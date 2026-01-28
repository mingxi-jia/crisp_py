from diff_eval_utils.controllers.simple_controller import SimpleSequentialController
from diff_eval_utils.diffusion_transforms import convert_action_from_fingertip_to_gripper
from diff_eval_utils.diffusion_visualization import visualize_pcd_and_actions
import time
import numpy as np
from pathlib import Path
import threading
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib import ticker
from std_msgs.msg import Int32MultiArray


class InterventionController(SimpleSequentialController):
    """Controller with human intervention capability via spacemouse and recording control.

    Two-layer State Machine:

    Recording States (outer layer):
    - READY: Robot is at start position, ready to start recording
    - RECORDING: Policy/intervention active

    Policy States (inner layer, only active when RECORDING):
    - POLICY_MODE: Execute actions from diffusion policy
    - INTERVENTION_MODE: Follow spacemouse input (human takeover)

    Key Controls:
    - 'r': Toggle recording
        - When RECORDING: Stop control, reset to start position → READY
        - When READY: Start policy inference → RECORDING
    - 'i': Resume policy from intervention (INTERVENTION → POLICY)
    - Spacemouse: Automatic intervention trigger (POLICY → INTERVENTION)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Recording state machine (outer layer)
        self.READY = 'ready'
        self.RECORDING = 'recording'
        self.recording_state = self.READY  # Start in READY mode

        # Policy state machine (inner layer)
        self.POLICY_MODE = 'policy'
        self.INTERVENTION_MODE = 'intervention'
        self.mode = self.POLICY_MODE

        # Spacemouse (initialized in run() using context manager)
        self.spacemouse = None

        # Keyboard listener state
        self.key_commands = {'r': False, 'i': False}
        self.listener_lock = threading.Lock()
        self.keyboard_listener = None

        # ROS2 publisher for intervention signals
        self.intervention_pub = None

        # Target pose tracking for intervention mode
        self.intervention_target_pose = None

        # Gripper button state tracking
        self.prev_button_pressed = False

        # Timing tracking for visualization
        self.timing_events = []  # List of (timestamp, event_type) tuples
        self.start_recording_time = None

        # Action recording for debugging
        self.recorded_actions = []  # List of executed actions

        self.ACTION_SCALE = self.config.spacemouse_action_scale
        self.DEADZONE = self.config.spacemouse_deadzone

    def _setup_keyboard_listener(self):
        """Initialize keyboard listener for 'r' and 'i' keys."""
        from pynput import keyboard

        def on_press(key):
            with self.listener_lock:
                try:
                    if hasattr(key, 'char'):
                        if key.char == 'r':
                            self.key_commands['r'] = True
                            print("\n[KEY: r] Recording toggle requested...")
                        elif key.char == 'i':
                            self.key_commands['i'] = True
                            print("\n[KEY: i] Resume policy requested...")
                except AttributeError:
                    pass

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
        print("Keyboard listener started:")
        print("  'r' - Toggle recording (ready <-> recording, with reset)")
        print("  'i' - Resume policy from intervention")

    def _setup_intervention_publisher(self):
        """Create ROS2 publisher for intervention signals."""
        from std_msgs.msg import Int32MultiArray

        self.intervention_pub = self.robot.node.create_publisher(
            Int32MultiArray,
            '/teleop/signals',
            30
        )
        print("Intervention signal publisher created on /teleop/signals")

    def _publish_intervention_state(self, state: int):
        """Publish intervention state (0=idle, 1=policy, 2=intervention).

        Args:
            state: 0 (idle), 1 (policy action), 2 (intervention action)
        """
        from std_msgs.msg import Int32MultiArray
        msg = Int32MultiArray()
        msg.data = [state, 0, 0, 0, 0, 0, 0, 0, 0]
        print(f"[PUBLISH] Publishing intervention state: {state}")
        self.intervention_pub.publish(msg)

    def _cleanup(self):
        """Clean up resources."""
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
            print("Keyboard listener stopped")

    def _check_for_intervention_trigger(self, spacemouse_motion):
        """Check if spacemouse motion triggers intervention.

        Args:
            spacemouse_motion: 6-element array [dx, dy, dz, droll, dpitch, dyaw]

        Returns:
            bool: True if intervention should be triggered
        """
        # Any non-zero motion triggers intervention
        motion_detected = np.any(np.abs(spacemouse_motion) > 1e-6)
        return motion_detected

    def _check_key_command(self, key):
        """Check if a key command was triggered and reset flag.

        Args:
            key: Key to check ('r' or 'i')

        Returns:
            bool: True if key was pressed
        """
        with self.listener_lock:
            if self.key_commands[key]:
                self.key_commands[key] = False
                return True
        return False

    def _handle_recording_toggle(self):
        """Handle 'r' key press: toggle between ready/recording states with reset."""
        if self.recording_state == self.READY:
            # Start recording: READY -> RECORDING
            print(f"\n{'='*60}")
            print("[START RECORDING] Starting policy inference...")
            print(f"{'='*60}")

            self.recording_state = self.RECORDING
            self.mode = self.POLICY_MODE  # Reset to policy mode
            self.intervention_target_pose = self.robot.end_effector_pose.copy()

            # Reset timing tracking
            self.timing_events = []
            self.start_recording_time = time.time()

            # Reset action recording
            self.recorded_actions = []

            print(f"[STATE] ready -> recording (policy mode)")
            print(f"{'='*60}\n")

        elif self.recording_state == self.RECORDING:
            # Stop recording and reset: RECORDING -> READY
            print(f"\n{'='*60}")
            print("[STOP RECORDING] Stopping robot actions and resetting to start position...")
            print(f"{'='*60}")

            # Generate timing visualization
            if len(self.timing_events) > 0:
                self._plot_timing_events()

            # Save recorded actions to .npy file
            if len(self.recorded_actions) > 0:
                self._save_recorded_actions()

            # Reset robot to start position
            self.robot.home()
            self._switch_to_impedance_controller()

            # Update target_pose to match the new robot position after homing
            self.target_pose = self.robot.end_effector_pose.copy()
            self.intervention_target_pose = self.target_pose.copy()
            # Transition to READY
            self.recording_state = self.READY
            self.mode = self.POLICY_MODE  # Reset to policy mode

            print(f"[STATE] recording rr-> ready (robot reset)")
            print(f"{'='*60}\n")

        self._update_buffer()

    def _execute_intervention_step(self, spacemouse, recording=True):
        """Execute one step of spacemouse control.

        Args:
            spacemouse: Spacemouse instance
        """
        from std_msgs.msg import Int32MultiArray

        # Get spacemouse motion and button state
        motion = spacemouse.get_motion_state_transformed()
        dx, dy, dz, droll, dpitch, dyaw = motion * self.ACTION_SCALE
        button_pressed = spacemouse.is_button_pressed(0)

        # Detect gripper toggle event
        gripper_toggle = 1 if (button_pressed and not self.prev_button_pressed) else 0

        # Publish intervention signals (always publish during intervention mode)
        has_movement = (dx != 0 or dy != 0 or dz != 0 or droll != 0 or dpitch != 0 or dyaw != 0)

        if (has_movement or gripper_toggle) and recording:
            # Publish intervention signals (for data collection/logging)
            msg = Int32MultiArray()
            # Convert to discrete signals (sign only)
            dx_int = (1 if dx > 0 else -1 if dx < 0 else 0)
            dy_int = (1 if dy > 0 else -1 if dy < 0 else 0)
            dz_int = (1 if dz > 0 else -1 if dz < 0 else 0)
            droll_int = (1 if droll > 0 else -1 if droll < 0 else 0)
            dpitch_int = (1 if dpitch > 0 else -1 if dpitch < 0 else 0)
            dyaw_int = (1 if dyaw > 0 else -1 if dyaw < 0 else 0)
            # Format: [state, dx, dy, dz, droll, dpitch, dyaw, gripper, reset]
            # state=2 means intervention action
            msg.data = [2, dx_int, dy_int, dz_int, droll_int, dpitch_int, dyaw_int, gripper_toggle, 0]
            self.intervention_pub.publish(msg)
        else:
            self._publish_intervention_state(0)
            return

        # Update target pose
        curr_pos = self.intervention_target_pose.position
        curr_euler = self.intervention_target_pose.orientation.as_euler('XYZ')

        # Apply deltas
        new_pos = np.array([
            curr_pos[0] + dx,
            curr_pos[1] + dy,
            max(curr_pos[2] + dz, 0.02)  # Safety: don't go below table
        ])

        new_euler = np.array([
            curr_euler[0] + droll * 2,
            curr_euler[1] - dpitch * 2,
            curr_euler[2] - dyaw * 2  # Match spacemouse_example.py convention
        ])

        # Update and send to robot
        self.intervention_target_pose.position = new_pos
        self.intervention_target_pose.orientation = R.from_euler('XYZ', new_euler)
        self.robot.set_target(pose=self.intervention_target_pose)
        # self.arm_rate.sleep()
        # Only print if there's movement
        if dx != 0 or dy != 0 or dz != 0 or droll != 0 or dpitch != 0 or dyaw != 0:
            print(f"[INTERVENTION] x={new_pos[0]:.4f} y={new_pos[1]:.4f} z={new_pos[2]:.4f} "
                  f"roll={new_euler[0]:.4f} pitch={new_euler[1]:.4f} yaw={new_euler[2]:.4f}")

        # Handle gripper button toggle (button_pressed already retrieved above)
        if gripper_toggle:
            # Toggle gripper state
            new_gripper_value = 1.0 - self.prev_grasp_value
            print(f"[GRIPPER] {'Closing' if new_gripper_value == 1.0 else 'Opening'} gripper...")
            self.gripper.set_target(1.0 - new_gripper_value)  # Invert for Franka convention
            self.gripper_rate.sleep()
            time.sleep(1.0)  # Wait for gripper (Franka driver limitation)
            self.prev_grasp_value = new_gripper_value

        # Update button state for next iteration
        self.prev_button_pressed = button_pressed

        self.arm_rate.sleep()

        # Trigger async buffer update (non-blocking)
        self._update_buffer()

    def _execute_action(self, action, move_to=False):
        """Execute a single action.

        Args:
            action: 10-element action array [x, y, z, rot6d(6), grasp(1)]
        """
        action, gripper_pose = convert_action_from_fingertip_to_gripper(action, self.rotation_transformer)

        # Position deltas
        new_position = action[:3]
        prev_position = self.target_pose.position
        dx = new_position[0] - prev_position[0]
        dy = new_position[1] - prev_position[1]
        dz = new_position[2] - prev_position[2]

        # Rotation deltas
        prev_euler = self.target_pose.orientation.as_euler('XYZ')
        new_euler = gripper_pose.as_euler('XYZ')
        droll = new_euler[0] - prev_euler[0]
        dpitch = new_euler[1] - prev_euler[1]
        dyaw = new_euler[2] - prev_euler[2]

        # Convert to discrete signals (sign only)
        dx_int = (1 if dx > 0 else -1 if dx < 0 else 0)
        dy_int = (1 if dy > 0 else -1 if dy < 0 else 0)
        dz_int = (1 if dz > 0 else -1 if dz < 0 else 0)
        droll_int = (1 if droll > 0 else -1 if droll < 0 else 0)
        dpitch_int = (1 if dpitch > 0 else -1 if dpitch < 0 else 0)
        dyaw_int = (1 if dyaw > 0 else -1 if dyaw < 0 else 0)

        # Extract gripper value
        grasp_value = int(np.round(np.clip(action[-1], 0, 1)))

        # Publish policy action state with deltas and gripper value
        msg = Int32MultiArray()
        # Format: [state, dx, dy, dz, droll, dpitch, dyaw, gripper, reset]
        # state=1 means policy action
        msg.data = [1, dx_int, dy_int, dz_int, droll_int, dpitch_int, dyaw_int, grasp_value, 0]


        self.intervention_pub.publish(msg)

        # Record message encoding timing
        self.timing_events.append((time.time(), 'message_sent'))

        # Record action for debugging
        self.recorded_actions.append(action.copy())

        self.target_pose.position = new_position
        # self.target_pose.orientation = R.from_euler('XYZ', [np.pi, 0, 0])
        self.target_pose.orientation = gripper_pose
        print(f"Moving to position: {new_position}, orientation (euler): {gripper_pose.as_euler('XYZ')}")
        if not move_to:
            self.robot.set_target(pose=self.target_pose)
            self.arm_rate.sleep()
        else:
            self.robot.move_to(pose=self.target_pose, speed=0.15)

        grasp_value = np.round(np.clip(action[-1], 0, 1))
        if grasp_value != self.prev_grasp_value:
            self.gripper.set_target(1 - grasp_value)
            self.gripper_rate.sleep()
            time.sleep(1.0)  # Wait for gripper (Franka driver limitation)
        self.prev_grasp_value = grasp_value

        # Trigger async buffer update (non-blocking)
        self._update_buffer()

    def _save_recorded_actions(self):
        """Save recorded actions to .npy file with timestamp."""
        from datetime import datetime

        # Create debug_data directory if it doesn't exist
        debug_dir = Path('debug_data')
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp-based filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = debug_dir / f'actions_{timestamp}.npy'

        # Convert to numpy array and save
        actions_array = np.array(self.recorded_actions)
        np.save(filename, actions_array)

        print(f"\nRecorded actions saved to: {filename.absolute()}")
        print(f"Total actions saved: {len(self.recorded_actions)}")

    def _plot_timing_events(self):
        """Generate timing visualization plot."""
        print("\nGenerating timing events visualization...")

        # Convert timing events to arrays
        timestamps = np.array([t - self.start_recording_time for t, _ in self.timing_events])
        event_types = [event_type for _, event_type in self.timing_events]

        # Map event types to numeric values for plotting
        event_map = {
            'inference_done': 0,
            'message_sent': 1,
            'action_executed': 2
        }

        # Create plot
        fig, ax = plt.subplots(figsize=(16, 7))

        # Plot events with vertical lines and time differences
        colors = {'inference_done': 'red', 'message_sent': 'orange', 'action_executed': 'green'}

        for event_type, y_val in event_map.items():
            mask = [e == event_type for e in event_types]
            event_times = timestamps[mask]

            if len(event_times) > 0:
                # Plot scatter points
                ax.scatter(event_times, [y_val] * len(event_times),
                          c=colors[event_type], s=80, alpha=0.8, label=event_type, zorder=3)

                # Draw vertical lines to x-axis
                for t in event_times:
                    ax.plot([t, t], [y_val, -0.3], color=colors[event_type],
                           alpha=0.3, linewidth=1, zorder=1)

                # Add time difference labels between consecutive events
                for i in range(1, len(event_times)):
                    time_diff = (event_times[i] - event_times[i-1]) * 1000  # Convert to ms
                    mid_time = (event_times[i] + event_times[i-1]) / 2
                    ax.text(mid_time, y_val + 0.15, f'{time_diff:.1f}ms',
                           ha='center', va='bottom', fontsize=8,
                           color=colors[event_type], weight='bold')

        # Configure axes with finer granularity
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Event Type', fontsize=12, fontweight='bold')
        ax.set_yticks(list(event_map.values()))
        ax.set_yticklabels(list(event_map.keys()))
        ax.set_ylim(-0.5, 2.5)

        # Finer x-axis ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

        ax.set_title('Policy Execution Timing Events', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x', which='major')
        ax.grid(True, alpha=0.15, axis='x', which='minor', linestyle=':')
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        filename = Path('debug_plots/timing_events.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Timing events plot saved to: {filename.absolute()}")
        plt.close()

    def run(self, n_steps: int = None):
        """Execute intervention-enabled control with recording state management.

        Args:
            n_steps: Not used in this mode (runs indefinitely until Ctrl+C)

        State Machine:
        1. Check for 'r' key (recording control with reset)
        2. If RECORDING:
           - POLICY_MODE: Execute policy, monitor spacemouse for intervention
           - INTERVENTION_MODE: Execute spacemouse, monitor 'i' key for resume
        3. If READY: Wait for 'r' to start recording
        """
        from crisp_py.spacemouse import Spacemouse

        # Setup
        self._setup_keyboard_listener()
        self._setup_intervention_publisher()

        policy_iter = 0  # Track policy iterations (for observation cycles)

        print("\n" + "="*60)
        print("INTERVENTION + RECORDING MODE ACTIVE")
        print("="*60)
        print("Controls:")
        print("  'r' - Toggle recording (ready <-> recording, with reset)")
        print("  Spacemouse - Automatic intervention during recording")
        print("  'i' - Resume policy from intervention")
        print("  Ctrl+C - Exit")
        print("="*60)
        print(f"\nInitial state: {self.recording_state}")
        print("="*60 + "\n")

        try:
            with Spacemouse(deadzone=self.DEADZONE) as sm:
                self.spacemouse = sm

                # Policy state (only used when RECORDING)
                current_actions = None
                action_idx = 0
                self.intervention_target_pose = self.robot.end_effector_pose.copy()

                while True:
                    iter_start = time.time()

                    # Check for 'r' key (toggle recording with reset)
                    if self._check_key_command('r'):
                        self._handle_recording_toggle()
                        # Reset policy state when toggling
                        current_actions = None
                        first = True
                        action_idx = 0

                    recording = self.recording_state == self.RECORDING

                    # Execute policy/intervention only when RECORDING
                    if self.mode == self.POLICY_MODE:
                        if recording:
                        # STATE: POLICY_MODE
                            # Get new actions if needed
                            if current_actions is None:
                                first = True
                                print("first True")
                            else:
                                first = False
                            
                            self._publish_intervention_state(0)
                            obs_dict = self._get_observation()
                            current_actions = self.policy_client.predict_action(obs_dict)
                            for action in current_actions[:self.config.action_exec_size]:
                                # Calculate deltas for monitoring (before executing action)
                                if first:
                                    time.sleep(0.1)
                                self._execute_action(action, move_to=False)

                                # Record action execution timing
                                self.timing_events.append((time.time(), 'action_executed'))
                                first = False
                                action_idx += 1
                            
                                motion = sm.get_motion_state_transformed()
                                if self._check_for_intervention_trigger(motion):
                                    print(f"\n{'='*60}")
                                    print("[INTERVENTION TRIGGERED] Spacemouse movement detected!")
                                    print(f"{'='*60}")
                                    self.mode = self.INTERVENTION_MODE

                                    # Initialize intervention target from current pose
                                    self.intervention_target_pose = self.robot.end_effector_pose.copy()
                                    # Invalidate current actions (will get fresh observation on resume)
                                    current_actions = None
                                    action_idx = 0
                                    break

                            iter_total = time.time() - iter_start
                            print(f"[RECORDING/POLICY] Action {action_idx}: {iter_total*1000:.1f} ms")

                            # if current_actions is None or action_idx >= self.config.action_exec_size:
                            #     # time.sleep(2)
                            #     # print("Prev Chunk Done")
                            #     self._publish_intervention_state(0)
                            #     obs_dict = self._get_observation()
                            #     current_actions = self.policy_client.predict_action(obs_dict)
                            #     action_idx = 0
                            #     policy_iter += 1
                            #     print(f"\n[POLICY #{policy_iter}] Got {len(current_actions)} actions")

                            #     # Record inference timing
                            #     self.timing_events.append((time.time(), 'inference_done'))

                            #     # visualize_pcd_and_actions(obs_dict['pcd'], current_actions)

                            # # Execute next action
                            # action = current_actions[action_idx].copy()
                            # # Calculate deltas for monitoring (before executing action)
                            # if first:
                            #     time.sleep(0.3)
                            # self._execute_action(action, move_to=False)

                            # # Record action execution timing
                            # self.timing_events.append((time.time(), 'action_executed'))
                            # action_idx += 1

                            # Check for intervention trigger
                            # motion = sm.get_motion_state_transformed()
                            # if self._check_for_intervention_trigger(motion):
                            #     print(f"\n{'='*60}")
                            #     print("[INTERVENTION TRIGGERED] Spacemouse movement detected!")
                            #     print(f"{'='*60}")
                            #     self.mode = self.INTERVENTION_MODE

                            #     # Initialize intervention target from current pose
                            #     self.intervention_target_pose = self.robot.end_effector_pose.copy()
                            #     # Invalidate current actions (will get fresh observation on resume)
                            #     current_actions = None
                            #     action_idx = 0
                            #     continue

                            # iter_total = time.time() - iter_start
                            # print(f"[RECORDING/POLICY] Action {action_idx}: {iter_total*1000:.1f} ms")

                    # STATE: INTERVENTION_MODE
                    if self.mode == self.INTERVENTION_MODE or not recording:

                        # Execute spacemouse control
                        self._execute_intervention_step(sm, recording=recording)

                        if recording:
                            # Check for resume request
                            if self._check_key_command('i'):
                                print(f"\n{'='*60}")
                                print("[RESUMING POLICY] Getting observation from current pose...")
                                print(f"{'='*60}")
                                self.mode = self.POLICY_MODE
                                # Sync target_pose with current intervention target
                                self.target_pose = self.intervention_target_pose.copy()
                                # Will get fresh observation on next iteration
                                continue
                    if not recording:
                        # Not recording, publish idle state and sleep briefly
                        self._publish_intervention_state(0)
                        time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nRecording stopped by user")
        finally:
            self._cleanup()
