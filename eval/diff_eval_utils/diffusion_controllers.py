"""Controller hierarchy for different diffusion policy execution modes."""

from abc import ABC, abstractmethod
import time
import numpy as np
from pathlib import Path
import threading
import shutil
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib import cm, ticker

from .diffusion_transforms import (
    get_pose_from_robot,
    convert_action_from_fingertip_to_gripper,
    franka_obs_to_diff_obs
)
from .diffusion_constants import GRIPPER_NORM_CONST
from std_msgs.msg import Int32MultiArray

class RobotController(ABC):
    """Abstract base class for robot controllers."""

    def __init__(self, robot, gripper, obs_manager, joint_state_subscriber,
                 policy_client, rotation_transformer,
                 config, ctrl_freq=10.0):
        """Initialize the controller.

        Args:
            robot: Robot instance
            gripper: Gripper instance
            obs_manager: PointCloudManager instance
            joint_state_subscriber: JointStateSubscriber instance
            policy_client: PolicyClient instance
            pcd_client: PcdProcessingClient instance
            rotation_transformer: RotationTransformer for 6D rotations
            config: DPEvalConfig instance
            ctrl_freq: Control frequency in Hz
        """
        self.robot = robot
        self.gripper = gripper
        self.obs_manager = obs_manager
        self.joint_state_subscriber = joint_state_subscriber
        self.policy_client = policy_client
        # self.pcd_client = pcd_client
        self.rotation_transformer = rotation_transformer
        self.config = config
        self.ctrl_freq = ctrl_freq

        # Common state
        self.target_pose = robot.end_effector_pose.copy()
        self.arm_rate = robot.node.create_rate(ctrl_freq)
        self.gripper_rate = gripper.node.create_rate(ctrl_freq)
        self.prev_grasp_value = 0.0

    @abstractmethod
    def run(self, n_steps: int):
        """Execute the controller for n_steps.

        Args:
            n_steps: Number of control steps to execute
        """
        pass

    def _get_observation(self):
        """Get current observation from sensors."""
        joint_state = self.joint_state_subscriber.joint_values
        gripper_state = self.prev_grasp_value
        print(f"gripper_state: {gripper_state}")

        time.sleep(0.05)  # Wait for robot stability

        eef_pose = get_pose_from_robot(self.robot.end_effector_pose)

        obs_dict = franka_obs_to_diff_obs(
            self.obs_manager, eef_pose, gripper_state,
             joint_state, visualize=self.config.visualize
        )
        # from diff_eval_utils.diffusion_visualization import visualize_pcd
        # visualize_pcd(obs_dict['pcd'])



        return obs_dict

    def _execute_action(self, action, move_to=False):
        """Execute a single action.

        Args:
            action: 10-element action array [x, y, z, rot6d(6), grasp(1)]
        """
        action, gripper_pose = convert_action_from_fingertip_to_gripper(action, self.rotation_transformer)

        # Collect debug data if enabled
        if hasattr(self, 'debug_data_collection') and self.debug_data_collection is not None:
            timestamp_ms = int(time.time() * 1000)

            # Save executed action (before transformations)
            self.debug_data_collection['executed_actions'].append(
                (timestamp_ms, action.copy())
            )

            # Save EEF pose
            current_pose = self.robot.end_effector_pose
            pose_dict = {
                'position': current_pose.position.copy(),
                'orientation_quat': current_pose.orientation.as_quat().copy(),
                'orientation_matrix': current_pose.orientation.as_matrix().copy()
            }
            self.debug_data_collection['eef_poses'].append(
                (timestamp_ms, pose_dict)
            )

            # Save actual EEF position
            self.debug_data_collection['actual_eef_pos'].append(
                (timestamp_ms, current_pose.position.copy())
            )

        # Safety check: verify pose change is reasonable
        new_position = action[:3]
        prev_position = self.target_pose.position

        # # Fast position delta check
        # position_delta = np.linalg.norm(new_position - prev_position)

        # # Fast rotation delta check using quaternion dot product
        # prev_quat = self.target_pose.orientation.as_quat()  # [x, y, z, w]
        # new_quat = gripper_pose.as_quat()
        # quat_dot = np.abs(np.dot(prev_quat, new_quat))  # |q1 · q2| ∈ [0, 1]
        # rotation_angle = 2 * np.arccos(np.clip(quat_dot, 0, 1))  # angle in radians

        # MAX_POSITION_DELTA = 0.1  # 10cm maximum movement per step
        # MAX_ROTATION_DELTA = np.deg2rad(10)  # 30 degrees maximum rotation per step

        # if position_delta > MAX_POSITION_DELTA or rotation_angle > MAX_ROTATION_DELTA:
        #     print(f"⚠ WARNING: Large pose jump detected!")
        #     print(f"  Position delta: {position_delta:.4f}m (max: {MAX_POSITION_DELTA}m)")
        #     print(f"  Rotation delta: {np.rad2deg(rotation_angle):.2f}° (max: {np.rad2deg(MAX_ROTATION_DELTA):.2f}°)")
        #     print(f"  Skipping this action for safety!")
        #     return

        self.target_pose.position = new_position
        # self.target_pose.orientation = R.from_euler('XYZ', [np.pi, 0, 0])
        self.target_pose.orientation = gripper_pose
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

    def _switch_to_impedance_controller(self):
        self.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
        self.robot.cartesian_controller_parameters_client.load_param_config(
            file_path="config/control/spacemouse_cartesian_impedance.yaml"
        )


class SimpleSequentialController(RobotController):
    """Simple controller that executes actions sequentially without buffering."""

    def run(self, n_steps: int):
        """Execute simple sequential control.

        Each iteration:
        1. Get observation
        2. Query policy for actions
        3. Execute all actions sequentially
        """
        n_steps_done = 0

        print("Starting simple sequential control...")
        while n_steps_done < n_steps:
            iter_start = time.time()

            # Get observation and action
            obs_dict = self._get_observation()
            actions = self.policy_client.predict_action(obs_dict)
            print(f"{actions[0][0] - actions[7][0]}\t{actions[0][1] - actions[7][1]}\t{actions[0][2] - actions[7][2]}")
            # Execute first 8 actions
            for action in actions[:8]:
                if n_steps_done >= n_steps:
                    break
                self._execute_action(action.copy())
                n_steps_done += 1

            iter_total = time.time() - iter_start
            print(f"Step {n_steps_done}: {iter_total*1000:.1f} ms")

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

    # Constants
    ACTION_SCALE = 0.01  # 1cm per spacemouse unit
    DEADZONE = 0.1  # Spacemouse deadzone threshold

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
            curr_euler[2] - dyaw * 6  # Match spacemouse_example.py convention
        ])

        # Update and send to robot
        self.intervention_target_pose.position = new_pos
        self.intervention_target_pose.orientation = R.from_euler('XYZ', new_euler)
        self.robot.set_target(pose=self.intervention_target_pose)

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
                        action_idx = 0

                    recording = self.recording_state == self.RECORDING

                    # Execute policy/intervention only when RECORDING
                    if self.mode == self.POLICY_MODE:
                        if recording:
                        # STATE: POLICY_MODE
                            # Get new actions if needed
                            if current_actions is None or action_idx >= self.config.action_exec_size:
                                self._publish_intervention_state(0)
                                obs_dict = self._get_observation()
                                current_actions = self.policy_client.predict_action(obs_dict)
                                action_idx = 0
                                policy_iter += 1
                                print(f"\n[POLICY #{policy_iter}] Got {len(current_actions)} actions")

                                # Record inference timing
                                self.timing_events.append((time.time(), 'inference_done'))

                            # Execute next action
                            action = current_actions[action_idx].copy()
                            # Calculate deltas for monitoring (before executing action)

                            self._execute_action(action, move_to=False)

                            # Record action execution timing
                            self.timing_events.append((time.time(), 'action_executed'))
                            action_idx += 1

                            # Check for intervention trigger
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
                                continue

                            iter_total = time.time() - iter_start
                            print(f"[RECORDING/POLICY] Action {action_idx}: {iter_total*1000:.1f} ms")

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


class ChunkingController(RobotController):
    """Controller with action buffering and policy_delay offset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_queue = []
        self.last_inference_time = None
        self.last_execution_time = None
        self.inference_count = 0
        self.enforce_fixed_freq = True

        self.inference_thread = None
        self.inference_in_progress = False
        self.inference_lock = threading.Lock()

        self.execute_dt = 1.0 / self.ctrl_freq

        # Debug plotting setup
        if self.config.debug_plotting:
            self._setup_debug_plotting()

        # Debug data collection
        if self.config.save_debug_data:
            self.debug_data_collection = {
                'action_chunks': [],      # (timestamp_ms, array(16,10))
                'executed_actions': [],   # (timestamp_ms, array(10,))
                'eef_poses': [],          # (timestamp_ms, pose_dict)
                'point_clouds': [],       # (timestamp_ms, array(N,6))
                'robot0_eef_pos': [],     # (timestamp_ms, array(3,))
                'actual_eef_pos': [],     # (timestamp_ms, array(3,))
                'pcd_timestamps': []     # (timestamp_ms, pcd_timestamp)
            }
        else:
            self.debug_data_collection = None

        # Live visualization
        self.live_plot_fig = None
        self.live_plot_ax = None
        self.live_events = []  # List of (timestamp, event_type) tuples
        self.start_time = None
        self.last_plot_update_time = 0  # Track last update time
        self.plot_update_interval = 1.0  # Update plot every 1 second
        if self.config.visualize:
            self._setup_live_plot()

    def _setup_debug_plotting(self):
        """Initialize debug plotting data structures."""
        if self.config.debug_output_dir.exists():
            shutil.rmtree(self.config.debug_output_dir)
        self.config.debug_output_dir.mkdir(parents=True, exist_ok=True)

        self.predicted_actions = []
        self.actual_poses = []
        self.control_loop_times = []
        self.all_predicted_chunks = []
        self.start_time = None

    def _need_inference(self):
        """Check if we need to run policy inference."""
        need_inf = len(self.action_queue) <= self.config.policy_delay
        if need_inf:
            print(f"\n[Inference Needed] Current action queue size: {len(self.action_queue)}")
        return need_inf

    def _run_inference(self):
        """Run policy inference and add actions to queue."""
        current_time = time.time()

        if self.last_inference_time is not None:
            time_since_last = current_time - self.last_inference_time
            print(f"\n{'='*50}")
            print(f"[Inference #{self.inference_count}] Time since last: {time_since_last*1000:.1f} ms")
            print(f"{'='*50}")

        # Get observation and predict
        obs_dict = self._get_observation()
        actions = self.policy_client.predict_action(obs_dict)

        # Store for debug plotting
        if self.config.debug_plotting:
            self.all_predicted_chunks.append((self.inference_count, actions.copy()))

        # Add actions to queue
        if len(self.action_queue) == 0:
            # First time: add first action_exec_size actions
            for action in actions[:self.config.action_exec_size]:
                self.action_queue.append(action.copy())
        else:
            # Subsequent: add actions starting from policy_delay
            start_idx = self.config.policy_delay
            end_idx = start_idx + self.config.action_exec_size
            for action in actions[start_idx:end_idx]:
                self.action_queue.append(action.copy())

        self.last_inference_time = time.time()
        print(f"Inference took {(self.last_inference_time - current_time)*1000:.1f} ms")
        self.inference_count += 1

    def _run_inference_async(self):
        """Run inference in background thread."""
        def inference_worker():
            try:
                current_time = time.time()

                if self.last_inference_time is not None:
                    time_since_last = current_time - self.last_inference_time
                    print(f"\n{'='*50}")
                    print(f"[Inference #{self.inference_count}] Time since last: {time_since_last*1000:.1f} ms")
                    print(f"{'='*50}")

                # Record observation arrival
                if self.config.visualize:
                    self.live_events.append((time.time(), 'obs_arrival'))

                # Get observation and predict
                obs_dict = self._get_observation()
                actions = self.policy_client.predict_action(obs_dict)
                inference_time = time.time()

                # Record inference done
                if self.config.visualize:
                    self.live_events.append((inference_time, 'inference_done'))
                # Store for debug plotting
                if self.config.debug_plotting:
                    self.all_predicted_chunks.append((self.inference_count, actions.copy()))
                
                # Thread-safe queue update
                with self.inference_lock:
                    if self.first_inference_round:
                        for action in actions[:self.config.action_exec_size]:
                            self.action_queue.append(action.copy())
                        self.first_inference_round = False
                    else:
                        start_idx = self.config.policy_delay
                        end_idx = start_idx + self.config.action_exec_size
                        for action in actions[start_idx:end_idx]:
                            self.action_queue.append(action.copy())

                    # Debug data collection
                    if self.debug_data_collection is not None:
                        timestamp_ms = int(inference_time * 1000)

                        # Save action chunk
                        self.debug_data_collection['action_chunks'].append(
                            (timestamp_ms, actions.copy())
                        )

                        # Save point cloud
                        pcd = obs_dict.get('pcd', None)
                        if pcd is not None:
                            self.debug_data_collection['point_clouds'].append(
                                (timestamp_ms, pcd.copy())
                            )

                        # Save robot0_eef_pos
                        robot0_eef_pos = obs_dict.get('robot0_eef_pos', None)
                        if robot0_eef_pos is not None:
                            self.debug_data_collection['robot0_eef_pos'].append(
                                (timestamp_ms, robot0_eef_pos.copy())
                            )

                        # Save RGBD timestamp
                        pcd_timestamp = obs_dict.get('pcd_timestamp', None)
                        if pcd_timestamp is not None:
                            # Ensure we create an independent copy (not a reference)
                            if isinstance(pcd_timestamp, np.ndarray):
                                ts_copy = pcd_timestamp.copy()
                            else:
                                ts_copy = np.array(pcd_timestamp).copy()
                            print(f"Saving RGBD timestamp: {ts_copy[0]}")
                            self.debug_data_collection['pcd_timestamps'].append(
                                (timestamp_ms, ts_copy[0])
                            )
                
                    self.last_inference_time = time.time()
                    print(f"Inference took {(self.last_inference_time - current_time)*1000:.1f} ms")
                    self.inference_count += 1
                    
            finally:
                self.inference_in_progress = False
        
        # Start thread
        self.inference_in_progress = True
        self.inference_thread = threading.Thread(target=inference_worker, daemon=True)
        self.inference_thread.start()
    
    def _enforce_fixed_frequency(self):
        curr_time = time.time()
        if self.last_execution_time is not None:
            timing_adjust_constant = 0.001
            time_since_last_exec = curr_time - self.last_execution_time
            time_since_last_exec += timing_adjust_constant
            if time_since_last_exec < self.execute_dt:
                time_to_wait = self.execute_dt - time_since_last_exec
                time.sleep(time_to_wait - timing_adjust_constant)

    def _setup_live_plot(self):
        """Initialize live matplotlib plot for event tracking."""
        plt.ion()  # Enable interactive mode
        self.live_plot_fig, self.live_plot_ax = plt.subplots(figsize=(12, 6))
        self.live_plot_ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        self.live_plot_ax.set_ylabel('Event Type', fontsize=12, fontweight='bold')
        self.live_plot_ax.set_title('Live Policy Execution Events', fontsize=14, fontweight='bold')
        self.live_plot_ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        print("Live visualization enabled")

    def _update_live_plot(self, force=False):
        """Update live plot with latest events (throttled to reduce overhead).

        Args:
            force: If True, update regardless of time interval
        """
        if self.live_plot_ax is None or len(self.live_events) == 0:
            return

        # Only update if enough time has passed or forced
        current_time = time.time()
        if not force and (current_time - self.last_plot_update_time) < self.plot_update_interval:
            return

        self.last_plot_update_time = current_time

        # Clear and redraw
        self.live_plot_ax.clear()

        # Convert timing events to arrays
        timestamps = np.array([t - self.start_time for t, _ in self.live_events])
        event_types = [event_type for _, event_type in self.live_events]

        # Map event types to numeric values
        event_map = {'obs_arrival': 0, 'inference_done': 1, 'action_executed': 2}
        colors = {'obs_arrival': 'blue', 'inference_done': 'red', 'action_executed': 'green'}

        # Plot events
        for event_type, y_val in event_map.items():
            mask = [e == event_type for e in event_types]
            event_times = timestamps[mask]
            if len(event_times) > 0:
                self.live_plot_ax.scatter(event_times, [y_val] * len(event_times),
                                         c=colors[event_type], s=60, alpha=0.8, label=event_type)

        # Configure axes
        self.live_plot_ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        self.live_plot_ax.set_ylabel('Event Type', fontsize=12, fontweight='bold')
        self.live_plot_ax.set_yticks(list(event_map.values()))
        self.live_plot_ax.set_yticklabels(list(event_map.keys()))
        self.live_plot_ax.set_ylim(-0.5, 2.5)
        self.live_plot_ax.set_title('Live Policy Execution Events', fontsize=14, fontweight='bold')
        self.live_plot_ax.grid(True, alpha=0.3)
        self.live_plot_ax.legend(loc='upper right', fontsize=10)

        # Refresh display
        self.live_plot_fig.canvas.draw()
        self.live_plot_fig.canvas.flush_events()

    def run(self, n_steps: int):
        """Execute chunking control with async inference."""
        n_steps_done = 0

        if self.config.debug_plotting:
            self.start_time = time.time()

        # Initialize start time for live visualization
        if self.config.visualize:
            self.start_time = time.time()

        # Initial inference (synchronous to populate queue)
        self._run_inference_async()
        while self.inference_in_progress:
            time.sleep(0.002)  # Wait for first inference

        print("\nStarting chunking control...")
        self.first_inference_round = True
        while n_steps_done < n_steps:
            loop_start = time.time()
            
            # Trigger async inference if needed (don't wait)
            if self._need_inference() and not self.inference_in_progress:
                self._run_inference_async()
            

            # Execute action from queue
            with self.inference_lock:
                queue_has_actions = len(self.action_queue) > 0
                if queue_has_actions:
                    action = self.action_queue.pop(0)
            
            if queue_has_actions:
                # Enforce fixed control frequency
                if self.enforce_fixed_freq:
                    self._enforce_fixed_frequency()
                self._execute_action(action)
                self.last_execution_time = time.time() # Update execution time

                # Record action execution
                if self.config.visualize:
                    self.live_events.append((time.time(), 'action_executed'))
                    self._update_live_plot()

                if self.config.debug_plotting:
                    current_pose = self.robot.end_effector_pose
                    rotmat = current_pose.orientation.as_matrix()
                    yaw = np.arctan2(rotmat[1, 0], rotmat[0, 0])
                    elapsed = time.time() - self.start_time
                    self.actual_poses.append((elapsed, current_pose.position[0],
                                             current_pose.position[1],
                                             current_pose.position[2], yaw))
                n_steps_done += 1
            else:
                print("Warning: Action queue empty, waiting for inference...")
                time.sleep(0.01)
            
            if self.config.debug_plotting:
                loop_time = (time.time() - loop_start) * 1000
                self.control_loop_times.append(loop_time)

        # Wait for any ongoing inference to complete
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=5.0)

        # Final plot update and close
        if self.live_plot_fig is not None:
            self._update_live_plot(force=True)  # Final update with all events
            plt.close(self.live_plot_fig)
            self.live_plot_fig = None
            self.live_plot_ax = None

        # Save debug data at end
        if self.debug_data_collection is not None and len(self.debug_data_collection['executed_actions']) > 0:
            self._save_debug_data()

        # Generate debug plots
        if self.config.debug_plotting:
            self._plot_debug_results()
            self._plot_action_chunks()
            self._plot_action_execution_timing()
            print(f"\n{'='*60}")
            print(f"Debug plots saved to: {self.config.debug_output_dir.absolute()}")
            print(f"{'='*60}")

    def _plot_debug_results(self):
        """Generate debug visualization plots."""
        print("\nGenerating control loop timing histogram...")

        loop_times = np.array(self.control_loop_times)

        # Create histogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(loop_times, bins=30, color='steelblue', alpha=0.7, edgecolor='black')

        # Add statistics
        mean_time = np.mean(loop_times)
        median_time = np.median(loop_times)
        std_time = np.std(loop_times)
        min_time = np.min(loop_times)
        max_time = np.max(loop_times)

        # Add vertical lines
        ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.2f} ms')
        ax.axvline(median_time, color='green', linestyle='--', linewidth=2, label=f'Median: {median_time:.2f} ms')

        # Add text box with statistics
        stats_text = (f'Statistics:\n'
                     f'Mean: {mean_time:.2f} ms\n'
                     f'Median: {median_time:.2f} ms\n'
                     f'Std Dev: {std_time:.2f} ms\n'
                     f'Min: {min_time:.2f} ms\n'
                     f'Max: {max_time:.2f} ms\n'
                     f'Total loops: {len(loop_times)}')

        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, family='monospace')

        ax.set_xlabel('Control Loop Execution Time (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Control Loop Timing Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper left', fontsize=10)

        plt.tight_layout()
        filename = self.config.debug_output_dir / "control_loop_timing.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Timing histogram saved to: {filename}")
        plt.close()

    def _plot_action_chunks(self):
        """Generate plots showing all predicted actions from each inference chunk."""
        print("Generating action chunk plots...")

        if len(self.all_predicted_chunks) == 0:
            print("No action chunks to plot")
            return

        n_inferences = len(self.all_predicted_chunks)
        colors = cm.rainbow(np.linspace(0, 1, n_inferences))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('All Predicted Action Chunks (16 actions per inference)', fontsize=14, fontweight='bold')

        labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)', 'Yaw (rad)']

        for ax, label in zip(axes.flatten(), labels):
            for inf_idx, (chunk_id, actions) in enumerate(self.all_predicted_chunks):
                if inf_idx == 0:
                    start_step = 0
                else:
                    start_step = self.config.action_exec_size * inf_idx

                x_vals = []
                y_vals = []

                for action_idx in range(len(actions)):
                    action = actions[action_idx]
                    n_step = start_step + action_idx

                    if label == 'X Position (m)':
                        value = action[0]
                    elif label == 'Y Position (m)':
                        value = action[1]
                    elif label == 'Z Position (m)':
                        value = action[2]
                    else:  # Yaw
                        rot6d = action[3:9]
                        rotmat = self.rotation_transformer.forward(rot6d.reshape(1, 6))[0]
                        value = np.arctan2(rotmat[1, 0], rotmat[0, 0])

                    x_vals.append(n_step)
                    y_vals.append(value)

                ax.plot(x_vals, y_vals, 'o-', color=colors[inf_idx],
                       linewidth=1, markersize=3, alpha=0.7,
                       label=f'Inference #{chunk_id}')

            ax.set_xlabel('n_step', fontsize=11, fontweight='bold')
            ax.set_ylabel(label, fontsize=11, fontweight='bold')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            ax.grid(which='major', color='black', linewidth=1)
            ax.grid(which='minor', color='gray', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8, ncol=2)

        plt.tight_layout()
        filename = self.config.debug_output_dir / "action_chunks.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Action chunks plot saved to: {filename}")
        plt.close()

    def _plot_action_execution_timing(self):
        """Generate plot showing execution time vs action index to identify delays."""
        print("Generating action execution timing plot...")

        if len(self.control_loop_times) == 0:
            print("No timing data to plot")
            return

        loop_times = np.array(self.control_loop_times)
        action_indices = np.arange(len(loop_times))

        # Calculate cumulative time for reference
        cumulative_time = np.cumsum(loop_times)

        # Calculate expected time at ideal frequency
        ideal_dt = 1000.0 / self.config.ctrl_freq  # ms per step
        expected_cumulative = action_indices * ideal_dt

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Action Execution Timing Analysis', fontsize=14, fontweight='bold')

        # Subplot 1: Loop time per action
        ax1.plot(action_indices, loop_times, 'o-', color='steelblue',
                linewidth=1.5, markersize=4, alpha=0.7, label='Actual loop time')
        ax1.axhline(ideal_dt, color='red', linestyle='--', linewidth=2,
                   label=f'Expected (1/{self.config.ctrl_freq} Hz = {ideal_dt:.1f} ms)')

        # Highlight outliers (>1.5x expected)
        outliers = loop_times > (ideal_dt * 1.5)
        if np.any(outliers):
            ax1.scatter(action_indices[outliers], loop_times[outliers],
                       color='red', s=80, zorder=5, label='Delays (>1.5x expected)', marker='x')

        ax1.set_xlabel('Action Index', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Loop Execution Time (ms)', fontsize=11, fontweight='bold')
        ax1.set_title('Control Loop Time per Action', fontsize=12, fontweight='bold')
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax1.grid(which='major', color='black', linewidth=0.5, alpha=0.3)
        ax1.grid(which='minor', color='gray', linestyle='--', alpha=0.2)
        ax1.legend(loc='best', fontsize=9)

        # Add statistics box
        mean_time = np.mean(loop_times)
        std_time = np.std(loop_times)
        outlier_count = np.sum(outliers)
        stats_text = (f'Mean: {mean_time:.2f} ms\n'
                     f'Std: {std_time:.2f} ms\n'
                     f'Delays: {outlier_count}/{len(loop_times)}')
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, family='monospace')

        # Subplot 2: Cumulative time deviation
        time_deviation = cumulative_time - expected_cumulative
        ax2.plot(action_indices, time_deviation, 'o-', color='purple',
                linewidth=1.5, markersize=4, alpha=0.7)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.fill_between(action_indices, 0, time_deviation,
                        where=(time_deviation >= 0), color='red', alpha=0.2, label='Behind schedule')
        ax2.fill_between(action_indices, 0, time_deviation,
                        where=(time_deviation < 0), color='green', alpha=0.2, label='Ahead of schedule')

        ax2.set_xlabel('Action Index', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Time Deviation (ms)', fontsize=11, fontweight='bold')
        ax2.set_title('Cumulative Timing Deviation from Expected', fontsize=12, fontweight='bold')
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax2.grid(which='major', color='black', linewidth=0.5, alpha=0.3)
        ax2.grid(which='minor', color='gray', linestyle='--', alpha=0.2)
        ax2.legend(loc='best', fontsize=9)

        # Add final deviation text
        final_deviation = time_deviation[-1]
        deviation_text = f'Final deviation: {final_deviation:.1f} ms'
        ax2.text(0.98, 0.98, deviation_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=9, family='monospace')

        plt.tight_layout()
        filename = self.config.debug_output_dir / "action_execution_timing.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Action execution timing plot saved to: {filename}")
        plt.close()

    def _save_debug_data(self):
        """Save all collected debug data at end of execution."""
        if self.debug_data_collection is None:
            return

        try:
            # Create directory structure
            debug_dir = Path('debug_data')
            subdirs = {
                'action_chunks': debug_dir / 'action_chunks',
                'executed_actions': debug_dir / 'actions',
                'eef_poses': debug_dir / 'eef_poses',
                'point_clouds': debug_dir / 'pcd',
                'robot0_eef_pos': debug_dir / 'robot0_eef_pos',
                'actual_eef_pos': debug_dir / 'actual_eef_pos',
                'pcd_timestamps': debug_dir / 'pcd_timestamps'
            }

            for subdir in subdirs.values():
                subdir.mkdir(parents=True, exist_ok=True)

            # Session timestamp for file naming
            session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save each data type
            if len(self.debug_data_collection['action_chunks']) > 0:
                self._save_timestamped_array_data(
                    self.debug_data_collection['action_chunks'],
                    subdirs['action_chunks'] / f'action_chunks_{session_ts}.npy'
                )

            if len(self.debug_data_collection['executed_actions']) > 0:
                self._save_timestamped_array_data(
                    self.debug_data_collection['executed_actions'],
                    subdirs['executed_actions'] / f'executed_actions_{session_ts}.npy'
                )

            if len(self.debug_data_collection['eef_poses']) > 0:
                self._save_eef_poses(
                    self.debug_data_collection['eef_poses'],
                    subdirs['eef_poses'] / f'eef_poses_{session_ts}.npy'
                )

            if len(self.debug_data_collection['point_clouds']) > 0:
                self._save_point_clouds(
                    self.debug_data_collection['point_clouds'],
                    subdirs['point_clouds'],
                    session_ts
                )

            if len(self.debug_data_collection['robot0_eef_pos']) > 0:
                self._save_timestamped_array_data(
                    self.debug_data_collection['robot0_eef_pos'],
                    subdirs['robot0_eef_pos'] / f'robot0_eef_pos_{session_ts}.npy'
                )

            if len(self.debug_data_collection['actual_eef_pos']) > 0:
                self._save_timestamped_array_data(
                    self.debug_data_collection['actual_eef_pos'],
                    subdirs['actual_eef_pos'] / f'actual_eef_pos_{session_ts}.npy'
                )

            if len(self.debug_data_collection['pcd_timestamps']) > 0:
                self._save_timestamped_array_data(
                    self.debug_data_collection['pcd_timestamps'],
                    subdirs['pcd_timestamps'] / f'pcd_timestamps_{session_ts}.npy'
                )

            # Summary
            print(f"\n{'='*60}")
            print(f"Debug data saved: {debug_dir.absolute()}")
            print(f"  Session: {session_ts}")
            print(f"  Action chunks: {len(self.debug_data_collection['action_chunks'])} samples")
            print(f"  Executed actions: {len(self.debug_data_collection['executed_actions'])} samples")
            print(f"  EEF poses: {len(self.debug_data_collection['eef_poses'])} samples")
            print(f"  Point clouds: {len(self.debug_data_collection['point_clouds'])} samples")
            print(f"  Robot0 EEF pos: {len(self.debug_data_collection['robot0_eef_pos'])} samples")
            print(f"  Actual EEF pos: {len(self.debug_data_collection['actual_eef_pos'])} samples")
            print(f"  RGBD timestamps: {len(self.debug_data_collection['pcd_timestamps'])} samples")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"WARNING: Failed to save debug data: {e}")
            import traceback
            traceback.print_exc()

    def _save_timestamped_array_data(self, data_list, output_path):
        """Save list of (timestamp_ms, array) tuples."""
        if len(data_list) == 0:
            return

        timestamps = np.array([ts for ts, _ in data_list], dtype=np.int64)
        arrays = np.array([arr for _, arr in data_list])

        np.save(output_path, {
            'timestamps_ms': timestamps,
            'data': arrays
        })

    def _save_eef_poses(self, poses_list, output_path):
        """Save end-effector poses with timestamps."""
        if len(poses_list) == 0:
            return

        timestamps = np.array([ts for ts, _ in poses_list], dtype=np.int64)
        positions = np.array([pose['position'] for _, pose in poses_list], dtype=np.float32)
        quats = np.array([pose['orientation_quat'] for _, pose in poses_list], dtype=np.float32)
        matrices = np.array([pose['orientation_matrix'] for _, pose in poses_list], dtype=np.float32)

        np.save(output_path, {
            'timestamps_ms': timestamps,
            'positions': positions,
            'orientations_quat': quats,
            'orientations_matrix': matrices
        })

    def _save_point_clouds(self, pcd_list, output_dir, session_timestamp):
        """Save point clouds with timestamps.

        Saves all PCDs in single .npy file for easy batch loading.
        """
        if self.config.save_pcd_format == 'npy':
            # Single file with all point clouds
            timestamps = np.array([ts for ts, _ in pcd_list], dtype=np.int64)
            point_clouds = [pcd for _, pcd in pcd_list]
            point_counts = np.array([len(pcd) for pcd in point_clouds], dtype=np.int32)

            output_path = output_dir / f'point_clouds_{session_timestamp}.npy'
            np.save(output_path, {
                'timestamps_ms': timestamps,
                'point_clouds': point_clouds,
                'point_counts': point_counts
            }, allow_pickle=True)

        elif self.config.save_pcd_format == 'ply':
            # Individual PLY files
            import open3d as o3d

            timestamps = []
            filenames = []

            for timestamp_ms, pcd_array in pcd_list:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcd_array[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(pcd_array[:, 3:6])

                filename = f'pcd_{session_timestamp}_{timestamp_ms}.ply'
                o3d.io.write_point_cloud(str(output_dir / filename), pcd)

                timestamps.append(timestamp_ms)
                filenames.append(filename)

            # Save metadata
            np.save(output_dir / f'metadata_{session_timestamp}.npy', {
                'timestamps_ms': np.array(timestamps, dtype=np.int64),
                'filenames': filenames
            })


class BlendingChunkingController(ChunkingController):
    """Controller with action blending in overlap regions."""

    def _setup_debug_plotting(self):
        """Initialize debug plotting with blending-specific data."""
        super()._setup_debug_plotting()
        self.action_queue_snapshots = []

    def _need_inference(self):
        """Check if we need to run policy inference (accounts for merge_range)."""
        return len(self.action_queue) <= self.config.policy_delay + self.config.merge_range

    def _run_inference_async(self):
        """Run inference with blending in background thread."""
        def inference_worker():
            try:
                current_time = time.time()

                if self.last_inference_time is not None:
                    time_since_last = current_time - self.last_inference_time
                    print(f"\n{'='*50}")
                    print(f"[Inference #{self.inference_count}] Time since last: {time_since_last*1000:.1f} ms")
                    print(f"{'='*50}")

                # Record observation arrival
                if self.config.visualize:
                    self.live_events.append((time.time(), 'obs_arrival'))

                # Get observation and predict
                obs_dict = self._get_observation()
                actions = self.policy_client.predict_action(obs_dict)
                inference_time = time.time()

                # Record inference done
                if self.config.visualize:
                    self.live_events.append((inference_time, 'inference_done'))

                # Store for debug plotting
                if self.config.debug_plotting:
                    self.all_predicted_chunks.append((self.inference_count, actions.copy()))

                # Thread-safe queue update with blending
                with self.inference_lock:
                    if self.first_inference_round:
                        # First time: add first action_exec_size actions
                        for action in actions[:self.config.action_exec_size]:
                            self.action_queue.append(action.copy())
                        self.first_inference_round = False
                    else:
                        # Merge overlapping actions
                        start_idx = self.config.policy_delay
                        end_idx = start_idx + self.config.action_exec_size

                        for action_idx, action in enumerate(actions[start_idx:end_idx]):
                            if action_idx < self.config.merge_range:
                                # Blend with existing action in queue
                                prior_weight = (self.config.merge_range - action_idx) / self.config.merge_range
                                prior_weight = prior_weight / 2  # Reduce blending strength

                                queue_idx = -(action_idx + 1)
                                action_merged = (prior_weight * self.action_queue[queue_idx] +
                                                (1 - prior_weight) * action.copy())
                                self.action_queue[queue_idx] = action_merged
                            else:
                                # Append new action
                                self.action_queue.append(action.copy())

                    # Store queue snapshot for debug plotting
                    if self.config.debug_plotting:
                        self.action_queue_snapshots.append(
                            (self.inference_count, [a.copy() for a in self.action_queue])
                        )

                    # Debug data collection
                    if self.debug_data_collection is not None:
                        timestamp_ms = int(inference_time * 1000)

                        # Save action chunk
                        self.debug_data_collection['action_chunks'].append(
                            (timestamp_ms, actions.copy())
                        )

                        # Save point cloud
                        pcd = obs_dict.get('pcd', None)
                        if pcd is not None:
                            self.debug_data_collection['point_clouds'].append(
                                (timestamp_ms, pcd.copy())
                            )

                        # Save robot0_eef_pos
                        robot0_eef_pos = obs_dict.get('robot0_eef_pos', None)
                        if robot0_eef_pos is not None:
                            self.debug_data_collection['robot0_eef_pos'].append(
                                (timestamp_ms, robot0_eef_pos.copy())
                            )

                        # Save RGBD timestamp
                        pcd_timestamp = obs_dict.get('pcd_timestamp', None)
                        if pcd_timestamp is not None:
                            # Ensure we create an independent copy (not a reference)
                            if isinstance(pcd_timestamp, np.ndarray):
                                ts_copy = pcd_timestamp.copy()
                            else:
                                ts_copy = np.array(pcd_timestamp).copy()
                            print(f"Saving RGBD timestamp: {ts_copy}")
                            self.debug_data_collection['pcd_timestamps'].append(
                                (timestamp_ms, ts_copy)
                            )

                    self.last_inference_time = time.time()
                    print(f"Inference took {(self.last_inference_time - current_time)*1000:.1f} ms")
                    self.inference_count += 1

            finally:
                self.inference_in_progress = False

        # Start thread
        self.inference_in_progress = True
        self.inference_thread = threading.Thread(target=inference_worker, daemon=True)
        self.inference_thread.start()

    def run(self, n_steps: int):
        """Execute blending chunking control with async inference."""
        n_steps_done = 0

        if self.config.debug_plotting:
            self.start_time = time.time()

        # Initialize start time for live visualization
        if self.config.visualize:
            self.start_time = time.time()

        # Initial inference (synchronous to populate queue)
        self._run_inference_async()
        while self.inference_in_progress:
            time.sleep(0.005)  # Wait for first inference

        self.first_inference_round = True
        print("\nStarting blending chunking control...")
        while n_steps_done < n_steps:
            loop_start = time.time()

            # Trigger async inference if needed (don't wait)
            if self._need_inference() and not self.inference_in_progress:
                self._run_inference_async()


            # Execute action from queue
            with self.inference_lock:
                queue_has_actions = len(self.action_queue) > 0
                if queue_has_actions:
                    action = self.action_queue.pop(0)

            if queue_has_actions:
                if self.config.debug_plotting:
                    current_pose = self.robot.end_effector_pose
                    rotmat = current_pose.orientation.as_matrix()
                    yaw = np.arctan2(rotmat[1, 0], rotmat[0, 0])
                    elapsed = time.time() - self.start_time
                    self.actual_poses.append((elapsed, current_pose.position[0],
                                             current_pose.position[1],
                                             current_pose.position[2], yaw))
                # Enforce fixed control frequency
                if self.enforce_fixed_freq:
                    self._enforce_fixed_frequency()
                self._execute_action(action)
                self.last_execution_time = time.time() # Update execution time

                # Record action execution
                if self.config.visualize:
                    self.live_events.append((time.time(), 'action_executed'))
                    self._update_live_plot()

                n_steps_done += 1
            else:
                print("Warning: Action queue empty, waiting for inference...")
                time.sleep(0.01)

            if self.config.debug_plotting:
                loop_time = (time.time() - loop_start) * 1000
                self.control_loop_times.append(loop_time)

        # Wait for any ongoing inference to complete
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=5.0)

        # Final plot update and close
        if self.live_plot_fig is not None:
            self._update_live_plot(force=True)  # Final update with all events
            plt.close(self.live_plot_fig)
            self.live_plot_fig = None
            self.live_plot_ax = None

        # Save debug data at end
        if self.debug_data_collection is not None and len(self.debug_data_collection['executed_actions']) > 0:
            self._save_debug_data()

        # Generate debug plots
        if self.config.debug_plotting:
            self._plot_debug_results()
            self._plot_action_chunks()
            self._plot_action_execution_timing()
            print(f"\n{'='*60}")
            print(f"Debug plots saved to: {self.config.debug_output_dir.absolute()}")
            print(f"{'='*60}")

    def _plot_debug_results(self):
        """Generate debug plots showing action blending."""
        # Call parent for timing histogram
        super()._plot_debug_results()

        # Add merged queue visualization
        print("Generating merged action queue plots...")

        if len(self.action_queue_snapshots) == 0:
            print("No action queue snapshots to plot")
            return

        n_snapshots = len(self.action_queue_snapshots)
        colors = cm.rainbow(np.linspace(0, 1, n_snapshots))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Merged Action Queue Snapshots (After Blending)', fontsize=14, fontweight='bold')

        labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)', 'Yaw (rad)']

        for ax, label in zip(axes.flatten(), labels):
            for snap_idx, (inf_idx, action_queue) in enumerate(self.action_queue_snapshots):
                if snap_idx == 0:
                    start_step = 0
                else:
                    start_step = (self.config.action_exec_size - self.config.merge_range) * snap_idx

                x_vals = []
                y_vals = []

                for action_idx, action in enumerate(action_queue):
                    n_step = start_step + action_idx

                    if label == 'X Position (m)':
                        value = action[0]
                    elif label == 'Y Position (m)':
                        value = action[1]
                    elif label == 'Z Position (m)':
                        value = action[2]
                    else:  # Yaw
                        rot6d = action[3:9]
                        rotmat = self.rotation_transformer.forward(rot6d.reshape(1, 6))[0]
                        value = np.arctan2(rotmat[1, 0], rotmat[0, 0])

                    x_vals.append(n_step)
                    y_vals.append(value)

                ax.plot(x_vals, y_vals, 'o-', color=colors[snap_idx],
                       linewidth=2, markersize=3, alpha=0.7,
                       label=f'After Inference #{inf_idx}')

            ax.set_xlabel('n_step', fontsize=11, fontweight='bold')
            ax.set_ylabel(label, fontsize=11, fontweight='bold')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            ax.grid(which='major', color='black', linewidth=1)
            ax.grid(which='minor', color='gray', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8, ncol=2)

        plt.tight_layout()
        filename = self.config.debug_output_dir / "merged_action_queue.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Merged action queue plot saved to: {filename}")
        plt.close()




def create_controller(mode: str, *args, **kwargs) -> RobotController:
    """Factory function to create controller based on mode.

    Args:
        mode: Controller mode ('simple', 'chunking', 'blending', 'intv')
        *args, **kwargs: Arguments passed to controller constructor

    Returns:
        RobotController instance
    """
    if mode == 'simple':
        return SimpleSequentialController(*args, **kwargs)
    elif mode == 'chunking':
        return ChunkingController(*args, **kwargs)
    elif mode == 'blending':
        return BlendingChunkingController(*args, **kwargs)
    elif mode == 'intv':
        return InterventionController(*args, **kwargs)
    else:
        raise ValueError(f"Unknown controller mode: {mode}")
