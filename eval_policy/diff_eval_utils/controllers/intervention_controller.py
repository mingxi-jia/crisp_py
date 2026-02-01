from diff_eval_utils.controllers.simple_controller import SimpleSequentialController
from diff_eval_utils.diffusion_transforms import ten_d_action_to_pose, convert_action_from_fingertip_to_gripper
import time
import numpy as np
from pathlib import Path
import threading
from scipy.spatial.transform import Rotation as R
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

    Controller Switching:
    - POLICY_MODE: Uses controller specified in config (joint or cartesian)
    - INTERVENTION_MODE: Always uses cartesian impedance controller for spacemouse
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

        # Action recording for debugging
        self.recorded_actions = []  # List of executed actions

        self.ACTION_SCALE = self.config.spacemouse_action_scale
        self.DEADZONE = self.config.spacemouse_deadzone

        self.policy_n_interpolation = self.config.n_interpolation
        self.intervention_n_interpolation = self.config.teleop['n_interpolation']

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
        # print(f"[PUBLISH] Publishing intervention state: {state}")
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

            # Ensure impedance controller is active
            self._switch_to_impedance_controller()

            self.recording_state = self.RECORDING
            self.mode = self.POLICY_MODE  # Reset to policy mode
            self.intervention_target_pose = self.robot.end_effector_pose.copy()

            # Reset action recording
            self.recorded_actions = []

            print(f"[STATE] ready -> recording (policy mode)")
            print(f"{'='*60}\n")

        elif self.recording_state == self.RECORDING:
            # Stop recording and reset: RECORDING -> READY
            print(f"\n{'='*60}")
            print("[STOP RECORDING] Stopping robot actions and resetting to start position...")
            print(f"{'='*60}")

            # Save recorded actions to .npy file
            if len(self.recorded_actions) > 0:
                self._save_recorded_actions()

            # Reset robot to start position
            self.robot.home()

            # Switch back to impedance controller (home() may have switched to position controller)
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

    def _execute_spacemouse_action(self):
        new_pos = self.intervention_target_pose.position
        new_orientation = self.intervention_target_pose.orientation


        print(new_pos)

        if self.control_space == 'cartesian':
            self._execute_cartesian_action(new_pos, new_orientation)
        elif self.control_space == 'joint':
            self._execute_joint_action(new_pos, new_orientation)
        else: 
            raise NotImplementedError(f"Unknown control space: {self.control_space}")

    def _execute_intervention_step(self, spacemouse, recording=True):
        """Execute one step of spacemouse control using cartesian action execution.

        Uses a similar pattern to base controller's _execute_cartesian_action but
        adapted for delta-based spacemouse control without interpolation.

        Args:
            spacemouse: Spacemouse instance
            recording: Whether recording is active (affects ROS message publishing)
        """
        from std_msgs.msg import Int32MultiArray

        # Get spacemouse motion and button state
        motion = spacemouse.get_motion_state_transformed()
        dx, dy, dz, droll, dpitch, dyaw = motion * self.ACTION_SCALE
        button_pressed = spacemouse.is_button_pressed(0)

        # Detect gripper toggle event
        gripper_toggle = 1 if (button_pressed and not self.prev_button_pressed) else 0

        # Check if there's any movement
        has_movement = (dx != 0 or dy != 0 or dz != 0 or droll != 0 or dpitch != 0 or dyaw != 0)

        # Early return only if no movement AND no gripper action
        if not has_movement and not gripper_toggle:
            self._publish_intervention_state(0)
            self._execute_spacemouse_action()
            return

        # Publish intervention signals only when recording (for data collection)
        if recording and (has_movement or gripper_toggle):
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

        # Update target pose using delta control
        curr_pos = self.intervention_target_pose.position
        curr_euler = self.intervention_target_pose.orientation.as_euler('XYZ')

        # Apply deltas to compute new target position
        new_pos = np.array([
            curr_pos[0] + dx,
            curr_pos[1] + dy,
            max(curr_pos[2] + dz, 0.02)  # Safety: don't go below table
        ])

        new_euler = np.array([
            curr_euler[0] + droll * 1,
            curr_euler[1] - dpitch * 1,
            curr_euler[2] - dyaw * 2  # Match spacemouse_example.py convention
        ])
        new_orientation = R.from_euler('XYZ', new_euler)

        self.intervention_target_pose.position = new_pos
        self.intervention_target_pose.orientation = new_orientation

        self.n_interpolation = self.intervention_n_interpolation

        self._execute_spacemouse_action()
        # Update intervention target pose

        # Only print if there's movement
        if has_movement:
            print(f"[INTERVENTION] x={new_pos[0]:.4f} y={new_pos[1]:.4f} z={new_pos[2]:.4f} "
                  f"roll={new_euler[0]:.4f} pitch={new_euler[1]:.4f} yaw={new_euler[2]:.4f}")

        # Handle gripper using base controller's gripper execution pattern
        if gripper_toggle:
            new_gripper_value = 1.0 - self.prev_grasp_value
            print(f"[GRIPPER] {'Closing' if new_gripper_value == 1.0 else 'Opening'} gripper...")
            self._execute_gripper_action(new_gripper_value)

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
        pose_action, grasp_action = ten_d_action_to_pose(action)
        new_position, gripper_pose = convert_action_from_fingertip_to_gripper(pose_action)

        # Position deltas
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

        self.n_interpolation = self.policy_n_interpolation

        print(f"self.n_interpolation: {self.n_interpolation}")

        if self.control_space == 'cartesian':
            self._execute_cartesian_action(new_position, gripper_pose, move_to=move_to)

        elif self.control_space == 'joint':
            assert move_to == False, "move_to not supported in joint control space"
            self._execute_joint_action(new_position, gripper_pose)
        
        else:
            raise NotImplementedError(f"Invalid control space: {self.control_space}")
        
        self._execute_gripper_action(grasp_action)

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
                                first = False
                                action_idx += 1
                            
                                motion = sm.get_motion_state_transformed()
                                if self._check_for_intervention_trigger(motion):
                                    print(f"\n{'='*60}")
                                    print("[INTERVENTION TRIGGERED] Spacemouse movement detected! Switching Controller")
                                    time.sleep(1)
                                    print(f"{'='*60}")

                                    self.mode = self.INTERVENTION_MODE
                                    self.robot

                                    # Initialize intervention target from current pose
                                    self.intervention_target_pose = self.robot.end_effector_pose.copy()
                                    # Invalidate current actions (will get fresh observation on resume)
                                    current_actions = None
                                    action_idx = 0
                                    break

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
