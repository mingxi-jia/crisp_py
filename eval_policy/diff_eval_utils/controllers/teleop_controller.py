from diff_eval_utils.controllers.base_controller import RobotController
import time
import copy
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Int32MultiArray
from diff_eval_utils.diffusion_transforms import get_pose_from_robot, convert_action_from_fingertip_to_gripper

# class TeleopController(RobotController):
#     """Pure teleoperation controller using spacemouse without policy inference.

#     Key Controls:
#     - 'r': Reset robot to home position
#     - Spacemouse: Control robot movement
#     - Spacemouse button: Toggle gripper
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # Spacemouse (initialized in run() using context manager)
#         self.spacemouse = None

#         # Keyboard listener state
#         self.key_commands = {'r': False}
#         self.listener_lock = threading.Lock()
#         self.keyboard_listener = None

#         # ROS2 publisher for teleop signals
#         self.teleop_pub = None

#         # Target pose tracking
#         self.teleop_target_pose = None

#         # Gripper button state tracking
#         self.prev_button_pressed = False

#         self.ACTION_SCALE = self.config.spacemouse_action_scale
#         self.DEADZONE = self.config.spacemouse_deadzone

#     def _setup_keyboard_listener(self):
#         """Initialize keyboard listener for 'r' key."""
#         from pynput import keyboard

#         def on_press(key):
#             with self.listener_lock:
#                 try:
#                     if hasattr(key, 'char'):
#                         if key.char == 'r':
#                             self.key_commands['r'] = True
#                             print("\n[KEY: r] Reset requested...")
#                 except AttributeError:
#                     pass

#         self.keyboard_listener = keyboard.Listener(on_press=on_press)
#         self.keyboard_listener.start()
#         print("Keyboard listener started:")
#         print("  'r' - Reset robot to home position")

#     def _setup_teleop_publisher(self):
#         """Create ROS2 publisher for teleop signals."""
#         self.teleop_pub = self.robot.node.create_publisher(
#             Int32MultiArray,
#             '/teleop/signals',
#             30
#         )
#         print("Teleop signal publisher created on /teleop/signals")

#     def _publish_teleop_state(self, state: int):
#         """Publish teleop state (0=idle, 2=teleop command).

#         Args:
#             state: 0 (idle), 2 (teleop command)
#         """
#         msg = Int32MultiArray()
#         msg.data = [state, 0, 0, 0, 0, 0, 0, 0, 0]
#         self.teleop_pub.publish(msg)

#     def _cleanup(self):
#         """Clean up resources."""
#         if self.keyboard_listener is not None:
#             self.keyboard_listener.stop()
#             print("Keyboard listener stopped")

#     def _check_key_command(self, key):
#         """Check if a key command was triggered and reset flag."""
#         with self.listener_lock:
#             if self.key_commands[key]:
#                 self.key_commands[key] = False
#                 return True
#         return False

#     def _handle_reset(self):
#         """Handle 'r' key press: reset robot to home position."""
#         print(f"\n{'='*60}")
#         print("[RESET] Resetting robot to home position...")
#         print(f"{'='*60}")

#         # Reset gripper state (open gripper)
#         if self.prev_grasp_value != 0.0:
#             print("[RESET] Opening gripper...")
#             self.gripper.set_target(1.0)  # 1.0 = open for Franka
#             self.gripper_rate.sleep()
#             time.sleep(1.0)
#         self.prev_grasp_value = 0.0

#         # Reset robot to start position
#         self.robot.home()
#         self._switch_to_impedance_controller()

#         # Update target_pose to match the new robot position after homing
#         time.sleep(1.0)
#         self.target_pose = self.robot.end_effector_pose.copy()
#         self.teleop_target_pose = self.target_pose.copy()

#         print(f"[RESET] Complete. Resuming teleoperation...")
#         print(f"{'='*60}\n")

#     def _execute_spacemouse_action(self, new_pos, new_orient, gripper_toggle, button_pressed):

#         if self.control_space == 'cartesian':
#             self._execute_cartesian_action(new_pos, new_orient)
#         elif self.control_space == 'joint':
#             self._execute_joint_action(new_pos, new_orient)
#         else: 
#             raise NotImplementedError(f"Unknown control space: {self.control_space}")

#         # Handle gripper button toggle
#         if gripper_toggle:
#             # Toggle gripper state
#             new_gripper_value = 1.0 - self.prev_grasp_value
#             self._execute_gripper_action(new_gripper_value)
#             # time.sleep(1.0)  # Wait for gripper (Franka driver limitation)
#             self.prev_grasp_value = new_gripper_value

#         # Update button state for next iteration
#         self.prev_button_pressed = button_pressed

#     def _execute_teleop_step(self, spacemouse):
#         """Execute one step of spacemouse control.

#         Args:
#             spacemouse: Spacemouse instance
#         """
#         # Get spacemouse motion and button state
#         motion = spacemouse.get_motion_state_transformed()
#         dx, dy, dz, droll, dpitch, dyaw = motion * self.ACTION_SCALE
#         if dz > 0:
#             dz = dz * 1.5
#         button_pressed = spacemouse.is_button_pressed(0)

#         # Detect gripper toggle event
#         gripper_toggle = 1 if (button_pressed and not self.prev_button_pressed) else 0

#         # Check if there's any movement
#         has_movement = (dx != 0 or dy != 0 or dz != 0 or droll != 0 or dpitch != 0 or dyaw != 0)

#         if not has_movement and not gripper_toggle:
#             # No input, publish idle state
#             self._publish_teleop_state(0)
#             self.arm_rate.sleep()
#             self.prev_button_pressed = button_pressed
#             return

#         # Publish teleop command with movement/gripper data
#         # Convert to discrete signals (sign only)
#         dx_int = (1 if dx > 0 else -1 if dx < 0 else 0)
#         dy_int = (1 if dy > 0 else -1 if dy < 0 else 0)
#         dz_int = (1 if dz > 0 else -1 if dz < 0 else 0)
#         droll_int = (1 if droll > 0 else -1 if droll < 0 else 0)
#         dpitch_int = (1 if dpitch > 0 else -1 if dpitch < 0 else 0)
#         dyaw_int = (1 if dyaw > 0 else -1 if dyaw < 0 else 0)

#         # Format: [state, dx, dy, dz, droll, dpitch, dyaw, gripper, reset]
#         # state=2 means teleop command
#         msg = Int32MultiArray()
#         msg.data = [2, dx_int, dy_int, dz_int, droll_int, dpitch_int, dyaw_int, gripper_toggle, 0]
#         self.teleop_pub.publish(msg)

#         # Update target pose
#         # curr_pos = self.teleop_target_pose.position
#         curr_pos = copy.deepcopy(self.robot.end_effector_pose.position)
#         # curr_euler = self.teleop_target_pose.orientation.as_euler('XYZ')
#         curr_euler = copy.deepcopy(self.robot.end_effector_pose.orientation.as_euler('XYZ'))
#         # Apply deltas
#         new_pos = np.array([
#             curr_pos[0] + dx,
#             curr_pos[1] + dy,
#             max(curr_pos[2] + dz, 0.02)  # Safety: don't go below table
#         ])

#         new_euler = np.array([
#             curr_euler[0] + droll * 10,
#             curr_euler[1] - dpitch * 2,
#             curr_euler[2] - dyaw * 10  # Match spacemouse_example.py convention
#         ])

#         new_orient = R.from_euler('XYZ', new_euler)

#         self._execute_spacemouse_action(new_pos, new_orient, gripper_toggle, button_pressed)
#         self.new_euler = new_euler

#     def run(self, n_steps: int = None):
#         """Execute pure teleoperation control.

#         Args:
#             n_steps: Not used (runs indefinitely until Ctrl+C)
#         """
#         from crisp_py.spacemouse import Spacemouse

#         # Setup
#         self._setup_keyboard_listener()
#         self._setup_teleop_publisher()

#         print("\n" + "="*60)
#         print("TELEOPERATION MODE ACTIVE")
#         print("="*60)
#         print("Controls:")
#         print("  'r' - Reset robot to home position")
#         print("  Spacemouse - Control robot movement")
#         print("  Spacemouse button - Toggle gripper")
#         print("  Ctrl+C - Exit")
#         print("="*60 + "\n")

#         try:
#             with Spacemouse(deadzone=self.DEADZONE) as sm:
#                 self.spacemouse = sm
#                 self.teleop_target_pose = self.robot.end_effector_pose.copy()

#                 while True:
#                     # Check for 'r' key (reset)
#                     if self._check_key_command('r'):
#                         self._handle_reset()

#                     # Execute spacemouse control
#                     self._execute_teleop_step(sm)

#         except KeyboardInterrupt:
#             print("\n\nTeleoperation stopped by user")
#         finally:
#             self._cleanup()


class TeleopController(RobotController):
    """Pure teleoperation controller using spacemouse without policy inference.

    Key Controls:
    - 'r': Reset robot to home position
    - Spacemouse: Control robot movement
    - Spacemouse button: Toggle gripper
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Spacemouse (initialized in run() using context manager)
        self.spacemouse = None

        # Keyboard listener state
        self.key_commands = {'r': False}
        self.listener_lock = threading.Lock()
        self.keyboard_listener = None

        # ROS2 publisher for teleop signals
        self.teleop_pub = None

        # Target pose tracking
        self.teleop_target_pose = None

        # Gripper button state tracking
        self.prev_button_pressed = False

        self.ACTION_SCALE = self.config.spacemouse_action_scale
        self.DEADZONE = self.config.spacemouse_deadzone

        # Override for teleop: no interpolation for fast response
        teleop_config = getattr(self.config, 'teleop', {})
        self.n_interpolation = teleop_config.get('n_interpolation', 0)


        # Loop frequency tracking
        self._loop_count = 0
        self._freq_log_interval = 5.0  # Log every 5 seconds
        self._last_freq_log_time = None

    def _setup_keyboard_listener(self):
        """Initialize keyboard listener for 'r' key."""
        from pynput import keyboard

        def on_press(key):
            with self.listener_lock:
                try:
                    if hasattr(key, 'char'):
                        if key.char == 'r':
                            self.key_commands['r'] = True
                            print("\n[KEY: r] Reset requested...")
                except AttributeError:
                    pass

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
        print("Keyboard listener started:")
        print("  'r' - Reset robot to home position")

    def _setup_teleop_publisher(self):
        """Create ROS2 publisher for teleop signals."""
        self.teleop_pub = self.robot.node.create_publisher(
            Int32MultiArray,
            '/teleop/signals',
            30
        )
        print("Teleop signal publisher created on /teleop/signals")

    def _publish_teleop_state(self, state: int):
        """Publish teleop state (0=idle, 2=teleop command).

        Args:
            state: 0 (idle), 2 (teleop command)
        """
        msg = Int32MultiArray()
        msg.data = [state, 0, 0, 0, 0, 0, 0, 0, 0]
        self.teleop_pub.publish(msg)

    def _cleanup(self):
        """Clean up resources."""
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
            print("Keyboard listener stopped")

    def _log_loop_frequency(self):
        """Log the control loop frequency every _freq_log_interval seconds."""
        current_time = time.time()

        # Initialize on first call
        if self._last_freq_log_time is None:
            self._last_freq_log_time = current_time
            self._loop_count = 0
            return

        self._loop_count += 1
        elapsed = current_time - self._last_freq_log_time

        if elapsed >= self._freq_log_interval:
            frequency = self._loop_count / elapsed
            print(f"[FREQ] Control loop: {frequency:.1f} Hz (avg over {elapsed:.1f}s)")
            self._last_freq_log_time = current_time
            self._loop_count = 0

    def _check_key_command(self, key):
        """Check if a key command was triggered and reset flag."""
        with self.listener_lock:
            if self.key_commands[key]:
                self.key_commands[key] = False
                return True
        return False

    def _handle_reset(self):
        """Handle 'r' key press: reset robot to home position."""
        print(f"\n{'='*60}")
        print("[RESET] Resetting robot to home position...")
        print(f"{'='*60}")

        # Reset gripper state (open gripper)
        if self.prev_grasp_value != 0.0:
            print("[RESET] Opening gripper...")
            self.gripper.set_target(1.0)  # 1.0 = open for Franka
            self.gripper_rate.sleep()
            time.sleep(1.0)
        self.prev_grasp_value = 0.0

        # Reset robot to start position
        self.robot.home()
        self._switch_to_impedance_controller()

        # Update target_pose to match the new robot position after homing
        time.sleep(1.0)
        self.target_pose = self.robot.end_effector_pose.copy()
        self.teleop_target_pose = self.target_pose.copy()

        print(f"[RESET] Complete. Resuming teleoperation...")
        print(f"{'='*60}\n")

    def _execute_spacemouse_action(self, new_pos, new_orient, gripper_toggle, button_pressed):
        pose_to_act = copy.deepcopy(self.teleop_target_pose)

        # Convert fingertip pose to gripper frame
        gripper_position, gripper_rotation = convert_action_from_fingertip_to_gripper(pose_to_act)
        # print(gripper_position, pose_to_act.position)
        if self.control_space == 'cartesian':
            self._execute_cartesian_action(gripper_position, gripper_rotation)
        elif self.control_space == 'joint':
            self._execute_joint_action(gripper_position, gripper_rotation)
        else: 
            raise NotImplementedError(f"Unknown control space: {self.control_space}")

        # Handle gripper button toggle
        if gripper_toggle:
            # Toggle gripper state
            new_gripper_value = 1.0 - self.prev_grasp_value
            self._execute_gripper_action(new_gripper_value)
            # time.sleep(1.0)  # Wait for gripper (Franka driver limitation)
            self.prev_grasp_value = new_gripper_value

        # Update button state for next iteration
        self.prev_button_pressed = button_pressed

    def _execute_teleop_step(self, spacemouse):
        """Execute one step of spacemouse control.

        Args:
            spacemouse: Spacemouse instance
        """
        # Get spacemouse motion and button state
        motion = spacemouse.get_motion_state_transformed()
        dx, dy, dz, droll, dpitch, dyaw = motion * self.ACTION_SCALE
        button_pressed = spacemouse.is_button_pressed(0)

        # Detect gripper toggle event
        gripper_toggle = 1 if (button_pressed and not self.prev_button_pressed) else 0

        # Check if there's any movement
        has_movement = (dx != 0 or dy != 0 or dz != 0 or droll != 0 or dpitch != 0 or dyaw != 0)

        if not has_movement and not gripper_toggle:
            # No input, publish idle state
            self._publish_teleop_state(0)
            # self.arm_rate.sleep()
            self._execute_spacemouse_action(None, None, gripper_toggle, button_pressed)
            self.prev_button_pressed = button_pressed
            return

        # Publish teleop command with movement/gripper data
        # Convert to discrete signals (sign only)
        dx_int = (1 if dx > 0 else -1 if dx < 0 else 0)
        dy_int = (1 if dy > 0 else -1 if dy < 0 else 0)
        dz_int = (1 if dz > 0 else -1 if dz < 0 else 0)
        droll_int = (1 if droll > 0 else -1 if droll < 0 else 0)
        dpitch_int = (1 if dpitch > 0 else -1 if dpitch < 0 else 0)
        dyaw_int = (1 if dyaw > 0 else -1 if dyaw < 0 else 0)

        # Format: [state, dx, dy, dz, droll, dpitch, dyaw, gripper, reset]
        # state=2 means teleop command
        msg = Int32MultiArray()
        msg.data = [2, dx_int, dy_int, dz_int, droll_int, dpitch_int, dyaw_int, gripper_toggle, 0]
        self.teleop_pub.publish(msg)

        # Update target pose
        curr_pos = self.teleop_target_pose.position
        curr_euler = self.teleop_target_pose.orientation.as_euler('XYZ')
        
        # curr_pose = get_pose_from_robot(self.robot.end_effector_pose, ret_pose=True)
        # curr_pos = curr_pose.position
        # curr_euler = curr_pose.orientation.as_euler('XYZ')
        # Apply deltas
        new_pos = np.array([
            curr_pos[0] + dx,
            curr_pos[1] + dy,
            max(curr_pos[2] + dz, 0.02)  # Safety: don't go below table
        ])

        new_euler = np.array([
            curr_euler[0] + droll,
            curr_euler[1] + dpitch,
            curr_euler[2] + dyaw 
        ])
        # print(f"{dx:.3f}, {dy:.3f}, {dz:.3f}, {dpitch:.3f}, {dyaw:.3f}, {droll:.3f}")
        # print(self.teleop_target_pose.position, self.robot.end_effector_pose.position)

        new_orient = R.from_euler('XYZ', new_euler)
        # Update teleop_target_pose for next iteration
        self.teleop_target_pose.position = new_pos
        self.teleop_target_pose.orientation = new_orient

        self._execute_spacemouse_action(new_pos, new_orient, gripper_toggle, button_pressed)
        self.new_euler = new_euler

    def run(self, n_steps: int = None):
        """Execute pure teleoperation control.

        Args:
            n_steps: Not used (runs indefinitely until Ctrl+C)
        """
        from crisp_py.spacemouse import Spacemouse

        # Setup
        self._setup_keyboard_listener()
        self._setup_teleop_publisher()

        print("\n" + "="*60)
        print("TELEOPERATION MODE ACTIVE")
        print("="*60)
        print("Controls:")
        print("  'r' - Reset robot to home position")
        print("  Spacemouse - Control robot movement")
        print("  Spacemouse button - Toggle gripper")
        print("  Ctrl+C - Exit")
        print("="*60 + "\n")

        try:
            with Spacemouse(deadzone=self.DEADZONE) as sm:
                self.spacemouse = sm
                self.teleop_target_pose = get_pose_from_robot(self.robot.end_effector_pose.copy(), ret_pose=True)

                while True:
                    # Check for 'r' key (reset)
                    if self._check_key_command('r'):
                        self._handle_reset()

                    # Execute spacemouse control
                    self._execute_teleop_step(sm)

                    # Log loop frequency
                    self._log_loop_frequency()

        except KeyboardInterrupt:
            print("\n\nTeleoperation stopped by user")
        finally:
            self._cleanup()
