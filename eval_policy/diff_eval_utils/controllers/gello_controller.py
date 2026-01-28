"""Gello controller that listens to joint control signals from ROS2 topic and executes them."""

from diff_eval_utils.controllers.base_controller import RobotController
import time
import numpy as np
import threading
from sensor_msgs.msg import JointState


# Default maximum joint velocity (rad/s) - conservative value for safety
DEFAULT_MAX_JOINT_VELOCITY = 1.0  # rad/s

# Calibration threshold - max allowed difference between gello and robot joints
CALIBRATION_THRESHOLD = 0.08  # rad

# Default end effector safety bounds (meters) - fallback if not in config
DEFAULT_EEF_BOUNDS = {
    'x': (0.3, 0.8),
    'y': (-0.35, 0.35),
    'z': (0.0, 0.61),
}

# Default joint delta threshold (rad) - fallback if not in config
DEFAULT_JOINT_DELTA_THRESHOLD = 0.2


class GelloController(RobotController):
    """Controller that subscribes to gello_control_signal topic and executes joint commands.

    This controller listens to joint position commands published on the 'gello_control_signal'
    topic and sends them to the robot's joint trajectory controller.

    Velocity limiting is enforced by computing the minimum time required to reach the target
    position without exceeding the maximum joint velocity.

    Key Controls:
    - 'r': Reset robot to home position and wait for gello calibration
    """

    def __init__(self, *args, max_joint_velocity: float = DEFAULT_MAX_JOINT_VELOCITY,
                 min_time_to_goal: float = 0.1,
                 calibration_threshold: float = CALIBRATION_THRESHOLD, **kwargs):
        """Initialize the GelloController.

        Args:
            *args: Arguments passed to base controller
            max_joint_velocity: Maximum allowed joint velocity in rad/s (default: 1.0)
            min_time_to_goal: Minimum time in seconds for any movement (default: 0.1)
            calibration_threshold: Max allowed joint difference for calibration (default: 0.02 rad)
            **kwargs: Keyword arguments passed to base controller
        """
        super().__init__(*args, **kwargs)

        # Gello command state
        self._latest_joint_command = None
        self._latest_joint_names = None
        self._command_lock = threading.Lock()
        self._new_command_event = threading.Event()

        # Velocity limiting configuration
        self.max_joint_velocity = max_joint_velocity
        self.min_time_to_goal = min_time_to_goal
        self.calibration_threshold = calibration_threshold

        # ROS2 subscriber for gello control signals
        self._gello_sub = None

        # Keyboard listener state
        self.key_commands = {'r': False}
        self.listener_lock = threading.Lock()
        self.keyboard_listener = None

        # Safety state
        self._safety_stop = False

        # Safety bounds from config (with fallbacks)
        self.eef_bounds = getattr(self.config, 'eef_bounds', DEFAULT_EEF_BOUNDS)
        self.joint_delta_threshold = getattr(self.config, 'joint_delta_threshold', DEFAULT_JOINT_DELTA_THRESHOLD)
        self.home_joint_position = np.array(self.config.home_joint_position[:7])

    def _setup_gello_subscriber(self):
        """Create ROS2 subscriber for gello control signals."""
        self._gello_sub = self.robot.node.create_subscription(
            JointState,
            'gello_control_signal',
            self._gello_callback,
            10
        )

    def _setup_keyboard_listener(self):
        """Initialize keyboard listener for 'r' key (reset)."""
        from pynput import keyboard

        def on_press(key):
            with self.listener_lock:
                try:
                    if hasattr(key, 'char'):
                        if key.char == 'r':
                            self.key_commands['r'] = True
                except AttributeError:
                    pass

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()

    def _check_key_command(self, key):
        """Check if a key command was triggered and reset flag."""
        with self.listener_lock:
            if self.key_commands[key]:
                self.key_commands[key] = False
                return True
        return False

    def _handle_reset(self):
        """Handle 'r' key press: reset robot to home position and wait for calibration."""
        print(f"\n{'='*60}")
        print("[RESET] Resetting robot to home position...")
        print(f"{'='*60}")

        # Reset gripper state (open gripper)
        if self.prev_grasp_value != 0.0:
            self.gripper.set_target(1.0)  # 1.0 = open for Franka
            self.gripper_rate.sleep()
            time.sleep(1.0)
        self.prev_grasp_value = 0.0

        # Reset robot to home position
        self.robot.home()
        self._switch_to_joint_controller()
        time.sleep(0.5)

        # Wait for gello calibration
        self._wait_for_gello_calibration()

        print(f"[RESET] Complete. Resuming gello control...")
        print(f"{'='*60}\n")

    def _wait_for_gello_calibration(self):
        """Wait until gello joint positions are close enough to robot's current positions."""
        print(f"\n[CALIBRATION] Waiting for gello to match robot position...")
        print(f"[CALIBRATION] Threshold: {self.calibration_threshold:.4f} rad")
        print(f"[CALIBRATION] Move the gello to match the robot's current position.\n")

        joint_names = self.robot.config.joint_names[:7]
        check_rate = self.robot.node.create_rate(10.0)

        while True:
            gello_names, gello_positions = self._get_latest_command(with_names=True)
            if gello_names is None:
                print("[CALIBRATION] Waiting for gello signal...", end='\r')
                check_rate.sleep()
                continue

            gello_dict = dict(zip(gello_names, gello_positions))
            robot_positions = self.joint_state_subscriber.joint_values[:7]

            # Calculate differences
            diffs = [gello_dict.get(name, float('nan')) - robot_positions[i]
                     for i, name in enumerate(joint_names)]
            abs_diffs = [abs(d) if not np.isnan(d) else float('inf') for d in diffs]

            # Print table
            print("\033[K" + f"{'Joint':<20} {'Diff':>10} {'Robot':>10} {'Gello':>10} {'Status':>8}")
            print("\033[K" + "-" * 60)
            for i, name in enumerate(joint_names):
                robot_val = robot_positions[i]
                gello_val = gello_dict.get(name)
                if gello_val is not None:
                    status = "OK" if abs_diffs[i] < self.calibration_threshold else "MOVE"
                    print(f"\033[K{name:<20} {diffs[i]:>10.4f} {robot_val:>10.4f} {gello_val:>10.4f} {status:>8}")
                else:
                    print(f"\033[K{name:<20} {'N/A':>10} {robot_val:>10.4f} {'N/A':>10} {'MISSING':>8}")

            if all(d < self.calibration_threshold for d in abs_diffs):
                print("\n[CALIBRATION] All joints calibrated! Starting control...")
                break

            print(f"\033[{len(joint_names) + 3}A", end='')
            check_rate.sleep()

    def _check_eef_safety(self) -> tuple[bool, str]:
        """Check if end effector position is within safe bounds.

        Returns:
            Tuple of (is_safe, message):
            - is_safe: True if within bounds, False if out of bounds
            - message: Description of which bound was violated (empty if safe)
        """
        eef_pose = self.robot.end_effector_pose
        x, y, z = eef_pose.position

        violations = []

        # Check X bounds
        if x < self.eef_bounds['x'][0]:
            violations.append(f"X={x:.3f}m < {self.eef_bounds['x'][0]}m (min)")
        elif x > self.eef_bounds['x'][1]:
            violations.append(f"X={x:.3f}m > {self.eef_bounds['x'][1]}m (max)")

        # Check Y bounds
        if y < self.eef_bounds['y'][0]:
            violations.append(f"Y={y:.3f}m < {self.eef_bounds['y'][0]}m (min)")
        elif y > self.eef_bounds['y'][1]:
            violations.append(f"Y={y:.3f}m > {self.eef_bounds['y'][1]}m (max)")

        # Check Z bounds
        if z < self.eef_bounds['z'][0]:
            violations.append(f"Z={z:.3f}m < {self.eef_bounds['z'][0]}m (min)")
        elif z > self.eef_bounds['z'][1]:
            violations.append(f"Z={z:.3f}m > {self.eef_bounds['z'][1]}m (max)")

        if violations:
            return False, "; ".join(violations)
        return True, ""

    def _check_joint_safety(self) -> tuple[bool, str]:
        """Check if joint positions are within safe delta from home pose.

        Returns:
            Tuple of (is_safe, message):
            - is_safe: True if within threshold, False if exceeded
            - message: Description of which joint violated (empty if safe)
        """
        current_joints = np.array(self.joint_state_subscriber.joint_values[:7])
        deltas = np.abs(current_joints - self.home_joint_position)

        violations = []
        joint_names = self.robot.config.joint_names[:7]

        for i, (delta, name) in enumerate(zip(deltas, joint_names)):
            if delta > self.joint_delta_threshold:
                violations.append(
                    f"J{i}({name[:8]}): delta={delta:.3f}rad > {self.joint_delta_threshold}rad"
                )

        if violations:
            return False, "; ".join(violations)
        return True, ""

    def _check_safety(self) -> tuple[bool, str]:
        """Check all safety conditions.

        Returns:
            Tuple of (is_safe, message):
            - is_safe: True if all checks pass, False otherwise
            - message: Description of violations (empty if safe)
        """
        all_violations = []

        # Check EEF bounds
        eef_safe, eef_msg = self._check_eef_safety()
        if not eef_safe:
            all_violations.append(f"[EEF] {eef_msg}")

        # Check joint delta from home
        joint_safe, joint_msg = self._check_joint_safety()
        if not joint_safe:
            all_violations.append(f"[JOINT] {joint_msg}")

        if all_violations:
            return False, " | ".join(all_violations)
        return True, ""

    def _handle_safety_stop(self, violation_msg: str):
        """Handle safety stop when safety bounds are violated.

        Args:
            violation_msg: Description of the safety violation
        """
        self._safety_stop = True
        print(f"\n{'!'*60}")
        print("[SAFETY STOP] Safety bounds violated!")
        print(f"{'!'*60}")
        print(f"Violation: {violation_msg}")
        print(f"\nSafe bounds:")
        print(f"  EEF X: [{self.eef_bounds['x'][0]}, {self.eef_bounds['x'][1]}] m")
        print(f"  EEF Y: [{self.eef_bounds['y'][0]}, {self.eef_bounds['y'][1]}] m")
        print(f"  EEF Z: [{self.eef_bounds['z'][0]}, {self.eef_bounds['z'][1]}] m")
        print(f"  Joint delta from home: {self.joint_delta_threshold} rad")
        eef_pose = self.robot.end_effector_pose
        current_joints = np.array(self.joint_state_subscriber.joint_values[:7])
        joint_deltas = np.abs(current_joints - self.home_joint_position)
        print(f"\nCurrent state:")
        print(f"  EEF position: X={eef_pose.position[0]:.3f}, Y={eef_pose.position[1]:.3f}, Z={eef_pose.position[2]:.3f}")
        print(f"  Joint deltas from home: {np.round(joint_deltas, 3)}")
        print(f"\nControl STOPPED. Press 'r' to reset robot to home and recalibrate.")
        print(f"{'!'*60}\n")

    def _gello_callback(self, msg: JointState):
        """Callback for gello control signal messages.

        Args:
            msg: JointState containing joint names and positions
        """
        with self._command_lock:
            self._latest_joint_command = np.array(msg.position, dtype=np.float64)
            self._latest_joint_names = list(msg.name)
        self._new_command_event.set()

    def _get_latest_command(self, with_names: bool = False):
        """Get the latest joint command (thread-safe).

        Args:
            with_names: If True, return (names, positions) tuple

        Returns:
            If with_names=False: numpy array of joint positions or None
            If with_names=True: tuple of (joint_names, joint_positions) or (None, None)
        """
        with self._command_lock:
            if self._latest_joint_command is not None and self._latest_joint_names is not None:
                if with_names:
                    return (list(self._latest_joint_names), self._latest_joint_command.copy())
                return self._latest_joint_command.copy()
            if with_names:
                return (None, None)
            return None

    def _compute_safe_time_to_goal(self, current_positions: np.ndarray,
                                    target_positions: np.ndarray) -> float:
        """Compute the minimum time required to reach target without exceeding max velocity.

        Args:
            current_positions: Current joint positions (rad)
            target_positions: Target joint positions (rad)

        Returns:
            Time in seconds required to complete the movement safely
        """
        # Calculate the maximum joint displacement
        displacements = np.abs(target_positions - current_positions)
        max_displacement = np.max(displacements)

        # Calculate minimum time required based on max velocity
        # time = distance / velocity
        required_time = max_displacement / self.max_joint_velocity

        # Use the larger of required time or minimum time
        safe_time = max(required_time, self.min_time_to_goal)

        return safe_time

    def _execute_joint_command(self, joint_positions: np.ndarray, blocking: bool = False):
        """Execute a joint position command with velocity limiting.

        Args:
            joint_positions: Array of joint positions (should match robot's joint count)
            blocking: Whether to wait for the movement to complete
        """
        # Validate joint positions length
        expected_joints = len(self.robot.config.joint_names)
        if len(joint_positions) < expected_joints:
            joint_positions = np.pad(
                joint_positions,
                (0, expected_joints - len(joint_positions)),
                mode='constant'
            )
        elif len(joint_positions) > expected_joints:
            joint_positions = joint_positions[:expected_joints]

        # Get current joint positions and compute safe time
        current_positions = np.array(self.robot.joint_values)
        time_to_goal = self._compute_safe_time_to_goal(current_positions, joint_positions)

        # Send joint configuration
        self.robot.joint_trajectory_controller_client.send_joint_config(
            self.robot.config.joint_names,
            joint_positions.tolist(),
            time_to_goal=time_to_goal,
            blocking=blocking
        )

    def _switch_to_joint_controller(self):
        """Switch to joint trajectory controller for joint space control."""
        self.robot.controller_switcher_client.switch_controller("joint_trajectory_controller")

    def run(self, n_steps: int = None):
        """Execute gello control loop.

        Listens to joint commands from the gello_control_signal topic and executes them.

        Args:
            n_steps: Maximum number of steps to execute (None for indefinite)
        """
        # Setup
        self._setup_gello_subscriber()
        self._setup_keyboard_listener()
        self._switch_to_joint_controller()

        # Wait for initial gello calibration before starting
        self._wait_for_gello_calibration()

        n_steps_done = 0

        try:
            while n_steps is None or n_steps_done < n_steps:
                # Check for 'r' key (reset)
                if self._check_key_command('r'):
                    self._handle_reset()
                    self._safety_stop = False  # Clear safety stop after reset
                    continue

                # If in safety stop, don't execute commands
                if self._safety_stop:
                    # Clear any pending commands
                    self._new_command_event.clear()
                    time.sleep(0.1)
                    continue

                # Check all safety conditions before processing commands
                is_safe, violation_msg = self._check_safety()
                if not is_safe:
                    self._handle_safety_stop(violation_msg)
                    continue

                # Wait for new command with timeout
                if self._new_command_event.wait(timeout=0.1):
                    self._new_command_event.clear()

                    # Get and execute the latest command
                    joint_command = self._get_latest_command()
                    if joint_command is not None:
                        self._execute_joint_command(joint_command, blocking=False)
                        n_steps_done += 1

                        # Check safety again after execution
                        is_safe, violation_msg = self._check_safety()
                        if not is_safe:
                            self._handle_safety_stop(violation_msg)
                            continue

                        # Update observation buffer
                        self._update_buffer()

        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up resources."""
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
        if self._gello_sub is not None:
            self.robot.node.destroy_subscription(self._gello_sub)
