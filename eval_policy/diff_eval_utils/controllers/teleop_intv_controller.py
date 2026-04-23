from diff_eval_utils.controllers.base_controller import RobotController
import time
import copy
import sys
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Int32MultiArray
from diff_eval_utils.diffusion_transforms import get_pose_from_robot, convert_action_from_fingertip_to_gripper


class InterventionMonitor:
    """Async thread that continuously prints intervention prediction to the terminal."""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._initialized = False

        self.intervention_state = None
        self.last_intervention_ms = None
        self.ee_pos = None
        self.ee_rot = None
        self.gripper = None

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    NUM_LINES = 6

    def _render(self):
        def fmt_ms(v):
            return f"{v:.1f} ms" if v is not None else "N/A"

        pos_str = (
            f"[{self.ee_pos[0]:.3f}, {self.ee_pos[1]:.3f}, {self.ee_pos[2]:.3f}]"
            if self.ee_pos is not None else "N/A"
        )
        if self.ee_rot is not None:
            try:
                euler = self.ee_rot.as_euler("xyz", degrees=True)
                rot_str = f"[{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] deg"
            except Exception:
                rot_str = str(self.ee_rot)
        else:
            rot_str = "N/A"

        gripper_str = f"{self.gripper:.2f}" if self.gripper is not None else "N/A"
        intervention_str = str(self.intervention_state) if self.intervention_state is not None else "N/A"

        lines = [
            "┌─────────────────────────────────────────────────┐",
            f"│  Intervention State      : {intervention_str:<22}│",
            f"│  Predict Intervention    : {fmt_ms(self.last_intervention_ms):<22}│",
            f"│  EE Position             : {pos_str:<22}│",
            f"│  EE Rotation (xyz)       : {rot_str:<22}│",
            f"│  Gripper                 : {gripper_str:<22}│",
            "└─────────────────────────────────────────────────┘",
        ]
        return lines

    def _display_loop(self):
        while self._running:
            with self._lock:
                lines = self._render()

            if self._initialized:
                sys.stdout.write(f"\033[{self.NUM_LINES + 1}A\033[J")
            else:
                self._initialized = True

            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(0.1)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)


class TeleopIntvController(RobotController):
    """Teleoperation controller with async intervention prediction monitor.

    Key Controls:
    - 'r': Reset robot to home position
    - Spacemouse: Control robot movement
    - Spacemouse button: Toggle gripper

    An async thread continuously queries the intervention predictor and
    prints the result to the terminal.
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
        self._freq_log_interval = 5.0
        self._last_freq_log_time = None
        self.predict_contact = True

        # Intervention monitor
        self._monitor = InterventionMonitor()
        self._intv_thread = None
        self._intv_running = False

    # ------------------------------------------------------------------
    # Intervention prediction async thread
    # ------------------------------------------------------------------

    def _intervention_loop(self):
        """Background thread: continuously predict intervention and update monitor."""
        while self._intv_running:
            try:
                obs_dict = self._get_observation()
                # print(f"inhand mean: {obs_dict['robot0_eye_in_hand_image'].mean()}")
                t0 = time.time()
                intervention_state = self._predict_intervention(obs_dict)
                elapsed_ms = (time.time() - t0) * 1000

                update_kwargs = dict(
                    intervention_state=intervention_state,
                    last_intervention_ms=elapsed_ms,
                )
                try:
                    ee_pose = self.robot.end_effector_pose
                    update_kwargs.update(
                        ee_pos=np.array(ee_pose.position),
                        ee_rot=ee_pose.orientation,
                        gripper=self.prev_grasp_value,
                    )
                except Exception:
                    pass

                self._monitor.update(**update_kwargs)
            except Exception:
                pass

    def _start_intervention_thread(self):
        if not self.predict_contact:
            return
        self._intv_running = True
        self._intv_thread = threading.Thread(target=self._intervention_loop, daemon=True)
        self._intv_thread.start()

    def _stop_intervention_thread(self):
        self._intv_running = False
        if self._intv_thread:
            self._intv_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Keyboard listener
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Teleop publisher
    # ------------------------------------------------------------------

    def _setup_teleop_publisher(self):
        """Create ROS2 publisher for teleop signals."""
        self.teleop_pub = self.robot.node.create_publisher(
            Int32MultiArray,
            '/teleop/signals',
            30
        )
        print("Teleop signal publisher created on /teleop/signals")

    def _publish_teleop_state(self, state: int):
        msg = Int32MultiArray()
        msg.data = [state, 0, 0, 0, 0, 0, 0, 0, 0]
        self.teleop_pub.publish(msg)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self):
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
            print("Keyboard listener stopped")
        self._stop_intervention_thread()
        self._monitor.stop()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _log_loop_frequency(self):
        current_time = time.time()
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
        with self.listener_lock:
            if self.key_commands[key]:
                self.key_commands[key] = False
                return True
        return False

    def _handle_reset(self):
        print(f"\n{'='*60}")
        print("[RESET] Resetting robot to home position...")
        print(f"{'='*60}")

        if self.prev_grasp_value != 0.0:
            print("[RESET] Opening gripper...")
            self.gripper.set_target(1.0)
            self.gripper_rate.sleep()
        self.prev_grasp_value = 0.0

        self.robot.home()
        self._switch_to_impedance_controller()

        self._last_joint_target = None
        self.target_pose = get_pose_from_robot(self.robot.end_effector_pose.copy(), ret_pose=True)
        self.teleop_target_pose = self.target_pose.copy()

        print(f"[RESET] Complete. Resuming teleoperation...")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Spacemouse action execution
    # ------------------------------------------------------------------

    def _execute_spacemouse_action(self, new_pos, new_orient, gripper_toggle, button_pressed):
        pose_to_act = copy.deepcopy(self.teleop_target_pose)

        gripper_position, gripper_rotation = convert_action_from_fingertip_to_gripper(pose_to_act)
        if self.control_space == 'cartesian':
            self._execute_cartesian_action(gripper_position, gripper_rotation)
        elif self.control_space == 'joint':
            self._execute_joint_action(gripper_position, gripper_rotation)
        else:
            raise NotImplementedError(f"Unknown control space: {self.control_space}")

        if gripper_toggle:
            new_gripper_value = 1.0 - self.prev_grasp_value
            self._execute_gripper_action(new_gripper_value)
            self.prev_grasp_value = new_gripper_value

        self.prev_button_pressed = button_pressed

    def _execute_teleop_step(self, spacemouse):
        motion = spacemouse.get_motion_state_transformed()
        dx, dy, dz, droll, dpitch, dyaw = motion * self.ACTION_SCALE
        button_pressed = spacemouse.is_button_pressed(0)

        gripper_toggle = 1 if (button_pressed and not self.prev_button_pressed) else 0

        has_movement = (dx != 0 or dy != 0 or dz != 0 or droll != 0 or dpitch != 0 or dyaw != 0)

        if not has_movement and not gripper_toggle:
            self._publish_teleop_state(0)
            self._execute_spacemouse_action(None, None, gripper_toggle, button_pressed)
            self.prev_button_pressed = button_pressed
            return

        dx_int = (1 if dx > 0 else -1 if dx < 0 else 0)
        dy_int = (1 if dy > 0 else -1 if dy < 0 else 0)
        dz_int = (1 if dz > 0 else -1 if dz < 0 else 0)
        droll_int = (1 if droll > 0 else -1 if droll < 0 else 0)
        dpitch_int = (1 if dpitch > 0 else -1 if dpitch < 0 else 0)
        dyaw_int = (1 if dyaw > 0 else -1 if dyaw < 0 else 0)

        msg = Int32MultiArray()
        msg.data = [2, dx_int, dy_int, dz_int, droll_int, dpitch_int, dyaw_int, gripper_toggle, 0]
        self.teleop_pub.publish(msg)

        curr_pos = self.teleop_target_pose.position
        curr_euler = self.teleop_target_pose.orientation.as_euler('XYZ')

        new_pos = np.array([
            curr_pos[0] + dx,
            curr_pos[1] + dy,
            max(curr_pos[2] + dz, 0.01)
        ])

        new_euler = np.array([
            curr_euler[0] + droll,
            curr_euler[1] + dpitch,
            curr_euler[2] + dyaw
        ])

        new_orient = R.from_euler('XYZ', new_euler)
        self.teleop_target_pose.position = new_pos
        self.teleop_target_pose.orientation = new_orient

        self._execute_spacemouse_action(new_pos, new_orient, gripper_toggle, button_pressed)
        self.new_euler = new_euler

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, n_steps: int = None):
        """Execute teleoperation with async intervention prediction monitor."""
        from crisp_py.spacemouse import Spacemouse

        self._setup_keyboard_listener()
        self._setup_teleop_publisher()

        # Start intervention monitor display
        self._monitor.start()
        # Start async intervention prediction thread
        self._start_intervention_thread()

        print("\n" + "="*60)
        print("TELEOPERATION + INTERVENTION MONITOR MODE ACTIVE")
        print("="*60)
        print("Controls:")
        print("  'r' - Reset robot to home position")
        print("  Spacemouse - Control robot movement")
        print("  Spacemouse button - Toggle gripper")
        print("  Ctrl+C - Exit")
        print("="*60 + "\n")

        n_steps_passed = 0

        try:
            with Spacemouse(deadzone=self.DEADZONE) as sm:
                self.spacemouse = sm
                self.teleop_target_pose = get_pose_from_robot(self.robot.end_effector_pose.copy(), ret_pose=True)

                while True:
                    if self._check_key_command('r'):
                        self._handle_reset()

                    self._execute_teleop_step(sm)
                    self._log_loop_frequency()
                    # Trigger async buffer update every step so _intervention_loop
                    # always reads from a fresh observation (non-blocking).
                    self._update_buffer()
                    n_steps_passed += 1
                    # Update EE state in monitor (always, regardless of predict_contact)
                    try:
                        ee_pose = self.robot.end_effector_pose
                        self._monitor.update(
                            ee_pos=np.array(ee_pose.position),
                            ee_rot=ee_pose.orientation,
                            gripper=self.prev_grasp_value,
                        )
                    except Exception:
                        pass

        except KeyboardInterrupt:
            print("\n\nTeleoperation stopped by user")
        finally:
            self._cleanup()
