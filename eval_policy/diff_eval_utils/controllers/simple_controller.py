from diff_eval_utils.controllers.base_controller import RobotController
import time
import threading
import sys
import numpy as np
from pynput import keyboard as kb


class StatusMonitor:
    """Terminal status monitor that refreshes in-place."""

    NUM_LINES = 13

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._initialized = False

        self.intervention_state = None
        self.last_inference_ms = None
        self.last_obs_time_ms = None
        self.last_intervention_ms = None
        self.n_interpolation = None
        self.ee_pos = None
        self.ee_rot = None  # scipy Rotation
        self.gripper = None
        self.gripper_qpos = None
        self.last_predict_action_ts = None
        self.paused = False

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def _render(self):
        def fmt_ms(v):
            return f"{v:.1f} ms" if v is not None else "N/A"

        def fmt_ts(ts):
            if ts is None:
                return "N/A"
            return time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int((ts % 1)*1000):03d}"

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
        gripper_qpos_str = f"{self.gripper_qpos:.4f}" if self.gripper_qpos is not None else "N/A"
        intervention_str = str(self.intervention_state) if self.intervention_state is not None else "N/A"
        interpolation_str = str(self.n_interpolation) if self.n_interpolation is not None else "N/A"
        pause_str = "PAUSED (press i to resume)" if self.paused else "RUNNING"

        lines = [
            "┌─────────────────────────────────────────────────┐",
            f"│  Status                  : {pause_str:<22}│",
            f"│  Intervention State      : {intervention_str:<22}│",
            f"│  Last Inference Time     : {fmt_ms(self.last_inference_ms):<22}│",
            f"│  Get Observation Time    : {fmt_ms(self.last_obs_time_ms):<22}│",
            f"│  Predict Intervention    : {fmt_ms(self.last_intervention_ms):<22}│",
            f"│  N Interpolation         : {interpolation_str:<22}│",
            f"│  EE Position             : {pos_str:<22}│",
            f"│  EE Rotation (xyz)       : {rot_str:<22}│",
            f"│  Gripper (cmd)           : {gripper_str:<22}│",
            f"│  Gripper (qpos)          : {gripper_qpos_str:<22}│",
            f"│  Last Predict Action     : {fmt_ts(self.last_predict_action_ts):<22}│",
            "└─────────────────────────────────────────────────┘",
        ]
        return lines

    def _display_loop(self):
        while self._running:
            with self._lock:
                lines = self._render()

            if self._initialized:
                sys.stdout.write(f"\033[{self.NUM_LINES}A\033[J")
            else:
                self._initialized = True

            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(0.033)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)


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

        monitor = StatusMonitor()
        monitor.start()

        # Keyboard listener for pause/resume
        _key_lock = threading.Lock()
        _key_flags = {'p': False, 'i': False}
        _paused = False

        def _on_press(key):
            with _key_lock:
                try:
                    if hasattr(key, 'char') and key.char in _key_flags:
                        _key_flags[key.char] = True
                except AttributeError:
                    pass

        listener = kb.Listener(on_press=_on_press)
        listener.start()

        print("Starting simple sequential control...  ('p' to pause  'i' to resume)")
        try:
            while n_steps_done < n_steps:
                # Check for pause between inference chunks
                with _key_lock:
                    if _key_flags['p']:
                        _key_flags['p'] = False
                        _paused = True
                        monitor.update(paused=True)
                        print("\n[PAUSED] Press 'i' to resume...")

                while _paused:
                    time.sleep(0.05)
                    with _key_lock:
                        if _key_flags['i']:
                            _key_flags['i'] = False
                            _paused = False
                            monitor.update(paused=False)
                            print("[RESUMED]")

                # Get observation and action
                self._update_buffer_sync()

                t_obs_start = time.time()
                obs_dict = self._get_observation()
                last_obs_time_ms = (time.time() - t_obs_start) * 1000

                t_inference_start = time.time()
                actions = self.policy_client.predict_action(obs_dict)
                last_inference_ms = (time.time() - t_inference_start) * 1000
                last_predict_action_ts = time.time()

                timing_upd = dict(
                    last_inference_ms=last_inference_ms,
                    last_obs_time_ms=last_obs_time_ms,
                    last_predict_action_ts=last_predict_action_ts,
                )
                monitor.update(**timing_upd)
                self._mpl_monitor.update(**timing_upd)

                actions_execute = self._post_process_action(actions[:n_steps])
                self.actions = actions_execute

                for action in actions_execute:
                    if n_steps_done >= n_steps:
                        break

                    self._execute_action(action)
                    monitor.update(
                        intervention_state=self._last_intervention_state,
                        last_intervention_ms=self._last_intervention_ms,
                        n_interpolation=self.n_interpolation,
                        ee_pos=np.array(self.robot.end_effector_pose.position),
                        ee_rot=self.robot.end_effector_pose.orientation,
                        gripper=self.prev_grasp_value,
                        gripper_qpos=1.0 - self.prev_grasp_value,
                    )
                    n_steps_done += 1
                    
                    if self._need_intervention:
                        break

        finally:
            listener.stop()
            monitor.stop()
