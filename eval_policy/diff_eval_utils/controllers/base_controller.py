from abc import ABC, abstractmethod
import time
import numpy as np
from pathlib import Path
import threading
import shutil
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import copy
from scipy.spatial.transform import Slerp


from diff_eval_utils.diffusion_transforms import (
    get_pose_from_robot,
    ten_d_action_to_pose,
    convert_action_from_fingertip_to_gripper,
    franka_obs_to_diff_obs,
    ten_d_action_to_pose_batch,
    convert_action_from_fingertip_to_gripper_batch,
)
from diff_eval_utils.diffusion_visualization import visualize_pcd_and_actions
from diff_eval_utils.diffusion_constants import GRIPPER_NORM_CONST
from std_msgs.msg import Int32MultiArray
from crisp_py.robot import Pose


class TkStatusMonitor:
    """Tkinter-based status monitor.

    Uses Label widgets updated via label.config() — no canvas, no OpenGL,
    no redraws. Works in a daemon thread on Linux with no extra dependencies.
    """

    MAX_ACTIONS = 8

    _BG       = '#1a1a1a'
    _BG_ROW0  = '#2a2a2a'
    _BG_ROW1  = '#242424'
    _FG_LABEL = '#aaaaaa'
    _FG_VALUE = '#00e676'
    _FG_HAND  = '#4CAF50'
    _FG_INTV  = '#F44336'
    _FG_OPEN  = '#2196F3'
    _FG_CLOSE = '#FF9800'
    _FG_HDR   = '#cccccc'
    _FG_OBS   = '#4fc3f7'
    _BG_OBS   = '#0d2233'
    _FONT     = ('Monospace', 10)
    _FONT_HDR = ('Monospace', 10, 'bold')

    _STATUS_ROWS = [
        ('intervention_state',    'Intervention State'),
        ('last_inference_ms',     'Last Inference (ms)'),
        ('last_obs_time_ms',      'Get Obs Time (ms)'),
        ('last_intervention_ms',  'Predict Intv (ms)'),
        ('n_interpolation',       'N Interpolation'),
        ('ee_pos',                'EE Position'),
        ('ee_rot',                'EE Rotation (xyz)'),
        ('gripper_cmd',           'Gripper (cmd)'),
        ('last_predict_action_ts','Last Predict Action'),
    ]
    _ACTION_COLS = ['#', 'X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']

    def __init__(self, controller):
        self._controller = controller
        self._lock = threading.Lock()
        self._cache = {}
        self._running = False
        self._thread = None
        # Label widget references, populated in _build_ui
        self._val_labels = {}          # key -> tk.Label
        self._action_labels = {}       # (row, col) -> tk.Label
        self._obs_row_labels = {}      # col -> tk.Label

    def update(self, **kwargs):
        with self._lock:
            self._cache.update(kwargs)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def _snapshot(self):
        c = self._controller
        fp = c.current_finger_pos
        try:
            ee_pos = np.array(fp[:3]) if fp is not None else None
            ee_rot = R.from_quat(fp[3:]) if fp is not None else None
        except Exception:
            ee_pos = ee_rot = None
        with self._lock:
            cache = dict(self._cache)
        return {
            'intervention_state':     c._last_intervention_state,
            'last_inference_ms':      cache.get('last_inference_ms'),
            'last_obs_time_ms':       cache.get('last_obs_time_ms'),
            'last_intervention_ms':   c._last_intervention_ms,
            'n_interpolation':        c.n_interpolation,
            'ee_pos':                 ee_pos,
            'ee_rot':                 ee_rot,
            'gripper':                c.prev_grasp_value,
            'last_predict_action_ts': cache.get('last_predict_action_ts'),
            'actions':                c.actions,
            'latest_obs_display':     c.latest_obs_display,
        }

    @staticmethod
    def _fmt_ms(v):
        return f"{v:.1f} ms" if v is not None else "N/A"

    @staticmethod
    def _fmt_ts(ts):
        if ts is None:
            return "N/A"
        return time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int((ts % 1)*1000):03d}"

    # ------------------------------------------------------------------
    # UI construction (called once inside the GUI thread)
    # ------------------------------------------------------------------
    def _build_ui(self, root):
        import tkinter as tk

        root.title('Robot Controller Status')
        root.configure(bg=self._BG)
        root.protocol('WM_DELETE_WINDOW', lambda: None)

        pad = {'padx': 6, 'pady': 2}

        # ── Status section ────────────────────────────────────────────
        tk.Label(root, text='Robot Status', bg=self._BG,
                 fg=self._FG_HDR, font=self._FONT_HDR).grid(
            row=0, column=0, columnspan=2, sticky='w', padx=6, pady=(6, 2))
        tk.Frame(root, bg='#444444', height=1).grid(
            row=1, column=0, columnspan=2, sticky='ew', padx=6)

        for i, (key, label) in enumerate(self._STATUS_ROWS):
            bg = self._BG_ROW0 if i % 2 == 0 else self._BG_ROW1
            row = i + 2
            tk.Label(root, text=label, bg=bg, fg=self._FG_LABEL,
                     font=self._FONT, anchor='w', width=26).grid(
                row=row, column=0, sticky='ew', **pad)
            lbl = tk.Label(root, text='N/A', bg=bg, fg=self._FG_VALUE,
                           font=self._FONT, anchor='w', width=36)
            lbl.grid(row=row, column=1, sticky='ew', **pad)
            self._val_labels[key] = lbl

        sep_row = len(self._STATUS_ROWS) + 2
        tk.Label(root, text='Predicted Actions', bg=self._BG,
                 fg=self._FG_HDR, font=self._FONT_HDR).grid(
            row=sep_row, column=0, columnspan=2, sticky='w', padx=6, pady=(8, 2))
        tk.Frame(root, bg='#444444', height=1).grid(
            row=sep_row + 1, column=0, columnspan=2, sticky='ew', padx=6)

        # ── Actions table ─────────────────────────────────────────────
        col_widths = [3, 8, 8, 8, 7, 7, 7, 16]
        tbl_start = sep_row + 2

        # Header
        for c, (col_name, w) in enumerate(zip(self._ACTION_COLS, col_widths)):
            tk.Label(root, text=col_name, bg='#333333', fg='#ffffff',
                     font=self._FONT_HDR, anchor='center', width=w).grid(
                row=tbl_start, column=c, padx=1, pady=1, sticky='ew')

        # Obs sent row (row 0, visually distinct)
        for c, w in enumerate(col_widths):
            lbl = tk.Label(root, text='', bg=self._BG_OBS, fg=self._FG_OBS,
                           font=self._FONT, anchor='center', width=w)
            lbl.grid(row=tbl_start + 1, column=c, padx=1, pady=1, sticky='ew')
            self._obs_row_labels[c] = lbl
        self._obs_row_labels[0].config(text='OBS')

        # Data rows (shifted down by 1 to make room for obs row)
        for r in range(self.MAX_ACTIONS):
            bg = self._BG_ROW0 if r % 2 == 0 else self._BG_ROW1
            for c in range(len(self._ACTION_COLS)):
                lbl = tk.Label(root, text='', bg=bg, fg=self._FG_VALUE,
                               font=self._FONT, anchor='center',
                               width=col_widths[c])
                lbl.grid(row=tbl_start + 2 + r, column=c, padx=1, pady=1, sticky='ew')
                self._action_labels[(r, c)] = lbl

        # Make columns stretch evenly
        for c in range(len(self._ACTION_COLS)):
            root.grid_columnconfigure(c, weight=1)

    # ------------------------------------------------------------------
    # Per-tick value update
    # ------------------------------------------------------------------
    def _update_display(self):
        snap = self._snapshot()
        # try:
        #     snap = self._snapshot()
        # except Exception:
        #     return

        # Intervention state
        intv = snap['intervention_state']
        self._val_labels['intervention_state'].config(
            text='INTERVENE' if intv else 'HAND',
            fg=self._FG_INTV if intv else self._FG_HAND)

        # Timing
        self._val_labels['last_inference_ms'].config(
            text=self._fmt_ms(snap['last_inference_ms']))
        self._val_labels['last_obs_time_ms'].config(
            text=self._fmt_ms(snap['last_obs_time_ms']))
        self._val_labels['last_intervention_ms'].config(
            text=self._fmt_ms(snap['last_intervention_ms']))
        self._val_labels['n_interpolation'].config(
            text=str(snap['n_interpolation']))

        # EE position
        p = snap['ee_pos']
        self._val_labels['ee_pos'].config(
            text=f"[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]" if p is not None else "N/A")

        # EE rotation
        rot = snap['ee_rot']
        if rot is not None:
            try:
                e = rot.as_euler('xyz', degrees=True)
                self._val_labels['ee_rot'].config(
                    text=f"[{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}] deg")
            except Exception:
                self._val_labels['ee_rot'].config(text='N/A')
        else:
            self._val_labels['ee_rot'].config(text='N/A')

        # Gripper
        g = snap['gripper']
        if g is not None:
            closed = bool(round(g))
            self._val_labels['gripper_cmd'].config(
                text='CLOSED' if closed else 'OPEN',
                fg=self._FG_CLOSE if closed else self._FG_OPEN)

        # Last predict action timestamp
        self._val_labels['last_predict_action_ts'].config(
            text=self._fmt_ts(snap['last_predict_action_ts']))

        # Obs sent row
        obs = snap.get('latest_obs_display')
        if obs is not None:
            try:
                euler = R.from_quat(obs['quat']).as_euler('xyz', degrees=True)
            except Exception:
                euler = [float('nan')] * 3
            g = float(obs['gripper'])
            is_contact = bool(obs.get('is_contact', False))
            gt = obs.get('gripper_torque')
            gt_str = f"|ft|={float(np.linalg.norm(gt[:3])):.2f}" if gt is not None else "N/A"
            obs_vals = [
                'OBS',
                f"{obs['pos'][0]:.3f}", f"{obs['pos'][1]:.3f}", f"{obs['pos'][2]:.3f}",
                f"{euler[0]:.1f}", f"{euler[1]:.1f}", f"{euler[2]:.1f}",
                f"{g:.4f} {'C' if is_contact else 'NC'} {gt_str}",
            ]
            for c, v in enumerate(obs_vals):
                cfg = {'text': v}
                if c == 7:
                    cfg['fg'] = '#F44336' if is_contact else self._FG_OBS
                self._obs_row_labels[c].config(**cfg)

        # Actions table
        actions = snap.get('actions') or []
        for r in range(self.MAX_ACTIONS):
            if r < len(actions):
                pos, rot, grasp = actions[r]
                try:
                    euler = rot.as_euler('xyz', degrees=True)
                except Exception:
                    euler = [float('nan')] * 3
                closed = bool(round(grasp))
                vals = [
                    str(r + 1),
                    f"{pos[0]:.3f}", f"{pos[1]:.3f}", f"{pos[2]:.3f}",
                    f"{euler[0]:.1f}", f"{euler[1]:.1f}", f"{euler[2]:.1f}",
                    f"{'CLOSED' if closed else 'OPEN'} ({float(grasp):.3f})",
                ]
                for c, v in enumerate(vals):
                    cfg = {'text': v}
                    if c == 7:  # gripper column
                        cfg['fg'] = self._FG_CLOSE if closed else self._FG_OPEN
                    self._action_labels[(r, c)].config(**cfg)
            else:
                for c in range(len(self._ACTION_COLS)):
                    self._action_labels[(r, c)].config(text='')

    # ------------------------------------------------------------------
    # GUI thread
    # ------------------------------------------------------------------
    def _run_gui(self):
        try:
            import tkinter as tk

            root = tk.Tk()
            self._build_ui(root)

            def _tick():
                if not self._running:
                    root.quit()
                    return
                try:
                    self._update_display()
                except Exception as e:
                    print(f"TkStatusMonitor render error: {e}")
                root.after(50, _tick)

            root.after(0, _tick)
            root.mainloop()

        except Exception as e:
            print(f"TkStatusMonitor: could not open window: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_gui, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)


class RobotController(ABC):
    """Abstract base class for robot controllers."""

    def __init__(self, robot, gripper, obs_manager, joint_state_subscriber,
                 policy_client, config, ctrl_freq=10.0, ik_client=None):
        """Initialize the controller.

        Args:
            robot: Robot instance
            gripper: Gripper instance
            obs_manager: PointCloudManager instance
            joint_state_subscriber: JointStateSubscriber instance
            policy_client: PolicyClient instance
            config: DPEvalConfig instance
            ctrl_freq: Control frequency in Hz
            ik_client: PinkIKClient instance (required when ctrl_space='joint')
        """
        self.robot = robot
        self.gripper = gripper
        self.obs_manager = obs_manager
        self.joint_state_subscriber = joint_state_subscriber
        self.policy_client = policy_client
        self.ik_client = ik_client
        self.config = config
        self.control_space = config.ctrl_space  # 'joint' or 'cartesian'
        if self.control_space not in ['joint', 'cartesian']:
            raise ValueError(f"Invalid ctrl_space: {self.control_space}")
        if self.control_space == 'joint':
            self.ctrl_freq = config.joint_ctrl_freq
            print(self.ctrl_freq)
        else:
            self.ctrl_freq = config.ctrl_freq
        # print(f"Current Control Frequency: {self.ctrl_freq}")
        self.n_interpolation = config.n_interpolation

        if self.control_space == 'joint' and self.ik_client is None:
            raise ValueError("ik_client is required when ctrl_space='joint'")

        # Common state
        self.target_pose = robot.end_effector_pose.copy()
        self.arm_rate = robot.node.create_rate(self.ctrl_freq)
        self.gripper_rate = gripper.node.create_rate(self.ctrl_freq)
        self.prev_grasp_value = 0.0

        # Unified observation buffer (stores all observation data per timestep)
        self.obs_buffer = []
        self._buffer_lock = threading.Lock()

        # Background buffer update thread
        self._buffer_update_thread = None
        self._buffer_update_running = False
        self._buffer_update_requested = threading.Event()
        self._buffer_update_complete = threading.Event()

        self.latest_obs_timestamp = None

        self.avg_init_gripper_torque = None

        self._last_joint_target = None
        self._joint_exec_time = None
        self._max_delta_joint_thresh = 8.0
        self._max_delta_joint_angle = 8.0
        self._joint_exec_time_tolerance = config.joint_exec_time_tolerance

        self._last_intervention_state = False
        self._need_intervention = False # this is intervention trigger
        self.intervention_status = False # this is intervention status
        self.actions = []  # latest predicted actions for display
        self.latest_obs_display = None  # latest obs sent to policy for display
        self.current_finger_pos = None  # real-time finger pose [x,y,z,qx,qy,qz,qw]

        # Intervention prediction thread (runs independently at max speed)
        self._intv_thread = None
        self._intv_thread_running = False
        self._last_intervention_ms = None

        if self.policy_client is None:
            self.predict_contact = False
        else:
            self.predict_contact = self.policy_client.predict_contact
            assert type(self.policy_client.predict_contact) == bool


        # Initial synchronous buffer update
        print("Performing initial observation buffer sync...")
        self._update_buffer_sync()

        print(f"self.predict_contact: {self.predict_contact}")
        # Start intervention thread after initial sync
        if self.predict_contact:
            self._start_intervention_thread()

        # Start matplotlib status GUI
        self._mpl_monitor = TkStatusMonitor(controller=self)
        self._mpl_monitor.start()

    def _start_background_buffer_thread(self):
        """Start the background buffer update thread."""
        if self._buffer_update_thread is not None and self._buffer_update_thread.is_alive():
            return

        self._buffer_update_running = True
        self._buffer_update_thread = threading.Thread(
            target=self._background_buffer_worker,
            daemon=True
        )
        self._buffer_update_thread.start()

    def _stop_background_buffer_thread(self):
        """Stop the background buffer update thread."""
        self._buffer_update_running = False
        self._buffer_update_requested.set()  # Wake up the thread to exit
        if self._buffer_update_thread is not None:
            self._buffer_update_thread.join(timeout=1.0)

    def _background_buffer_worker(self):
        """Background worker that updates buffer when requested."""
        while self._buffer_update_running:
            # Wait for update request
            self._buffer_update_requested.wait()

            if not self._buffer_update_running:
                break

            self._buffer_update_requested.clear()

            # Perform the buffer update
            self._update_buffer_sync()

            # Signal that update is complete
            self._buffer_update_complete.set()

    def _update_buffer_sync(self):
        """Synchronously update the observation buffer (called by background thread)."""
        t_start = time.time()
        self.obs_manager.clear_cache()
        self.joint_state_subscriber.clear_cache()
        while not self.joint_state_subscriber.is_ready:
            time.sleep(0.01)
        print(f"time took for the obs to be ready: {(time.time() - t_start)*1000} ms")

        joint_state = self.joint_state_subscriber.joint_values
        gripper_state = 1.0 - self.prev_grasp_value
        if not self.config.ft_sensor_on:
            gripper_torque = None
        elif self.avg_init_gripper_torque is not None:
            gripper_torque = self.joint_state_subscriber.gripper_torque - self.avg_init_gripper_torque
        else:
            gripper_torque = self.joint_state_subscriber.gripper_torque
        # print(f"gripper_state {gripper_state}")
        # print(f"gripper_torque {gripper_torque}")

        # Capture images and sensor data
        cam3_rgb, _ = self.obs_manager.get_latest_rgbd('cam3')
        print(1)
        cam4_rgb, cam4_depth = self.obs_manager.get_latest_rgbd('cam4')
        print(1)
        pcd = self.obs_manager.get_latest_pointcloud()
        print(1)

        robot_pose = copy.deepcopy(self.robot.end_effector_pose)
        # orientation_offset = R.from_euler('xyz', [np.pi, 0, 0]) # Rotate 180 degrees around X-axis
        # robot_pose.orientation = robot_pose.orientation * orientation_offset
        # orientation_offset = R.from_euler('xyz', [0, 0, -np.pi / 2]) # Rotate 90 degrees around Z-axis
        # robot_pose.orientation = robot_pose.orientation * orientation_offset
        
        eef_pose = get_pose_from_robot(robot_pose)

        # Store latest obs for GUI display

        print(f"obs ready")
        self.latest_obs_display = {
            'pos': eef_pose[:3].copy(),
            'quat': eef_pose[3:].copy(),
            'gripper': gripper_state,
            'is_contact': self.intervention_status,
            'gripper_torque': gripper_torque,
        }

        # Build raw observation snapshot
        obs_snapshot = {
            'eef_pos': eef_pose[:3].astype(np.float32),
            'eef_quat': eef_pose[3:].astype(np.float32),
            'gripper_qpos': np.array([gripper_state, gripper_state], dtype=np.float32),
            'joint_pos': joint_state.astype(np.float32),
            'cam4_rgb': cam4_rgb.copy(),
            'cam4_depth': cam4_depth.copy(),
            'cam3_rgb': cam3_rgb.copy(),
            'pcd': pcd,
        }

        # Convert to diffusion policy observation format
        obs_dict = franka_obs_to_diff_obs(
            [obs_snapshot],  # Single frame
            img_policy=self.config.img_policy,
            visualize=self.config.visualize
        )
        obs_dict['is_contact'] = np.array([self.intervention_status], dtype=np.float32)
        update_end_time = time.time()
        # obs_dict['is_contact'] = np.array([1], dtype=np.float32)
        obs_dict['timestamp'] = update_end_time

        # Thread-safe buffer update
        with self._buffer_lock:
            self.obs_buffer.append(obs_dict)

            # Initialize buffer with duplicate if needed (for policies requiring 2 frames)
            while len(self.obs_buffer) < 2:
                self.obs_buffer.insert(0, obs_dict.copy())
        # print(f"update_end_time {update_end_time}")
        return update_end_time

    def _predict_intervention(self, obs_dict):
        """Get intervention prediction from the policy server."""
        predicted_label, timing = self.policy_client.predict_intervention(obs_dict)
        # print(f"Intervention prediction: {predicted_label}")
        return predicted_label
        # except Exception as e:
        #     print(f"Warning: Failed to get intervention prediction: {e}")
        #     return 0  # Default to no intervention on error

    def _start_intervention_thread(self):
        """Start the continuous intervention prediction thread."""
        if self._intv_thread is not None and self._intv_thread.is_alive():
            return
        self._intv_thread_running = True
        self._intv_thread = threading.Thread(
            target=self._intervention_worker,
            daemon=True
        )
        self._intv_thread.start()

    def _stop_intervention_thread(self):
        """Stop the intervention prediction thread."""
        self._intv_thread_running = False
        if self._intv_thread is not None:
            self._intv_thread.join(timeout=2.0)

    def _intervention_worker(self):
        """Continuously predict intervention using only eef_pos, eef_quat, cam4_rgb, cam4_depth."""
        while self._intv_thread_running:
            try:
                t_intv = time.time()
                cam4_rgb, cam4_depth = self.obs_manager.get_latest_rgbd('cam4')

                robot_pose = copy.deepcopy(self.robot.end_effector_pose)
                
                # orientation_offset = R.from_euler('xyz', [np.pi, 0, 0]) # Rotate 180 degrees around X-axis
                # robot_pose.orientation = robot_pose.orientation * orientation_offset
                # orientation_offset = R.from_euler('xyz', [0, 0, -np.pi / 2]) # Rotate 90 degrees around Z-axis
                # robot_pose.orientation = robot_pose.orientation * orientation_offset
                
                eef_pose = get_pose_from_robot(robot_pose)

                gripper_state = 1.0 - self.prev_grasp_value
                obs_dict = {
                    'robot0_eef_pos': eef_pose[:3].astype(np.float32),
                    'robot0_eef_quat': eef_pose[3:].astype(np.float32),
                    'robot0_gripper_qpos': np.array([gripper_state, gripper_state], dtype=np.float32),
                    'robot0_eye_in_hand_image': cam4_rgb.copy().astype(np.uint8),
                    'robot0_eye_in_hand_depth': cam4_depth.copy().astype(np.float32),
                }

                
                predicted_label = self._predict_intervention(obs_dict)
                # predicted_label = False
                self._last_intervention_ms = (time.time() - t_intv) * 1000
                print(f"\n\n\n\npredicted_label = {predicted_label}\n\n\n\n")
                self.intervention_status = predicted_label
                if self._last_intervention_state == False and predicted_label == True:
                    self._need_intervention = True
                    self._last_intervention_state = predicted_label
                    time.sleep(0.5)
                else:
                    self._need_intervention = False
                    self._last_intervention_state = predicted_label
            except Exception as e:
                print(f"Warning: Intervention worker error: {e}")

    def _update_buffer(self):
        """Request a buffer update (non-blocking, runs in background thread)."""
        # Make sure background thread is running
        if not self._buffer_update_running:
            self._start_background_buffer_thread()

        # Clear completion flag and request update
        self._buffer_update_complete.clear()
        self._buffer_update_requested.set()

    def _clear_buffer(self):
        """Clear the observation buffer and repopulate with fresh observations."""
        with self._buffer_lock:
            self.obs_buffer.clear()
        # Synchronously update to repopulate the buffer
        self._update_buffer_sync()

    def _wait_for_buffer_update(self, timeout=1.0):
        """Wait for the background buffer update to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if update completed, False if timeout
        """
        return self._buffer_update_complete.wait(timeout=timeout)

    @abstractmethod
    def run(self, n_steps: int):
        """Execute the controller for n_steps.

        Args:
            n_steps: Number of control steps to execute
        """
        pass

    def _get_observation(self):
        # Keep buffer size limited
        with self._buffer_lock:
            if len(self.obs_buffer) > 20:
                self.obs_buffer.pop(0)

            if not self.obs_buffer:
                raise RuntimeError("Observation buffer is empty")

            # Get the latest observation
            obs_dict = self.obs_buffer[-1].copy()
        self.latest_obs_timestamp = obs_dict.get('timestamp')
        # Remove timestamp from obs_dict before returning (not needed by policy)
        obs_dict.pop('timestamp', None)

        # Always override eef pos/quat with current live robot pose
        eef_pose = get_pose_from_robot(copy.deepcopy(self.robot.end_effector_pose))
        obs_dict['robot0_eef_pos'] = eef_pose[:3].astype(np.float32)
        obs_dict['robot0_eef_quat'] = eef_pose[3:].astype(np.float32)

        # Always override gripper_qpos based on is_contact label
        if bool(obs_dict.get('is_contact', np.array([False]))[0]):
            gripper_override = self.prev_grasp_value
        else:
            gripper_override = 1.0 - self.prev_grasp_value
        if getattr(self.config, 'flip_gripper_obs', True):
            gripper_override = self.prev_grasp_value
        obs_dict['robot0_gripper_qpos'] = np.array(
            [gripper_override, gripper_override], dtype=np.float32)

        return obs_dict

    def _get_inhand_rgb(self):
        return self.obs_manager.get_latest_rgbd('cam4')[0]
    
    def _post_process_action(self, actions):
        # time.sleep(10)

        poses, grasps = ten_d_action_to_pose_batch(actions)
        gripper_positions, gripper_rotations = convert_action_from_fingertip_to_gripper_batch(poses)
        actions_ret = []

        for index, grasp in enumerate(grasps[1:7]):
            prev_grasp = grasps[index-1]
            if grasp != prev_grasp and grasps[index+1] == prev_grasp:
                print(f"!!! Detected potential grasp noise at index {index}, smoothing grasp value")
                grasps[index] = prev_grasp
        
        grasps[-1] = grasps[-2]

        for idx in range(len(grasps)):
            actions_ret.append((gripper_positions[idx], gripper_rotations[idx], grasps[idx]))

        return actions_ret
    
    def _execute_gripper_action(self, grasp_value):
        # print(f"grasp value: {grasp_value}")
        grasp_value = np.round(np.clip(grasp_value, 0, 1))
        if grasp_value != self.prev_grasp_value:
            # print(f"setting target to {1 - grasp_value}")
            time.sleep(0.2)
            if self._need_intervention:
                return
            self.gripper.set_target(1 - grasp_value)
            self.gripper_rate.sleep()
            time.sleep(0.5)
            # time.sleep(1.0)  # Wait for gripper (Franka driver limitation)
        self.prev_grasp_value = grasp_value
    
    def _execute_cartesian_action(self, action, gripper_pose, move_to=False):
        assert len(action) >= 3, "Action must have at least 3 elements for position"
        new_position = action[:3]
        current_position = copy.deepcopy(self.robot.end_effector_pose.position)
        # print(f"{new_position[0]:.3f}, {new_position[1]:.3f}, {new_position[2]:.3f}\t{current_position[0]:.3f}, {current_position[1]:.3f}, {current_position[2]:.3f}")
        current_orientation = copy.deepcopy(self.robot.end_effector_pose.orientation)

        if move_to:
            self.target_pose.position = new_position
            self.target_pose.orientation = gripper_pose
            self.robot.move_to(pose=self.target_pose, speed=0.15)
            return
        
        # Create SLERP interpolator for orientation

        if self.n_interpolation == 0:
            self.target_pose.position = new_position
            self.target_pose.orientation = gripper_pose
            self.robot.set_target(pose=self.target_pose)
            self.arm_rate.sleep()
            self._per_action()
            return

        key_rots = R.concatenate([current_orientation, gripper_pose])
        slerp = Slerp([0, 1], key_rots)
        
        for step in range(self.n_interpolation):
            alpha = (step + 1) / self.n_interpolation
            # Interpolate position (linear)
            interp_position = (1 - alpha) * current_position + alpha * new_position
            # Interpolate orientation (SLERP)
            interp_orientation = slerp(alpha)

            self.target_pose.position = interp_position
            self.target_pose.orientation = interp_orientation
            self.robot.set_target(pose=self.target_pose)
            self.arm_rate.sleep()
            self._per_action()

        # Set final pose
        self.target_pose.position = new_position
        self.target_pose.orientation = gripper_pose

    def _per_action(self):
        self.current_finger_pos = get_pose_from_robot(copy.deepcopy(self.robot.end_effector_pose))
    
    def _compute_q_target(self, action, gripper_pose):
        assert len(action) >= 3, "Action must have at least 3 elements for position"
        if self._last_joint_target is None:
            q_current = self.robot.joint_values.copy()
        else: 
            q_current = self._last_joint_target
        target_pos = action[:3]
        target_quat = gripper_pose.as_quat()
        
        if self._joint_exec_time is not None:
            t_since_last_exec = (time.time() - self._joint_exec_time) * 1000 # convert to ms
            if t_since_last_exec > self._joint_exec_time_tolerance:
                print(f"WARNING: Control loop too slow, took {t_since_last_exec} for new action to arrive")
    
        q_target, success, timing = self.ik_client.solve_ik(target_pos, target_quat, q_current)

        if not success:
            print(f"  WARNING: IK did not converge. Using current joint values as target to ensure safety.")
            return q_current, q_current

        # print(self.robot.end_effector_pose.position, target_pos)
        # print(f"IK time={timing.get('total_time_ms', 0):.4f}ms, iters={timing.get('num_iterations', 0)}, success: {success}")

        return q_target, q_current
    
    def _joint_interpolation(self, q_target, q_current):
        max_delta = np.degrees(np.max(np.abs(q_target - q_current)))
        # print(f"max_delta: {max_delta}, self._max_delta_joint_angle: {self._max_delta_joint_angle}")
        self._max_delta_joint_angle = max(self._max_delta_joint_angle, max_delta)
                
        if max_delta > self._max_delta_joint_thresh:
            print(f"!!! Max delta joint angle ever: {self._max_delta_joint_angle:.2f} deg")
            print(f"!!! Max delta joint angle ever: {self._max_delta_joint_angle:.2f} deg")
            print(f"!!! Max delta joint angle ever: {self._max_delta_joint_angle:.2f} deg")
            qs = []
            n_interp_rounds_needed = int((max_delta // self._max_delta_joint_thresh + 1) * self.n_interpolation)
            print(f"!!! Doing {n_interp_rounds_needed} extra rounds of interpolation to ensure robot safety")
            print(f"!!! Doing {n_interp_rounds_needed} extra rounds of interpolation to ensure robot safety")
            print(f"!!! Doing {n_interp_rounds_needed} extra rounds of interpolation to ensure robot safety")
            for step in range(n_interp_rounds_needed):
                alpha = (step + 1) / n_interp_rounds_needed
                q_interp = (1 - alpha) * q_current + alpha * q_target
                qs.append(q_interp)
            return qs
        
        if self.n_interpolation == 0:
            qs = [q_target]
        else:
            qs = []
            for step in range(self.n_interpolation):
                alpha = (step + 1) / self.n_interpolation
                q_interp = (1 - alpha) * q_current + alpha * q_target
                qs.append(q_interp)

        return qs
    
    def _joint_execution(self, qs):
        for q_interp in qs:
            self.robot.set_target_joint(q_interp)
            self.arm_rate.sleep()
            self._per_action()

    def _execute_joint_action(self, action, gripper_pose):
        q_target, q_current = self._compute_q_target(action, gripper_pose)
        q_interps = self._joint_interpolation(q_target, q_current)
        self._joint_execution(q_interps)
        print("Executed joint action to target: ", q_target)
        print("Executed joint action to target: ", q_target)

        print("Executed joint action to target: ", q_target)

        print("Executed joint action to target: ", q_target)

        print("Executed joint action to target: ", q_target)

        self._last_joint_target = q_target
        self._joint_exec_time = time.time()

    def _execute_action(self, action, move_to=False):
        """Execute a single action.

        Args:
            
        """
        gripper_position, gripper_rotation, gripper_action = action

        if self.control_space == 'cartesian':
            self._execute_cartesian_action(gripper_position, gripper_rotation, move_to=move_to)

        elif self.control_space == 'joint':
            assert move_to == False, "move_to not supported in joint control space"
            self._execute_joint_action(gripper_position, gripper_rotation)
        
        else:
            raise NotImplementedError(f"Invalid control space: {self.control_space}")
        
        self._execute_gripper_action(gripper_action)

        # Trigger async buffer update (non-blocking)
        # self._update_buffer_sync()
        self._update_buffer()

    def _switch_to_impedance_controller(self):
        if self.control_space == 'cartesian':
            self.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
            self.robot.cartesian_controller_parameters_client.load_param_config(
                file_path="config/control/spacemouse_cartesian_impedance.yaml"
            )
        elif self.control_space == 'joint':
            self.robot.controller_switcher_client.switch_controller("joint_impedance_controller")
            self.robot.joint_controller_parameters_client.load_param_config(
                file_path="config/control/joint_impedance_controller.yaml"
            )
        else:
            raise NotImplementedError(f"Invalid control space: {self.control_space}")
        
    def _pre_run(self):
        if not self.config.ft_sensor_on:
            print("ft_sensor_on=False, skipping gripper torque calibration.")
            return

        print("collecting initial gripper torque data for relative force-torque observation...")
        time.sleep(0.5)
        # TODO: Collect forch torque data for 1 sec to compute relative force torque observation.
        self.joint_state_subscriber.clear_cache()
        init_gripper_torques = []

        while not self.joint_state_subscriber.is_ready:
            time.sleep(0.01)
        # print(f"time took for the obs to be ready: {(time.time() - t_start)*1000} ms")
        while len(init_gripper_torques) < 20:
            if self.joint_state_subscriber.is_ready:
                init_gripper_torques.append(self.joint_state_subscriber.gripper_torque)
                self.joint_state_subscriber.clear_cache()

        self.avg_init_gripper_torque = np.mean(init_gripper_torques, axis=0)
        print(f"Average initial gripper torque: {self.avg_init_gripper_torque}")

    def cleanup(self):
        """Clean up resources. Call this when done with the controller."""
        self.gripper.set_target(1.0)  # Open gripper
        self._stop_background_buffer_thread()
        self._stop_intervention_thread()
        self._mpl_monitor.stop()
