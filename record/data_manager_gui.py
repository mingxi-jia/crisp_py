#!/usr/bin/env python3
"""
Episode Data Manager GUI with frame viewer.

Left panel  – episode list with stats (frames, duration, FPS, status).
Right panel – frame viewer: 2x2 camera grid, grasp bar, contact bar, scrub slider.
"""

import os
import shutil
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Optional
from datetime import datetime
import numpy as np
import argparse

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from PIL import Image as PILImage
    from PIL import ImageTk as PILImageTk
    _RESAMPLE = PILImage.Resampling.LANCZOS  # type: ignore[attr-defined]
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

ACTION_TRIM = 0.0001  # metres — values below this are zeroed
AUTO_CLEAN_MIN_FRAMES = 10
DEFAULT_DATA_DIR = "./raw_datasets/episodes"
REFRESH_INTERVAL_MS = 5000
GRASP_BAR_H = 24        # height of the grasp / contact timeline bars in pixels

# ── Episode info ───────────────────────────────────────────────────────────────


def get_episode_info(episode_path: str) -> dict:
    """Extract statistics from an episode directory."""
    info: dict[str, Any] = {
        "path": episode_path,
        "name": os.path.basename(episode_path),
        "frames": 0,
        "duration": 0.0,
        "fps": 0.0,
        "bad": False,
        "date": "",
        "sort_key": "",
    }

    try:
        suffix = info["name"].split("_", 1)[1]
        dt = datetime.strptime(suffix, "%Y%m%d_%H%M%S_%f")
        info["date"] = dt.strftime("%Y-%m-%d %H:%M:%S")
        info["sort_key"] = suffix
    except (ValueError, IndexError):
        info["date"] = info["name"]
        info["sort_key"] = info["name"]

    state_file = os.path.join(episode_path, "state", "grasp.npy")
    if os.path.exists(state_file):
        try:
            info["frames"] = len(np.load(state_file))
        except Exception:
            pass

    rgb_path = os.path.join(episode_path, "cam1", "rgb")
    if info["frames"] == 0 and os.path.exists(rgb_path):
        try:
            info["frames"] = len([f for f in os.listdir(rgb_path) if f.endswith(".png")])
        except OSError:
            pass

    if os.path.exists(rgb_path):
        try:
            pngs = sorted(f for f in os.listdir(rgb_path) if f.endswith(".png"))
            if len(pngs) >= 2:
                def parse_ts(fname: str) -> float:
                    base = fname.rsplit(".", 1)[0]
                    sec, nanosec = base.split("_")
                    return int(sec) + int(nanosec) * 1e-9
                t0, t1 = parse_ts(pngs[0]), parse_ts(pngs[-1])
                info["duration"] = t1 - t0
                if info["duration"] > 0:
                    info["fps"] = (len(pngs) - 1) / info["duration"]
        except Exception:
            pass

    info["bad"] = os.path.exists(os.path.join(episode_path, "state", "bad.txt"))
    return info


# ── GUI ────────────────────────────────────────────────────────────────────────


class DataManagerGUI:
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR,
                 refresh_interval: int = REFRESH_INTERVAL_MS,
                 hdf5_path: Optional[str] = None):
        self.data_dir = os.path.abspath(data_dir)
        self.refresh_interval = refresh_interval
        self._loading = False
        self._episodes: list = []

        # HDF5 mode
        self._hdf5_path = hdf5_path
        self._hdf5_file: Any = None          # open h5py.File handle
        self._hdf5_cam_keys: dict[int, Any] = {}  # {cam_idx: h5py.Dataset}

        # Viewer state
        self._viewer_episode_path: str = ""
        self._frame_files_by_cam: dict[int, list[str]] = {}
        self._n_frames: int = 0
        self._actions_xyz: Optional[np.ndarray] = None  # shape (N, 3), trimmed
        self._grasp_data: Optional[np.ndarray] = None   # shape (N,)
        self._contact_data: Optional[np.ndarray] = None  # shape (N,) is_contact label
        self._pose_data: Optional[np.ndarray] = None    # shape (N, 6): x y z roll pitch yaw
        self._gripper_qpos_data: Optional[np.ndarray] = None  # shape (N,) raw qpos 0–1
        self._current_frame_idx: int = 0
        self._photo_refs: list = []  # strong refs to prevent GC (up to 4)

        if self._hdf5_path:
            if not H5PY_AVAILABLE:
                raise RuntimeError("h5py is required for HDF5 mode: pip install h5py")
            self._hdf5_file = h5py.File(self._hdf5_path, 'r+')

        self.root = tk.Tk()
        self.root.title("Episode Data Manager")
        self.root.geometry("1500x750")
        self.root.configure(bg="#1e1e1e")

        self._build_ui()
        self._schedule_refresh()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _btn(self, parent: tk.Widget, text: str, command: Any,
             bg: str, fg: str = "white", width: Optional[int] = None) -> tk.Button:
        b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                      relief=tk.FLAT, cursor="hand2", padx=10, pady=4)
        b.configure(font=("TkDefaultFont", 10))
        if width:
            b.configure(width=width)
        return b

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self.root, bg="#252526", pady=6)
        header.pack(fill=tk.X)
        lbl = tk.Label(header, text=f"  {self.data_dir}", bg="#252526", fg="#9cdcfe", anchor="w")
        lbl.configure(font=("Courier", 10))
        lbl.pack(side=tk.LEFT)
        self.stats_var = tk.StringVar(value="")
        slbl = tk.Label(header, textvariable=self.stats_var, bg="#252526", fg="#b5cea8")
        slbl.configure(font=("Courier", 10))
        slbl.pack(side=tk.RIGHT, padx=12)

        # Button bar
        btn_frame = tk.Frame(self.root, bg="#1e1e1e", pady=6)
        btn_frame.pack(fill=tk.X, padx=10)
        self._btn(btn_frame, "⟳  Refresh", self._refresh, "#3c3c3c").pack(side=tk.LEFT, padx=4)
        del_btn = self._btn(btn_frame, "🗑  Delete Selected", self._delete_selected, "#c0392b")
        del_btn.pack(side=tk.LEFT, padx=4)
        clean_btn = self._btn(btn_frame, f"🧹  Auto Clean  (< {AUTO_CLEAN_MIN_FRAMES} frames)",
                              self._auto_clean, "#e67e22")
        clean_btn.pack(side=tk.LEFT, padx=4)
        if self._hdf5_path:
            clean_btn.configure(state=tk.DISABLED, bg="#555555")
        self.status_var = tk.StringVar(value="")
        slbl2 = tk.Label(btn_frame, textvariable=self.status_var, bg="#1e1e1e", fg="#888888")
        slbl2.configure(font=("TkDefaultFont", 9))
        slbl2.pack(side=tk.RIGHT, padx=8)

        # Main paned area
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#1e1e1e",
                               sashwidth=5, sashrelief=tk.FLAT)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        left = tk.Frame(paned, bg="#1e1e1e")
        right = tk.Frame(paned, bg="#1e1e1e")
        paned.add(left, minsize=500)
        paned.add(right, minsize=380)
        self.root.update_idletasks()
        paned.sash_place(0, max(600, self.root.winfo_width() * 6 // 10), 0)

        self._build_list_panel(left)
        self._build_viewer_panel(right)

    # ── Episode list panel ─────────────────────────────────────────────────────

    def _build_list_panel(self, parent: tk.Frame) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#252526", foreground="#d4d4d4",
                        rowheight=26, fieldbackground="#252526")
        style.configure("Treeview", font=("Courier", 10))
        style.configure("Treeview.Heading", background="#3c3c3c", foreground="#ffffff")
        style.configure("Treeview.Heading", font=("TkDefaultFont", 10, "bold"))
        style.map("Treeview",
                  background=[("selected", "#094771")],
                  foreground=[("selected", "white")])

        columns = ("name", "date", "frames", "duration", "fps", "status")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", selectmode="extended")

        col_cfg: list[tuple[str, str, int, str]] = [
            ("name",     "Episode Name",   230, "w"),
            ("date",     "Date / Time",    160, "center"),
            ("frames",   "Frames",          70, "center"),
            ("duration", "Duration  (s)",  110, "center"),
            ("fps",      "Eff. FPS",        80, "center"),
            ("status",   "Status",         100, "center"),
        ]
        for col, heading, width, anchor in col_cfg:
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, anchor=anchor, stretch=(col == "name"))  # type: ignore[arg-type]

        self.tree.tag_configure("bad",        background="#3b1e1e", foreground="#f48771")
        self.tree.tag_configure("few_frames", background="#2e2b1a", foreground="#dcdcaa")
        self.tree.tag_configure("ok",         background="#1e2a1e", foreground="#b5cea8")

        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Delete>", lambda _e: self._delete_selected())
        self.tree.bind("<<TreeviewSelect>>", self._on_row_select)
        # Up/Down arrows for episode navigation (move selection + load)
        self.tree.bind("<KeyPress-Up>",   lambda _e: self.root.after(10, self._on_row_select))
        self.tree.bind("<KeyPress-Down>", lambda _e: self.root.after(10, self._on_row_select))

    # ── Frame viewer panel ─────────────────────────────────────────────────────

    def _build_viewer_panel(self, parent: tk.Frame) -> None:
        title = tk.Label(parent, text="Frame Viewer  (↑↓ episodes · ← → frames)",
                         bg="#1e1e1e", fg="#cccccc", anchor="w")
        title.configure(font=("TkDefaultFont", 11, "bold"))
        title.pack(fill=tk.X, padx=6, pady=(4, 2))

        # 2x2 image canvas
        self._canvas = tk.Canvas(parent, bg="#000000", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self._canvas.bind("<Configure>", lambda _e: self.root.after(50, lambda: self._show_frame(self._current_frame_idx)))
        self._canvas.bind("<KeyPress-Left>",  lambda _e: self._step_frame(-1))
        self._canvas.bind("<KeyPress-Right>", lambda _e: self._step_frame(+1))
        self._canvas.bind("<KeyPress-Up>",    lambda _e: self._step_episode(-1))
        self._canvas.bind("<KeyPress-Down>",  lambda _e: self._step_episode(+1))
        self._canvas.bind("<Button-1>", lambda _e: self._canvas.focus_set())
        self._canvas.config(takefocus=True)

        if not PIL_AVAILABLE:
            self._canvas.create_text(200, 150, text="Install Pillow to enable viewer\n(pip install Pillow)",
                                     fill="#888888", font=("Courier", 12), justify=tk.CENTER)

        # Gripper state bar (solid color indicator)
        self._grasp_canvas = tk.Canvas(parent, bg="#2c2c2c", height=GRASP_BAR_H,
                                       highlightthickness=0)
        self._grasp_canvas.pack(fill=tk.X, padx=6, pady=(4, 1))
        self._grasp_canvas.bind("<Configure>", lambda _e: self._update_grasp_bar())

        # Contact / intervention state bar
        self._contact_canvas = tk.Canvas(parent, bg="#2c2c2c", height=GRASP_BAR_H,
                                         highlightthickness=0)
        self._contact_canvas.pack(fill=tk.X, padx=6, pady=(1, 1))
        self._contact_canvas.bind("<Configure>", lambda _e: self._update_contact_bar())

        # Gripper qpos progress bar
        self._qpos_canvas = tk.Canvas(parent, bg="#1e1e1e", height=GRASP_BAR_H,
                                      highlightthickness=0)
        self._qpos_canvas.pack(fill=tk.X, padx=6, pady=(1, 2))
        self._qpos_canvas.bind("<Configure>", lambda _e: self._update_qpos_bar())

        # Pose display
        self._pose_var = tk.StringVar(value="")
        pose_lbl = tk.Label(parent, textvariable=self._pose_var, bg="#1e1e1e", fg="#9cdcfe",
                            anchor="center")
        pose_lbl.configure(font=("Courier", 10))
        pose_lbl.pack(fill=tk.X, padx=6, pady=(2, 0))

        # Frame info label
        self._frame_info_var = tk.StringVar(value="← select an episode")
        info_lbl = tk.Label(parent, textvariable=self._frame_info_var,
                            bg="#1e1e1e", fg="#888888", anchor="center")
        info_lbl.configure(font=("Courier", 9))
        info_lbl.pack(fill=tk.X, padx=6)

        # Scrub slider
        self._slider_var = tk.DoubleVar(value=0.0)
        self._slider = ttk.Scale(parent, from_=0, to=0, orient=tk.HORIZONTAL,
                                 variable=self._slider_var, command=self._on_slider_move)
        self._slider.pack(fill=tk.X, padx=6, pady=(2, 6))

    # ── Viewer logic ───────────────────────────────────────────────────────────

    def _on_row_select(self, _event: Any = None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        name = self.tree.item(sel[0], "values")[0]
        identifier = name if self._hdf5_path else os.path.join(self.data_dir, name)
        if identifier != self._viewer_episode_path:
            self._viewer_episode_path = identifier
            self._load_frames(identifier)

    def _load_frames(self, episode_path: str) -> None:
        if self._hdf5_path:
            self._load_frames_hdf5(episode_path)
            return

        # Load frame file lists for all four cameras
        self._frame_files_by_cam = {}
        for cam in range(1, 5):
            rgb_dir = os.path.join(episode_path, f"cam{cam}", "rgb")
            if os.path.exists(rgb_dir):
                try:
                    files = sorted(
                        os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith(".png")
                    )
                    if files:
                        self._frame_files_by_cam[cam] = files
                except OSError:
                    pass

        if not self._frame_files_by_cam:
            self._n_frames = 0
            self._actions_xyz = None
            self._grasp_data = None
            self._contact_data = None
            self._gripper_qpos_data = None
            self._frame_info_var.set("No frames found")
            self._canvas.delete("all")
            self._grasp_canvas.delete("all")
            self._contact_canvas.delete("all")
            self._qpos_canvas.delete("all")
            return

        self._n_frames = max(len(v) for v in self._frame_files_by_cam.values())
        self._actions_xyz = self._load_actions(episode_path)
        self._grasp_data = self._load_grasp(episode_path)
        self._contact_data = self._load_contact(episode_path)
        self._pose_data = self._load_pose(episode_path)
        self._gripper_qpos_data = self._load_gripper_qpos(episode_path)

        self._slider.configure(to=max(0, self._n_frames - 1))
        self._slider_var.set(0)
        self._current_frame_idx = 0
        self._show_frame(0)

    def _load_frames_hdf5(self, demo_name: str) -> None:
        """Load frame data for a demo from the open HDF5 file."""
        self._frame_files_by_cam = {}  # not used in HDF5 mode
        self._hdf5_cam_keys = {}
        self._actions_xyz = None

        cam_key_map = {
            'cam1_image': 1,
            'cam2_image': 2,
            'cam3_image': 3,
            'robot0_eye_in_hand_image': 4,
        }

        try:
            demo = self._hdf5_file['data'][demo_name]
            obs = demo.get('obs', {})
            for key, cam_idx in cam_key_map.items():
                if key in obs:
                    self._hdf5_cam_keys[cam_idx] = obs[key]

            if not self._hdf5_cam_keys:
                self._n_frames = 0
                self._grasp_data = None
                self._contact_data = None
                self._frame_info_var.set("No camera data in this demo")
                self._canvas.delete("all")
                self._grasp_canvas.delete("all")
                self._contact_canvas.delete("all")
                return

            self._n_frames = next(iter(self._hdf5_cam_keys.values())).shape[0]

            # Gripper state
            self._grasp_data = None
            if 'robot0_gripper_state' in obs:
                gs = obs['robot0_gripper_state'][:]
                if gs.ndim == 2:
                    gs = gs[:, 0]
                self._grasp_data = gs.astype(float)

            # Contact / intervention state
            self._contact_data = None
            if 'is_contact' in obs:
                ct = obs['is_contact'][:]
                if ct.ndim == 2:
                    ct = ct[:, 0]
                self._contact_data = ct.astype(float)

            # Pose: eef_pos + eef_quat → xyz + rpy
            self._pose_data = None
            if 'robot0_eef_pos' in obs and 'robot0_eef_quat' in obs:
                xyz = obs['robot0_eef_pos'][:]           # (N, 3)
                quat = obs['robot0_eef_quat'][:]         # (N, 4) xyzw
                rpy = self._quat_to_rpy(quat)
                self._pose_data = np.concatenate([xyz, rpy], axis=1)

        except Exception as exc:
            self._n_frames = 0
            self._grasp_data = None
            self._contact_data = None
            self._frame_info_var.set(f"Error loading {demo_name}: {exc}")
            return

        self._slider.configure(to=max(0, self._n_frames - 1))
        self._slider_var.set(0)
        self._current_frame_idx = 0
        self._show_frame(0)

    @staticmethod
    def _quat_to_rpy(qxyzw: np.ndarray) -> np.ndarray:
        """Convert (N,4) [qx,qy,qz,qw] to (N,3) [roll,pitch,yaw] in radians."""
        qx, qy, qz, qw = qxyzw[:, 0], qxyzw[:, 1], qxyzw[:, 2], qxyzw[:, 3]
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        yaw   = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        return np.stack([roll, pitch, yaw], axis=1)

    def _load_pose(self, episode_path: str) -> Optional[np.ndarray]:
        """Load pose_wrt_world.npy and return (N,6) [x,y,z,roll,pitch,yaw]."""
        pose_file = os.path.join(episode_path, "state", "pose_wrt_world.npy")
        if not os.path.exists(pose_file):
            return None
        try:
            poses = np.load(pose_file)       # (N, 7): x y z qx qy qz qw
            if poses.ndim != 2 or poses.shape[1] < 7:
                return None
            xyz = poses[:, :3]
            rpy = self._quat_to_rpy(poses[:, 3:7])
            return np.concatenate([xyz, rpy], axis=1)
        except Exception:
            return None

    def _load_actions(self, episode_path: str) -> Optional[np.ndarray]:
        """Load pose_wrt_world.npy and compute trimmed XYZ actions."""
        pose_file = os.path.join(episode_path, "state", "pose_wrt_world.npy")
        if not os.path.exists(pose_file):
            return None
        try:
            poses = np.load(pose_file)          # (N, 7): x y z qx qy qz qw
            if poses.ndim != 2 or poses.shape[1] < 3 or len(poses) < 2:
                return None
            xyz = poses[:, :3]                  # (N, 3)
            delta = np.diff(xyz, axis=0)        # (N-1, 3): a_t = pose[t+1] - pose[t]
            delta[np.abs(delta) < ACTION_TRIM] = 0.0
            padding = np.zeros((1, 3), dtype=delta.dtype)
            actions = np.concatenate([delta, padding], axis=0)  # (N, 3)
            return actions
        except Exception:
            return None

    def _load_grasp(self, episode_path: str) -> Optional[np.ndarray]:
        """Load state/grasp.npy — expected shape (N,) with 0=open, 1=grasp."""
        grasp_file = os.path.join(episode_path, "state", "grasp.npy")
        if not os.path.exists(grasp_file):
            return None
        try:
            data = np.load(grasp_file)
            if data.ndim == 1:
                return data.astype(float)
            return None
        except Exception:
            return None

    def _load_contact(self, episode_path: str) -> Optional[np.ndarray]:
        """Load state/is_contact.npy — expected shape (N,) with 0=no contact, 1=contact."""
        contact_file = os.path.join(episode_path, "state", "is_contact.npy")
        if not os.path.exists(contact_file):
            return None
        try:
            data = np.load(contact_file)
            if data.ndim == 1:
                return data.astype(float)
            return None
        except Exception:
            return None

    def _load_gripper_qpos(self, episode_path: str) -> Optional[np.ndarray]:
        """Load state/gripper_qpos.npy — expected shape (N,) raw float 0–1."""
        qpos_file = os.path.join(episode_path, "state", "gripper_qpos.npy")
        if not os.path.exists(qpos_file):
            return None
        try:
            data = np.load(qpos_file)
            if data.ndim == 1:
                return data.astype(float)
            return None
        except Exception:
            return None

    def _on_slider_move(self, val: str) -> None:
        idx = int(float(val))
        if idx != self._current_frame_idx:
            self._current_frame_idx = idx
            self._show_frame(idx)

    def _step_frame(self, delta: int) -> None:
        if self._n_frames == 0:
            return
        new_idx = max(0, min(self._n_frames - 1, self._current_frame_idx + delta))
        if new_idx != self._current_frame_idx:
            self._current_frame_idx = new_idx
            self._slider_var.set(new_idx)
            self._show_frame(new_idx)

    def _step_episode(self, delta: int) -> None:
        """Move episode selection up (-1) or down (+1) in the tree."""
        children = self.tree.get_children()
        if not children:
            return
        sel = self.tree.selection()
        if sel:
            current_iid = sel[0]
            try:
                idx = list(children).index(current_iid)
            except ValueError:
                idx = 0
        else:
            idx = 0 if delta > 0 else len(children) - 1
            delta = 0
        new_idx = max(0, min(len(children) - 1, idx + delta))
        new_iid = children[new_idx]
        self.tree.selection_set(new_iid)
        self.tree.see(new_iid)
        self._on_row_select()

    def _show_frame(self, idx: int) -> None:
        if not PIL_AVAILABLE or self._n_frames == 0 or idx >= self._n_frames:
            return

        cw = self._canvas.winfo_width() or 400
        ch = self._canvas.winfo_height() or 300
        if cw < 10 or ch < 10:
            return

        cell_w = cw // 2
        cell_h = ch // 2

        # Build 2x2 composite
        composite = PILImage.new("RGB", (cw, ch), (0, 0, 0))  # type: ignore[possibly-undefined]
        self._photo_refs = []

        cam_positions = [(1, 0, 0), (2, cell_w, 0), (3, 0, cell_h), (4, cell_w, cell_h)]
        for cam, ox, oy in cam_positions:
            img = None
            if self._hdf5_path:
                ds = self._hdf5_cam_keys.get(cam)
                if ds is not None and idx < ds.shape[0]:
                    try:
                        arr = np.array(ds[idx])  # (H, W, 3) RGB uint8
                        img = PILImage.fromarray(arr)  # type: ignore[possibly-undefined]
                    except Exception:
                        pass
            else:
                files = self._frame_files_by_cam.get(cam)
                if files and idx < len(files):
                    try:
                        img = PILImage.open(files[idx])  # type: ignore[possibly-undefined]
                    except Exception:
                        pass
            if img is not None:
                img = img.resize((cell_w, cell_h), _RESAMPLE)  # type: ignore[possibly-undefined]
                composite.paste(img, (ox, oy))
            # Draw cam label overlay
            self._draw_cam_label_on_composite(composite, cam, ox, oy, cell_w, cell_h)

        photo = PILImageTk.PhotoImage(composite)  # type: ignore[possibly-undefined]
        self._photo_refs.append(photo)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        # Draw action overlay on top
        if self._actions_xyz is not None and idx < len(self._actions_xyz):
            self._draw_action_overlay(self._actions_xyz[idx], cw, ch, idx == self._n_frames - 1)

        # Update grasp, contact, and qpos bars
        self._update_grasp_bar()
        self._update_contact_bar()
        self._update_qpos_bar()

        # Update pose label
        if self._pose_data is not None and idx < len(self._pose_data):
            x, y, z, r, p, yw = self._pose_data[idx]
            self._pose_var.set(
                f"X:{x:+8.4f}  Y:{y:+8.4f}  Z:{z:+8.4f}    "
                f"R:{r:+7.4f}  P:{p:+7.4f}  Yaw:{yw:+7.4f}"
            )
        else:
            self._pose_var.set("")

        n = self._n_frames
        cams_available = (sorted(self._hdf5_cam_keys.keys()) if self._hdf5_path
                          else sorted(self._frame_files_by_cam.keys()))
        self._frame_info_var.set(
            f"cams {cams_available}   frame {idx + 1} / {n}   ↑↓ episodes · ← → frames"
        )

    def _draw_cam_label_on_composite(self, img: Any, cam: int, ox: int, oy: int,
                                     cell_w: int, cell_h: int) -> None:
        """Draw a small 'Cam N' text label in the top-left of each cell using PIL."""
        try:
            from PIL import ImageDraw  # type: ignore[import]
            draw = ImageDraw.Draw(img)
            text = f"Cam {cam}"
            tx, ty = ox + 4, oy + 2
            # Shadow
            draw.text((tx + 1, ty + 1), text, fill=(0, 0, 0))
            draw.text((tx, ty), text, fill=(200, 200, 200))
        except Exception:
            pass

    def _draw_action_overlay(self, action: np.ndarray, cw: int, ch: int, is_last: bool) -> None:
        """Draw XYZ action bars in the bottom-left corner of the canvas."""
        ax, ay, az = float(action[0]), float(action[1]), float(action[2])

        pad_x, pad_y = 12, 12
        row_h = 22
        label_w = 18
        bar_max_px = 90
        bar_h = 10
        box_w = label_w + 12 + bar_max_px * 2 + 50
        box_h = row_h * 3 + pad_y
        bx = pad_x
        by = ch - pad_y - box_h

        self._canvas.create_rectangle(bx, by, bx + box_w, by + box_h,
                                      fill="#1a1a1a", outline="#444444",
                                      stipple="gray50")

        axis_labels = ["X", "Y", "Z"]
        values = [ax, ay, az]
        colors_pos = ["#e74c3c", "#2ecc71", "#3498db"]
        colors_neg = ["#c0392b", "#27ae60", "#2980b9"]
        bar_scale = bar_max_px / 0.005

        for i, (label, val, cp, cn) in enumerate(zip(axis_labels, values, colors_pos, colors_neg)):
            ry = by + pad_y // 2 + i * row_h
            center_x = bx + label_w + 12 + bar_max_px

            self._canvas.create_text(bx + label_w, ry + row_h // 2,
                                     text=f"{label}:", fill="#cccccc",
                                     font=("Courier", 9, "bold"), anchor="e")
            self._canvas.create_line(center_x, ry + 2, center_x, ry + bar_h + 2,
                                     fill="#555555", width=1)

            bar_len = int(abs(val) * bar_scale)
            bar_len = min(bar_len, bar_max_px)
            color = cp if val >= 0 else cn
            if bar_len > 0:
                if val >= 0:
                    x0, x1 = center_x, center_x + bar_len
                else:
                    x0, x1 = center_x - bar_len, center_x
                self._canvas.create_rectangle(x0, ry + 3, x1, ry + bar_h + 1,
                                              fill=color, outline="")

            suffix = "" if is_last else "m"
            display_val = 0.0 if is_last else val
            txt = f"{display_val:+.4f}{suffix}" if not is_last else "  ——"
            self._canvas.create_text(center_x + bar_max_px + 6, ry + row_h // 2,
                                     text=txt, fill=color if bar_len > 0 else "#666666",
                                     font=("Courier", 9), anchor="w")

    def _update_grasp_bar(self) -> None:
        """Redraw the gripper state bar as a solid color for the current frame."""
        self._grasp_canvas.delete("all")
        bw = self._grasp_canvas.winfo_width() or 200
        bh = self._grasp_canvas.winfo_height() or GRASP_BAR_H
        if bw < 4:
            return

        idx = self._current_frame_idx
        grasp = self._grasp_data

        if grasp is not None and idx < len(grasp) and self._n_frames > 0:
            closed = float(grasp[idx]) >= 0.5
            color = "#27ae60" if closed else "#444444"
            label = "GRIPPER: CLOSED" if closed else "GRIPPER: OPEN"
            text_color = "#ffffff" if closed else "#888888"
        else:
            color = "#2c2c2c"
            label = "GRIPPER: —"
            text_color = "#555555"

        self._grasp_canvas.create_rectangle(0, 0, bw, bh, fill=color, outline="")
        self._grasp_canvas.create_text(bw // 2, bh // 2, text=label, fill=text_color,
                                       font=("Courier", 9, "bold"), anchor="center")

    def _update_contact_bar(self) -> None:
        """Redraw the contact/intervention state bar as a solid color for the current frame."""
        self._contact_canvas.delete("all")
        bw = self._contact_canvas.winfo_width() or 200
        bh = self._contact_canvas.winfo_height() or GRASP_BAR_H
        if bw < 4:
            return

        idx = self._current_frame_idx
        contact = self._contact_data

        if contact is not None and idx < len(contact) and self._n_frames > 0:
            active = float(contact[idx]) > 0.5
            color = "#e67e22" if active else "#444444"
            label = "CONTACT: YES" if active else "CONTACT: NO"
            text_color = "#ffffff" if active else "#888888"
        else:
            color = "#2c2c2c"
            label = "CONTACT: —"
            text_color = "#555555"

        self._contact_canvas.create_rectangle(0, 0, bw, bh, fill=color, outline="")
        self._contact_canvas.create_text(bw // 2, bh // 2, text=label, fill=text_color,
                                         font=("Courier", 9, "bold"), anchor="center")

    def _update_qpos_bar(self) -> None:
        """Redraw the gripper qpos progress bar for the current frame."""
        self._qpos_canvas.delete("all")
        bw = self._qpos_canvas.winfo_width() or 200
        bh = self._qpos_canvas.winfo_height() or GRASP_BAR_H
        if bw < 4:
            return

        idx = self._current_frame_idx
        qpos_data = self._gripper_qpos_data

        # Background
        self._qpos_canvas.create_rectangle(0, 0, bw, bh, fill="#1e1e1e", outline="")

        if qpos_data is not None and idx < len(qpos_data) and self._n_frames > 0:
            qpos = float(np.clip(qpos_data[idx], 0.0, 1.0))
            fill_w = int((bw - 2) * qpos)

            # Gradient: blue (open) → orange (closed)
            r = int(255 * qpos)
            g = int(100 * (1.0 - qpos))
            b = int(220 * (1.0 - qpos))
            color = f"#{r:02x}{g:02x}{b:02x}"

            if fill_w > 0:
                self._qpos_canvas.create_rectangle(1, 1, 1 + fill_w, bh - 1,
                                                   fill=color, outline="")
            # Border
            self._qpos_canvas.create_rectangle(0, 0, bw - 1, bh - 1,
                                               fill="", outline="#555555")
            label = f"Gripper qpos: {qpos_data[idx]:.3f}"
            text_color = "#ffffff"
        else:
            self._qpos_canvas.create_rectangle(0, 0, bw - 1, bh - 1,
                                               fill="", outline="#333333")
            label = "Gripper qpos: —"
            text_color = "#555555"

        self._qpos_canvas.create_text(bw // 2, bh // 2, text=label, fill=text_color,
                                      font=("Courier", 9, "bold"), anchor="center")

    # ── Episode data loading ───────────────────────────────────────────────────

    def _load_episodes(self) -> list:
        if self._hdf5_path:
            return self._load_episodes_hdf5()
        if not os.path.exists(self.data_dir):
            return []
        episodes = []
        try:
            for entry in os.listdir(self.data_dir):
                if not entry.startswith("episode_"):
                    continue
                path = os.path.join(self.data_dir, entry)
                if os.path.isdir(path):
                    episodes.append(get_episode_info(path))
        except OSError:
            return []
        episodes.sort(key=lambda e: e["sort_key"], reverse=True)
        return episodes

    def _load_episodes_hdf5(self) -> list:
        """Build episode list from the open HDF5 file."""
        episodes = []
        try:
            data_group = self._hdf5_file['data']
            for demo_name in sorted(data_group.keys()):
                demo = data_group[demo_name]
                obs = demo.get('obs', {})
                # Count frames from first available camera
                n_frames = 0
                for key in ('cam1_image', 'cam2_image', 'cam3_image', 'robot0_eye_in_hand_image'):
                    if key in obs:
                        n_frames = obs[key].shape[0]
                        break
                episodes.append({
                    'name': demo_name,
                    'path': None,
                    'frames': n_frames,
                    'duration': 0.0,
                    'fps': 0.0,
                    'bad': False,
                    'date': '',
                    'sort_key': demo_name,
                })
        except Exception as exc:
            print(f"Error reading HDF5 episodes: {exc}")
        return episodes

    def _populate_table(self, episodes: list) -> None:
        selected_names = {self.tree.item(iid, "values")[0] for iid in self.tree.selection()}
        self.tree.delete(*self.tree.get_children())

        total_frames = sum(ep["frames"] for ep in episodes)
        total_duration = sum(ep["duration"] for ep in episodes)

        for ep in episodes:
            frames = ep["frames"]
            if ep["bad"]:
                status_text, tag = "⚠ BAD", "bad"
            elif frames < AUTO_CLEAN_MIN_FRAMES:
                status_text, tag = "⚠ SHORT", "few_frames"
            else:
                status_text, tag = "✓ OK", "ok"

            values = (
                ep["name"],
                ep["date"],
                str(frames),
                f"{ep['duration']:.1f}" if ep["duration"] > 0 else "—",
                f"{ep['fps']:.1f}" if ep["fps"] > 0 else "—",
                status_text,
            )
            iid = self.tree.insert("", tk.END, values=values, tags=(tag,))
            if ep["name"] in selected_names:
                self.tree.selection_add(iid)

        n = len(episodes)
        self.stats_var.set(
            f"  {n} episode{'s' if n != 1 else ''}   "
            f"{total_frames} frames total   "
            f"{total_duration / 60:.1f} min total   "
            f"refreshed {datetime.now().strftime('%H:%M:%S')}  "
        )

    # ── Refresh ────────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        if self._loading:
            return
        self._loading = True
        self.status_var.set("Refreshing…")

        def _load() -> None:
            episodes = self._load_episodes()
            self.root.after(0, lambda: self._on_loaded(episodes))

        threading.Thread(target=_load, daemon=True).start()

    def _on_loaded(self, episodes: list) -> None:
        self._episodes = episodes
        self._populate_table(episodes)
        self.status_var.set("")
        self._loading = False

    def _schedule_refresh(self) -> None:
        self._refresh()
        self.root.after(self.refresh_interval, self._schedule_refresh)

    # ── Delete actions ─────────────────────────────────────────────────────────

    def _selected_episode_info(self) -> list:
        result = []
        for iid in self.tree.selection():
            vals = self.tree.item(iid, "values")
            result.append({"name": vals[0], "path": os.path.join(self.data_dir, vals[0]), "frames": vals[2]})
        return result

    def _clear_viewer_if(self, path: str) -> None:
        if path == self._viewer_episode_path:
            self._viewer_episode_path = ""
            self._frame_files_by_cam = {}
            self._n_frames = 0
            self._actions_xyz = None
            self._grasp_data = None
            self._contact_data = None
            self._pose_data = None
            self._gripper_qpos_data = None
            self._canvas.delete("all")
            self._grasp_canvas.delete("all")
            self._contact_canvas.delete("all")
            self._qpos_canvas.delete("all")
            self._frame_info_var.set("← select an episode")
            self._pose_var.set("")

    def _delete_selected(self) -> None:
        if self._hdf5_path:
            self._delete_selected_hdf5()
            return

        selected = self._selected_episode_info()
        if not selected:
            messagebox.showinfo("No Selection", "Select one or more episodes to delete.")
            return
        names_str = "\n".join(f"  • {ep['name']}  ({ep['frames']} frames)" for ep in selected)
        if not messagebox.askyesno("Confirm Delete",
                                   f"Permanently delete {len(selected)} episode(s)?\n\n{names_str}"):
            return
        deleted = 0
        for ep in selected:
            try:
                if os.path.exists(ep["path"]):
                    shutil.rmtree(ep["path"])
                    deleted += 1
                    self._clear_viewer_if(ep["path"])
            except Exception as exc:
                messagebox.showerror("Error", f"Could not delete {ep['name']}:\n{exc}")
        self.status_var.set(f"Deleted {deleted} episode(s)")
        self._refresh()

    def _delete_selected_hdf5(self) -> None:
        selected = self._selected_episode_info()
        if not selected:
            messagebox.showinfo("No Selection", "Select one or more demos to delete.")
            return
        names_str = "\n".join(f"  • {ep['name']}  ({ep['frames']} frames)" for ep in selected)
        if not messagebox.askyesno("Confirm Delete",
                                   f"Permanently delete {len(selected)} demo(s) from HDF5?\n\n{names_str}"):
            return
        deleted = 0
        for ep in selected:
            demo_name = ep["name"]
            try:
                data_group = self._hdf5_file['data']
                if demo_name in data_group:
                    del data_group[demo_name]
                    self._hdf5_file.flush()
                    deleted += 1
                    if demo_name == self._viewer_episode_path:
                        self._viewer_episode_path = ""
                        self._frame_files_by_cam = {}
                        self._hdf5_cam_keys = {}
                        self._n_frames = 0
                        self._actions_xyz = None
                        self._grasp_data = None
                        self._contact_data = None
                        self._pose_data = None
                        self._gripper_qpos_data = None
                        self._canvas.delete("all")
                        self._grasp_canvas.delete("all")
                        self._contact_canvas.delete("all")
                        self._qpos_canvas.delete("all")
                        self._frame_info_var.set("← select an episode")
                        self._pose_var.set("")
            except Exception as exc:
                messagebox.showerror("Error", f"Could not delete {demo_name}:\n{exc}")
        self.status_var.set(f"Deleted {deleted} demo(s) from HDF5")
        self._refresh()

    def _auto_clean(self) -> None:
        episodes = self._load_episodes()
        to_delete = [ep for ep in episodes if ep["frames"] < AUTO_CLEAN_MIN_FRAMES]
        if not to_delete:
            messagebox.showinfo("Auto Clean",
                                f"No episodes with fewer than {AUTO_CLEAN_MIN_FRAMES} frames.")
            return
        names_str = "\n".join(f"  • {ep['name']}  ({ep['frames']} frames)" for ep in to_delete)
        if not messagebox.askyesno("Auto Clean",
                                   f"Delete {len(to_delete)} short episode(s)?\n\n{names_str}"):
            return
        deleted = 0
        for ep in to_delete:
            try:
                if os.path.exists(ep["path"]):
                    shutil.rmtree(ep["path"])
                    deleted += 1
                    self._clear_viewer_if(ep["path"])
            except Exception as exc:
                messagebox.showerror("Error", f"Could not delete {ep['name']}:\n{exc}")
        self.status_var.set(f"Auto clean: deleted {deleted} episode(s)")
        self._refresh()

    # ── Entry point ────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            if self._hdf5_file is not None:
                self._hdf5_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Episode Data Manager GUI")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"Path to episodes directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--refresh", type=int, default=5,
                        help="Auto-refresh interval in seconds (default: 5)")
    parser.add_argument("--hdf5", default=None, metavar="PATH",
                        help="Open an HDF5 dataset file instead of an episode directory")
    args = parser.parse_args()

    gui = DataManagerGUI(data_dir=args.data_dir, refresh_interval=args.refresh * 1000,
                         hdf5_path=args.hdf5)
    gui.run()


if __name__ == "__main__":
    main()
