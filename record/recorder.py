"""HTTP observation recorder for Franka teleoperation datasets.

Episodes use MP4 for RGB because raw frames are roughly 40 times larger and a
browser can scrub a video natively without shipping a decoder.  The state and
actions live in Parquet because they are typed columnar data that pandas and
pyarrow read in one call, and are already close to the shape LeRobot expects;
later conversion is mechanical.  JSON holds the metadata because, a year from
now, the most likely missing fact is what the numeric columns meant.

This process is also the spacemouse teleop process.  Cameras belong to the
robot server, so observations are sampled over HTTP and recording never opens
a RealSense device.

Schema version 2 adds the held gripper command and derived grasp label, raw
spacemouse motion, applied motion delta, button state, and paired action
timestamp to each row.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from control.robot_client import RobotClient
from control.teleop import (SpacemouseConfig, SpacemousePolicy, TeleopDriver,
                            resolve_spacemouse_config as _resolve_spacemouse_config)
from control.teleop.tuning import TELEOP_CONFIG_FILE
from control.ik import DifferentialIK

SCHEMA_VERSION = 2
DEFAULT_GRIPPER_STROKE_MM = 85.0


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _array(value, size: int, default=0.0) -> list[float]:
    if value is None:
        return [float(default)] * size
    result = np.asarray(value, dtype=np.float32).ravel()
    if result.size != size:
        raise ValueError(f"expected vector of length {size}, got {result.size}")
    return result.tolist()


def _pose(pose):
    return _array(pose.position, 3), _array(pose.orientation.as_quat(), 4)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _config_snapshot(name: str) -> str | None:
    path = Path(__file__).resolve().parents[1] / "config" / name
    return path.read_text() if path.exists() else None


def camera_specs_from_config(path="config/cameras.yaml") -> dict[str, dict]:
    """Read the stable role, dimensions, and fps from the server camera config."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return {name: dict(spec) for name, spec in (data.get("cameras") or {}).items()}


@dataclass
class RecorderConfig:
    output_dir: Path | str
    record_fps: float = 15.0
    queue_size: int = 60
    crf: int = 18
    depth: bool = False
    min_frames: int = 16
    task: str = ""
    notes: str = ""
    control_space: str = "cartesian"
    gripper_stroke_mm: float = DEFAULT_GRIPPER_STROKE_MM
    control_rate: float = 50.0
    teleop: dict | None = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


def resolve_spacemouse_config(args, control_space: str, ik=None) -> SpacemouseConfig:
    """Resolve recorder teleop tuning without opening hardware or a network client."""
    return _resolve_spacemouse_config(
        control_space, ik=ik, action_size=args.action_size,
        rotation_size=args.rotation_size, deadzone=args.deadzone,
        max_lead_m=args.max_lead, max_lead_rad=args.max_lead_rad,
        brake_on_release=args.brake, brake_release_ticks=args.brake_ticks,
        keyboard_reset=False)


def teleop_metadata(policy_config: SpacemouseConfig, control_space: str,
                   control_rate: float) -> dict:
    return {
        "action_size": policy_config.action_size,
        "rotation_size": policy_config.rotation_size,
        "max_lead_m": policy_config.max_lead_m,
        "max_lead_rad": policy_config.max_lead_rad,
        "brake_on_release": policy_config.brake_on_release,
        "brake_release_ticks": policy_config.brake_release_ticks,
        "deadzone": policy_config.deadzone,
        "control_space": control_space,
        "control_rate": control_rate,
        "config_path": str(TELEOP_CONFIG_FILE),
        "config_mtime": TELEOP_CONFIG_FILE.stat().st_mtime,
    }


@dataclass
class RecordedSample:
    """One sampler result paired with the most recent command."""

    observation: object
    action: object
    q: np.ndarray | None = None
    action_timestamp: float | None = None


def _schema(camera_names: list[str]) -> pa.Schema:
    # Plain Arrow list columns permit a null *whole vector* for optional
    # actions; every non-null value is validated to the documented length by
    # _array before it reaches this schema.
    vector = lambda n: pa.list_(pa.float32())
    fixed_vector = lambda n: pa.list_(pa.float32(), n)
    fields = [
        pa.field("frame_index", pa.int32()), pa.field("timestamp", pa.float64()),
        pa.field("t_rel", pa.float32()), pa.field("joint_position", vector(7)),
        pa.field("ee_position", vector(3)), pa.field("ee_quat_xyzw", vector(4)),
        pa.field("fingertip_position", vector(3)), pa.field("fingertip_quat_xyzw", vector(4)),
        pa.field("gripper_value", pa.float32()), pa.field("gripper_width_mm", pa.float32()),
        pa.field("ft_wrench", vector(6)),
        pa.field("action_fingertip_position", vector(3)),
        pa.field("action_fingertip_quat_xyzw", vector(4)),
        pa.field("action_grasp", pa.int8()),
        # Held command (1.0 open, 0.0 closed), not the edge-only action.gripper.
        pa.field("action_gripper", pa.float32()),
        pa.field("action_delta", fixed_vector(6)),
        pa.field("spacemouse_motion", fixed_vector(6)),
        pa.field("spacemouse_button", pa.bool_()),
        pa.field("action_timestamp", pa.float64()),
        pa.field("action_joint_position", vector(7)),
    ]
    for name in camera_names:
        fields.extend([pa.field(f"{name}_age_s", pa.float32()),
                       pa.field(f"{name}_stale", pa.bool_())])
    return pa.schema(fields)


class EpisodeWriter:
    """Synchronous episode writer; safe to drive directly in synthetic tests.

    ``write`` is deliberately independent of HTTP and spacemouse code.  A
    background ``RecordingSession`` calls it in production, while tests feed
    synthetic observations directly.
    """

    def __init__(self, config: RecorderConfig, episode_id: int,
                 camera_specs: dict[str, dict], camera_names: list[str] | None = None):
        self.config = config
        self.episode_id = int(episode_id)
        self.camera_specs = camera_specs
        self.camera_names = list(camera_names or camera_specs)
        self.path = config.output_dir / f"episode_{self.episode_id:06d}"
        self.rows: list[dict] = []
        self._last_rgb: dict[str, np.ndarray] = {}
        self._containers: dict[str, object] = {}
        self._streams: dict[str, object] = {}
        self._start_wall = _iso_now()
        self._first_timestamp: float | None = None
        self._opened = False

    @property
    def n_frames(self):
        return len(self.rows)

    def start(self):
        if self._opened:
            return self
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "video").mkdir()
        if self.config.depth:
            (self.path / "depth").mkdir()
        for name in self.camera_names:
            spec = self.camera_specs[name]
            container = av.open(str(self.path / "video" / f"{name}.mp4"), "w",
                                options={"movflags": "+faststart"})
            # PyAV requires an int or Fraction rate; the CLI deliberately keeps
            # record_fps as a float for arithmetic and JSON metadata.
            stream = container.add_stream(
                "libx264", rate=Fraction(self.config.record_fps).limit_denominator(1000))
            stream.width, stream.height = int(spec["width"]), int(spec["height"])
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(self.config.crf)}
            stream.gop_size = int(round(self.config.record_fps))
            self._containers[name], self._streams[name] = container, stream
            if self.config.depth:
                (self.path / "depth" / name).mkdir()
        self._opened = True
        return self

    def _rgb_for_camera(self, observation, name: str) -> tuple[np.ndarray, bool, float | None]:
        frame = (getattr(observation, "cameras", {}) or {}).get(name)
        rgb = None if frame is None else frame.rgb
        stale = frame is None or bool(frame.stale) or rgb is None
        age_s = None if frame is None else frame.age_s
        spec = self.camera_specs[name]
        if stale:
            rgb = self._last_rgb.get(name)
            if rgb is None:
                rgb = np.zeros((int(spec["height"]), int(spec["width"]), 3), dtype=np.uint8)
        else:
            rgb = np.asarray(rgb)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"{name} RGB must be HxWx3, got {rgb.shape}")
            if rgb.shape[:2] != (int(spec["height"]), int(spec["width"])):
                raise ValueError(f"{name} RGB shape {rgb.shape[:2]} does not match configured "
                                 f"{spec['height']}x{spec['width']}")
            rgb = np.ascontiguousarray(rgb.astype(np.uint8, copy=False))
            self._last_rgb[name] = rgb.copy()
        return rgb, stale, age_s

    def write(self, sample: RecordedSample):
        """Encode exactly one frame per camera and append its aligned row.

        Row/frame alignment is a hard invariant: row i is frame i in every
        camera video.  A stale camera is repeated (or black initially), never
        skipped, because loaders and review tools rely on this mapping.
        """
        if not self._opened:
            self.start()
        obs, action = sample.observation, sample.action
        timestamp = float(obs.timestamp)
        action_timestamp = sample.action_timestamp
        if action_timestamp is None or not np.isfinite(action_timestamp):
            raise ValueError("RecordedSample.action_timestamp must be finite")
        if self._first_timestamp is None:
            self._first_timestamp = timestamp
        camera_values = {}
        for name in self.camera_names:
            rgb, stale, age_s = self._rgb_for_camera(obs, name)
            video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in self._streams[name].encode(video_frame):
                self._containers[name].mux(packet)
            camera_values[name] = (stale, age_s)
            if self.config.depth:
                frame = (getattr(obs, "cameras", {}) or {}).get(name)
                if frame is not None and frame.depth is not None:
                    depth = np.asarray(frame.depth, dtype=np.uint16)
                    cv2.imwrite(str(self.path / "depth" / name /
                                    f"{len(self.rows):06d}.png"), depth)

        fingertip_position, fingertip_quat = _pose(obs.fingertip_pose)
        gripper = getattr(obs, "gripper_value", None)
        row = {
            "frame_index": len(self.rows), "timestamp": timestamp,
            "t_rel": np.float32(timestamp - self._first_timestamp).item(),
            "joint_position": _array(obs.joint_values, 7),
            "ee_position": _array(obs.ee_position, 3),
            "ee_quat_xyzw": _array(obs.ee_quat_xyzw, 4),
            "fingertip_position": fingertip_position, "fingertip_quat_xyzw": fingertip_quat,
            "gripper_value": None if gripper is None else np.float32(gripper).item(),
            "gripper_width_mm": (getattr(obs, "gripper_width_mm", None) if gripper is not None
                                  else None),
            "ft_wrench": _array(getattr(obs, "ft_wrench", None), 6),
            "action_fingertip_position": _array(action.position, 3),
            "action_fingertip_quat_xyzw": _array(action.quat_xyzw, 4),
            "action_grasp": np.int8(action.gripper_latched < 0.5).item(),
            "action_gripper": np.float32(action.gripper_latched).item(),
            "action_delta": _array(action.delta, 6),
            "spacemouse_motion": _array(action.motion, 6),
            "spacemouse_button": bool(action.button),
            "action_timestamp": float(action_timestamp),
            "action_joint_position": None if sample.q is None else _array(sample.q, 7),
        }
        if row["gripper_width_mm"] is None and gripper is not None:
            row["gripper_width_mm"] = np.float32(
                gripper * self.config.gripper_stroke_mm).item()
        for name, (stale, age_s) in camera_values.items():
            row[f"{name}_age_s"] = None if age_s is None else np.float32(age_s).item()
            row[f"{name}_stale"] = stale
        self.rows.append(row)

    def _close_video(self):
        for name in self.camera_names:
            stream, container = self._streams[name], self._containers[name]
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self._streams.clear()
        self._containers.clear()

    def _meta(self):
        cameras = []
        for name in self.camera_names:
            spec = self.camera_specs[name]
            cameras.append({"name": name, "width": int(spec["width"]),
                            "height": int(spec["height"]), "codec": "libx264",
                            "crf": self.config.crf,
                            "gop": int(round(self.config.record_fps))})
        return {"schema_version": SCHEMA_VERSION, "episode_id": self.episode_id,
                "start_wall_time": self._start_wall, "end_wall_time": _iso_now(),
                "n_frames": self.n_frames, "record_fps": self.config.record_fps,
                "control_space": self.config.control_space, "task": self.config.task,
                "operator_notes": self.config.notes, "bad": False, "cameras": cameras,
                "gripper_stroke_mm": self.config.gripper_stroke_mm,
                "teleop": self.config.teleop,
                "git_commit": _git_commit(), "cameras_yaml": _config_snapshot("cameras.yaml"),
                "gripper_geometry_yaml": _config_snapshot("gripper_geometry.yaml")}

    def finalize(self, discard=False) -> Path | None:
        if not self._opened:
            return None
        self._close_video()
        self._opened = False
        if discard or self.n_frames < self.config.min_frames:
            shutil.rmtree(self.path)
            return None
        pq.write_table(pa.Table.from_pylist(self.rows, schema=_schema(self.camera_names)),
                       self.path / "data.parquet")
        (self.path / "meta.json").write_text(json.dumps(self._meta(), indent=2) + "\n")
        rewrite_dataset_manifest(self.config.output_dir, self.camera_names)
        return self.path


def rewrite_dataset_manifest(root: Path | str, camera_names: list[str]):
    root = Path(root)
    episodes = sorted(path.name for path in root.glob("episode_*") if (path / "meta.json").exists())
    payload = {"schema_version": SCHEMA_VERSION, "created_at": _iso_now(),
               "robot_description": "Franka FR3 with Robotiq 2F-85",
               "camera_roles": list(camera_names), "episodes": episodes}
    (root / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n")


class RecordingSession:
    """Bounded producer/consumer wrapper that keeps disk/video work off teleop."""

    def __init__(self, config: RecorderConfig, camera_specs: dict[str, dict]):
        self.config, self.camera_specs = config, camera_specs
        self.queue: queue.Queue[RecordedSample] = queue.Queue(maxsize=config.queue_size)
        self.writer: EpisodeWriter | None = None
        self.thread: threading.Thread | None = None
        self.recording = False
        self.dropped = 0
        self._state_lock = threading.Lock()
        self._stopping = False

    def _next_id(self):
        ids = [int(p.name.split("_")[-1]) for p in self.config.output_dir.glob("episode_*")
               if p.name.split("_")[-1].isdigit()]
        return max(ids, default=0) + 1

    def start(self, camera_names: list[str]):
        with self._state_lock:
            if self.recording or self._stopping:
                return
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            self.writer = EpisodeWriter(
                self.config, self._next_id(), self.camera_specs, camera_names).start()
            self.queue = queue.Queue(maxsize=self.config.queue_size)
            self.dropped = 0
            self.recording = True
            self.thread = threading.Thread(target=self._drain, daemon=True)
            self.thread.start()

    def submit(self, sample: RecordedSample):
        with self._state_lock:
            if not self.recording or self._stopping:
                return
            try:
                self.queue.put_nowait(sample)
            except queue.Full:
                self.dropped += 1
                if self.dropped == 1 or self.dropped % 100 == 0:
                    print(f"[recorder] queue full; dropped {self.dropped} samples")

    def _drain(self):
        while True:
            try:
                sample = self.queue.get(timeout=0.1)
            except queue.Empty:
                with self._state_lock:
                    if not self.recording:
                        return
                continue
            try:
                self.writer.write(sample)  # type: ignore[union-attr]
            finally:
                self.queue.task_done()

    def stop(self, discard=False):
        with self._state_lock:
            if not self.writer or self._stopping:
                return None
            # A producer can pass a bare recording check just before this
            # point and enqueue after the drain loop exits, leaving queue.join
            # blocked forever.  Gate the check and enqueue with this same lock.
            self.recording = False
            self._stopping = True
            thread, writer = self.thread, self.writer
        if thread:
            thread.join()
        while True:
            try:
                sample = self.queue.get_nowait()
            except queue.Empty:
                break
            try:
                writer.write(sample)
            finally:
                self.queue.task_done()
        result = writer.finalize(discard=discard)
        with self._state_lock:
            self.writer = None
            self.thread = None
            self._stopping = False
        return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:7000")
    p.add_argument("--output-dir", default="recordings")
    p.add_argument("--rate", type=float, default=50.0, help="teleop control rate [Hz]")
    p.add_argument("--record-fps", type=float, default=15.0)
    p.add_argument("--queue-size", type=int, default=60)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--depth", action="store_true")
    p.add_argument("--min-frames", type=int, default=16)
    p.add_argument("--task", default="")
    p.add_argument("--notes", default="")
    p.add_argument("--action-size", type=float, default=None,
                   help="per-step translation gain [m] (default: control-space "
                        "value in config/teleop.yaml)")
    p.add_argument("--rotation-size", type=float, default=None,
                   help="per-step rotation gain (default: follow --action-size)")
    p.add_argument("--deadzone", type=float, default=None)
    p.add_argument("--max-lead", type=float, default=None,
                   help="maximum target lead [m] (per-space default when omitted)")
    p.add_argument("--max-lead-rad", type=float, default=None,
                   help="maximum target lead [rad] (per-space default when omitted)")
    p.add_argument("--brake-ticks", type=int, default=None,
                   help="idle ticks before the release brake fires")
    p.add_argument("--no-brake", dest="brake", action="store_false", default=None,
                   help="do not snap the target onto the arm on release")
    p.add_argument("--max-step-rad", type=float, default=0.05)
    p.add_argument("--damping", type=float, default=0.05)
    return p.parse_args(argv)


def main():
    args = parse_args()  # --help exits before any hardware/network object exists.
    robot = RobotClient(args.url)
    ctrl_space = robot.ctrl_space
    if ctrl_space not in (None, "cartesian", "joint"):
        raise SystemExit(f"the robot server at {args.url} reported unrecognised control "
                         f"space {ctrl_space!r}; expected 'cartesian' or 'joint'")
    drive_space = ctrl_space or "cartesian"
    ik = (DifferentialIK(frame="fingertip", damping=args.damping,
                         max_step_rad=args.max_step_rad) if drive_space == "joint" else None)
    policy_config = resolve_spacemouse_config(args, drive_space, ik)
    config = RecorderConfig(
        args.output_dir, args.record_fps, args.queue_size, args.crf, args.depth,
        args.min_frames, args.task, args.notes, drive_space, control_rate=args.rate,
        teleop=teleop_metadata(policy_config, drive_space, args.rate))
    specs = camera_specs_from_config()
    policy = SpacemousePolicy(policy_config)
    session = RecordingSession(config, specs)
    latest = {"action": None, "q": None, "timestamp": None}
    latest_lock = threading.Lock()
    stop = threading.Event()

    # One listener owns all recorder keys; policy keyboard_reset is disabled.
    from pynput import keyboard
    keys: queue.Queue[str] = queue.Queue()
    def on_press(key):
        char = getattr(key, "char", None)
        if char in (" ", "r", "f", "h", "q"):
            keys.put(char)
        elif key == keyboard.Key.space:
            keys.put(" ")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    def sampler():
        period = 1.0 / args.record_fps
        while not stop.is_set():
            tick = time.perf_counter()
            if session.recording:
                obs = robot.get_obs(include_depth=args.depth)
                with latest_lock:
                    action, q, action_timestamp = (latest["action"], latest["q"],
                                                   latest["timestamp"])
                if action is not None:
                    session.submit(RecordedSample(obs, action, q, action_timestamp))
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))
    sampler_thread = threading.Thread(target=sampler, daemon=True)
    last_status_print = 0.0

    print(f"Driving in {drive_space} control space. space=start/stop r=save+next f=discard+next h=home q=quit")
    try:
        with TeleopDriver(robot, policy, drive_space, ik=ik) as driver:
            sampler_thread.start()
            while not stop.is_set():
                tick = time.perf_counter()
                try:
                    while True:
                        key = keys.get_nowait()
                        if key == " ":
                            if session.recording: session.stop()
                            else: session.start(list(specs))
                        elif key == "r":
                            if session.recording: session.stop()
                            session.start(list(specs))
                        elif key == "f":
                            if session.recording: session.stop(discard=True)
                            session.start(list(specs))
                        elif key == "h":
                            print("[h] re-homing..."); robot.home(); driver.reset_anchor()
                        elif key == "q": stop.set()
                except queue.Empty:
                    pass
                teleop = driver.step()
                with latest_lock:
                    latest["action"], latest["q"], latest["timestamp"] = (
                        teleop.action, teleop.q, time.time())
                if session.writer and tick - last_status_print >= 0.1:
                    print(f"\rE{session.writer.episode_id:06d} {session.writer.n_frames:5d} "
                          f"{session.writer.n_frames / args.record_fps:5.1f}s drop={session.dropped} "
                          f"held={teleop.held} lag={driver.lagging}", end="", flush=True)
                    last_status_print = tick
                time.sleep(max(0.0, 1.0 / args.rate - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        print()
    finally:
        stop.set()
        if session.recording: session.stop()
        listener.stop()
        sampler_thread.join(timeout=2)


if __name__ == "__main__":
    main()
