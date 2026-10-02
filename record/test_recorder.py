"""Synthetic end-to-end checks for the recorder format.

Run with ``python -m record.test_recorder``.  This deliberately creates no
robot client and opens no input devices.
"""

import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from control.teleop import Action
from control.ik import MAX_LEAD_M, MAX_LEAD_RAD
from record.dataset import load_dataset, load_episode
from record.export_legacy import export_dataset
from record.recorder import (EpisodeWriter, RecordedSample, RecorderConfig,
                             RecordingSession, parse_args, resolve_spacemouse_config,
                             teleop_metadata)


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    assert cond, name


class Rotation:
    def __init__(self, quat): self.quat = np.asarray(quat, dtype=np.float32)
    def as_quat(self): return self.quat


class Pose:
    def __init__(self, position, quat):
        self.position = np.asarray(position, dtype=np.float32)
        self.orientation = Rotation(quat)


class Camera:
    def __init__(self, rgb, stale=False, age_s=0.01, depth=None):
        self.rgb, self.stale, self.age_s, self.depth = rgb, stale, age_s, depth


class Observation:
    def __init__(self, index, cameras):
        self.timestamp = 1_700_000_000.0 + index / 15.0
        self.joint_values = np.arange(7, dtype=np.float32) + index
        self.ee_position = np.array([index, index + 1, index + 2], dtype=np.float32) / 100
        self.ee_quat_xyzw = np.array([0, 0, 0, 1], dtype=np.float32)
        self.gripper_value = float(index % 2)
        self.gripper_width_mm = self.gripper_value * 85.0
        self.ft_wrench = np.arange(6, dtype=np.float32) + index / 10
        self.fingertip_pose = Pose(self.ee_position + .06, self.ee_quat_xyzw)
        self.cameras = cameras


def sample(index, stale=False):
    agent = np.full((24, 32, 3), (index * 5) % 255, dtype=np.uint8)
    inhand = np.full((12, 20, 3), (index * 7) % 255, dtype=np.uint8)
    cameras = {"agentview": Camera(None if stale else agent, stale=stale,
                                    depth=np.full((24, 32), index, dtype=np.uint16)),
               "inhand": Camera(inhand, depth=np.full((12, 20), index, dtype=np.uint16))}
    obs = Observation(index, cameras)
    latched = 1.0 if index < 20 else 0.0
    motion = np.arange(6, dtype=np.float32) + index / 10
    delta = motion * np.array([.003, .003, .003, .006, -.003, -.009], dtype=np.float32)
    action = Action(position=obs.fingertip_pose.position + .001,
                    quat_xyzw=np.array([0, 0, 0, 1], dtype=np.float32),
                    gripper=0.0 if index == 20 else None,
                    gripper_latched=latched,
                    motion=motion,
                    delta=delta,
                    button=bool(index % 4 == 0))
    q = None if index % 2 else obs.joint_values + .01
    return RecordedSample(obs, action, q, action_timestamp=1_700_000_100.0 + index / 50.0)


def main():
    print("=== spacemouse tuning ===")
    joint_ik = SimpleNamespace(max_lead_m=MAX_LEAD_M * 0.8,
                               max_lead_rad=MAX_LEAD_RAD * 0.8)
    joint_config = resolve_spacemouse_config(parse_args([]), "joint", joint_ik)
    check("joint defaults derive shared tuning", joint_config.action_size == 0.008 and
          joint_config.max_lead_m == 0.9 * min(MAX_LEAD_M, joint_ik.max_lead_m) and
          joint_config.max_lead_rad == 0.9 * min(MAX_LEAD_RAD, joint_ik.max_lead_rad))
    cartesian_config = resolve_spacemouse_config(parse_args([]), "cartesian")
    check("cartesian defaults derive shared tuning", cartesian_config.action_size == 0.015 and
          cartesian_config.max_lead_m == 0.10 and cartesian_config.max_lead_rad == 0.50)
    overrides = parse_args(["--action-size", "0.012", "--max-lead", "0.04"])
    check("command line overrides apply in both spaces", all(
        resolve_spacemouse_config(overrides, space, joint_ik if space == "joint" else None).action_size == 0.012 and
        resolve_spacemouse_config(overrides, space, joint_ik if space == "joint" else None).max_lead_m == 0.04
        for space in ("joint", "cartesian")))
    check("recorder keeps reset key ownership", not cartesian_config.keyboard_reset)

    specs = {"agentview": {"width": 32, "height": 24},
             "inhand": {"width": 20, "height": 12}}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "dataset"
        cfg = RecorderConfig(root, record_fps=15.0, crf=18, depth=True, min_frames=16,
                             task="synthetic test", notes="no hardware", control_space="joint",
                             control_rate=50.0,
                             teleop=teleop_metadata(joint_config, "joint", 50.0))
        writer = EpisodeWriter(cfg, 7, specs).start()
        original = [sample(i, stale=(i == 5)) for i in range(40)]
        for item in original:
            writer.write(item)
        episode_path = writer.finalize()
        episode = load_episode(episode_path)

        print("=== recorder round trip ===")
        check("all state rows round-trip", episode.n_frames == 40)
        check("float CLI-shaped fps writes an episode", episode_path.exists())
        check("all state columns round-trip", all((
            np.allclose(episode.table.timestamp, [x.observation.timestamp for x in original]),
            np.allclose(episode.table.t_rel, np.arange(40) / 15, atol=1e-6),
            np.allclose(np.asarray(episode.table.joint_position.tolist()),
                        np.asarray([x.observation.joint_values for x in original]), atol=1e-6),
            np.allclose(np.asarray(episode.table.ee_position.tolist()),
                        np.asarray([x.observation.ee_position for x in original]), atol=1e-6),
            np.allclose(np.asarray(episode.table.ee_quat_xyzw.tolist()),
                        np.asarray([x.observation.ee_quat_xyzw for x in original]), atol=1e-6),
            np.allclose(np.asarray(episode.table.fingertip_position.tolist()),
                        np.asarray([x.observation.fingertip_pose.position for x in original]), atol=1e-6),
            np.allclose(np.asarray(episode.table.fingertip_quat_xyzw.tolist()),
                        np.asarray([x.observation.fingertip_pose.orientation.as_quat() for x in original]), atol=1e-6),
            np.allclose(episode.table.gripper_value, [x.observation.gripper_value for x in original]),
            np.allclose(episode.table.gripper_width_mm, [x.observation.gripper_width_mm for x in original]),
            np.allclose(np.asarray(episode.table.ft_wrench.tolist()),
                        np.asarray([x.observation.ft_wrench for x in original]), atol=1e-6),
        )))
        check("action joint nulls survive", episode.table.action_joint_position.isna().sum() == 20)
        expected_grasp = np.asarray([int(x.action.gripper_latched < 0.5) for x in original], dtype=np.int8)
        check("closed gripper command has grasp polarity 1, open has 0",
              expected_grasp[19] == 0 and expected_grasp[20] == 1)
        check("grasp transition follows the latched command",
              np.array_equal(np.asarray(episode.table.action_grasp, dtype=np.int8), expected_grasp))
        check("held action gripper is populated on every row",
              episode.table.action_gripper.notna().all() and np.allclose(
                  episode.table.action_gripper, [x.action.gripper_latched for x in original]))
        check("motion and applied delta round-trip as six-vectors",
              all(len(value) == 6 for value in episode.table.action_delta) and
              all(len(value) == 6 for value in episode.table.spacemouse_motion) and
              np.allclose(np.asarray(episode.table.action_delta.tolist()),
                          np.asarray([x.action.delta for x in original])) and
              np.allclose(np.asarray(episode.table.spacemouse_motion.tolist()),
                          np.asarray([x.action.motion for x in original])))
        check("action timestamps are present and finite",
              episode.table.action_timestamp.notna().all() and
              np.isfinite(np.asarray(episode.table.action_timestamp)).all())
        check("one video frame per row", all(sum(1 for _ in episode.frames(camera["name"])) == episode.n_frames
                                                for camera in episode.meta["cameras"]))
        check("stale frame is marked", bool(episode.table.loc[5, "agentview_stale"]))
        check("stale frame still has alignment", episode.frame("agentview", 5).shape == (24, 32, 3))
        listed = list(episode.frames("inhand"))
        check("random access matches sequential decode", np.array_equal(episode.frame("inhand", 17), listed[17]))

        print("\n=== metadata ===")
        meta = json.loads((episode_path / "meta.json").read_text())
        manifest = json.loads((root / "dataset.json").read_text())
        check("meta has documented keys", {"schema_version", "episode_id", "n_frames", "cameras",
                                             "gripper_stroke_mm", "cameras_yaml", "gripper_geometry_yaml"} <= set(meta))
        teleop_keys = {"action_size", "rotation_size", "max_lead_m", "max_lead_rad",
                       "brake_on_release", "brake_release_ticks", "deadzone",
                       "control_space", "control_rate", "config_path", "config_mtime"}
        check("meta records resolved teleop tuning", teleop_keys <= set(meta["teleop"]) and
              meta["teleop"] == teleop_metadata(joint_config, "joint", 50.0))
        check("dataset manifest indexes episode", manifest["episodes"] == ["episode_000007"])
        check("dataset loads listed episodes", len(load_dataset(root).episodes) == 1)

        print("\n=== short episode cleanup ===")
        short = EpisodeWriter(cfg, 8, specs).start()
        for index in range(3): short.write(sample(index))
        check("short episode is deleted", short.finalize() is None and not (root / "episode_000008").exists())

        print("\n=== legacy export ===")
        legacy_root = Path(temp) / "legacy"
        exported = export_dataset(root, legacy_root)
        state = exported[0] / "state"
        pose = np.load(state / "pose_wrt_world.npy")
        joint = np.load(state / "joint_states.npy")
        wrench = np.load(state / "ft_sensor.npy")
        grasp = np.load(state / "grasp.npy")
        qpos = np.load(state / "gripper_qpos.npy")
        check("legacy documented state files", all((state / name).exists() for name in
              ("grasp.npy", "pose_wrt_world.npy", "joint_states.npy", "ft_sensor.npy", "gripper_qpos.npy")))
        check("legacy dtypes and shapes", pose.dtype == np.float32 and pose.shape == (40, 7) and
              joint.dtype == np.float32 and joint.shape == (40, 7) and wrench.shape == (40, 6) and
              grasp.dtype == np.int32 and grasp.shape == (40,) and qpos.dtype == np.float32 and qpos.shape == (40,))
        check("legacy grasp is the recorded action grasp", grasp.dtype == np.int32 and
              np.array_equal(grasp, np.asarray(episode.table.action_grasp, dtype=np.int32)))
        check("legacy RGB count matches episode", len(list((exported[0] / "cam1" / "rgb").glob("*.png"))) == 40 and
              len(list((exported[0] / "cam2" / "rgb").glob("*.png"))) == 40)

        print("\n=== recorder shutdown ===")
        stop_cfg = RecorderConfig(Path(temp) / "stop_dataset", queue_size=4, min_frames=1)
        session = RecordingSession(stop_cfg, {})
        session.start([])
        keep_submitting = threading.Event()
        keep_submitting.set()
        producer_started = threading.Event()

        def produce():
            producer_started.set()
            while keep_submitting.is_set():
                session.submit(sample(0))
                time.sleep(0.001)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        check("shutdown producer starts", producer_started.wait(timeout=1.0))
        time.sleep(0.02)
        stop_finished = threading.Event()

        def stop_session():
            session.stop()
            stop_finished.set()

        stopper = threading.Thread(target=stop_session, daemon=True)
        stopper.start()
        stopped = stop_finished.wait(timeout=3.0)
        keep_submitting.clear()
        producer.join(timeout=1.0)
        if stopped:
            stopper.join(timeout=1.0)
        check("stop returns while a producer is submitting", stopped)

    print("\nAll recorder tests passed.")


if __name__ == "__main__":
    main()
