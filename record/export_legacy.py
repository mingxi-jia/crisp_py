"""Export compact HTTP-recorded episodes into diffusion-policy's old layout."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from record.dataset import load_dataset, load_episode


def _timestamp_filename(timestamp: float) -> str:
    """Match recorder_old.py: ``<sec:010d>_<nanosec:09d>.png``."""
    sec = math.floor(float(timestamp))
    nanosec = int(round((float(timestamp) - sec) * 1_000_000_000))
    if nanosec == 1_000_000_000:
        sec, nanosec = sec + 1, 0
    return f"{sec:010d}_{nanosec:09d}.png"


def export_episode(episode, out_dir) -> Path:
    """Export one loaded :class:`record.dataset.Episode`."""
    out = Path(out_dir) / episode.path.name
    for cam_number, camera in enumerate(("agentview", "inhand"), 1):
        rgb_dir = out / f"cam{cam_number}" / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        for timestamp, image in zip(episode.table["timestamp"], episode.frames(camera)):
            # OpenCV writes BGR arrays; decoder gives RGB.
            cv2.imwrite(str(rgb_dir / _timestamp_filename(timestamp)),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    state = out / "state"
    state.mkdir(parents=True, exist_ok=True)
    table = episode.table
    gripper = np.asarray(table["gripper_value"].fillna(1.0), dtype=np.float32)
    np.save(state / "grasp.npy", np.asarray(table["action_grasp"], dtype=np.int32))
    np.save(state / "pose_wrt_world.npy", np.asarray(
        [np.concatenate((p, q)) for p, q in zip(table["ee_position"], table["ee_quat_xyzw"])],
        dtype=np.float32))
    np.save(state / "joint_states.npy", np.asarray(table["joint_position"].tolist(), dtype=np.float32))
    np.save(state / "ft_sensor.npy", np.asarray(table["ft_wrench"].tolist(), dtype=np.float32))
    np.save(state / "gripper_qpos.npy", gripper.astype(np.float32))
    return out


def export_dataset(dataset_dir, out_dir, episodes=None) -> list[Path]:
    dataset = load_dataset(dataset_dir)
    wanted = set(episodes or [])
    result = []
    for episode in dataset.episodes:
        if wanted and episode.path.name not in wanted and str(episode.meta["episode_id"]) not in wanted:
            continue
        result.append(export_episode(episode, out_dir))
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset_dir")
    p.add_argument("out_dir")
    p.add_argument("--episodes", nargs="*", help="episode directory names or numeric ids")
    return p.parse_args()


def main():
    args = parse_args()
    exported = export_dataset(args.dataset_dir, args.out_dir, args.episodes)
    print(f"Exported {len(exported)} episode(s).")


if __name__ == "__main__":
    main()
