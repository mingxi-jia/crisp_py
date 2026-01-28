#!/usr/bin/env python3
"""
Visualize HDF5 robomimic dataset using rerun.io.
Shows point clouds and hand trajectory over time.
"""

import argparse
import h5py
import numpy as np
import rerun as rr
from pathlib import Path
from scipy.spatial.transform import Rotation
import sys


def visualize_episode(f, episode_key, timeline="frame", modalities=None):
    """Visualize a single episode from the HDF5 file.

    Args:
        f: HDF5 file handle
        episode_key: Key of the episode (e.g., "demo_0")
        timeline: Timeline name for rerun
        modalities: List of modalities to visualize. If None, visualize all.
                   Options: pcd, pcd_render, robot0_eye_in_hand_image, robot_eef, action_pose, robot_state
    """
    ep_grp = f[f"data/{episode_key}"]

    # Get episode data
    actions = np.array(ep_grp["actions"])
    num_samples = ep_grp.attrs["num_samples"]

    print(f"[*] {episode_key}: {num_samples} samples")

    # Get observation keys
    obs_keys = list(ep_grp["obs"].keys())
    print(f"    Observation keys: {obs_keys}")

    # Load 'pcd' point cloud
    obs_pcd = None
    if "pcd" in obs_keys:
        obs_pcd = np.array(ep_grp["obs/pcd"])
        print(f"    Point cloud 'pcd' shape: {obs_pcd.shape}")

    # Load 'pcd_render' or 'render_pcd' point cloud
    obs_pcd_render = None
    if "pcd_render" in obs_keys:
        obs_pcd_render = np.array(ep_grp["obs/pcd_render"])
        print(f"    Point cloud 'pcd_render' shape: {obs_pcd_render.shape}")
    elif "render_pcd" in obs_keys:
        obs_pcd_render = np.array(ep_grp["obs/render_pcd"])
        print(f"    Point cloud 'render_pcd' shape: {obs_pcd_render.shape}")

    # Load robot0_eye_in_hand_image
    robot0_eye_in_hand_image = None
    if "robot0_eye_in_hand_image" in obs_keys:
        robot0_eye_in_hand_image = np.array(ep_grp["obs/robot0_eye_in_hand_image"])
        print(f"    robot0_eye_in_hand_image shape: {robot0_eye_in_hand_image.shape}")

    # Load robot end-effector pose from observations if available
    eef_pos = None
    eef_quat = None
    if "robot0_eef_pos" in obs_keys:
        eef_pos = np.array(ep_grp["obs/robot0_eef_pos"])
    if "robot0_eef_quat" in obs_keys:
        eef_quat = np.array(ep_grp["obs/robot0_eef_quat"])

    # Helper function to check if modality should be visualized
    def should_visualize(modality_name):
        return modalities is None or modality_name in modalities

    # Visualize each frame
    for i in range(num_samples):
        rr.set_time(timeline, sequence=i)

        # Log 'pcd' point cloud if available
        if obs_pcd is not None and should_visualize("pcd"):
            pcd_frame = obs_pcd[i]  # Shape: (N, 3) or (N, 6) with colors

            if pcd_frame.shape[1] >= 6:
                positions = pcd_frame[:, :3]
                colors = pcd_frame[:, 3:6]
                if colors.max() > 1.0:
                    colors = colors / 255.0
                rr.log(
                    f"{episode_key}/pcd",
                    rr.Points3D(positions, colors=colors, radii=0.005)
                )
            else:
                positions = pcd_frame[:, :3]
                rr.log(
                    f"{episode_key}/pcd",
                    rr.Points3D(positions, radii=0.005)
                )

        # Log 'pcd_render' point cloud if available
        if obs_pcd_render is not None and should_visualize("pcd_render"):
            pcd_render_frame = obs_pcd_render[i]

            if pcd_render_frame.shape[1] >= 6:
                positions = pcd_render_frame[:, :3]
                colors = pcd_render_frame[:, 3:6]
                if colors.max() > 1.0:
                    colors = colors / 255.0
                rr.log(
                    f"{episode_key}/pcd_render",
                    rr.Points3D(positions, colors=colors, radii=0.005)
                )
            else:
                positions = pcd_render_frame[:, :3]
                rr.log(
                    f"{episode_key}/pcd_render",
                    rr.Points3D(positions, radii=0.005)
                )

        # Log robot0_eye_in_hand_image if available
        if robot0_eye_in_hand_image is not None and should_visualize("robot0_eye_in_hand_image"):
            img = robot0_eye_in_hand_image[i]
            rr.log(
                f"{episode_key}/robot0_eye_in_hand_image",
                rr.Image(img)
            )

        # Log robot end-effector pose from observations
        if eef_pos is not None and eef_quat is not None and should_visualize("robot_eef"):
            t = eef_pos[i]
            q = eef_quat[i]  # [qx, qy, qz, qw] or [qw, qx, qy, qz] depending on convention

            # Create rotation matrix from quaternion (scipy expects xyzw)
            R_mat = Rotation.from_quat(q).as_matrix()

            # Log as transform3d
            rr.log(
                f"{episode_key}/robot_eef",
                rr.Transform3D(
                    translation=t,
                    mat3x3=R_mat,
                )
            )

            # Log the trajectory as a line path
            if i > 0:
                rr.log(
                    f"{episode_key}/robot_trajectory",
                    rr.LineStrips3D([eef_pos[:i+1]], colors=[0, 255, 0])
                )

        # Log hand pose from actions if actions contain pose data
        # Assuming actions are [tx, ty, tz, qx, qy, qz, qw]
        if actions.shape[1] >= 7 and should_visualize("action_pose"):
            action = actions[i]
            t = action[:3]
            q = action[3:7]  # [qx, qy, qz, qw]

            # Create rotation matrix from quaternion
            R_mat = Rotation.from_quat(q).as_matrix()

            # Log as transform3d
            rr.log(
                f"{episode_key}/action_pose",
                rr.Transform3D(
                    translation=t,
                    mat3x3=R_mat,
                )
            )

            # Also log the action trajectory as a line path
            if i > 0:
                rr.log(
                    f"{episode_key}/action_trajectory",
                    rr.LineStrips3D([actions[:i+1, :3]], colors=[255, 128, 0])
                )

        # Log states if available (optional)
        if "states" in ep_grp and should_visualize("robot_state"):
            states = np.array(ep_grp["states"])
            if states.shape[1] >= 7:
                state = states[i]
                state_t = state[:3]
                state_q = state[3:7]

                state_R = Rotation.from_quat(state_q).as_matrix()

                rr.log(
                    f"{episode_key}/robot_state",
                    rr.Transform3D(
                        translation=state_t,
                        mat3x3=state_R,
                    )
                )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize robomimic HDF5 dataset with rerun.io"
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        help="Path to the HDF5 file"
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Episodes to visualize (e.g., '0,1,2' or '0-5'). Default: all"
    )
    parser.add_argument(
        "--app-id",
        type=str,
        default="robomimic_viz",
        help="Rerun application ID (default: robomimic_viz)"
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Comma-separated list of modalities to visualize. Options: pcd, pcd_render, robot0_eye_in_hand_image, robot_eef, action_pose, robot_state. Default: all"
    )

    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_path)
    if not hdf5_path.exists():
        print(f"Error: HDF5 file not found: {hdf5_path}")
        sys.exit(1)

    # Parse modalities
    modalities = None
    if args.modalities is not None:
        modalities = [m.strip() for m in args.modalities.split(",")]
        print(f"Selected modalities: {modalities}")

    # Initialize rerun
    rr.init(args.app_id, spawn=True)

    # Open HDF5 file
    print(f"Opening HDF5 file: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        # Get all episode keys
        all_episodes = sorted([k for k in f["data"].keys() if k.startswith("demo_")])

        # Parse episode selection
        if args.episodes is None:
            episodes_to_viz = all_episodes
        else:
            episodes_to_viz = []
            for part in args.episodes.split(","):
                if "-" in part:
                    # Range
                    start, end = part.split("-")
                    for i in range(int(start), int(end) + 1):
                        episodes_to_viz.append(f"demo_{i}")
                else:
                    # Single episode
                    episodes_to_viz.append(f"demo_{part}")

        print(f"Total episodes: {len(all_episodes)}")
        print(f"Visualizing {len(episodes_to_viz)} episode(s)")
        print()

        # Visualize the origin (world coordinate frame)
        axis_length = 0.1
        rr.log(
            "world/origin",
            rr.Arrows3D(
                origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                vectors=[
                    [axis_length, 0, 0],  # X-axis (red)
                    [0, axis_length, 0],  # Y-axis (green)
                    [0, 0, axis_length],  # Z-axis (blue)
                ],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                radii=0.005,
            ),
        )

        # Visualize each episode
        for ep_key in episodes_to_viz:
            if ep_key not in all_episodes:
                print(f"Warning: Episode {ep_key} not found, skipping")
                continue
            visualize_episode(f, ep_key, modalities=modalities)

        print()
        print("Visualization complete!")
        print("Open the Rerun viewer to see the results.")
        print("Use the timeline slider to scrub through frames.")


if __name__ == "__main__":
    main()
