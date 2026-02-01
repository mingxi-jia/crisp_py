#!/usr/bin/env python3
"""Visualize predicted policy trajectories from collected workspace observations.

This script loads observations collected from collect_workspace_obs.py, runs policy
inference on each observation, and visualizes the predicted trajectories in 3D using
Open3D.
                                             
Example usage:
    python visualize_policy_trajectories.py \
        --input workspace_obs_data.pkl \
        --show_every 2
"""

import argparse
import pickle
import time
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

# Add external dependencies
sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy')

# Import policy client
from diff_eval_utils.diffusion_clients import PolicyClient
from diff_eval_utils.diffusion_transforms import ten_d_action_to_pose, convert_action_from_fingertip_to_gripper


class PolicyTrajectoryVisualizer:
    """Visualizes predicted policy trajectories in 3D."""

    def __init__(self, policy_client, hand=False):
        """Initialize the visualizer.

        Args:
            policy_client: PolicyClient instance
        """
        self.policy_client = policy_client
        self.hand = hand

    def predict_trajectory(self, observation):
        """Predict trajectory from observation.

        Args:
            observation: Observation dictionary

        Returns:
            Array of shape (N, 7) containing [x, y, z, qx, qy, qz, qw] for each step
        """
        # Get action predictions
        if self.hand:
            observation_togo = {
                'render_pcd': observation['render_pcd'],
                'robot0_eef_pos': observation['robot0_eef_pos'],
                'robot0_eef_quat': observation['robot0_eef_quat'],
                'robot0_gripper_qpos': observation['robot0_gripper_qpos'],
            }
        else:
            observation_togo = observation
        actions = self.policy_client.predict_action(observation_togo)

        trajectory = []
        for action in actions:
            # Convert from fingertip to gripper frame
            pose, grasp = ten_d_action_to_pose(action)
            position, gripper_rotation = convert_action_from_fingertip_to_gripper(pose)

            # Use the gripper rotation from the conversion
            orientation_quat = gripper_rotation.as_quat()  # (x, y, z, w)

            # Combine into pose
            pose = np.concatenate([position, orientation_quat])
            trajectory.append(pose)

        return np.array(trajectory)

    def create_trajectory_geometries(self, start_position, start_orientation,
                                     trajectory, color=[1, 0, 0]):
        """Create Open3D geometries for visualizing a trajectory.

        Args:
            start_position: Starting position (x, y, z)
            start_orientation: Starting orientation as quaternion (x, y, z, w)
            trajectory: Array of shape (N, 7) with poses [x, y, z, qx, qy, qz, qw]
            color: RGB color for trajectory

        Returns:
            Dictionary with 'static' and 'orientation_frames' geometry lists
        """
        static_geometries = []
        orientation_frames = []

        # Create start point sphere (smaller)
        # start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        # start_sphere.translate(start_position)
        # start_sphere.paint_uniform_color([0, 1, 0])  # Green for start
        # static_geometries.append(start_sphere)

        # Create coordinate frame at start position
        start_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.015, origin=[0, 0, 0]
        )
        # Apply rotation to coordinate frame
        rot_matrix = R.from_quat(start_orientation).as_matrix()
        start_frame.rotate(rot_matrix, center=[0, 0, 0])
        start_frame.translate(start_position)
        orientation_frames.append(start_frame)

        # Create trajectory line (using cylinders for thickness)
        points = trajectory[:, :3]  # Extract xyz positions

        # Create thick trajectory using cylinders between points
        for i in range(len(points) - 1):
            # Create cylinder between consecutive points
            start = points[i]
            end = points[i + 1]

            # Calculate cylinder parameters
            direction = end - start
            length = np.linalg.norm(direction)

            if length > 0:
                # Create cylinder
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=0.001,  # 1mm radius
                    height=length
                )
                cylinder.paint_uniform_color(color)

                # Align cylinder with direction
                cylinder_direction = np.array([0, 0, 1])  # Default cylinder axis
                rotation_axis = np.cross(cylinder_direction, direction / length)

                if np.linalg.norm(rotation_axis) > 1e-6:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    angle = np.arccos(np.clip(np.dot(cylinder_direction, direction / length), -1, 1))
                    rot_matrix = R.from_rotvec(angle * rotation_axis).as_matrix()
                    cylinder.rotate(rot_matrix, center=[0, 0, 0])

                # Position cylinder
                cylinder.translate(start + direction / 2)
                static_geometries.append(cylinder)

        # Create coordinate frames at each predicted timestep
        for pose in trajectory:
            position = pose[:3]
            orientation_quat = pose[3:7]  # (qx, qy, qz, qw)

            # Create coordinate frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.015, origin=[0, 0, 0]
            )
            # Apply rotation
            rot_matrix = R.from_quat(orientation_quat).as_matrix()
            frame.rotate(rot_matrix, center=[0, 0, 0])
            frame.translate(position)
            orientation_frames.append(frame)

        # Create end point sphere (smaller)
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        end_sphere.translate(points[-1])
        end_sphere.paint_uniform_color([1, 0, 0])  # Red for end
        static_geometries.append(end_sphere)

        return {
            'static': static_geometries,
            'orientation_frames': orientation_frames
        }

    def create_grid_visualization(self, grid_positions, color=[0.5, 0.5, 0.5]):
        """Create visualization of grid points.

        Args:
            grid_positions: List of (x, y, z) tuples
            color: RGB color for grid points

        Returns:
            Open3D PointCloud
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(grid_positions))
        pcd.paint_uniform_color(color)
        return pcd

    def visualize_trajectories(self, collected_data, show_every=1,
                               trajectory_colors=None):
        """Visualize all predicted trajectories.

        Args:
            collected_data: List of data points from collection
            show_every: Only show every Nth trajectory (for clarity)
            trajectory_colors: Optional list of colors for trajectories
        """
        print("\nRunning policy inference on collected observations...")
        print("=" * 60)

        static_geometries = []
        orientation_frames = []

        # Collect grid positions
        grid_positions = [d['position'] for d in collected_data]

        # Add grid points
        grid_pcd = self.create_grid_visualization(grid_positions)
        static_geometries.append(grid_pcd)

        # Add point cloud from observation with highest Z, lowest X, lowest Y
        if len(collected_data) > 0:
            # Find the observation at the position with highest Z, then lowest X, then lowest Y
            positions = np.array([d['position'] for d in collected_data])

            # Sort by: highest Z (descending), lowest X (ascending), lowest Y (ascending)
            # Create a sort key: (-z, x, y)
            sort_keys = np.column_stack((-positions[:, 2], positions[:, 0], positions[:, 1]))
            sorted_indices = np.lexsort((sort_keys[:, 2], sort_keys[:, 1], sort_keys[:, 0]))
            best_idx = sorted_indices[0]

            best_position = collected_data[best_idx]['position']
            best_obs = collected_data[best_idx]['observation']

            print(f"✓ Selected observation at position ({best_position[0]:.3f}, {best_position[1]:.3f}, {best_position[2]:.3f})")

            if 'pcd' in best_obs:
                point_cloud_data = best_obs['pcd']  # Shape: (N, 6) - XYZ + RGB
                obs_pcd = o3d.geometry.PointCloud()
                obs_pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])  # XYZ only
                obs_pcd.colors = o3d.utility.Vector3dVector(point_cloud_data[:, 3:6])  # RGB
                static_geometries.append(obs_pcd)
                print(f"✓ Added point cloud from selected observation ({len(point_cloud_data)} points)")
            else:
                print(f"⚠ Warning: No 'pcd' found in selected observation. Keys: {list(best_obs.keys())}")

        # Generate colors if not provided
        if trajectory_colors is None:
            # Use colormap for trajectories
            n_trajectories = len(collected_data[::show_every])
            cmap = plt.cm.get_cmap('rainbow')
            trajectory_colors = [cmap(i / n_trajectories)[:3]
                                for i in range(n_trajectories)]

        # Predict and visualize trajectories
        for idx, data_point in enumerate(collected_data[::show_every]):
            position = data_point['position']
            observation = data_point['observation']

            # Get actual eef pose from observation
            eef_pos = observation['robot0_eef_pos']

            print(f"[{idx+1}/{len(collected_data[::show_every])}] "
                  f"Processing position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
            print(f"  EEF pose: pos=({eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f})")

            try:
                # Predict trajectory
                trajectory = self.predict_trajectory(observation)

                if len(trajectory) == 0:
                    print(f"  ✗ No trajectory predicted")
                    continue

                # Use first trajectory point's orientation for start frame
                start_quat = trajectory[0, 3:7]  # Extract quat from first trajectory point

                # Create visualization geometries using trajectory orientations
                color = trajectory_colors[idx] if idx < len(trajectory_colors) else [1, 0, 0]
                traj_geoms = self.create_trajectory_geometries(
                    eef_pos, start_quat, trajectory, color=color
                )
                static_geometries.extend(traj_geoms['static'])
                orientation_frames.extend(traj_geoms['orientation_frames'])

                print(f"  ✓ Trajectory predicted ({len(trajectory)} steps)")

            except Exception as e:
                print(f"  ✗ Error predicting trajectory: {e}")
                continue

        print("\n" + "=" * 60)
        print("Launching Open3D visualization...")
        print("Controls:")
        print("  - Mouse: Rotate view")
        print("  - Scroll: Zoom")
        print("  - Shift + Mouse: Pan")
        print("  - R: Toggle orientation frames visibility")
        print("  - Q or ESC: Close")

        # Create visualizer with keyboard callback
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name='Policy Trajectory Visualization',
                         width=1600, height=1200)

        # Add static geometries
        for geom in static_geometries:
            vis.add_geometry(geom)

        # Add orientation frames (initially visible)
        for geom in orientation_frames:
            vis.add_geometry(geom)

        # Track visibility state
        frames_visible = [True]  # Use list to allow modification in nested function

        # Define keyboard callback for toggling orientation frames
        def toggle_orientation_frames(vis):
            frames_visible[0] = not frames_visible[0]
            for frame in orientation_frames:
                if frames_visible[0]:
                    vis.add_geometry(frame, reset_bounding_box=False)
                else:
                    vis.remove_geometry(frame, reset_bounding_box=False)
            print(f"Orientation frames: {'ON' if frames_visible[0] else 'OFF'}")
            return False

        # Register keyboard callback for 'R' key (ASCII 82)
        vis.register_key_callback(ord('R'), toggle_orientation_frames)

        # Set view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, -1, -0.5])
        ctr.set_lookat(np.mean(grid_positions, axis=0))
        ctr.set_up([0, 0, 1])

        # Run visualizer
        vis.run()
        vis.destroy_window()

        print("\nVisualization closed.")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize predicted policy trajectories from workspace observations'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input pickle file from collect_workspace_obs.py')
    parser.add_argument('--policy-port', type=int, default=5000,
                        help='Policy server port')
    parser.add_argument('--show_every', type=int, default=1,
                        help='Show every Nth trajectory (default: 1 = all)')
    parser.add_argument('--rotation_repr', type=str, default='rotation_6d',
                        choices=['rotation_6d', 'quaternion', 'euler'],
                        help='Rotation representation')
    parser.add_argument('--hand', action='store_true',
                        help='Use Hand policy, which changes the observation')

    args = parser.parse_args()

    # Load collected data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    print(f"Loading data from {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    collected_data = data['data']
    metadata = data['metadata']

    print(f"✓ Loaded {len(collected_data)} observations")
    print(f"  Workspace: X=[{metadata['x_range'][0]}, {metadata['x_range'][1]}], "
          f"Y=[{metadata['y_range'][0]}, {metadata['y_range'][1]}], "
          f"Z=[{metadata['z_range'][0]}, {metadata['z_range'][1]}]")
    print(f"  Grid spacing: {metadata['spacing']} m")

    # Initialize policy client
    print(f"\nConnecting to policy server at localhost:{args.policy_port}...")
    policy_client = PolicyClient(f"http://localhost:{args.policy_port}")
    time.sleep(1.0)  # Wait for connection

    # Create visualizer
    visualizer = PolicyTrajectoryVisualizer(
        policy_client=policy_client,
        hand=args.hand
    )

    # Visualize trajectories
    visualizer.visualize_trajectories(
        collected_data=collected_data,
        show_every=args.show_every
    )

    print("\nDone!")


if __name__ == '__main__':
    # Import matplotlib for colormap (lazy import to avoid display issues)
    import matplotlib.pyplot as plt
    main()
