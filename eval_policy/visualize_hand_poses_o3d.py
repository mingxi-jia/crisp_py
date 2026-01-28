"""Visualize hand poses trajectory in Open3D before robot execution."""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os
import h5py


def load_scene_pcd(pose_folder, pcd_index=0, use_no_hand=True):
    """Load a scene point cloud from the episode folder.

    Args:
        pose_folder: Path to the episode folder
        pcd_index: Which frame's point cloud to load (default: 0 for first frame)
        use_no_hand: If True, use pcd_no_hand folder (without hand visible)

    Returns:
        Open3D PointCloud object
    """
    pcd_folder = "pcd_no_hand" if use_no_hand else "pcd"
    pcd_path = os.path.join(pose_folder, pcd_folder, f"{pcd_index:06d}.npy")

    if not os.path.exists(pcd_path):
        print(f"Warning: Point cloud not found at {pcd_path}")
        return None

    pcd_data = np.load(pcd_path)
    # Format: (N, 6) with XYZ + RGB
    points = pcd_data[:, :3]
    colors = pcd_data[:, 3:6]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def create_coordinate_frame(pose, scale=0.05):
    """Create a coordinate frame mesh at the given pose.

    Args:
        pose: 7D array [x, y, z, qx, qy, qz, qw]
        scale: Size of the coordinate frame axes

    Returns:
        Open3D TriangleMesh coordinate frame
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)

    # Create transformation matrix from pose
    position = pose[:3]
    quat = pose[3:7]  # [qx, qy, qz, qw]
    rotation = R.from_quat(quat).as_matrix()

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position

    frame.transform(transform)
    return frame


def create_trajectory_line(poses, color=[0, 0, 1]):
    """Create a line set showing the trajectory path.

    Args:
        poses: Nx7 array of poses
        color: RGB color for the line

    Returns:
        Open3D LineSet
    """
    points = poses[:, :3]
    lines = [[i, i+1] for i in range(len(points)-1)]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


def create_grasp_markers(poses, grasp_values, open_color=[0, 1, 0], closed_color=[1, 0, 0]):
    """Create spheres at grasp change points.

    Args:
        poses: Nx7 array of poses
        grasp_values: N array of grasp values (0=open, 1=closed)
        open_color: Color for open gripper
        closed_color: Color for closed gripper

    Returns:
        List of sphere meshes
    """
    spheres = []
    grasp_rounded = np.round(np.clip(grasp_values, 0, 1))

    # Find grasp change points
    for i in range(1, len(grasp_rounded)):
        if grasp_rounded[i] != grasp_rounded[i-1]:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere.translate(poses[i, :3])

            if grasp_rounded[i] == 1:
                sphere.paint_uniform_color(closed_color)  # Closing
            else:
                sphere.paint_uniform_color(open_color)    # Opening

            spheres.append(sphere)

    return spheres


def load_from_hdf5(hdf5_path, demo_key="demo_0"):
    """Load hand poses and grasp data from HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file
        demo_key: Key for the demo in the HDF5 file (default: "demo_0")

    Returns:
        hand_poses: Nx7 array of poses [x, y, z, qx, qy, qz, qw]
        hand_grasp: N array of grasp values
    """
    dataset = h5py.File(hdf5_path, "r")
    data = dataset['data'][demo_key]
    hand_pos = data['obs']['robot0_eef_pos'][:]
    hand_quat = data['obs']['robot0_eef_quat'][:]
    hand_grasp = data['obs']['robot0_gripper_qpos'][:, 0]
    hand_poses = np.hstack((hand_pos, hand_quat)).astype(float)
    dataset.close()
    return hand_poses, hand_grasp


def list_demos_in_hdf5(hdf5_path):
    """List all demo keys in an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file

    Returns:
        List of demo keys
    """
    dataset = h5py.File(hdf5_path, "r")
    demos = list(dataset['data'].keys())
    dataset.close()
    return sorted(demos)


def visualize_hand_poses(hdf5_path, demo_key="demo_0", frame_step=10, pcd_folder=None, pcd_index=0):
    """Visualize hand poses from an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file
        demo_key: Key for the demo in the HDF5 file (default: "demo_0")
        frame_step: Show coordinate frame every N poses (to reduce clutter)
        pcd_folder: Optional path to folder containing point clouds
        pcd_index: Which frame's point cloud to display (default: 0)
    """
    # Load data from HDF5
    hand_poses, hand_grasp = load_from_hdf5(hdf5_path, demo_key)

    print(f"Loaded {len(hand_poses)} poses from {hdf5_path} [{demo_key}]")
    print(f"Position range:")
    print(f"  X: [{hand_poses[:, 0].min():.3f}, {hand_poses[:, 0].max():.3f}]")
    print(f"  Y: [{hand_poses[:, 1].min():.3f}, {hand_poses[:, 1].max():.3f}]")
    print(f"  Z: [{hand_poses[:, 2].min():.3f}, {hand_poses[:, 2].max():.3f}]")

    geometries = []

    # Load and add scene point cloud if folder provided
    if pcd_folder is not None:
        scene_pcd = load_scene_pcd(pcd_folder, pcd_index=pcd_index, use_no_hand=True)
        if scene_pcd is not None:
            print(f"Loaded scene point cloud with {len(scene_pcd.points)} points")
            geometries.append(scene_pcd)

    # Add world coordinate frame at origin
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(world_frame)

    # Add trajectory line
    trajectory = create_trajectory_line(hand_poses, color=[0.2, 0.2, 0.8])
    geometries.append(trajectory)

    # Add coordinate frames at regular intervals
    for i in range(0, len(hand_poses), frame_step):
        frame = create_coordinate_frame(hand_poses[i], scale=0.03)
        geometries.append(frame)

    # Add start point (green sphere)
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    start_sphere.translate(hand_poses[0, :3])
    start_sphere.paint_uniform_color([0, 0.8, 0])
    geometries.append(start_sphere)

    # Add end point (red sphere)
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    end_sphere.translate(hand_poses[-1, :3])
    end_sphere.paint_uniform_color([0.8, 0, 0])
    geometries.append(end_sphere)

    # Add grasp change markers
    grasp_markers = create_grasp_markers(hand_poses, hand_grasp)
    geometries.extend(grasp_markers)
    print(f"Found {len(grasp_markers)} grasp change points")

    # Create center point for reference (from original script)
    center = np.array([0.4, 0.0, 0.4])
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
    center_sphere.translate(center)
    center_sphere.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(center_sphere)

    # Visualize
    print("\nVisualization Legend:")
    print("  - Large frame at origin: World coordinate frame")
    print("  - Blue line: Trajectory path")
    print("  - Small frames: End-effector orientations (every {} poses)".format(frame_step))
    print("  - Green sphere: Start position")
    print("  - Red sphere: End position")
    print("  - Green small spheres: Gripper opening")
    print("  - Red small spheres: Gripper closing")
    print("  - Gray sphere: Center point [0.4, 0.0, 0.4]")
    print("\nPress Q to close the visualization window.")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Hand Poses Trajectory Preview",
        width=1280,
        height=720,
        left=50,
        top=50
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize hand poses in Open3D from HDF5 file")
    parser.add_argument(
        "--hdf5_path",
        type=str,
        default="/media/mingxi/T7/XEMB_Experiment/coffee_prep/replay_hand_test/test_2.hdf5",
        help="Path to the HDF5 file containing robot trajectory data"
    )
    parser.add_argument(
        "--demo_key",
        type=str,
        default="demo_0",
        help="Demo key in the HDF5 file (default: demo_0)"
    )
    parser.add_argument(
        "--list_demos",
        action="store_true",
        help="List all demos in the HDF5 file and exit"
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=10,
        help="Show coordinate frame every N poses"
    )
    parser.add_argument(
        "--pcd_folder",
        type=str,
        default=None,
        help="Optional path to folder containing point clouds (pcd_no_hand/)"
    )
    parser.add_argument(
        "--pcd_index",
        type=int,
        default=0,
        help="Which frame's point cloud to display"
    )

    args = parser.parse_args()

    if args.list_demos:
        demos = list_demos_in_hdf5(args.hdf5_path)
        print(f"Demos in {args.hdf5_path}:")
        for demo in demos:
            print(f"  - {demo}")
    else:
        visualize_hand_poses(args.hdf5_path, args.demo_key, args.frame_step, args.pcd_folder, args.pcd_index)
