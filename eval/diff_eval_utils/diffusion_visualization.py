"""Visualization utilities for point clouds and actions."""

import numpy as np
import open3d as o3d
from diffusion_policy.model.common.rotation_transformer import RotationTransformer


def visualize_pcd(pcd: np.ndarray, robot_pcd=None):
    """Visualize point cloud using Open3D.

    Args:
        pcd: Point cloud as a numpy array of shape (N, 6) where the first 3 columns are XYZ and the next 3 are RGB.
        robot_pcd: Optional robot point cloud geometry to visualize alongside the scene.
    """
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    pcd_o3d.colors = o3d.utility.Vector3dVector(pcd[:, 3:])

    geometries = [pcd_o3d]
    if robot_pcd is not None:
        geometries.append(robot_pcd)

    o3d.visualization.draw_geometries(geometries)


def visualize_robot_pcd(raw_pcd, robot_seg, joint_state):
    """Generate robot mesh point cloud for visualization.

    Args:
        raw_pcd: Raw point cloud
        robot_seg: Robot segmentation object with URDF
        joint_state: Current joint state

    Returns:
        Open3D point cloud geometry of robot
    """
    joint_names = sorted([j.name for j in robot_seg.robot_urdf.actuated_joints])
    joint_angles = dict(zip(joint_names, joint_state))

    # Sample robot points
    robot_mesh_dict = robot_seg.robot_urdf.visual_trimesh_fk(cfg=joint_angles)
    sampled_points = []
    for mesh, pose in robot_mesh_dict.items():
        transformed = mesh.copy()
        transformed.apply_transform(pose)
        transformed.apply_transform(robot_seg.T_world_urdf)
        sampled_points.append(transformed.sample(2 * 2500))

    robot_points = np.vstack(sampled_points)
    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.points = o3d.utility.Vector3dVector(robot_points)
    robot_pcd.paint_uniform_color([0.75, 0.75, 0.75])  # Grey color

    return robot_pcd


def visualize_pcd_and_actions(pcd, actions, robot_pcd=None):
    """Visualize point cloud with action trajectories.

    Args:
        pcd: Point cloud array
        actions: Array of actions to visualize
        robot_pcd: Optional robot point cloud
    """
    action_frames = []
    for action in actions:
        x, y, z = action[:3]
        rot6d = action[3:9]
        rotation_transformer = RotationTransformer(from_rep='rotation_6d', to_rep='matrix')
        rotmat = rotation_transformer.forward(rot6d.reshape(1, 6))

        # Create coordinate frame at action pose
        action_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        action_transform = np.eye(4)
        action_transform[:3, :3] = rotmat
        action_transform[:3, 3] = np.array([x, y, z])
        action_frame.transform(action_transform)
        action_frames.append(action_frame)

    # Visualize filtered pcd with robot model and EEF frame
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    pcd_o3d.colors = o3d.utility.Vector3dVector(pcd[:, 3:])

    if robot_pcd is not None:
        o3d.visualization.draw_geometries([pcd_o3d, robot_pcd] + action_frames)
    else:
        o3d.visualization.draw_geometries([pcd_o3d] + action_frames)
