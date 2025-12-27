"""Coordinate transformation utilities for diffusion policy."""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from crisp_py.robot import Pose
from PIL import Image

from .diffusion_constants import FINGER_HAND_OFFSET


def get_pose_from_robot(robot_pose: Pose) -> np.ndarray:
    """Convert robot pose to gripper pose with fingertip offset.

    Args:
        robot_pose: Robot end-effector pose

    Returns:
        7-element array [x, y, z, qx, qy, qz, qw] representing gripper pose
    """
    xyz = robot_pose.position
    rotmat = robot_pose.orientation.as_matrix()
    eef_pose = np.eye(4)
    eef_pose[:3, :3] = rotmat
    eef_pose[:3, 3] = xyz

    gripper_offset = np.eye(4)
    gripper_offset[:3, 3] = np.array([0, 0, FINGER_HAND_OFFSET])

    gripper_pose = eef_pose @ gripper_offset
    gripper_xyz = gripper_pose[:3, 3]
    gripper_orientation = R.from_matrix(gripper_pose[:3, :3]).as_quat()
    return np.concatenate([gripper_xyz, gripper_orientation], axis=0)


def franka_obs_to_diff_obs(obs_manager, eef_pose, gripper_state, joint_state, visualize=False):
    """Convert Franka observation to diffusion model observation format.

    Args:
        obs_manager: Point cloud manager for getting raw point clouds
        eef_pose: End effector pose
        gripper_state: Gripper state
        pcd_client: PcdProcessingClient for processing point clouds
        joint_state: Joint state array
        visualize: Whether to visualize

    Returns:
        dict: A dictionary containing the formatted observation
    """
    pcd = obs_manager.get_latest_pointcloud()
    # pcd, render_pcd = pcd_client.process_pcd(pcd, eef_pose, joint_state)

    inhand_cam = 'cam4'
    ih_rgb, ih_depth = obs_manager.get_latest_rgbd(inhand_cam)
    h, w = ih_rgb.shape[:2]
    target_size = 84
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    ih_rgb_cropped = ih_rgb[top:top+min_dim, left:left+min_dim, :]
    resized = np.array(Image.fromarray(ih_rgb_cropped).resize((target_size, target_size), Image.BILINEAR))
    ih_rgb_resized = np.transpose(resized, (2, 0, 1)) / 255.0

    # print(ih_depth)
    # rgb_dict, depth_dict = {inhand_cam: ih_rgb}, {inhand_cam: ih_depth}
    # rgb_dict, depth_dict, is_contact = pcd_client.process_images(rgb_dict, depth_dict)
    # print(max(rgb_dict[inhand_cam].flatten()), min(rgb_dict[inhand_cam].flatten()))

    robot0_eef_pos = eef_pose[:3]
    robot0_eef_quat = eef_pose[3:]
    robot0_gripper_qpos = np.array([gripper_state, gripper_state], dtype=int)

    # Visualize for debugging
    if visualize:
        import open3d as o3d
        pcd_visualize = o3d.geometry.PointCloud()
        pcd_visualize.points = o3d.utility.Vector3dVector(pcd[:, :3])
        pcd_visualize.colors = o3d.utility.Vector3dVector(pcd[:, 3:])
        o3d.visualization.draw_geometries([pcd_visualize])

    # Create observation dictionary
    obs = {
        'pcd': pcd,
        # 'render_pcd': render_pcd,
        'robot0_eye_in_hand_image': ih_rgb_resized,
        'robot0_eye_in_hand_depth': ih_depth[None,...],
        'robot0_eef_pos': robot0_eef_pos.astype(np.float32),
        'robot0_eef_quat': robot0_eef_quat.astype(np.float32),
        'robot0_gripper_qpos': robot0_gripper_qpos.astype(np.float32),
        'robot0_joint_pos': joint_state.astype(np.float32),
        # 'is_contact': np.array([is_contact]).astype(np.float32)
    }

    return obs


def convert_action_from_fingertip_to_gripper(action, rot6d_to_mat):
    """Convert action from fingertip frame to gripper frame.

    Args:
        action: 10-element action [x, y, z, rot6d(6), grasp(1)]
        rot6d_to_mat: Rotation transformer for 6D rotation representation

    Returns:
        Converted action in gripper frame
    """
    if not action.flags.writeable:
        action = action.copy()

    rot6d = action[3:9]
    rotmat = rot6d_to_mat.forward(rot6d.reshape(1, 6))

    # Safety limits - clip Z to prevent collisions
    action[0] = np.clip(action[0], 0.3, 0.8)
    action[1] = np.clip(action[1], -0.35, 0.35)
    action[2] = np.clip(action[2], 0.02, 0.6)

    finger_pose = np.eye(4)
    finger_pose[:3, :3] = rotmat[0]
    finger_pose[:3, 3] = action[:3]

    gripper_offset = np.eye(4)
    gripper_offset[:3, 3] = np.array([0, 0, -FINGER_HAND_OFFSET])

    gripper_pose = finger_pose @ gripper_offset
    gripper_xyz = gripper_pose[:3, 3]
    gripper_rot6d = rot6d_to_mat.inverse(gripper_pose[:3, :3].reshape(1, 3, 3))[0]
    grasp = action[-1:]

    action_converted = np.concatenate([gripper_xyz, gripper_rot6d, grasp], axis=0)
    gripper_rotation = R.from_matrix(gripper_pose[:3, :3])
    return action_converted, gripper_rotation