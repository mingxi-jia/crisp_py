"""Coordinate transformation utilities for diffusion policy."""

import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from crisp_py.robot import Pose
from PIL import Image
import copy
import sys

try:
    from .diffusion_constants import FINGER_HAND_OFFSET, ROBOTIQ_ROTATION_OFFSET, DPEvalConfig
except ImportError:
    from diffusion_constants import FINGER_HAND_OFFSET, ROBOTIQ_ROTATION_OFFSET, DPEvalConfig

# Add diffusion_policy to path and create module-level rotation transformer
_config = DPEvalConfig
if _config.diffusion_policy_path not in sys.path:
    sys.path.append(_config.diffusion_policy_path)

from diffusion_policy.model.common.rotation_transformer import RotationTransformer

# Module-level rotation transformer singleton (rot6d <-> matrix)
rot6d_to_mat = RotationTransformer('rotation_6d', 'matrix')
mat_to_rot6d = RotationTransformer('matrix', 'rotation_6d')


def eef_to_tip(robot_pose, xyz_offset, euler_offset) -> Pose: 
    pose_out = copy.deepcopy(robot_pose)
    
    xyz = robot_pose.position
    rotmat = robot_pose.orientation.as_matrix()
    eef_pose = np.eye(4)
    eef_pose[:3, :3] = rotmat
    eef_pose[:3, 3] = xyz

    gripper_offset = np.eye(4)
    gripper_offset[:3, 3] = xyz_offset
    gripper_offset[:3, :3] = R.from_euler('XYZ', euler_offset).as_matrix()

    gripper_pose = eef_pose @ gripper_offset
    gripper_xyz = gripper_pose[:3, 3]
    pose_out.position = gripper_xyz
    pose_out.orientation = R.from_matrix(gripper_pose[:3, :3])
    return pose_out


def get_pose_from_robot(robot_pose: Pose, ret_orig=False, ret_pose=False) -> np.ndarray:
    """Convert robot pose to gripper pose with fingertip offset.

    Args:
        robot_pose: Robot end-effector pose

    Returns:
        7-element array [x, y, z, qx, qy, qz, qw] representing gripper pose
    """
    if ret_orig:
        xyz_offset = np.array([0, 0, 0])
    else:
        xyz_offset = np.array([0, 0, FINGER_HAND_OFFSET])
    pose_out = eef_to_tip(robot_pose, xyz_offset, ROBOTIQ_ROTATION_OFFSET)
    
    if ret_pose: 
        return pose_out

    gripper_xyz = pose_out.position
    gripper_orientation = pose_out.orientation.as_quat()
    return np.concatenate([gripper_xyz, gripper_orientation], axis=0)


def process_rgb(rgb, target_size=84):
    """Crop, resize, and normalize an RGB image to (C, H, W) format."""
    h, w = rgb.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    cropped = rgb[top:top+min_dim, left:left+min_dim, :]
    resized = np.array(Image.fromarray(cropped).resize((target_size, target_size), Image.BILINEAR))
    return np.transpose(resized, (2, 0, 1)) / 255.0

def franka_obs_to_diff_obs(obs_buffer, img_policy=False, visualize=False):
    """Convert Franka observation buffer to diffusion model observation format.

    Args:
        obs_buffer: List of observation snapshots, each containing:
            - eef_pos: End effector position (3,)
            - eef_quat: End effector quaternion (4,)
            - gripper_qpos: Gripper joint positions (2,)
            - joint_pos: Joint positions (7,)
            - cam4_rgb: In-hand camera RGB image
            - cam3_rgb: External camera RGB image (img_policy only)
            - cam4_depth: In-hand camera depth (non-img_policy only)
            - pcd: Point cloud (non-img_policy only)
        img_policy: Whether to use image-only policy (excludes pcd, depth, joint_pos)
        visualize: Whether to visualize point cloud

    Returns:
        dict: A dictionary containing the formatted observation
    """
    # Get last 2 frames from buffer
    frames = obs_buffer[-2:]

    if img_policy:
        # Stack images from last 2 frames: (2, 3, 84, 84)
        ih_rgb_resized = np.stack([process_rgb(f['cam4_rgb']) for f in frames], axis=0)
        cam3_rgb_resized = np.stack([process_rgb(f['cam3_rgb']) for f in frames], axis=0)

        # import matplotlib.pyplot as plt
        # plt.imshow(np.transpose(ih_rgb_resized[0], (1,2,0)))
        # plt.show()
        # plt.imshow(np.transpose(cam3_rgb_resized[0], (1,2,0)))
        # plt.show()
        # Stack states from last 2 frames
        obs = {
            'robot0_eef_pos': np.stack([f['eef_pos'] for f in frames], axis=0),
            'robot0_eef_quat': np.stack([f['eef_quat'] for f in frames], axis=0),
            'robot0_gripper_qpos': np.stack([f['gripper_qpos'] for f in frames], axis=0),
            'robot0_eye_in_hand_image': ih_rgb_resized.astype(np.float32),
            'cam3_image': cam3_rgb_resized.astype(np.float32),
        }
    else:
        # Use latest frame only
        latest = frames[-1]
        pcd = latest['pcd']
        ih_rgb_resized = process_rgb(latest['cam4_rgb'])
        ih_depth = latest['cam4_depth']

        # Visualize for debugging
        if visualize:
            print(f"visualize: {visualize}")
            import open3d as o3d
            pcd_visualize = o3d.geometry.PointCloud()
            pcd_visualize.points = o3d.utility.Vector3dVector(pcd[:, :3])
            pcd_visualize.colors = o3d.utility.Vector3dVector(pcd[:, 3:])
            # o3d.visualization.draw_geometries([pcd_visualize])

        obs = {
            'robot0_eef_pos': latest['eef_pos'],
            'robot0_eef_quat': latest['eef_quat'],
            'robot0_gripper_qpos': latest['gripper_qpos'],
            'robot0_eye_in_hand_image': ih_rgb_resized.astype(np.float32),
            'pcd': pcd,
            'robot0_eye_in_hand_depth': ih_depth.astype(np.float32),
            'robot0_joint_pos': latest['joint_pos'],
        }

    return obs

def ten_d_action_to_pose(action, clip=True):
    """Convert 10D policy action to Pose and grasp value.

    This should be called immediately after policy prediction to convert
    the action from policy output format to Pose format.

    Args:
        action: 10-element action [x, y, z, rot6d(6), grasp(1)]
        clip: Whether to apply safety limits on position

    Returns:
        pose: Pose object with position and orientation (in fingertip frame)
        grasp: Grasp value (0-1)
    """

    # Extract components
    position = action[:3].copy()
    rot6d = action[3:9]
    grasp = float(action[9])

    # Safety limits - clip position to prevent collisions
    if clip:
        position[0] = np.clip(position[0], 0.3, 0.8)
        position[1] = np.clip(position[1], -0.35, 0.35)
        position[2] = np.clip(position[2], 0, 0.61)

    # Convert rot6d to rotation matrix then to scipy Rotation
    rotmat = rot6d_to_mat.forward(rot6d.reshape(1, 6))[0]
    orientation = R.from_matrix(rotmat)

    # Create Pose object
    pose = Pose(position=position, orientation=orientation)

    return pose, grasp


def convert_action_from_fingertip_to_gripper(pose: Pose, ret_orig=False):
    """Convert pose from fingertip frame to gripper frame.

    Args:
        pose: Pose object in fingertip frame
        ret_orig: If True, don't apply fingertip offset

    Returns:
        action: Position array [x, y, z] in gripper frame
        gripper_rotation: scipy Rotation object for gripper orientation
    """
    # Get position and orientation from pose
    position = pose.position
    rotmat = pose.orientation.as_matrix()

    # Build fingertip pose matrix
    finger_pose = np.eye(4)
    finger_pose[:3, :3] = rotmat
    finger_pose[:3, 3] = position

    # Apply offset (fingertip to gripper)
    if ret_orig:
        xyz_offset = np.array([0, 0, 0])
    else:
        xyz_offset = np.array([0, 0, -FINGER_HAND_OFFSET])

    gripper_offset = np.eye(4)
    gripper_offset[:3, 3] = xyz_offset
    gripper_offset[:3, :3] = R.from_euler('XYZ', -1 * ROBOTIQ_ROTATION_OFFSET).as_matrix()

    gripper_pose_mat = finger_pose @ gripper_offset
    gripper_position = gripper_pose_mat[:3, 3]
    gripper_rotation = R.from_matrix(gripper_pose_mat[:3, :3])

    return gripper_position, gripper_rotation


def ten_d_action_to_pose_batch(actions, clip=True):
    """Convert batch of 10D policy actions to Poses and grasp values.

    Uses vectorized operations for speed.

    Args:
        actions: (N, 10) array of actions from policy_client.predict_action()
        clip: Whether to apply safety limits on position

    Returns:
        poses: List of Pose objects
        grasps: List of grasp values
    """
    assert type(actions) == np.ndarray, f"actions needs to be np array, not {type(actions)}"
    n = actions.shape[0]

    # Extract components (vectorized)
    positions = actions[:, :3].copy()  # (N, 3)
    rot6ds = actions[:, 3:9]  # (N, 6)
    grasps = actions[:, 9].tolist()  # List of N floats

    # Safety limits - vectorized clipping
    if clip:
        positions[:, 0] = np.clip(positions[:, 0], 0.3, 0.8)
        positions[:, 1] = np.clip(positions[:, 1], -0.35, 0.35)
        positions[:, 2] = np.clip(positions[:, 2], 0, 0.61)

    # Batch convert rot6d to rotation matrices (single forward pass)
    rotmats = rot6d_to_mat.forward(rot6ds)  # (N, 3, 3)

    # Create Pose objects
    poses = [
        Pose(position=positions[i], orientation=R.from_matrix(rotmats[i]))
        for i in range(n)
    ]

    return poses, grasps


def convert_action_from_fingertip_to_gripper_batch(poses, ret_orig=False):
    """Convert batch of poses from fingertip frame to gripper frame.

    Uses vectorized operations for speed.

    Args:
        poses: List of Pose objects in fingertip frame
        ret_orig: If True, don't apply fingertip offset

    Returns:
        gripper_positions: (N, 3) array of positions
        gripper_rotations: List of scipy Rotation objects
    """
    n = len(poses)

    # Extract positions and rotmats into arrays
    positions = np.array([p.position for p in poses])  # (N, 3)
    rotmats = np.array([p.orientation.as_matrix() for p in poses])  # (N, 3, 3)

    # Build finger pose matrices (N, 4, 4)
    finger_poses = np.zeros((n, 4, 4))
    finger_poses[:, :3, :3] = rotmats
    finger_poses[:, :3, 3] = positions
    finger_poses[:, 3, 3] = 1.0

    # Build gripper offset matrix (shared for all)
    if ret_orig:
        xyz_offset = np.array([0, 0, 0])
    else:
        xyz_offset = np.array([0, 0, -FINGER_HAND_OFFSET])

    gripper_offset = np.eye(4)
    gripper_offset[:3, 3] = xyz_offset
    gripper_offset[:3, :3] = R.from_euler('XYZ', -1 * ROBOTIQ_ROTATION_OFFSET).as_matrix()

    # Batch matrix multiplication
    gripper_pose_mats = finger_poses @ gripper_offset  # (N, 4, 4)

    # Extract results
    gripper_positions = gripper_pose_mats[:, :3, 3]  # (N, 3)
    gripper_rotations = [
        R.from_matrix(gripper_pose_mats[i, :3, :3])
        for i in range(n)
    ]

    return gripper_positions, gripper_rotations