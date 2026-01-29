"""Coordinate transformation utilities for diffusion policy."""

import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from crisp_py.robot import Pose
from PIL import Image

try:
    from .diffusion_constants import FINGER_HAND_OFFSET, ROBOTIQ_ROTATION_OFFSET
except ImportError:
    from diffusion_constants import FINGER_HAND_OFFSET, ROBOTIQ_ROTATION_OFFSET


def get_pose_from_robot(robot_pose: Pose, ret_orig=True) -> np.ndarray:
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
    gripper_offset[:3, :3] = R.from_euler('XYZ', ROBOTIQ_ROTATION_OFFSET).as_matrix()
    if ret_orig:
        gripper_offset[:3, 3] = np.array([0, 0, 0])


    gripper_pose = eef_pose @ gripper_offset
    gripper_xyz = gripper_pose[:3, 3]
    gripper_orientation = R.from_matrix(gripper_pose[:3, :3]).as_quat()
    return np.concatenate([gripper_xyz, gripper_orientation], axis=0)

def convert_pose_from_robot_to_fingertip(ee_poses: dict) -> dict:
    corrected_hand_poss = []
    for frame_idx, hand_pos in enumerate(ee_poses):
        # Apply corrective rotation (calculated from first frame)
        hand_mat = np.eye(4)
        hand_mat[:3, :3] = R.from_quat(hand_pos[3:]).as_matrix()
        hand_mat[:3, 3] = hand_pos[:3]

        # Apply fixed translation offset from robot EE to hand fingertip
        eTf = np.eye(4)
        eTf[:3,3] = np.array([0.0, 0.0, 0.06])

        corrected_hand_mat = hand_mat @ eTf
        hand_pos[:3] = corrected_hand_mat[:3, 3]
        hand_pos[3:] = R.from_matrix(corrected_hand_mat[:3,:3]).as_quat()
        hand_pos[2] = np.clip(hand_pos[2], 0.0, None)  # prevent z from going below 0
        corrected_hand_poss.append(hand_pos)
         
    return np.array(corrected_hand_poss)

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


def convert_action_from_fingertip_to_gripper(action, rot6d_to_mat, ret_orig=True, clip=True):
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
    if clip:
        action[0] = np.clip(action[0], 0.3, 0.8)
        action[1] = np.clip(action[1], -0.35, 0.35)
        action[2] = np.clip(action[2], 0, 0.61)

    finger_pose = np.eye(4)
    finger_pose[:3, :3] = rotmat[0]
    finger_pose[:3, 3] = action[:3]
    # print(f"FINGER_HAND_OFFSET: {FINGER_HAND_OFFSET}")
    gripper_offset = np.eye(4)
    gripper_offset[:3, 3] = np.array([0, 0, -FINGER_HAND_OFFSET])
    gripper_offset[:3, :3] = R.from_euler('XYZ', -1 * ROBOTIQ_ROTATION_OFFSET).as_matrix()
    if ret_orig:
        gripper_offset[:3, 3] = np.array([0, 0, 0])

    gripper_pose = finger_pose @ gripper_offset
    gripper_xyz = gripper_pose[:3, 3]
    gripper_rot6d = rot6d_to_mat.inverse(gripper_pose[:3, :3].reshape(1, 3, 3))[0]
    grasp = action[-1:]

    action_converted = np.concatenate([gripper_xyz, gripper_rot6d, grasp], axis=0)
    gripper_rotation = R.from_matrix(gripper_pose[:3, :3])
    return action_converted, gripper_rotation


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add parent directory to path for imports when running directly
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir.parent))
    sys.path.insert(0, str(script_dir))

    from diffusion_constants import DPEvalConfig

    example_data = {
        "data_1": {
            "eef_pos": np.array([0.5, 0.0, 0.4]),
            "eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        },

        "data_2": {
            "eef_pos": np.array([-0.3, 0.2, 0.6]),
            "eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        },

        "data_3": {
            "eef_pos": np.array([0.4, -0.2, 0.3]),
            "eef_quat": np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)]),
        },

        "data_4": {
            "eef_pos": np.array([0.6, 0.1, 0.5]),
            "eef_quat": np.array([0.0, np.sin(np.pi/4), 0.0, np.cos(np.pi/4)]),
        },

        "data_5": {
            "eef_pos": np.array([0.45, 0.05, 0.35]),
            "eef_quat": np.array([np.sin(np.pi/4), 0.0, 0.0, np.cos(np.pi/4)]),
        },

        "data_6": {
            "eef_pos": np.array([0.55, -0.15, 0.45]),
            "eef_quat": np.array([
                0.182574,  # x
                0.365148,  # y
                0.547723,  # z
                0.730297,  # w
            ]),  # already normalized
        },

        "data_7": {
            "eef_pos": np.array([0.3, 0.0, 0.01]),
            "eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        },

        "data_8": {
            "eef_pos": np.array([0.5, 0.2, 0.4]),
            "eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        },

        "data_9": {
            "eef_pos": np.array([0.62, -0.08, 0.52]),
            "eef_quat": np.array([0.12, -0.23, 0.41, 0.87]),
        },

        "data_10": {
            "eef_pos": np.array([0.4, 0.1, -0.05]),
            "eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        },
    }

    config = DPEvalConfig()
    sys.path.append(config.diffusion_policy_path)
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer

    # Transformer for rot6d <-> matrix conversion (used by convert_action_from_fingertip_to_gripper)
    # forward: rot6d -> matrix, inverse: matrix -> rot6d
    rot6d_to_mat = RotationTransformer('rotation_6d', 'matrix')

    # Transformer for matrix -> rot6d conversion (used for building actions)
    mat_to_rot6d = RotationTransformer('matrix', 'rotation_6d')

    ee_poses = np.stack([
        np.concatenate([v["eef_pos"], v["eef_quat"]])
        for v in example_data.values()
    ], axis=0)

    new_hand_poses = convert_pose_from_robot_to_fingertip(ee_poses)

    # Build actions array - use full quaternion (not sliced)
    # mat_to_rot6d.forward: matrix -> rot6d
    actions = np.stack([
        np.concatenate([
            v["eef_pos"],
            mat_to_rot6d.forward(
                R.from_quat(v['eef_quat']).as_matrix().reshape(1, 3, 3).astype(np.float32)
            )[0],
            np.array([1.0])
        ])
        for v in example_data.values()
    ], axis=0)

    # Process each action individually since convert_action_from_fingertip_to_gripper expects single action
    converted_positions = []
    for action in actions:
        action_converted, gripper_rotation = convert_action_from_fingertip_to_gripper(
            action, rot6d_to_mat, ret_orig=False, clip=False
        )
        converted_positions.append(action_converted[:3])

    converted_positions = np.array(converted_positions)

    print("Converted action positions:")
    print(converted_positions)
    print("\nNew hand poses positions:")
    print(new_hand_poses[:, :3])


    actions_hand_poses = np.stack([
        np.concatenate([
            pos,
            mat_to_rot6d.forward(
                R.from_quat(quat).as_matrix().reshape(1, 3, 3).astype(np.float32)
            )[0],
            np.array([1.0])
        ])
        for pos, quat in zip(new_hand_poses[:, :3], new_hand_poses[:, 3:])
    ], axis=0)

    converted_actions = []
    for action in actions_hand_poses:
        action_converted, gripper_rotation = convert_action_from_fingertip_to_gripper(
            action, rot6d_to_mat, ret_orig=False, clip=False
        )
        converted_actions.append(action_converted)

    converted_actions = np.array(converted_actions)

    for (action_done, action_raw) in zip(converted_actions, example_data.values()):
        pos = action_done[:3]
        # print(type(action[3:9]))
        # print(action[3:9])
        quat = R.from_matrix(
            rot6d_to_mat.forward(action_done[3:9].reshape(1,6))[0]
        ).as_quat()
        print(f"Action:\n\t{pos}\n\t{quat}\n\t{action_raw['eef_pos']}\n\t{action_raw['eef_quat']}")

    print("### Testing on preprocess + action conversion Done ###")

    # ==========================================================================
    # Test: get_pose_from_robot + convert_action_from_fingertip_to_gripper
    # ==========================================================================
    print("\n" + "="*70)
    print("### Testing get_pose_from_robot + convert_action_from_fingertip_to_gripper ###")
    print("="*70)

    # Create a mock Pose class for testing (mimics crisp_py.robot.Pose)
    class MockPose:
        def __init__(self, position: np.ndarray, quat: np.ndarray):
            self.position = position
            self.orientation = R.from_quat(quat)

    # Test with each example data point
    for name, data in example_data.items():
        # Create mock robot pose
        robot_pose = MockPose(data["eef_pos"], data["eef_quat"])

        # Step 1: get_pose_from_robot (robot EE -> fingertip, when ret_orig=False)
        fingertip_pose = get_pose_from_robot(robot_pose, ret_orig=False)
        fingertip_pos = fingertip_pose[:3]
        fingertip_quat = fingertip_pose[3:]

        # Step 2: Convert fingertip pose to action format (with rot6d)
        fingertip_action = np.concatenate([
            fingertip_pos,
            mat_to_rot6d.forward(
                R.from_quat(fingertip_quat).as_matrix().reshape(1, 3, 3).astype(np.float32)
            )[0],
            np.array([1.0])  # grasp value
        ])

        # Step 3: convert_action_from_fingertip_to_gripper (fingertip -> robot EE)
        gripper_action, gripper_rotation = convert_action_from_fingertip_to_gripper(
            fingertip_action, rot6d_to_mat, ret_orig=False, clip=False
        )

        # Extract recovered position and quaternion
        recovered_pos = gripper_action[:3]
        recovered_quat = R.from_matrix(
            rot6d_to_mat.forward(gripper_action[3:9].reshape(1, 6))[0]
        ).as_quat()

        # Compare original vs recovered
        pos_error = np.linalg.norm(data["eef_pos"] - recovered_pos)
        # Handle quaternion sign ambiguity (q and -q represent same rotation)
        quat_dot = np.abs(np.dot(data["eef_quat"], recovered_quat))
        quat_error = 1.0 - quat_dot

        print(f"\n{name}:")
        print(f"  Original pos:  {data['eef_pos']}")
        print(f"  Recovered pos: {recovered_pos}")
        print(f"  Position error: {pos_error:.6f}")
        print(f"  Original quat:  {data['eef_quat']}")
        print(f"  Recovered quat: {recovered_quat}")
        print(f"  Quaternion error: {quat_error:.6f}")

        # Check if round-trip is successful (within tolerance)
        if pos_error < 1e-5 and quat_error < 1e-5:
            print(f"  Status: PASS")
        else:
            print(f"  Status: FAIL")

    # Also test with ret_orig=False (no offset applied)
    print("\n" + "-"*70)
    print("Testing with ret_orig=False (no fingertip offset):")
    print("-"*70)

    for name, data in list(example_data.items())[:3]:  # Test first 3 only
        robot_pose = MockPose(data["eef_pos"], data["eef_quat"])

        # get_pose_from_robot with ret_orig=False (no offset)
        pose_no_offset = get_pose_from_robot(robot_pose, ret_orig=False)

        # convert_action_from_fingertip_to_gripper with ret_orig=False (no offset)
        action_no_offset = np.concatenate([
            pose_no_offset[:3],
            mat_to_rot6d.forward(
                R.from_quat(pose_no_offset[3:]).as_matrix().reshape(1, 3, 3).astype(np.float32)
            )[0],
            np.array([1.0])
        ])
        recovered_action, _ = convert_action_from_fingertip_to_gripper(
            action_no_offset, rot6d_to_mat, ret_orig=False, clip=False
        )

        recovered_pos = recovered_action[:3]
        pos_error = np.linalg.norm(data["eef_pos"] - recovered_pos)

        print(f"\n{name}:")
        print(f"  Original pos:  {data['eef_pos']}")
        print(f"  Recovered pos: {recovered_pos}")
        print(f"  Position error: {pos_error:.6f}")
        print(f"  Status: {'PASS' if pos_error < 1e-5 else 'FAIL'}")

    print("\n### Testing get_pose_from_robot + convert_action_from_fingertip_to_gripper Done ###")