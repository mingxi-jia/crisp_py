"""A ROS2-based point cloud processing example."""

# %%
import os
import time
from pathlib import Path
import rclpy
import open3d as o3d

from crisp_py.camera.pointcloud import PointCloudManager
from crisp_py.robot import Robot, Pose
from crisp_py.gripper.gripper import Gripper, GripperConfig

import sys
toolbox_path = '/home/mingxi/mingxi_ws/handpi/robot-vision-toolbox'
sys.path.append(toolbox_path)
from robot_filter.arm_segmentor import RobotArmSegmentation

from utils.pcd_utils import render_pcd_from_pose
from hand.trajectory_loader import PointCloudProcessor

# Diffusion Imports / inits
sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy')
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from scipy.spatial.transform import Rotation as R

import torch
import numpy as np
import dill
import hydra


finger_hand_offset = 0.06  # Distance from finger tip to gripper base along z-axis
gripper_norm_const = 0.05  # Normalization constant for gripper value
home_position = np.array([0.60, 0., 0.32 + finger_hand_offset])
joint_thresholds = {
        "panda_link0": 0.08,
        "panda_link1": 0.08,
        "panda_link2": 0.08,
        "panda_link3": 0.08,
        "panda_link4": 0.08,
        "panda_link5": 0.08,
        "panda_link6": 0.08,
        "panda_link7": 0.08,
        "panda_link8": 0.08,
        "panda_hand": 0.02,
        "panda_leftfinger": 0.02,
        "panda_rightfinger": 0.02,
    }

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

def visualize_robot_pcd(raw_pcd, robot_seg, joint_state, no_vis=False):

    joint_names = [j.name for j in robot_seg.robot_urdf.actuated_joints]
    joint_angles = dict(zip(joint_names, joint_state))
    # Sample robot points
    robot_mesh_dict = robot_seg.robot_urdf.visual_trimesh_fk(cfg=joint_angles)
    sampled_points = []
    for mesh, pose in robot_mesh_dict.items():
        transformed = mesh.copy()
        transformed.apply_transform(pose)
        transformed.apply_transform(robot_seg.T_world_urdf)
        sampled_points.append(transformed.sample(2*2500))

    robot_points = np.vstack(sampled_points)
    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.points = o3d.utility.Vector3dVector(robot_points)
    robot_pcd.paint_uniform_color([0.75, 0.75, 0.75])  # Grey color

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(raw_pcd[:, :3])
    pcd_o3d.colors = o3d.utility.Vector3dVector(raw_pcd[:, 3:])
    if no_vis:
        return robot_pcd
    # o3d.visualization.draw_geometries([pcd_o3d, robot_pcd])
    o3d.visualization.draw_geometries([pcd_o3d])

def visualize_pcd_and_actions(pcd, actions, robot_pcd=None):
    
    action_frames = []
    for action in actions:
        x, y, z = action[:3]
        rot6d = action[3:9]
        rotation_transformer = RotationTransformer(
            from_rep='rotation_6d', to_rep='matrix')
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

def franka_obs_to_diff_obs(pcd, eef_pose, gripper_state):
    """Convert Franka observation to diffusion model observation format.
    
    Args:
        pcd (np.ndarray): The input point cloud.
        end_effector_pose (np.ndarray): The robot's end effector pose.
    
    Returns:
        dict: A dictionary containing the formatted observation.
    """

    robot0_eef_pos = eef_pose[:3]
    robot0_eef_quat = eef_pose[3:]  # Assuming quaternion is in (x, y, z, w) format   
    robot0_gripper_qpos = np.array([gripper_state], dtype=int) # !!! Check
    
    # Create observation dictionary
    obs = {
        'pcd': pcd, # !!! Check the format, diffusion expects [1024, 6]
        'robot0_eef_pos': robot0_eef_pos.astype(np.float32), 
        'robot0_eef_quat': robot0_eef_quat.astype(np.float32),
        'robot0_gripper_qpos': robot0_gripper_qpos.astype(np.float32),
    }

    return obs

# !! Not sure how necessary this is 
def warm_up_policy(obs, policy, cfg):
    with torch.no_grad():
        policy.reset()
        device = torch.device('cuda')
        obs_dict_np = get_real_obs_dict(
            env_obs=obs, shape_meta=cfg.task.shape_meta)
        obs_dict = dict_apply(obs_dict_np, 
            lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
        result = policy.predict_action(obs_dict)
        action = result['action'][0].detach().to('cpu').numpy()
        assert action.shape[-1] == 2
        del result


def get_action(obs, policy, cfg):
    with torch.no_grad():
        s = time.time()
        device = torch.device('cuda')
        obs_dict_np = get_real_obs_dict(
            env_obs=obs, shape_meta=cfg.task.shape_meta)
        obs_dict = dict_apply(obs, 
            lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
        print(obs_dict.keys())
        result = policy.predict_action(obs_dict)
        # this action starts from the first obs step
        action = result['action'][0].detach().to('cpu').numpy()
        print('Inference latency:', time.time() - s)
    
    return action

def get_pose_from_robot(robot_pose: Pose):
    xyz = robot_pose.position
    rotmat = robot_pose.orientation.as_matrix()
    eef_pose = np.eye(4)
    eef_pose[:3, :3] = rotmat
    eef_pose[:3, 3] = xyz

    gripper_offset = np.eye(4)
    gripper_offset[:3, 3] = np.array([0, 0, finger_hand_offset])  # 10 cm offset along z-axis

    gripper_pose = eef_pose @ gripper_offset
    gripper_xyz = gripper_pose[:3, 3]
    gripper_orientation = R.from_matrix(gripper_pose[:3, :3]).as_quat()
    return np.concatenate([gripper_xyz, gripper_orientation], axis=0)

def convert_action_from_fingertip_to_gripper(action, rot6d_to_mat):
    print(f"Original action: {action}")
    rot6d = action[3:9]
    rotmat = rot6d_to_mat.forward(rot6d.reshape(1, 6))

    action[2] = np.clip(action[2], 0.0, 0.6)
    finger_pose = np.eye(4)
    finger_pose[:3, :3] = rotmat[0]
    finger_pose[:3, 3] = action[:3]

    gripper_offset = np.eye(4)
    gripper_offset[:3, 3] = np.array([0, 0, -finger_hand_offset])  # 10 cm offset along z-axis

    gripper_pose = finger_pose @ gripper_offset
    gripper_xyz = gripper_pose[:3, 3]
    gripper_rot6d = rot6d_to_mat.inverse(gripper_pose[:3, :3].reshape(1, 3, 3))[0]
    grasp = action[-1:]

    action_converted = np.concatenate([gripper_xyz, gripper_rot6d, grasp], axis=0)
    print(f"Converted action: {action_converted}")
    return action_converted

# %%
def main():

    ctrl_freq = 10.0 # Hz
    ckpt_path = "/home/mingxi/mingxi_ws/handpi/data/data/outputs/2025.11.21/21.01.28_diff_voxel_lift_block_realworld_38_None/checkpoints/latest.ckpt"
    # ckpt_path = "/home/mingxi/mingxi_ws/handpi/data/data/outputs/2025.11.20/19.43.13_diff_voxel_lift_block_realworld_38_None/checkpoints/epoch=0110-val_loss=0.032.ckpt"

    ### ---- Policy Setup ----- ###
    # Load the diffusion model
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.logging.resume = False
    cfg.logging.mode = 'offline'  # Disable logging
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Setup the diffusion model
    delta_action = False
    policy: BaseImagePolicy
    policy = workspace.model # !!! didn't include use ema
    device = torch.device('cuda')
    policy.eval().to(device)
    policy.num_inference_steps = 16 
    n_action_steps = 8
    policy.n_action_steps = n_action_steps
    policy.reset()
    rotation_transformer = RotationTransformer(
            from_rep='rotation_6d', to_rep='matrix')

    ### ---- Robot Setup ----- ###
    """Test the PointCloudManager."""
    rclpy.init()
    # Initialize robot
    robot = Robot(namespace="")
    robot.wait_until_ready()
    print(f"Robot ready. Current joint values: {robot.joint_values}")
    print(robot.end_effector_pose)
    print(robot.joint_values)
    print("Going to home position...")
    robot.home()

    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.cartesian_controller_parameters_client.load_param_config(
        file_path="config/control/default_cartesian_impedance.yaml"
    )
    
    print("Going to start position...")
    robot.move_to(position=home_position, speed=0.15)

    # Initialize gripper
    gripper_config = GripperConfig.from_yaml("./config/gripper_right.yaml")
    gripper = Gripper(gripper_config=gripper_config, namespace="/right/gripper")
    gripper.wait_until_ready()
    gripper.set_target(1.0)
    print("Gripper ready")

    target_pose = robot.end_effector_pose.copy()
    print(f"target_pose: {target_pose}")
    arm_rate = robot.node.create_rate(ctrl_freq)
    gripper_rate = gripper.node.create_rate(ctrl_freq)

    ### ---- Point Cloud Setup ----- ###
    # Initialize robot filter
    config_path = os.path.join(toolbox_path, "configs", "camera_info.yaml")
    manager = PointCloudManager(config_path) # Create point cloud manager

    robot_seg = RobotArmSegmentation(joint_thresholds=joint_thresholds)
    robot_seg.load_urdf(toolbox_path + "/robot_filter/panda_description/urdf/panda_arm_hand_finray.urdf")
    # robot_seg.base_pose = np.array([0.1, 0, 0])

    pcd_processor = PointCloudProcessor()

    import threading # Spin in background thread to receive messages
    spin_thread = threading.Thread(target=rclpy.spin, args=(manager,), daemon=True)
    # spin_thread = threading.Thread(target=rclpy.spin, args=(manager,))
    spin_thread.start()


    ### ---- Inference Loop ----- ###
    n_steps_done = 0 # -1 means warm up
    n_steps_todo = 40 # !! Number of steps to run
    prev_grasp_value = 0.0 # Initialize previous grasp value !! Check value
    t = 0.0
    
    print("Starting inference loop...\n====================\n====================")
    while True:

        if n_steps_done >= n_steps_todo:
            break
    
        # Prepare robot obs
        joint_state = robot.joint_values
        gripper_val = gripper.value
        print(f"gripper_val: {gripper_val}")
        joint_state = np.concatenate([joint_state, [gripper_norm_const * gripper_val]])
        gripper_state = not gripper.is_open()

        # Prepare robot obs
        time.sleep(0.1) # wait till robot is stable (TODO: better way to do this)
        eef_pose = get_pose_from_robot(robot.end_effector_pose)
        pcd = manager.get_latest_pointcloud()
        
        pcd = pcd_processor.process_raw_pcd(pcd, eef_pose)
        pcd = robot_seg.segment(pcd, joint_state)
        obs_dict = franka_obs_to_diff_obs(pcd, eef_pose, gripper_state)

        robot_pcd = visualize_robot_pcd(pcd, robot_seg, joint_state, no_vis=True)

        if n_steps_done == -1: # Warm up policy
            # !! Update the obs input to the policy
            warm_up_policy(obs=obs_dict, policy=policy, cfg=cfg)  # Replace None with actual observation if available
            n_steps_done += 1
            continue
        
        # Get action from policy
        actions = get_action(obs=obs_dict, policy=policy, cfg=cfg)
        print(f"Policy inference: {actions}")
        visualize_pcd_and_actions(pcd=pcd, actions=actions, robot_pcd=robot_pcd)
        for index, action in enumerate(actions):
            action = convert_action_from_fingertip_to_gripper(action, rotation_transformer)
            
            target_pose.position = action[:3]
            target_pose.orientation = R.from_euler('XYZ', [np.pi, 0, 0])

            # robot.move_to(position=np.array([x, y, z]), speed=0.15)
            robot.set_target(pose=target_pose)
            arm_rate.sleep()

            grasp_value = np.round(np.clip(action[-1], 0, 1))
            if grasp_value != prev_grasp_value:
                gripper.set_target(1-grasp_value)
                gripper_rate.sleep()
                time.sleep(1.0)  # wait for gripper to move (due to franka driver limitation)
            prev_grasp_value = grasp_value

        n_steps_done += 1

    # Cleanup
    robot.home()
    robot.shutdown()
    manager.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
        try:
            rclpy.shutdown()
        except Exception:
            pass
        try:
            sys.exit(0)
        except SystemExit:
            pass
