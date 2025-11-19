"""A ROS2-based point cloud processing example."""

# %%
import time
from pathlib import Path
import rclpy
import open3d as o3d

from crisp_py.camera.pointcloud import PointCloudManager
from crisp_py.robot import Robot
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

def franka_obs_to_diff_obs(pcd, end_effector_pose, gripper_state):
    """Convert Franka observation to diffusion model observation format.
    
    Args:
        pcd (np.ndarray): The input point cloud.
        end_effector_pose (np.ndarray): The robot's end effector pose.
    
    Returns:
        dict: A dictionary containing the formatted observation.
    """

    robot0_eef_pos = end_effector_pose.position.copy() 
    quat = end_effector_pose.orientation.as_quat()
    robot0_eef_quat = quat  
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


# %%
def main():
    NO_ACTION = False

    ctrl_freq = 10.0 # Hz
    ckpt_path = "/home/mingxi/mingxi_ws/handpi/data/data/outputs/2025.11.19/12.03.00_diff_voxel_block_lift_realworld_39_None/checkpoints/latest.ckpt" # TODO: Specify

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
    action_offset = 0 # !!! Not sure about these params
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
    homing_pose = robot.end_effector_pose.copy()

    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.cartesian_controller_parameters_client.load_param_config(
        file_path="config/control/default_cartesian_impedance.yaml"
    )
    time.sleep(2.0)
    

    print("Going to start position...")
    robot.move_to(position=np.array([0.5, 0., 0.3]), speed=0.15)

    # Initialize gripper
    gripper_config = GripperConfig.from_yaml("./config/gripper_right.yaml")
    gripper = Gripper(gripper_config=gripper_config, namespace="/right/gripper")
    gripper.wait_until_ready()
    gripper.set_target(1.0)
    print("Gripper ready")

    target_pose = robot.end_effector_pose.copy()
    arm_rate = robot.node.create_rate(ctrl_freq)
    gripper_rate = gripper.node.create_rate(ctrl_freq)


    ### ---- Point Cloud Setup ----- ###
    # Initialize robot filter
    config_path = Path(__file__).parent.parent / "config" / "camera_info.yaml"
    print(f"debug: config_path = {config_path}")
    manager = PointCloudManager(str(config_path)) # Create point cloud manager

    # Define per-link thresholds for adaptive filtering
    # Stricter thresholds for fingers, looser for arm body

    robot_seg = RobotArmSegmentation()
    robot_seg.load_urdf(toolbox_path + "/robot_filter/panda_description/urdf/panda_arm_hand_finray.urdf")
    # robot_seg.base_pose = np.array([0.1, 0, 0])

    pcd_processor = PointCloudProcessor()

    import threading # Spin in background thread to receive messages
    # spin_thread = threading.Thread(target=rclpy.spin, args=(manager,), daemon=True)
    spin_thread = threading.Thread(target=rclpy.spin, args=(manager,))
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
    
        # Get current joint state from robot
        joint_state = robot.joint_values
        gripper_val = gripper.value
        print(f"gripper_val: {gripper_val}")
        joint_state = np.concatenate([joint_state, [0.055 * gripper_val]])
        # Use adaptive threshold segmentation to filter robot from point cloud
        # filtered_pcd = pcd # Debug: Check misalignment
        end_effector_pose = robot.end_effector_pose
        gripper_state = not gripper.is_open()

        # Generate robot point cloud for visualization
        joint_names = [j.name for j in robot_seg.robot_urdf.actuated_joints]
        joint_angles = dict(zip(joint_names, joint_state))
        robot_mesh_dict = robot_seg.robot_urdf.visual_trimesh_fk(cfg=joint_angles)

        # Sample robot points
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

        # Prepare robot obs
        eef_pose = np.concatenate([end_effector_pose.position, end_effector_pose.orientation.as_quat()], axis=0)

        pcd = manager.get_latest_pointcloud()
        if pcd is None:
            print("No point cloud received yet.")
            time.sleep(1.0)
            continue
        filtered_pcd = robot_seg.segment(pcd, joint_state)
        filtered_pcd = pcd_processor.get_render_pcd(filtered_pcd, eef_pose)

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(filtered_pcd[:, :3])
        pcd_o3d.colors = o3d.utility.Vector3dVector(filtered_pcd[:, 3:])
        o3d.visualization.draw_geometries([pcd_o3d, robot_pcd])
        # o3d.visualization.draw_geometries([pcd_o3d])

        obs_dict = franka_obs_to_diff_obs(filtered_pcd, end_effector_pose, gripper_state)

        if n_steps_done == -1: # Warm up policy
            # !! Update the obs input to the policy
            warm_up_policy(obs=obs_dict, policy=policy, cfg=cfg)  # Replace None with actual observation if available
            n_steps_done += 1
            continue
        
        # Visualize current end effector euler rotation
        current_euler = end_effector_pose.orientation.as_euler('XYZ')

        # Get action from policy
        actions = get_action(obs=obs_dict, policy=policy, cfg=cfg)
        print(f"Policy inference: {actions}")
        for index, action in enumerate(actions):
            
            print(action.shape)
            x, y, z = action[:3]
            rot6d = action[3:9]
            rotmat = rotation_transformer.forward(rot6d.reshape(1, 6))
            
            # # Create coordinate frame at end effector to visualize orientation
            # eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # eef_transform = np.eye(4)
            # eef_transform[:3, :3] = rotmat
            # eef_transform[:3, 3] = np.array([x, y, z])
            # eef_frame.transform(eef_transform)

            # # Visualize filtered pcd with robot model and EEF frame
            # pcd_o3d = o3d.geometry.PointCloud()
            # pcd_o3d.points = o3d.utility.Vector3dVector(filtered_pcd[:, :3])
            # pcd_o3d.colors = o3d.utility.Vector3dVector(filtered_pcd[:, 3:])
            # o3d.visualization.draw_geometries([pcd_o3d, robot_pcd, eef_frame])
            # o3d.visualization.draw_geometries([pcd_o3d])

            z = np.clip(z, 0.06+0.02, 0.6)
            target_pose.position = np.array([x, y, z])
            print(rotmat.shape)
            target_pose.orientation = R.from_matrix(rotmat[0])
            time.sleep(0.1)
            if not NO_ACTION:
                print(f"robo action {np.array([x, y, z])}")
                # robot.move_to(position=np.array([x, y, z]), speed=0.15)
                robot.set_target(pose=target_pose)
                arm_rate.sleep()

            grasp_value = np.clip(action[-1], 0, 1)
            if grasp_value != prev_grasp_value:
                print(f"Setting gripper to {grasp_value}")
                gripper.set_target(1-grasp_value)
                gripper_rate.sleep()
                time.sleep(1.0)  # wait for gripper to move
            prev_grasp_value = grasp_value
            if index > 2:
                break

        manager.latest_pcd = None  # Clear latest point cloud to force getting a new one
        # t += 1.0 / ctrl_freq
        # i += 1

        
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
