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
sys.path.append('/home/mingxi/mingxi_ws/handpi/robot-vision-toolbox')
from robot_filter.arm_segmentor import RobotArmSegmentation

# Diffusion Imports / inits
sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy')
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply

import torch
import numpy as np
import dill
import hydra


ctrl_freq = 10.0 # Hz
ckpt_path = "/home/mingxi/mingxi_ws/handpi/data/data/outputs/2025.11.16/14.49.15_diff_voxel_block_lift_realworld_39_None/checkpoints/latest.ckpt" # TODO: Specify


def franka_obs_to_diff_obs(pcd_o3d: o3d.geometry.PointCloud, joint_values: np.ndarray):
    """Convert Franka observation to diffusion model observation format.
    
    Args:
        pcd_o3d (o3d.geometry.PointCloud): The input point cloud.
        joint_values (np.ndarray): The robot's joint values.
    
    Returns:
        dict: A dictionary containing the formatted observation.
    """

    robot0_eef_pos = joint_values[:3] 
    robot0_eef_quat = joint_values[3:7]  # !!! Assuming quaternion is in (x, y, z, w) format
    robot0_gripper_qpos = joint_values[:-1] # !!! Check

    # Create observation dictionary
    obs = {
        'pcd': {
            pcd_o3d # !!! Check the format, diffusion expects [1024, 6]
        },
        'robot0_eef_pos': robot0_eef_pos.astype(np.float32),
        'robot0_eef_quat': robot0_eef_quat.astype(np.float32),
        'robot0_gripper_qpos': robot0_gripper_qpos.astype(np.float32),
    }

    return obs

# !! Not sure how necessary this is 
def warm_up_policy(obs, policy, cfg):
    with torch.no_grad():
        policy.reset()
        obs_dict_np = get_real_obs_dict(
            env_obs=obs, shape_meta=cfg.task.shape_meta)
        obs_dict = dict_apply(obs_dict_np, 
            lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        result = policy.predict_action(obs_dict)
        action = result['action'][0].detach().to('cpu').numpy()
        assert action.shape[-1] == 2
        del result


def get_action(obs, policy, cfg):
    with torch.no_grad():
        s = time.time()
        obs_dict_np = get_real_obs_dict(
            env_obs=obs, shape_meta=cfg.task.shape_meta)
        obs_dict = dict_apply(obs_dict_np, 
            lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        result = policy.predict_action(obs_dict)
        # this action starts from the first obs step
        action = result['action'].detach().to('cpu').numpy()
        print('Inference latency:', time.time() - s)
    
    return action


# %%
def main():
    """Test the PointCloudManager."""
    rclpy.init()

    NO_ACTION = True

    ### ---- Policy Setup ----- ###
    # Load the diffusion model
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
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

    ### ---- Robot Setup ----- ###
    # Initialize robot
    robot = Robot(namespace="")
    robot.wait_until_ready()
    print(f"Robot ready. Current joint values: {robot.joint_values}")
    print(robot.end_effector_pose)
    print(robot.joint_values)
    print("Going to home position...")
    robot.home()
    homing_pose = robot.end_effector_pose.copy()

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
    manager = PointCloudManager(str(config_path)) # Create point cloud manager
    robot_seg = RobotArmSegmentation() 
    robot_seg.load_urdf("robot_filter/panda_description/urdf/panda_arm_hand.urdf")

    import threading # Spin in background thread to receive messages
    spin_thread = threading.Thread(target=rclpy.spin, args=(manager,), daemon=True)
    spin_thread.start()


    ### ---- Inference Loop ----- ###
    n_steps_done = -1 # -1 means warm up
    n_steps_todo = 40 # !! Number of steps to run
    prev_grasp_value = 0.0 # Initialize previous grasp value !! Check value
    t = 0.0
    xxxxx
    while True:

        if n_steps_done >= n_steps_todo:
            break
        pcd = manager.get_latest_pointcloud()


        if pcd is None:
            print("No point cloud received yet.")
            continue

        # TODO? crop pcd using workspace
        
        # Get current joint state from robot
        joint_state = robot.joint_values
        filtered_pcd = robot_seg.segment(pcd, joint_state)

        filtered_pcd_o3d = o3d.geometry.PointCloud()
        filtered_pcd_o3d.points = o3d.utility.Vector3dVector(filtered_pcd[:, :3])
        filtered_pcd_o3d.colors = o3d.utility.Vector3dVector(filtered_pcd[:, 3:])
        
        # o3d.visualization.draw_geometries(
        #     [filtered_pcd_o3d],
        #     window_name="Filtered Point Cloud",
        #     width=800,
        #     height=600
        # )

        # Prepare robot obs
        obs_dict = franka_obs_to_diff_obs(filtered_pcd_o3d, joint_state)

        if n_steps_done == -1: # Warm up policy
            # !! Update the obs input to the policy
            warm_up_policy(obs=obs_dict, policy=policy)  # Replace None with actual observation if available
            n_steps_done += 1
            continue
        
        # Get action from policy
        actions = get_action(obs=obs_dict, policy=policy, cfg=cfg)
        print(f"Moving to pose {action}")
        for index, action in enumerate(actions):
            x, y, z = action[:3]
            z = np.clip(z-0.02, 0.06, 0.6)
            target_pose.position = np.array([x, y, z])

            if not NO_ACTION:
                robot.set_target(pose=target_pose)
                arm_rate.sleep()

            grasp_value = action[-1]
            if grasp_value != prev_grasp_value:
                print(f"Setting gripper to {grasp_value}")
                gripper.set_target(1-grasp_value)
                gripper_rate.sleep()
                time.sleep(1.0)  # wait for gripper to move
            prev_grasp_value = grasp_value
    
        # t += 1.0 / ctrl_freq
        # i += 1

        
        n_steps_done += 1

    # Cleanup
    robot.home()
    robot.shutdown()
    manager.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
