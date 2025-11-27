"""Instrumented version of pcd_diff_eval_debug_inf_time.py
to diagnose why identical observations cause variable inference times.

This adds diagnostics to capture:
1. GPU state before each inference
2. Thread activity
3. System state
4. CUDA context
"""

import os
import time
from pathlib import Path
import rclpy
import open3d as o3d
import threading
import subprocess

from crisp_py.camera.pointcloud import PointCloudManager
from crisp_py.robot import Robot, Pose
from crisp_py.gripper.gripper import Gripper, GripperConfig
from sensor_msgs.msg import JointState

import sys
toolbox_path = '/home/mingxi/mingxi_ws/handpi/robot-vision-toolbox'
sys.path.append(toolbox_path)
from robot_filter.arm_segmentor import RobotArmSegmentation

from utils.pcd_utils import render_pcd_from_pose
from hand.trajectory_loader import ObservationProcessor

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


finger_hand_offset = 0.06
gripper_norm_const = 0.05
home_position = np.array([0.60, 0., 0.32 + finger_hand_offset])
joint_thresholds = {
    "panda_link0": 0.08, "panda_link1": 0.08, "panda_link2": 0.08,
    "panda_link3": 0.08, "panda_link4": 0.08, "panda_link5": 0.08,
    "panda_link6": 0.08, "panda_link7": 0.08, "panda_link8": 0.08,
    "panda_hand": 0.02, "panda_leftfinger": 0.02, "panda_rightfinger": 0.02,
}


def get_gpu_state():
    """Get current GPU state."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,clocks.current.graphics,power.draw,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=0.5
        )
        if result.returncode == 0:
            util, clock, power, temp = result.stdout.strip().split(',')
            return {
                'util': float(util),
                'clock_mhz': float(clock),
                'power_w': float(power),
                'temp_c': float(temp)
            }
    except Exception:
        pass
    return None


def get_system_state():
    """Get current system state."""
    state = {
        'threads': threading.active_count(),
        'thread_names': [t.name for t in threading.enumerate()],
        'cuda_mem_allocated_mb': torch.cuda.memory_allocated(0) / 1024**2,
        'cuda_mem_reserved_mb': torch.cuda.memory_reserved(0) / 1024**2,
    }
    return state


class JointStateSubscriber:
    """A simple ROS2 subscriber to get joint states directly from /joint_states topic."""

    def __init__(self, node, joint_names: list[str], topic: str = "/joint_states"):
        self._node = node
        self._received = False
        self.joint_array = None

        self._subscription = node.create_subscription(
            JointState, topic, self._callback, 10)

    def _callback(self, msg: JointState):
        joint_names = msg.name
        joint_positions = msg.position
        sorted_indices = sorted(range(len(joint_names)), key=lambda i: joint_names[i])
        sorted_positions = [joint_positions[i] for i in sorted_indices]
        self.joint_array = np.array(sorted_positions, dtype=np.float32)
        self._received = True

    @property
    def joint_values(self) -> np.ndarray:
        return self.joint_array[1:]

    @property
    def is_ready(self) -> bool:
        return self._received


def franka_obs_to_diff_obs(obs_manager: PointCloudManager, eef_pose, gripper_state, pcd_processor, joint_state):
    """Convert Franka observation to diffusion model observation format."""
    t0 = time.time()
    pcd = obs_manager.get_latest_pointcloud()
    t_latest_pcd = time.time() - t0

    t0 = time.time()
    pcd, render_pcd = pcd_processor.get_policy_obs(pcd, eef_pose, joint_state)
    t_process_pcd = time.time() - t0

    t0 = time.time()
    inhand_cam = 'cam4'
    ih_rgb, ih_depth = obs_manager.get_latest_rgbd(inhand_cam)
    rgb_dict, depth_dict = {inhand_cam: ih_rgb}, {inhand_cam: ih_depth}
    rgb_dict, depth_dict = pcd_processor.get_policy_images(rgb_dict, depth_dict)
    t_process_images = time.time() - t0

    robot0_eef_pos = eef_pose[:3]
    robot0_eef_quat = eef_pose[3:]
    robot0_gripper_qpos = np.array([gripper_state, gripper_state], dtype=int)

    obs = {
        'pcd': pcd,
        'render_pcd': render_pcd,
        'robot0_eye_in_hand_image': np.transpose(rgb_dict[inhand_cam], (2, 0, 1)) / 255.0,
        'robot0_eef_pos': robot0_eef_pos.astype(np.float32),
        'robot0_eef_quat': robot0_eef_quat.astype(np.float32),
        'robot0_gripper_qpos': robot0_gripper_qpos.astype(np.float32),
    }

    return obs


def get_action_instrumented(obs, policy, cfg, device, step_num):
    """Instrumented version of get_action with detailed diagnostics."""

    # Capture state BEFORE inference
    gpu_state_before = get_gpu_state()
    sys_state_before = get_system_state()

    with torch.no_grad():
        # Time tensor conversion and GPU transfer
        t0 = time.time()
        obs_dict = dict_apply(obs,
            lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(1).to(device))
        torch.cuda.synchronize()
        t_to_gpu = time.time() - t0

        # Time policy inference - THIS IS WHERE VARIANCE HAPPENS
        torch.cuda.synchronize()
        t0 = time.time()
        result = policy.predict_action(obs_dict)
        torch.cuda.synchronize()
        t_predict = time.time() - t0

        # Time result transfer back to CPU
        t0 = time.time()
        action = result['action'][0].detach().to('cpu').numpy()
        t_to_cpu = time.time() - t0

    # Capture state AFTER inference
    gpu_state_after = get_gpu_state()
    sys_state_after = get_system_state()

    # Print detailed diagnostics
    print(f"\n{'='*80}")
    print(f"STEP {step_num} INFERENCE DIAGNOSTICS")
    print(f"{'='*80}")

    print(f"\n[Timing Breakdown]")
    print(f"  to GPU:          {t_to_gpu*1000:6.2f} ms")
    print(f"  predict_action:  {t_predict*1000:6.2f} ms  ← VARIABLE COMPONENT")
    print(f"  to CPU:          {t_to_cpu*1000:6.2f} ms")
    print(f"  TOTAL:           {(t_to_gpu + t_predict + t_to_cpu)*1000:6.2f} ms")

    if gpu_state_before and gpu_state_after:
        print(f"\n[GPU State]")
        print(f"  Before:  util={gpu_state_before['util']:5.1f}%, clock={gpu_state_before['clock_mhz']:4.0f}MHz, "
              f"power={gpu_state_before['power_w']:5.1f}W, temp={gpu_state_before['temp_c']:4.1f}°C")
        print(f"  After:   util={gpu_state_after['util']:5.1f}%, clock={gpu_state_after['clock_mhz']:4.0f}MHz, "
              f"power={gpu_state_after['power_w']:5.1f}W, temp={gpu_state_after['temp_c']:4.1f}°C")

        # Check if clock changed
        clock_change = gpu_state_after['clock_mhz'] - gpu_state_before['clock_mhz']
        if abs(clock_change) > 50:
            print(f"  ⚠ GPU CLOCK CHANGED BY {clock_change:+.0f} MHz DURING INFERENCE!")

    print(f"\n[System State]")
    print(f"  Active threads:   {sys_state_before['threads']}")
    print(f"  CUDA mem alloc:   {sys_state_before['cuda_mem_allocated_mb']:.2f} MB")
    print(f"  CUDA mem reserved: {sys_state_before['cuda_mem_reserved_mb']:.2f} MB")

    # Check for thread activity
    if sys_state_before['threads'] != sys_state_after['threads']:
        print(f"  ⚠ Thread count changed during inference: {sys_state_before['threads']} → {sys_state_after['threads']}")

    print(f"{'='*80}\n")

    return action, {
        't_predict': t_predict * 1000,
        't_to_gpu': t_to_gpu * 1000,
        't_to_cpu': t_to_cpu * 1000,
        'gpu_before': gpu_state_before,
        'gpu_after': gpu_state_after,
        'sys_before': sys_state_before,
        'sys_after': sys_state_after,
    }


def get_pose_from_robot(robot_pose: Pose):
    xyz = robot_pose.position
    rotmat = robot_pose.orientation.as_matrix()
    eef_pose = np.eye(4)
    eef_pose[:3, :3] = rotmat
    eef_pose[:3, 3] = xyz

    gripper_offset = np.eye(4)
    gripper_offset[:3, 3] = np.array([0, 0, finger_hand_offset])

    gripper_pose = eef_pose @ gripper_offset
    gripper_xyz = gripper_pose[:3, 3]
    gripper_orientation = R.from_matrix(gripper_pose[:3, :3]).as_quat()
    return np.concatenate([gripper_xyz, gripper_orientation], axis=0)


def main():
    ctrl_freq = 10.0
    ckpt_path = "/home/mingxi/Downloads/epoch=0040-val_loss=0.008.ckpt"

    ### ---- Policy Setup ----- ###
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.logging.resume = False
    cfg.logging.mode = 'offline'
    cfg.real_robot_eval = True
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy: BaseImagePolicy = workspace.model
    device = torch.device('cuda')
    policy.eval()
    policy.to(device)
    policy.num_inference_steps = 20
    n_action_steps = 8
    policy.n_action_steps = n_action_steps
    policy.reset()

    rotation_transformer = RotationTransformer(from_rep='rotation_6d', to_rep='matrix')

    ### ---- Robot Setup ----- ###
    rclpy.init()
    robot = Robot(namespace="")
    robot.wait_until_ready()
    print(f"Robot ready. Current joint values: {robot.joint_values}")
    robot.home()

    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.cartesian_controller_parameters_client.load_param_config(
        file_path="config/control/default_cartesian_impedance.yaml"
    )

    print("Going to start position...")
    robot.move_to(position=home_position, speed=0.15)

    gripper_config = GripperConfig.from_yaml("./config/gripper_right.yaml")
    gripper = Gripper(gripper_config=gripper_config, namespace="/right/gripper")
    gripper.wait_until_ready()
    gripper.set_target(1.0)

    target_pose = robot.end_effector_pose.copy()
    arm_rate = robot.node.create_rate(ctrl_freq)
    gripper_rate = gripper.node.create_rate(ctrl_freq)

    ### ---- Point Cloud Setup ----- ###
    config_path = os.path.join(toolbox_path, "configs", "camera_info.yaml")
    manager = PointCloudManager(config_path)

    pcd_processor = ObservationProcessor()

    joint_names = sorted([j.name for j in pcd_processor.robot_filter.robot_urdf.actuated_joints])
    joint_state_subscriber = JointStateSubscriber(manager, joint_names, topic="/joint_states")

    spin_thread = threading.Thread(target=rclpy.spin, args=(manager,))
    spin_thread.start()

    while not joint_state_subscriber.is_ready:
        time.sleep(0.1)

    ### ---- Inference Loop ----- ###
    n_steps_done = 0
    n_steps_todo = 20  # Reduced for testing

    # Storage for diagnostics
    all_diagnostics = []

    print("\n" + "="*80)
    print("STARTING INSTRUMENTED INFERENCE LOOP")
    print("="*80)
    print("\nThis will run inference with detailed diagnostics to identify")
    print("why identical observations cause variable inference times.\n")

    while n_steps_done < n_steps_todo:
        iter_start = time.time()

        # Prepare observation
        joint_state = joint_state_subscriber.joint_values
        gripper_val = gripper.value
        joint_state = np.concatenate([joint_state, [gripper_norm_const * gripper_val]])
        gripper_state = not gripper.is_open()

        time.sleep(0.05)
        eef_pose = get_pose_from_robot(robot.end_effector_pose)

        # Get observation
        obs_dict = franka_obs_to_diff_obs(manager, eef_pose, gripper_state, pcd_processor, joint_state)

        # INSTRUMENTED INFERENCE - This will show detailed diagnostics
        actions, diagnostics = get_action_instrumented(obs_dict, policy, cfg, device, n_steps_done)

        all_diagnostics.append(diagnostics)
        n_steps_done += 1

        iter_total = time.time() - iter_start
        print(f"Total iteration time: {iter_total*1000:6.1f} ms\n")

    # Analyze all diagnostics
    print("\n" + "="*80)
    print("AGGREGATE ANALYSIS")
    print("="*80)

    predict_times = [d['t_predict'] for d in all_diagnostics]
    gpu_clocks_before = [d['gpu_before']['clock_mhz'] for d in all_diagnostics if d['gpu_before']]

    print(f"\nInference times (predict_action):")
    print(f"  Mean:   {np.mean(predict_times):6.2f} ms")
    print(f"  Std:    {np.std(predict_times):6.2f} ms")
    print(f"  Min:    {np.min(predict_times):6.2f} ms")
    print(f"  Max:    {np.max(predict_times):6.2f} ms")
    print(f"  Range:  {np.max(predict_times) - np.min(predict_times):6.2f} ms")

    if gpu_clocks_before:
        print(f"\nGPU clocks before inference:")
        print(f"  Mean:   {np.mean(gpu_clocks_before):6.0f} MHz")
        print(f"  Std:    {np.std(gpu_clocks_before):6.0f} MHz")
        print(f"  Min:    {np.min(gpu_clocks_before):6.0f} MHz")
        print(f"  Max:    {np.max(gpu_clocks_before):6.0f} MHz")

        # Correlation
        from scipy.stats import pearsonr
        corr, p_val = pearsonr(gpu_clocks_before, predict_times)
        print(f"\nCorrelation (GPU clock ↔ inference time): r = {corr:+.3f} (p = {p_val:.4f})")

        if abs(corr) > 0.5:
            print("  ⚠ STRONG CORRELATION - GPU clock speed is affecting inference time!")

    # Cleanup
    robot.home()
    robot.shutdown()
    manager.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received, shutting down...")
        try:
            rclpy.shutdown()
        except Exception:
            pass
