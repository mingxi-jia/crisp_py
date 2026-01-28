"""Try to follow a "figure eight" target on the yz plane."""


import matplotlib.pyplot as plt
import numpy as np
import time
from crisp_py.robot import Robot
import h5py
import sys

from scipy.spatial.transform import Rotation as R
from go import setup_robot
from diff_eval_utils.diffusion_constants import DPEvalConfig
from diff_eval_utils.diffusion_transforms import convert_action_from_fingertip_to_gripper

config = DPEvalConfig()

sys.path.append(config.diffusion_policy_path)

from diffusion_policy.model.common.rotation_transformer import RotationTransformer

rot6d_to_mat = RotationTransformer('rotation_6d', 'matrix')
mat_to_rot6d = RotationTransformer('matrix', 'rotation_6d')

left_arm, gripper = setup_robot(config)
homing_pose = left_arm.end_effector_pose.copy()

hdf5_file_path = "/media/mingxi/T7/XEMB_Experiment/coffee_prep/replay_hand_test/test_2.hdf5"
hdf5_file_path = "/media/mingxi/T7/XEMB_Experiment/nutella_sort/test_nutella.hdf5"

# Gate the magenitude differences across frames
dataset = h5py.File(hdf5_file_path, "r")
data = dataset['data']['demo_0']
hand_pos = data['obs']['robot0_eef_pos'][:]
hand_quat = data['obs']['robot0_eef_quat'][:]
hand_grasp = data['obs']['robot0_gripper_qpos'][:, 0]
hand_poses = np.hstack((hand_pos, hand_quat)).astype(float)
# pose_folder = "/home/mingxi/mingxi_ws/crisp/crisp_py/raw_datasets/episodes/episode_20260123_174009_768/state"
# hand_poses = np.load(f"{pose_folder}/pose_wrt_world.npy", allow_pickle=True)[()].astype(float)
# hand_grasp = np.load(f"{pose_folder}/grasp.npy", allow_pickle=True)
assert len(hand_poses) == len(hand_grasp), f"Length mismatch: {len(hand_poses)} vs {len(hand_grasp)}"
print(hand_poses)
# %%
# The move_to function will publish a pose to /target_pose while interpolation linearly
# center = np.array([0.4, 0.0, 0.4])
ctrl_freq = 5.0
# left_arm.move_to(position=center, speed=0.15)
left_arm.move_to(position=hand_poses[0][:3], speed=0.15)

t = 0.0
target_pose = left_arm.end_effector_pose.copy()
arm_rate = left_arm.node.create_rate(ctrl_freq)
gripper_rate = gripper.node.create_rate(ctrl_freq)

# load hand pose
i=0
prev_grasp_value = hand_grasp[0]
for time_step, hand_pose in enumerate(hand_poses):
    print(f"Moving to pose {hand_pose}")
    action = np.concatenate([
        hand_pose[:3], 
        mat_to_rot6d.forward(
            R.from_quat(hand_pose[3:7]).as_matrix().reshape(1, 3, 3).astype(np.float32)
        )[0],
        [hand_grasp[time_step]]
    ])
    action_converted, gripper_rotation = convert_action_from_fingertip_to_gripper(
            action, rot6d_to_mat, ret_orig=False, clip=False
        )

    target_pose.position = action_converted[:3]
    target_pose.orientation = gripper_rotation

    # robot.move_to(position=np.array([x, y, z]), speed=0.15)
    left_arm.set_target(pose=target_pose)
    arm_rate.sleep()

    grasp_value = np.round(np.clip(hand_grasp[time_step], 0, 1))
    if grasp_value != prev_grasp_value:
        gripper.set_target(1-grasp_value)
        gripper_rate.sleep()
        time.sleep(1.0)  # wait for gripper to move (due to franka driver limitation)
    prev_grasp_value = grasp_value


# t = 0.0
# while t < 1.0:
#     # Just wait a bit for the end effector to settle
#     rate.sleep()
#     t += 1.0 / ctrl_freq


left_arm.home()

# %%
left_arm.shutdown()
