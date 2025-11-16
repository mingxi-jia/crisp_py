"""Try to follow a "figure eight" target on the yz plane."""

# %%
import matplotlib.pyplot as plt
import numpy as np
import time
from crisp_py.robot import Robot

left_arm = Robot(namespace="")
left_arm.wait_until_ready()

# %%
print(left_arm.end_effector_pose)
print(left_arm.joint_values)

# %%
print("Going to home position...")
left_arm.home()
homing_pose = left_arm.end_effector_pose.copy()

# %%
left_arm.controller_switcher_client.switch_controller("cartesian_impedance_controller")
left_arm.cartesian_controller_parameters_client.load_param_config(
    # file_path="config/control/gravity_compensation.yaml"
    # file_path="config/control/default_operational_space_controller.yaml"
    # file_path="config/control/clipped_cartesian_impedance.yaml"
    file_path="config/control/default_cartesian_impedance.yaml"
)

# set gripper
from crisp_py.gripper.gripper import Gripper, GripperConfig

gripper_config = GripperConfig.from_yaml("./config/gripper_right.yaml")
gripper = Gripper(gripper_config=gripper_config, namespace="/right/gripper")
gripper.wait_until_ready()
gripper.set_target(1.0)

hand_poses = np.load("examples/hand_poses_wrt_world.npy", allow_pickle=True)[()]
hand_grasp = np.load("examples/grasp.npy", allow_pickle=True)[:-1]
assert len(hand_poses) == len(hand_grasp), f"Length mismatch: {len(hand_poses)} vs {len(hand_grasp)}"
print(hand_grasp)
# %%
# The move_to function will publish a pose to /target_pose while interpolation linearly
center = np.array([0.4, 0.0, 0.4])
ctrl_freq = 10.0
left_arm.move_to(position=center, speed=0.15)
left_arm.move_to(position=hand_poses['000000'][:3], speed=0.15)

print("Starting to draw a circle...")
t = 0.0
target_pose = left_arm.end_effector_pose.copy()
arm_rate = left_arm.node.create_rate(ctrl_freq)
gripper_rate = gripper.node.create_rate(ctrl_freq)

# load hand pose
i=0
prev_grasp_value = hand_grasp[0]
for time_step, hand_pose in hand_poses.items():
    print(f"Moving to pose {hand_pose}")
    x, y, z = hand_pose[:3]
    z = np.clip(z-0.02, 0.06, 0.6)
    target_pose.position = np.array([x, y, z])

    # left_arm.move_to(position=np.array([x, y, z]), speed=0.2)
    left_arm.set_target(pose=target_pose)
    arm_rate.sleep()

    grasp_value = hand_grasp[i]
    if grasp_value != prev_grasp_value:
        print(f"Setting gripper to {grasp_value}")
        gripper.set_target(1-grasp_value)
        gripper_rate.sleep()
        time.sleep(1.0)  # wait for gripper to move

    t += 1.0 / ctrl_freq
    i += 1

    prev_grasp_value = grasp_value

    if time_step == '000075':
        break

# t = 0.0
# while t < 1.0:
#     # Just wait a bit for the end effector to settle
#     rate.sleep()
#     t += 1.0 / ctrl_freq


left_arm.home()

# %%
left_arm.shutdown()
