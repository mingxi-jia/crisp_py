from crisp_py.spacemouse import Spacemouse
import time


from crisp_py.robot import Robot
from crisp_py.gripper.gripper import Gripper, GripperConfig

import sys

from scipy.spatial.transform import Rotation as R

import numpy as np
import copy

from pynput import keyboard

def main():
    ctrl_freq = 10.0 # Hz
    action_scale = 0.02  # 1 cm per action unit

    # Shared state for keyboard reset trigger
    reset_requested = {'flag': False}

    def on_press(key):
        try:
            if key.char == 'x':
                reset_requested['flag'] = True
                print("\n[RESET] 'x' key pressed - resetting to start position...")
        except AttributeError:
            pass

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

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

    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.cartesian_controller_parameters_client.load_param_config(
        file_path="config/control/spacemouse_cartesian_impedance.yaml"
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

    target_pose = robot.end_effector_pose
    target_xyz = np.array(target_pose.position)
    target_orientation = np.array(target_pose.orientation.as_euler('XYZ'))

    # Save initial start position and orientation for reset
    start_position = np.array([0.5, 0., 0.3])
    start_orientation = target_orientation.copy()
    arm_rate = robot.node.create_rate(ctrl_freq)
    gripper_rate = gripper.node.create_rate(ctrl_freq)
    prev_grasp_value = 0
    
    print("Starting Spacemouse control loop...\n====================\n====================")
    print("Press 'r' to reset gripper and move to start position [0.5, 0., 0.3]")
    with Spacemouse(deadzone=0.3) as sm:
        while True:

            # Check for reset request
            if reset_requested['flag']:
                print("Resetting gripper to open...")
                gripper.set_target(1.0)
                time.sleep(1.0)

                print("Moving to start position and orientation...")
                robot.move_to(position=start_position, speed=0.15)

                # Update target pose and orientation
                target_pose = robot.end_effector_pose
                target_xyz = np.array(target_pose.position)
                target_orientation = start_orientation.copy()

                print("Reset complete!\n")
                reset_requested['flag'] = False
                prev_grasp_value = 0

            if np.linalg.norm(target_pose.position - robot.end_effector_pose.position) > 0.02:
                # print("Warning: Large deviation from target pose. Stopping control.")
                continue


            # Get action from sapcemouse
            spacemouse_eef_action = sm.get_motion_state_transformed()
            spacemouse_gripper_action = sm.is_button_pressed(0) # is pressed -> 1
            # print(f"{spacemouse_eef_action}, {spacemouse_gripper_action}")
            dx, dy, dz, droll, dpitch, dyaw = spacemouse_eef_action * action_scale
            # print(f"Spacemouse action: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}, droll={droll:.4f}, dpitch={dpitch:.4f}, dyaw={dyaw:.4f}, gripper_action={spacemouse_gripper_action}")

            curr_x, curr_y, curr_z = target_xyz
            curr_roll, curr_pitch, curr_yaw = target_orientation
            # print(f"Moving to position: x={curr_x:.4f}, y={curr_y:.4f}, z={curr_z:.4f}")
            # if (daction==0).all():
            x, y, z = curr_x + dx, curr_y + dy, curr_z + dz
            roll, pitch, yaw = curr_roll + droll, curr_pitch + dpitch, curr_yaw - dyaw*2
            target_xyz = x, y, z
            target_orientation = np.array([roll, pitch, yaw])
            print(f"x={x:.4f}\ty={y:.4f}\tz={z:.4f}\troll={roll:.4f}\tpitch={pitch:.4f}\tyaw={yaw:.4f}")
            # print(f"Moving to position: x={x:.4f}, y={y:.4f}, z={z:.4f}")
            # z = np.clip(z-0.02, 0.06, 0.6)
            target_pose.position = np.array([x, y, z])
            target_pose.orientation = R.from_euler('XYZ', target_orientation)
            robot.set_target(pose=target_pose)        
            arm_rate.sleep()

            grasp_value = np.clip(spacemouse_gripper_action, 0, 1)
            if grasp_value != prev_grasp_value:
                print(f"Setting gripper to {grasp_value}")
                gripper.set_target(1-grasp_value)
                gripper_rate.sleep()
                time.sleep(1.0)  # wait for gripper to move
            prev_grasp_value = grasp_value


            # # Get current state

            # if (action!=0).any():
            #     target_xyz = np.array(target_pose.position)
                # target_orientation = np.array(target_pose.orientation.as_euler('XYZ'))
            # gripper_val = gripper.value
            # target_pose = robot.end_effector_pose
            # gripper_state = not gripper.is_open()
            # curr_xyz = target_pose.position 
            # curr_orientation = target_pose.orientation.as_euler('XYZ')

        # Cleanup
        listener.stop()
        robot.home()
        robot.shutdown()

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
