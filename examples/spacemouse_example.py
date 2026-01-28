from crisp_py.spacemouse import Spacemouse
import time
import sys
import threading

import rclpy
from std_msgs.msg import Int32MultiArray

from crisp_py.robot import Robot, Pose
from crisp_py.gripper.gripper import Gripper, GripperConfig

from scipy.spatial.transform import Rotation as R

import numpy as np

from pynput import keyboard
from eval.eval import setup_robot
from eval.diff_eval_utils.diffusion_constants import DPEvalConfig

def main():
    ctrl_freq = 10.0  # Hz
    action_scale = 0.015  # 1 cm per action unit

    config = DPEvalConfig()

    # Shared state for keyboard reset trigger
    reset_requested = {'flag': False}
    gripper_closed = {'value': False}
    lock = threading.Lock()

    def on_press(key):
        with lock:
            try:
                if key.char == 'r':
                    reset_requested['flag'] = True
            except AttributeError:
                pass

    def on_release(key):
        pass

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    ### ---- Robot Setup ----- ###
    robot, gripper = setup_robot(config)

    target_pose = robot.end_effector_pose
    target_xyz = np.array(target_pose.position)
    target_orientation = np.array(target_pose.orientation.as_euler('XYZ'))

    # Save initial start position and orientation for reset
    start_orientation = target_orientation.copy()
    arm_rate = robot.node.create_rate(ctrl_freq)
    gripper_rate = gripper.node.create_rate(ctrl_freq)
    prev_button_pressed = False

    # Create spacemouse signal publisher
    # Format: [dx, dy, dz, droll, dpitch, dyaw, gripper_toggle, reset]
    spacemouse_pub = robot.node.create_publisher(Int32MultiArray, '/teleop/signals', 30)

    print("\n" + "=" * 50)
    print("Spacemouse Control Active")
    print("=" * 50)
    print("\nControls:")
    print("  Spacemouse - 6DOF control (position + rotation)")
    print("  Button 0 - Toggle gripper open/close")
    print("  R - Reset to start position")
    print("  Ctrl+C - Exit")
    print("=" * 50 + "\n")
    with Spacemouse(deadzone=0.1) as sm:
        while True:
            with lock:
                # Check for reset request
                if reset_requested['flag']:
                    print("\n[RESET] Resetting to start position...")
                    reset_requested['flag'] = False
                    gripper.set_target(1.0)
                    gripper_closed['value'] = False
                    time.sleep(1.0)

                    robot.move_to(position=[0.6, 0.0, 0.3], pose=R.from_euler('XYZ', [0, np.pi/12, 0]), speed=0.1)

                    target_pose = robot.end_effector_pose
                    target_xyz = np.array(target_pose.position)
                    target_orientation = start_orientation.copy()
                    print("Reset complete!\n")
                    continue

            # Check for large deviation
            if np.linalg.norm(target_pose.position - robot.end_effector_pose.position) > 0.02:
                arm_rate.sleep()
                continue

            # Get action from spacemouse
            spacemouse_eef_action = sm.get_motion_state_transformed()
            button_pressed = sm.is_button_pressed(0)  # is pressed -> True
            dx, dy, dz, droll, dpitch, dyaw = spacemouse_eef_action * action_scale
            # droll, dpitch, dyaw = 0, 0, 0
            # Publish spacemouse signals
            # Format: [dx, dy, dz, droll, dpitch, dyaw, gripper_toggle, reset]
            spacemouse_msg = Int32MultiArray()
            gripper_toggle = 1 if (button_pressed and not prev_button_pressed) else 0
            # Convert float deltas to int (scaled by 100 to preserve precision)
            dx_int = (1 if dx > 0 else -1 if dx < 0 else 0)
            dy_int = (1 if dy > 0 else -1 if dy < 0 else 0)
            dz_int = (1 if dz > 0 else -1 if dz < 0 else 0)
            droll_int = (1 if droll > 0 else -1 if droll < 0 else 0)
            dpitch_int = (1 if dpitch > 0 else -1 if dpitch < 0 else 0)
            dyaw_int = (1 if dyaw > 0 else -1 if dyaw < 0 else 0)
            if dx_int != 0 or dy_int != 0 or dz_int != 0 or droll_int != 0 or dpitch_int != 0 or dyaw_int != 0:
                intv_state = 2
            else:
                intv_state = 0
            spacemouse_msg.data = [intv_state, dx_int, dy_int, dz_int, droll_int, dpitch_int, dyaw_int, gripper_toggle, 0]
            spacemouse_pub.publish(spacemouse_msg)

            if np.linalg.norm(target_pose.position - robot.end_effector_pose.position) > 0.05:
                arm_rate.sleep()
                continue
            

            # Apply position and rotation deltas
            curr_x, curr_y, curr_z = target_xyz
            curr_roll, curr_pitch, curr_yaw = target_orientation

            x = curr_x + dx
            y = curr_y + dy
            z = curr_z + dz
            roll = curr_roll + droll
            pitch = curr_pitch + dpitch
            yaw = curr_yaw - dyaw * 2

            # clip z to be above table height
            z = max(z, 0.02)

            target_xyz = np.array([x, y, z])
            target_orientation = np.array([roll, pitch, yaw])

            # Only print if there's movement
            if dx != 0 or dy != 0 or dz != 0 or droll != 0 or dpitch != 0 or dyaw != 0:
                print(f"x={x:.4f}\ty={y:.4f}\tz={z:.4f}\troll={roll:.4f}\tpitch={pitch:.4f}\tyaw={yaw:.4f}")

            target_pose.position = target_xyz
            target_pose.orientation = R.from_euler('XYZ', target_orientation)
            robot.set_target(pose=target_pose)
            arm_rate.sleep()

            # Handle gripper toggle
            if button_pressed and not prev_button_pressed:
                gripper_closed['value'] = not gripper_closed['value']
                gripper_target = 0.0 if gripper_closed['value'] else 1.0
                print(f"Gripper {'closing' if gripper_closed['value'] else 'opening'}...")
                gripper.set_target(gripper_target)
                gripper_rate.sleep()
                time.sleep(0.5)
            prev_button_pressed = button_pressed

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
            import rclpy
            rclpy.shutdown()
        except Exception:
            pass
        sys.exit(0)
