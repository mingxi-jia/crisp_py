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


def main():
    ctrl_freq = 10.0  # Hz
    position_scale = 0.01  # 1 cm per key press
    rotation_scale = 0.05  # radians per key press

    # Shared state for keyboard input
    key_state = {
        # Position controls (WASD + QE)
        'w': False,  # +X (forward)
        's': False,  # -X (backward)
        'a': False,  # +Y (left)
        'd': False,  # -Y (right)
        # Z-axis controls (Arrow Up/Down)
        'up': False,    # +Z (up)
        'down': False,  # -Z (down)
        # Rotation controls (IJKL + UO)
        'i': False,  # +roll
        'k': False,  # -roll
        'j': False,  # +pitch
        'l': False,  # -pitch
        'u': False,  # +yaw
        'o': False,  # -yaw
        # Gripper control
        'space': False,  # toggle gripper
        # Reset
        'r': False,  # reset to start position
    }

    gripper_closed = {'value': False}
    lock = threading.Lock()

    def on_press(key):
        with lock:
            try:
                if hasattr(key, 'char') and key.char in key_state:
                    key_state[key.char] = True
            except AttributeError:
                pass
            if key == keyboard.Key.space:
                key_state['space'] = True
            if key == keyboard.Key.up:
                key_state['up'] = True
            if key == keyboard.Key.down:
                key_state['down'] = True

    def on_release(key):
        with lock:
            try:
                if hasattr(key, 'char') and key.char in key_state:
                    key_state[key.char] = False
            except AttributeError:
                pass
            if key == keyboard.Key.space:
                key_state['space'] = False
            if key == keyboard.Key.up:
                key_state['up'] = False
            if key == keyboard.Key.down:
                key_state['down'] = False

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    ### ---- Robot Setup ----- ###
    robot = Robot(namespace="")
    robot.wait_until_ready()
    print(f"Robot ready. Current joint values: {robot.joint_values}")
    print(robot.end_effector_pose)
    print(robot.joint_values)
    print("Going to home position...")
    robot.home()

    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.cartesian_controller_parameters_client.load_param_config(
        file_path="config/control/spacemouse_cartesian_impedance.yaml"
    )
    time.sleep(2.0)

    print("Going to start position...")
    home_pose = Pose(position=np.array([0.6, 0., 0.35]), orientation=R.from_euler('XYZ', [np.pi, 0, 0]))
    robot.move_to(pose=home_pose, speed=0.15)

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
    start_orientation = target_orientation.copy()
    arm_rate = robot.node.create_rate(ctrl_freq)
    gripper_rate = gripper.node.create_rate(ctrl_freq)
    prev_space_pressed = False

    # Create keyboard signal publisher
    # Format: [dx, dy, dz, droll, dpitch, dyaw, gripper_toggle, reset]
    keyboard_pub = robot.node.create_publisher(Int32MultiArray, '/teleop/signals', 30)

    print("\n" + "=" * 50)
    print("Keyboard Control Active")
    print("=" * 50)
    print("\nPosition Controls:")
    print("  W/S - Forward/Backward (X axis)")
    print("  A/D - Left/Right (Y axis)")
    print("  Arrow Up/Down - Up/Down (Z axis)")
    print("\nRotation Controls:")
    print("  I/K - Roll +/-")
    print("  J/L - Pitch +/-")
    print("  U/O - Yaw +/-")
    print("\nOther Controls:")
    print("  SPACE - Toggle gripper open/close")
    print("  R - Reset to start position")
    print("  Ctrl+C - Exit")
    print("=" * 50 + "\n")

    while True:
        with lock:
            # Check for reset request
            if key_state['r']:
                print("\n[RESET] Resetting to start position...")
                key_state['r'] = False
                gripper.set_target(1.0)
                gripper_closed['value'] = False
                time.sleep(1.0)

                robot.move_to(pose=home_pose, speed=0.15)

                target_pose = robot.end_effector_pose
                target_xyz = np.array(target_pose.position)
                target_orientation = start_orientation.copy()
                print("Reset complete!\n")
                continue

            # Calculate position delta from WASD + Arrow keys
            dx = (1 if key_state['s'] else 0) - (1 if key_state['w'] else 0)
            dy = (1 if key_state['d'] else 0) - (1 if key_state['a'] else 0)
            dz = (1 if key_state['up'] else 0) - (1 if key_state['down'] else 0)

            # Calculate rotation delta from IJKL + UO
            droll = (1 if key_state['i'] else 0) - (1 if key_state['k'] else 0)
            dpitch = (1 if key_state['j'] else 0) - (1 if key_state['l'] else 0)
            dyaw = (1 if key_state['u'] else 0) - (1 if key_state['o'] else 0)

            # Handle gripper toggle (on press, not hold)
            space_pressed = key_state['space']
            reset_pressed = key_state['r']

        # Publish keyboard signals
        # Format: [dx, dy, dz, droll, dpitch, dyaw, gripper_toggle, reset]
        keyboard_msg = Int32MultiArray()
        gripper_toggle = 1 if (space_pressed and not prev_space_pressed) else 0
        keyboard_msg.data = [dx, dy, dz, droll, dpitch, dyaw, gripper_toggle, int(reset_pressed)]
        keyboard_pub.publish(keyboard_msg)

        # Check for large deviation
        if np.linalg.norm(target_pose.position - robot.end_effector_pose.position) > 0.02:
            arm_rate.sleep()
            continue

        # Apply position and rotation deltas
        curr_x, curr_y, curr_z = target_xyz
        curr_roll, curr_pitch, curr_yaw = target_orientation

        x = curr_x + dx * position_scale
        y = curr_y + dy * position_scale
        z = curr_z + dz * position_scale
        roll = curr_roll + droll * rotation_scale
        pitch = curr_pitch + dpitch * rotation_scale
        yaw = curr_yaw + dyaw * rotation_scale

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
        if space_pressed and not prev_space_pressed:
            gripper_closed['value'] = not gripper_closed['value']
            gripper_target = 0.0 if gripper_closed['value'] else 1.0
            print(f"Gripper {'closing' if gripper_closed['value'] else 'opening'}...")
            gripper.set_target(gripper_target)
            gripper_rate.sleep()
            time.sleep(0.5)
        prev_space_pressed = space_pressed

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
