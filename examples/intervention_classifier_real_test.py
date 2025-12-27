from crisp_py.spacemouse import Spacemouse
import time
import sys
import threading

import rclpy
from std_msgs.msg import Int32MultiArray

from crisp_py.robot import Robot, Pose
from crisp_py.gripper.gripper import Gripper, GripperConfig
from crisp_py.camera.pointcloud import PointCloudManager

from scipy.spatial.transform import Rotation as R

import numpy as np

from pynput import keyboard

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """
        ResNet-based binary classifier.

        Args:
            num_classes: Number of output classes (default: 2 for binary classification)
            pretrained: Whether to use pretrained weights
        """
        super(ResNetClassifier, self).__init__()

        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def resize_image(image: np.ndarray) -> np.ndarray:
    target_size = 84
    h, w = image.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    cropped = image[top:top+min_dim, left:left+min_dim]
    if image.ndim == 2:  # depth image (single channel)
        resized = np.array(Image.fromarray(cropped).resize((target_size, target_size), Image.BILINEAR))
    elif image.shape[2] == 1:
        resized = np.array(Image.fromarray(cropped.squeeze(-1)).resize((target_size, target_size), Image.BILINEAR))
        resized = resized[:, :, None]

    else:  # RGB image (3 channels)
        resized = np.array(Image.fromarray(cropped).resize((target_size, target_size), Image.BILINEAR))
    return resized

def predict_image(image, model, device='cuda'):
    """
    Predict the label for a single image.

    Args:
        image: Input image as numpy array (H, W, C) with values in [0, 255] (uint8)
               or [0, 1] (float32), or as a PyTorch tensor
        model: Trained ResNet classifier
        device: Device to run inference on

    Returns:
        predicted_label: Predicted class (0 or 1)
        confidence: Confidence score for the predicted class
        probabilities: Dictionary with probabilities for both classes
    """
    image = resize_image(image)
    model.eval()

    # Convert numpy array to tensor if needed
    if isinstance(image, np.ndarray):
        # Convert to float32 and normalize to [0, 1] if needed
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Convert from (H, W, C) to (C, H, W) if needed
        if len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
            image = np.transpose(image, (2, 0, 1))

        # Convert to tensor
        image = torch.from_numpy(image).float()

    # Apply ImageNet normalization (same as training)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std

    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_label = output.argmax(1).item()
        confidence = probabilities[0, predicted_label].item()

    # Create probability dictionary
    prob_dict = {
        0: probabilities[0, 0].item(),
        1: probabilities[0, 1].item()
    }

    return predicted_label, confidence, prob_dict

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the saved model checkpoint (.pth file)
        device: Device to load the model on ('cuda' or 'cpu')

    Returns:
        model: Loaded ResNet classifier ready for inference
    """
    # Create model architecture
    model = ResNetClassifier(num_classes=2, pretrained=False)

    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    return model


def camera_visualization_thread(manager, model, camera_name='cam4', device='cuda', window_name='Wrist Camera - Intervention Prediction'):
    """
    Thread function for continuous camera visualization with intervention prediction.

    Args:
        manager: PointCloudManager to capture frames from
        model: Loaded intervention classifier model
        camera_name: Name of the camera to visualize (default: 'cam4' for wrist camera)
        device: Device to run inference on
        window_name: Name of the OpenCV window
    """
    print(f"Starting camera visualization thread for {camera_name}...")

    while True:
        try:
            # Capture frame from specified camera
            rgb, depth = manager.get_latest_rgbd(camera_name)
            if rgb is None:
                time.sleep(0.1)
                continue

            # Run prediction
            predicted_label, confidence, prob_dict = predict_image(rgb, model, device)

            # Create display frame
            display_frame = rgb.copy()

            # Add prediction text
            label_text = "INTERVENTION" if predicted_label == 1 else "NO INTERVENTION"
            color = (0, 0, 255) if predicted_label == 1 else (0, 255, 0)  # Red for intervention, green for no intervention

            # Add text to image
            cv2.putText(display_frame, f"Prediction: {label_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_frame, f"Confidence: {confidence:.2%}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_frame, f"P(No Int): {prob_dict[0]:.2%}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"P(Int): {prob_dict[1]:.2%}", (10, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display (convert RGB to BGR for OpenCV)
            cv2.imshow(window_name, cv2.cvtColor(display_frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error in camera visualization: {e}")
            time.sleep(0.1)

    cv2.destroyAllWindows()


def main():
    intervention_classifier_path = "examples/best_intervention_classifier.pth"
    ctrl_freq = 10.0  # Hz
    action_scale = 0.01  # 1 cm per action unit
    camera_name = 'cam4'  # Wrist camera name
    config_path = "config/camera_info.yaml"  # Camera config path

    # Initialize ROS2
    rclpy.init()

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
    robot = Robot(namespace="")
    robot.wait_until_ready()
    print(f"Robot ready. Current joint values: {robot.joint_values}")
    print(robot.end_effector_pose)
    print(robot.joint_values)
    print("Going to home position...")
    robot.home()

    # exit()

    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.cartesian_controller_parameters_client.load_param_config(
        file_path="config/control/spacemouse_cartesian_impedance.yaml"
    )
    time.sleep(2.0)

    # print("Going to start position...")
    # home_pose = Pose(position=np.array([0.6, 0., 0.35]), orientation=R.from_euler('XYZ', [np.pi, 0, 0]))
    # robot.move_to(pose=home_pose, speed=0.15)

    # Initialize gripper
    gripper_config = GripperConfig.from_yaml("./config/gripper_right.yaml")
    gripper = Gripper(gripper_config=gripper_config, namespace="/right/gripper")
    gripper.wait_until_ready()
    gripper.set_target(1.0)
    print("Gripper ready")

    # Initialize point cloud manager for camera access
    print(f"Loading point cloud manager with config: {config_path}")
    manager = PointCloudManager(config_path)

    # Spin manager in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(manager,), daemon=True)
    spin_thread.start()
    print("Point cloud manager ready")

    # Load intervention classifier model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    intervention_model = load_model_from_checkpoint(intervention_classifier_path, device=device)
    print(f"Intervention classifier loaded on {device}")

    # Start camera visualization thread
    viz_thread = threading.Thread(
        target=camera_visualization_thread,
        args=(manager, intervention_model, camera_name, device),
        daemon=True
    )
    viz_thread.start()
    print(f"Camera visualization thread started for {camera_name}")

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
    print("\nCamera visualization window shows live intervention predictions")
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

                    robot.move_to(pose=home_pose, speed=0.15)

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
            spacemouse_msg.data = [dx_int, dy_int, dz_int, droll_int, dpitch_int, dyaw_int, gripper_toggle, 0]
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
    manager.destroy_node()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("Shutdown complete.")
        sys.exit(0)
