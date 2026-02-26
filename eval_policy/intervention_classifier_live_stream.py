import time
import sys
import threading

import rclpy

from crisp_py.camera.pointcloud import PointCloudManager

import numpy as np

import torch
import torch.nn as nn
from torchvision import models
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error in camera visualization: {e}")
            time.sleep(0.1)

    cv2.destroyAllWindows()


def main():
    intervention_classifier_path = "examples/best_intervention_classifier.pth"
    intervention_classifier_path = "eval/best_intervention_classifier_top_left_corner.pth"
    intervention_classifier_path = "eval_policy/best_intervention_classifier_left_side.pth"
    intervention_classifier_path = "/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/best_intervention_classifier_nutella_d0.pth"
    intervention_classifier_path = "/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/best_intervention_classifier_nutella_d0_extended.pth"
    intervention_classifier_path = "/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d0/best_intervention_classifier_nutella_d0_15_modified_all.pth"

    camera_name = 'cam4'  # Wrist camera name
    config_path = "/home/mingxi/mingxi_ws/handpi/diffusion_policy/robotool/robot_configs/camera_info.yaml"  # Camera config path

    # Initialize ROS2
    rclpy.init()

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

    print("\n" + "=" * 50)
    print("Live Intervention Prediction Stream")
    print("=" * 50)
    print(f"\nCamera: {camera_name}")
    print("Controls:")
    print("  Q - Quit (when focused on window)")
    print("  Ctrl+C - Exit")
    print("=" * 50 + "\n")

    # Run camera visualization in main thread
    try:
        camera_visualization_thread(manager, intervention_model, camera_name, device)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")

    # Cleanup
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
