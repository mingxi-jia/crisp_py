"""Client classes for policy and point cloud processing servers."""

import base64
import numpy as np
import requests


class PolicyClient:
    """Client for communicating with the policy server."""

    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url
        self._check_health()

    def _check_health(self):
        """Check if server is healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print("Connected to policy server successfully")
            else:
                raise ConnectionError("Policy server unhealthy")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to policy server: {e}")

    def predict_action(self, obs_dict: dict) -> np.ndarray:
        """Get action prediction from server.

        Args:
            obs_dict: Dictionary of observations

        Returns:
            Action array
        """
        # Encode observations as base64
        data = {}
        for key, value in obs_dict.items():
            array_bytes = value.tobytes()
            array_b64 = base64.b64encode(array_bytes).decode('utf-8')
            data[key] = {
                'data': array_b64,
                'dtype': str(value.dtype),
                'shape': value.shape
            }

        # Send request
        response = requests.post(
            f"{self.server_url}/predict",
            json=data,
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.json()}")

        result = response.json()

        # Decode action
        action_bytes = base64.b64decode(result['action']['data'])
        action = np.frombuffer(action_bytes, dtype=result['action']['dtype'])
        action = action.reshape(result['action']['shape'])

        return action

    def reset(self):
        """Reset the policy."""
        response = requests.post(f"{self.server_url}/reset", timeout=5)
        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.json()}")


class PcdProcessingClient:
    """Client for communicating with the point cloud processing server."""

    def __init__(self, server_url: str = "http://localhost:5001"):
        self.server_url = server_url
        self._check_health()

    def _check_health(self):
        """Check if server is healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print("Connected to PCD processing server successfully")
            else:
                raise ConnectionError("PCD processing server unhealthy")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to PCD processing server: {e}")

    def process_pcd(self, pcd: np.ndarray, eef_pose: np.ndarray, joint_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process point cloud on the server.

        Args:
            pcd: Raw point cloud array
            eef_pose: End-effector pose
            joint_state: Joint state array

        Returns:
            Tuple of (processed_pcd, render_pcd)
        """
        # Encode inputs as base64
        data = {
            'pcd': {
                'data': base64.b64encode(pcd.tobytes()).decode('utf-8'),
                'dtype': str(pcd.dtype),
                'shape': pcd.shape
            },
            'eef_pose': {
                'data': base64.b64encode(eef_pose.tobytes()).decode('utf-8'),
                'dtype': str(eef_pose.dtype),
                'shape': eef_pose.shape
            },
            'joint_state': {
                'data': base64.b64encode(joint_state.tobytes()).decode('utf-8'),
                'dtype': str(joint_state.dtype),
                'shape': joint_state.shape
            }
        }

        # Send request
        response = requests.post(
            f"{self.server_url}/process_pcd",
            json=data,
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.json()}")

        result = response.json()

        # Decode processed point clouds
        processed_pcd_bytes = base64.b64decode(result['processed_pcd']['data'])
        processed_pcd = np.frombuffer(processed_pcd_bytes, dtype=result['processed_pcd']['dtype'])
        processed_pcd = processed_pcd.reshape(result['processed_pcd']['shape'])

        render_pcd_bytes = base64.b64decode(result['render_pcd']['data'])
        render_pcd = np.frombuffer(render_pcd_bytes, dtype=result['render_pcd']['dtype'])
        render_pcd = render_pcd.reshape(result['render_pcd']['shape'])

        return processed_pcd, render_pcd

    def process_images(self, rgb_dict: dict, depth_dict: dict) -> tuple[dict, dict]:
        """Process RGB and depth images on the server.

        Args:
            rgb_dict: Dictionary of RGB images
            depth_dict: Dictionary of depth images

        Returns:
            Tuple of (processed_rgb_dict, processed_depth_dict)
        """
        # Encode inputs as base64
        data = {
            'rgb_dict': {},
            'depth_dict': {}
        }

        for cam_name, img in rgb_dict.items():
            data['rgb_dict'][cam_name] = {
                'data': base64.b64encode(img.tobytes()).decode('utf-8'),
                'dtype': str(img.dtype),
                'shape': img.shape
            }

        for cam_name, img in depth_dict.items():
            data['depth_dict'][cam_name] = {
                'data': base64.b64encode(img.tobytes()).decode('utf-8'),
                'dtype': str(img.dtype),
                'shape': img.shape
            }

        # Send request
        response = requests.post(
            f"{self.server_url}/process_images",
            json=data,
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.json()}")

        result = response.json()

        # Decode processed images
        processed_rgb = {}
        processed_depth = {}

        for cam_name, img_data in result['rgb_dict'].items():
            img_bytes = base64.b64decode(img_data['data'])
            processed_rgb[cam_name] = np.frombuffer(img_bytes, dtype=img_data['dtype']).reshape(img_data['shape'])

        for cam_name, img_data in result['depth_dict'].items():
            img_bytes = base64.b64decode(img_data['data'])
            processed_depth[cam_name] = np.frombuffer(img_bytes, dtype=img_data['dtype']).reshape(img_data['shape'])

        return processed_rgb, processed_depth
