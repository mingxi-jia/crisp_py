"""Client classes for policy and point cloud processing servers."""

import base64
import numpy as np
import requests
import sys
import time
import torch
import dill
import hydra
from omegaconf import OmegaConf

sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy')
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply


class PolicyClient:
    """Client for communicating with the policy server."""

    def __init__(self, server_url: str = "http://localhost:5000", img_policy: bool = False):
        self.server_url = server_url
        self.img_policy = img_policy
        self.predict_contact = False  # Whether policy supports intervention prediction
        self._check_health()

    def _check_health(self):
        """Check if server is healthy and get policy capabilities."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                result = response.json()
                self.predict_contact = result.get('predict_contact', False)
                print(f"Connected to policy server successfully (predict_contact={self.predict_contact})")
            else:
                raise ConnectionError("Policy server unhealthy")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to policy server: {e}")
        
    def predict_intervention(self, obs_dict: dict) -> tuple[int, dict]:
        """Get intervention prediction from server using the policy.

        Args:
            obs_dict: Dictionary of observations (same format as predict_action)

        Returns:
            Tuple of (predicted_label, timing_dict)
        """
        # Encode observations as base64
        data = {}
        for key, value in obs_dict.items():
            if key in ['pcd_timestamp']:
                continue
            array_bytes = value.tobytes()
            array_b64 = base64.b64encode(array_bytes).decode('utf-8')
            data[key] = {
                'data': array_b64,
                'dtype': str(value.dtype),
                'shape': value.shape
            }

        # Send request
        response = requests.post(
            f"{self.server_url}/predict_intv",
            json=data,
            timeout=30
        )

        if response.status_code != 200:
            try:
                error_msg = response.json()
            except:
                error_msg = response.text
            raise RuntimeError(f"Server error (status {response.status_code}): {error_msg}")

        result = response.json()

        predicted_label = result['predicted_label']
        timing = result.get('timing', {})

        return predicted_label, timing

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
            if key in ['pcd_timestamp']:
                continue
            array_bytes = value.tobytes()
            array_b64 = base64.b64encode(array_bytes).decode('utf-8')
            data[key] = {
                'data': array_b64,
                'dtype': str(value.dtype),
                'shape': value.shape
            }

        # Add img_policy flag
        data['_img_policy'] = self.img_policy

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
            'depth_dict': {},
            'is_contact': None
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

        is_contact = result['is_contact']

        return processed_rgb, processed_depth, is_contact


class PinkIKClient:
    """Client for communicating with the Pink IK server."""

    def __init__(self, server_url: str = "http://localhost:5002"):
        self.server_url = server_url
        self._check_health()

    def _check_health(self):
        """Check if server is healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print("Connected to Pink IK server successfully")
            else:
                raise ConnectionError("Pink IK server unhealthy")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Pink IK server: {e}")

    def solve_ik(self, position: np.ndarray, orientation_quat: np.ndarray, q_init: np.ndarray = None) -> tuple[np.ndarray, bool]:
        """Solve IK for a single pose.

        Args:
            position: End-effector position [x, y, z]
            orientation_quat: Orientation quaternion [qx, qy, qz, qw]
            q_init: Initial joint configuration for warm-starting (optional)

        Returns:
            Tuple of (joint_config, success)
            - joint_config: Joint configuration (7 DOF)
            - success: Whether IK converged
        """
        payload = {
            "position": position.tolist() if isinstance(position, np.ndarray) else position,
            "orientation": orientation_quat.tolist() if isinstance(orientation_quat, np.ndarray) else orientation_quat,
        }
        if q_init is not None:
            payload["q_init"] = q_init.tolist() if isinstance(q_init, np.ndarray) else q_init

        response = requests.post(f"{self.server_url}/solve_ik", json=payload, timeout=10)

        if response.status_code != 200:
            raise RuntimeError(f"Pink IK server error: {response.text}")

        result = response.json()
        return np.array(result.get("q", [])), result.get("success", False), result.get("timing", {})

    def solve_trajectory(self, poses: list[tuple[np.ndarray, np.ndarray]], q_init: np.ndarray = None) -> tuple[list[np.ndarray], bool]:
        """Solve IK for a trajectory of poses (batch processing with warm-starting).

        Args:
            poses: List of (position, orientation_quat) tuples
            q_init: Initial joint configuration for first pose (optional)

        Returns:
            Tuple of (joint_trajectory, success)
            - joint_trajectory: List of joint configurations
            - success: Whether all IK solutions converged
        """
        pose_dicts = []
        for pos, quat in poses:
            pose_dicts.append({
                "position": pos.tolist() if isinstance(pos, np.ndarray) else pos,
                "orientation": quat.tolist() if isinstance(quat, np.ndarray) else quat,
            })

        payload = {"poses": pose_dicts}
        if q_init is not None:
            payload["q_init"] = q_init.tolist() if isinstance(q_init, np.ndarray) else q_init

        response = requests.post(f"{self.server_url}/solve_trajectory", json=payload, timeout=30)

        if response.status_code != 200:
            raise RuntimeError(f"Pink IK server error: {response.text}")

        result = response.json()
        trajectory = [np.array(q) for q in result.get("trajectory", [])]
        return trajectory, result.get("success", False)


class DirectPolicyWrapper:
    """Direct policy wrapper that mimics PolicyClient interface but runs policy locally.

    This is useful for debugging to bypass server communication overhead.
    """

    def __init__(self, ckpt_path: str, img_policy: bool = False):
        """Initialize policy directly from checkpoint.

        Args:
            ckpt_path: Path to policy checkpoint file
            img_policy: Whether to use image-only policy (excludes is_contact)
        """
        self.ckpt_path = ckpt_path
        self.img_policy = img_policy
        self.policy = None
        self.device = None
        self._initialize_policy()

    def _initialize_policy(self):
        """Initialize the diffusion policy model."""
        print(f"Loading policy directly from {self.ckpt_path}")
        payload = torch.load(open(self.ckpt_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']

        if 'task_name' not in cfg.policy:
            OmegaConf.set_struct(cfg.policy, False)
            cfg.policy.task_name = "realworld"
            OmegaConf.set_struct(cfg.policy, True)
        # Force is_simulation=False for real-robot evaluation
        OmegaConf.set_struct(cfg.policy, False)
        cfg.policy.is_simulation = False
        OmegaConf.set_struct(cfg.policy, True)

        cfg.logging.resume = False
        cfg.logging.mode = 'offline'
        if hasattr(cfg, 'real_robot_eval'):
            cfg.real_robot_eval = True
        if hasattr(cfg, 'policy'):
            if hasattr(cfg.policy, 'predict_contact'):
                delattr(cfg.policy, 'predict_contact')
            if hasattr(cfg.policy, 'control_mode'):
                delattr(cfg.policy, 'control_mode')

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, real_robot_eval=True)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        self.policy = workspace.ema_model
        self.device = torch.device('cuda')
        self.policy.eval()
        self.policy.to(self.device)
        self.policy.num_inference_steps = 20
        self.policy.n_action_steps = 8
        self.policy.reset()

        self.predict_contact = True

        print("Policy initialized successfully (direct mode)")

    def predict_action(self, obs_dict: dict) -> np.ndarray:
        """Get action prediction directly from policy.

        Args:
            obs_dict: Dictionary of numpy array observations

        Returns:
            Action array
        """
        with torch.no_grad():
            # Convert numpy observations to torch tensors
            # if not self.img_policy:
            #     obs_dict['is_contact'] = np.array([0], dtype=np.float32)
            if self.img_policy:
                # img_policy: frames already stacked with time dim, only add batch dim
                obs_dict_torch = dict_apply(obs_dict,
                    lambda x: torch.from_numpy(x.copy()).unsqueeze(0).to(self.device))
            else:
                obs_dict_torch = dict_apply(obs_dict,
                    lambda x: torch.from_numpy(x.copy()).unsqueeze(0).unsqueeze(1).to(self.device))

            # Run inference
            result = self.policy.predict_action(obs_dict_torch)

            # Convert back to numpy
            action = result['action'][0].detach().to('cpu').numpy()

        return action
    
    def predict_intervention(self, obs_dict: dict):
        with torch.no_grad():
            obs_dict_torch = dict_apply(obs_dict,
                    lambda x: torch.from_numpy(x[None, None, ...].copy()).to(self.device))
            result = self.policy.predict_intervention(obs_dict_torch)
            intv = result.to('cpu').numpy()[0]
            return intv, None

    def reset(self):
        """Reset the policy."""
        self.policy.reset()