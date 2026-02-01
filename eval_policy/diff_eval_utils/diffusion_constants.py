"""Constants and configuration for diffusion policy control."""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# Robot-specific constants
FINGER_HAND_OFFSET = 0.10  # Distance from finger tip to gripper base along z-axis
# FINGER_HAND_OFFSET = 0.08  # Distance from finger tip to gripper base along z-axis

GRIPPER_NORM_CONST = 0.05  # Normalization constant for gripper value
START_POSITION = np.array([0.55, 0., 0.25 + FINGER_HAND_OFFSET])

ROBOTIQ_ROTATION_OFFSET = np.array([0, 0, np.pi / 4])  # Rotation offset for Robotiq gripper
# ROBOTIQ_ROTATION_OFFSET = np.array([0, 0, 0])  # Rotation offset for Robotiq gripper


@dataclass
class DPEvalConfig:
    """Configuration for diffusion policy control modes."""

    # Server configuration
    policy_server_port: int = 5000
    pcd_server_port: int = 5001
    ik_server_port: int = 5002

    home_joint_position: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0015795138042423453, 0.11460111156789562, 0.00012723805921852443, -1.9088541631957334, 0.007562769235242739, 2.16821183580947, 0.0]
        )
    )

    # Safety bounds
    eef_bounds: dict = field(
        default_factory=lambda: {
            'x': (0.3, 0.8),
            'y': (-0.35, 0.35),
            'z': (0.0, 0.61),
        }
    )
    joint_delta_threshold: float = 0.2  # Max allowed deviation from home pose (rad)
    n_steps: int = 20

    # Control parameters
    ctrl_freq: float = 14
    n_interpolation: int = 6
    joint_ctrl_freq: float = 24

    # The Action Scale --> 
    # unit_action = nutella_sort dataset action,
    # unit_action Freq = joint_ctrl_freq / n_interpolation 

    # These params are working perfectly with the current joint controller
    n_steps_per_unit_action: int = 20
    joint_ctrl_freq: float = 100
    n_interpolation = n_steps_per_unit_action - 1  

    joint_exec_time_tolerance: float = 10.0 # 10 ms tolerance for the delay outside execute action

    teleop: dict = field(default_factory=lambda: {"n_interpolation": 0})

    # Spacemouse action scale
    spacemouse_action_size: float = 0.007
    spacemouse_action_scaling_factor: np.ndarray = field(
        default_factory=lambda: np.array([1, 1, 1, 2, -0.5, -2])
    )
    spacemouse_deadzone: float = 0.1

    # Controller mode: 'simple', 'chunking', 'blending', 'intv'
    mode: str = 'simple'

    # Control space: 'joint' or 'cartesian'
    ctrl_space: str = 'cartesian'

    horizon: int = 16

    # Chunking parameters (used when mode != 'simple')
    policy_delay: int = 2
    action_exec_size: int = 8


    # Blending parameters (only when mode == 'blending')
    merge_range: int = 4

    # Debug options
    debug_plotting: bool = False
    debug_output_dir: Path = field(default_factory=lambda: Path("debug_plots"))
    visualize: bool = False
    img_policy: bool = False

    # Debug data saving
    save_debug_data: bool = False
    save_pcd_format: str = 'npy'  # 'npy' or 'ply'

    # External paths
    diffusion_policy_path: str = '/home/mingxi/mingxi_ws/handpi/diffusion_policy'
    toolbox_path: str = diffusion_policy_path + '/robotool'

    def __post_init__(self):
        """Validate configuration."""

        self.spacemouse_action_scale = self.spacemouse_action_size * self.spacemouse_action_scaling_factor

        valid_modes = ['simple', 'chunking', 'blending', 'intv', 'controlnet', 'teleop', 'gello', 'test', 'test_teleop']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {valid_modes}.")

        if self.ctrl_space not in ['joint', 'cartesian']:
            raise ValueError(f"Invalid ctrl_space: {self.ctrl_space}. Must be 'joint' or 'cartesian'.")

        if self.mode in ['chunking', 'blending']:
            if self.action_exec_size + self.policy_delay > self.horizon:
                raise ValueError(
                    f"action_exec_size ({self.action_exec_size}) + policy_delay ({self.policy_delay}) "
                    f"must be <= horizon ({self.horizon})"
                )

        if self.mode == 'blending':
            if self.merge_range > self.action_exec_size / 2:
                raise ValueError(
                    f"merge_range ({self.merge_range}) must be <= action_exec_size/2 ({self.action_exec_size/2})"
                )

    @classmethod
    def from_cli_args(cls, args):
        """Create config from argparse arguments."""
        return cls(
            mode=args.mode,
            n_steps=args.n_steps,
            debug_plotting=args.debug_plotting,
            policy_server_port=args.policy_port,
            pcd_server_port=args.pcd_port,
            ik_server_port=args.ik_port,
            visualize=args.visualize,
            img_policy=args.img_policy,
            ctrl_space=args.ctrl_space,
        )
