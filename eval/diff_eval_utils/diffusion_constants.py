"""Constants and configuration for diffusion policy control."""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# Robot-specific constants
FINGER_HAND_OFFSET = 0.06  # Distance from finger tip to gripper base along z-axis
GRIPPER_NORM_CONST = 0.05  # Normalization constant for gripper value
START_POSITION = np.array([0.60, 0., 0.32 + FINGER_HAND_OFFSET])

JOINT_THRESHOLDS = {
    "panda_link0": 0.08,
    "panda_link1": 0.08,
    "panda_link2": 0.08,
    "panda_link3": 0.08,
    "panda_link4": 0.08,
    "panda_link5": 0.08,
    "panda_link6": 0.08,
    "panda_link7": 0.08,
    "panda_link8": 0.08,
    "panda_hand": 0.02,
    "panda_leftfinger": 0.02,
    "panda_rightfinger": 0.02,
}


@dataclass
class DPEvalConfig:
    """Configuration for diffusion policy control modes."""

    # Server configuration
    policy_server_port: int = 5000
    pcd_server_port: int = 5001

    # Control parameters
    ctrl_freq: float = 10.0
    n_steps: int = 40

    # Controller mode: 'simple', 'chunking', 'blending'
    mode: str = 'simple'

    # Chunking parameters (used when mode != 'simple')
    policy_delay: int = 4
    horizon: int = 16
    action_exec_size: int = 10

    # Blending parameters (only when mode == 'blending')
    merge_range: int = 4

    # Debug options
    debug_plotting: bool = False
    debug_output_dir: Path = field(default_factory=lambda: Path("debug_plots"))
    visualize: bool = False

    # External paths
    toolbox_path: str = '/home/mingxi/mingxi_ws/handpi/robot-vision-toolbox'
    diffusion_policy_path: str = '/home/mingxi/mingxi_ws/handpi/diffusion_policy'

    def __post_init__(self):
        """Validate configuration."""
        if self.mode not in ['simple', 'chunking', 'blending']:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'simple', 'chunking', or 'blending'")

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
            ctrl_freq=args.ctrl_freq,
            debug_plotting=args.debug_plotting,
            policy_server_port=args.policy_port,
            pcd_server_port=args.pcd_port,
            visualize=args.visualize,
        )
