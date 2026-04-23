"""Constants and configuration for diffusion policy control."""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# Robot-specific constants
FINGER_HAND_OFFSET = 0.10  # Distance from finger tip to gripper base along z-axis
FINGER_HAND_OFFSET = 0.08   # Distance from finger tip to gripper base along z-axis
# FINGER_HAND_OFFSET = 0.06  # Distance from finger tip to gripper base along z-axis

GRIPPER_NORM_CONST = 0.05  # Normalization constant for gripper value
START_POSITION = np.array([0.53, 0.0, 0.22 + FINGER_HAND_OFFSET]) # nutella
# START_POSITION = np.array([0.53, 0.15, 0.22 + FINGER_HAND_OFFSET]) # coffee making
START_POSITION = np.array([0.53, 0.0, 0.27 + FINGER_HAND_OFFSET]) # nutella Temp
START_POSITION = np.array([0.53, 0.0, 0.32 + FINGER_HAND_OFFSET]) 
START_POSITION = np.array([0.53, 0.05, 0.32 + FINGER_HAND_OFFSET]) 
START_POSITION = np.array([0.53, 0.10, 0.32 + FINGER_HAND_OFFSET]) 

# START_POSITION = np.array([0.53, -0.05, 0.32 + FINGER_HAND_OFFSET]) # desk_cleanriiiiirriirri_up 02 08 Temp
MIN_Z = 0.02
# MIN_Z = 0.0


ROBOTIQ_ROTATION_OFFSET = np.array([0, 0, np.pi / 4])  # Rotation offset for Robotiq gripper
# ROBOTIQ_ROTATION_OFFSET = np.array([0, 0, 0])  # Rotation offset for Robotiq gripper


@dataclass
class DPEvalConfig:
    """Configuration for diffusion policy control modes."""

    # Server configuration
    policy_server_port: int = 5000
    pcd_server_port: int = 5001
    ik_server_port: int = 5002

    # home_joint_position: np.ndarray = field(
    #     default_factory=lambda: np.array(
    #         [0.003614710905754158, -0.11320468520732234, -0.0001438466713307361, -2.2916872754536546, 0.012506755483931742, 2.3405799781608208, 0.026043922792643427]
    #     )
    # ) # 0.53, 0.0, 0.22 + FINGER_HAND_OFFSET

    # home_joint_position: np.ndarray = field(
    #     default_factory=lambda: np.array(
    #         [0.0010592495343525858, -0.1855868288347739, 0.002223996584916949, -2.223416249304472, 0.015297965884714102, 2.2045127972035674, 0.012424510319882112]
    #     )
    # ) # 0.53, 0.0, 0.27 + FINGER_HAND_OFFSET

    # home_joint_position: np.ndarray = field(
    #     default_factory=lambda: np.array(
    #         [-0.00021036731172055314, -0.2114014540483003, 0.003076851296918837, -2.1276072564005877, 0.01883355535690998, 2.0901015289004694, -0.0007363878236070589]
    #     )
    # ) # 0.53, 0.0, 0.32 + FINGER_HAND_OFFSET

    home_joint_position: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.010627665550159485, -0.13752133567683, -0.00031197058307286284, -2.0999258051963947, -0.0026502434610639236, 2.1077846475228178, 0.020550094808096893]
        )
    ) # 0.53, 0.0, 0.32 + FINGER_HAND_OFFSET No Rotational Offset

    home_joint_position: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.020231719674765586, -0.13386455797511426, 0.07162067359927972, -2.096127669802142, 0.0001639917924899637, 1.9977632412655684, 0.0820641116711499]
        )
    ) # 0.53, 0.05, 0.32 + FINGER_HAND_OFFSET No Rotational Offset

    home_joint_position_1: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.05300830878672895, -0.12505050015354602, 0.13231264456667063, -2.0618399494036668, -0.0033282643974355017, 1.9778674369275133, 0.17595177801320064]
        )
    ) # 0.53, 0.10, 0.32 + FINGER_HAND_OFFSET


    


    # home_joint_position: np.ndarray = field(
    #     default_factory=lambda: np.array(
    #         [-0.026631299793012958, -0.20883834842587998, -0.06749117694827121, -2.1280879622930162, 0.008251831251039166, 2.0957079420315865, -0.09929068380960085]
    #     )
    # ) # 0.53, -0.05, 0.32 + FINGER_HAND_OFFSET




    # Safety bounds
    eef_bounds: dict = field(
        default_factory=lambda: {
            'x': (0.3, 0.8),
            'y': (-0.35, 0.35),
            'z': (MIN_Z, 0.61),
        }
    )
    joint_delta_threshold: float = 0.2  # Max allowed deviation from home pose (rad)
    n_steps: int = 9999

    # Control parameters
    ctrl_freq: float = 100
    n_interpolation: int = 6
    joint_ctrl_freq: float = 24

    # The Action Scale --> 
    # unit_action = nutella_sort dataset action,
    # unit_action Freq = joint_ctrl_freq / n_interpolation 

    # These params are ok with the current joint controller
    n_steps_per_unit_action: int = 8 # 15 for joint
    joint_ctrl_freq: float = 100
    n_interpolation = n_steps_per_unit_action - 1  

    n_steps_per_unit_hand_action: int = 20
    n_steps_per_unit_intv_action: int = 10

    # For coffee intv
    # n_steps_per_unit_action: int = 25
    # joint_ctrl_freq: float = 100
    # n_interpolation = n_steps_per_unit_action - 1  

    # For party host
    # n_steps_per_unit_action: int = 50 #  ~ 1 Hz
    # joint_ctrl_freq: float = 100
    # n_interpolation = n_steps_per_unit_action - 1  

    joint_exec_time_tolerance: float = 10.0 # 10 ms tolerance for the delay outside execute action

    teleop: dict = field(default_factory=lambda: {"n_interpolation": 0})

    # Spacemouse action scale
    spacemouse_action_size: float = 0.003
    spacemouse_action_scaling_factor: np.ndarray = field(
        default_factory=lambda: np.array([1, 1, 1, 2, -1, -3])
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

    # FT sensor
    ft_sensor_on: bool = False

    # Debug options
    debug_plotting: bool = False
    debug_output_dir: Path = field(default_factory=lambda: Path("debug_plots"))
    visualize: bool = False
    img_policy: bool = False
    flip_gripper_obs: bool = False
    policy_action_freq_override: str | None = None

    # Debug data saving
    save_debug_data: bool = False
    save_pcd_format: str = 'npy'  # 'npy' or 'ply'

    # External paths
    diffusion_policy_path: str = '/home/mingxi/mingxi_ws/handpi/diffusion_policy'
    toolbox_path: str = diffusion_policy_path + '/robotool'

    def __post_init__(self):
        """Validate configuration."""

        self.spacemouse_action_scale = self.spacemouse_action_size * self.spacemouse_action_scaling_factor

        valid_modes = ['simple', 'chunking', 'blending', 'intv', 'intv_party', 'controlnet', 'teleop', 'teleop_intv', 'gello', 'test', 'test_teleop']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {valid_modes}.")

        if self.ctrl_space not in ['joint', 'cartesian']:
            raise ValueError(f"Invalid ctrl_space: {self.ctrl_space}. Must be 'joint' or 'cartesian'.")

        valid_freq_overrides = [None, 'intv', 'hand']
        if self.policy_action_freq_override not in valid_freq_overrides:
            raise ValueError(
                f"Invalid policy_action_freq_override: {self.policy_action_freq_override}. "
                f"Must be one of {valid_freq_overrides}."
            )

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
            flip_gripper_obs=getattr(args, 'flip_gripper_obs', False),
            policy_action_freq_override='intv' if getattr(args, 'intv_freq', False) else 'hand' if getattr(args, 'hand_freq', False) else None,
        )
