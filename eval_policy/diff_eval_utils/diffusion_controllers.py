"""Controller hierarchy for different diffusion policy execution modes."""

from diff_eval_utils.controllers.base_controller import RobotController
from diff_eval_utils.controllers.simple_controller import SimpleSequentialController
from diff_eval_utils.controllers.controlnet_controller import ControlNetController
from diff_eval_utils.controllers.intervention_controller import InterventionController
from diff_eval_utils.controllers.teleop_controller import TeleopController
from diff_eval_utils.controllers.chunking_controller import ChunkingController
from diff_eval_utils.controllers.blending_controller import BlendingChunkingController
from diff_eval_utils.controllers.gello_controller import GelloController


def create_controller(mode: str, *args, **kwargs) -> RobotController:
    """Factory function to create controller based on mode.

    Args:
        mode: Controller mode ('simple', 'chunking', 'blending', 'intv', 'controlnet', 'teleop', 'gello')
        *args, **kwargs: Arguments passed to controller constructor

    Returns:
        RobotController instance
    """
    if mode == 'simple':
        return SimpleSequentialController(*args, **kwargs)
    elif mode == 'chunking':
        return ChunkingController(*args, **kwargs)
    elif mode == 'blending':
        return BlendingChunkingController(*args, **kwargs)
    elif mode == 'intv':
        return InterventionController(*args, **kwargs)
    elif mode == 'controlnet':
        return ControlNetController(*args, **kwargs)
    elif mode == 'teleop':
        return TeleopController(*args, **kwargs)
    elif mode == 'gello':
        return GelloController(*args, **kwargs)
    else:
        raise ValueError(f"Unknown controller mode: {mode}")
