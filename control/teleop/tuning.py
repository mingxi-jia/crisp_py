"""Strict, shared spacemouse tuning resolution.

The numeric teleop gains live in ``config/teleop.yaml``. Point
``$CRISP_TELEOP_CONFIG`` at another file to select a different tuning set.
"""

import os
from pathlib import Path

from control.ik import DifferentialIK, MAX_LEAD_M, MAX_LEAD_RAD
from control.teleop.spacemouse_policy import SpacemouseConfig


TELEOP_CONFIG_FILE = Path(os.environ.get(
    "CRISP_TELEOP_CONFIG",
    Path(__file__).resolve().parents[2] / "config" / "teleop.yaml"))


def _required(mapping: dict, key: str, path: Path, section: str):
    try:
        return mapping[key]
    except KeyError as e:
        raise KeyError(f"{path}: {section} needs {key}; teleop gains must be "
                       "configured explicitly, not silently assumed") from e


def load_teleop_config(path=None) -> dict:
    """Read the complete teleop tuning YAML, raising on absent configuration."""
    import yaml

    path = Path(path or TELEOP_CONFIG_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"teleop tuning file not found: {path}. It holds the gains every "
            f"spacemouse caller uses; set $CRISP_TELEOP_CONFIG if it lives elsewhere.")
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"{path}: teleop tuning must be a YAML mapping")

    _required(config, "deadzone", path, "top level")
    brake = _required(config, "brake", path, "top level")
    if not isinstance(brake, dict):
        raise ValueError(f"{path}: brake must be a mapping")
    _required(brake, "on_release", path, "brake")
    _required(brake, "release_ticks", path, "brake")
    for control_space in ("cartesian", "joint"):
        space = _required(config, control_space, path, "top level")
        if not isinstance(space, dict):
            raise ValueError(f"{path}: {control_space} must be a mapping")
        for key in ("action_size", "rotation_size", "max_lead_m", "max_lead_rad"):
            _required(space, key, path, control_space)
    takeover = _required(config, "takeover", path, "top level")
    if not isinstance(takeover, dict):
        raise ValueError(f"{path}: takeover must be a mapping")
    _required(takeover, "speed", path, "takeover")
    return config


def _lead(override, configured, derived):
    """Resolve a lead cap; explicit zero keeps the CLI's disable-leash meaning."""
    if override is not None:
        return None if override <= 0 else float(override)
    return derived if configured is None else float(configured)


def resolve_spacemouse_config(control_space, *, ik=None, action_size=None,
                              rotation_size=None, deadzone=None,
                              max_lead_m=None, max_lead_rad=None,
                              brake_on_release=None,
                              brake_release_ticks=None,
                              keyboard_reset=True, speed=1.0) -> SpacemouseConfig:
    """Return policy tuning for one control space, with explicit overrides winning.

    ``speed`` scales translation and an explicitly decoupled rotation gain. A
    coupled (``null``) rotation gain keeps following the scaled action size.
    """
    if control_space not in ("cartesian", "joint"):
        raise ValueError(f"unrecognised control space {control_space!r}; expected "
                         "'cartesian' or 'joint'")
    config = load_teleop_config()
    space = config[control_space]
    speed = float(speed)
    base_action_size = space["action_size"] if action_size is None else action_size
    resolved_action_size = float(base_action_size) * speed
    base_rotation_size = space["rotation_size"] if rotation_size is None else rotation_size
    resolved_rotation_size = (None if base_rotation_size is None
                              else float(base_rotation_size) * speed)

    if control_space == "joint":
        ik = ik or DifferentialIK()
        derived_m = 0.9 * min(MAX_LEAD_M, ik.max_lead_m)
        derived_rad = 0.9 * min(MAX_LEAD_RAD, ik.max_lead_rad)
    else:
        derived_m = derived_rad = None

    return SpacemouseConfig(
        action_size=resolved_action_size,
        rotation_size=resolved_rotation_size,
        deadzone=float(config["deadzone"] if deadzone is None else deadzone),
        max_lead_m=_lead(max_lead_m, space["max_lead_m"], derived_m),
        max_lead_rad=_lead(max_lead_rad, space["max_lead_rad"], derived_rad),
        brake_on_release=bool(config["brake"]["on_release"]
                              if brake_on_release is None else brake_on_release),
        brake_release_ticks=int(config["brake"]["release_ticks"]
                                 if brake_release_ticks is None else brake_release_ticks),
        keyboard_reset=keyboard_reset,
    )
