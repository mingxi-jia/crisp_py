"""Offline checks for the shared spacemouse tuning resolver.

Run with ``python -m control.teleop.test_tuning``. This neither opens a robot
connection nor a spacemouse.
"""

import importlib
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from control.ik import MAX_LEAD_M, MAX_LEAD_RAD
from control.teleop import load_teleop_config, resolve_spacemouse_config
import control.teleop.tuning as tuning


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    assert cond, name


def _tuning_values(config):
    return {key: getattr(config, key) for key in (
        "action_size", "rotation_size", "deadzone", "max_lead_m",
        "max_lead_rad", "brake_on_release", "brake_release_ticks")}


def main():
    print("=== YAML configuration ===")
    config = load_teleop_config()
    documented = {
        "deadzone", "brake", "cartesian", "joint", "takeover",
    }
    check("config carries every documented top-level key", documented <= set(config))
    check("brake carries every documented key",
          {"on_release", "release_ticks"} <= set(config["brake"]))
    for space in ("cartesian", "joint"):
        check(f"{space} carries every documented key",
              {"action_size", "rotation_size", "max_lead_m", "max_lead_rad"}
              <= set(config[space]))
    check("takeover carries speed", "speed" in config["takeover"])

    print("\n=== resolution ===")
    cartesian = resolve_spacemouse_config("cartesian")
    check("cartesian action size is configured", cartesian.action_size == 0.015)
    check("cartesian leads are configured",
          cartesian.max_lead_m == 0.10 and cartesian.max_lead_rad == 0.50)

    ik = SimpleNamespace(max_lead_m=MAX_LEAD_M * 0.8,
                         max_lead_rad=MAX_LEAD_RAD * 0.8)
    joint = resolve_spacemouse_config("joint", ik=ik)
    expected_m = 0.9 * min(MAX_LEAD_M, ik.max_lead_m)
    expected_rad = 0.9 * min(MAX_LEAD_RAD, ik.max_lead_rad)
    check("joint action size is configured", joint.action_size == 0.008)
    check("joint leads derive from imported IK constants",
          joint.max_lead_m == expected_m and joint.max_lead_rad == expected_rad)

    override = resolve_spacemouse_config(
        "cartesian", action_size=0.012, rotation_size=0.02, deadzone=0.2,
        max_lead_m=0.04, max_lead_rad=0.3, brake_on_release=False,
        brake_release_ticks=3)
    check("every explicit override beats config",
          _tuning_values(override) == {
              "action_size": 0.012, "rotation_size": 0.02, "deadzone": 0.2,
              "max_lead_m": 0.04, "max_lead_rad": 0.3,
              "brake_on_release": False, "brake_release_ticks": 3,
          })
    scaled = resolve_spacemouse_config("cartesian", action_size=0.012,
                                       rotation_size=0.02, speed=2.5)
    check("speed scales action and explicit rotation gains",
          scaled.action_size == 0.03 and scaled.rotation_size == 0.05)

    print("\n=== all three callers agree ===")
    import deploy.deploy_spacemouse as deploy_spacemouse
    from record.recorder import parse_args as recorder_args
    from record.recorder import resolve_spacemouse_config as recorder_config
    from control.takeover import SpacemouseTakeover

    for space in ("cartesian", "joint"):
        ik = (SimpleNamespace(max_lead_m=MAX_LEAD_M,
                              max_lead_rad=MAX_LEAD_RAD)
              if space == "joint" else None)
        deploy_config = deploy_spacemouse.resolve_spacemouse_config(
            space, ik=ik, keyboard_reset=False, speed=1.0)
        recorded_config = recorder_config(recorder_args([]), space, ik)
        takeover_config = SpacemouseTakeover(
            robot=None, control_space=space, speed=1.0).policy_config
        check(f"all three callers agree for {space} at speed 1.0",
              _tuning_values(deploy_config) == _tuning_values(recorded_config)
              == _tuning_values(takeover_config))

    print("\n=== strict path handling ===")
    missing = Path(tempfile.gettempdir()) / "missing-teleop-config.yaml"
    try:
        load_teleop_config(missing)
    except FileNotFoundError as error:
        check("missing config names $CRISP_TELEOP_CONFIG",
              "$CRISP_TELEOP_CONFIG" in str(error))
    else:
        raise AssertionError("missing teleop config did not raise")

    original_env = os.environ.get("CRISP_TELEOP_CONFIG")
    try:
        with tempfile.TemporaryDirectory() as temp:
            alternate = Path(temp) / "teleop.yaml"
            alternate.write_text("""deadzone: 0.2
brake:
  on_release: false
  release_ticks: 2
cartesian:
  action_size: 0.02
  rotation_size: null
  max_lead_m: 0.1
  max_lead_rad: 0.5
joint:
  action_size: 0.01
  rotation_size: null
  max_lead_m: null
  max_lead_rad: null
takeover:
  speed: 1.0
""")
            os.environ["CRISP_TELEOP_CONFIG"] = str(alternate)
            importlib.reload(tuning)
            check("$CRISP_TELEOP_CONFIG temp file is honoured",
                  tuning.resolve_spacemouse_config("cartesian").action_size == 0.02)
    finally:
        if original_env is None:
            os.environ.pop("CRISP_TELEOP_CONFIG", None)
        else:
            os.environ["CRISP_TELEOP_CONFIG"] = original_env
        importlib.reload(tuning)

    print("\nAll tuning tests passed.")


if __name__ == "__main__":
    main()
