"""Tests for control/takeover.py and the Jacobian it jogs with.

    python -m control.test_takeover

No robot and no spacemouse: everything here is the maths between the stick and
the joint target, which is where a takeover can go wrong quietly. The clamps
in particular have to be checked, because nothing about a missing clamp shows
up until somebody is holding the stick.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from control.projection import Chain, damped_least_squares, frame_offset
from control.takeover import SpacemouseTakeover

Q = np.array([0.0, -0.6, 0.1, -2.1, 0.05, 1.9, 0.7])


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    assert cond, name


def achieved_twist(chain, offset, q0, q1):
    """The twist that actually resulted from a joint step."""
    a, b = chain.fk(q0) @ offset, chain.fk(q1) @ offset
    return np.concatenate([b[:3, 3] - a[:3, 3],
                           Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).as_rotvec()])


def main():
    chain = Chain.from_urdf()
    offset = frame_offset("fingertip")

    print("=== Jacobian ===")
    jac = chain.jacobian(Q, offset)
    check("shape is 6 x joints", jac.shape == (6, 7), str(jac.shape))
    eps = 1e-6
    numeric = np.zeros((6, 7))
    for i in range(7):
        step = np.zeros(7); step[i] = eps
        numeric[:, i] = achieved_twist(chain, offset, Q, Q + step) / eps
    err = float(np.abs(jac - numeric).max())
    check("matches finite differences of FK", err < 1e-5, f"max {err:.1e}")

    # The reference point matters: jogging about the flange instead of the
    # fingertip rotates the gripper about the wrong place.
    check("the tool offset changes the linear rows",
          not np.allclose(chain.jacobian(Q, offset)[:3], chain.jacobian(Q)[:3]))
    check("but not the angular ones",
          np.allclose(chain.jacobian(Q, offset)[3:], chain.jacobian(Q)[3:]))

    print("\n=== damped least squares ===")
    for name, twist in [("translation", [0.01, 0, 0, 0, 0, 0]),
                        ("rotation", [0, 0, 0, 0, 0, 0.03]),
                        ("both", [0.004, -0.002, 0.003, 0.01, 0, 0.01])]:
        dq = damped_least_squares(jac, twist, damping=0.02)
        got = achieved_twist(chain, offset, Q, Q + dq)
        rel = np.linalg.norm(got - np.asarray(twist)) / np.linalg.norm(twist)
        check(f"{name} is produced to within 5%", rel < 0.05, f"{rel*100:.1f}%")

    # Near a singularity the plain pseudo-inverse explodes; damping is the
    # whole reason a human can jog there without the arm bolting.
    straight = np.array([0.0, 0.0, 0.0, -0.2, 0.0, 0.2, 0.0])
    sing = chain.jacobian(straight, offset)
    undamped = np.linalg.pinv(sing) @ np.array([0.0, 0.0, 0.001, 0, 0, 0])
    damped = damped_least_squares(sing, [0.0, 0.0, 0.001, 0, 0, 0], damping=0.05)
    check("damping bounds the step near a singularity",
          np.abs(damped).max() < np.abs(undamped).max(),
          f"{np.abs(damped).max():.4f} vs {np.abs(undamped).max():.4f} rad")

    print("\n=== the jog step follows a Cartesian target ===")
    jog = SpacemouseTakeover(robot=None, chain=chain)
    here = chain.fk(Q) @ offset
    pos, quat = here[:3, 3], Rotation.from_matrix(here[:3, :3]).as_quat()

    check("already there means no motion",
          np.allclose(jog.solve(Q, pos, quat), Q, atol=1e-9))

    # A target one tick ahead is reached in about one tick: this is the
    # property that makes the stick feel direct instead of dragging.
    step = jog.solve(Q, pos + [0.003, 0, 0], quat)
    got = achieved_twist(chain, offset, Q, step)
    check("a 3 mm lead is closed in one tick",
          abs(got[0] - 0.003) < 3e-4 and np.linalg.norm(got[1:3]) < 3e-4,
          f"{np.round(got[:3], 4)}")

    turned = Rotation.from_rotvec([0, 0, 0.01]) * Rotation.from_quat(quat)
    step = jog.solve(Q, pos, turned.as_quat())
    got = achieved_twist(chain, offset, Q, step)
    check("a small rotation lead is closed in one tick",
          abs(got[5] - 0.01) < 1e-3, f"{np.round(got[3:], 4)}")

    print("\n=== the target may lead, but only so far ===")
    # The reference integrates its target regardless of whether the arm keeps
    # up. That is what stops tracking lag dragging the motion backwards -- but
    # it means the gap has to be bounded, or releasing the stick after pushing
    # into a limit would leave the arm chasing a distant target.
    far = jog.solve(Q, pos + [2.0, 0, 0], quat)
    check("an absurd lead is capped per joint",
          np.abs(far - Q).max() <= jog.max_step_rad + 1e-12,
          f"{np.abs(far - Q).max():.4f} rad")
    check("and reported as lagging", jog.status()["lagging"])
    jog.solve(Q, pos + [0.003, 0, 0], quat)
    check("a followed target is not lagging", not jog.status()["lagging"])

    print("\n=== joint limits ===")
    limits = chain.joint_limits
    check("limits are read from the URDF",
          np.all(np.isfinite(limits)) and limits.shape == (7, 2))
    beyond = limits[:, 1] + 0.05
    here_bad = chain.fk(beyond) @ offset
    stepped = jog.solve(beyond, here_bad[:3, 3] + [0.01, 0, 0],
                        Rotation.from_matrix(here_bad[:3, :3]).as_quat())
    check("no target ever exceeds a joint limit",
          np.all(stepped <= limits[:, 1] + 1e-9) and np.all(stepped >= limits[:, 0] - 1e-9),
          f"max overshoot {float((stepped - limits[:, 1]).max()):.2e}")
    check("and says so", jog.status()["at_joint_limit"])

    print("\n=== the feel comes from the reference, not from here ===")
    # Whatever deploy_spacemouse does, this must do: the resolved shared
    # config, not a number copied into either caller.
    from control.teleop import load_teleop_config, resolve_spacemouse_config

    for space in ("cartesian", "joint"):
        for speed in (1.0, 2.5):
            ref = resolve_spacemouse_config(space, speed=speed)
            jog = SpacemouseTakeover(robot=None, chain=chain,
                                     control_space=space, speed=speed)
            check(f"{space} gain follows the shared resolver at speed {speed}",
                  abs(jog.action_size - ref.action_size) < 1e-12,
                  f"{jog.action_size} vs {ref.action_size}")

    # The configured default is part of the contract. It must not silently
    # drift back to SpacemouseConfig's mechanism-level fallback.
    configured = load_teleop_config()
    for space in ("cartesian", "joint"):
        check(f"{space} default gain comes from config",
              abs(resolve_spacemouse_config(space).action_size
                  - configured[space]["action_size"]) < 1e-12)
    import inspect
    source = inspect.getsource(SpacemouseTakeover)
    for forbidden in ("as_euler", "AXIS_SCALING", "get_motion_state"):
        check(f"the deltas are not recomputed here ({forbidden})",
              forbidden not in source)

    print("\n=== one shared joint solver ===")
    from control.ik import DifferentialIK
    import deploy.deploy_spacemouse as deploy_spacemouse
    takeover_source = inspect.getsource(SpacemouseTakeover)
    deploy_source = inspect.getsource(deploy_spacemouse)
    check("takeover delegates to DifferentialIK", isinstance(jog._ik, DifferentialIK))
    check("deploy uses the same solver", "DifferentialIK" in deploy_source)
    check("takeover does not reimplement the joint step",
          "damped_least_squares" not in takeover_source)

    print("\n=== status ===")
    status = SpacemouseTakeover(robot=None, chain=chain).status()
    for key in ("active", "error", "frame", "ticks", "lagging",
                "at_joint_limit", "max_step_rad", "speed"):
        check(f"status reports {key}", key in status)
    import json
    check("status serialises for the panel", json.dumps(status) is not None)

    print("\n=== an optional feature cannot take down /health ===")
    # RobotClient calls /health before anything else, so a throw there stops
    # every client from connecting -- a far worse failure than the feature
    # being unavailable. This is what a half-applied edit looked like.
    import control.robot_server as rs

    class HalfBuilt:
        """A server missing the takeover attribute entirely."""
        paused, ctrl_space, inhand_camera, camera_names = False, "joint", None, []

    half = HalfBuilt()
    half.takeover_status = lambda: rs.RobotServer.takeover_status(half)
    half.set_takeover = lambda on: rs.RobotServer.set_takeover(half, on)
    client = rs.build_app(half).test_client()

    response = client.get("/health")
    check("/health still answers", response.status_code == 200,
          str(response.status_code))
    check("and still carries the fields clients need",
          set(response.get_json()) >= {"ok", "ctrl_space", "paused"})
    check("/takeover reports the problem instead of a 500 page",
          client.get("/takeover").status_code == 200)
    posted = client.post("/takeover", json={"on": True}).get_json()
    check("and a failed takeover names the reason",
          posted["active"] is False and "AttributeError" in posted["error"],
          posted["error"][:60])

    class Broken:
        def status(self): raise RuntimeError("device fell over")

    class WithBroken(HalfBuilt):
        takeover = Broken()

    broken = WithBroken()
    check("a throwing takeover object is reported, not raised",
          "device fell over" in rs.RobotServer.takeover_status(broken)["error"])

    print("\nAll takeover tests passed.")


if __name__ == "__main__":
    main()
