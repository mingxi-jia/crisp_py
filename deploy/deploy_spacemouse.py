#!/usr/bin/env python3
"""Spacemouse teleoperation against robot_server.py.

A policy class and a control loop. Nothing else.

Start the server first:
    python -m control.robot_server --camera-config config/camera_info.yaml --port 7000

Or, for joint-space control:
    python -m control.robot_server --camera-config config/camera_info.yaml --port 7000 --ctrl-space joint

Then, from the repository root:
    python -m deploy.deploy_spacemouse

Controls:
    spacemouse        move the fingertip
    spacemouse button toggle the gripper
    'r'               re-home and re-anchor
    Ctrl+C            quit

Measuring instead of driving:
    python -m deploy.deploy_spacemouse --test        # holding a pose
    python -m deploy.deploy_spacemouse --test-traj   # following a moving one

    Commands a star of poses around wherever the arm is, holds each one, and
    reports the settled error. No spacemouse needed. Cartesian space only --
    it measures how well the impedance controller holds a pose, which is not
    a question joint space answers.

Swapping the human for a network means replacing SpacemousePolicy with any
object exposing ``reset(pose)`` and ``predict_action(obs) -> Action``; the loop
below does not change.

In joint space the server-side spacemouse takeover (control/takeover.py, from
the panel button) owns the device while it is active, so it and this script
cannot run at the same time.
"""

import argparse
import time

from control.robot_client import RobotClient
from control.teleop import SpacemouseConfig, SpacemousePolicy
from control.ik import DifferentialIK, MAX_LEAD_M, MAX_LEAD_RAD
from control.teleop.driver import TeleopDriver


# Joint space asks more of the arm than Cartesian space does for the same
# gain. DifferentialIK converts a Cartesian step into joint motion and clips
# it at max_step_rad (0.05 rad/tick = 2.5 rad/s at 50 Hz), which is sized to
# the FR3's own joint velocity limit of 2.62 rad/s on j1-j4. Measured over 400
# random configurations, the fraction of them where the required step exceeds
# that clip is:
#
#     action_size 0.003 -> 0%     0.010 -> 7%
#     action_size 0.008 -> 1%     0.015 -> 30%
#
# A clipped step is motion the arm is physically unable to make, so the
# command runs ahead and the arm drags. Cartesian space has no such clip --
# the impedance controller just tracks a pose -- which is why the same gain
# feels direct there and draggy here.
JOINT_ACTION_SIZE = 0.008


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:7000",
                   help="robot_server base URL")
    p.add_argument("--rate", type=float, default=50.0,
                   help="control loop frequency [Hz]")
    p.add_argument("--n-steps", type=int, default=None,
                   help="stop after this many steps (default: run until Ctrl+C)")
    p.add_argument("--action-size", type=float, default=None,
                   help="per-step translation gain [m] (default: 0.015 in "
                        "cartesian space, 0.008 in joint space, where "
                        "DifferentialIK's max_step_rad clips anything "
                        "faster)")
    p.add_argument("--rotation-size", type=float, default=None,
                   help="per-step rotation gain, decoupled from "
                        "--action-size (default: follow it). Capped by "
                        "--max-lead-rad, which is a speed ceiling: raising "
                        "this past cap/(d/k) has no effect")
    p.add_argument("--deadzone", type=float, default=0.1,
                   help="spacemouse deadzone, 0..1")
    p.add_argument("--max-lead", type=float, default=None,
                   help="how far the target may run ahead of the arm [m]; "
                        "bounds how far the arm coasts after you release the "
                        "puck. 0 disables the leash")
    p.add_argument("--max-lead-rad", type=float, default=None,
                   help="rotational counterpart of --max-lead [rad]. Must be "
                        "far below the translation cap: rotation is commanded "
                        "3x faster and settles 2x slower, so a cap that looks "
                        "equivalent never engages. 0 disables")
    p.add_argument("--max-step-rad", type=float, default=0.05,
                   help="maximum joint step [rad] in joint mode only")
    p.add_argument("--damping", type=float, default=0.05,
                   help="DLS damping in joint mode only")
    p.add_argument("--brake-ticks", type=int, default=1,
                   help="idle ticks before the release brake fires. 1 stops "
                        "instantly; 2-3 debounces a jittery deadzone at the "
                        "cost of a little coast")
    p.add_argument("--no-brake", dest="brake", action="store_false",
                   help="do not snap the target onto the arm when the puck is "
                        "released; the arm then coasts out its lead")
    p.add_argument("--test", action="store_true",
                   help="do not teleoperate: command a few target poses and "
                        "report how accurately the controller holds them")
    p.add_argument("--test-offset", type=float, default=0.05,
                   help="--test probe distance from the start pose [m]")
    p.add_argument("--test-settle", type=float, default=2.0,
                   help="--test seconds to hold each target before measuring")
    p.add_argument("--test-speed", type=float, default=0.05,
                   help="--test move speed between points [m/s]")
    p.add_argument("--test-traj", action="store_true",
                   help="do not teleoperate: drive a figure-eight path and "
                        "report how well the controller tracked it. --test "
                        "measures holding a pose; this measures following a "
                        "moving one")
    p.add_argument("--traj-radius", type=float, default=0.06,
                   help="--test-traj figure-eight half-extent [m]")
    p.add_argument("--traj-time", type=float, default=8.0,
                   help="--test-traj duration [s]")
    p.add_argument("--traj-rate", type=float, default=50.0,
                   help="--test-traj command rate [Hz]")
    p.add_argument("--traj-fy", type=float, default=0.25,
                   help="--test-traj y frequency [Hz]; 2x --traj-fz gives "
                        "the eight")
    p.add_argument("--traj-fz", type=float, default=0.125,
                   help="--test-traj z frequency [Hz]")
    p.add_argument("--traj-rot-deg", type=float, default=0.0,
                   help="--test-traj: also oscillate the wrist by +/- this "
                        "many degrees about the tool axis, so rotation "
                        "tracking is measured too (0 = translation only)")
    p.add_argument("--traj-fr", type=float, default=None,
                   help="--test-traj wrist oscillation frequency [Hz] "
                        "(default: same as --traj-fz)")
    p.add_argument("--traj-csv", default=None,
                   help="--test-traj: also write per-sample data to this CSV")
    p.add_argument("--yes", action="store_true",
                   help="skip the --test confirmation prompt")
    p.add_argument("--home", action="store_true",
                   help="home the robot before starting")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Accuracy test (--test)
# ---------------------------------------------------------------------------

def _test_targets(start_pos, offset):
    """A star of offsets around the start pose, clipped into EEF_BOUNDS.

    One axis at a time, both signs, so a systematic error shows up as a
    consistent sign on one axis rather than being smeared across a diagonal.
    The pair of +/- probes per axis is what separates a bias (z always low,
    say, from gravity) from plain noise.
    """
    import numpy as np
    from control.kinematics import clamp_to_workspace

    targets = []
    for axis in range(3):
        for sign in (+1.0, -1.0):
            d = np.zeros(3)
            d[axis] = sign * offset
            p = clamp_to_workspace(np.asarray(start_pos, float) + d)
            if np.linalg.norm(p - start_pos) > 1e-4:   # skip if clipped away
                targets.append((f"{'xyz'[axis]}{'+' if sign > 0 else '-'}", p))
    return targets


def _settle_and_measure(robot, target_pos, target_quat, settle, sample_hz=50.0):
    """Hold a target, then report where the arm actually sits.

    Returns (mean_pos_err_vec, pos_err_m, rot_err_rad, jitter_m).
    The last half second is averaged, so the number is steady-state tracking
    error and not a snapshot of the tail of the transient.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    period = 1.0 / sample_hz
    t_end = time.time() + settle
    window, target_rot = [], R.from_quat(np.asarray(target_quat, float))
    while time.time() < t_end:
        tick = time.perf_counter()
        # Keep commanding: servo() publishes one target and returns, so the
        # controller would otherwise just hold whatever it last received.
        res = robot.servo(target_pos, target_quat)
        if res.get("paused") or res.get("takeover"):
            return None
        st = robot.get_state()
        fp = st.fingertip_pose
        window.append((time.time(),
                       np.asarray(fp.position, float),
                       R.from_quat(fp.orientation.as_quat())))
        time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    cutoff = window[-1][0] - 0.5
    tail = [w for w in window if w[0] >= cutoff] or window[-1:]
    pos = np.array([w[1] for w in tail])
    mean_pos = pos.mean(axis=0)
    err_vec = mean_pos - np.asarray(target_pos, float)
    rot_errs = [np.linalg.norm((w[2] * target_rot.inv()).as_rotvec()) for w in tail]
    jitter = float(np.max(np.linalg.norm(pos - mean_pos, axis=1))) if len(pos) > 1 else 0.0
    return err_vec, float(np.linalg.norm(err_vec)), float(np.mean(rot_errs)), jitter


def run_accuracy_test(robot, args):
    """Command a few poses, measure where the arm actually settles."""
    import numpy as np

    if (robot.ctrl_space or "cartesian") != "cartesian":
        raise SystemExit(
            f"--test measures Cartesian tracking, but the server is in "
            f"'{robot.ctrl_space}' space. Restart it with --ctrl-space cartesian.")

    start = robot.get_state().fingertip_pose
    start_pos = np.asarray(start.position, float).copy()
    start_quat = start.orientation.as_quat()
    targets = _test_targets(start_pos, args.test_offset)
    if not targets:
        raise SystemExit("every probe clipped against EEF_BOUNDS; "
                         "move the arm nearer the middle, or lower --test-offset")

    print(f"\nAccuracy test: {len(targets)} points, "
          f"+/-{args.test_offset*1000:.0f} mm about the current fingertip pose.")
    print(f"start {np.round(start_pos, 4)}   settle {args.test_settle}s each")
    print("THE ARM WILL MOVE. Keep the e-stop in reach.")
    if not args.yes:
        if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("aborted")

    rows = []
    for name, tgt in targets:
        print(f"  {name} -> {np.round(tgt, 4)} ", end="", flush=True)
        robot.goto(tgt, start_quat, use_move_to=True, speed=args.test_speed)
        m = _settle_and_measure(robot, tgt, start_quat, args.test_settle)
        if m is None:
            print("SKIPPED (server paused or in takeover)")
            continue
        err_vec, err, rot_err, jitter = m
        rows.append((name, tgt, err_vec, err, rot_err, jitter))
        print(f"err {err*1000:6.2f} mm  rot {np.degrees(rot_err):5.2f} deg  "
              f"jitter {jitter*1000:4.2f} mm")

    print("\nreturning to the start pose")
    robot.goto(start_pos, start_quat, use_move_to=True, speed=args.test_speed)

    if not rows:
        raise SystemExit("no points measured")

    errs = np.array([r[3] for r in rows])
    rots = np.array([r[4] for r in rows])
    vecs = np.array([r[2] for r in rows])

    print("\n" + "=" * 68)
    print(f"{'point':>6} {'err mm':>8} {'dx':>7} {'dy':>7} {'dz':>7} "
          f"{'rot deg':>8} {'jitter':>8}")
    for name, _, v, e, r, j in rows:
        print(f"{name:>6} {e*1000:>8.2f} {v[0]*1000:>7.2f} {v[1]*1000:>7.2f} "
              f"{v[2]*1000:>7.2f} {np.degrees(r):>8.2f} {j*1000:>8.2f}")
    print("-" * 68)
    print(f"{'mean':>6} {errs.mean()*1000:>8.2f} "
          f"{vecs[:,0].mean()*1000:>7.2f} {vecs[:,1].mean()*1000:>7.2f} "
          f"{vecs[:,2].mean()*1000:>7.2f} {np.degrees(rots.mean()):>8.2f}")
    print(f"{'max':>6} {errs.max()*1000:>8.2f}")
    print("=" * 68)
    print("\nThe dx/dy/dz MEANS are the informative part: a consistent sign on")
    print("one axis is a systematic bias (a negative dz mean is the arm sitting")
    print("below its target, i.e. the controller not holding against gravity),")
    print("while signs that cancel out are just tracking error.")


# ---------------------------------------------------------------------------
# Trajectory-following test (--test-traj)
# ---------------------------------------------------------------------------

def _figure_eight(t, center, radius, f_y, f_z):
    """Lissajous 2:1 in the yz plane -- f_y = 2*f_z draws the eight.

    Same shape as deploy/example_figure_eight.py, but centred on wherever the
    arm is rather than a fixed point, and in fingertip frame.
    """
    import numpy as np
    t = np.asarray(t, dtype=np.float64)
    out = np.empty(np.shape(t) + (3,))
    out[..., 0] = center[0]
    out[..., 1] = center[1] + radius * np.sin(2 * np.pi * f_y * t)
    out[..., 2] = center[2] + radius * np.sin(2 * np.pi * f_z * t)
    return out


def _lag_and_residual(cmd, meas, dt, max_lag_s=0.5):
    """Split tracking error into 'late' and 'wrong'.

    A controller that follows the path perfectly but a fixed time behind is a
    very different problem from one that cuts the corners, and the raw RMS
    cannot tell them apart. Shifting the measured signal back by the lag that
    minimises RMS separates them: what is left after the shift is the part
    that is not merely late.
    """
    import numpy as np
    best = (0, float("inf"))
    for k in range(int(max_lag_s / dt) + 1):
        if k >= len(cmd):
            break
        e = meas[k:] - cmd[:len(cmd) - k] if k else meas - cmd
        rms = float(np.sqrt((np.linalg.norm(e, axis=1) ** 2).mean()))
        if rms < best[1]:
            best = (k, rms)
    k, rms_shifted = best
    return k * dt, rms_shifted


def _path_decomposition(cmd, meas):
    """Error along the path (lag-like) vs across it (corner-cutting)."""
    import numpy as np
    tang = np.gradient(cmd, axis=0)
    n = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = np.divide(tang, n, out=np.zeros_like(tang), where=n > 1e-12)
    err = meas - cmd
    along = np.einsum("ij,ij->i", err, tang)
    cross = err - along[:, None] * tang
    return along, np.linalg.norm(cross, axis=1)


def run_traj_test(robot, args):
    """Drive a figure eight and report how well the controller followed it."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from control.kinematics import EEF_BOUNDS

    if (robot.ctrl_space or "cartesian") != "cartesian":
        raise SystemExit(
            f"--test-traj measures Cartesian tracking, but the server is in "
            f"'{robot.ctrl_space}' space. Restart with --ctrl-space cartesian.")

    start = robot.get_state().fingertip_pose
    center = np.asarray(start.position, float).copy()
    quat = start.orientation.as_quat()

    dt = 1.0 / args.traj_rate
    ts = np.arange(0.0, args.traj_time, dt)
    path = _figure_eight(ts, center, args.traj_radius, args.traj_fy, args.traj_fz)

    # Clamping would silently deform the shape and make the tracking numbers
    # meaningless, so refuse rather than measure a different curve.
    lo = np.array([EEF_BOUNDS[k][0] for k in "xyz"])
    hi = np.array([EEF_BOUNDS[k][1] for k in "xyz"])
    if (path < lo).any() or (path > hi).any():
        worst = np.maximum((lo - path).max(axis=0), (path - hi).max(axis=0))
        raise SystemExit(
            f"the figure eight leaves EEF_BOUNDS by up to "
            f"{np.round(worst * 1000, 1)} mm on xyz.\n"
            f"Move the arm nearer the middle of the workspace, or lower "
            f"--traj-radius (currently {args.traj_radius * 1000:.0f} mm).")

    # Orientation path. Rotation is applied in the TOOL frame (right
    # multiply), so it is a wrist twist about the gripper axis rather than a
    # world-frame roll -- that is what a teleoperator actually commands.
    amp = np.radians(args.traj_rot_deg)
    f_r = args.traj_fr if args.traj_fr is not None else args.traj_fz
    base_rot = R.from_quat(quat)
    if amp > 0:
        angles = amp * np.sin(2 * np.pi * f_r * ts)
        rots = [base_rot * R.from_rotvec([0.0, 0.0, a]) for a in angles]
    else:
        angles = np.zeros_like(ts)
        rots = [base_rot] * len(ts)
    quats = np.array([r.as_quat() for r in rots])

    print(f"\nFigure eight: radius {args.traj_radius*1000:.0f} mm, "
          f"{args.traj_time:.1f} s at {args.traj_rate:.0f} Hz")
    print(f"  centre {np.round(center, 4)}  (the current fingertip pose)")
    print(f"  f_y {args.traj_fy} Hz, f_z {args.traj_fz} Hz "
          f"-> {args.traj_fy/args.traj_fz:.0f}:1 Lissajous")
    print(f"  peak commanded speed "
          f"{2*np.pi*args.traj_fy*args.traj_radius:.3f} m/s")
    if amp > 0:
        print(f"  wrist twist +/-{args.traj_rot_deg:.1f} deg at {f_r} Hz "
              f"(peak {np.degrees(2*np.pi*f_r*amp):.1f} deg/s)")
    print("THE ARM WILL MOVE. Keep the e-stop in reach.")
    if not args.yes:
        if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("aborted")

    print("  moving to the start of the path")
    robot.goto(path[0], quats[0], use_move_to=True, speed=args.test_speed)

    cmd, meas, stamp, rot_err = [], [], [], []
    t0 = time.perf_counter()
    for i, t in enumerate(ts):
        tick = time.perf_counter()
        p = path[i]
        res = robot.servo(p, quats[i])
        if res.get("paused") or res.get("takeover"):
            raise SystemExit("server paused or in takeover; aborting")
        st = robot.get_state()
        cmd.append(p)
        meas.append(np.asarray(st.fingertip_pose.position, float))
        m_rot = R.from_quat(st.fingertip_pose.orientation.as_quat())
        rot_err.append((m_rot * rots[i].inv()).as_rotvec())
        stamp.append(time.perf_counter() - t0)
        time.sleep(max(0.0, dt - (time.perf_counter() - tick)))

    cmd = np.array(cmd); meas = np.array(meas); stamp = np.array(stamp)
    rot_err = np.array(rot_err)

    print("  returning to the centre")
    robot.goto(center, quat, use_move_to=True, speed=args.test_speed)

    # Did the loop actually hold its rate? A loop running slow would look
    # like controller lag, so check before blaming the controller.
    gaps = np.diff(stamp)
    achieved = 1.0 / gaps.mean()

    err = np.linalg.norm(meas - cmd, axis=1)
    lag_s, rms_after = _lag_and_residual(cmd, meas, dt)
    along, cross = _path_decomposition(cmd, meas)
    rms = float(np.sqrt((err ** 2).mean()))

    print("\n" + "=" * 64)
    print(f"loop rate      : {achieved:.1f} Hz commanded {args.traj_rate:.0f} Hz"
          f"   (jitter {gaps.std()*1000:.2f} ms)")
    print(f"samples        : {len(cmd)}")
    print("-" * 64)
    print(f"RMS error      : {rms*1000:8.2f} mm")
    print(f"max error      : {err.max()*1000:8.2f} mm")
    print(f"mean error     : {err.mean()*1000:8.2f} mm")
    print("-" * 64)
    print(f"best-fit lag   : {lag_s*1000:8.1f} ms   "
          f"({lag_s/dt:.1f} ticks)")
    print(f"RMS after lag  : {rms_after*1000:8.2f} mm   "
          f"({100*(1-rms_after/rms):.0f}% of the error was just being late)")
    print("-" * 64)
    print(f"along-path err : {np.abs(along).mean()*1000:8.2f} mm mean  "
          f"{np.abs(along).max()*1000:6.2f} max   (lag-like)")
    print(f"cross-path err : {cross.mean()*1000:8.2f} mm mean  "
          f"{cross.max()*1000:6.2f} max   (corner-cutting)")
    if amp > 0:
        ang = np.linalg.norm(rot_err, axis=1)
        # Rotational lag, by the same shift search used for position.
        # rot_err is measured*commanded^-1 as a rotvec in the WORLD frame,
        # while the commanded twist is about the TOOL axis -- with the
        # gripper pointing down those differ by nearly a sign flip, so the
        # error has to be projected onto the actual axis rather than read off
        # the world z component.
        axis_world = base_rot.apply([0.0, 0.0, 1.0])
        sig_c = angles
        sig_m = angles + rot_err @ axis_world
        best = (0, float("inf"))
        for k in range(int(0.5 / dt) + 1):
            if k >= len(sig_c):
                break
            e = sig_m[k:] - sig_c[:len(sig_c) - k] if k else sig_m - sig_c
            rms = float(np.sqrt((e ** 2).mean()))
            if rms < best[1]:
                best = (k, rms)
        print("-" * 64)
        print(f"rotation RMS   : {np.degrees(np.sqrt((ang**2).mean())):8.3f} deg"
              f"   max {np.degrees(ang.max()):.3f}")
        twist_err = rot_err @ axis_world
        print(f"rotation lag   : {best[0]*dt*1000:8.1f} ms   "
              f"(translation lag was {lag_s*1000:.0f} ms)")
        print(f"  about the twist axis : bias {np.degrees(twist_err.mean()):+7.3f} deg"
              f"   swing {np.degrees(twist_err.std()):6.3f} deg")
        off = np.linalg.norm(rot_err - np.outer(twist_err, axis_world), axis=1)
        print(f"  off-axis (not commanded): mean {np.degrees(off.mean()):6.3f} deg"
              f"   max {np.degrees(off.max()):6.3f} deg")
    print("=" * 64)
    print("\nA large along-path error that mostly vanishes once the lag is")
    print("removed means the controller is LATE, not inaccurate -- raise")
    print("stiffness or lower the speed. A stubborn cross-path error means it")
    print("is cutting corners, which stiffness fixes and speed does not.")

    if args.traj_csv:
        import csv
        with open(args.traj_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "cx", "cy", "cz", "mx", "my", "mz", "err_m"])
            for i in range(len(cmd)):
                w.writerow([f"{stamp[i]:.4f}", *[f"{v:.6f}" for v in cmd[i]],
                            *[f"{v:.6f}" for v in meas[i]], f"{err[i]:.6f}"])
        print(f"\nwrote {args.traj_csv}")
    return {"rms": rms, "max": float(err.max()), "lag_s": lag_s,
            "rms_after_lag": rms_after}


def main():
    args = parse_args()

    robot = RobotClient(args.url)

    ctrl_space = robot.ctrl_space
    if ctrl_space not in (None, "cartesian", "joint"):
        raise SystemExit(
            f"the robot server at {args.url} reported unrecognised control "
            f"space {ctrl_space!r}; expected 'cartesian' or 'joint'")
    drive_space = ctrl_space or "cartesian"

    if args.test or args.test_traj:
        if args.home:
            print("Homing...")
            robot.home()
        if args.test:
            run_accuracy_test(robot, args)
        if args.test_traj:
            run_traj_test(robot, args)
        return

    ik = (DifferentialIK(frame="fingertip", damping=args.damping,
                         max_step_rad=args.max_step_rad)
          if drive_space == "joint" else None)

    # Two lead clamps run in series in joint mode: this policy's, and the one
    # inside DifferentialIK. Only the tighter of the pair ever binds, so if
    # the IK's is tighter it clamps on every push and sets .lagging, which
    # the loop reports -- a stream of "[IK lagging]" that means nothing
    # except that the wrong clamp is in charge. Keep the policy's just inside
    # the IK's so the policy is the one that acts and the IK stays a
    # backstop. Explicit flags still win.
    action_size = args.action_size
    if action_size is None:
        action_size = JOINT_ACTION_SIZE if drive_space == "joint" else 0.015

    if drive_space == "joint":
        default_lead = 0.9 * min(MAX_LEAD_M, ik.max_lead_m)
        default_lead_rad = 0.9 * min(MAX_LEAD_RAD, ik.max_lead_rad)
    else:
        default_lead, default_lead_rad = 0.10, 0.50
    max_lead = default_lead if args.max_lead is None else args.max_lead
    max_lead_rad = (default_lead_rad if args.max_lead_rad is None
                    else args.max_lead_rad)

    policy = SpacemousePolicy(SpacemouseConfig(
        action_size=action_size,
        deadzone=args.deadzone,
        max_lead_m=max_lead if max_lead > 0 else None,
        max_lead_rad=max_lead_rad if max_lead_rad > 0 else None,
        rotation_size=args.rotation_size,
        brake_on_release=args.brake,
        brake_release_ticks=args.brake_ticks,
    ))

    if args.home:
        print("Homing...")
        robot.home()

    period = 1.0 / args.rate
    step = 0

    print(f"Driving in {drive_space} control space, "
          f"action_size={action_size:.3f}.")
    print(f"Teleoperating at {args.rate:.0f} Hz. 'r' to re-home, Ctrl+C to quit.")
    with TeleopDriver(robot, policy, drive_space, ik=ik) as driver:

        try:
            while args.n_steps is None or step < args.n_steps:
                tick = time.perf_counter()

                teleop_step = driver.step()
                action = teleop_step.action

                if action.reset:
                    print("[r] re-homing...")
                    robot.home()
                    driver.reset_anchor()
                    continue

                if "lagging" in teleop_step.events:
                        print("  [IK lagging] target lead capped; near a limit or singularity")
                if "at_joint_limit" in teleop_step.events:
                        print("  [IK joint limit] joint target clipped at a limit")
                if "held" in teleop_step.events:
                    hold_reason = ("paused" if teleop_step.result.get("paused")
                               else "spacemouse takeover")
                    print(f"  [server {hold_reason}] holding; the target will re-anchor on resume")
                elif "resumed" in teleop_step.events:
                    print("  [server resumed] target re-anchored to the current pose")

                step += 1
                time.sleep(max(0.0, period - (time.perf_counter() - tick)))
        except KeyboardInterrupt:
            print("\nStopped.")

    print(f"Ran {step} steps.")


if __name__ == "__main__":
    main()
