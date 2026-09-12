#!/usr/bin/env python3
"""Deploy the π0.5-DROID checkpoint on this robot.

Everything π0.5 lives under deploy/pi05/:
    policy.py    the policy client and the DROID<->crisp conversions
    features.py  PCA and cosine-similarity maths (pure numpy)
    viz.py       the local OpenCV window
    panel.py     the web panel, served on its own port
This file is the CLI and the control loop.

Prerequisites
-------------
1. Policy server, on a machine with a decent GPU (~4090). Note the `cd`:
   the server script needs openpi's environment.

       cd third_party/openpi
       GIT_LFS_SKIP_SMUDGE=1 uv sync                    # first time only
       uv run python ../../deploy/pi05_policy_server.py \\
           --config pi05_droid \\
           --dir gs://openpi-assets/checkpoints/pi05_droid

   That wrapper also returns vision features, which the panel needs.
   openpi's stock `scripts/serve_policy.py --env=DROID` still works; you
   just get no feature panes.

2. Robot server, in JOINT control space -- π0.5-DROID emits joint-space
   actions, so the Cartesian controller would ignore them:

       ./scripts/start_all.sh --ctrl-space joint

3. This script:

       python -m deploy.deploy_pi05 --prompt "pick up the red block" \\
           --policy-host localhost --panel

Going slower
------------
    --speed-scale 0.3     Each joint step is a third as far; the loop stays at
                          15 Hz, so the policy's observation timing is
                          unchanged and it re-plans from where it got to.
                          Start here.
    --max-step-rad 0.05   Hard ceiling on any one joint step, applied after
                          the scale.
    --rate 5              Same commanded path, executed 3x slower, but the
                          policy then observes at 5 Hz instead of 15 Hz.
                          Prefer --speed-scale.
"""

import argparse
import time

import numpy as np

from control.robot_client import RobotClient
from deploy.pi05.policy import (
    DROID_CONTROL_HZ,
    MAX_JOINT_DELTA,
    Pi05DroidPolicy,
    StaleCameraError,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:7000",
                   help="robot_server base URL")
    p.add_argument("--policy-host", default="0.0.0.0",
                   help="openpi policy server host")
    p.add_argument("--policy-port", type=int, default=8000,
                   help="openpi policy server port")
    p.add_argument("--api-key", default=None,
                   help="policy server API key, if it requires one")
    p.add_argument("--prompt", default=None,
                   help="task instruction; prompted for interactively if omitted")
    p.add_argument("--external-camera", default="cam1",
                   help="camera used as the exterior view")
    p.add_argument("--wrist-camera", default="cam4",
                   help="camera used as the wrist view")
    p.add_argument("--exterior-fit", choices=["box", "crop", "pad"], default="box",
                   help="how the side view is made square: box (default, crop "
                        "the explicit --exterior-crop region), crop "
                        "(centre-crop), or pad (letterbox, as DROID trained)")
    p.add_argument("--exterior-crop", default="254,130,574,450", metavar="X0,Y0,X1,Y1",
                   help="crop region for --exterior-fit box, in source pixels. "
                        "Default suits the 848x480 side view; it is validated "
                        "against the actual frame, so a stale box errors rather "
                        "than silently cropping the wrong place")
    p.add_argument("--wrist-fit", choices=["box", "crop", "pad"], default="pad",
                   help="same for the wrist view (default pad)")
    p.add_argument("--wrist-crop", default=None, metavar="X0,Y0,X1,Y1",
                   help="crop region for --wrist-fit box")
    p.add_argument("--max-steps", type=int, default=600,
                   help="stop after this many control steps")
    p.add_argument("--open-loop-horizon", type=int, default=8,
                   help="actions executed per inference (8 ~ 0.5 s at 15 Hz)")
    p.add_argument("--speed-scale", type=float, default=1.0,
                   help="scale every joint step (0.3 = a third as far per "
                        "step). Keeps the 15 Hz loop the policy expects; the "
                        "preferred way to go slower")
    p.add_argument("--max-step-rad", type=float, default=MAX_JOINT_DELTA,
                   help="hard cap on any single joint step [rad], applied "
                        "after --speed-scale")
    p.add_argument("--rate", type=float, default=DROID_CONTROL_HZ,
                   help="control rate [Hz]. Lowering it executes the same "
                        "path proportionally slower, but the policy then sees "
                        "the world less often than it was trained to")
    p.add_argument("--no-binarize-gripper", action="store_true",
                   help="send the raw gripper value instead of snapping to 0/1")
    p.add_argument("--home", action="store_true",
                   help="home the robot before starting")
    p.add_argument("--on-stale", choices=["stop", "hold", "recover"],
                   default="stop",
                   help="what to do when a camera stops delivering: stop the "
                        "rollout (default), hold position until it returns, or "
                        "try restarting the camera nodes and resume")
    p.add_argument("--stale-timeout", type=float, default=20.0,
                   help="with --on-stale hold/recover, give up after this long")
    p.add_argument("--panel", action="store_true",
                   help="serve the π0.5 panel (obs, PCA feature maps, "
                        "click-to-cosine-similarity) on --panel-port")
    p.add_argument("--panel-port", type=int, default=7100)
    p.add_argument("--pca-components", type=int, default=64,
                   help="PCA components kept for the panel's similarity maths")
    p.add_argument("--pca-fit-frames", type=int, default=8,
                   help="frames used to fit the PCA basis before it is held fixed")
    p.add_argument("--visualize", action="store_const", const="window",
                   default=None, dest="visualize",
                   help="also open a local OpenCV window (needs a display). "
                        "The web panel is --panel and is usually better.")
    p.add_argument("--dry-run", action="store_true",
                   help="query the policy and print actions, but never move")
    return p.parse_args()


def parse_box(text, flag: str):
    """'x0,y0,x1,y1' -> (x0, y0, x1, y1). None passes through."""
    if not text:
        return None
    try:
        parts = [int(v) for v in str(text).replace(" ", "").split(",")]
    except ValueError as e:
        raise SystemExit(f"{flag}: expected four integers 'x0,y0,x1,y1', got {text!r} ({e})")
    if len(parts) != 4:
        raise SystemExit(f"{flag}: expected four integers 'x0,y0,x1,y1', got {text!r}")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise SystemExit(f"{flag}: x1 must exceed x0 and y1 must exceed y0, got {parts}")
    w, h = x1 - x0, y1 - y0
    if w != h:
        print(f"NOTE {flag}: the region is {w}x{h}, not square, so resizing to "
              f"224x224 will stretch it by {max(w, h) / min(w, h):.2f}x")
    return (x0, y0, x1, y1)


def wait_for_cameras(robot, policy, args, stale_since: float) -> bool:
    """Hold (and optionally restart the cameras) until frames return.

    Returns True if the cameras came back within --stale-timeout. The arm is
    not commanded while we wait, so it holds its last target.
    """
    needed = {args.external_camera, args.wrist_camera}
    attempted_restart = False

    while time.time() - stale_since < args.stale_timeout:
        status = robot.cameras_status()
        stale = set(status.get("stale") or [])
        if not (needed & stale):
            print(f"        cameras recovered after "
                  f"{time.time() - stale_since:.1f}s; resuming")
            return True

        if args.on_stale == "recover" and not attempted_restart:
            attempted_restart = True
            print("        restarting the camera nodes...")
            try:
                robot.restart_cameras()
            except Exception as e:
                print(f"        restart failed: {e}")
        time.sleep(1.0)

    return False


def main():
    args = parse_args()

    robot = RobotClient(args.url)

    # Preflight, before touching the policy server. pi0.5-DROID emits joint
    # actions, so a server in cartesian space would accept nothing we send.
    ctrl_space = robot.ctrl_space
    if ctrl_space is not None and ctrl_space != "joint":
        raise SystemExit(
            f"the robot server at {args.url} is running in "
            f"'{ctrl_space}' control space, but pi0.5-DROID emits joint-space "
            f"actions that the cartesian controller ignores.\n"
            f"Restart it in joint space:\n"
            f"    ./scripts/start_all.sh --stop\n"
            f"    ./scripts/start_all.sh --ctrl-space joint")

    state = robot.get_state()
    if state.joint_values is None or len(state.joint_values) < 7:
        raise SystemExit(f"expected 7 joint values, got {state.joint_values}")

    # Fail before touching the policy server if the cameras it needs are absent.
    obs = robot.get_obs(include_depth=False)
    for role, cam in (("exterior", args.external_camera),
                      ("wrist", args.wrist_camera)):
        if obs.rgb(cam) is None:
            raise SystemExit(
                f"no frames from the {role} camera {cam!r}. "
                f"Cameras with frames: {obs.ready_cameras or 'none'}. "
                f"Start them with ./scripts/start_all.sh, or pass "
                f"--{role.replace('exterior', 'external')}-camera")

    for role, cam, fit, box in (
            ("exterior", args.external_camera, args.exterior_fit,
             parse_box(args.exterior_crop, "--exterior-crop")),
            ("wrist", args.wrist_camera, args.wrist_fit,
             parse_box(args.wrist_crop, "--wrist-crop"))):
        img = obs.rgb(cam)
        src = f"{img.shape[1]}x{img.shape[0]}" if img is not None else "?"
        if fit == "box" and box is not None:
            x0, y0, x1, y1 = box
            print(f"  {role:8s} {cam}: {src} -> crop x[{x0}:{x1}] y[{y0}:{y1}] "
                  f"({x1-x0}x{y1-y0}) -> 224x224")
        else:
            print(f"  {role:8s} {cam}: {src} -> {fit} -> 224x224")

    prompt = args.prompt if args.prompt is not None else input("Task instruction: ")

    print(f"Connecting to policy server at {args.policy_host}:{args.policy_port} ...")
    policy = Pi05DroidPolicy(
        host=args.policy_host,
        port=args.policy_port,
        prompt=prompt,
        external_camera=args.external_camera,
        wrist_camera=args.wrist_camera,
        open_loop_horizon=args.open_loop_horizon,
        exterior_fit=args.exterior_fit,
        wrist_fit=args.wrist_fit,
        want_features=bool(args.panel),
        exterior_box=parse_box(args.exterior_crop, "--exterior-crop"),
        wrist_box=parse_box(args.wrist_crop, "--wrist-crop"),
        binarize_gripper=not args.no_binarize_gripper,
        speed_scale=args.speed_scale,
        max_step_rad=args.max_step_rad,
        api_key=args.api_key,
    )
    print(f"  server metadata: {policy.server_metadata}")
    if args.panel and not policy.server_metadata.get("features"):
        print("  NOTE: this policy server does not advertise vision features. "
              "The panel will show observations but no PCA/similarity maps. "
              "Start deploy/pi05_policy_server.py instead of openpi's "
              "scripts/serve_policy.py to get them.")

    if args.home:
        print("Homing...")
        robot.home()

    top_speed = args.speed_scale * args.max_step_rad * args.rate
    print(f"Speed: {args.speed_scale:g}x scale, {args.max_step_rad:g} rad/step cap, "
          f"{args.rate:g} Hz -> at most {top_speed:.2f} rad/s per joint "
          f"(stock pi0.5-DROID is {MAX_JOINT_DELTA * DROID_CONTROL_HZ:.2f} rad/s)")
    if abs(args.rate - DROID_CONTROL_HZ) > 1e-6:
        print(f"NOTE: the loop is at {args.rate:g} Hz, not the {DROID_CONTROL_HZ:g} Hz "
              f"the policy was trained at. The commanded path is unchanged and "
              f"simply executes {DROID_CONTROL_HZ / args.rate:.1f}x slower, but the "
              f"policy observes less often, so closed-loop behaviour will differ. "
              f"--speed-scale slows the arm without changing the feedback rate.")

    panel_state = panel_analyser = None
    if args.panel:
        from deploy.pi05 import panel as pi05_panel
        panel_state = pi05_panel.PanelState()
        panel_analyser = pi05_panel.FeatureAnalyser(
            n_components=args.pca_components, fit_frames=args.pca_fit_frames)
        pi05_panel.serve(panel_state, panel_analyser, port=args.panel_port)
        print(f"π0.5 panel: http://localhost:{args.panel_port}/")

    viz = None
    if args.visualize == "window":
        from deploy.pi05.viz import Pi05Visualizer
        viz = Pi05Visualizer(args.external_camera, args.wrist_camera)
    elif args.visualize == "panel":
        print(f"Streaming policy inputs to {args.url}/panel")

    period = 1.0 / args.rate
    policy.reset()
    prev_gripper = None
    step = 0

    print(f'\nTask: "{prompt}"')
    print(f"Running up to {args.max_steps} steps at {args.rate:.0f} Hz. "
          f"Ctrl+C to stop" + (", or q in the window" if args.visualize else "") + "." + ("  [DRY RUN — no motion]" if args.dry_run else ""))

    try:
        while step < args.max_steps:
            tick = time.perf_counter()

            # Cheap per-step read: the joint delta is applied to the *measured*
            # position, as DROID does. Images are only fetched when the chunk
            # runs out, since /get_obs is ~10x the cost of /get_state.
            needs_inference = (policy._chunk is None
                               or policy._chunk_index >= policy.open_loop_horizon
                               or policy._chunk_index >= len(policy._chunk))
            obs = (robot.get_obs(include_depth=False) if needs_inference
                   else robot.get_state())

            try:
                action = policy.predict_action(obs)
            except StaleCameraError as e:
                # Never keep driving on a frozen image. The arm holds position
                # because we simply stop sending new targets; the impedance
                # controller keeps the last one.
                print(f"\n[STALE] {e}")
                if args.on_stale == "stop":
                    print("        stopping (--on-stale hold or recover to keep going)")
                    break
                if not wait_for_cameras(robot, policy, args, stale_since=time.time()):
                    print("        giving up")
                    break
                policy.reset()
                continue

            if not args.dry_run:
                robot.servo_joint(action.joint_positions)
                if action.gripper != prev_gripper:
                    robot.set_gripper(action.gripper)
                    prev_gripper = action.gripper

            if panel_state is not None:
                from deploy.pi05.panel import build_payload
                panel_state.set(build_payload(policy, action, args, step, panel_analyser))

            if viz is not None:
                keep_going = viz.update(
                    policy.last_request, action,
                    {"step": f"{step}/{args.max_steps}",
                     "infer": (f"{policy.last_inference_ms:.0f} ms"
                               if policy.last_inference_ms else "--"),
                     "chunk": f"{policy._chunk_index}/{policy.open_loop_horizon}",
                     "rate": f"{args.rate:g} Hz",
                     "speed": f"{args.speed_scale:g}x",
                     "mode": "DRY RUN" if args.dry_run else "live"})
                if not keep_going:
                    print("\nStopped from the visualizer (q).")
                    break

            if step % 15 == 0:
                infer = (f"{policy.last_inference_ms:.0f} ms"
                         if policy.last_inference_ms else "—")
                feat = (f" feat={policy.last_feature_ms:.0f}ms"
                        if getattr(policy, "last_feature_ms", None) else "")
                print(f"  step {step:4d}  |v|max={np.abs(action.raw_velocity).max():.2f}"
                      f"  grip={action.gripper:.0f}  infer={infer}{feat}")

            step += 1
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if viz is not None:
            viz.close()

    print(f"\nRan {step} steps, {policy.n_inferences} inferences.")


if __name__ == "__main__":
    main()
