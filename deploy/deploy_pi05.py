#!/usr/bin/env python3
"""Deploy the π0.5-DROID checkpoint on this robot.

Everything π0.5 lives under deploy/pi05/:
    policy.py    the policy client and the DROID<->crisp conversions
    features.py  PCA and cosine-similarity maths (pure numpy)
    overlay.py   FK + camera projection: the action chunk as a path on the image
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

Seeing what it intends
----------------------
`--panel` draws the predicted chunk on the camera frame: the joint velocities
are integrated into joint targets, run through forward kinematics, and
projected with config/camera_info.yaml, so the plan can be read against the
scene. Two things are worth checking the first time:

    the FK residual the preflight prints. It compares FK against the pose the
    robot reports for itself; millimetres means the chain is right, and
    centimetres means the drawn path is fiction.

    which calibration entry belongs to the camera. The names in
    config/camera_info.yaml are from whichever rig was calibrated, not the
    role names this repo uses, so each camera carries the mapping in
    config/cameras.yaml (`agentview` is calibrated as `cam3`).
    --overlay-calibration overrides it for a one-off.

The path is drawn on the *raw* frame by default, with the --exterior-crop box
on it, because the gripper often sits near the crop edge and a path drawn only
inside the crop would be silently clipped. The panel's "policy input" tab
shows the same path on the 224x224 the model is actually given.

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


def build_overlays(args, state, frame_sizes=None):
    """The panel's trajectory projectors, one per camera, with a preflight.

    Checks forward kinematics against the pose the robot reports for itself
    before anything is drawn: FK, the tip frame and the joint order all have to
    be right, and a residual in millimetres is the proof. Centimetres means
    something is wrong upstream of the drawing, so say so loudly rather than
    putting a confident-looking line in the wrong place.

    A camera with no calibration entry is reported and skipped, not fatal: the
    agentview overlay is useful on its own.
    """
    from deploy.pi05.overlay import Overlay

    frame_sizes = frame_sizes or {}
    wanted = [("exterior", args.external_camera, args.exterior_fit,
               parse_box(args.exterior_crop, "--exterior-crop"),
               args.overlay_calibration),
              ("wrist", args.wrist_camera, args.wrist_fit,
               parse_box(args.wrist_crop, "--wrist-crop"),
               args.wrist_calibration)]

    overlays = []
    for role, camera, fit, box, key in wanted:
        overlay = Overlay(camera=camera, fit=fit, box=box, role=role,
                          frame=args.overlay_frame, urdf=args.urdf,
                          camera_info=args.camera_info, calibration_key=key)
        if not overlay.available:
            print(f"  overlay: {role} camera {camera!r} not drawn — "
                  f"{overlay.unavailable_reason()}")
            continue

        mount = overlay.calibration.parent
        where = ("static, in the robot base frame" if not overlay.calibration.moving
                 else f"mounted on {mount}, placed by FK each frame")
        print(f"  overlay: {role} {camera} via the "
              f"{overlay.calibration_key!r} calibration ({where})")

        if frame_sizes.get(role):
            drift = overlay.calibration.aspect_mismatch(frame_sizes[role])
            if drift > 0.002:
                cw, ch = overlay.calibration.calib_size
                w, h = frame_sizes[role]
                print(f"           NOTE: calibrated at {cw}x{ch} but streaming "
                      f"{w}x{h}, a {drift*100:.1f}% aspect difference. "
                      f"Intrinsics are scaled, so expect the drawn point to "
                      f"sit a few pixels off near the frame edges.")
        overlays.append(overlay)

    if overlays:
        pos_mm, rot_deg = overlays[0].residual(state)
        verdict = "ok" if pos_mm < 5.0 else "TOO LARGE -- do not trust the overlay"
        print(f"           FK to {args.overlay_frame} vs the robot's reported "
              f"pose: {pos_mm:.1f} mm / {rot_deg:.2f} deg ({verdict})")
        if pos_mm >= 5.0:
            print("           the model, the tip frame or the joint order "
                  "disagrees with the driver. Check "
                  "config/gripper_geometry.yaml and --urdf before believing "
                  "the drawn path.")
    return overlays


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
    p.add_argument("--external-camera", default="agentview",
                   help="camera used as the exterior view (a name from "
                        "config/cameras.yaml)")
    p.add_argument("--wrist-camera", default="inhand",
                   help="camera used as the wrist view (a name from "
                        "config/cameras.yaml)")
    p.add_argument("--exterior-fit", choices=["box", "crop", "pad"], default="box",
                   help="how the side view is made square: box (default, crop "
                        "the explicit --exterior-crop region), crop "
                        "(centre-crop), or pad (letterbox, as DROID trained)")
    p.add_argument("--exterior-crop", default="254,130,574,450", metavar="X0,Y0,X1,Y1",
                   help="crop region for --exterior-fit box, in source pixels. "
                        "Default suits the 848x480 agentview; it is validated "
                        "against the actual frame, so a stale box errors rather "
                        "than silently cropping the wrong place")
    p.add_argument("--wrist-fit", choices=["box", "crop", "pad"], default="pad",
                   help="same for the wrist view (default pad)")
    p.add_argument("--wrist-crop", default=None, metavar="X0,Y0,X1,Y1",
                   help="crop region for --wrist-fit box")
    p.add_argument("--max-steps", type=int, default=600,
                   help="stop after this many control steps; 0 runs until "
                        "Ctrl+C. The bound is deliberate: unlike teleop, "
                        "where the arm only moves while a hand is on the "
                        "stick, nothing here stops a policy that is quietly "
                        "doing the wrong thing. 600 is 40 s at 15 Hz")
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
                        "click-to-cosine-similarity, predicted action chunk) "
                        "on --panel-port")
    p.add_argument("--panel-port", type=int, default=7100)
    p.add_argument("--prompt-history", default=None, metavar="PATH",
                   help="where the panel's task history is kept "
                        "(default ~/.cache/crisp/pi05_prompts.json)")
    p.add_argument("--no-prompt-history", action="store_true",
                   help="do not read or write the task history")
    p.add_argument("--listen", action="store_true",
                   help="take the task from the microphone: say it, then say "
                        "\"done\". Speech between one \"done\" and the next "
                        "accumulates, so the task can be said in several "
                        "breaths. Needs --panel (that is where it is shown)")
    p.add_argument("--listen-model", default="base.en",
                   help="faster-whisper model. base.en is real-time on a CPU "
                        "core; small.en is better on accents and costs ~3x")
    p.add_argument("--listen-device", default="cpu", choices=["cpu", "cuda"],
                   help="where to run it. cpu by default: the GPU is busy "
                        "serving the policy, and a short utterance transcribes "
                        "in a fraction of a second either way")
    p.add_argument("--listen-alsa", default=None, metavar="DEVICE",
                   help="ALSA capture device, e.g. plughw:2,0 (default: the "
                        "system default). `arecord -l` lists them")
    p.add_argument("--listen-terminator", default="done",
                   help="the word that ends a spoken task")
    p.add_argument("--listen-cancel", default="nevermind",
                   help="the word that abandons what is being said, or -- said "
                        "on its own -- puts the previous task back")
    p.add_argument("--no-speak", action="store_true",
                   help="do not read the task back through the speaker")
    p.add_argument("--speak-engine", default="auto",
                   choices=["auto", "piper", "spd-say", "none"],
                   help="how to talk back. auto prefers Piper, a small neural "
                        "voice that runs locally; spd-say drives espeak, which "
                        "is robotic enough that Whisper mis-hears it")
    p.add_argument("--speak-voice", default=None, metavar="NAME_OR_PATH",
                   help="Piper voice, e.g. en_US-lessac-medium or a path to "
                        "an .onnx (default: whatever is in ~/.cache/piper)")
    p.add_argument("--samples", type=int, default=1, metavar="N",
                   help="ask the policy for N plans per inference instead of "
                        "one and draw them all on the panel. pi0.5 samples "
                        "from noise, so they differ; where they fan out is "
                        "where the policy is unsure. They come back from one "
                        "batched forward pass, but that batch is still N times "
                        "the work -- watch the inference time")
    p.add_argument("--samples-every", type=int, default=1, metavar="K",
                   help="only ask for the extra plans every Kth inference, so "
                        "the control loop pays the bigger batch less often")
    p.add_argument("--pca-components", type=int, default=64,
                   help="PCA components kept for the panel's similarity maths")
    p.add_argument("--pca-fit-frames", type=int, default=8,
                   help="frames used to fit the PCA basis before it is held fixed")
    p.add_argument("--overlay-frame", choices=["fingertip", "ee"], default="fingertip",
                   help="which frame the panel draws on the image: the Robotiq "
                        "fingertip (default, what the policy and the training "
                        "data speak) or the end-effector frame the driver "
                        "reports, ~6 cm up the gripper")
    p.add_argument("--overlay-calibration", default=None, metavar="NAME",
                   help="camera_info.yaml entry to use for the exterior "
                        "overlay, overriding the camera's `calibration:` in "
                        "config/cameras.yaml (agentview -> cam3)")
    p.add_argument("--wrist-calibration", default=None, metavar="NAME",
                   help="same for the wrist camera. A wrist camera needs an "
                        "entry with `frame: <link>` -- its pose is fixed "
                        "relative to the link it rides, not to the robot base")
    p.add_argument("--camera-info", default=None, metavar="PATH",
                   help="camera calibration for the overlay "
                        "(default config/camera_info.yaml)")
    p.add_argument("--urdf", default=None, metavar="PATH",
                   help="robot model used for the overlay's forward kinematics "
                        "(default control/fr3_robot.urdf)")
    p.add_argument("--no-overlay", action="store_true",
                   help="do not draw the predicted trajectory on the panel's "
                        "observation")
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

    overlays = []
    if args.panel and not args.no_overlay:
        sizes = {}
        for role, cam in (("exterior", args.external_camera),
                          ("wrist", args.wrist_camera)):
            img = obs.rgb(cam)
            if img is not None:
                sizes[role] = (img.shape[1], img.shape[0])
        overlays = build_overlays(args, state, sizes)

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
        n_samples=args.samples if args.panel else 1,
        samples_every=args.samples_every,
        exterior_box=parse_box(args.exterior_crop, "--exterior-crop"),
        wrist_box=parse_box(args.wrist_crop, "--wrist-crop"),
        binarize_gripper=not args.no_binarize_gripper,
        speed_scale=args.speed_scale,
        max_step_rad=args.max_step_rad,
        api_key=args.api_key,
    )
    print(f"  server metadata: {policy.server_metadata}")
    if args.samples > 1 and not args.panel:
        print("  NOTE: --samples only feeds the panel, and --panel is off; "
              "asking for one plan per inference.")
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

    panel_state = panel_analyser = panel_history = None
    if args.panel:
        from deploy.pi05 import panel as pi05_panel
        panel_state = pi05_panel.PanelState()
        panel_analyser = pi05_panel.FeatureAnalyser(
            n_components=args.pca_components, fit_frames=args.pca_fit_frames)
        panel_history = pi05_panel.PromptHistory(
            False if args.no_prompt_history else args.prompt_history)
        # The task this rollout started with belongs in the history too, so it
        # is one click away next time.
        panel_history.add(prompt)
        pi05_panel.serve(panel_state, panel_analyser, port=args.panel_port,
                         policy=policy, history=panel_history)
        print(f"π0.5 panel: http://localhost:{args.panel_port}/  "
              f"(the task can be retyped there)")

    listener = None
    if args.listen:
        if not args.panel:
            print("  NOTE: --listen without --panel; you will only see what "
                  "was heard in this terminal.")
        from control.speech import SpeechPrompter

        # What has actually been asked for this session, oldest first. The
        # persistent history is a different thing: it remembers across
        # sessions and reorders by use, so it cannot answer "what was I doing
        # before this one".
        spoken_stack = [prompt]

        def spoken(text):
            """Same hand-off as the panel's text box: queue it, let the
            control loop adopt it at a safe point."""
            policy.request_prompt(text)
            if panel_history is not None:
                panel_history.add(text)
            spoken_stack.append(text)
            print(f'  [heard] task -> "{text}"')

        def reverted():
            """'nevermind' with nothing part-said: go back one task."""
            if len(spoken_stack) < 2:
                return None
            spoken_stack.pop()
            previous = spoken_stack[-1]
            policy.request_prompt(previous)
            print(f'  [heard] back to "{previous}"')
            return previous

        def vocabulary():
            """Bias the transcriber towards the tasks actually used here.

            Whisper's initial_prompt is a strong hint, and the task history is
            exactly the right vocabulary: these are short commands, repeated,
            full of object names no general model expects.
            """
            words = [] if panel_history is None else panel_history.prompts(8)
            words += [args.listen_terminator, args.listen_cancel]
            return ". ".join(words)

        listener = SpeechPrompter(
            spoken, on_cancel=reverted, model=args.listen_model,
            device=args.listen_device, alsa_device=args.listen_alsa,
            terminator=args.listen_terminator, cancel=args.listen_cancel,
            speak=not args.no_speak, speak_engine=args.speak_engine,
            speak_voice=args.speak_voice, hint=vocabulary)
        listener.start()
        voice = listener.status().get("voice")
        print(f'Listening: say the task, then "{args.listen_terminator}"; '
              f'"{args.listen_cancel}" to take it back. '
              f"(loading {args.listen_model} the first time takes a moment)")
        print(f"  talking back with: {voice or 'nothing — no speaker found'}")

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
    interrupted = False
    last_prompt = policy.prompt

    print(f'\nTask: "{prompt}"')
    bound = (f"up to {args.max_steps} steps"
             if args.max_steps > 0 else "until stopped")
    print(f"Running {bound} at {args.rate:.0f} Hz. "
          f"Ctrl+C to stop" + (", or q in the window" if args.visualize else "") + "." + ("  [DRY RUN — no motion]" if args.dry_run else ""))

    try:
        while args.max_steps <= 0 or step < args.max_steps:
            tick = time.perf_counter()

            # Cheap per-step read: the joint delta is applied to the *measured*
            # position, as DROID does. Images are only fetched when the policy
            # says it will re-infer, since /get_obs is ~10x the cost of
            # /get_state. Ask it rather than re-deriving the condition here:
            # a copy of it went stale the moment retasking could also force an
            # inference, and the rollout died on a state-only observation.
            obs = (robot.get_obs(include_depth=False) if policy.needs_inference
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
                panel_state.set(build_payload(policy, action, args, step,
                                              panel_analyser, obs, overlays,
                                              panel_history, listener))

            if viz is not None:
                keep_going = viz.update(
                    policy.last_request, action,
                    {"step": (f"{step}/{args.max_steps}" if args.max_steps > 0
                              else str(step)),
                     "infer": (f"{policy.last_inference_ms:.0f} ms"
                               if policy.last_inference_ms else "--"),
                     "chunk": f"{policy._chunk_index}/{policy.open_loop_horizon}",
                     "rate": f"{args.rate:g} Hz",
                     "speed": f"{args.speed_scale:g}x",
                     "mode": "DRY RUN" if args.dry_run else "live"})
                if not keep_going:
                    print("\nStopped from the visualizer (q).")
                    break

            if policy.prompt != last_prompt:
                print(f'  task -> "{policy.prompt}"')
                last_prompt = policy.prompt

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
        interrupted = True
    finally:
        if viz is not None:
            viz.close()
        if listener is not None:
            listener.stop()

    print(f"\nRan {step} steps, {policy.n_inferences} inferences.")

    # The panel dies with this process, and a rollout is short -- 600 steps at
    # 15 Hz is 40 seconds, which is not long enough to finish looking at it.
    # Ctrl+C during the rollout means "I am done", so only linger when the
    # rollout ended by itself.
    if panel_state is not None and not interrupted:
        print(f"The panel is still serving the last step at "
              f"http://localhost:{args.panel_port}/ — Ctrl+C to quit.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
