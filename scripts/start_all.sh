#!/usr/bin/env bash
#
# All-in-one bring-up: environment, cameras, robot server, control panel.
#
#   ./scripts/start_all.sh                # everything, open the panel
#   ./scripts/start_all.sh --home         # ALSO move the arm to its home pose
#                                         #   (off by default: bring-up does
#                                         #    not move the robot)
#   ./scripts/start_all.sh --no-panel     # skip opening a browser
#   ./scripts/start_all.sh --port 7001
#   ./scripts/start_all.sh --no-cameras   # reuse cameras started elsewhere
#   ./scripts/start_all.sh --stop         # stop everything and exit (no hold)
#   ./scripts/start_all.sh --keep-running # leave everything up when it exits
#   ./scripts/start_all.sh --camera-launch control/launch_cameras.launch.py
#   ./scripts/start_all.sh --ctrl-space joint   # required for deploy_pi05
#   ./scripts/start_all.sh --camera-backend ros  # ROS camera topics instead
#   ./scripts/start_all.sh --compliance compliant  # gives way on contact
#
# Anything already running is reused rather than started twice. Ctrl+C stops
# the whole stack -- cameras and robot server -- whether this invocation
# started them or adopted them, so there is no separate teardown step.
# Use --keep-running to leave everything up on exit.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------- defaults --
PORT=7000
CAMERA_LAUNCH="control/launch_cameras.launch.py"
CONDA_ENV="ros_env"
CONFIG_PATH=""
CTRL_SPACE="cartesian"
CAMERA_BACKEND="realsense"
COMPLIANCE="normal"
LOG_DIR="${TMPDIR:-/tmp}/crisp-startup"
OPEN_PANEL=1
SKIP_CAMERAS=0
FORCE_CAMERAS=0
STOP_ONLY=0
KEEP_RUNNING=0
SKIP_SERVER=0
SERVER_ARGS=()
CAMERA_TIMEOUT=90
SERVER_TIMEOUT=180

while [ $# -gt 0 ]; do
  case "$1" in
    --port)           PORT="$2"; shift 2 ;;
    --camera-launch)  CAMERA_LAUNCH="$2"; shift 2 ;;
    --conda-env)      CONDA_ENV="$2"; shift 2 ;;
    --config-path)    CONFIG_PATH="$2"; shift 2 ;;
    --ctrl-space)     CTRL_SPACE="$2"; shift 2 ;;
    --camera-backend) CAMERA_BACKEND="$2"; shift 2 ;;
    --compliance)     COMPLIANCE="$2"; shift 2 ;;
    --log-dir)        LOG_DIR="$2"; shift 2 ;;
    --no-panel)       OPEN_PANEL=0; shift ;;
    --no-cameras)     SKIP_CAMERAS=1; shift ;;
    --force-cameras)  FORCE_CAMERAS=1; shift ;;
    --stop)           STOP_ONLY=1; shift ;;
    --keep-running)   KEEP_RUNNING=1; shift ;;
    --no-server)      SKIP_SERVER=1; shift ;;
    --home)           SERVER_ARGS+=("--home"); shift ;;
    --no-home)        shift ;;   # now the default; accepted so old invocations still work
    --ft-sensor-on)   SERVER_ARGS+=("--ft-sensor-on"); shift ;;
    -h|--help)        sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config}"
[ -d "$CONFIG_PATH" ] || { echo "config dir not found: $CONFIG_PATH" >&2; exit 2; }

# Everything below is derived from $CAMERA_LAUNCH so any launch file works.
CAMERA_LAUNCH_NAME="$(basename "$CAMERA_LAUNCH" .launch.py)"
# Camera node names declared in the launch file, e.g. "cam1 cam4".
launch_camera_names() {
  [ -f "$CAMERA_LAUNCH" ] || return 0
  {
    grep -oE "camera_name'[[:space:]]*:[[:space:]]*'[a-z0-9_]+" "$CAMERA_LAUNCH" \
      | grep -oE "'[a-z0-9_]+$" | tr -d "'"
    grep -oE "name='cam[0-9]+'" "$CAMERA_LAUNCH" | grep -oE "cam[0-9]+"
  } 2>/dev/null | sort -u
}

mkdir -p "$LOG_DIR"
CAM_LOG="$LOG_DIR/cameras.log"
SRV_LOG="$LOG_DIR/robot_server.log"
PANEL_URL="http://localhost:${PORT}/panel"

# ------------------------------------------------------------------ output --
if [ -t 1 ]; then B=$'\033[1m'; DIM=$'\033[2m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
else B=""; DIM=""; G=""; Y=""; R=""; N=""; fi
step() { printf "%s\n" "${B}==> $*${N}"; }
info() { printf "    %s\n" "$*"; }
ok()   { printf "    %s\n" "${G}ok${N} $*"; }
warn() { printf "    %s\n" "${Y}!!${N} $*"; }
die()  { printf "%s\n" "${R}xx $*${N}" >&2; exit 1; }

# ----------------------------------------------------------------- cleanup --
STARTED_PIDS=()
STARTED_CAMERAS=0
# Stops the whole stack, not just what this invocation happened to start.
# Reused processes are adopted: running the script makes it the owner of the
# cameras and the robot server, so one Ctrl+C leaves nothing behind and there
# is no separate teardown step to remember.
cleanup() {
  local code=$?
  trap - EXIT INT TERM

  if [ "$KEEP_RUNNING" -eq 1 ]; then
    echo
    info "--keep-running: leaving cameras and the robot server up"
    exit "$code"
  fi

  echo
  step "Shutting down"

  # -- robot server ----------------------------------------------------
  if server_up || pgrep -f 'control\.robot_server' >/dev/null 2>&1; then
    info "stopping the robot server"
    # Wait on the PROCESS, not on HTTP. server_up() is a curl with a 2 s
    # timeout, so polling it 20 times could burn ~45 s just to notice the
    # process had already exited.
    srv_pids="$(pgrep -f 'control\.robot_server' 2>/dev/null | tr '\n' ' ')"
    if [ -n "$srv_pids" ]; then
      kill -TERM $srv_pids 2>/dev/null || true
      waited=0
      while [ "$waited" -lt 60 ] && pgrep -f 'control\.robot_server' >/dev/null 2>&1; do
        sleep 0.25; waited=$((waited + 1))
      done
      if pgrep -f 'control\.robot_server' >/dev/null 2>&1; then
        warn "server ignored TERM after 15 s; forcing"
        pkill -KILL -f 'control\.robot_server' 2>/dev/null || true
        sleep 0.5
      fi
    fi
    pgrep -f 'control\.robot_server' >/dev/null 2>&1 \
      && warn "server still running" || ok "robot server stopped"
  fi

  # -- our own launch process groups (kills ros2 launch and its children) --
  local pid
  for pid in "${STARTED_PIDS[@]-}"; do
    [ -n "$pid" ] || continue
    kill -0 "$pid" 2>/dev/null || continue
    kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  pkill -f "$CAMERA_LAUNCH_NAME" 2>/dev/null || true

  # -- camera nodes, whoever started them -------------------------------
  local n
  n="$(realsense_count)"
  if [ "$n" -gt 0 ]; then
    info "stopping $n camera node(s)"
    realsense_pids | xargs -r kill -TERM 2>/dev/null || true
    for _ in $(seq 24); do [ "$(realsense_count)" -eq 0 ] && break; sleep 0.25; done
    if [ "$(realsense_count)" -ne 0 ]; then
      warn "camera nodes did not exit on TERM; forcing"
      realsense_pids | xargs -r kill -KILL 2>/dev/null || true
      sleep 1
    fi
    # The nodes run under setsid, so they never see the terminal's Ctrl+C --
    # this trap is the only thing that stops them. Confirm the USB devices
    # actually came free, or the next launch (and realsense-viewer) find them
    # busy.
    [ "$(realsense_count)" -eq 0 ] \
      && ok "cameras released" \
      || warn "still holding devices; pids: $(realsense_pids | tr '\n' ' ')"
  else
    ok "no camera nodes running"
  fi

  ok "stopped"
  exit "$code"
}

trap cleanup EXIT INT TERM

server_up() { curl -sf --max-time 2 "http://localhost:${PORT}/health" >/dev/null 2>&1; }
# NOTE: do not write this as `ros2 topic list | grep -q ...`. Under
# `set -o pipefail`, grep -q exits on the first match and closes the pipe,
# ros2 dies of SIGPIPE, and pipefail propagates that failure -- so the check
# reports "no cameras" exactly when it *does* find them, at random. Capture
# the output first so nothing can receive SIGPIPE.
# Ready when every camera the launch file declares is publishing colour.
# Both /color/image_raw and /color/image_rect_raw are accepted -- in-hand
# cameras are commonly configured to publish the rectified topic instead.
cameras_up() {
  local topics cam found=0 total=0
  topics="$(timeout 20 ros2 topic list 2>/dev/null)" || true
  [ -n "$topics" ] || return 1
  for cam in $CAMERA_NAMES; do
    total=$((total + 1))
    case "$topics" in
      *"/$cam/color/image_raw"*|*"/$cam/color/image_rect_raw"*) found=$((found + 1)) ;;
    esac
  done
  [ "$total" -gt 0 ] && [ "$found" -eq "$total" ]
}
# Device-level guard: launching a second set of realsense nodes fights the
# first over the USB devices and can take BOTH sets down.
# Identify camera nodes by their actual executable, read from /proc/<pid>/exe.
# The two obvious shortcuts are both wrong:
#   pgrep -x realsense2_camera_node  -> never matches. Linux caps /proc/<pid>/comm
#                                       at 15 chars, so the name is
#                                       "realsense2_came".
#   pgrep -f realsense2_camera_node  -> over-matches: this script's own command
#                                       line contains the pattern.
realsense_pids() {
  local d pid exe
  for d in /proc/[0-9]*; do
    pid="${d#/proc/}"
    exe="$(readlink -f "$d/exe" 2>/dev/null)" || continue
    case "$exe" in */realsense2_camera_node) echo "$pid" ;; esac
  done
}
realsense_count()   { realsense_pids | wc -l | tr -d ' '; }
realsense_running() { [ "$(realsense_count)" -gt 0 ]; }

wait_for_cameras() {   # $1 = seconds, $2 = pid to watch (optional)
  printf "    waiting for camera topics"
  local i
  for i in $(seq "$1"); do
    if [ -n "${2:-}" ] && ! kill -0 "$2" 2>/dev/null; then
      echo; return 2
    fi
    cameras_up && { echo " (${i}s)"; return 0; }
    printf "."; sleep 1
  done
  echo; return 1
}

# ================================================================== --stop ==
# Independent of the trap: recovers from `kill -9`, a crash, or cameras started
# by hand. Use this before running realsense-viewer, which cannot open a device
# another process still holds.
if [ "$STOP_ONLY" -eq 1 ]; then
  step "Stopping cameras and robot server"
  CAMERA_NAMES="$(launch_camera_names | tr '\n' ' ')"

  if server_up; then
    info "robot server is up on port $PORT; stopping"
    pkill -f 'control\.robot_server' 2>/dev/null || true
    # Wait on the process, not on a 2 s-timeout curl (see cleanup()).
    waited=0
    while [ "$waited" -lt 60 ] && pgrep -f 'control\.robot_server' >/dev/null 2>&1; do
      sleep 0.25; waited=$((waited + 1))
    done
    pgrep -f 'control\.robot_server' >/dev/null 2>&1 \
      && warn "server still running" || ok "robot server stopped"
  else
    info "no robot server on port $PORT"
  fi

  n="$(realsense_count)"
  if [ "$n" -eq 0 ]; then
    ok "no realsense2_camera_node processes running"
  else
    info "stopping $n realsense2_camera_node process(es)"
    pkill -f "$CAMERA_LAUNCH_NAME" 2>/dev/null || true
    realsense_pids | xargs -r kill -TERM 2>/dev/null || true
    for _ in $(seq 20); do [ "$(realsense_count)" -eq 0 ] && break; sleep 0.25; done
    if [ "$(realsense_count)" -ne 0 ]; then
      warn "did not exit on TERM; sending KILL"
      realsense_pids | xargs -r kill -KILL 2>/dev/null || true
      sleep 1
    fi
    [ "$(realsense_count)" -eq 0 ] && ok "camera nodes stopped" \
      || die "camera nodes still running: $(realsense_pids | tr '\n' ' ')"
  fi

  # Prove the USB devices are actually claimable, not just that the pids died.
  step "USB devices"
  if command -v rs-enumerate-devices >/dev/null 2>&1; then
    if out="$(timeout 30 rs-enumerate-devices -s 2>/dev/null)" \
       && [ "$(printf '%s\n' "$out" | grep -c 'Intel RealSense')" -gt 0 ]; then
      printf '%s\n' "$out" | sed 's/^/    /'
      ok "devices are free and enumerable"
    else
      warn "enumeration returned nothing — try unplugging/replugging the cameras"
    fi
  else
    info "rs-enumerate-devices not on PATH; skipping enumeration check"
  fi

  echo
  info "realsense-viewer can now open the cameras"
  info "bring everything back with: $0"
  trap - EXIT INT TERM
  exit 0
fi

# =========================================================== 1. environment ==
step "1/4  Environment"

[ -f init_crispy.sh ] || die "init_crispy.sh not found in $REPO_ROOT"

# Initialise conda *before* sourcing init_crispy.sh, which calls
# `conda activate` at the top.
# `conda activate` needs the shell *function*, which only exists after
# conda.sh is sourced; the bare binary on PATH is not enough.
if [ "$(type -t conda || true)" != "function" ]; then
  for hook in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$hook" ] && { . "$hook"; break; }
  done
fi
[ "$(type -t conda || true)" = "function" ] \
  || die "conda shell function unavailable; run 'conda init bash', or activate '$CONDA_ENV' yourself and re-run"

# init_crispy.sh's first line is `myconda`, which is an *alias* in ~/.bashrc.
# Aliases are not expanded in non-interactive shells, so it would abort the
# source. conda is already initialised above, so shim it as a no-op function
# (functions win over an undefined alias and work non-interactively).
myconda() { :; }

# init_crispy.sh hardcodes the *other* checkout's config dir, but uses
# ${VAR:-default}, so exporting first wins. Set these unconditionally rather
# than deferring to whatever a previous `source init_crispy.sh` left in the
# shell -- bring-up should not depend on inherited state.
export CRISP_CONFIG_PATH="$CONFIG_PATH"
export CYCLONEDDS_URI="$CONFIG_PATH/cyclone_config.xml"

# Tolerate errors inside init_crispy.sh; we verify what matters below.
# -u is lifted too: the script references ${pwd}, a typo for $(pwd), which is
# an unbound variable.
set +eu
# shellcheck disable=SC1091
source ./init_crispy.sh
init_rc=$?
set -eu
[ "$init_rc" -eq 0 ] || warn "init_crispy.sh returned $init_rc (continuing)"

if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
  conda activate "$CONDA_ENV" || die "could not activate conda env '$CONDA_ENV'"
fi

# Make `python -m control...` / `deploy...` resolve from the repo root
# regardless of how PYTHONPATH was set.
case ":${PYTHONPATH:-}:" in
  *":$REPO_ROOT:"*) ;;
  *) export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

# ros_env's activate hook exports PYTHONPATH and its deactivate hook leaves it
# set; harmless inside ros_env itself, but print it so surprises are visible.
info "repo            $REPO_ROOT"
info "conda env       ${CONDA_DEFAULT_ENV:-<none>}  ($(python -V 2>&1))"
info "ROS_DOMAIN_ID   ${ROS_DOMAIN_ID:-<unset>}"
info "RMW             ${RMW_IMPLEMENTATION:-<unset>}"
info "CRISP_CONFIG    $CRISP_CONFIG_PATH"
info "logs            $LOG_DIR"
python -c "import rclpy" 2>/dev/null || die "rclpy not importable in '$CONDA_DEFAULT_ENV'"
ok "environment ready"

# =============================================================== 2. cameras ==
step "2/4  Cameras"
if [ "$CAMERA_BACKEND" = "realsense" ]; then
  # The robot server opens the devices itself (config/cameras.yaml). Starting
  # ROS camera nodes here would contend for the same USB devices -- only one
  # process may own a RealSense.
  info "backend 'realsense': the robot server opens the cameras directly"
  if realsense_running; then
    warn "ROS camera nodes are running and hold the devices; stopping them"
    pkill -f "$CAMERA_LAUNCH_NAME" 2>/dev/null || true
    realsense_pids | xargs -r kill -TERM 2>/dev/null || true
    for _ in $(seq 24); do [ "$(realsense_count)" -eq 0 ] && break; sleep 0.25; done
    [ "$(realsense_count)" -eq 0 ] && ok "ROS camera nodes stopped" \
      || warn "camera nodes still running: $(realsense_pids | tr '\n' ' ')"
  fi
  info "cameras      from config/cameras.yaml"
else
CAMERA_NAMES="$(launch_camera_names | tr '\n' ' ')"
CAMERA_NAMES="${CAMERA_NAMES% }"
if [ -z "$CAMERA_NAMES" ]; then
  warn "could not read camera names from $CAMERA_LAUNCH; assuming cam1"
  CAMERA_NAMES="cam1"
fi
info "launch file  $CAMERA_LAUNCH"
info "cameras      $CAMERA_NAMES"

# The first `ros2 topic list` in a fresh process tree can return a partial
# list while the daemon warms up. Burn one call before deciding anything.
timeout 20 ros2 topic list >/dev/null 2>&1 || true

if [ "$SKIP_CAMERAS" -eq 1 ]; then
  warn "skipped (--no-cameras)"
elif cameras_up; then
  ok "already publishing /cam1/color/image_raw — reusing"
else
  if realsense_running && [ "$FORCE_CAMERAS" -eq 0 ]; then
    # Nodes exist but topics are not visible. Starting a second launch here
    # would contend for the USB devices, so wait instead of stacking.
    warn "realsense nodes are already running but topics are not visible yet"
    if wait_for_cameras "$CAMERA_TIMEOUT"; then
      ok "existing cameras came up — reusing"
    else
      die "realsense nodes are running but not publishing after ${CAMERA_TIMEOUT}s.
    Refusing to start a second launch (it would fight over the USB devices and
    can take both sets down). Either:
      pkill -f realsense2_camera_node   # then re-run
      $0 --force-cameras                # launch anyway, at your own risk"
    fi
  else
    [ -f "$CAMERA_LAUNCH" ] || die "$CAMERA_LAUNCH not found"
    [ "$FORCE_CAMERAS" -eq 1 ] && realsense_running \
      && warn "--force-cameras: launching alongside existing realsense nodes"
    info "ros2 launch $CAMERA_LAUNCH   (starts: $CAMERA_NAMES)"
    info "log: $CAM_LOG"
    setsid ros2 launch "$CAMERA_LAUNCH" >"$CAM_LOG" 2>&1 &
    cam_pid=$!
  # Detach from job control so bash does not print "Killed" on stop.
  disown "$cam_pid" 2>/dev/null || true
    STARTED_PIDS+=("$cam_pid")
    STARTED_CAMERAS=1
    # Call it directly and read $?. Capturing it in $( ) would swallow the
    # progress output into the value being matched, so the case could never
    # match "0" and every launch looked like a failure.
    wait_for_cameras "$CAMERA_TIMEOUT" "$cam_pid" && wait_rc=0 || wait_rc=$?
    case "$wait_rc" in
      0) ok "cameras streaming: $CAMERA_NAMES (pid $cam_pid)" ;;
      2) tail -20 "$CAM_LOG" >&2; die "camera launch died — see $CAM_LOG" ;;
      *) tail -20 "$CAM_LOG" >&2
         die "cameras did not publish within ${CAMERA_TIMEOUT}s — see $CAM_LOG" ;;
    esac
  fi
fi

fi

# ========================================================== 3. robot server ==
step "3/4  Robot server"
if [ -z "${CAMERA_NAMES:-}" ]; then
  CAMERA_NAMES="$(launch_camera_names | tr '\n' ' ')"
  CAMERA_NAMES="${CAMERA_NAMES% }"
fi
if [ "$SKIP_SERVER" -eq 1 ]; then
  warn "skipped (--no-server)"
elif server_up; then
  ok "already answering on port $PORT — reusing"
else
  # Hand the server the cameras this launch file actually brings up; without
  # it the server auto-discovers, which on a shared ROS domain can also find
  # other machines' cameras.
  server_cameras="$(printf '%s' "$CAMERA_NAMES" | tr ' ' ',')"
  info "python -m control.robot_server --camera-backend $CAMERA_BACKEND --ctrl-space $CTRL_SPACE --port $PORT ${SERVER_ARGS[*]:-}"
  if [[ " ${SERVER_ARGS[*]:-} " == *" --home "* ]]; then
    warn "--home given: THE ARM WILL MOVE to its home pose on startup"
  else
    info "the arm will not move on startup (pass --home if you want homing)"
  fi
  info "log: $SRV_LOG"
  setsid python -m control.robot_server \
      --camera-backend "$CAMERA_BACKEND" \
      --compliance "$COMPLIANCE" \
      --ctrl-space "$CTRL_SPACE" \
      --port "$PORT" \
      "${SERVER_ARGS[@]+"${SERVER_ARGS[@]}"}" >"$SRV_LOG" 2>&1 &
  srv_pid=$!
  # Detach from job control so bash does not print "Killed" on stop.
  disown "$srv_pid" 2>/dev/null || true
  STARTED_PIDS+=("$srv_pid")
  printf "    waiting for /health"
  for i in $(seq "$SERVER_TIMEOUT"); do
    kill -0 "$srv_pid" 2>/dev/null || { echo; echo; tail -25 "$SRV_LOG" >&2; echo; die "robot server died — see $SRV_LOG"; }
    server_up && break
    printf "."; sleep 1
  done
  echo
  server_up || die "server not healthy after ${SERVER_TIMEOUT}s — see $SRV_LOG"
  ok "robot server ready (pid $srv_pid)"
fi

# ================================================================= 4. panel ==
step "4/4  Control panel"
if server_up; then
  ok "$PANEL_URL"
  if [ "$OPEN_PANEL" -eq 1 ]; then
    if command -v xdg-open >/dev/null; then
      (xdg-open "$PANEL_URL" >/dev/null 2>&1 &)
      info "opened in your browser"
    else
      warn "xdg-open not found — open the URL above manually"
    fi
  fi
else
  warn "server not reachable; panel unavailable"
fi

echo
step "Ready"
info "panel     $PANEL_URL"
info "teleop    python -m deploy.deploy_spacemouse --url http://localhost:${PORT}"
info "logs      tail -f $SRV_LOG"
echo
if [ "$KEEP_RUNNING" -eq 1 ]; then
  info "${DIM}--keep-running: exiting now, leaving everything up${N}"
  trap - EXIT INT TERM
  exit 0
fi
info "${DIM}Ctrl+C stops the cameras and the robot server${N}"
echo

# Hold the foreground even when everything was already running and this
# invocation started nothing itself: the script adopts what it found, so it
# has to stay alive to be the thing you Ctrl+C. A bare `wait` returns straight
# away with no jobs of its own, so park on a sleep instead.
while :; do
  sleep 3600 &
  wait $! || true
done
