# third_party

Upstream checkouts, cloned rather than vendored. Nothing here is modified.

| Directory | Upstream | Used for |
|---|---|---|
| `openpi/` | github.com/Physical-Intelligence/openpi | pi0.5-DROID checkpoint, policy server, `openpi_client` websocket client |
| `droid/` | github.com/droid-dataset/droid | Reference for the DROID action space and gripper convention |

## Why droid/ is here

It is not imported at runtime. It is the authoritative source for two things
`deploy/deploy_pi05.py` must get right, both cited in `control/pi05_droid.py`:

- `droid/robot_ik/robot_ik_solver.py` - `max_joint_delta = 0.2`, so a
  normalised joint velocity of 1.0 means 0.2 rad per tick at 15 Hz.
- `droid/franka/robot.py:123` - `width = max_width * (1 - command)`, i.e. the
  gripper is 0 = open, 1 = closed. That is the opposite of crisp_py.

## Installing the client on the robot machine

Only `openpi_client` is needed here (few dependencies):

    pip install --no-deps -e third_party/openpi/packages/openpi-client
    pip install websockets msgpack pillow

`dm-tree` is listed as a dependency but is only imported by
`action_chunk_broker.py` and a test, neither of which this repo uses.

## Running the policy server

On a GPU machine (~4090). The `cd` matters - `scripts/serve_policy.py` is
openpi's, not this repo's, and running it from the crisp_py_new root fails
with "Failed to spawn: scripts/serve_policy.py".

    cd third_party/openpi
    GIT_LFS_SKIP_SMUDGE=1 uv sync          # first time only, installs jax etc.
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_droid \
        --policy.dir=gs://openpi-assets/checkpoints/pi05_droid

The `aloha` and `libero` submodules openpi's README mentions are not path
dependencies and are not needed for DROID inference.
