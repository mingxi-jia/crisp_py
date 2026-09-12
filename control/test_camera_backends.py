"""The two camera backends must be interchangeable.

robot_server.get_obs() once called cameras.frame(), which only CameraHub has,
so /get_obs worked on the ROS backend and raised AttributeError on the direct
one. Nothing caught it because "drop-in compatible" was checked against a list
written by hand rather than against what the consumers actually call.

    python -m control.test_camera_backends
"""

import inspect
import re
from pathlib import Path

# Every attribute a consumer is allowed to use on a camera backend.
REQUIRED = [
    "camera_names", "rgb", "depth", "is_stale", "status",
    "ready_cameras", "stale_cameras",
    "wait_for_any", "wait_for_all", "clear_cache", "destroy_node",
]

CONSUMERS = ["control/robot_server.py", "control/observer.py"]
ROOT = Path(__file__).resolve().parent.parent


def backends():
    from control.cameras import CameraHub
    from control.realsense_direct import RealsenseCameras
    return {"CameraHub": CameraHub, "RealsenseCameras": RealsenseCameras}


def main():
    ok = True

    print("=== every backend implements the required surface ===")
    for name, cls in backends().items():
        # camera_names is set in __init__, so hasattr on the class is False;
        # look for the assignment instead of instantiating (which needs hardware).
        src = inspect.getsource(cls)
        missing = [m for m in REQUIRED
                   if not hasattr(cls, m) and f"self.{m} =" not in src]
        print(f"  {'PASS' if not missing else 'FAIL'}  {name}"
              + (f"  missing: {missing}" if missing else ""))
        ok &= not missing

    print("\n=== compatible signatures for the methods consumers call ===")
    hub, direct = backends()["CameraHub"], backends()["RealsenseCameras"]
    for m in ("rgb", "depth", "is_stale", "status"):
        a = inspect.signature(getattr(hub, m)).parameters
        b = inspect.signature(getattr(direct, m)).parameters
        shared = set(a) & set(b)
        agree = "name" in shared or m == "status"
        print(f"  {'PASS' if agree else 'FAIL'}  {m}({', '.join(list(a)[1:])}) "
              f"vs ({', '.join(list(b)[1:])})")
        ok &= agree

    print("\n=== consumers only touch the shared surface ===")
    allowed = set(REQUIRED)
    for rel in CONSUMERS:
        src = (ROOT / rel).read_text()
        # Observation.cameras is a dict of frames, not a backend; scanning its
        # class body would flag .get()/.items() as backend calls.
        src = re.sub(r'\nclass Observation\b.*?(?=\nclass |\n# ---|\Z)', '\n', src, flags=re.S)
        used = set()
        for line in src.splitlines():
            for m in re.finditer(r'\bself\.(?:_)?cameras\.([A-Za-z_]\w*)', line):
                used.add(m.group(1))
        extra = sorted(used - allowed)
        print(f"  {'PASS' if not extra else 'FAIL'}  {rel}"
              + (f"  uses {extra}, not on both backends" if extra else
                 f"  ({len(used)} call sites, all shared)"))
        ok &= not extra

    print("\n" + ("ALL BACKEND CONFORMANCE CHECKS PASSED" if ok else "FAILURES ABOVE"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
