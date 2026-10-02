"""Forward kinematics and camera projection for this robot.

Where a point on the arm lands in a camera image, which is a question several
things ask: the robot panel drawing the gripper on its feeds, the pi0.5 panel
drawing a predicted trajectory, the hand-eye calibration checking its own
answer. The maths is the robot's, not any one consumer's, so it lives here
rather than in whichever tool needed it first.

    q --FK(URDF)--> T_base_tip --extrinsics--> p_cam --K,d--> pixel

Pure numpy/scipy on purpose: pinocchio lives in the separate `pink_solver`
conda env, and nothing here may depend on it.

Two kinds of camera are described by config/camera_info.yaml. One bolted to
the room has a constant pose in the robot base frame. One bolted to the
gripper does not -- only a constant pose relative to the link it rides, which
has to be composed with FK every frame. `CameraCalibration.pose_in_base`
hides the difference; `frame:` in the YAML entry is what distinguishes them.
"""

import dataclasses
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from control.kinematics import GEOMETRY_FILE, fingertip_offset_matrix

DEFAULT_URDF = Path(__file__).resolve().parent / "fr3_robot.urdf"
DEFAULT_CAMERA_INFO = Path(__file__).resolve().parent.parent / "config" / "camera_info.yaml"
# `base`, so FK over this chain is comparable to the measured pose.
BASE_LINK = "base"
# What the robot driver reports as the end-effector (crisp_py FrankaConfig
# target_frame), so FK to this frame can be checked against the measured pose.
TIP_LINK = "fr3_hand_tcp"

# ...which is the *Franka* hand TCP, not where the Robotiq actually grips. The
# fingertip -- the frame the policy and the training data speak -- is offset
# from it by the tool geometry in config/gripper_geometry.yaml. Drawing the
# fingertip means the overlay lands on the fingers in the picture; drawing
# 'ee' lands ~6 cm up the gripper body.
FRAMES = ("fingertip", "ee")


def rigid(rot: np.ndarray, xyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = xyz
    return T


@dataclasses.dataclass
class _Joint:
    name: str
    kind: str                 # 'revolute', 'continuous', 'fixed', ...
    origin: np.ndarray        # 4x4, parent -> joint
    axis: np.ndarray          # unit, in the joint frame
    limits: tuple | None = None   # (lower, upper) [rad], from the URDF


class Chain:
    """The base -> tip kinematic chain of a URDF, as plain matrices."""

    def __init__(self, joints: list[_Joint]):
        self.joints = joints
        self.actuated = [j for j in joints if j.kind in ("revolute", "continuous")]

    @property
    def n_joints(self) -> int:
        return len(self.actuated)

    @property
    def joint_limits(self) -> np.ndarray:
        """(n_joints, 2) lower/upper bounds, infinite where unstated."""
        return np.array([j.limits or (-np.inf, np.inf) for j in self.actuated])

    @classmethod
    def from_urdf(cls, path=None, base: str = BASE_LINK, tip: str = TIP_LINK) -> "Chain":
        """Parse `path` and keep only the joints between `base` and `tip`."""
        path = Path(path or DEFAULT_URDF)
        if not path.exists():
            raise FileNotFoundError(f"no URDF at {path}")
        root = ET.parse(path).getroot()

        by_child = {}
        for el in root.findall("joint"):
            child = el.find("child").get("link")
            parent = el.find("parent").get("link")
            origin = el.find("origin")
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0").split())] if origin is not None else [0, 0, 0]
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split())] if origin is not None else [0, 0, 0]
            axis_el = el.find("axis")
            axis = np.array([float(v) for v in axis_el.get("xyz").split()]) if axis_el is not None \
                else np.array([0.0, 0.0, 1.0])
            norm = np.linalg.norm(axis)
            limit = el.find("limit")
            limits = (None if limit is None or limit.get("lower") is None
                      else (float(limit.get("lower")), float(limit.get("upper"))))
            by_child[child] = (parent, _Joint(
                el.get("name"), el.get("type", "fixed"),
                rigid(Rotation.from_euler("xyz", rpy).as_matrix(), xyz),
                axis / norm if norm > 0 else axis, limits))

        # Walk up from the tip: a URDF is a tree, so each link has one parent.
        chain, link, guard = [], tip, 0
        while link != base:
            if link not in by_child:
                raise ValueError(
                    f"{path.name}: link {link!r} has no parent joint, so there is "
                    f"no chain from {base!r} to {tip!r}")
            parent, joint = by_child[link]
            chain.append(joint)
            link = parent
            guard += 1
            if guard > 512:
                raise ValueError(f"{path.name}: cycle while walking up from {tip!r}")
        chain.reverse()
        return cls(chain)

    def fk(self, q) -> np.ndarray:
        """Joint angles -> 4x4 pose of the tip in the base frame."""
        q = np.asarray(q, dtype=np.float64).ravel()
        if q.size < self.n_joints:
            raise ValueError(f"chain needs {self.n_joints} joint angles, got {q.size}")
        T = np.eye(4)
        i = 0
        for j in self.joints:
            T = T @ j.origin
            if j.kind in ("revolute", "continuous"):
                T = T @ rigid(Rotation.from_rotvec(j.axis * q[i]).as_matrix(), (0, 0, 0))
                i += 1
        return T

    def jacobian(self, q, offset: np.ndarray | None = None) -> np.ndarray:
        """Geometric Jacobian at the tip, (6, n_joints), in the base frame.

        Rows are [vx vy vz wx wy wz]: joint velocity -> tip twist. For a
        revolute joint the tip's linear velocity is the axis crossed into the
        lever arm from that joint to the tip, and the angular velocity is the
        axis itself.

        `offset` moves the reference point (the fingertip rather than the
        flange, say), which changes the lever arms and therefore the linear
        rows -- jogging about the wrong point feels like the arm pivoting
        around somewhere behind the gripper.
        """
        q = np.asarray(q, dtype=np.float64).ravel()
        if q.size < self.n_joints:
            raise ValueError(f"chain needs {self.n_joints} joint angles, got {q.size}")

        # One pass down the chain recording each joint's world axis and origin.
        T = np.eye(4)
        axes, origins = [], []
        i = 0
        for j in self.joints:
            T = T @ j.origin
            if j.kind in ("revolute", "continuous"):
                axes.append(T[:3, :3] @ j.axis)
                origins.append(T[:3, 3].copy())
                T = T @ rigid(Rotation.from_rotvec(j.axis * q[i]).as_matrix(), (0, 0, 0))
                i += 1
        tip = (T @ offset)[:3, 3] if offset is not None else T[:3, 3]

        jac = np.zeros((6, len(axes)))
        for k, (axis, origin) in enumerate(zip(axes, origins)):
            jac[:3, k] = np.cross(axis, tip - origin)
            jac[3:, k] = axis
        return jac

    def fk_batch(self, qs) -> np.ndarray:
        """(N, n_joints) -> (N, 4, 4). Loops; N is a chunk, so tens at most."""
        return np.stack([self.fk(q) for q in np.atleast_2d(qs)])


def fk_residual(chain: Chain, q, ee_position, ee_quat_xyzw=None) -> tuple[float, float]:
    """(position error [m], orientation error [deg]) of FK against the robot.

    The robot reports its own end-effector pose, so the chain can be checked
    against ground truth instead of trusted. A residual of millimetres means
    the model, the tip frame and the joint order all agree; centimetres means
    the overlay is drawing the wrong thing and should not be believed.
    """
    T = chain.fk(q)
    pos_err = float(np.linalg.norm(T[:3, 3] - np.asarray(ee_position, dtype=np.float64)))
    if ee_quat_xyzw is None:
        return pos_err, float("nan")
    dR = Rotation.from_matrix(T[:3, :3]).inv() * Rotation.from_quat(ee_quat_xyzw)
    return pos_err, float(np.degrees(np.linalg.norm(dR.as_rotvec())))


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class CameraCalibration:
    """Intrinsics + extrinsics for one camera.

    `k` and the image size are whatever the calibration was done at; `scaled_k`
    adapts them to the resolution the camera is actually streaming now, which
    is not the same (config/cameras.yaml runs the agentview at 848x480, the
    calibration file records 480x270).

    `parent` is which link the extrinsic is relative to. A camera bolted to the
    room is calibrated against `base` and its pose is constant. A camera bolted
    to the gripper is not: its pose is only fixed relative to the link it rides,
    and where it is in the robot's base frame depends on the arm's
    configuration, so it has to be composed with forward kinematics every
    frame. `pose_in_base` is what hides that difference from callers.
    """

    name: str
    k: np.ndarray                  # 3x3, at (calib_width, calib_height)
    dist: np.ndarray               # plumb_bob [k1, k2, p1, p2, k3]
    rot_base_cam: np.ndarray       # 3x3, p_parent = R @ p_cam + t
    pos_base_cam: np.ndarray       # 3,
    calib_size: tuple[int, int]    # (width, height)
    parent: str = BASE_LINK        # link the extrinsic is relative to

    @property
    def moving(self) -> bool:
        """True if this camera rides the arm and needs FK to be placed."""
        return self.parent != BASE_LINK

    def pose_in_base(self, chain: "Chain | None" = None, q=None):
        """(R, t) taking camera coordinates into the base frame, right now.

        For a static camera this is the calibration as stored. For one on a
        moving link it is T_base_parent(q) composed with the stored
        parent->camera transform, so the caller passes the same joint
        configuration the image was taken at.
        """
        if not self.moving:
            return self.rot_base_cam, self.pos_base_cam
        if chain is None or q is None:
            raise ValueError(
                f"camera {self.name!r} is calibrated against {self.parent!r}, "
                f"which moves: placing it needs the arm's joint positions")
        T = chain.fk(q)
        return T[:3, :3] @ self.rot_base_cam, T[:3, :3] @ self.pos_base_cam + T[:3, 3]

    @classmethod
    def load(cls, camera: str, path=None) -> "CameraCalibration | None":
        """Read one camera out of camera_info.yaml, or None if it is absent.

        Absent is the normal case for the wrist camera: it moves with the arm,
        so a static base->camera transform does not describe it and there is
        no hand-eye calibration for it in this repo.
        """
        import yaml

        path = Path(path or DEFAULT_CAMERA_INFO)
        if not path.exists():
            return None
        data = yaml.safe_load(path.read_text()) or {}
        cfg = data.get(camera)
        if not cfg or "k" not in cfg or "q" not in cfg or "t" not in cfg:
            return None
        return cls(
            name=camera,
            k=np.asarray(cfg["k"], dtype=np.float64).reshape(3, 3),
            dist=np.asarray(cfg.get("d") or [0.0] * 5, dtype=np.float64).ravel(),
            # scipy's from_quat is [x, y, z, w], which is the order the file
            # documents in its header.
            rot_base_cam=Rotation.from_quat(np.asarray(cfg["q"], dtype=np.float64)).as_matrix(),
            pos_base_cam=np.asarray(cfg["t"], dtype=np.float64),
            calib_size=(int(cfg["width"]), int(cfg["height"])),
            # Absent means the historical meaning of this file: a static
            # camera whose t/q are in the robot base frame.
            parent=str(cfg.get("frame") or cfg.get("parent") or BASE_LINK),
        )

    def scaled_k(self, frame_size) -> np.ndarray:
        """Intrinsics for a frame of `frame_size` = (width, height)."""
        w, h = frame_size
        cw, ch = self.calib_size
        k = self.k.copy()
        k[0, :] *= w / cw
        k[1, :] *= h / ch
        return k

    def aspect_mismatch(self, frame_size) -> float:
        """Relative aspect-ratio difference between the live frame and the
        calibration. Scaling intrinsics is only exact when this is 0; a few
        percent shifts the projection by a few pixels near the edges."""
        w, h = frame_size
        cw, ch = self.calib_size
        return abs((w / h) / (cw / ch) - 1.0)

    def project(self, points_base, frame_size, chain=None, q=None):
        """(N, 3) base-frame points -> ((N, 2) raw pixels, (N,) in-front mask).

        `chain`/`q` are only needed for a camera on a moving link; see
        `pose_in_base`.
        """
        rot, pos = self.pose_in_base(chain, q)
        pts = np.atleast_2d(np.asarray(points_base, dtype=np.float64))
        cam = (pts - pos) @ rot                                  # R^T (p - t)
        z = cam[:, 2]
        in_front = z > 1e-6
        safe_z = np.where(in_front, z, 1.0)
        x, y = cam[:, 0] / safe_z, cam[:, 1] / safe_z

        # plumb_bob, the model the calibration file names.
        k1, k2, p1, p2, k3 = (list(self.dist) + [0.0] * 5)[:5]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

        k = self.scaled_k(frame_size)
        u = k[0, 0] * xd + k[0, 1] * yd + k[0, 2]
        v = k[1, 1] * yd + k[1, 2]
        return np.stack([u, v], axis=1), in_front



def calibration_key_for(camera: str) -> str:
    """config/cameras.yaml's `calibration:` for this camera, else its name."""
    try:
        from control.realsense_direct import calibration_key
    except Exception:
        return camera
    return calibration_key(camera)


def frame_offset(frame: str = "fingertip") -> np.ndarray:
    """Right-multiplied offset from the FK tip frame to `frame`.

    Uses the same configured transform as the observation/action pipeline
    (control.kinematics), so the overlay cannot drift from the frame the
    policy is actually commanded in: both read
    config/gripper_geometry.yaml.
    """
    if frame == "ee":
        return np.eye(4)
    if frame != "fingertip":
        raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
    return fingertip_offset_matrix()


def axis_tips(pose: np.ndarray, length: float = 0.05) -> np.ndarray:
    """The three axis endpoints of a pose, for drawing a triad."""
    return pose[:3, 3] + pose[:3, :3].T * length




def damped_least_squares(jacobian: np.ndarray, twist, damping: float = 0.05):
    """Joint step that best produces `twist`, staying finite near singularities.

    The plain pseudo-inverse blows up when the arm approaches a singularity --
    a small commanded twist asks for an enormous joint velocity, which is
    exactly the moment a human is jogging near a limit. The damping term trades
    a little tracking accuracy for a bounded answer:

        dq = J^T (J J^T + lambda^2 I)^-1 * twist
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    twist = np.asarray(twist, dtype=np.float64).ravel()
    jjt = jacobian @ jacobian.T
    return jacobian.T @ np.linalg.solve(
        jjt + (damping ** 2) * np.eye(jjt.shape[0]), twist)


class ToolProjector:
    """Where the gripper is, in one camera's image.

    The smallest useful thing that can be built out of this module: FK to the
    tool frame, the camera's calibration, and the pixel that comes out. The
    robot panel draws it on its feeds; it is also the check that a calibration
    is right, since the gripper is visible in the picture and either the cross
    sits on it or the numbers are wrong.

    Handles both kinds of camera. A static one is projected with its stored
    pose; one that rides the arm is placed by FK to its mount link first,
    which is why it needs the joint positions either way.
    """

    def __init__(self, camera: str, calibration: CameraCalibration,
                 chain: Chain, mount_chain: Chain | None = None,
                 frame: str = "fingertip"):
        self.camera = camera
        self.calibration = calibration
        self.chain = chain
        self.mount_chain = mount_chain
        self.frame = frame
        self.offset = frame_offset(frame)

    @classmethod
    def for_camera(cls, camera: str, urdf=None, camera_info=None,
                   chain: Chain | None = None, frame: str = "fingertip",
                   calibration_key: str | None = None) -> "ToolProjector | None":
        """Build one, or None if this camera has no calibration to build from."""
        calibration = CameraCalibration.load(
            calibration_key or calibration_key_for(camera), camera_info)
        if calibration is None:
            return None
        chain = chain or Chain.from_urdf(urdf)
        mount = (Chain.from_urdf(urdf, tip=calibration.parent)
                 if calibration.moving else None)
        return cls(camera, calibration, chain, mount, frame)

    def project(self, joints, frame_size, axis_length: float = 0.05,
                frame: str | None = None) -> dict | None:
        """Joint positions -> the tool's pixel and a small frame triad.

        Coordinates are normalised to the frame, so a caller drawing on a
        scaled copy of the image does not need to know the resolution.

        `frame` overrides the projector's own for this call only: one
        projector is shared by every request the server answers, so a caller
        asking for 'ee' must not change what the next caller gets.
        """
        joints = np.asarray(joints, dtype=np.float64).ravel()[:7]
        offset = self.offset if frame in (None, self.frame) else frame_offset(frame)
        pose = self.chain.fk(joints) @ offset
        points = np.vstack([pose[:3, 3], axis_tips(pose, axis_length)])
        uv, front = self.calibration.project(
            points, frame_size, self.mount_chain, joints)
        uv = uv / np.asarray(frame_size, dtype=np.float64)
        return {
            "camera": self.camera,
            "frame": frame or self.frame,
            "mounted_on": self.calibration.parent,
            "moving": self.calibration.moving,
            "point": [round(float(v), 5) for v in uv[0]],
            "axes": [[round(float(v), 5) for v in row] for row in uv[1:]],
            "front": bool(front[0]),
            # Outside 0..1 the tool is behind the camera or out of frame; the
            # caller can say "off screen" instead of drawing at the edge.
            "in_frame": bool(front[0] and 0.0 <= uv[0][0] <= 1.0
                             and 0.0 <= uv[0][1] <= 1.0),
        }
