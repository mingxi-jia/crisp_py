"""Hand-eye calibration maths for a camera that rides the arm.

Separate from the script that drives it (scripts/calibrate_handeye.py) so the
solver can be run on saved samples, and tested against synthetic data where
the answer is known -- see scripts/test_calibrate_handeye.py.

What is being solved
--------------------
The in-hand camera is bolted to the gripper, so the unknown is one fixed
transform, T_gripper_cam. With a target that does not move, every sample gives

    T_base_target = T_base_gripper(i) @ T_gripper_cam @ T_cam_target(i)

and the left side is the same for all i. That is the classic AX = XB problem;
cv2.calibrateHandEye solves it. The camera's own intrinsics come out of the
same images (cv2.calibrateCamera), so nothing has to be known in advance.

The number that matters is not the solver's residual but `spread`: how much
T_base_target actually varies once the solution is applied. A target that is
bolted to the table cannot move, so whatever spread is left is error.
"""

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation

# Enough views to constrain intrinsics; hand-eye alone needs fewer, but the
# two are solved from the same captures.
MIN_SAMPLES = 6


def rigid(rot, pos) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = np.asarray(pos).ravel()
    return T


def invert(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
class ArucoMarker:
    """A single printed ArUco marker.

    Four corners per view: enough to fix the marker's pose, not enough to also
    determine the lens, so intrinsics have to come from somewhere else. In
    exchange it is the one target that needs no printing beyond a tag that is
    probably already on the table.

    Object points are in OpenCV's marker frame -- origin at the centre, +z out
    of the face, corners clockwise from the top-left -- so the pose returned
    matches what cv2.aruco would give.
    """

    kind = "aruco"

    def __init__(self, size: float, dictionary: str = "DICT_ARUCO_ORIGINAL",
                 marker_id: int | None = None):
        import cv2

        if not size or size <= 0:
            raise ValueError(
                "the marker's side length is required and cannot be guessed: "
                "every distance in the result scales with it, so a 5 mm error "
                "on a 50 mm tag is a 10% error on the camera offset")
        self.size = float(size)
        self.dictionary_name = dictionary
        self.marker_id = marker_id
        half = self.size / 2.0
        self.object_points = np.array([[-half, half, 0.0], [half, half, 0.0],
                                       [half, -half, 0.0], [-half, -half, 0.0]],
                                      dtype=np.float32)
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)))

    def __str__(self):
        which = "largest in view" if self.marker_id is None else f"id {self.marker_id}"
        return (f"aruco {self.size*1000:.1f} mm, {self.dictionary_name}, {which}")

    def detect(self, gray):
        import cv2

        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or not len(ids):
            return None
        ids = ids.ravel().tolist()
        if self.marker_id is not None:
            if self.marker_id not in ids:
                return None
            chosen = corners[ids.index(self.marker_id)]
        else:
            # Several tags in frame and no id given: the biggest one is the
            # one being looked at, and the small ones are usually noise.
            chosen = max(corners, key=lambda c: cv2.contourArea(c.reshape(-1, 2)))
        return self.object_points.copy(), chosen.reshape(-1, 2).astype(np.float32)


class Checkerboard:
    """A plain chessboard: `cols` x `rows` *inner* corners, `square` metres.

    Needs the whole board in view, which is a real constraint for a wrist
    camera working at 10 cm. Use charuco when that bites.
    """

    kind = "checkerboard"

    def __init__(self, rows: int, cols: int, square: float):
        self.rows, self.cols, self.square = int(rows), int(cols), float(square)
        grid = np.zeros((self.rows * self.cols, 3), np.float32)
        grid[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        self.object_points = grid * self.square

    def __str__(self):
        return (f"checkerboard {self.cols}x{self.rows} inner corners, "
                f"{self.square*1000:.1f} mm squares")

    def detect(self, gray):
        """Grayscale -> (object points, image points), or None if not found."""
        import cv2

        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, (self.cols, self.rows), flags=flags)
        if not ok or corners is None or len(corners) != len(self.object_points):
            return None
        return self.object_points.copy(), corners.reshape(-1, 2).astype(np.float32)


class CharucoBoard:
    """A chessboard with ArUco markers in the white squares.

    Every corner carries an id, so a partial view still yields usable points.
    That is what makes it workable at wrist-camera range, where the board
    routinely overflows the frame.
    """

    kind = "charuco"

    def __init__(self, rows: int, cols: int, square: float, marker: float,
                 dictionary: str = "DICT_4X4_50"):
        import cv2

        self.rows, self.cols = int(rows), int(cols)
        self.square, self.marker = float(square), float(marker)
        self.dictionary_name = dictionary
        adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        # OpenCV takes (squaresX, squaresY) -- squares, not inner corners.
        self.board = cv2.aruco.CharucoBoard((self.cols, self.rows),
                                            self.square, self.marker, adict)
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def __str__(self):
        return (f"charuco {self.cols}x{self.rows} squares, "
                f"{self.square*1000:.1f} mm / {self.marker*1000:.1f} mm marker, "
                f"{self.dictionary_name}")

    def detect(self, gray):
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) < 6:
            return None
        obj, img = self.board.matchImagePoints(corners, ids)
        if obj is None or len(obj) < 6:
            return None
        return (obj.reshape(-1, 3).astype(np.float32),
                img.reshape(-1, 2).astype(np.float32))


def make_target(kind: str, rows: int = 0, cols: int = 0, square: float = 0.0,
                marker: float = 0.0, dictionary: str = "DICT_4X4_50",
                marker_size: float = 0.0, marker_id: int | None = None):
    """Build a target from CLI-shaped arguments."""
    if kind == "aruco":
        return ArucoMarker(marker_size, dictionary, marker_id)
    if kind == "checkerboard":
        return Checkerboard(rows, cols, square)
    if kind == "charuco":
        return CharucoBoard(rows, cols, square, marker or square * 0.75, dictionary)
    raise ValueError(f"unknown target {kind!r}")


def detect_into(target, sample) -> int:
    """Run the detector on a sample's image and store the points on it.

    Returns how many points were found, 0 for no detection.
    """
    import cv2

    found = target.detect(cv2.cvtColor(sample.rgb, cv2.COLOR_RGB2GRAY))
    if found is None:
        sample.object_points = sample.image_points = None
        return 0
    sample.object_points, sample.image_points = found
    return len(sample.image_points)


def points_per_view_enough_for_intrinsics(target) -> bool:
    """Whether this target can determine the lens as well as the pose.

    Four coplanar points fix a homography and therefore a pose, but leave
    focal length and distortion trading off against each other. Boards give
    tens of points per view and do constrain them.
    """
    return getattr(target, "kind", None) in ("charuco", "checkerboard")


# ---------------------------------------------------------------------------
# Coverage: whether the captures can constrain the answer at all
# ---------------------------------------------------------------------------
def rotation_coverage(rotations) -> dict:
    """How well a set of gripper orientations constrains the hand-eye rotation.

    AX = XB is degenerate when every relative rotation is about the same axis:
    the component of X around that axis is then unobservable, and the solver
    still returns a confident-looking answer. Stacking the relative rotation
    axes and looking at their singular values says whether a second and third
    axis were actually used.
    """
    rotations = list(rotations)
    if len(rotations) < 2:
        return {"angles_deg": 0.0, "axes": [0.0, 0.0, 0.0], "ok": False}
    axes, angles = [], []
    for i in range(len(rotations) - 1):
        rel = (Rotation.from_matrix(rotations[i]).inv()
               * Rotation.from_matrix(rotations[i + 1])).as_rotvec()
        angle = float(np.linalg.norm(rel))
        if angle > np.deg2rad(3.0):          # ignore noise-sized motions
            axes.append(rel / angle)
            angles.append(angle)
    if len(axes) < 2:
        return {"angles_deg": float(np.degrees(sum(angles))), "axes": [0.0]*3, "ok": False}
    sv = np.linalg.svd(np.asarray(axes), compute_uv=False)
    sv = (sv / sv[0]).tolist() + [0.0, 0.0]
    total_deg = float(np.degrees(np.sum(angles)))
    return {
        "angles_deg": total_deg,
        "axes": [round(v, 3) for v in sv[:3]],
        # A second axis carrying at least a fifth of the first is enough to
        # pin the rotation; below that the answer is partly unconstrained.
        # bool(), not the numpy scalar `>` returns: this dict is handed
        # straight to the panel's JSON encoder, which rejects np.bool_.
        "ok": bool(sv[1] > 0.2 and total_deg > 60.0),
    }


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class HandEyeResult:
    """Everything a caller needs to judge and use a calibration."""

    k: np.ndarray                    # 3x3 intrinsics at image_size
    dist: np.ndarray                 # plumb_bob [k1 k2 p1 p2 k3]
    image_size: tuple               # (width, height)
    T_gripper_cam: np.ndarray       # 4x4, the answer
    T_base_target: np.ndarray       # 4x4, where the target ended up
    reprojection_rms: float          # px, from the intrinsics fit
    spread_mm: float                 # target position wobble across samples
    spread_deg: float                # target orientation wobble
    per_sample_mm: list              # each sample's distance from the mean
    n_samples: int
    coverage: dict
    method: str

    @property
    def position(self) -> np.ndarray:
        return self.T_gripper_cam[:3, 3]

    @property
    def quat_xyzw(self) -> np.ndarray:
        return Rotation.from_matrix(self.T_gripper_cam[:3, :3]).as_quat()

    def verdict(self) -> tuple[bool, str]:
        """Is this good enough to use, and why not."""
        problems = []
        if self.reprojection_rms > 1.0:
            problems.append(f"intrinsics fit is loose ({self.reprojection_rms:.2f} px)")
        if self.spread_mm > 5.0:
            problems.append(f"the target moves {self.spread_mm:.1f} mm between "
                            f"samples, but it is bolted down")
        if not self.coverage["ok"]:
            problems.append("the arm was not rotated about enough distinct axes")
        return (not problems), "; ".join(problems)


def solve(samples, image_size, fixed_k=None, fixed_dist=None,
          method: str = "TSAI") -> HandEyeResult:
    """Detected samples -> intrinsics and the gripper->camera transform.

    `samples` need `.object_points`, `.image_points`, `.ee_position` and
    `.ee_quat_xyzw`; anything without a detection is ignored by the caller,
    not here.
    """
    import cv2

    if len(samples) < 3:
        raise ValueError(f"need at least 3 detected samples, got {len(samples)}")

    obj = [s.object_points.reshape(-1, 1, 3).astype(np.float32) for s in samples]
    img = [s.image_points.reshape(-1, 1, 2).astype(np.float32) for s in samples]

    if fixed_k is None:
        rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj, img, tuple(image_size), None, None)
    else:
        # Intrinsics already known: only the board poses are needed, and
        # solvePnP gives them one view at a time.
        k = np.asarray(fixed_k, dtype=np.float64).reshape(3, 3)
        dist = np.asarray(fixed_dist if fixed_dist is not None else np.zeros(5),
                          dtype=np.float64).reshape(-1, 1)
        rvecs, tvecs, errs = [], [], []
        for o, i in zip(obj, img):
            ok, rvec, tvec = cv2.solvePnP(o, i, k, dist)
            if not ok:
                raise ValueError("solvePnP failed on a sample")
            rvecs.append(rvec); tvecs.append(tvec)
            projected, _ = cv2.projectPoints(o, rvec, tvec, k, dist)
            errs.append(np.linalg.norm(projected.reshape(-1, 2) - i.reshape(-1, 2),
                                       axis=1))
        rms = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))

    # T_base_gripper, straight from what the robot reports.
    R_g2b = [Rotation.from_quat(s.ee_quat_xyzw).as_matrix() for s in samples]
    t_g2b = [np.asarray(s.ee_position, dtype=np.float64).reshape(3, 1) for s in samples]
    # T_cam_target, from the board poses above.
    R_t2c = [cv2.Rodrigues(r)[0] for r in rvecs]
    t_t2c = [np.asarray(t, dtype=np.float64).reshape(3, 1) for t in tvecs]

    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c,
        method=getattr(cv2, f"CALIB_HAND_EYE_{method.upper()}"))
    T_gripper_cam = rigid(R_c2g, t_c2g)

    # The honest error measure: a bolted-down target must land in the same
    # place from every viewpoint.
    targets = [rigid(R_g2b[i], t_g2b[i]) @ T_gripper_cam @ rigid(R_t2c[i], t_t2c[i])
               for i in range(len(samples))]
    positions = np.array([T[:3, 3] for T in targets])
    mean_pos = positions.mean(axis=0)
    per_sample_mm = (np.linalg.norm(positions - mean_pos, axis=1) * 1000.0)
    mean_rot = Rotation.from_matrix(np.array([T[:3, :3] for T in targets])).mean()
    spread_deg = float(np.degrees(np.max([
        (mean_rot.inv() * Rotation.from_matrix(T[:3, :3])).magnitude()
        for T in targets])))

    return HandEyeResult(
        k=np.asarray(k, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64).ravel()[:5],
        image_size=tuple(int(v) for v in image_size),
        T_gripper_cam=T_gripper_cam,
        T_base_target=rigid(mean_rot.as_matrix(), mean_pos),
        reprojection_rms=float(rms),
        spread_mm=float(per_sample_mm.max()),
        spread_deg=spread_deg,
        per_sample_mm=[round(float(v), 2) for v in per_sample_mm],
        n_samples=len(samples),
        coverage=rotation_coverage(R_g2b),
        method=method.upper(),
    )


def yaml_block(result: HandEyeResult, name: str = "cam4",
               frame: str = "fr3_hand_tcp") -> str:
    """The result as an entry for config/camera_info.yaml, ready to paste."""
    k = result.k.ravel()
    lines = [
        f"# Hand-eye calibration: {result.n_samples} views, "
        f"{result.reprojection_rms:.2f} px reprojection, "
        f"target stationary to {result.spread_mm:.1f} mm / {result.spread_deg:.2f} deg.",
        f"{name}:",
        f"  height: {result.image_size[1]}",
        f"  width: {result.image_size[0]}",
        f"  frame: {frame}          # rides this link; t/q are relative to it",
        f"  distortion_model: plumb_bob",
        "  d:",
    ]
    lines += [f"    - {v!r}" for v in result.dist.tolist()]
    lines.append("  k:")
    lines += [f"    - {v!r}" for v in k.tolist()]
    lines.append("  t:")
    lines += [f"    - {v!r}" for v in result.position.tolist()]
    lines.append("  q:")
    lines += [f"    - {v!r}" for v in result.quat_xyzw.tolist()]
    return "\n".join(lines) + "\n"
