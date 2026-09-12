"""Wire format shared by robot_server.py and robot_client.py.

Numpy arrays travel as base64-encoded ``.npy`` blobs, which keeps dtype and
shape exact without trusting pickle. This matches the convention used by the
other crisp_py Flask servers (e.g. pink_ik_server.py).
"""

import base64
import io

import numpy as np


def encode_array(arr) -> dict | None:
    """Encode a numpy array as a JSON-serialisable dict (None passes through)."""
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return {
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def decode_array(payload: dict | None) -> np.ndarray | None:
    """Inverse of :func:`encode_array` (None passes through)."""
    if payload is None:
        return None
    raw = base64.b64decode(payload["data"])
    return np.load(io.BytesIO(raw), allow_pickle=False)


def as_vector(value, dtype=np.float64) -> np.ndarray | None:
    """Accept either an encoded array or a plain JSON list, return an ndarray."""
    if value is None:
        return None
    if isinstance(value, dict):
        return decode_array(value)
    return np.asarray(value, dtype=dtype)
