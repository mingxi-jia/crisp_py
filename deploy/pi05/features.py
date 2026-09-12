"""Vision-feature analysis for the π0.5 panel. Pure numpy.

No ROS, no GPU, no network: everything here runs on a saved array, which is
what makes the PCA and similarity maths testable without a robot or a
checkpoint.

The tokens come from SigLIP So400m/14 as configured in openpi
(third_party/openpi/src/openpi/models/pi0.py:82-91). At 224x224 with patch 14
that is a 16x16 grid of 256 tokens, projected to PaliGemma's width (2048).
They are the *projected* tokens the language model consumes, not raw 1152-dim
SigLIP output -- the right thing to visualise, but a learned space, so do not
expect it to look like a DINO feature map.
"""

import dataclasses

import numpy as np

GRID = 16                      # 224 / 14
N_TOKENS = GRID * GRID         # 256
DEFAULT_COMPONENTS = 64        # what ships to the browser for cosine similarity


@dataclasses.dataclass
class FeatureMap:
    """One camera's patch tokens at one timestep."""

    tokens: np.ndarray                     # (N_TOKENS, dim) float32
    camera: str = ""
    grid: int = GRID

    def __post_init__(self):
        self.tokens = np.asarray(self.tokens, dtype=np.float32)
        if self.tokens.ndim != 2:
            raise ValueError(f"tokens must be 2-D (n_patches, dim), got {self.tokens.shape}")
        if self.tokens.shape[0] != self.grid * self.grid:
            raise ValueError(
                f"{self.tokens.shape[0]} patches does not fill a "
                f"{self.grid}x{self.grid} grid ({self.grid ** 2} expected)")

    @property
    def dim(self) -> int:
        return int(self.tokens.shape[1])

    def as_grid(self, values: np.ndarray) -> np.ndarray:
        """Reshape a per-patch vector into the (grid, grid) image layout."""
        values = np.asarray(values)
        if values.shape[0] != self.tokens.shape[0]:
            raise ValueError(f"expected {self.tokens.shape[0]} values, got {values.shape[0]}")
        return values.reshape(self.grid, self.grid, *values.shape[1:])

    def patch_index(self, row: int, col: int) -> int:
        if not (0 <= row < self.grid and 0 <= col < self.grid):
            raise IndexError(f"({row}, {col}) outside a {self.grid}x{self.grid} grid")
        return row * self.grid + col


class PCABasis:
    """A PCA basis fitted once and reused across frames.

    Refitting every frame makes the visualisation flicker: component order and
    sign are arbitrary, so a patch can swap colour between timesteps for no
    physical reason. Fit once over a few frames, then hold it.
    """

    def __init__(self, mean: np.ndarray, components: np.ndarray,
                 explained_variance_ratio: np.ndarray | None = None):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.components = np.asarray(components, dtype=np.float32)   # (k, dim)
        self.explained_variance_ratio = (
            None if explained_variance_ratio is None
            else np.asarray(explained_variance_ratio, dtype=np.float32))

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @classmethod
    def fit(cls, tokens: np.ndarray, n_components: int = DEFAULT_COMPONENTS) -> "PCABasis":
        """Fit on (n_patches, dim) or a stack of several frames."""
        x = np.asarray(tokens, dtype=np.float32)
        if x.ndim == 3:                       # (frames, patches, dim) -> pool
            x = x.reshape(-1, x.shape[-1])
        mean = x.mean(axis=0)
        centred = x - mean
        k = int(min(n_components, *centred.shape))
        # full_matrices=False: we only ever need the top components.
        _u, s, vt = np.linalg.svd(centred, full_matrices=False)
        var = s ** 2
        total = var.sum()
        ratio = (var[:k] / total) if total > 0 else np.zeros(k, np.float32)
        # Sign is arbitrary in SVD; pin it so colours stay stable across refits.
        comps = vt[:k]
        flip = np.where(np.abs(comps).argmax(axis=1)[:, None] >= 0,
                        np.sign(comps[np.arange(k), np.abs(comps).argmax(axis=1)])[:, None], 1.0)
        comps = comps * np.where(flip == 0, 1.0, flip)
        return cls(mean, comps, ratio)

    def project(self, tokens: np.ndarray) -> np.ndarray:
        """(n_patches, dim) -> (n_patches, k)."""
        x = np.asarray(tokens, dtype=np.float32) - self.mean
        return x @ self.components.T

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(),
                "components": self.components.tolist(),
                "explained_variance_ratio":
                    None if self.explained_variance_ratio is None
                    else self.explained_variance_ratio.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "PCABasis":
        return cls(np.asarray(d["mean"]), np.asarray(d["components"]),
                   None if d.get("explained_variance_ratio") is None
                   else np.asarray(d["explained_variance_ratio"]))


def pca_rgb(feature_map: FeatureMap, basis: PCABasis,
            percentile: float = 2.0) -> np.ndarray:
    """Top-3 PCA components as a (grid, grid, 3) uint8 image.

    Each channel is normalised over its own robust range: outlier patches
    otherwise crush everything else into a flat mid-grey.
    """
    if basis.n_components < 3:
        raise ValueError(f"need at least 3 components, basis has {basis.n_components}")
    proj = basis.project(feature_map.tokens)[:, :3]
    lo = np.percentile(proj, percentile, axis=0)
    hi = np.percentile(proj, 100.0 - percentile, axis=0)
    span = np.where(hi - lo < 1e-8, 1.0, hi - lo)
    norm = np.clip((proj - lo) / span, 0.0, 1.0)
    return feature_map.as_grid((norm * 255).astype(np.uint8))


def l2_normalise(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.where(n < 1e-12, 1.0, n)


def cosine_map(tokens: np.ndarray, index: int, grid: int = GRID) -> np.ndarray:
    """Cosine similarity of patch `index` against every patch, as (grid, grid).

    Reference implementation. The panel computes this in the browser on
    PCA-reduced features; this is what that approximation is checked against.
    """
    x = np.asarray(tokens, dtype=np.float32)
    if not 0 <= index < x.shape[0]:
        raise IndexError(f"patch {index} outside 0..{x.shape[0] - 1}")
    unit = l2_normalise(x)
    return (unit @ unit[index]).reshape(grid, grid)


def reduce_for_client(feature_map: FeatureMap, basis: PCABasis,
                      n_components: int = DEFAULT_COMPONENTS) -> np.ndarray:
    """(n_patches, k) float32 to ship to the browser.

    Cosine similarity is computed there on this reduced form. PCA is a linear
    projection, so similarity is not preserved exactly -- but because the
    dropped components carry little variance, the *ranking* of patches is
    nearly unchanged, which is what a similarity map is read for.
    Centring is part of the projection, so cosine here is on centred features:
    the reference cosine_map is also fed centred tokens when comparing.

    Measured against the exact 2048-dim map (see test_features.py), with 64
    components retaining ~95% of variance: Pearson r = 1.000, Spearman rank
    = 0.99, worst-case absolute value error ~0.05 on a [-1, 1] scale, and the
    true similarity given up by trusting the approximation's top-8 is 0.001.
    In short: values shift slightly, the pattern does not.
    """
    k = min(n_components, basis.n_components)
    return basis.project(feature_map.tokens)[:, :k].astype(np.float32)
