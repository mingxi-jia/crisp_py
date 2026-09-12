"""Tests for deploy/pi05/features.py. No hardware, no checkpoint, no network.

    python -m deploy.pi05.test_features
"""

import numpy as np

from deploy.pi05.features import (
    DEFAULT_COMPONENTS, GRID, N_TOKENS, FeatureMap, PCABasis,
    cosine_map, l2_normalise, pca_rgb, reduce_for_client,
)

DIM = 2048


def synth_tokens(seed=0, n_clusters=5, dim=DIM):
    """Patch tokens with real structure: clusters of similar patches, as a
    scene of distinct objects produces. Random noise has no similarity
    structure and would make the approximation test meaningless."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1, (n_clusters, dim)).astype(np.float32)
    labels = rng.integers(0, n_clusters, N_TOKENS)
    return (centres[labels] + rng.normal(0, 0.25, (N_TOKENS, dim))).astype(np.float32), labels


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    assert cond, name


def main():
    print("=== FeatureMap ===")
    tokens, labels = synth_tokens()
    fm = FeatureMap(tokens, camera="cam1")
    check("shape", fm.tokens.shape == (N_TOKENS, DIM), f"{fm.tokens.shape}")
    check("dtype float32", fm.tokens.dtype == np.float32)
    check("patch_index round trip", fm.patch_index(3, 7) == 3 * GRID + 7)
    check("as_grid", fm.as_grid(np.arange(N_TOKENS)).shape == (GRID, GRID))
    for bad, why in [((10, DIM), "wrong patch count"), ((N_TOKENS,), "1-D")]:
        try:
            FeatureMap(np.zeros(bad)); ok = False
        except ValueError:
            ok = True
        check(f"rejects {why}", ok)
    try:
        fm.patch_index(GRID, 0); ok = False
    except IndexError:
        ok = True
    check("patch_index bounds", ok)

    print("\n=== PCABasis ===")
    basis = PCABasis.fit(tokens, DEFAULT_COMPONENTS)
    check("n_components", basis.n_components == DEFAULT_COMPONENTS)
    check("components orthonormal",
          np.allclose(basis.components @ basis.components.T,
                      np.eye(basis.n_components), atol=1e-3))
    proj = basis.project(tokens)
    check("projection shape", proj.shape == (N_TOKENS, DEFAULT_COMPONENTS))
    evr = basis.explained_variance_ratio
    check("variance ratios descend", bool(np.all(np.diff(evr) <= 1e-6)))
    print(f"        top-3 explain {evr[:3].sum()*100:.1f}%, "
          f"top-{DEFAULT_COMPONENTS} explain {evr.sum()*100:.1f}%")
    check("serialisation round trip",
          np.allclose(PCABasis.from_dict(basis.to_dict()).project(tokens), proj, atol=1e-5))
    check("fit accepts a stack of frames",
          PCABasis.fit(np.stack([tokens, tokens]), 8).n_components == 8)

    print("\n=== pca_rgb ===")
    rgb = pca_rgb(fm, basis)
    check("shape/dtype", rgb.shape == (GRID, GRID, 3) and rgb.dtype == np.uint8, f"{rgb.shape}")
    check("uses the range", rgb.ptp() > 100, f"ptp={rgb.ptp()}")
    flat = FeatureMap(np.ones((N_TOKENS, DIM), np.float32))
    check("constant input does not divide by zero",
          np.all(np.isfinite(pca_rgb(flat, basis))))
    try:
        pca_rgb(fm, PCABasis.fit(tokens, 2)); ok = False
    except ValueError:
        ok = True
    check("rejects a basis with <3 components", ok)

    print("\n=== cosine_map (reference) ===")
    cm = cosine_map(tokens, 0)
    check("shape", cm.shape == (GRID, GRID))
    check("self-similarity is 1", np.isclose(cm.flat[0], 1.0, atol=1e-5), f"{cm.flat[0]:.6f}")
    check("in [-1, 1]", bool(cm.min() >= -1.0001 and cm.max() <= 1.0001))
    same = labels == labels[0]
    check("same-cluster patches score higher",
          cm.flatten()[same].mean() > cm.flatten()[~same].mean() + 0.2,
          f"{cm.flatten()[same].mean():.3f} vs {cm.flatten()[~same].mean():.3f}")
    check("l2_normalise handles a zero row",
          np.all(np.isfinite(l2_normalise(np.zeros((3, 4), np.float32)))))
    try:
        cosine_map(tokens, N_TOKENS); ok = False
    except IndexError:
        ok = True
    check("index bounds", ok)

    print("\n=== THE approximation that matters: PCA-64 vs full 2048-D ===")
    reduced = reduce_for_client(fm, basis)
    check("payload shape", reduced.shape == (N_TOKENS, DEFAULT_COMPONENTS))
    kb = reduced.nbytes / 1024
    print(f"        payload {kb:.0f} KB/camera vs "
          f"{tokens.nbytes/1024:.0f} KB full ({tokens.nbytes/reduced.nbytes:.0f}x smaller)")

    # The browser sees centred features, so the reference must be centred too.
    centred = tokens - basis.mean
    # Raw top-k overlap is the wrong metric here: a flat region of the scene
    # produces dozens of patches at near-identical similarity, whose ordering
    # is numerical noise. What a similarity map is actually read for is
    # "the patches it highlights really are similar", so measure the quality
    # of what the approximation selects, not whether it ties-breaks identically.
    worst_rank, worst_corr, worst_gap, worst_err = 1.0, 1.0, 0.0, 0.0
    for idx in range(0, N_TOKENS, 7):
        exact = cosine_map(centred, idx).flatten()
        approx = cosine_map(reduced, idx).flatten()
        worst_corr = min(worst_corr, float(np.corrcoef(exact, approx)[0, 1]))
        worst_rank = min(worst_rank, float(np.corrcoef(
            np.argsort(np.argsort(-exact)), np.argsort(np.argsort(-approx)))[0, 1]))
        worst_err = max(worst_err, float(np.abs(exact - approx).max()))
        # How much true similarity is lost by trusting the approximation's top-8?
        best8 = exact[np.argsort(-exact)[:8]].mean()
        picked8 = exact[np.argsort(-approx)[:8]].mean()
        worst_gap = max(worst_gap, float(best8 - picked8))
    n_q = len(range(0, N_TOKENS, 7))
    print(f"        over {n_q} query patches, worst case:")
    print(f"          Pearson r            {worst_corr:.4f}")
    print(f"          Spearman rank        {worst_rank:.4f}")
    print(f"          max abs value error  {worst_err:.4f}")
    print(f"          true-similarity lost by trusting approx top-8: {worst_gap:.4f}")
    # ~0.05 absolute error on a [-1, 1] scale is expected and harmless for a
    # heat map: the dropped components carry ~5% of the variance, so values
    # shift slightly while the pattern does not. Ranking and selection, which
    # are what the map is read for, are checked separately below.
    check("values track the exact map", worst_corr > 0.99 and worst_err < 0.1)
    check("spatial ordering preserved", worst_rank > 0.95)
    check("highlighted patches really are the similar ones", worst_gap < 0.01)

    print("\n=== degenerate inputs ===")
    tiny = FeatureMap(np.zeros((N_TOKENS, 4), np.float32))
    b = PCABasis.fit(tiny.tokens, 8)
    check("zero features: basis clamps k to dim", b.n_components <= 4, f"k={b.n_components}")
    check("zero features: rgb finite", np.all(np.isfinite(pca_rgb(tiny, PCABasis.fit(tiny.tokens, 3)))))
    check("zero features: cosine finite", np.all(np.isfinite(cosine_map(tiny.tokens, 0))))

    print("\nALL FEATURE TESTS PASSED")


if __name__ == "__main__":
    main()
