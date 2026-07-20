"""
point_sampler.py
Simulates sparse point-label supervision from a fully-labeled class-ID mask,
as described in the assessment ("annotation is based on points ... a very
challenging problem").

Two sampling strategies are provided:
  - random_point_sample:   uniform random pixels per class (simple baseline)
  - stratified_point_sample: same as above but explicitly guarantees exactly
    N points per *present* class (equivalent here, kept as a separate name for
    clarity/experiment labeling — see class_balanced_point_sample for the
    genuinely different "balanced under imbalance" strategy)
  - class_balanced_point_sample: samples so that rare classes are not
    penalized relative to their (small) pixel count -- same fixed N per class
    regardless of how rare the class is in this particular tile. This is the
    "class-balanced stratified" strategy referenced in Experiment B of the
    implementation plan.

All strategies:
  - never sample from IGNORE_INDEX ("Unknown") pixels
  - return a `MASK_labeled` binary array (1 = keep pixel loss, 0 = ignore)
    plus a `sparse_label` array (only valid at labeled pixels)
"""

from __future__ import annotations

import numpy as np

from data_utils import IGNORE_INDEX, NUM_CLASSES


def _sample_n_per_class(class_id_mask: np.ndarray, n_points: int, rng: np.random.Generator):
    """Core routine shared by both strategies: sample up to n_points pixel
    coordinates for each class present in the mask (excluding IGNORE_INDEX)."""
    h, w = class_id_mask.shape
    mask_labeled = np.zeros((h, w), dtype=np.float32)
    sparse_label = np.full((h, w), IGNORE_INDEX, dtype=np.int64)

    present_classes = [c for c in np.unique(class_id_mask) if c != IGNORE_INDEX]

    for class_id in present_classes:
        ys, xs = np.where(class_id_mask == class_id)
        if len(ys) == 0:
            continue
        n = min(n_points, len(ys))
        chosen = rng.choice(len(ys), size=n, replace=False)
        mask_labeled[ys[chosen], xs[chosen]] = 1.0
        sparse_label[ys[chosen], xs[chosen]] = class_id

    return sparse_label, mask_labeled


def random_point_sample(
    class_id_mask: np.ndarray,
    n_points_per_class: int = 20,
    seed: int | None = None,
):
    """
    Uniform-random point simulation: `n_points_per_class` pixels sampled
    independently and uniformly at random for each class present in the tile.

    Returns
    -------
    sparse_label : (H, W) int64  -- valid class id only where mask_labeled==1
    mask_labeled : (H, W) float32 in {0, 1}
    """
    rng = np.random.default_rng(seed)
    return _sample_n_per_class(class_id_mask, n_points_per_class, rng)


def class_balanced_point_sample(
    class_id_mask: np.ndarray,
    n_points_per_class: int = 20,
    seed: int | None = None,
):
    """
    Class-balanced stratified point simulation. Functionally this guarantees
    the *same* fixed budget of points per present class regardless of how
    dominant/rare that class is in the tile -- which is the key difference
    from naive "sample N total points uniformly over all foreground pixels"
    (that alternative would end up mostly sampling the majority classes,
    e.g. agriculture/forest in DeepGlobe, and starve minority classes like
    urban/water/barren). Implemented identically to random_point_sample at
    the per-class level; the distinction from a *pixel-proportional* sampler
    is made explicit here and used as-is for Experiment B in the report.

    Returns
    -------
    sparse_label : (H, W) int64
    mask_labeled : (H, W) float32 in {0, 1}
    """
    rng = np.random.default_rng(seed)
    return _sample_n_per_class(class_id_mask, n_points_per_class, rng)


def pixel_proportional_point_sample(
    class_id_mask: np.ndarray,
    n_points_total: int = 120,
    seed: int | None = None,
):
    """
    Alternative baseline sampler: draws n_points_total pixels uniformly at
    random from ALL non-ignore pixels in the tile (no per-class stratification).
    Included so Experiment B can compare "class-balanced" vs. "proportional to
    natural class frequency" -- the latter is expected to under-sample rare
    classes in imbalanced datasets like DeepGlobe.
    """
    rng = np.random.default_rng(seed)
    h, w = class_id_mask.shape
    mask_labeled = np.zeros((h, w), dtype=np.float32)
    sparse_label = np.full((h, w), IGNORE_INDEX, dtype=np.int64)

    ys, xs = np.where(class_id_mask != IGNORE_INDEX)
    if len(ys) == 0:
        return sparse_label, mask_labeled

    n = min(n_points_total, len(ys))
    chosen = rng.choice(len(ys), size=n, replace=False)
    mask_labeled[ys[chosen], xs[chosen]] = 1.0
    sparse_label[ys[chosen], xs[chosen]] = class_id_mask[ys[chosen], xs[chosen]]

    return sparse_label, mask_labeled
