"""
make_synthetic_data.py
Generates small synthetic "DeepGlobe-like" RGB image + RGB color-coded mask
pairs, purely to smoke-test the pipeline end-to-end in environments without
access to the real Kaggle dataset. NOT a substitute for real data -- swap in
real DeepGlobe images/masks by pointing the loader at the actual dataset
directory (see notebook Section 1 for instructions).
"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image

from data_utils import CLASS_COLORS


def make_synthetic_pair(size: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)

    # Random blocky "land cover" pattern: assign a random class per coarse
    # cell, then upsample -- crude but gives multi-class contiguous regions,
    # which is what real land-cover masks look like.
    cell = 40
    n_cells = size // cell + 1
    class_grid = rng.integers(0, 6, size=(n_cells, n_cells))  # classes 0-5 (exclude Unknown=6 for simplicity, add patches below)
    class_mask = np.kron(class_grid, np.ones((cell, cell), dtype=int))[:size, :size]

    # sprinkle a bit of "Unknown" (class 6) near the border to mimic real data
    class_mask[:20, :] = 6
    class_mask[-20:, :] = 6

    rgb_mask = CLASS_COLORS[class_mask]  # (H, W, 3)

    # fabricate a plausible-looking RGB image: base color per class + noise
    base_colors = {
        0: (150, 150, 150),  # urban -> grayish
        1: (200, 190, 80),   # agriculture -> tan/yellow
        2: (170, 190, 120),  # rangeland -> olive
        3: (40, 110, 40),    # forest -> green
        4: (40, 80, 180),    # water -> blue
        5: (210, 200, 180),  # barren -> beige
        6: (0, 0, 0),
    }
    image = np.zeros((size, size, 3), dtype=np.uint8)
    for cls, color in base_colors.items():
        image[class_mask == cls] = color
    noise = rng.normal(0, 12, size=image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, rgb_mask


def generate_dataset(out_dir: str, n_images: int = 6, size: int = 600, seed: int = 0):
    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(n_images):
        image, rgb_mask = make_synthetic_pair(size=size, seed=seed + i)
        Image.fromarray(image).save(os.path.join(img_dir, f"{i:03d}_sat.jpg"), quality=95)
        Image.fromarray(rgb_mask).save(os.path.join(mask_dir, f"{i:03d}_mask.png"))

    print(f"Generated {n_images} synthetic image/mask pairs in {out_dir}")


if __name__ == "__main__":
    generate_dataset("/home/claude/project/sample_data", n_images=6, size=600)
