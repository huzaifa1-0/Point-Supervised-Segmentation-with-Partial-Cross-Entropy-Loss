from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils import IGNORE_INDEX, NUM_CLASSES
from point_sampler import random_point_sample, class_balanced_point_sample, pixel_proportional_point_sample

SAMPLERS = {
    "random": random_point_sample,
    "class_balanced": class_balanced_point_sample,
    "pixel_proportional": pixel_proportional_point_sample,
}


def _to_tensor_image(img: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> (3, H, W) float32 in [0, 1]."""
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).contiguous()


class PointSupervisedTileDataset(Dataset):
    """
    Parameters
    ----------
    tiles : list of (image_tile, class_id_mask_tile) numpy arrays
    n_points_per_class : point budget per present class (or total, for
        the "pixel_proportional" strategy)
    strategy : one of SAMPLERS keys
    transform : optional albumentations-style transform applied jointly to
        (image, sparse_label, mask_labeled); must accept/return keys
        'image', 'mask', 'mask_labeled'
    base_seed : if set, sampling becomes deterministic per (index, epoch)
        rather than fully random -- useful for reproducible experiments
    """

    def __init__(
        self,
        tiles: List[Tuple[np.ndarray, np.ndarray]],
        n_points_per_class: int = 20,
        strategy: str = "random",
        transform: Callable | None = None,
        base_seed: int | None = None,
    ):
        assert strategy in SAMPLERS, f"unknown strategy {strategy!r}, choose from {list(SAMPLERS)}"
        self.tiles = tiles
        self.n_points_per_class = n_points_per_class
        self.strategy = strategy
        self.transform = transform
        self.base_seed = base_seed
        self._epoch = 0

    def set_epoch(self, epoch: int):
        
        self._epoch = epoch

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        image, class_id_mask = self.tiles[idx]

        seed = None
        if self.base_seed is not None:
            seed = self.base_seed + idx * 9973 + self._epoch  

        sampler_fn = SAMPLERS[self.strategy]
        if self.strategy == "pixel_proportional":
            sparse_label, mask_labeled = sampler_fn(
                class_id_mask, n_points_total=self.n_points_per_class, seed=seed
            )
        else:
            sparse_label, mask_labeled = sampler_fn(
                class_id_mask, n_points_per_class=self.n_points_per_class, seed=seed
            )

        if self.transform is not None:
            out = self.transform(image=image, mask=sparse_label, mask_labeled=mask_labeled)
            image, sparse_label, mask_labeled = out["image"], out["mask"], out["mask_labeled"]

        image_t = _to_tensor_image(image)
        label_t = torch.from_numpy(np.ascontiguousarray(sparse_label)).long()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_labeled)).float()

        return image_t, label_t, mask_t


class FullMaskTileDataset(Dataset):
    """Validation/test dataset -- always returns the full dense class-ID mask."""

    def __init__(self, tiles: List[Tuple[np.ndarray, np.ndarray]], transform: Callable | None = None):
        self.tiles = tiles
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        image, class_id_mask = self.tiles[idx]

        if self.transform is not None:
            out = self.transform(image=image, mask=class_id_mask)
            image, class_id_mask = out["image"], out["mask"]

        image_t = _to_tensor_image(image)
        label_t = torch.from_numpy(np.ascontiguousarray(class_id_mask)).long()
        return image_t, label_t
