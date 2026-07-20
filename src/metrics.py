"""
metrics.py
Segmentation evaluation metrics: per-class IoU, mean IoU, pixel accuracy.
All metrics exclude IGNORE_INDEX pixels (DeepGlobe's "Unknown" class).
"""

from __future__ import annotations

import numpy as np
import torch


class ConfusionMatrixMeter:
    """Accumulates a confusion matrix across batches, then derives metrics."""

    def __init__(self, num_classes: int, ignore_index: int | None = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion.fill(0)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        """
        logits : (B, C, H, W)
        target : (B, H, W) long
        """
        preds = logits.argmax(dim=1)  # (B, H, W)
        preds_np = preds.cpu().numpy().reshape(-1)
        target_np = target.cpu().numpy().reshape(-1)

        if self.ignore_index is not None:
            valid = target_np != self.ignore_index
            preds_np = preds_np[valid]
            target_np = target_np[valid]

        # guard against any out-of-range prediction (shouldn't happen, but safe)
        valid_range = (preds_np >= 0) & (preds_np < self.num_classes) & \
                      (target_np >= 0) & (target_np < self.num_classes)
        preds_np = preds_np[valid_range]
        target_np = target_np[valid_range]

        idx = target_np * self.num_classes + preds_np
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += binc.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self) -> np.ndarray:
        cm = self.confusion
        intersection = np.diag(cm)
        union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
        iou = np.full(self.num_classes, np.nan)
        nonzero = union > 0
        iou[nonzero] = intersection[nonzero] / union[nonzero]
        return iou

    def mean_iou(self) -> float:
        iou = self.per_class_iou()
        return float(np.nanmean(iou))

    def pixel_accuracy(self) -> float:
        cm = self.confusion
        correct = np.diag(cm).sum()
        total = cm.sum()
        return float(correct / total) if total > 0 else float("nan")
