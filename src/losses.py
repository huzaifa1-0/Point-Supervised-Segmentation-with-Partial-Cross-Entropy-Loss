"""
losses.py
Partial (Focal) Cross-Entropy loss for point-supervised segmentation.

    pfCE = sum( FocalLoss(pred, GT) * MASK_labeled ) / sum( MASK_labeled )

Only pixels flagged in `mask_labeled` (i.e. the simulated point labels)
contribute to the loss; everything else is ignored -- this is what makes the
loss "partial" rather than a standard dense-mask cross-entropy / focal loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialFocalCE(nn.Module):
    """
    Parameters
    ----------
    num_classes : int
        Number of trainable classes (exclude any ignore/unknown class).
    gamma : float
        Focal-loss focusing parameter. gamma=0 reduces this to a plain
        partial cross-entropy loss.
    alpha : torch.Tensor or None, shape (num_classes,)
        Optional per-class weights (e.g. inverse class frequency) to counter
        class imbalance. If None, all classes weighted equally.
    eps : float
        Numerical-stability epsilon.
    """

    def __init__(self, num_classes: int, gamma: float = 2.0, alpha: torch.Tensor | None = None, eps: float = 1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.eps = eps
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask_labeled: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, C, H, W) float  -- raw network output (pre-softmax)
        target : (B, H, W) long      -- class ids; value at unlabeled pixels
                                        is irrelevant (masked out below), but
                                        MUST be a valid index in [0, C-1] to
                                        avoid one_hot errors -- callers should
                                        fill unlabeled pixels with any valid
                                        placeholder class (e.g. 0), never a
                                        value >= C.
        mask_labeled : (B, H, W) float in {0, 1} -- 1 where a point label exists

        Returns
        -------
        scalar loss tensor
        """
        assert logits.dim() == 4, f"logits must be (B, C, H, W), got {logits.shape}"
        b, c, h, w = logits.shape
        assert c == self.num_classes, f"logits has {c} classes, expected {self.num_classes}"
        assert target.shape == (b, h, w), f"target shape {target.shape} != {(b, h, w)}"
        assert mask_labeled.shape == (b, h, w), f"mask_labeled shape {mask_labeled.shape} != {(b, h, w)}"

        log_probs = F.log_softmax(logits, dim=1)         
        probs = log_probs.exp()

        target_clamped = target.clamp(min=0, max=self.num_classes - 1)
        target_onehot = F.one_hot(target_clamped, self.num_classes)      
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()      

        pt = (probs * target_onehot).sum(dim=1)           
        logpt = (log_probs * target_onehot).sum(dim=1)     

        focal_term = (1.0 - pt).clamp(min=self.eps) ** self.gamma
        per_pixel_loss = -focal_term * logpt              

        if self.alpha is not None:
            alpha_map = (target_onehot * self.alpha.view(1, -1, 1, 1)).sum(dim=1)
            per_pixel_loss = per_pixel_loss * alpha_map

        masked_loss = per_pixel_loss * mask_labeled
        denom = mask_labeled.sum().clamp(min=1.0)
        return masked_loss.sum() / denom


class DenseFocalCE(nn.Module):
    """
    Standard dense (fully-supervised) Focal Cross-Entropy, used only as the
    "upper bound" baseline for comparison against the point-supervised model.
    Supports an ignore_index (e.g. DeepGlobe's "Unknown" class).
    """

    def __init__(self, num_classes: int, gamma: float = 2.0, alpha: torch.Tensor | None = None,
                 ignore_index: int | None = None, eps: float = 1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = logits.shape
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            target_clamped = target.clamp(min=0, max=self.num_classes - 1)
        else:
            valid = torch.ones_like(target, dtype=torch.bool)
            target_clamped = target

        target_onehot = F.one_hot(target_clamped, self.num_classes).permute(0, 3, 1, 2).float()
        pt = (probs * target_onehot).sum(dim=1)
        logpt = (log_probs * target_onehot).sum(dim=1)
        focal_term = (1.0 - pt).clamp(min=self.eps) ** self.gamma
        per_pixel_loss = -focal_term * logpt

        if self.alpha is not None:
            alpha_map = (target_onehot * self.alpha.view(1, -1, 1, 1)).sum(dim=1)
            per_pixel_loss = per_pixel_loss * alpha_map

        valid = valid.float()
        masked_loss = per_pixel_loss * valid
        denom = valid.sum().clamp(min=1.0)
        return masked_loss.sum() / denom
