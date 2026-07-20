"""
train.py
Training / validation loops for the point-supervised segmentation model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader

from metrics import ConfusionMatrixMeter
from losses import PartialFocalCE
from data_utils import IGNORE_INDEX, NUM_CLASSES


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 20


@dataclass
class TrainHistory:
    train_loss: list = field(default_factory=list)
    val_miou: list = field(default_factory=list)
    val_pixel_acc: list = field(default_factory=list)


def train_one_epoch(model, loader: DataLoader, criterion: PartialFocalCE,
                     optimizer, device: str, log_every: int = 20) -> float:
    model.train()
    running_loss, n_batches = 0.0, 0

    for i, (images, labels, mask_labeled) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        mask_labeled = mask_labeled.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels, mask_labeled)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

        if log_every and (i + 1) % log_every == 0:
            print(f"    batch {i + 1}/{len(loader)}  loss={loss.item():.4f}")

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str, num_classes: int,
             ignore_index: Optional[int] = None) -> ConfusionMatrixMeter:
    model.eval()
    meter = ConfusionMatrixMeter(num_classes=num_classes, ignore_index=ignore_index)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        meter.update(logits, labels)

    return meter


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: PartialFocalCE,
    config: TrainConfig,
    train_dataset_for_epoch_hook=None,
    verbose: bool = True,
) -> TrainHistory:
    """
    Full training loop with per-epoch validation.

    `train_dataset_for_epoch_hook`: if the underlying PointSupervisedTileDataset
    supports `.set_epoch(epoch)` (to vary the simulated points each epoch),
    pass it here so it gets called automatically.
    """
    device = config.device
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history = TrainHistory()

    for epoch in range(config.epochs):
        t0 = time.time()

        if train_dataset_for_epoch_hook is not None:
            train_dataset_for_epoch_hook.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, log_every=config.log_every
        )
        scheduler.step()

        meter = evaluate(model, val_loader, device, num_classes=criterion.num_classes,
                          ignore_index=IGNORE_INDEX)
        val_miou = meter.mean_iou()
        val_acc = meter.pixel_accuracy()

        history.train_loss.append(train_loss)
        history.val_miou.append(val_miou)
        history.val_pixel_acc.append(val_acc)

        if verbose:
            dt = time.time() - t0
            print(f"Epoch {epoch + 1:3d}/{config.epochs}  "
                  f"train_loss={train_loss:.4f}  val_mIoU={val_miou:.4f}  "
                  f"val_pixAcc={val_acc:.4f}  ({dt:.1f}s)")

    return history
