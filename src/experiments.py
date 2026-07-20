"""
experiments.py
Runners for the two ablation experiments described in the implementation plan:

  Experiment A -- Point density: mIoU vs. number of points per class
  Experiment B -- Sampling strategy: class-balanced vs. pixel-proportional
                  point sampling, at fixed point budget

Both build a fresh model per run (no weight sharing across conditions) so
results are directly comparable and independent.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import PointSupervisedTileDataset, FullMaskTileDataset
from model import build_model
from losses import PartialFocalCE
from train import TrainConfig, fit
from data_utils import IGNORE_INDEX, NUM_CLASSES


def run_point_density_experiment(
    train_tiles: List[Tuple],
    val_tiles: List[Tuple],
    point_counts: List[int] = (5, 20, 50, 100),
    strategy: str = "random",
    train_config: TrainConfig | None = None,
    encoder_name: str = "resnet34",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Trains one model per entry in `point_counts` (points-per-class budget),
    evaluates on the same held-out val_tiles (full dense masks), and returns
    a results DataFrame: columns = [points_per_class, val_mIoU, val_pixel_acc].
    """
    train_config = train_config or TrainConfig()
    results = []

    val_ds = FullMaskTileDataset(val_tiles)
    val_loader = DataLoader(val_ds, batch_size=train_config.batch_size,
                             shuffle=False, num_workers=train_config.num_workers)

    for n_points in point_counts:
        print(f"\n=== Experiment A: {n_points} points/class, strategy={strategy} ===")
        torch.manual_seed(seed)

        train_ds = PointSupervisedTileDataset(
            train_tiles, n_points_per_class=n_points, strategy=strategy, base_seed=seed
        )
        train_loader = DataLoader(train_ds, batch_size=train_config.batch_size,
                                   shuffle=True, num_workers=train_config.num_workers)

        model = build_model(num_classes=NUM_CLASSES, encoder_name=encoder_name)
        criterion = PartialFocalCE(num_classes=NUM_CLASSES, gamma=2.0)

        history = fit(model, train_loader, val_loader, criterion, train_config,
                       train_dataset_for_epoch_hook=train_ds)

        results.append({
            "points_per_class": n_points,
            "final_train_loss": history.train_loss[-1],
            "val_mIoU": history.val_miou[-1],
            "val_pixel_acc": history.val_pixel_acc[-1],
            "best_val_mIoU": max(history.val_miou),
        })

    return pd.DataFrame(results)


def run_sampling_strategy_experiment(
    train_tiles: List[Tuple],
    val_tiles: List[Tuple],
    n_points: int = 20,
    strategies: List[str] = ("class_balanced", "pixel_proportional"),
    train_config: TrainConfig | None = None,
    encoder_name: str = "resnet34",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Trains one model per sampling strategy at a fixed point budget, evaluates
    per-class IoU on val_tiles to check whether class-balanced sampling
    improves minority-class performance more than overall accuracy.

    Returns a long-format DataFrame: columns = [strategy, class_name, iou],
    plus a summary row per strategy for mIoU / pixel_acc.
    """
    from data_utils import CLASS_NAMES
    from metrics import ConfusionMatrixMeter

    train_config = train_config or TrainConfig()
    rows = []

    val_ds = FullMaskTileDataset(val_tiles)
    val_loader = DataLoader(val_ds, batch_size=train_config.batch_size,
                             shuffle=False, num_workers=train_config.num_workers)

    for strategy in strategies:
        print(f"\n=== Experiment B: strategy={strategy}, {n_points} points/class ===")
        torch.manual_seed(seed)

        # NOTE: pixel_proportional_point_sample takes a *total* point budget;
        # scale it up so the comparison is roughly apples-to-apples in total
        # annotation effort per tile (n_points * num_present_classes ~= n_points * NUM_CLASSES).
        effective_n = n_points if strategy != "pixel_proportional" else n_points * NUM_CLASSES

        train_ds = PointSupervisedTileDataset(
            train_tiles, n_points_per_class=effective_n, strategy=strategy, base_seed=seed
        )
        train_loader = DataLoader(train_ds, batch_size=train_config.batch_size,
                                   shuffle=True, num_workers=train_config.num_workers)

        model = build_model(num_classes=NUM_CLASSES, encoder_name=encoder_name)
        criterion = PartialFocalCE(num_classes=NUM_CLASSES, gamma=2.0)

        fit(model, train_loader, val_loader, criterion, train_config,
            train_dataset_for_epoch_hook=train_ds)

        meter = ConfusionMatrixMeter(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(train_config.device)
                labels = labels.to(train_config.device)
                logits = model(images)
                meter.update(logits, labels)

        per_class = meter.per_class_iou()
        for class_id, iou in enumerate(per_class):
            rows.append({
                "strategy": strategy,
                "class_name": CLASS_NAMES[class_id],
                "iou": iou,
            })
        rows.append({
            "strategy": strategy,
            "class_name": "MEAN (mIoU)",
            "iou": meter.mean_iou(),
        })

    return pd.DataFrame(rows)
