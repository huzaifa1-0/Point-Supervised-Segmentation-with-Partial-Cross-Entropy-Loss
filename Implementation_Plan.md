# Implementation Plan — Partial Cross-Entropy Loss for Point-Supervised Remote Sensing Segmentation

## 1. Problem Restatement

Standard semantic segmentation requires a **dense pixel mask** for every training image.
In this assessment, supervision is instead given as **sparse point annotations** (a few
labeled pixels per class per image). The goal is to train a segmentation network using
only these points, via a **Partial Cross-Entropy (pCE) loss** that ignores unlabeled pixels.

Formula (from the assessment):

```
pfCE = Σ( FocalLoss(pred, GT) × MASK_labeled ) / Σ( MASK_labeled )
```

Where `MASK_labeled` is a binary map: 1 at annotated (point) pixels, 0 elsewhere.

## 2. Deliverables

- [ ] `partial_ce_loss.py` — loss implementation
- [ ] `point_sampler.py` — utility to simulate point labels from full masks
- [ ] `train.ipynb` / `train.py` — full training + evaluation pipeline
- [ ] `experiments.ipynb` — ablation study (1–2 factors)
- [ ] `technical_report.md` — method + experiments + results write-up

## 3. Environment / Tools

- Python 3.10+
- PyTorch
- `segmentation-models-pytorch` (U-Net / DeepLabV3+ with pretrained ImageNet encoders)
- `albumentations` (augmentation)
- `rasterio` / `Pillow` (image I/O)
- `scikit-learn`, `numpy`, `matplotlib`

```bash
pip install torch torchvision segmentation-models-pytorch albumentations rasterio scikit-learn matplotlib --break-system-packages
```

## 4. Dataset Selection

**Recommended: LandCover.ai** (aerial imagery, 4 classes: building, woodland, water,
background; ~40 large orthophotos, pre-split into small tiles). Reasons:
- Small enough to train quickly on CPU/single GPU
- Full pixel masks available → lets us *simulate* point labels and also measure
  "upper bound" performance with full supervision for comparison
- Simple download script, no registration wall

**Alternatives** (if LandCover.ai is unavailable): DeepGlobe Land Cover Classification,
ISPRS Potsdam/Vaihingen (2D Semantic Labeling).

## 5. Point-Label Simulation

For each training image + full mask:
1. For each class present in the mask, randomly sample `N` pixel coordinates
   belonging to that class (stratified — guarantees every class gets some points).
2. Build `MASK_labeled` (same H×W as image): 1 at sampled coordinates, 0 elsewhere.
3. Store `(image, sparse_label_map, MASK_labeled)` as the training triplet.
   The sparse label map has real class values only at labeled pixels; unlabeled
   pixels can be set to any placeholder (ignored via the mask, not via `ignore_index`).

This simulation step directly reproduces the "very limited info" scenario shown in
the assessment (points vs. mask).

## 6. Partial Cross-Entropy (Focal) Loss — Design

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialFocalCE(nn.Module):
    """
    Partial Cross-Entropy loss using Focal Loss as the base per-pixel loss.
    Only pixels where mask_labeled == 1 contribute to the loss.
    """
    def __init__(self, num_classes, gamma=2.0, alpha=None, eps=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha  # optional per-class weights, tensor of shape [C]
        self.eps = eps

    def forward(self, logits, target, mask_labeled):
        """
        logits:       [B, C, H, W]  raw network output
        target:       [B, H, W]     integer class ids (garbage allowed where mask=0)
        mask_labeled: [B, H, W]     1 where a point label exists, 0 otherwise
        """
        log_probs = F.log_softmax(logits, dim=1)                # [B, C, H, W]
        probs = log_probs.exp()

        target_onehot = F.one_hot(target, self.num_classes)     # [B, H, W, C]
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        pt = (probs * target_onehot).sum(dim=1)                 # [B, H, W]
        logpt = (log_probs * target_onehot).sum(dim=1)          # [B, H, W]

        focal_term = (1 - pt).clamp(min=self.eps) ** self.gamma
        per_pixel_loss = -focal_term * logpt                    # [B, H, W]

        if self.alpha is not None:
            alpha_map = (target_onehot * self.alpha.view(1, -1, 1, 1)).sum(dim=1)
            per_pixel_loss = per_pixel_loss * alpha_map

        masked_loss = per_pixel_loss * mask_labeled
        denom = mask_labeled.sum().clamp(min=1.0)
        return masked_loss.sum() / denom
```

Key points to call out in the report:
- Denominator is `Σ(MASK_labeled)`, **not** `H×W×B` — this is what makes it "partial."
- Focal term `(1-pt)^gamma` down-weights easy/confident pixels, matching the
  assessment's explicit choice of Focal Loss as the base term.
- `gamma`, `alpha` are tunable — worth a short note on default choices (gamma=2 is
  the standard focal-loss default).

## 7. Model Architecture

- **Backbone**: U-Net with ResNet-34 encoder, pretrained on ImageNet
  (`segmentation_models_pytorch.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=N)`)
- Justifies the "transfer learning" component of the original project description.
- Optionally try DeepLabV3+ as a second architecture if time allows (not required).

## 8. Training Pipeline

1. Load LandCover.ai tiles, split train/val/test.
2. Apply point simulation to *training* masks only (val/test keep full masks, used
   purely for fair evaluation of mIoU / pixel accuracy — this is standard practice
   in point-supervised segmentation literature).
3. Standard augmentations (flip, rotate, color jitter) via `albumentations` — apply
   the **same** spatial transform to image, sparse label, and `MASK_labeled`.
4. Optimizer: AdamW, cosine LR schedule, ~30–50 epochs (dataset-size dependent).
5. Loss: `PartialFocalCE`.
6. Evaluation: mIoU, per-class IoU, pixel accuracy — computed against **full**
   validation masks (never the sparse ones).
7. Baseline for comparison: same model trained with **full** dense-mask CE loss,
   to show the performance gap point-supervision incurs.

## 9. Experiments (choose 1–2 factors)

### Factor A — Point density
Train identical models with `N ∈ {5, 20, 50, 100, "full mask"}` points per class per
image. Plot mIoU vs. N. 
**Hypothesis:** performance rises steeply at first and plateaus, approaching the
full-supervision baseline as N grows.

### Factor B — Sampling strategy
Compare **uniform random** point sampling vs. **class-balanced stratified** sampling
at fixed N. 
**Hypothesis:** class-balanced sampling improves minority-class IoU (e.g., "water")
more than overall pixel accuracy, since random sampling under-represents rare classes.

(Optional Factor C, if time allows: focal-loss `gamma` value, or backbone choice
ResNet18 vs ResNet34.)

Each experiment should report: mIoU, per-class IoU, a qualitative prediction grid
(input / GT / prediction), and a training-loss curve.

## 10. Technical Report Structure

```
1. Method
   1.1 Problem formulation
   1.2 Point-label simulation procedure
   1.3 Partial Focal CE loss (formula + implementation notes)
   1.4 Network architecture & training setup

2. Experiments
   2.1 Experiment 1 — Point Density
       - Purpose / Hypothesis
       - Experimental process (setup, splits, hyperparameters)
       - Results (table + plot + qualitative examples)
   2.2 Experiment 2 — Sampling Strategy
       - Purpose / Hypothesis
       - Experimental process
       - Results

3. Discussion
   - Key findings, gap to full supervision, failure cases

4. Conclusion & Future Work
```

## 11. Suggested Timeline

| Step | Task |
|---|---|
| 1 | Data download + point simulation utility |
| 2 | Implement & unit-test Partial Focal CE loss |
| 3 | Training pipeline + sanity-check run (few epochs) |
| 4 | Full training run (baseline + point-supervised) |
| 5 | Run Experiment A (point density sweep) |
| 6 | Run Experiment B (sampling strategy) |
| 7 | Write technical report, finalize notebook |

---

**Next step:** I can build out the actual notebook/Python files (loss module, point
sampler, training script, experiment runner) — just say the word and I'll implement
them end-to-end.
