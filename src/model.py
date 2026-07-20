"""
model.py
Segmentation model builder. Uses segmentation_models_pytorch for a U-Net with
an ImageNet-pretrained encoder (transfer learning component referenced in the
original project description).
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    num_classes: int,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    architecture: str = "unet",
) -> nn.Module:
    """
    Parameters
    ----------
    num_classes : number of trainable output classes (exclude ignore/unknown)
    encoder_name : any encoder supported by segmentation_models_pytorch,
        e.g. 'resnet18', 'resnet34', 'resnet50'
    encoder_weights : 'imagenet' for pretrained weights, or None for
        random init (used as an ablation baseline if desired)
    architecture : 'unet' or 'deeplabv3plus'
    """
    architecture = architecture.lower()
    common_kwargs = dict(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )

    try:
        if architecture == "unet":
            model = smp.Unet(**common_kwargs)
        elif architecture in ("deeplabv3plus", "deeplabv3+"):
            model = smp.DeepLabV3Plus(**common_kwargs)
        else:
            raise ValueError(f"unknown architecture {architecture!r}")
    except Exception as e:
        if encoder_weights is not None:
            
            print(f"[build_model] WARNING: failed to download '{encoder_weights}' weights "
                  f"for encoder '{encoder_name}' ({type(e).__name__}: {e}). "
                  f"Falling back to encoder_weights=None (random init).")
            common_kwargs["encoder_weights"] = None
            if architecture == "unet":
                model = smp.Unet(**common_kwargs)
            else:
                model = smp.DeepLabV3Plus(**common_kwargs)
        else:
            raise

    return model
