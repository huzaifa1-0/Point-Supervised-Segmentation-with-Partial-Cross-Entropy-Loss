from __future__ import annotations

import numpy as np


CLASS_NAMES = [
    "Urban land",
    "Agriculture land",
    "Rangeland",
    "Forest land",
    "Water",
    "Barren land",
    "Unknown",
]

CLASS_COLORS = np.array(
    [
        (0, 255, 255),   
        (255, 255, 0),   
        (255, 0, 255),   
        (0, 255, 0),     
        (0, 0, 255),     
        (255, 255, 255), 
        (0, 0, 0),      
    ],
    dtype=np.uint8,
)

NUM_CLASSES = 6          
IGNORE_INDEX = 6          


def rgb_mask_to_class_id(rgb_mask: np.ndarray) -> np.ndarray:
    """
    Convert an (H, W, 3) RGB color-coded mask into an (H, W) integer class-ID
    mask using the DeepGlobe color table. Any pixel whose color does not match
    one of the 7 known colors exactly is mapped to IGNORE_INDEX (robust to
    JPEG-artifact-style color drift near class boundaries, which is common in
    this dataset's PNG masks if they were ever re-encoded).

    Parameters
    ----------
    rgb_mask : np.ndarray, shape (H, W, 3), dtype uint8

    Returns
    -------
    class_id_mask : np.ndarray, shape (H, W), dtype int64
    """
    assert rgb_mask.ndim == 3 and rgb_mask.shape[2] == 3, \
        f"expected (H, W, 3) RGB mask, got shape {rgb_mask.shape}"

    h, w, _ = rgb_mask.shape
    class_id_mask = np.full((h, w), IGNORE_INDEX, dtype=np.int8)

    # Memory-efficient encoding without large intermediate arrays
    encoded = (rgb_mask[..., 0].astype(np.int32) * 65536 + 
               rgb_mask[..., 1].astype(np.int32) * 256 + 
               rgb_mask[..., 2].astype(np.int32))

    for class_id, color in enumerate(CLASS_COLORS):
        color_code = int(color[0]) * 65536 + int(color[1]) * 256 + int(color[2])
        class_id_mask[encoded == color_code] = class_id

    return class_id_mask


def class_id_to_rgb_mask(class_id_mask: np.ndarray) -> np.ndarray:
    """Inverse of rgb_mask_to_class_id -- useful for visualizing predictions."""
    h, w = class_id_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        rgb[class_id_mask == class_id] = color
    return rgb




def tile_image_and_mask(
    image: np.ndarray,
    class_id_mask: np.ndarray,
    tile_size: int = 512,
    stride: int | None = None,
    drop_partial: bool = True,
):
    """
    Split a large image + class-ID mask into a grid of smaller tiles.

    Parameters
    ----------
    image : (H, W, 3) uint8
    class_id_mask : (H, W) int64
    tile_size : side length of square tiles
    stride : step between tile origins; defaults to tile_size (non-overlapping)
    drop_partial : if True, discard tiles that would run off the image edge
                   (simplest, avoids padding artifacts); if False, pad with
                   IGNORE_INDEX / zeros.

    Returns
    -------
    list of (image_tile, mask_tile) tuples
    """
    if stride is None:
        stride = tile_size

    h, w = class_id_mask.shape
    tiles = []

    ys = list(range(0, h - tile_size + 1, stride)) if drop_partial else list(range(0, h, stride))
    xs = list(range(0, w - tile_size + 1, stride)) if drop_partial else list(range(0, w, stride))

    if not drop_partial:
        
        if ys[-1] + tile_size < h:
            ys.append(h - tile_size)
        if xs[-1] + tile_size < w:
            xs.append(w - tile_size)

    for y in ys:
        for x in xs:
            y0, y1 = y, y + tile_size
            x0, x1 = x, x + tile_size
            if y1 > h or x1 > w:
                if drop_partial:
                    continue
                
                img_tile = np.zeros((tile_size, tile_size, 3), dtype=image.dtype)
                mask_tile = np.full((tile_size, tile_size), IGNORE_INDEX, dtype=class_id_mask.dtype)
                y1c, x1c = min(y1, h), min(x1, w)
                img_tile[: y1c - y0, : x1c - x0] = image[y0:y1c, x0:x1c]
                mask_tile[: y1c - y0, : x1c - x0] = class_id_mask[y0:y1c, x0:x1c]
            else:
                img_tile = image[y0:y1, x0:x1].copy()
                mask_tile = class_id_mask[y0:y1, x0:x1].copy()
            tiles.append((img_tile, mask_tile))

    return tiles
