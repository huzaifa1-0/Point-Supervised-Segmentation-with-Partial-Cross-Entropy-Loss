import sys
import os
import glob
from PIL import Image
import numpy as np

# Add src directory to the module path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_utils import tile_image_and_mask, rgb_mask_to_class_id
from experiments import run_point_density_experiment
from train import TrainConfig

def run_actual_data_check():
    print("Starting Diagnostics on Actual Dataset...")
    
    # 1. Discovering Image Pairs
    data_dir = r"u:\assesment\DeepGlobe Land Cover Classification Dataset\train"
    print(f"\n1. Discovering images in {data_dir}...")
    
    # Get all satellite images
    all_images = sorted(glob.glob(os.path.join(data_dir, "*_sat.jpg")))
    
    if not all_images:
        raise ValueError(f"No *_sat.jpg images found in {data_dir}")
        
    print(f"Total source images found: {len(all_images)}")
    
    # IMPORTANT: Loading all 800+ high-res images at once into memory (via build_tiles) 
    # will cause an out-of-memory crash. For this diagnostic test, we use a 5-image subset.
    subset_size = 5
    print(f"Taking a subset of {subset_size} images to prevent memory crash during quick test...")
    subset = all_images[:subset_size]
    
    pairs = []
    for img_path in subset:
        basename = os.path.basename(img_path).replace("_sat.jpg", "")
        mask_path = os.path.join(data_dir, basename + "_mask.png")
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))

    print(f"Successfully matched {len(pairs)} image/mask pairs.")
    
    # 2. Tiling Images
    print("\n2. Tiling high-resolution images (512x512 patches)...")
    tiles = []
    for idx, (img_path, mask_path) in enumerate(pairs):
        print(f"  Tiling image {idx+1}/{len(pairs)}: {os.path.basename(img_path)}")
        img_np = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        mask_id = rgb_mask_to_class_id(mask_rgb)
        
        # Use stride=512, drop_partial=True like the recommended notebook settings
        tiles.extend(tile_image_and_mask(img_np, mask_id, tile_size=512, stride=512, drop_partial=True))
    
    print(f"Created {len(tiles)} 512x512 tiles.")
    
    # 80/20 train/val split (Note: split should ideally be on source images like in the notebook, doing tiles here for simplicity of the smoke test)
    n_train = int(0.8 * len(tiles))
    train_tiles = tiles[:n_train]
    val_tiles = tiles[n_train:]
    
    # 3. Run a tiny test experiment
    print("\n3. Running 1 epoch as smoke test on real tiles...")
    # Use config that runs fast
    config = TrainConfig(epochs=1, batch_size=4, num_workers=0)
    
    try:
        results = run_point_density_experiment(
            train_tiles, 
            val_tiles, 
            point_counts=[5], 
            strategy="random", 
            train_config=config, 
            encoder_name="resnet18"
        )
        print("\n==============================================")
        print("    REAL DATASET PROJECT CHECK SUCCESSFUL!")
        print("==============================================")
        print("Results generated:")
        print(results)
    except Exception as e:
        print("\n==============================================")
        print("          PROJECT CHECK FAILED")
        print("==============================================")
        print(f"Error encountered: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_actual_data_check()
