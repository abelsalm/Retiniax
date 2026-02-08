#!/usr/bin/env python3
"""
Quick visual sanity-check for the MONAI augmentation pipeline defined in transforms.py.

Usage:
    python test_da.py --folder /path/to/images --limit 16 --seed 42

The script loads the first N images it finds in the folder, applies the MONAI
transform sequence, and renders a single matplotlib figure showing the originals
and their augmented counterparts so you can eyeball the impact of the data
augmentation.
"""

from __future__ import annotations

import argparse
import random
import traceback
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirstD
from monai.utils import set_determinism
from PIL import Image

from transforms import monai_transform_sequence_test

# Key used by the MONAI dictionary transforms; keep in sync with transforms.py
IMG_KEY = "image"

# Supported raster extensions for quick-and-dirty folder crawling.
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# These come from the NormalizeIntensityd call inside transforms.py
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MONAI data augmentation.")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder that contains the raw fundus images (any mix of PNG/JPEG/TIFF).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=16,
        help="Number of images to sample (default: 16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for MONAI's random transforms to obtain reproducible runs.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the final grid (e.g. /tmp/da_grid.png).",
    )
    return parser.parse_args()


def collect_image_paths(folder: str | Path, limit: int) -> List[Path]:
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    image_paths: List[Path] = []
    for path in sorted(folder_path.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
            image_paths.append(path)
        if len(image_paths) >= limit:
            break

    if not image_paths:
        raise ValueError(f"No images with extensions {SUPPORTED_EXTENSIONS} found in {folder_path}")

    if len(image_paths) < limit:
        print(f"[WARN] Only {len(image_paths)} images found; requested {limit}. Proceeding with available files.")

    return image_paths


def build_transform_with_channel_first() -> Compose:
    """Inject EnsureChannelFirstD at the start if the user forgot to add it in transforms.py."""
    transforms_list: List = list(monai_transform_sequence_test.transforms)
    if not any(isinstance(t, EnsureChannelFirstD) for t in transforms_list):
        transforms_list.insert(0, EnsureChannelFirstD(keys=[IMG_KEY], channel_dim=-1))
    return Compose(transforms_list)


def load_image(path: Path) -> np.ndarray:
    """Return an RGB image as float32 HWC numpy array scaled to [0, 1]."""
    with Image.open(path) as img:
        image = img.convert("RGB")
        print('Loaded image :', path)
        return np.asarray(image, dtype=np.float32) / 255.0



def prepare_for_transform(image_hwcx: np.ndarray, path: Path) -> dict:
    """Wrap the numpy image and lightweight metadata for MONAI dictionary transforms."""
    meta_dict = {
        "filename_or_obj": str(path),
        "spatial_shape": image_hwcx.shape[:2],
        "original_channel_dim": -1,  # we know images are HWC
    }
    return {
        IMG_KEY: image_hwcx.copy(),
        f"{IMG_KEY}_meta_dict": meta_dict,
    }


def prepare_batch_for_transform(images: list[np.ndarray], paths: list[Path]) -> dict:
    """
    Prepare a batch of images for MONAI dictionary transforms.
    Converts list of (H, W, C) images to a batch tensor (B, C, H, W).
    """
    # Ensure all images have the same shape
    first_shape = images[0].shape
    if not all(img.shape == first_shape for img in images):
        # Resize all images to the first image's shape
        from PIL import Image as PILImage
        resized_images = []
        for img in images:
            if img.shape != first_shape:
                # Convert to PIL, resize, convert back
                pil_img = PILImage.fromarray((img * 255).astype(np.uint8))
                pil_img = pil_img.resize((first_shape[1], first_shape[0]), PILImage.Resampling.BILINEAR)
                resized = np.asarray(pil_img, dtype=np.float32) / 255.0
                resized_images.append(resized)
            else:
                resized_images.append(img)
        images = resized_images
    
    # Stack images: (B, H, W, C) -> (B, C, H, W) after channel-first conversion
    # First convert each image to channel-first: (H, W, C) -> (C, H, W)
    images_channel_first = [np.moveaxis(img, -1, 0) for img in images]  # List of (C, H, W)
    
    # Stack into batch: (B, C, H, W)
    batch_tensor = np.stack(images_channel_first, axis=0)  # (B, C, H, W)
    
    # Create meta_dict for the batch (using first image's metadata)
    meta_dict = {
        "filename_or_obj": [str(p) for p in paths],
        "spatial_shape": [img.shape[:2] for img in images],
        "original_channel_dim": -1,
    }
    
    return {
        IMG_KEY: batch_tensor,
        f"{IMG_KEY}_meta_dict": meta_dict,
    }


def tensor_to_display_image(arr: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert a (possibly normalized) tensor/array back to an RGB image suitable for plt.imshow.
    
    Note: This function tries to detect if normalization was applied by checking if values
    are in a normalized range (typically negative or > 1). If not normalized, it just
    ensures values are in [0, 1] range.
    """
    if torch.is_tensor(arr):
        data = arr.detach().cpu().numpy()
    else:
        data = np.asarray(arr)

    # Move channel-first back to channel-last if needed.
    if data.ndim == 3 and data.shape[0] in (1, 3) and data.shape[2] not in (1, 3):
        data = np.moveaxis(data, 0, -1)

    # Check if data appears to be normalized (has negative values or values > 1)
    # If normalized, undo it. Otherwise, just clip to [0, 1]
    data_min = data.min()
    data_max = data.max()
    
    # If data has negative values or max > 1.5, it's likely normalized
    if data_min < -0.5 or data_max > 1.5:
        # Undo normalization (NormalizeIntensityd applied per channel)
        data = data * RGB_STD + RGB_MEAN
    
    # Ensure values are in [0, 1] for display
    data = np.clip(data, 0.0, 1.0)
    return data


def build_dataset_samples(
    image_paths: Sequence[Path], transform: Compose
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Path]]:
    """
    Load images and apply transforms. 
    Processes images individually (for 2D transforms) but groups them for CutOutd.
    """
    from monai.transforms import CutOutd
    
    # Load all original images
    originals: List[np.ndarray] = []
    valid_paths: List[Path] = []
    
    for path in image_paths:
        try:
            original = load_image(path)
            originals.append(original)
            valid_paths.append(path)
        except Exception as exc:
            print(f"[WARN] Failed to load {path.name}: {exc}")
            continue
    
    if not originals:
        return [], [], []
    
    # Split transforms into: before CutOutd, CutOutd, after CutOutd
    transforms_list = list(transform.transforms)
    cutout_idx = None
    for i, t in enumerate(transforms_list):
        if isinstance(t, CutOutd):
            cutout_idx = i
            break
    
    if cutout_idx is None:
        # No CutOutd found, process normally image by image
        augmenteds: List[np.ndarray] = []
        for i, path in enumerate(valid_paths):
            original = originals[i]
            try:
                transformed = transform(prepare_for_transform(original, path))[IMG_KEY]
                augmenteds.append(tensor_to_display_image(transformed))
            except Exception as e:
                print(f"[WARN] Failed to transform {path.name}: {e}")
                continue
        return originals, augmenteds, valid_paths
    
    # Split pipeline: before CutOutd, CutOutd, after CutOutd
    from monai.transforms import Compose
    transforms_before = Compose(transforms_list[:cutout_idx])
    cutout_transform = transforms_list[cutout_idx]
    transforms_after = Compose(transforms_list[cutout_idx + 1:])
    
    # Process images individually up to CutOutd
    images_before_cutout: List[np.ndarray] = []
    successful_originals: List[np.ndarray] = []
    successful_paths: List[Path] = []
    
    for i, path in enumerate(valid_paths):
        original = originals[i]
        try:
            transformed = transforms_before(prepare_for_transform(original, path))[IMG_KEY]
            # Ensure it's numpy and channel-first (C, H, W)
            if torch.is_tensor(transformed):
                transformed = transformed.detach().cpu().numpy()
            images_before_cutout.append(transformed)
            successful_originals.append(original)
            successful_paths.append(path)
        except Exception as e:
            print(f"[WARN] Failed to transform {path.name} before CutOutd: {e}")
            # Skip this image
            continue
    
    if not images_before_cutout:
        return [], [], []
    
    # Group into batch for CutOutd: (B, C, H, W)
    # Ensure all images have the same shape
    first_shape = images_before_cutout[0].shape
    if not all(img.shape == first_shape for img in images_before_cutout):
        # Resize all to first image's shape using PIL
        resized = []
        for img in images_before_cutout:
            if img.shape != first_shape:
                # Convert (C, H, W) to (H, W, C) for PIL
                img_hwc = np.moveaxis(img, 0, -1)
                pil_img = Image.fromarray((img_hwc * 255).astype(np.uint8))
                target_h, target_w = first_shape[1], first_shape[2]
                pil_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
                img_resized = np.asarray(pil_img, dtype=np.float32) / 255.0
                # Convert back to (C, H, W)
                img_resized = np.moveaxis(img_resized, -1, 0)
                resized.append(img_resized)
            else:
                resized.append(img)
        images_before_cutout = resized
    
    # Stack into batch: (B, C, H, W)
    batch_tensor = np.stack(images_before_cutout, axis=0)
    
    # Apply CutOutd to batch
    try:
        batch_data = {IMG_KEY: batch_tensor}
        batch_after_cutout = cutout_transform(batch_data)[IMG_KEY]
        
        # Convert back to list of (C, H, W)
        if torch.is_tensor(batch_after_cutout):
            batch_np = batch_after_cutout.detach().cpu().numpy()
        else:
            batch_np = np.asarray(batch_after_cutout)
        
        images_after_cutout = [batch_np[i] for i in range(batch_np.shape[0])]
    except Exception as e:
        print(f"[WARN] CutOutd failed, skipping it: {e}")
        images_after_cutout = images_before_cutout
    
    # Process remaining transforms individually
    augmenteds: List[np.ndarray] = []
    kept_originals: List[np.ndarray] = []
    kept_paths: List[Path] = []
    
    for i, img in enumerate(images_after_cutout):
        if i >= len(successful_paths):
            break
        path = successful_paths[i]
        try:
            # Prepare as single image dict
            img_dict = {IMG_KEY: img}
            transformed = transforms_after(img_dict)[IMG_KEY]
            augmenteds.append(tensor_to_display_image(transformed))
            kept_originals.append(successful_originals[i])
            kept_paths.append(path)
        except Exception as e:
            print(f"[WARN] Failed to transform {path.name} after CutOutd: {e}")
            continue
    
    return kept_originals, augmenteds, kept_paths


def make_grid_plot(
    originals: Sequence[np.ndarray],
    augmenteds: Sequence[np.ndarray],
    image_paths: Sequence[Path],
    save_path: str | None = None,
) -> None:
    # Use the minimum length to ensure both lists have corresponding elements
    num_images = min(len(originals), len(augmenteds))
    if num_images == 0:
        raise RuntimeError("No samples available to plot.")
    
    if len(originals) != len(augmenteds):
        print(f"[WARN] Mismatch: {len(originals)} originals vs {len(augmenteds)} augmenteds. Using {num_images} samples.")

    cols = 8  # 2 rows of 8 for originals, 2 rows for augmentations -> 4 total rows
    rows = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    fig.suptitle("Data augmentation preview (top: originals, bottom: augmented)", fontsize=16)

    # Flatten axes for easier indexing
    axes = axes.reshape(rows, cols)

    def plot_row(row_offset: int, imgs: Sequence[np.ndarray], label: str) -> None:
        for idx in range(cols * 2):  # two rows per section
            row = row_offset + idx // cols
            col = idx % cols
            ax = axes[row, col]
            data_idx = idx
            if data_idx >= num_images:
                ax.axis("off")
                continue
            ax.imshow(imgs[data_idx])
            ax.axis("off")
            ax.set_title(f"{label} #{data_idx+1}", fontsize=8)

    plot_row(0, originals, "Orig")
    plot_row(2, augmenteds, "Aug")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        out_path = Path(save_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"[INFO] Saved grid to {out_path}")
    plt.savefig('data_aug_visualize.png')


def main() -> None:
    '''args = parse_args()'''

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    set_determinism(0)

    folder = '/Users/abelsalmona/Documents/Retinax/Training Repo/data/test_sample'
    limit = 16

    image_paths = collect_image_paths(folder, limit)
    transform = build_transform_with_channel_first()

    originals, augmenteds, kept_paths = build_dataset_samples(image_paths, transform)

    if not kept_paths:
        raise RuntimeError("No samples were successfully transformed. Check your input folder and transforms.")

    print(f"[INFO] Prepared {len(kept_paths)} samples from {Path(folder).resolve()}")
    make_grid_plot(originals, augmenteds, kept_paths, save_path='/Users/abelsalmona/Documents/Retinax/repository/transforms.png')


if __name__ == "__main__":
    main()
