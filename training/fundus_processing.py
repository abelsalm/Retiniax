"""
Fundus image enhancement transforms inspired by:

    "Ensemble Deep Learning for Diabetic Retinopathy Detection"

Each function takes a uint8 RGB image (H, W, 3) and returns
an enhanced uint8 image (H, W, 3).

    1. min_pooling          — Ben Graham's local-average subtraction (RGB)
    2. green_ben_graham     — Ben Graham on the green channel only, with
                              optional CLAHE post-processing (3-channel output)
    3. rgb_clahe            — CLAHE applied on every RGB channel
    4. lab_clahe            — CLAHE applied on L channel of CIE-Lab
    5. maxgreengsc_clahe    — CLAHE on [Grayscale, Green, MaxPixel] composite
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1. Min-pooling  (Ben Graham, Kaggle DR competition)
# ──────────────────────────────────────────────────────────────────────────────
def min_pooling(
    image: np.ndarray,
    alpha: float = 4.0,
    beta: float = -4.0,
    sigma: float = 10.0,
    gamma: float = 128.0,
) -> np.ndarray:
    """
    Subtract the local average colour and re-centre on middle grey.

        output = α · I  +  β · (I ∗ G_σ)  +  γ

    With the default parameters (α=4, β=−4, σ=10, γ=128) this simplifies to

        output = 4 · (I − blur(I)) + 128

    which removes low-frequency illumination variations and highlights
    fine-grained lesions such as micro-aneurysms and exudates.

    Args:
        image:  uint8 RGB image, shape (H, W, 3).
        alpha:  weight on the original image.
        beta:   weight on the Gaussian-blurred image (negative → subtraction).
        sigma:  standard deviation of the Gaussian kernel.
        gamma:  constant baseline shift (128 = middle grey).

    Returns:
        uint8 RGB image, shape (H, W, 3).
    """
    img_f = image.astype(np.float32)

    # Gaussian blur to estimate local average colour
    # ksize=(0,0) lets OpenCV auto-compute kernel size from sigma
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma)

    result = alpha * img_f + beta * blurred + gamma
    return np.clip(result, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Green Ben Graham
# ──────────────────────────────────────────────────────────────────────────────
def green_ben_graham(
    image: np.ndarray,
    alpha: float = 4.0,
    beta: float = -4.0,
    sigma: float = 10.0,
    gamma: float = 128.0,
    clahe_clip: float = 2.0,
    clahe_tile: tuple[int, int] = (8, 8),
    apply_clahe: bool = True,
) -> np.ndarray:
    """
    Green-channel variant of the Ben Graham preprocessing.

    The green channel of a fundus image carries the highest contrast for
    retinal structures (vessels, micro-aneurysms, haemorrhages) because
    haemoglobin absorbs green light strongly. By isolating this channel
    before applying the local-average subtraction we get:

        1. Better signal-to-noise than the full RGB Ben Graham
           (red is often saturated, blue is noisy).
        2. Stronger contrast on vascular and haemorrhagic lesions.

    Pipeline:
        ① Extract the green channel.
        ② Apply Ben Graham:  out = α·G + β·blur(G) + γ
        ③ (optional) Apply CLAHE to further boost local contrast.
        ④ Stack into a 3-channel image [green_bg, green_bg, green_bg]
           so the output is compatible with standard 3-channel models.

    Args:
        image:      uint8 RGB image, shape (H, W, 3).
        alpha:      weight on the original green channel.
        beta:       weight on the Gaussian-blurred green channel.
        sigma:      standard deviation of the Gaussian kernel.
        gamma:      constant baseline shift (128 = middle grey).
        clahe_clip: CLAHE clip limit (ignored if apply_clahe=False).
        clahe_tile: CLAHE tile grid size (ignored if apply_clahe=False).
        apply_clahe: if True, apply CLAHE after the Ben Graham step.

    Returns:
        uint8 image, shape (H, W, 3) — 3 identical channels.
    """
    green = image[:, :, 1].astype(np.float32)

    blurred = cv2.GaussianBlur(green, (0, 0), sigmaX=sigma)
    result = alpha * green + beta * blurred + gamma
    result = np.clip(result, 0, 255).astype(np.uint8)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
        result = clahe.apply(result)

    return np.stack([result, result, result], axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# 3. RGB-CLAHE
# ──────────────────────────────────────────────────────────────────────────────
def rgb_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalisation (CLAHE)
    independently to each RGB channel.

    CLAHE equalises contrast on small tiles and merges them with bilinear
    interpolation, avoiding the over-amplification artefacts of plain AHE.

    Args:
        image:          uint8 RGB image, shape (H, W, 3).
        clip_limit:     threshold for contrast limiting per tile.
        tile_grid_size: number of tiles in each dimension (rows, cols).

    Returns:
        uint8 RGB image, shape (H, W, 3).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    channels = cv2.split(image)
    eq_channels = [clahe.apply(ch) for ch in channels]
    return cv2.merge(eq_channels)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Lab-CLAHE
# ──────────────────────────────────────────────────────────────────────────────
def lab_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE only to the luminance (L) channel of the CIE-Lab colour
    space, then convert back to RGB.

    This equalises brightness while leaving the chrominance (a, b) channels
    untouched, so colours are preserved.

    Args:
        image:          uint8 RGB image, shape (H, W, 3).
        clip_limit:     threshold for contrast limiting per tile.
        tile_grid_size: number of tiles in each dimension (rows, cols).

    Returns:
        uint8 RGB image, shape (H, W, 3).
    """
    # OpenCV expects BGR for cvtColor, but input is RGB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # equalise L channel only

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ──────────────────────────────────────────────────────────────────────────────
# 4. MaxGreenGsc-CLAHE
# ──────────────────────────────────────────────────────────────────────────────
def maxgreengsc_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Build a composite 3-channel image from:

        channel 0 — grayscale
        channel 1 — green channel  (most informative for retinal vessels)
        channel 2 — max-pixel      (max(R, G, B) per pixel)

    then apply CLAHE to each channel independently.

    The green channel carries the highest contrast for retinal structures
    (vessels, micro-aneurysms). The max-pixel channel preserves bright
    lesions (exudates) that may be dominant in only one colour channel.

    Args:
        image:          uint8 RGB image, shape (H, W, 3).
        clip_limit:     threshold for contrast limiting per tile.
        tile_grid_size: number of tiles in each dimension (rows, cols).

    Returns:
        uint8 image, shape (H, W, 3)  [Grayscale, Green, MaxPixel].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    # (H, W)
    green = image[:, :, 1]                             # (H, W)
    max_pixel = np.max(image, axis=2)                  # (H, W)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    gray_eq = clahe.apply(gray)
    green_eq = clahe.apply(green)
    max_eq = clahe.apply(max_pixel)

    return np.stack([gray_eq, green_eq, max_eq], axis=-1)
