"""
image_quality.py — Local, no-API image quality assessment (IQA) for car photos.

============================================================================
CHOSEN APPROACH: Classical CV IQA  (no external LLM / Vision API)
============================================================================

Why Classical CV instead of NIMA / MUSIQ / CLIP-IQA?
  1. The assignment forbids external LLM/Vision APIs (GPT-4V, Claude Vision,
     Gemini Vision). Deep-Learning IQA models (NIMA, MUSIQ, HyperIQA, MANIQA)
     and CLIP-IQA are *local* and therefore allowed, BUT they need PyTorch +
     pretrained weight files (tens to hundreds of MB) downloaded from the
     internet. This sandbox does not ship torch/transformers/open_clip, and
     relying on a weight download makes the pipeline fragile and slow when
     batch-scoring hundreds of images.

  2. Classical CV metrics run instantly with OpenCV/scikit-image, need no
     weights, and every sub-score is explainable and maps to a factor a human
     reviewer cares about for a car listing photo: sharpness (blur), exposure
     (too dark / too bright), contrast/tonal range, and colour richness.

  3. The resulting 0-10 score is a weighted blend of these explainable
     sub-scores, so a low score can be diagnosed ("blurry", "underexposed"),
     which is more useful for a data team than an opaque number.

"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np
from skimage import exposure


# ---------------------------------------------------------------------------
# Sub-metric scorers. Each returns a float in [0, 10] (10 = best).
# ---------------------------------------------------------------------------

def sharpness_score(gray: np.ndarray) -> float:
    """Blur detection via the variance of the Laplacian.

    The Laplacian highlights second-order intensity changes (edges). A sharp
    photo has many strong edges => high variance; a blurry photo has soft
    edges => low variance. We map the variance to a 0-10 score with a
    saturating curve so that beyond a threshold extra sharpness yields
    diminishing returns (a photo can be "sharp enough").
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    # Empirically, car listing thumbnails have Laplacian variance roughly in
    # [0, ~1500]. A variance of ~400 already looks acceptably sharp, ~1000+
    # looks crisp. Use a square-root saturating mapping.
    # 10 * (1 - exp(-var / 350))  -> var=350  => ~6.3, var=800 => ~9.0,
    # var=1500 => ~9.9. Tuned by eye on a sample of taladrod thumbnails.
    score = 10.0 * (1.0 - np.exp(-var / 350.0))
    return float(np.clip(score, 0.0, 10.0))


def brightness_score(gray: np.ndarray) -> float:
    """Exposure quality from mean luminance.

    Penalize both under-exposed (too dark) and over-exposed (too bright)
    images. The ideal mean luminance for an outdoor/object photo is around
    110-150 on the 0-255 scale. We score 10 at the sweet spot and fall off
    on either side.
    """
    mean = float(gray.mean())
    # Ideal band ~128. Distance from 128, mapped through a Gaussian-like falloff.
    # sigma chosen so that mean<40 or mean>210 scores poorly (<3).
    sigma = 60.0
    score = 10.0 * np.exp(-((mean - 128.0) ** 2) / (2.0 * sigma * sigma))
    return float(np.clip(score, 0.0, 10.0))


def contrast_score(gray: np.ndarray) -> float:
    """Tonal range / contrast from the standard deviation of luminance.

    A flat, hazy photo has low std-dev (everything mid-grey); a photo with
    good dynamic range has higher std-dev. Too high can mean harsh/clipped,
    but for listing photos more contrast is generally better up to a point.
    """
    std = float(gray.std())
    # Typical std for natural photos: 40-80. Map with saturating curve.
    # std=40 => ~7.3, std=60 => ~8.8, std=80+ => ~9.5.
    score = 10.0 * (1.0 - np.exp(-std / 35.0))
    return float(np.clip(score, 0.0, 10.0))


def saturation_score(bgr: np.ndarray) -> float:
    """Saturation / vividness from the mean saturation in HSV space.

    Greyscale-ish or washed-out photos (low saturation) look dull; overly
    saturated photos look unnatural. Score peaks at a healthy mid saturation.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    mean_s = float(sat.mean())  # 0-255
    # Ideal mean saturation ~90-120. Falloff on both sides.
    sigma = 70.0
    score = 10.0 * np.exp(-((mean_s - 100.0) ** 2) / (2.0 * sigma * sigma))
    return float(np.clip(score, 0.0, 10.0))


def colorfulness_score(bgr: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colorfulness metric.

    rg = R - G, yb = 0.5*(R+G) - B. Colorfulness = sqrt(std_rg^2 + std_yb^2)
    + 0.3*sqrt(mean_rg^2 + mean_yb^2). Higher = more colourful. Natural
    photos cluster around 15-60. We map with a saturating curve.
    """
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg = float(np.std(rg))
    std_yb = float(np.std(yb))
    mean_rg = float(np.mean(rg))
    mean_yb = float(np.mean(yb))
    color = np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    # color ~0 => monochrome, ~15 => mild, ~50 => colourful, ~100 => very vivid
    score = 10.0 * (1.0 - np.exp(-color / 30.0))
    return float(np.clip(score, 0.0, 10.0))


def clipping_penalty(gray: np.ndarray) -> float:
    """Penalty for clipped (pure-black / pure-white) pixel ratio.

    Returns a multiplier in [0.7, 1.0]: a small clipped fraction is fine, but
    a large one (blown highlights / crushed shadows) reduces the final score.
    """
    total = float(gray.size)
    black = float(np.count_nonzero(gray <= 4)) / total
    white = float(np.count_nonzero(gray >= 251)) / total
    clipped = black + white
    # Allow up to ~3% clipped with no penalty; beyond that, scale down.
    if clipped <= 0.03:
        return 1.0
    # Map 3%->1.0, 20%->~0.7
    penalty = 1.0 - 0.3 * min(1.0, (clipped - 0.03) / 0.17)
    return float(max(0.7, penalty))


# ---------------------------------------------------------------------------
# Weighted fusion
# ---------------------------------------------------------------------------

# Weights reflect what matters most for a usable car listing photo:
#   - sharpness is the single most important factor (a blurry photo is useless)
#   - brightness/exposure is next (you must be able to see the car)
#   - contrast and colourfulness matter for appeal
#   - saturation is a tie-breaker / appeal factor
WEIGHTS = {
    "sharpness": 0.40,
    "brightness": 0.25,
    "contrast": 0.15,
    "colorfulness": 0.12,
    "saturation": 0.08,
}


@dataclass
class QualityResult:
    score: float                 # final 0-10 quality score (1 decimal)
    sharpness: float
    brightness: float
    contrast: float
    colorfulness: float
    saturation: float
    clipping_penalty: float
    width: int
    height: int
    note: str = ""               # short human-readable diagnosis


def assess_image(path_or_array) -> QualityResult:
    """Score one image. Accepts a file path (str) or a BGR uint8 numpy array."""
    if isinstance(path_or_array, str):
        if not os.path.exists(path_or_array):
            raise FileNotFoundError(path_or_array)
        bgr = cv2.imread(path_or_array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"OpenCV could not read image: {path_or_array}")
    else:
        bgr = path_or_array.copy()

    # Resize very large images to a working size for speed and so the
    # Laplacian variance is comparable across photos of different resolution.
    h, w = bgr.shape[:2]
    max_side = 512
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = exposure.rescale_intensity(gray, out_range=(0, 255)).astype(np.uint8)

    s_sharp = sharpness_score(gray)
    s_bright = brightness_score(gray)
    s_contrast = contrast_score(gray)
    s_color = colorfulness_score(bgr)
    s_sat = saturation_score(bgr)
    clip = clipping_penalty(gray)

    raw = sum(WEIGHTS[k] * v for k, v in [
        ("sharpness", s_sharp),
        ("brightness", s_bright),
        ("contrast", s_contrast),
        ("colorfulness", s_color),
        ("saturation", s_sat),
    ])
    final = float(np.clip(raw * clip, 0.0, 10.0))

    # Short human-readable note: list the weakest sub-scores for diagnosis.
    subs = {
        "sharpness": s_sharp, "brightness": s_bright, "contrast": s_contrast,
        "colorfulness": s_color, "saturation": s_sat,
    }
    weak = [k for k, v in sorted(subs.items(), key=lambda kv: kv[1]) if v < 5.0]
    if clip < 1.0:
        weak.append("clipped highlights/shadows")
    note = "low: " + ", ".join(weak) if weak else "balanced"

    return QualityResult(
        score=round(final, 1),
        sharpness=round(s_sharp, 2),
        brightness=round(s_bright, 2),
        contrast=round(s_contrast, 2),
        colorfulness=round(s_color, 2),
        saturation=round(s_sat, 2),
        clipping_penalty=round(clip, 2),
        width=int(w), height=int(h),
        note=note,
    )


def score_to_dict(path_or_array) -> dict:
    """Convenience wrapper returning the result as a plain dict (JSON-safe)."""
    return asdict(assess_image(path_or_array))


if __name__ == "__main__":
    # Quick CLI smoke test: `python image_quality.py some.jpg`
    import sys
    if len(sys.argv) < 2:
        print("usage: python image_quality.py <image_path>")
        sys.exit(1)
    r = assess_image(sys.argv[1])
    for k, v in asdict(r).items():
        print(f"  {k:18s}: {v}")
