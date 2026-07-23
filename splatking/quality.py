"""Blur / sharpness scoring for flat (non-equirectangular) frames.

The 360 Camera plugin's sharpness filter is equirectangular-only, and
SplatReady/`frame_timecodes.csv` carry no per-frame quality score. This module
provides a Laplacian-variance sharpness metric that works on SplatKing's flat
wide/ultra frames, plus helpers to keep only sufficiently sharp frames.

OpenCV is used when present; otherwise a pure-numpy Laplacian is used. If
neither is available the functions degrade to "accept everything" so the rest
of the pipeline still runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class FrameScore:
    path: str
    sharpness: float  # Laplacian variance; higher = sharper
    kept: bool = True


def _load_gray(path: str):
    """Load an image as a 2D float grayscale numpy array, or None."""
    try:
        import numpy as np
    except ImportError:
        return None

    # Prefer OpenCV (fast, handles many formats).
    try:
        import cv2  # type: ignore

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return img.astype("float32")
    except ImportError:
        pass

    # Fallback: Pillow.
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            return np.asarray(im.convert("L"), dtype="float32")
    except Exception:
        return None


def laplacian_variance(path: str) -> Optional[float]:
    """Return the variance of the Laplacian of a grayscale image (sharpness)."""
    gray = _load_gray(path)
    if gray is None:
        return None

    try:
        import cv2  # type: ignore

        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except ImportError:
        pass

    import numpy as np

    # Discrete Laplacian via a 3x3 kernel, implemented with slicing (no scipy).
    g = gray
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1] = (
        g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    )
    return float(lap[1:-1, 1:-1].var())


def score_frames(paths: list[str]) -> list[FrameScore]:
    scores: list[FrameScore] = []
    for p in paths:
        s = laplacian_variance(p)
        scores.append(FrameScore(path=p, sharpness=s if s is not None else float("inf")))
    return scores


def filter_frames(
    paths: list[str],
    abs_threshold: Optional[float] = None,
    percentile: Optional[float] = None,
) -> list[FrameScore]:
    """Score then flag frames to keep.

    * ``abs_threshold`` - keep frames with sharpness >= threshold.
    * ``percentile``    - drop the blurriest ``percentile`` fraction (0-1),
                          adaptive to the capture. Applied after abs_threshold.

    If scoring is unavailable (no numpy/cv2/PIL), every frame is kept.
    """
    scores = score_frames(paths)

    finite = [s.sharpness for s in scores if s.sharpness != float("inf")]
    if not finite:
        return scores  # scoring unavailable -> keep all

    cutoff = None
    if percentile is not None and 0.0 < percentile < 1.0:
        ordered = sorted(finite)
        idx = int(len(ordered) * percentile)
        idx = min(max(idx, 0), len(ordered) - 1)
        cutoff = ordered[idx]

    for s in scores:
        keep = True
        if abs_threshold is not None and s.sharpness < abs_threshold:
            keep = False
        if cutoff is not None and s.sharpness < cutoff:
            keep = False
        s.kept = keep
    return scores


def scoring_available() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        from PIL import Image  # noqa: F401

        return True
    except ImportError:
        return False
