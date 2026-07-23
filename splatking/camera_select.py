"""Training-camera subsampling for VRAM-constrained training.

Reconstruction can keep every registered image; training does not have to.
This module writes a filtered COLMAP images.txt (or an image list) so LichtFeld
can train on every-N / random-% cameras while the sparse geometry stays intact.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

from .colmap_model import ColmapImage, ColmapModel, read_model


@dataclass
class CameraSelectOptions:
    mode: str = "every_n"          # every_n | random_pct | keep_all
    every_n: int = 2
    random_pct: float = 0.5        # 0..1
    seed: int = 42
    prefer_high_quality: bool = False
    # Optional map of image basename -> quality score (LiDAR path).
    quality_scores: Optional[dict[str, float]] = None


def select_image_indices(n: int, opts: CameraSelectOptions) -> list[int]:
    if n <= 0:
        return []
    if opts.mode == "keep_all":
        return list(range(n))
    if opts.mode == "every_n":
        step = max(1, int(opts.every_n))
        return list(range(0, n, step))
    if opts.mode == "random_pct":
        pct = min(max(float(opts.random_pct), 0.0), 1.0)
        k = max(1, int(round(n * pct)))
        rng = random.Random(opts.seed)
        return sorted(rng.sample(range(n), k))
    raise ValueError(f"Unknown camera-select mode: {opts.mode}")


def subsample_model(model: ColmapModel, opts: CameraSelectOptions) -> ColmapModel:
    """Return a new ColmapModel keeping a subset of images (points3D unchanged).

    Track entries that reference dropped images are left as-is; LichtFeld and
    COLMAP ignore 2D-3D tracks for missing images during training, and we
    deliberately keep the full sparse cloud as the geometric prior.
    """
    images = list(model.images)
    if opts.prefer_high_quality and opts.quality_scores:
        images = sorted(
            images,
            key=lambda im: opts.quality_scores.get(os.path.basename(im.name), 0.0),
            reverse=True,
        )
    keep_idx = set(select_image_indices(len(images), opts))
    kept = [im for i, im in enumerate(images) if i in keep_idx]
    # Re-number image_ids consecutively for cleanliness.
    renumbered: list[ColmapImage] = []
    for new_id, im in enumerate(kept, start=1):
        renumbered.append(
            ColmapImage(
                image_id=new_id,
                qw=im.qw, qx=im.qx, qy=im.qy, qz=im.qz,
                tx=im.tx, ty=im.ty, tz=im.tz,
                camera_id=im.camera_id,
                name=im.name,
                points2d=list(im.points2d),
            )
        )
    return ColmapModel(
        cameras=list(model.cameras),
        images=renumbered,
        points3d=list(model.points3d),
    )


def write_training_subset(
    sparse_dir: str,
    out_sparse_dir: str,
    opts: CameraSelectOptions,
) -> dict:
    """Read sparse_dir, subsample images, write to out_sparse_dir."""
    model = read_model(sparse_dir)
    before = len(model.images)
    subset = subsample_model(model, opts)
    after = len(subset.images)
    os.makedirs(out_sparse_dir, exist_ok=True)
    subset.write(out_sparse_dir)
    return {
        "source_images": before,
        "training_images": after,
        "mode": opts.mode,
        "every_n": opts.every_n,
        "random_pct": opts.random_pct,
        "out_sparse_dir": out_sparse_dir,
    }
