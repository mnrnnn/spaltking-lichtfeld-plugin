"""Photo (dual-lens stills) reconstruction preparation."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field, asdict

from .colmap_model import ColmapModel
from .intrinsics import ColmapCamera, colmap_camera_from_device
from .pack import PhotoPack, PhotoFrame
from .video_pipeline import (
    ColmapSettings,
    build_colmap_commands,
    run_colmap_sequence,
    _write_run_script,
    _clear_database,
)


@dataclass
class PhotoPrepOptions:
    out_dir: str
    cameras: list[str] = field(default_factory=lambda: ["wide", "ultra"])
    blur_percentile: float = 0.15
    inject_intrinsics: bool = True
    colmap_bin: str = "colmap"
    run_colmap: bool = False
    colmap: ColmapSettings = field(default_factory=ColmapSettings)

    @property
    def matcher(self) -> str:
        return self.colmap.matcher

    @property
    def vocab_tree_path(self) -> str:
        return self.colmap.vocab_tree_path


@dataclass
class PhotoPrepResult:
    out_dir: str
    image_dir: str
    cameras: list[ColmapCamera]
    extracted: dict[str, list[str]]
    rejected: dict[str, list[str]]
    commands: list[list[str]]
    report_path: str


def _blur_key(frame: PhotoFrame) -> float:
    if frame.blur_sharpness is not None:
        return float(frame.blur_sharpness)
    if frame.quality_score is not None:
        return float(frame.quality_score)
    return 1.0


def _select_frames(
    frames: list[PhotoFrame], blur_percentile: float
) -> tuple[list[PhotoFrame], list[PhotoFrame]]:
    if not frames:
        return [], []
    if blur_percentile <= 0 or len(frames) < 3:
        return list(frames), []
    ranked = sorted(frames, key=_blur_key)
    drop_n = int(round(len(ranked) * blur_percentile))
    drop_n = min(max(drop_n, 0), len(ranked) - 1)
    rejected = ranked[:drop_n]
    kept_set = set(id(f) for f in ranked[drop_n:])
    kept = [f for f in frames if id(f) in kept_set]
    return kept, rejected


def prepare_photo_dataset(
    pack: PhotoPack,
    opts: PhotoPrepOptions,
    dry_run: bool = False,
    on_progress=None,
) -> PhotoPrepResult:
    image_root = os.path.join(opts.out_dir, "images")
    if not dry_run:
        os.makedirs(image_root, exist_ok=True)

    cameras: list[ColmapCamera] = []
    extracted: dict[str, list[str]] = {}
    rejected: dict[str, list[str]] = {}

    for cid, cam in enumerate(opts.cameras, start=1):
        frames = pack.frames_for(cam)
        device = pack.representative_device(cam)
        if device is None and frames:
            device = frames[0].device
        if device is None:
            continue
        cameras.append(colmap_camera_from_device(device, cid))

        kept_frames, dropped_frames = _select_frames(frames, opts.blur_percentile)
        cam_dir = os.path.join(image_root, cam)
        if not dry_run:
            os.makedirs(cam_dir, exist_ok=True)

        kept_paths: list[str] = []
        if on_progress:
            on_progress(f"[{cid}/{len(opts.cameras)}] Copying {cam} ({len(kept_frames)} images)...")
        for fr in kept_frames:
            src = os.path.join(pack.root, fr.image_file)
            dst_name = fr.basename or f"{cam}_{fr.capture_id}.jpg"
            dst = os.path.join(cam_dir, dst_name)
            if dry_run:
                kept_paths.append(dst)
                continue
            if not os.path.isfile(src):
                continue
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)
            kept_paths.append(dst)

        extracted[cam] = kept_paths
        rejected[cam] = [os.path.join(pack.root, fr.image_file) for fr in dropped_frames]

        list_path = os.path.join(image_root, f"_list_{cam}.txt")
        if not dry_run:
            with open(list_path, "w", encoding="utf-8") as f:
                for p in kept_paths:
                    f.write(f"{cam}/{os.path.basename(p)}\n")

    database_path = os.path.join(opts.out_dir, "database.db")
    sparse_dir = os.path.join(opts.out_dir, "sparse")
    if not dry_run:
        os.makedirs(sparse_dir, exist_ok=True)

    commands = build_colmap_commands(opts, cameras, image_root, database_path, sparse_dir)

    if not dry_run and opts.inject_intrinsics:
        model = ColmapModel(cameras=cameras)
        model.write(os.path.join(opts.out_dir, "cameras_injected"))

    report = {
        "capture_type": "photo_dual",
        "folder": pack.folder_name,
        "colmap": asdict(opts.colmap),
        "cameras": [c.to_line() for c in cameras],
        "kept_counts": {k: len(v) for k, v in extracted.items()},
        "rejected_counts": {k: len(v) for k, v in rejected.items()},
        "colmap_commands": [" ".join(c) for c in commands],
    }
    report_path = os.path.join(opts.out_dir, "splatking_prep_report.json")
    if not dry_run:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        _write_run_script(opts.out_dir, commands)
        if on_progress:
            on_progress(f"Wrote COLMAP script: {os.path.join(opts.out_dir, 'run_colmap.bat')}")

    if not dry_run and opts.run_colmap:
        if on_progress:
            on_progress("Running COLMAP...")
        _clear_database(database_path)
        run_colmap_sequence(commands, on_log=on_progress)

    return PhotoPrepResult(
        out_dir=opts.out_dir,
        image_dir=image_root,
        cameras=cameras,
        extracted=extracted,
        rejected=rejected,
        commands=commands,
        report_path=report_path,
    )
