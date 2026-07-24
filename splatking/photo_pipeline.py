"""Photo (dual-lens stills) reconstruction preparation."""

from __future__ import annotations

import json
import os
import re
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
    write_colmap_script: bool = False  # opt-in: run_colmap.bat / .sh
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


def _safe_capture_id(cid: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(cid).strip()) or "frame"
    return s[:80]


def _select_pairs(
    by_cam: dict[str, dict[str, PhotoFrame]],
    cameras: list[str],
    blur_percentile: float,
) -> tuple[list[str], list[str]]:
    """Return (kept_capture_ids, dropped_capture_ids) with pair-aware blur."""
    if len(cameras) < 2:
        cam = cameras[0]
        frames = list(by_cam.get(cam, {}).values())
        if blur_percentile <= 0 or len(frames) < 3:
            return [f.capture_id for f in frames], []
        ranked = sorted(frames, key=_blur_key)
        drop_n = int(round(len(ranked) * blur_percentile))
        drop_n = min(max(drop_n, 0), len(ranked) - 1)
        dropped = {f.capture_id for f in ranked[:drop_n]}
        kept = [f.capture_id for f in frames if f.capture_id not in dropped]
        return kept, list(dropped)

    shared = set(by_cam[cameras[0]].keys())
    for cam in cameras[1:]:
        shared &= set(by_cam.get(cam, {}).keys())
    shared_ids = sorted(shared)
    if not shared_ids:
        return [], []
    if blur_percentile <= 0 or len(shared_ids) < 3:
        return shared_ids, []

    # Pair score = min sharpness across lenses
    scored: list[tuple[float, str]] = []
    for cid in shared_ids:
        vals = [_blur_key(by_cam[cam][cid]) for cam in cameras]
        scored.append((min(vals), cid))
    ranked = sorted(scored, key=lambda t: t[0])
    drop_n = int(round(len(ranked) * blur_percentile))
    drop_n = min(max(drop_n, 0), len(ranked) - 1)
    dropped = {cid for _, cid in ranked[:drop_n]}
    kept = [cid for cid in shared_ids if cid not in dropped]
    return kept, sorted(dropped)


def prepare_photo_dataset(
    pack: PhotoPack,
    opts: PhotoPrepOptions,
    dry_run: bool = False,
    on_progress=None,
) -> PhotoPrepResult:
    from .rig import write_cross_lens_pairs_file, run_dual_colmap

    image_root = os.path.join(opts.out_dir, "images")
    if not dry_run:
        os.makedirs(image_root, exist_ok=True)

    cameras: list[ColmapCamera] = []
    extracted: dict[str, list[str]] = {}
    rejected: dict[str, list[str]] = {}

    # Index frames by capture_id per camera
    by_cam: dict[str, dict[str, PhotoFrame]] = {}
    for cam in opts.cameras:
        frames = pack.frames_for(cam)
        by_cam[cam] = {}
        for fr in frames:
            cid = fr.capture_id or fr.basename or os.path.basename(fr.image_file)
            by_cam[cam][str(cid)] = fr

        device = pack.representative_device(cam)
        if device is None and frames:
            device = frames[0].device
        if device is None:
            continue
        cameras.append(colmap_camera_from_device(device, len(cameras) + 1))

    kept_ids, dropped_ids = _select_pairs(by_cam, opts.cameras, opts.blur_percentile)
    dual = bool(opts.colmap.dual_mode) and len(opts.cameras) >= 2

    for cid_i, cam in enumerate(opts.cameras, start=1):
        cam_dir = os.path.join(image_root, cam)
        if not dry_run:
            os.makedirs(cam_dir, exist_ok=True)
        kept_paths: list[str] = []
        if on_progress:
            on_progress(
                f"[{cid_i}/{len(opts.cameras)}] Copying {cam} ({len(kept_ids)} images)..."
            )
        for cap_id in kept_ids:
            fr = by_cam.get(cam, {}).get(cap_id)
            if fr is None:
                continue
            src = os.path.join(pack.root, fr.image_file)
            if dual:
                dst_name = f"frame_{_safe_capture_id(cap_id)}.jpg"
            else:
                dst_name = fr.basename or f"{cam}_{_safe_capture_id(cap_id)}.jpg"
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
        rejected[cam] = [
            os.path.join(pack.root, by_cam[cam][d].image_file)
            for d in dropped_ids
            if d in by_cam.get(cam, {})
        ]

        list_path = os.path.join(image_root, f"_list_{cam}.txt")
        if not dry_run:
            with open(list_path, "w", encoding="utf-8") as f:
                for p in kept_paths:
                    f.write(f"{cam}/{os.path.basename(p)}\n")

    if dual and not dry_run:
        shared = [os.path.basename(p) for p in extracted.get(opts.cameras[0], [])]
        write_cross_lens_pairs_file(
            os.path.join(image_root, "_cross_lens_pairs.txt"), shared
        )

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
        "sync_method": "capture_id",
        "dual_pairs_kept": len(kept_ids),
        "dual_pairs_dropped": len(dropped_ids),
    }
    report_path = os.path.join(opts.out_dir, "splatking_prep_report.json")
    if not dry_run:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        if opts.write_colmap_script:
            _write_run_script(opts.out_dir, commands)
            if on_progress:
                on_progress(
                    f"Wrote COLMAP script: {os.path.join(opts.out_dir, 'run_colmap.bat')}"
                )

    if not dry_run and opts.run_colmap:
        if on_progress:
            on_progress("Running COLMAP...")
        _clear_database(database_path)
        dual_report = run_dual_colmap(
            colmap_bin=opts.colmap_bin,
            image_root=image_root,
            database_path=database_path,
            sparse_dir=sparse_dir,
            cameras=list(opts.cameras),
            settings=opts.colmap,
            inject_cameras=cameras,
            inject_intrinsics=opts.inject_intrinsics,
            on_log=on_progress,
            run_sequence=run_colmap_sequence,
        )
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["dual_colmap"] = dual_report
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except (OSError, ValueError):
            pass

    return PhotoPrepResult(
        out_dir=opts.out_dir,
        image_dir=image_root,
        cameras=cameras,
        extracted=extracted,
        rejected=rejected,
        commands=commands,
        report_path=report_path,
    )
