"""Video (dual-lens) reconstruction preparation.

Turns a ``video_dual`` splatpack into a COLMAP-ready dataset that respects the
things the GUI plugin cannot:

* extracts temporally-synchronized wide + ultra frames with ffmpeg,
* filters blurry frames with a Laplacian-variance metric,
* injects known per-lens PINHOLE intrinsics (two distinct cameras),
* orchestrates COLMAP with a selectable matcher including vocab-tree for
  large image counts.

Everything here is plain Python + subprocess; no LichtFeld import.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

from .colmap_model import ColmapModel
from .intrinsics import ColmapCamera, colmap_camera_from_device
from .pack import VideoPack, VideoStream
from . import quality


@dataclass
class VideoPrepOptions:
    out_dir: str
    cameras: list[str] = field(default_factory=lambda: ["wide", "ultra"])
    stride: int = 15                   # keep every Nth frame per lens (safe default)
    max_frames_per_lens: int = 200     # 0 = unlimited; cap protects first runs
    resize_width: int = 1920           # 0 = native; 1920 is VRAM-friendly
    # Quality filtering.
    blur_percentile: float = 0.15      # drop the blurriest 15%
    blur_abs_threshold: float = 0.0    # 0 = disabled
    # COLMAP orchestration.
    matcher: str = "sequential"        # exhaustive | sequential | vocab_tree
    vocab_tree_path: str = ""          # required for matcher == vocab_tree
    inject_intrinsics: bool = True     # write known per-lens PINHOLE params
    colmap_bin: str = "colmap"
    ffmpeg_bin: str = "ffmpeg"
    run_colmap: bool = False           # False = prepare dataset + script only


@dataclass
class VideoPrepResult:
    out_dir: str
    image_dir: str
    cameras: list[ColmapCamera]
    extracted: dict[str, list[str]]           # camera -> kept image paths
    rejected: dict[str, list[str]]            # camera -> dropped (blurry) paths
    commands: list[list[str]]                 # COLMAP argv sequence
    report_path: str


@dataclass
class ExtractEstimate:
    """Pre-flight summary shown in the UI before decoding large MOVs."""

    cameras: list[str]
    source_frames: dict[str, int]
    planned_frames: dict[str, int]
    total_planned: int
    video_bytes: dict[str, int]
    total_video_bytes: int
    suggested_matcher: str
    warnings: list[str] = field(default_factory=list)


# Density presets: (label, stride, max_frames, resize_width)
DENSITY_PRESETS: list[tuple[str, int, int, int]] = [
    ("Quick (first try)", 30, 80, 1280),
    ("Balanced", 15, 200, 1920),
    ("Dense", 5, 500, 1920),
    ("Full (slow)", 1, 0, 0),
]


def select_frame_indices(stream: VideoStream, stride: int, max_frames: int) -> list[int]:
    n = stream.accepted_frame_count or stream.frame_count or len(stream.frame_times)
    if n <= 0:
        return []
    stride = max(1, stride)
    idx = list(range(0, n, stride))
    if max_frames and len(idx) > max_frames:
        # Evenly subsample to the cap while preserving coverage.
        step = len(idx) / float(max_frames)
        idx = [idx[int(i * step)] for i in range(max_frames)]
    return idx


def estimate_extract(pack: VideoPack, cameras: list[str], stride: int, max_frames: int) -> ExtractEstimate:
    source: dict[str, int] = {}
    planned: dict[str, int] = {}
    sizes: dict[str, int] = {}
    warnings: list[str] = []
    for cam in cameras:
        stream = pack.stream(cam)
        if stream is None:
            continue
        n = stream.accepted_frame_count or stream.frame_count or len(stream.frame_times)
        source[cam] = n
        planned[cam] = len(select_frame_indices(stream, stride, max_frames))
        vpath = os.path.join(pack.root, stream.raw_video_file)
        try:
            sizes[cam] = os.path.getsize(vpath) if os.path.isfile(vpath) else 0
        except OSError:
            sizes[cam] = 0

    total = sum(planned.values())
    total_bytes = sum(sizes.values())
    if total_bytes >= 500 * 1024 * 1024:
        warnings.append(
            "Dual MOV files are large (~GB). ffmpeg must scan each video once; "
            "start with Quick/Balanced, not Full."
        )
    if total > 1500:
        suggested = "vocab_tree"
        warnings.append(f"~{total} frames planned: use vocab_tree matcher (exhaustive will explode).")
    elif total > 400:
        suggested = "sequential"
        warnings.append(f"~{total} frames: sequential matcher is recommended; avoid exhaustive.")
    else:
        suggested = "exhaustive" if total <= 120 else "sequential"

    if any(v == 0 for v in sizes.values()):
        warnings.append("One or more video files are missing from the pack folder.")

    return ExtractEstimate(
        cameras=list(planned.keys()),
        source_frames=source,
        planned_frames=planned,
        total_planned=total,
        video_bytes=sizes,
        total_video_bytes=total_bytes,
        suggested_matcher=suggested,
        warnings=warnings,
    )


def _ffmpeg_extract(
    ffmpeg_bin: str,
    video_path: str,
    indices: list[int],
    out_dir: str,
    prefix: str,
    resize_width: int = 0,
    dry_run: bool = False,
    on_progress=None,
) -> tuple[list[str], list[str]]:
    """Extract the given frame indices; return (output_paths, ffmpeg_argv).

    Uses ffmpeg's ``select`` filter to grab specific frames in a single decode
    pass, then renames the sequential outputs to their original frame index so
    wide/ultra frames with the same index stay paired.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not indices:
        return [], []

    select_expr = "+".join(f"eq(n\\,{i})" for i in indices)
    vf = f"select='{select_expr}'"
    if resize_width and resize_width > 0:
        vf += f",scale={resize_width}:-2"

    tmp_pattern = os.path.join(out_dir, f"{prefix}_seq_%06d.jpg")
    argv = [
        ffmpeg_bin, "-y", "-i", video_path,
        "-vf", vf,
        "-vsync", "0",
        "-q:v", "2",
        tmp_pattern,
    ]

    output_paths: list[str] = []
    if dry_run:
        for i in indices:
            output_paths.append(os.path.join(out_dir, f"{prefix}_{i:06d}.jpg"))
        return output_paths, argv

    if on_progress:
        on_progress(f"ffmpeg decoding {prefix} ({len(indices)} frames)...")
    subprocess.run(argv, check=True, capture_output=True)

    # Rename sequential outputs to original-index filenames.
    for seq, orig in enumerate(indices, start=1):
        src = os.path.join(out_dir, f"{prefix}_seq_{seq:06d}.jpg")
        if not os.path.isfile(src):
            continue
        dst = os.path.join(out_dir, f"{prefix}_{orig:06d}.jpg")
        os.replace(src, dst)
        output_paths.append(dst)
    if on_progress:
        on_progress(f"Extracted {len(output_paths)} frames from {prefix}")
    return output_paths, argv


def build_colmap_commands(
    opts: VideoPrepOptions,
    cameras: list[ColmapCamera],
    image_root: str,
    database_path: str,
    sparse_dir: str,
) -> list[list[str]]:
    """Build the COLMAP argv sequence for the video path.

    Feature extraction runs once per lens subfolder so each lens becomes its own
    camera with injected PINHOLE params (the CLI-only per-camera path). Matching
    supports exhaustive / sequential / vocab_tree; vocab_tree is the answer to
    the O(N^2) wall at thousands of images.
    """
    cmds: list[list[str]] = []
    cam_by_src = {c.source_camera: c for c in cameras}

    for cam in opts.cameras:
        folder = os.path.join(image_root, cam)
        fe = [
            opts.colmap_bin, "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_root,
            "--image_list_path", os.path.join(image_root, f"_list_{cam}.txt"),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "PINHOLE",
        ]
        cc = cam_by_src.get(cam)
        if opts.inject_intrinsics and cc is not None:
            fe += ["--ImageReader.camera_params", ",".join(f"{p:.9g}" for p in cc.params)]
        cmds.append(fe)

    if opts.matcher == "exhaustive":
        cmds.append([opts.colmap_bin, "exhaustive_matcher", "--database_path", database_path])
    elif opts.matcher == "vocab_tree":
        m = [opts.colmap_bin, "vocab_tree_matcher", "--database_path", database_path]
        if opts.vocab_tree_path:
            m += ["--VocabTreeMatching.vocab_tree_path", opts.vocab_tree_path]
        cmds.append(m)
    else:  # sequential
        cmds.append([opts.colmap_bin, "sequential_matcher", "--database_path", database_path])

    cmds.append([
        opts.colmap_bin, "mapper",
        "--database_path", database_path,
        "--image_path", image_root,
        "--output_path", sparse_dir,
    ])
    return cmds


def prepare_video_dataset(
    pack: VideoPack,
    opts: VideoPrepOptions,
    dry_run: bool = False,
    on_progress=None,
) -> VideoPrepResult:
    image_root = os.path.join(opts.out_dir, "images")
    os.makedirs(image_root, exist_ok=True)

    cameras: list[ColmapCamera] = []
    extracted: dict[str, list[str]] = {}
    rejected: dict[str, list[str]] = {}

    for cid, cam in enumerate(opts.cameras, start=1):
        stream = pack.stream(cam)
        if stream is None:
            continue

        out_w = opts.resize_width or stream.device.width
        out_h = None
        if opts.resize_width:
            # keep aspect ratio; ffmpeg scale=-2 handles the actual height
            ar = stream.device.height / stream.device.width if stream.device.width else 0
            out_h = int(round(opts.resize_width * ar)) if ar else None
        cameras.append(
            colmap_camera_from_device(
                stream.device, cid,
                out_width=(opts.resize_width or None),
                out_height=out_h,
            )
        )

        indices = select_frame_indices(stream, opts.stride, opts.max_frames_per_lens)
        cam_dir = os.path.join(image_root, cam)
        video_path = os.path.join(pack.root, stream.raw_video_file)
        if on_progress:
            on_progress(f"[{cid}/{len(opts.cameras)}] Extracting {cam}...")
        paths, _ = _ffmpeg_extract(
            opts.ffmpeg_bin, video_path, indices, cam_dir, cam,
            resize_width=opts.resize_width, dry_run=dry_run, on_progress=on_progress,
        )

        # Blur / quality filtering.
        if not dry_run and (opts.blur_percentile > 0 or opts.blur_abs_threshold > 0):
            if on_progress:
                on_progress(f"Scoring sharpness for {cam} ({len(paths)} frames)...")
            scores = quality.filter_frames(
                paths,
                abs_threshold=opts.blur_abs_threshold or None,
                percentile=opts.blur_percentile or None,
            )
            kept, dropped = [], []
            for s in scores:
                if s.kept:
                    kept.append(s.path)
                else:
                    dropped.append(s.path)
                    try:
                        os.remove(s.path)
                    except OSError:
                        pass
            extracted[cam] = kept
            rejected[cam] = dropped
        else:
            extracted[cam] = paths
            rejected[cam] = []

        # Per-lens image list (paths relative to image_root) for COLMAP.
        list_path = os.path.join(image_root, f"_list_{cam}.txt")
        if not dry_run:
            with open(list_path, "w", encoding="utf-8") as f:
                for p in extracted[cam]:
                    f.write(f"{cam}/{os.path.basename(p)}\n")

    database_path = os.path.join(opts.out_dir, "database.db")
    sparse_dir = os.path.join(opts.out_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    commands = build_colmap_commands(opts, cameras, image_root, database_path, sparse_dir)

    # Always write the injected 2-camera model so the intrinsics survive even if
    # the user runs COLMAP separately or triangulates with known poses.
    if not dry_run and opts.inject_intrinsics:
        model = ColmapModel(cameras=cameras)
        model.write(os.path.join(opts.out_dir, "cameras_injected"))

    report = {
        "capture_type": "video_dual",
        "folder": pack.folder_name,
        "options": opts.__dict__,
        "cameras": [c.to_line() for c in cameras],
        "kept_counts": {k: len(v) for k, v in extracted.items()},
        "rejected_counts": {k: len(v) for k, v in rejected.items()},
        "colmap_commands": [" ".join(c) for c in commands],
        "quality_scoring_available": quality.scoring_available(),
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
        run_colmap_sequence(commands, on_log=on_progress)

    return VideoPrepResult(
        out_dir=opts.out_dir,
        image_dir=image_root,
        cameras=cameras,
        extracted=extracted,
        rejected=rejected,
        commands=commands,
        report_path=report_path,
    )


def run_colmap_sequence(commands: list[list[str]], on_log=None) -> None:
    for cmd in commands:
        if on_log:
            on_log(" ".join(cmd))
        subprocess.run(cmd, check=True)


def _write_run_script(out_dir: str, commands: list[list[str]]) -> None:
    sh = os.path.join(out_dir, "run_colmap.sh")
    bat = os.path.join(out_dir, "run_colmap.bat")
    with open(sh, "w", encoding="utf-8", newline="\n") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        for c in commands:
            f.write(" ".join(_sh_quote(x) for x in c) + "\n")
    with open(bat, "w", encoding="utf-8") as f:
        f.write("@echo off\r\n")
        for c in commands:
            f.write(" ".join(_bat_quote(x) for x in c) + "\r\n")


def _sh_quote(s: str) -> str:
    if any(ch in s for ch in " \t\"'\\"):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


def _bat_quote(s: str) -> str:
    return f'"{s}"' if " " in s else s
