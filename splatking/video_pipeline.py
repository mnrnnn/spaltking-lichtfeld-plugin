"""Video (dual-lens) reconstruction preparation.

Extracts frames with ffmpeg (uniform keep%), filters blur, injects dual PINHOLE
intrinsics, and builds/runs COLMAP with shared ColmapSettings.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .colmap_model import ColmapModel, read_model
from .intrinsics import ColmapCamera, colmap_camera_from_device
from .pack import VideoPack, VideoStream
from . import quality


# --------------------------------------------------------------------------- #
# COLMAP settings (single source for Prepare + Run COLMAP section)
# Defaults are documented in README — keep in sync when changing.
# --------------------------------------------------------------------------- #
@dataclass
class ColmapSettings:
    matcher: str = "sequential"  # exhaustive | sequential | vocab_tree
    vocab_tree_path: str = ""
    use_gpu: bool = True
    max_image_size: int = 3200  # 0 = unlimited
    max_num_features: int = 8192
    sequential_overlap: int = 10
    min_num_matches: int = 15


@dataclass
class VideoPrepOptions:
    out_dir: str
    cameras: list[str] = field(default_factory=lambda: ["wide", "ultra"])
    keep_pct: float = 0.10  # fraction of source frames to keep (uniform)
    resize_width: int = 1920  # 0 = native
    blur_percentile: float = 0.15
    blur_abs_threshold: float = 0.0
    inject_intrinsics: bool = True
    colmap_bin: str = "colmap"
    ffmpeg_bin: str = "ffmpeg"
    run_colmap: bool = False
    colmap: ColmapSettings = field(default_factory=ColmapSettings)

    # Back-compat aliases used by older callers
    @property
    def matcher(self) -> str:
        return self.colmap.matcher

    @property
    def vocab_tree_path(self) -> str:
        return self.colmap.vocab_tree_path


@dataclass
class VideoPrepResult:
    out_dir: str
    image_dir: str
    cameras: list[ColmapCamera]
    extracted: dict[str, list[str]]
    rejected: dict[str, list[str]]
    commands: list[list[str]]
    report_path: str


@dataclass
class ExtractEstimate:
    cameras: list[str]
    source_frames: dict[str, int]
    planned_frames: dict[str, int]
    total_planned: int
    video_bytes: dict[str, int]
    total_video_bytes: int
    suggested_matcher: str
    keep_pct: float = 0.1
    step: int = 10
    warnings: list[str] = field(default_factory=list)


def step_from_keep_pct(keep_pct: float) -> int:
    """Uniform stride: keep ~keep_pct of frames (1.0 → every frame)."""
    p = min(max(float(keep_pct), 0.01), 1.0)
    if p >= 0.999:
        return 1
    return max(1, int(round(1.0 / p)))


def select_frame_indices_pct(n: int, keep_pct: float) -> list[int]:
    if n <= 0:
        return []
    step = step_from_keep_pct(keep_pct)
    return list(range(0, n, step))


def estimate_extract(
    pack: VideoPack, cameras: list[str], keep_pct: float
) -> ExtractEstimate:
    source: dict[str, int] = {}
    planned: dict[str, int] = {}
    sizes: dict[str, int] = {}
    warnings: list[str] = []
    step = step_from_keep_pct(keep_pct)
    for cam in cameras:
        stream = pack.stream(cam)
        if stream is None:
            continue
        n = stream.accepted_frame_count or stream.frame_count or len(stream.frame_times)
        source[cam] = n
        planned[cam] = len(select_frame_indices_pct(n, keep_pct))
        vpath = os.path.join(pack.root, stream.raw_video_file)
        try:
            sizes[cam] = os.path.getsize(vpath) if os.path.isfile(vpath) else 0
        except OSError:
            sizes[cam] = 0

    total = sum(planned.values())
    total_bytes = sum(sizes.values())
    if total_bytes >= 500 * 1024 * 1024:
        warnings.append(
            "Dual MOV files are large (~GB). Start around 10% keep + resize 1920."
        )
    if total > 1500:
        suggested = "vocab_tree"
        warnings.append(f"~{total} frames planned: use vocab_tree matcher.")
    elif total > 400:
        suggested = "sequential"
        warnings.append(f"~{total} frames: sequential matcher recommended.")
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
        keep_pct=keep_pct,
        step=step,
        warnings=warnings,
    )


def _ffmpeg_error_message(argv: list[str], completed: subprocess.CompletedProcess) -> str:
    err = (completed.stderr or completed.stdout or b"").decode("utf-8", errors="replace")
    tail = err.strip()[-800:] if err.strip() else "(no stderr)"
    return f"ffmpeg failed (exit {completed.returncode}): {' '.join(argv[:6])}...\n{tail}"


def _ffmpeg_extract(
    ffmpeg_bin: str,
    video_path: str,
    step: int,
    n_frames: int,
    out_dir: str,
    prefix: str,
    resize_width: int = 0,
    dry_run: bool = False,
    on_progress=None,
) -> tuple[list[str], list[str]]:
    """Extract every ``step``-th frame with a short select filter (no O(N) eq chain)."""
    os.makedirs(out_dir, exist_ok=True)
    step = max(1, int(step))
    indices = list(range(0, n_frames, step)) if n_frames > 0 else []
    if not indices and n_frames <= 0:
        # Unknown count: still run mod filter; discover outputs after
        indices = []

    vf = f"select=not(mod(n\\,{step}))"
    if resize_width and resize_width > 0:
        vf += f",scale={resize_width}:-2"

    tmp_pattern = os.path.join(out_dir, f"{prefix}_seq_%06d.jpg")
    argv = [
        ffmpeg_bin, "-y", "-i", video_path,
        "-vf", vf,
        "-fps_mode", "passthrough",
        "-q:v", "2",
        tmp_pattern,
    ]

    if dry_run:
        if not indices and n_frames <= 0:
            # placeholder
            return [], argv
        paths = [os.path.join(out_dir, f"{prefix}_{i:06d}.jpg") for i in indices]
        return paths, argv

    if on_progress:
        on_progress(f"ffmpeg decoding {prefix} (step={step})...")
    completed = subprocess.run(argv, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(_ffmpeg_error_message(argv, completed))

    # Map sequential outputs back to original frame indices.
    output_paths: list[str] = []
    if indices:
        for seq, orig in enumerate(indices, start=1):
            src = os.path.join(out_dir, f"{prefix}_seq_{seq:06d}.jpg")
            if not os.path.isfile(src):
                continue
            dst = os.path.join(out_dir, f"{prefix}_{orig:06d}.jpg")
            os.replace(src, dst)
            output_paths.append(dst)
    else:
        # Fall back: rename whatever seq files appeared
        seq = 1
        while True:
            src = os.path.join(out_dir, f"{prefix}_seq_{seq:06d}.jpg")
            if not os.path.isfile(src):
                break
            orig = (seq - 1) * step
            dst = os.path.join(out_dir, f"{prefix}_{orig:06d}.jpg")
            os.replace(src, dst)
            output_paths.append(dst)
            seq += 1

    if on_progress:
        on_progress(f"Extracted {len(output_paths)} frames from {prefix}")
    return output_paths, argv


def build_colmap_commands(
    opts,  # VideoPrepOptions | PhotoPrepOptions | object with .colmap / .cameras / ...
    cameras: list[ColmapCamera],
    image_root: str,
    database_path: str,
    sparse_dir: str,
) -> list[list[str]]:
    """Build COLMAP argv using ColmapSettings (GPU/SIFT/matcher/mapper)."""
    cm: ColmapSettings = getattr(opts, "colmap", None) or ColmapSettings()
    colmap_bin = getattr(opts, "colmap_bin", "colmap")
    inject = getattr(opts, "inject_intrinsics", True)
    cam_names = getattr(opts, "cameras", [c.source_camera for c in cameras])

    cmds: list[list[str]] = []
    cam_by_src = {c.source_camera: c for c in cameras}

    for cam in cam_names:
        fe = [
            colmap_bin, "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_root,
            "--image_list_path", os.path.join(image_root, f"_list_{cam}.txt"),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "PINHOLE",
            # COLMAP ≥3.13 / 4.x: FeatureExtraction.* (not SiftExtraction.use_gpu)
            "--FeatureExtraction.use_gpu", "1" if cm.use_gpu else "0",
            "--SiftExtraction.max_num_features", str(int(cm.max_num_features)),
        ]
        if cm.max_image_size and cm.max_image_size > 0:
            fe += ["--FeatureExtraction.max_image_size", str(int(cm.max_image_size))]
        cc = cam_by_src.get(cam)
        if inject and cc is not None:
            fe += ["--ImageReader.camera_params", ",".join(f"{p:.9g}" for p in cc.params)]
        cmds.append(fe)

    gpu = "1" if cm.use_gpu else "0"
    if cm.matcher == "exhaustive":
        cmds.append([colmap_bin, "exhaustive_matcher", "--database_path", database_path,
                      "--FeatureMatching.use_gpu", gpu])
    elif cm.matcher == "vocab_tree":
        m = [colmap_bin, "vocab_tree_matcher", "--database_path", database_path,
             "--FeatureMatching.use_gpu", gpu]
        if cm.vocab_tree_path:
            m += ["--VocabTreeMatching.vocab_tree_path", cm.vocab_tree_path]
        cmds.append(m)
    else:
        cmds.append([
            colmap_bin, "sequential_matcher",
            "--database_path", database_path,
            "--FeatureMatching.use_gpu", gpu,
            "--SequentialMatching.overlap", str(int(cm.sequential_overlap)),
        ])

    cmds.append([
        colmap_bin, "mapper",
        "--database_path", database_path,
        "--image_path", image_root,
        "--output_path", sparse_dir,
        "--Mapper.min_num_matches", str(int(cm.min_num_matches)),
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
    step = step_from_keep_pct(opts.keep_pct)

    cameras: list[ColmapCamera] = []
    extracted: dict[str, list[str]] = {}
    rejected: dict[str, list[str]] = {}
    n_cams = len(opts.cameras)

    for cid, cam in enumerate(opts.cameras, start=1):
        stream = pack.stream(cam)
        if stream is None:
            continue

        out_h = None
        if opts.resize_width:
            ar = stream.device.height / stream.device.width if stream.device.width else 0
            out_h = int(round(opts.resize_width * ar)) if ar else None
        cameras.append(
            colmap_camera_from_device(
                stream.device, cid,
                out_width=(opts.resize_width or None),
                out_height=out_h,
            )
        )

        n = stream.accepted_frame_count or stream.frame_count or len(stream.frame_times)
        cam_dir = os.path.join(image_root, cam)
        video_path = os.path.join(pack.root, stream.raw_video_file)
        if on_progress:
            on_progress(f"[{cid}/{n_cams}] Extracting {cam} (keep={opts.keep_pct:.0%}, step={step})...")
        paths, _ = _ffmpeg_extract(
            opts.ffmpeg_bin, video_path, step, n, cam_dir, cam,
            resize_width=opts.resize_width, dry_run=dry_run, on_progress=on_progress,
        )

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

        list_path = os.path.join(image_root, f"_list_{cam}.txt")
        if not dry_run:
            with open(list_path, "w", encoding="utf-8") as f:
                for p in extracted[cam]:
                    f.write(f"{cam}/{os.path.basename(p)}\n")

    database_path = os.path.join(opts.out_dir, "database.db")
    sparse_dir = os.path.join(opts.out_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    commands = build_colmap_commands(opts, cameras, image_root, database_path, sparse_dir)

    if not dry_run and opts.inject_intrinsics:
        model = ColmapModel(cameras=cameras)
        model.write(os.path.join(opts.out_dir, "cameras_injected"))

    report = {
        "capture_type": "video_dual",
        "folder": pack.folder_name,
        "keep_pct": opts.keep_pct,
        "step": step,
        "resize_width": opts.resize_width,
        "colmap": asdict(opts.colmap),
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
        _clear_database(database_path)
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


def _clear_database(database_path: str) -> None:
    for p in (database_path, database_path + "-journal"):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass


def run_colmap_on_prep(
    prep_dir: str,
    colmap_bin: str,
    settings: ColmapSettings,
    cameras: Optional[list[str]] = None,
    inject_intrinsics: bool = True,
    on_progress=None,
) -> list[list[str]]:
    """Rebuild and run COLMAP for an existing prep directory."""
    image_root = os.path.join(prep_dir, "images")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"No images/ in prep dir: {prep_dir}")

    cam_names = cameras or [
        d for d in ("wide", "ultra") if os.path.isdir(os.path.join(image_root, d))
    ]
    if not cam_names:
        cam_names = [
            name for name in os.listdir(image_root)
            if os.path.isdir(os.path.join(image_root, name)) and not name.startswith("_")
        ]

    colmap_cams: list[ColmapCamera] = []
    inj = os.path.join(prep_dir, "cameras_injected")
    if os.path.isdir(inj):
        try:
            model = read_model(inj)
            colmap_cams = list(model.cameras)
            for i, name in enumerate(cam_names):
                if i < len(colmap_cams) and not getattr(colmap_cams[i], "source_camera", ""):
                    colmap_cams[i].source_camera = name
        except Exception:
            colmap_cams = []

    class _Opts:
        pass

    o = _Opts()
    o.colmap = settings
    o.colmap_bin = colmap_bin
    o.inject_intrinsics = inject_intrinsics and bool(colmap_cams)
    o.cameras = cam_names

    database_path = os.path.join(prep_dir, "database.db")
    sparse_dir = os.path.join(prep_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    _clear_database(database_path)

    commands = build_colmap_commands(o, colmap_cams, image_root, database_path, sparse_dir)
    _write_run_script(prep_dir, commands)
    report_path = os.path.join(prep_dir, "splatking_prep_report.json")
    if os.path.isfile(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["colmap"] = asdict(settings)
            data["colmap_commands"] = [" ".join(c) for c in commands]
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except (OSError, ValueError):
            pass

    run_colmap_sequence(commands, on_log=on_progress)
    return commands


def run_colmap_sequence(commands: list[list[str]], on_log=None) -> None:
    n = len(commands)
    for i, cmd in enumerate(commands, start=1):
        if on_log:
            on_log(f"COLMAP [{i}/{n}] {' '.join(cmd[:4])}...")
        completed = subprocess.run(cmd, capture_output=True)
        if completed.returncode != 0:
            err = (completed.stderr or completed.stdout or b"").decode("utf-8", errors="replace")
            raise RuntimeError(
                f"COLMAP step failed (exit {completed.returncode}): {' '.join(cmd)}\n{err.strip()[-800:]}"
            )


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


# Deprecated name kept for imports
DENSITY_PRESETS: list[tuple[str, int, int, int]] = []
