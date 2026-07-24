"""Video (dual-lens) reconstruction preparation.

Extracts frames with ffmpeg (uniform keep%), filters blur, injects dual PINHOLE
intrinsics, and builds/runs COLMAP with shared ColmapSettings.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

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
    # Dual-lens integration (video_dual / photo_dual with wide+ultra)
    dual_mode: bool = True
    base_lens: str = "ultra"  # reconstruction skeleton
    dual_method: str = "auto"  # auto | registrator | rig | wide_only
    registrator_min_ratio: float = 0.50


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
    write_colmap_script: bool = False  # opt-in: run_colmap.bat / .sh
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
    indices: Optional[list[int]] = None,
    name_for_index: Optional[Callable[[int, int], str]] = None,
) -> tuple[list[str], list[str]]:
    """Extract frames. With ``indices``, select those frame numbers; else every ``step``-th."""
    os.makedirs(out_dir, exist_ok=True)
    step = max(1, int(step))
    if indices is None:
        indices = list(range(0, n_frames, step)) if n_frames > 0 else []

    if indices:
        # Exact indices (dual pairing). Chunk if huge to keep argv reasonable.
        eqs = "+".join(f"eq(n\\,{i})" for i in indices)
        vf = f"select='{eqs}'"
    else:
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

    def _dst_name(seq: int, orig: int) -> str:
        if name_for_index is not None:
            return name_for_index(seq - 1, orig)
        return f"{prefix}_{orig:06d}.jpg"

    if dry_run:
        if not indices and n_frames <= 0:
            return [], argv
        paths = [
            os.path.join(out_dir, _dst_name(seq, orig))
            for seq, orig in enumerate(indices, start=1)
        ]
        return paths, argv

    if on_progress:
        tag = f"indices={len(indices)}" if indices else f"step={step}"
        on_progress(f"ffmpeg decoding {prefix} ({tag})...")
    completed = subprocess.run(argv, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(_ffmpeg_error_message(argv, completed))

    output_paths: list[str] = []
    if indices:
        for seq, orig in enumerate(indices, start=1):
            src = os.path.join(out_dir, f"{prefix}_seq_{seq:06d}.jpg")
            if not os.path.isfile(src):
                continue
            dst = os.path.join(out_dir, _dst_name(seq, orig))
            os.replace(src, dst)
            output_paths.append(dst)
    else:
        seq = 1
        while True:
            src = os.path.join(out_dir, f"{prefix}_seq_{seq:06d}.jpg")
            if not os.path.isfile(src):
                break
            orig = (seq - 1) * step
            dst = os.path.join(out_dir, _dst_name(seq, orig))
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
    """Build COLMAP argv using ColmapSettings (GPU/SIFT/matcher/mapper).

    For dual_mode + 2 cameras this returns a *preview* of FE+match+cross+mapper A
    (Path1 start). The live runner is ``run_dual_colmap`` which adds fallbacks.
    """
    from .rig import (
        build_cross_lens_importer_cmd,
        build_feature_match_commands,
        shared_basenames,
        write_cross_lens_pairs_file,
    )

    cm: ColmapSettings = getattr(opts, "colmap", None) or ColmapSettings()
    colmap_bin = getattr(opts, "colmap_bin", "colmap")
    cam_names = getattr(opts, "cameras", [c.source_camera for c in cameras])
    dual = bool(getattr(cm, "dual_mode", True)) and len(cam_names) >= 2

    cmds = build_feature_match_commands(
        opts, cameras, image_root, database_path, dual=dual
    )
    if dual:
        shared = shared_basenames(image_root, list(cam_names))
        pairs_path = os.path.join(image_root, "_cross_lens_pairs.txt")
        # Preview: write pairs if images already exist; else empty placeholder path
        if shared:
            write_cross_lens_pairs_file(pairs_path, shared)
        cmds.append(
            build_cross_lens_importer_cmd(colmap_bin, database_path, pairs_path, cm)
        )
        base = getattr(cm, "base_lens", "ultra")
        if base not in cam_names:
            base = cam_names[0]
        cmds.append(
            [
                colmap_bin,
                "mapper",
                "--database_path",
                database_path,
                "--image_path",
                image_root,
                "--output_path",
                os.path.join(sparse_dir, f"{base}_base"),
                "--Mapper.min_num_matches",
                str(int(cm.min_num_matches)),
                "--Mapper.image_list_path",
                os.path.join(image_root, f"_list_{base}.txt"),
            ]
        )
    else:
        cmds.append(
            [
                colmap_bin,
                "mapper",
                "--database_path",
                database_path,
                "--image_path",
                image_root,
                "--output_path",
                sparse_dir,
                "--Mapper.min_num_matches",
                str(int(cm.min_num_matches)),
            ]
        )
    return cmds


def prepare_video_dataset(
    pack: VideoPack,
    opts: VideoPrepOptions,
    dry_run: bool = False,
    on_progress=None,
) -> VideoPrepResult:
    from .rig import (
        frame_basename,
        pair_by_frame_times,
        select_pairs_keep_pct,
        write_cross_lens_pairs_file,
    )

    image_root = os.path.join(opts.out_dir, "images")
    os.makedirs(image_root, exist_ok=True)
    step = step_from_keep_pct(opts.keep_pct)

    cameras: list[ColmapCamera] = []
    extracted: dict[str, list[str]] = {}
    rejected: dict[str, list[str]] = {}
    n_cams = len(opts.cameras)
    sync_meta: dict[str, Any] = {}

    dual = (
        bool(opts.colmap.dual_mode)
        and set(opts.cameras) >= {"wide", "ultra"}
        and pack.stream("wide") is not None
        and pack.stream("ultra") is not None
    )

    # Build cameras list in opts.cameras order
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
                stream.device,
                cid,
                out_width=(opts.resize_width or None),
                out_height=out_h,
            )
        )

    if dual:
        wide_s = pack.stream("wide")
        ultra_s = pack.stream("ultra")
        assert wide_s is not None and ultra_s is not None
        pairs = pair_by_frame_times(wide_s.frame_times, ultra_s.frame_times)
        if not pairs:
            # Fallback: index-aligned if same length
            n = min(
                wide_s.accepted_frame_count or wide_s.frame_count or 0,
                ultra_s.accepted_frame_count or ultra_s.frame_count or 0,
            )
            from .rig import TimePair

            pairs = [
                TimePair(pair_id=i, wide_idx=i, ultra_idx=i, dt_s=0.0) for i in range(n)
            ]
            sync_meta["sync_method"] = "index_aligned_fallback"
        else:
            sync_meta["sync_method"] = "frame_times"
        kept_pairs = select_pairs_keep_pct(pairs, opts.keep_pct)
        sync_meta["dual_pairs_total"] = len(pairs)
        sync_meta["dual_pairs_kept"] = len(kept_pairs)
        sync_meta["max_pair_dt_ms"] = (
            max((p.dt_s for p in kept_pairs), default=0.0) * 1000.0
        )

        # Extract per lens with shared basenames frame_XXXXXX.jpg
        for cam, idx_attr in (("wide", "wide_idx"), ("ultra", "ultra_idx")):
            stream = pack.stream(cam)
            assert stream is not None
            cam_dir = os.path.join(image_root, cam)
            video_path = os.path.join(pack.root, stream.raw_video_file)
            indices = [getattr(p, idx_attr) for p in kept_pairs]
            pair_ids = [p.pair_id for p in kept_pairs]
            id_by_pos = {pos: pid for pos, pid in enumerate(pair_ids)}

            def _name(pos: int, _orig: int, _m=id_by_pos) -> str:
                return frame_basename(_m[pos])

            if on_progress:
                on_progress(
                    f"Extracting {cam} paired ({len(indices)} frames, keep={opts.keep_pct:.0%})..."
                )
            paths, _ = _ffmpeg_extract(
                opts.ffmpeg_bin,
                video_path,
                step,
                0,
                cam_dir,
                cam,
                resize_width=opts.resize_width,
                dry_run=dry_run,
                on_progress=on_progress,
                indices=indices,
                name_for_index=_name,
            )
            extracted[cam] = paths
            rejected[cam] = []

        # Pair-aware blur: drop both sides if either is blurry
        if not dry_run and (opts.blur_percentile > 0 or opts.blur_abs_threshold > 0):
            if on_progress:
                on_progress("Pair-aware blur filter...")
            by_base = {
                cam: {os.path.basename(p): p for p in extracted[cam]}
                for cam in ("wide", "ultra")
            }
            bases = sorted(set(by_base["wide"]) & set(by_base["ultra"]))
            # Score min sharpness per pair
            pair_paths = [(by_base["wide"][b], by_base["ultra"][b]) for b in bases]
            # Use quality on all paths then intersect kept
            all_paths = [p for pair in pair_paths for p in pair]
            scores = {
                s.path: s
                for s in quality.filter_frames(
                    all_paths,
                    abs_threshold=opts.blur_abs_threshold or None,
                    percentile=None,  # apply percentile on pair mins
                )
            }
            # Build pair score = min of both
            pair_scores = []
            for w, u in pair_paths:
                sw = scores.get(w)
                su = scores.get(u)
                val = min(
                    float(getattr(sw, "sharpness", 0.0) or 0.0),
                    float(getattr(su, "sharpness", 0.0) or 0.0),
                )
                kept_flag = True
                if opts.blur_abs_threshold > 0:
                    kept_flag = val >= opts.blur_abs_threshold
                pair_scores.append((val, w, u, kept_flag))
            if opts.blur_percentile > 0 and len(pair_scores) >= 3:
                ranked = sorted(pair_scores, key=lambda t: t[0])
                drop_n = int(round(len(ranked) * opts.blur_percentile))
                drop_n = min(max(drop_n, 0), len(ranked) - 1)
                drop_bases = {os.path.basename(t[1]) for t in ranked[:drop_n]}
                pair_scores = [
                    (v, w, u, False if os.path.basename(w) in drop_bases else k)
                    for v, w, u, k in pair_scores
                ]
            kept_w, kept_u, drop_w, drop_u = [], [], [], []
            for val, w, u, keep in pair_scores:
                if keep:
                    kept_w.append(w)
                    kept_u.append(u)
                else:
                    drop_w.append(w)
                    drop_u.append(u)
                    for p in (w, u):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            extracted["wide"] = kept_w
            extracted["ultra"] = kept_u
            rejected["wide"] = drop_w
            rejected["ultra"] = drop_u

        for cam in ("wide", "ultra"):
            list_path = os.path.join(image_root, f"_list_{cam}.txt")
            if not dry_run:
                with open(list_path, "w", encoding="utf-8") as f:
                    for p in extracted.get(cam, []):
                        f.write(f"{cam}/{os.path.basename(p)}\n")
        if not dry_run:
            shared = [os.path.basename(p) for p in extracted.get("wide", [])]
            write_cross_lens_pairs_file(
                os.path.join(image_root, "_cross_lens_pairs.txt"), shared
            )
    else:
        # Legacy single-lens / independent extract
        for cid, cam in enumerate(opts.cameras, start=1):
            stream = pack.stream(cam)
            if stream is None:
                continue
            n = stream.accepted_frame_count or stream.frame_count or len(stream.frame_times)
            cam_dir = os.path.join(image_root, cam)
            video_path = os.path.join(pack.root, stream.raw_video_file)
            if on_progress:
                on_progress(
                    f"[{cid}/{n_cams}] Extracting {cam} (keep={opts.keep_pct:.0%}, step={step})..."
                )
            paths, _ = _ffmpeg_extract(
                opts.ffmpeg_bin,
                video_path,
                step,
                n,
                cam_dir,
                cam,
                resize_width=opts.resize_width,
                dry_run=dry_run,
                on_progress=on_progress,
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
        sync_meta["sync_method"] = "independent"

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
        **sync_meta,
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
        from .rig import run_dual_colmap

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
    write_colmap_script: bool = False,
    on_progress=None,
) -> list[list[str]]:
    """Rebuild and run COLMAP for an existing prep directory."""
    from .rig import run_dual_colmap

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
    if write_colmap_script:
        _write_run_script(prep_dir, commands)
        if on_progress:
            on_progress(f"Wrote COLMAP script: {os.path.join(prep_dir, 'run_colmap.bat')}")
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

    dual_report = run_dual_colmap(
        colmap_bin=colmap_bin,
        image_root=image_root,
        database_path=database_path,
        sparse_dir=sparse_dir,
        cameras=list(cam_names),
        settings=settings,
        inject_cameras=colmap_cams,
        inject_intrinsics=inject_intrinsics and bool(colmap_cams),
        on_log=on_progress,
        run_sequence=run_colmap_sequence,
    )
    if os.path.isfile(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["dual_colmap"] = dual_report
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except (OSError, ValueError):
            pass
    return commands


def _colmap_step_title(cmd: list[str]) -> str:
    """Bar / Status title, e.g. 'COLMAP Feature Extractor (wide)'."""
    if len(cmd) < 2:
        return "COLMAP"
    sub = cmd[1]
    titles = {
        "feature_extractor": "COLMAP Feature Extractor",
        "sequential_matcher": "COLMAP Sequential Matcher",
        "exhaustive_matcher": "COLMAP Exhaustive Matcher",
        "vocab_tree_matcher": "COLMAP Vocab Tree Matcher",
        "matches_importer": "COLMAP Cross-lens Match",
        "mapper": "COLMAP Mapper",
        "image_registrator": "COLMAP Image Registrator",
        "point_triangulator": "COLMAP Point Triangulator",
        "bundle_adjuster": "COLMAP Bundle Adjuster",
        "rig_configurator": "COLMAP Rig Configurator",
    }
    title = titles.get(sub, f"COLMAP {sub}")
    if sub == "feature_extractor":
        joined = " ".join(cmd)
        for cam in ("wide", "ultra"):
            if f"_list_{cam}.txt" in joined or f"_list_{cam}" in joined:
                return f"{title} ({cam})"
    return title


# Kept for tests / older imports
def _colmap_step_label(cmd: list[str]) -> str:
    return _colmap_step_title(cmd)


_PROCESSED_FILE_RE = re.compile(
    r"Processed file\s*\[(\d+)\s*/\s*(\d+)\]", re.IGNORECASE
)


def _format_colmap_bar(
    step: int,
    n_steps: int,
    title: str,
    cur: Optional[int] = None,
    total: Optional[int] = None,
) -> str:
    """e.g. '[1/4] COLMAP Feature Extractor (wide) (100/200, 50%)'."""
    head = f"[{step}/{n_steps}] {title}"
    if cur is not None and total is not None and total > 0:
        pct = int(round(100.0 * cur / total))
        return f"{head} ({cur}/{total}, {pct}%)"
    return head


def _cmd_flag(cmd: list[str], name: str) -> str:
    for i, a in enumerate(cmd):
        if a == name and i + 1 < len(cmd):
            return cmd[i + 1]
    return ""


def _read_colmap_db_counts(database_path: str) -> dict[str, int]:
    """Read-only snapshot of COLMAP SQLite tables (best effort)."""
    out = {
        "images": 0,
        "keypoints_images": 0,
        "keypoints_total": 0,
        "match_pairs": 0,
        "verified_pairs": 0,
    }
    if not database_path or not os.path.isfile(database_path):
        return out
    try:
        import sqlite3

        con = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True, timeout=2.0)
        try:
            cur = con.cursor()

            def _count(sql: str) -> int:
                try:
                    row = cur.execute(sql).fetchone()
                    return int(row[0] or 0) if row else 0
                except sqlite3.Error:
                    return 0

            out["images"] = _count("SELECT COUNT(*) FROM images")
            out["keypoints_images"] = _count("SELECT COUNT(*) FROM keypoints")
            out["keypoints_total"] = _count("SELECT COALESCE(SUM(rows), 0) FROM keypoints")
            out["match_pairs"] = _count("SELECT COUNT(*) FROM matches")
            out["verified_pairs"] = _count(
                "SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0"
            )
        finally:
            con.close()
    except Exception:
        pass
    return out


def _estimate_sequential_attempted(num_images: int, overlap: int) -> int:
    """Upper bound on pairs sequential_matcher tries (forward neighbors)."""
    if num_images <= 1 or overlap <= 0:
        return 0
    o = min(int(overlap), num_images - 1)
    # For each i, match i+1 .. i+o → sum_{k=1..n-1} min(o, n-1-k+1) = o*(n-o) + o*(o-1)/2? 
    # Simpler exact: sum_{i=0}^{n-2} min(o, n-1-i)
    total = 0
    n = int(num_images)
    for i in range(n - 1):
        total += min(o, n - 1 - i)
    return total


def _sparse_model_summary(sparse_dir: str) -> str:
    """Summarize first sparse model folder if present."""
    if not sparse_dir or not os.path.isdir(sparse_dir):
        return "no sparse model yet"
    # Prefer numeric subdirs (0, 1, ...)
    subs = []
    for name in os.listdir(sparse_dir):
        path = os.path.join(sparse_dir, name)
        if os.path.isdir(path) and name.isdigit():
            subs.append((int(name), path))
    subs.sort()
    if not subs:
        return "sparse/ empty"
    model_dir = subs[0][1]
    idx = subs[0][0]
    n_img = 0
    n_pts = 0
    images_txt = os.path.join(model_dir, "images.txt")
    points_txt = os.path.join(model_dir, "points3D.txt")
    images_bin = os.path.join(model_dir, "images.bin")
    points_bin = os.path.join(model_dir, "points3D.bin")
    try:
        if os.path.isfile(images_txt):
            with open(images_txt, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    # TEXT format: image lines alternate with POINTS2D lines
                    parts = s.split()
                    if len(parts) >= 10 and parts[0].isdigit():
                        n_img += 1
        elif os.path.isfile(images_bin):
            n_img = -1  # unknown without full parser
        if os.path.isfile(points_txt):
            with open(points_txt, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        n_pts += 1
        elif os.path.isfile(points_bin):
            n_pts = -1
    except OSError:
        pass
    img_s = "?" if n_img < 0 else str(n_img)
    pts_s = "?" if n_pts < 0 else f"{n_pts:,}"
    return f"sparse/{idx}: {img_s} registered, {pts_s} points"


def _colmap_step_result_line(
    cmd: list[str],
    *,
    step: int,
    n_steps: int,
    title: str,
) -> str:
    """Human summary for Results panel after a step finishes."""
    sub = cmd[1] if len(cmd) > 1 else ""
    db = _cmd_flag(cmd, "--database_path")
    counts = _read_colmap_db_counts(db)
    head = f"[{step}/{n_steps}] {title}"

    if sub == "feature_extractor":
        return (
            f"{head} — {counts['keypoints_images']} images with features, "
            f"{counts['keypoints_total']:,} keypoints"
        )
    if sub in ("sequential_matcher", "exhaustive_matcher", "vocab_tree_matcher"):
        verified = counts["verified_pairs"]
        attempted = counts["match_pairs"]
        if sub == "sequential_matcher":
            try:
                overlap = int(_cmd_flag(cmd, "--SequentialMatching.overlap") or "0")
            except ValueError:
                overlap = 0
            est = _estimate_sequential_attempted(counts["images"], overlap)
            if est > attempted:
                attempted = est
        if attempted <= 0:
            attempted = max(verified, 1)
        return (
            f"{head} — verified pairs {verified:,} / attempted {attempted:,} "
            f"(images {counts['images']})"
        )
    if sub == "mapper":
        sparse = _cmd_flag(cmd, "--output_path")
        return f"{head} — {_sparse_model_summary(sparse)}"
    return f"{head} — done"


def _run_colmap_cmd_streaming(
    cmd: list[str],
    *,
    step: int,
    n_steps: int,
    title: str,
    on_log: Optional[Callable[..., Any]] = None,
) -> None:
    """Run one COLMAP argv, stream logs, emit bar-friendly progress lines."""

    def emit(
        msg: str,
        *,
        status: bool = True,
        frac: Optional[float] = None,
        result: bool = False,
    ) -> None:
        if not on_log:
            return
        try:
            on_log(msg, status=status, frac=frac, result=result)
        except TypeError:
            on_log(msg)

    # Start of step → overall fraction at beginning of this step.
    base = (step - 1) / max(n_steps, 1)
    emit(_format_colmap_bar(step, n_steps, title), status=True, frac=base)

    # COLMAP logs via glog often go to stderr; merge both.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    tail: list[str] = []
    last_cur = 0
    last_total = 0
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n\r")
            if not line:
                continue
            tail.append(line)
            if len(tail) > 80:
                tail = tail[-80:]
            m = _PROCESSED_FILE_RE.search(line)
            if m:
                cur, total = int(m.group(1)), max(int(m.group(2)), 1)
                last_cur, last_total = cur, total
                sub = cur / total
                frac = base + sub / max(n_steps, 1)
                # CLI-style Status: stream progress lines (panel may coalesce).
                emit(
                    _format_colmap_bar(step, n_steps, title, cur, total),
                    status=True,
                    frac=frac,
                )
            else:
                # Pass through short COLMAP log lines (no huge paths).
                low = line.lower()
                if "processed file" in low or "registering" in low or " => " in line:
                    if len(line) > 200:
                        line = line[:199] + "…"
                    if "\\" not in line and ".bat" not in low:
                        emit(line, status=True, frac=None)
    finally:
        code = proc.wait()

    if code != 0:
        err = "\n".join(tail)[-800:]
        raise RuntimeError(
            f"COLMAP step failed (exit {code}): {title}\n"
            f"{' '.join(cmd)}\n{err}"
        )

    # Step complete → Results summary + Status done line.
    summary = _colmap_step_result_line(cmd, step=step, n_steps=n_steps, title=title)
    emit(summary, status=False, result=True, frac=step / max(n_steps, 1))
    emit(
        _format_colmap_bar(
            step,
            n_steps,
            title,
            last_total if last_total > 0 else None,
            last_total if last_total > 0 else None,
        )
        + " done",
        status=True,
        frac=step / max(n_steps, 1),
    )


def run_colmap_sequence(commands: list[list[str]], on_log=None) -> None:
    n = len(commands)
    for i, cmd in enumerate(commands, start=1):
        title = _colmap_step_title(cmd)
        _run_colmap_cmd_streaming(
            cmd, step=i, n_steps=n, title=title, on_log=on_log
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
