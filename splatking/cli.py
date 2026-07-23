"""Standalone CLI for SplatKing -> COLMAP/LichtFeld preparation.

Works without LichtFeld Studio installed. Typical large-scale flow::

    python -m splatking.cli prepare  path/to/Video_...  --out out/video_prep
    python -m splatking.cli prepare  path/to/Photo_...  --out out/photo_prep
    python -m splatking.cli prepare  path/to/Lidar_...  --out out/lidar_prep
    python -m splatking.cli cameras  out/video_prep/sparse/0 --every-n 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _add_package_root() -> None:
    """Allow ``python splatking/cli.py`` and ``python -m splatking.cli``."""
    here = os.path.dirname(os.path.abspath(__file__))
    plugin_root = os.path.dirname(here)
    if plugin_root not in sys.path:
        sys.path.insert(0, plugin_root)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="splatking-import",
        description="Prepare SplatKing photo/video/LiDAR packs for LichtFeld / COLMAP.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Detect pack type and prepare a dataset")
    prep.add_argument("pack", help="Path to a SplatKing capture folder")
    prep.add_argument("--out", required=True, help="Output directory")
    prep.add_argument("--cameras", default="wide,ultra",
                      help="Comma-separated lenses for photo/video packs (default: wide,ultra)")
    prep.add_argument("--stride", type=int, default=15, help="Keep every Nth video frame")
    prep.add_argument("--max-frames", type=int, default=200, help="Cap frames per lens (0=unlimited)")
    prep.add_argument("--resize", type=int, default=1920, help="Resize width (0=native)")
    prep.add_argument("--blur-percentile", type=float, default=0.15,
                      help="Drop blurriest fraction (0..1); 0 disables")
    prep.add_argument("--blur-threshold", type=float, default=0.0,
                      help="Absolute Laplacian variance floor; 0 disables")
    prep.add_argument("--matcher", choices=["exhaustive", "sequential", "vocab_tree"],
                      default="sequential")
    prep.add_argument("--vocab-tree", default="", help="Path to COLMAP vocab tree")
    prep.add_argument("--no-inject-intrinsics", action="store_true",
                      help="Do not write known PINHOLE params")
    prep.add_argument("--run-colmap", action="store_true",
                      help="Run COLMAP immediately after preparation")
    prep.add_argument("--colmap-bin", default="colmap")
    prep.add_argument("--ffmpeg-bin", default="ffmpeg")
    prep.add_argument("--confidence-min", type=int, default=1,
                      help="LiDAR depth: min ARConfidenceLevel to keep (0/1/2)")
    prep.add_argument("--depth-format", choices=["npy", "png16", "both"], default="npy")
    prep.add_argument("--dry-run", action="store_true")

    col = sub.add_parser("colmap", help="Run the prepared COLMAP script in a prep out-dir")
    col.add_argument("prep_dir", help="Directory produced by `prepare`")
    col.add_argument("--colmap-bin", default="colmap")

    cam = sub.add_parser("cameras", help="Subsample registered cameras for training")
    cam.add_argument("sparse_dir", help="COLMAP sparse/0 directory")
    cam.add_argument("--out", required=True, help="Output sparse directory for training")
    cam.add_argument("--mode", choices=["every_n", "random_pct", "keep_all"], default="every_n")
    cam.add_argument("--every-n", type=int, default=2)
    cam.add_argument("--random-pct", type=float, default=0.5)
    cam.add_argument("--seed", type=int, default=42)

    info = sub.add_parser("info", help="Print pack summary without writing outputs")
    info.add_argument("pack", help="Path to a SplatKing capture folder")

    return p


def cmd_info(args: argparse.Namespace) -> int:
    from splatking.pack import CaptureType, load_pack, detect_capture_type
    from splatking.intrinsics import colmap_camera_from_device

    ct = detect_capture_type(args.pack)
    print(f"capture_type: {ct.value}")
    if ct == CaptureType.UNKNOWN:
        return 1
    pack = load_pack(args.pack)
    if ct == CaptureType.VIDEO_DUAL:
        print(f"folder: {pack.folder_name}")
        print(f"parity_valid: {pack.recording_parity_valid}")
        for s in pack.streams:
            cam = colmap_camera_from_device(s.device, camera_id=1)
            print(
                f"  {s.camera}: {s.frame_count} frames, "
                f"{s.device.width}x{s.device.height}, "
                f"FOV={s.device.field_of_view:.2f}°, "
                f"PINHOLE fx={cam.params[0]:.2f}"
            )
        if pack.thermal_fps_events:
            print(f"thermal_throttle_events: {len(pack.thermal_fps_events)}")
    elif ct == CaptureType.PHOTO_DUAL:
        print(f"folder: {pack.folder_name}")
        print(f"pairs: {pack.pair_count}")
        print(f"frames: {len(pack.frames)}")
        for cam in pack.cameras:
            dev = pack.representative_device(cam)
            n = len(pack.frames_for(cam))
            if dev:
                cc = colmap_camera_from_device(dev, 1)
                print(
                    f"  {cam}: {n} images, {dev.width}x{dev.height}, "
                    f"FOV={dev.field_of_view:.2f}°, fx={cc.params[0]:.2f}"
                )
            else:
                print(f"  {cam}: {n} images")
    else:
        print(f"folder: {pack.folder_name}")
        print(f"pairs: {pack.pair_count}")
        print(f"colmap_model: {pack.colmap_model_dir}")
        print(f"sensor_data: {pack.sensor_data_dir}")
        qs = [p.quality_score for p in pack.pairs if p.quality_score is not None]
        if qs:
            print(f"quality_score: min={min(qs):.2f} max={max(qs):.2f} mean={sum(qs)/len(qs):.2f}")
        depth_n = sum(1 for p in pack.pairs if p.depth_available)
        print(f"depth_available: {depth_n}/{len(pack.pairs)}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from splatking.pack import CaptureType, load_pack, detect_capture_type
    from splatking.video_pipeline import VideoPrepOptions, prepare_video_dataset
    from splatking.photo_pipeline import PhotoPrepOptions, prepare_photo_dataset
    from splatking.lidar_pipeline import LidarPrepOptions, prepare_lidar_dataset
    from splatking.prefs import load_prefs, apply_tool_defaults

    prefs = apply_tool_defaults(load_prefs())
    ffmpeg_bin = args.ffmpeg_bin if args.ffmpeg_bin != "ffmpeg" else (prefs.get("ffmpeg_bin") or "ffmpeg")
    colmap_bin = args.colmap_bin if args.colmap_bin != "colmap" else (prefs.get("colmap_bin") or "colmap")

    ct = detect_capture_type(args.pack)
    print(f"detected: {ct.value}")
    if ct == CaptureType.UNKNOWN:
        print("error: not a SplatKing pack", file=sys.stderr)
        return 1
    pack = load_pack(args.pack)
    os.makedirs(args.out, exist_ok=True)
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]

    if ct == CaptureType.VIDEO_DUAL:
        opts = VideoPrepOptions(
            out_dir=args.out,
            cameras=cams,
            stride=args.stride,
            max_frames_per_lens=args.max_frames,
            resize_width=args.resize,
            blur_percentile=args.blur_percentile,
            blur_abs_threshold=args.blur_threshold,
            matcher=args.matcher,
            vocab_tree_path=args.vocab_tree or prefs.get("vocab_tree_path", ""),
            inject_intrinsics=not args.no_inject_intrinsics,
            colmap_bin=colmap_bin,
            ffmpeg_bin=ffmpeg_bin,
            run_colmap=args.run_colmap,
        )
        result = prepare_video_dataset(pack, opts, dry_run=args.dry_run)
        print(json.dumps({
            "out_dir": result.out_dir,
            "cameras": [c.to_line() for c in result.cameras],
            "kept": {k: len(v) for k, v in result.extracted.items()},
            "rejected": {k: len(v) for k, v in result.rejected.items()},
            "report": result.report_path,
            "colmap_commands": len(result.commands),
            "ffmpeg": ffmpeg_bin,
            "colmap": colmap_bin,
        }, indent=2))
    elif ct == CaptureType.PHOTO_DUAL:
        opts = PhotoPrepOptions(
            out_dir=args.out,
            cameras=cams,
            blur_percentile=args.blur_percentile,
            matcher=args.matcher,
            vocab_tree_path=args.vocab_tree or prefs.get("vocab_tree_path", ""),
            inject_intrinsics=not args.no_inject_intrinsics,
            colmap_bin=colmap_bin,
            run_colmap=args.run_colmap,
        )
        result = prepare_photo_dataset(pack, opts, dry_run=args.dry_run)
        print(json.dumps({
            "out_dir": result.out_dir,
            "cameras": [c.to_line() for c in result.cameras],
            "kept": {k: len(v) for k, v in result.extracted.items()},
            "rejected": {k: len(v) for k, v in result.rejected.items()},
            "report": result.report_path,
            "colmap_commands": len(result.commands),
            "colmap": colmap_bin,
        }, indent=2))
    else:
        opts = LidarPrepOptions(
            out_dir=args.out,
            confidence_min=args.confidence_min,
            depth_format=args.depth_format,
        )
        result = prepare_lidar_dataset(pack, opts)
        print(json.dumps({
            "colmap_model_dir": result.colmap_model_dir,
            "sparse_dir": result.sparse_dir,
            "registered_images": result.registered_images,
            "num_points3d": result.num_points3d,
            "depth_dir": result.depth_dir,
            "depth_written": result.depth_written,
            "manifest": result.manifest_path,
        }, indent=2))
    return 0


def cmd_colmap(args: argparse.Namespace) -> int:
    from splatking.video_pipeline import run_colmap_sequence

    report = os.path.join(args.prep_dir, "splatking_prep_report.json")
    if not os.path.isfile(report):
        print(f"error: no report at {report}", file=sys.stderr)
        return 1
    with open(report, "r", encoding="utf-8") as f:
        data = json.load(f)
    cmds = []
    for line in data.get("colmap_commands", []):
        import shlex
        argv = shlex.split(line, posix=os.name != "nt")
        if argv and args.colmap_bin and argv[0] == "colmap":
            argv[0] = args.colmap_bin
        cmds.append(argv)
    if not cmds:
        print("error: report has no colmap_commands", file=sys.stderr)
        return 1
    print(f"running {len(cmds)} COLMAP steps...")
    run_colmap_sequence(cmds, on_log=print)
    return 0


def cmd_cameras(args: argparse.Namespace) -> int:
    from splatking.camera_select import CameraSelectOptions, write_training_subset

    opts = CameraSelectOptions(
        mode=args.mode,
        every_n=args.every_n,
        random_pct=args.random_pct,
        seed=args.seed,
    )
    summary = write_training_subset(args.sparse_dir, args.out, opts)
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    _add_package_root()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "info":
        return cmd_info(args)
    if args.cmd == "prepare":
        return cmd_prepare(args)
    if args.cmd == "colmap":
        return cmd_colmap(args)
    if args.cmd == "cameras":
        return cmd_cameras(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
