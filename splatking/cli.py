"""Standalone CLI for SplatKing -> COLMAP/LichtFeld preparation."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _add_package_root() -> None:
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
    prep.add_argument("--cameras", default="wide,ultra")
    prep.add_argument("--keep-pct", type=float, default=0.10,
                      help="Fraction of video frames to keep (uniform stride)")
    prep.add_argument("--resize", type=int, default=1920, help="Resize width (0=native)")
    prep.add_argument("--blur-percentile", type=float, default=0.15)
    prep.add_argument("--blur-threshold", type=float, default=0.0)
    prep.add_argument("--matcher", choices=["exhaustive", "sequential", "vocab_tree"],
                      default="sequential")
    prep.add_argument("--vocab-tree", default="")
    prep.add_argument("--no-inject-intrinsics", action="store_true")
    prep.add_argument("--run-colmap", action="store_true")
    prep.add_argument(
        "--write-colmap-script",
        action="store_true",
        help="Write run_colmap.bat / .sh (off by default)",
    )
    prep.add_argument("--colmap-bin", default="colmap")
    prep.add_argument("--ffmpeg-bin", default="ffmpeg")
    prep.add_argument("--confidence-min", type=int, default=1)
    prep.add_argument("--depth-format", choices=["npy", "png16", "both"], default="npy")
    prep.add_argument("--dry-run", action="store_true")
    prep.add_argument("--use-gpu", action="store_true", default=True)
    prep.add_argument("--no-gpu", action="store_true")
    prep.add_argument("--max-image-size", type=int, default=3200)
    prep.add_argument("--max-num-features", type=int, default=8192)
    prep.add_argument("--seq-overlap", type=int, default=10)
    prep.add_argument("--min-num-matches", type=int, default=15)
    prep.add_argument("--no-dual", action="store_true", help="Disable dual-lens merge")
    prep.add_argument("--base-lens", choices=["ultra", "wide"], default="ultra")
    prep.add_argument(
        "--dual-method",
        choices=["auto", "registrator", "rig", "wide_only"],
        default="auto",
    )

    col = sub.add_parser("colmap", help="Run COLMAP on a prep out-dir")
    col.add_argument("prep_dir", help="Directory produced by `prepare`")
    col.add_argument("--colmap-bin", default="colmap")
    col.add_argument("--matcher", choices=["exhaustive", "sequential", "vocab_tree"],
                      default="sequential")
    col.add_argument("--vocab-tree", default="")
    col.add_argument("--no-gpu", action="store_true")
    col.add_argument("--max-image-size", type=int, default=3200)
    col.add_argument("--max-num-features", type=int, default=8192)
    col.add_argument("--seq-overlap", type=int, default=10)
    col.add_argument("--min-num-matches", type=int, default=15)
    col.add_argument("--no-dual", action="store_true")
    col.add_argument("--base-lens", choices=["ultra", "wide"], default="ultra")
    col.add_argument(
        "--dual-method",
        choices=["auto", "registrator", "rig", "wide_only"],
        default="auto",
    )
    col.add_argument(
        "--write-colmap-script",
        action="store_true",
        help="Write run_colmap.bat / .sh (off by default)",
    )

    cam = sub.add_parser("cameras", help="Subsample registered cameras for training")
    cam.add_argument("sparse_dir")
    cam.add_argument("--out", required=True)
    cam.add_argument("--mode", choices=["every_n", "random_pct", "keep_all"], default="every_n")
    cam.add_argument("--every-n", type=int, default=2)
    cam.add_argument("--random-pct", type=float, default=0.5)
    cam.add_argument("--seed", type=int, default=42)

    info = sub.add_parser("info", help="Print pack summary")
    info.add_argument("pack")

    return p


def _colmap_settings_from_args(args, prefs) -> "ColmapSettings":
    from splatking.video_pipeline import ColmapSettings

    use_gpu = not getattr(args, "no_gpu", False)
    return ColmapSettings(
        matcher=getattr(args, "matcher", "sequential"),
        vocab_tree_path=getattr(args, "vocab_tree", "") or prefs.get("vocab_tree_path", ""),
        use_gpu=use_gpu,
        max_image_size=int(getattr(args, "max_image_size", 3200)),
        max_num_features=int(getattr(args, "max_num_features", 8192)),
        sequential_overlap=int(getattr(args, "seq_overlap", 10)),
        min_num_matches=int(getattr(args, "min_num_matches", 15)),
        dual_mode=not getattr(args, "no_dual", False),
        base_lens=getattr(args, "base_lens", "ultra") or "ultra",
        dual_method=getattr(args, "dual_method", "auto") or "auto",
    )


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
        for s in pack.streams:
            cam = colmap_camera_from_device(s.device, camera_id=1)
            print(
                f"  {s.camera}: {s.frame_count} frames, "
                f"{s.device.width}x{s.device.height}, "
                f"FOV={s.device.field_of_view:.2f}°, fx={cam.params[0]:.2f}"
            )
    elif ct == CaptureType.PHOTO_DUAL:
        print(f"folder: {pack.folder_name}")
        print(f"pairs: {pack.pair_count} frames: {len(pack.frames)}")
        for cam in pack.cameras:
            dev = pack.representative_device(cam)
            n = len(pack.frames_for(cam))
            if dev:
                cc = colmap_camera_from_device(dev, 1)
                print(f"  {cam}: {n} images, FOV={dev.field_of_view:.2f}°, fx={cc.params[0]:.2f}")
            else:
                print(f"  {cam}: {n} images")
    else:
        print(f"folder: {pack.folder_name}")
        print(f"pairs: {pack.pair_count}")
        print(f"colmap_model: {pack.colmap_model_dir}")
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
    settings = _colmap_settings_from_args(args, prefs)

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
            keep_pct=args.keep_pct,
            resize_width=args.resize,
            blur_percentile=args.blur_percentile,
            blur_abs_threshold=args.blur_threshold,
            inject_intrinsics=not args.no_inject_intrinsics,
            colmap_bin=colmap_bin,
            ffmpeg_bin=ffmpeg_bin,
            run_colmap=args.run_colmap,
            write_colmap_script=bool(args.write_colmap_script),
            colmap=settings,
        )
        result = prepare_video_dataset(pack, opts, dry_run=args.dry_run)
        print(json.dumps({
            "out_dir": result.out_dir,
            "kept": {k: len(v) for k, v in result.extracted.items()},
            "report": result.report_path,
            "colmap_commands": len(result.commands),
        }, indent=2))
    elif ct == CaptureType.PHOTO_DUAL:
        opts = PhotoPrepOptions(
            out_dir=args.out,
            cameras=cams,
            blur_percentile=args.blur_percentile,
            inject_intrinsics=not args.no_inject_intrinsics,
            colmap_bin=colmap_bin,
            run_colmap=args.run_colmap,
            write_colmap_script=bool(args.write_colmap_script),
            colmap=settings,
        )
        result = prepare_photo_dataset(pack, opts, dry_run=args.dry_run)
        print(json.dumps({
            "out_dir": result.out_dir,
            "kept": {k: len(v) for k, v in result.extracted.items()},
            "report": result.report_path,
        }, indent=2))
    else:
        opts = LidarPrepOptions(
            out_dir=args.out,
            confidence_min=args.confidence_min,
            depth_format=args.depth_format,
        )
        result = prepare_lidar_dataset(pack, opts)
        print(json.dumps({
            "registered_images": result.registered_images,
            "depth_written": result.depth_written,
            "manifest": result.manifest_path,
        }, indent=2))
    return 0


def cmd_colmap(args: argparse.Namespace) -> int:
    from splatking.prefs import load_prefs, apply_tool_defaults
    from splatking.video_pipeline import run_colmap_on_prep

    prefs = apply_tool_defaults(load_prefs())
    colmap_bin = args.colmap_bin if args.colmap_bin != "colmap" else (prefs.get("colmap_bin") or "colmap")
    settings = _colmap_settings_from_args(args, prefs)
    print(f"running COLMAP in {args.prep_dir}...")
    cmds = run_colmap_on_prep(
        args.prep_dir,
        colmap_bin,
        settings,
        write_colmap_script=bool(args.write_colmap_script),
        on_progress=print,
    )
    print(f"done ({len(cmds)} steps)")
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
