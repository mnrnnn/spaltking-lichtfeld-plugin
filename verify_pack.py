"""Quick self-check against a local SplatKing pack (no LichtFeld)."""

from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from splatking.pack import (
    detect_capture_type,
    load_pack,
    CaptureType,
    load_lidar_frame_metadata,
    default_out_dir,
)
from splatking.intrinsics import colmap_camera_from_device, pinhole_from_fov, fov_from_focal_35mm
from splatking.colmap_model import read_model, find_sparse_dir
from splatking.camera_select import CameraSelectOptions, subsample_model
from splatking.lidar_pipeline import LidarPrepOptions, prepare_lidar_dataset
from splatking.video_pipeline import (
    VideoPrepOptions,
    prepare_video_dataset,
    ColmapSettings,
    step_from_keep_pct,
)
from splatking.photo_pipeline import PhotoPrepOptions, prepare_photo_dataset


def _pack_root() -> str:
    env = os.environ.get("SPALTKING_PACK", "").strip()
    if env and os.path.isdir(env):
        return env
    local = os.path.join(ROOT, "spaltking_pack")
    if os.path.isdir(local):
        return local
    raise SystemExit(
        "No sample pack found.\n"
        "Set SPALTKING_PACK or place gitignored spaltking_pack/ next to plugin root."
    )


def check_video(pack_root: str):
    path = os.path.join(pack_root, "3_video")
    if not os.path.isdir(path):
        if detect_capture_type(pack_root) == CaptureType.VIDEO_DUAL:
            path = pack_root
        else:
            print(f"[skip] video sample not found: {path}")
            return
    assert detect_capture_type(path) == CaptureType.VIDEO_DUAL
    pack = load_pack(path)
    assert pack.stream("wide") and pack.stream("ultra")
    wide = colmap_camera_from_device(pack.stream("wide").device, 1)
    ultra = colmap_camera_from_device(pack.stream("ultra").device, 2)
    assert wide.params[0] > ultra.params[0]
    assert step_from_keep_pct(0.10) == 10
    print(f"[video] wide fx={wide.params[0]:.2f}  ultra fx={ultra.params[0]:.2f}")

    with tempfile.TemporaryDirectory() as td:
        opts = VideoPrepOptions(
            out_dir=td,
            cameras=["wide", "ultra"],
            keep_pct=0.05,
            blur_percentile=0.0,
            inject_intrinsics=True,
            run_colmap=False,
            colmap=ColmapSettings(),
        )
        result = prepare_video_dataset(pack, opts, dry_run=True)
        assert len(result.cameras) == 2
        assert "wide" in result.extracted and "ultra" in result.extracted
        # ensure short select filter in commands path exists via build
        assert any("feature_extractor" in " ".join(c) for c in result.commands)
        assert any("SiftExtraction.use_gpu" in " ".join(c) for c in result.commands)
        print(f"[video] dry-run kept={[len(v) for v in result.extracted.values()]} cmds={len(result.commands)}")


def check_photo(pack_root: str):
    path = os.path.join(pack_root, "2_photo")
    if not os.path.isdir(path):
        if detect_capture_type(pack_root) == CaptureType.PHOTO_DUAL:
            path = pack_root
        else:
            print(f"[skip] photo sample not found: {path}")
            return
    assert detect_capture_type(path) == CaptureType.PHOTO_DUAL
    pack = load_pack(path)
    assert pack.pair_count >= 1
    wide_dev = pack.representative_device("wide")
    ultra_dev = pack.representative_device("ultra")
    assert wide_dev and ultra_dev
    print(f"[photo] pairs={pack.pair_count} FOV w={wide_dev.field_of_view:.1f} u={ultra_dev.field_of_view:.1f}")

    with tempfile.TemporaryDirectory() as td:
        result = prepare_photo_dataset(
            pack,
            PhotoPrepOptions(
                out_dir=td,
                cameras=["wide", "ultra"],
                blur_percentile=0.0,
                inject_intrinsics=True,
                run_colmap=False,
                colmap=ColmapSettings(matcher="sequential", sequential_overlap=10),
            ),
            dry_run=False,
        )
        assert os.path.isfile(os.path.join(td, "run_colmap.bat"))
        bat = open(os.path.join(td, "run_colmap.bat"), encoding="utf-8").read()
        assert "SequentialMatching.overlap" in bat or "sequential_matcher" in bat
        assert "Mapper.min_num_matches" in bat
        print(f"[photo] kept wide={len(result.extracted['wide'])} ultra={len(result.extracted['ultra'])}")


def check_lidar(pack_root: str):
    path = os.path.join(pack_root, "4_lidar")
    if not os.path.isdir(path):
        if detect_capture_type(pack_root) == CaptureType.PHOTO_LIDAR_SINGLE:
            path = pack_root
        else:
            print(f"[skip] lidar sample not found: {path}")
            return
    assert detect_capture_type(path) == CaptureType.PHOTO_LIDAR_SINGLE
    pack = load_pack(path)
    assert pack.has_colmap_model
    sparse = find_sparse_dir(pack.colmap_model_dir)
    model = read_model(sparse)
    print(f"[lidar] images={len(model.images)} points={len(model.points3d)}")
    subset = subsample_model(model, CameraSelectOptions(mode="every_n", every_n=2))
    print(f"[lidar] every_n=2 → {len(subset.images)}")
    pair = pack.pairs[0]
    load_lidar_frame_metadata(pack, pair)
    with tempfile.TemporaryDirectory() as td:
        result = prepare_lidar_dataset(
            pack, LidarPrepOptions(out_dir=td, confidence_min=1, depth_format="npy")
        )
        print(f"[lidar] depth_written={result.depth_written}")
        assert result.registered_images == len(model.images)


def main():
    pack_root = _pack_root()
    print(f"using pack root: {pack_root}")
    check_photo(pack_root)
    check_video(pack_root)
    check_lidar(pack_root)
    print("OK")


if __name__ == "__main__":
    main()
