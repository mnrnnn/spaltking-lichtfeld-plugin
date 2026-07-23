"""Quick self-check against a local SplatKing pack (no LichtFeld).

Sample packs are NOT in the repo. Point SPALTKING_PACK at a local folder, e.g.:

    set SPALTKING_PACK=D:\\captures\\spaltking_pack
    python verify_pack.py

Or place a folder named ``spaltking_pack`` next to this repo (gitignored).
"""

from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from splatking.pack import detect_capture_type, load_pack, CaptureType, load_lidar_frame_metadata
from splatking.intrinsics import colmap_camera_from_device, pinhole_from_fov
from splatking.colmap_model import read_model, find_sparse_dir
from splatking.camera_select import CameraSelectOptions, subsample_model
from splatking.lidar_pipeline import LidarPrepOptions, prepare_lidar_dataset
from splatking.video_pipeline import VideoPrepOptions, prepare_video_dataset


def _pack_root() -> str:
    env = os.environ.get("SPALTKING_PACK", "").strip()
    if env and os.path.isdir(env):
        return env
    local = os.path.join(ROOT, "spaltking_pack")
    if os.path.isdir(local):
        return local
    raise SystemExit(
        "No sample pack found.\n"
        "Set SPALTKING_PACK to a local capture folder, or place gitignored "
        f"spaltking_pack/ next to the plugin root:\n  {ROOT}"
    )


def check_video(pack_root: str):
    path = os.path.join(pack_root, "3_video")
    if not os.path.isdir(path):
        # Allow pointing SPALTKING_PACK directly at a video_dual folder.
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
    fx_w, _, _, _ = pinhole_from_fov(3840, 2160, 74.597122192382812)
    fx_u, _, _, _ = pinhole_from_fov(3840, 2160, 106.20069885253906)
    assert abs(wide.params[0] - fx_w) < 1e-3
    assert abs(ultra.params[0] - fx_u) < 1e-3
    assert wide.params[0] > ultra.params[0]  # narrower FOV → longer focal
    print(f"[video] wide fx={wide.params[0]:.2f}  ultra fx={ultra.params[0]:.2f}")
    print(f"[video] frames wide={pack.stream('wide').frame_count} ultra={pack.stream('ultra').frame_count}")

    with tempfile.TemporaryDirectory() as td:
        opts = VideoPrepOptions(
            out_dir=td,
            cameras=["wide", "ultra"],
            stride=50,
            max_frames_per_lens=3,
            blur_percentile=0.0,
            inject_intrinsics=True,
            run_colmap=False,
        )
        result = prepare_video_dataset(pack, opts, dry_run=True)
        assert len(result.cameras) == 2
        assert "wide" in result.extracted and "ultra" in result.extracted
        print(f"[video] dry-run kept={[len(v) for v in result.extracted.values()]} cmds={len(result.commands)}")


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
    print(f"[lidar] images={len(model.images)} points={len(model.points3d)} cams={len(model.cameras)}")
    assert len(model.images) > 0
    assert len(model.cameras) == 1

    subset = subsample_model(model, CameraSelectOptions(mode="every_n", every_n=2))
    print(f"[lidar] every_n=2 → {len(subset.images)} training cameras")

    pair = pack.pairs[0]
    load_lidar_frame_metadata(pack, pair)
    print(f"[lidar] sample quality={pair.quality_score} band={pair.quality_band} depth={pair.depth_available}")

    with tempfile.TemporaryDirectory() as td:
        result = prepare_lidar_dataset(
            pack,
            LidarPrepOptions(out_dir=td, confidence_min=1, depth_format="npy"),
        )
        print(
            f"[lidar] depth_written={result.depth_written} "
            f"registered={result.registered_images} points={result.num_points3d}"
        )
        assert result.registered_images == len(model.images)
        if result.depth_written:
            assert os.path.isdir(result.depth_dir)


def main():
    pack_root = _pack_root()
    print(f"using pack root: {pack_root}")
    check_video(pack_root)
    check_lidar(pack_root)
    print("OK")


if __name__ == "__main__":
    main()
