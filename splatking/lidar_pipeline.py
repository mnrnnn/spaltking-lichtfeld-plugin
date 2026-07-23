"""LiDAR (photo + LiDAR) reconstruction preparation.

The LiDAR path skips SfM entirely: SplatKing writes an on-device
``COLMAP_Text_Model`` (poses + sparse cloud) that LichtFeld ingests directly.
The only value-add here is unlocking the native Depth Loss by decoding the raw
ARKit depth + confidence maps into per-image metric depth maps.

Depth format (from sensor_data metadata):
* ``*_depth.bin``       - 256x192 Float32, metric meters (kCVPixelFormatType_DepthFloat32)
* ``*_confidence.bin``  - 256x192 UInt8, ARConfidenceLevel {0=low,1=med,2=high}
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Optional

from .colmap_model import find_sparse_dir, read_model
from .pack import LidarPack, LidarPair, load_lidar_frame_metadata


@dataclass
class LidarPrepOptions:
    out_dir: str
    confidence_min: int = 1            # 0=low,1=med,2=high; mask below this
    upsample_to_image: bool = False    # resize depth to image resolution
    depth_format: str = "npy"          # "npy" | "png16" (millimeters) | "both"
    max_depth_meters: float = 0.0      # 0 = no clamp


@dataclass
class LidarPrepResult:
    colmap_model_dir: Optional[str]
    sparse_dir: Optional[str]
    registered_images: int
    num_points3d: int
    depth_dir: Optional[str]
    depth_written: int
    manifest_path: Optional[str]


def _read_float32_grid(path: str, width: int, height: int, bytes_per_row: int):
    """Read a Float32 depth .bin honoring row padding; return numpy 2D array."""
    import numpy as np

    with open(path, "rb") as f:
        raw = f.read()
    if bytes_per_row and bytes_per_row != width * 4:
        rows = []
        for r in range(height):
            start = r * bytes_per_row
            row = raw[start:start + width * 4]
            rows.append(np.frombuffer(row, dtype="<f4", count=width))
        return np.vstack(rows)
    return np.frombuffer(raw, dtype="<f4", count=width * height).reshape(height, width)


def _read_uint8_grid(path: str, width: int, height: int, bytes_per_row: int):
    import numpy as np

    with open(path, "rb") as f:
        raw = f.read()
    if bytes_per_row and bytes_per_row != width:
        rows = []
        for r in range(height):
            start = r * bytes_per_row
            rows.append(np.frombuffer(raw[start:start + width], dtype="u1", count=width))
        return np.vstack(rows)
    return np.frombuffer(raw, dtype="u1", count=width * height).reshape(height, width)


def decode_depth_map(
    depth_path: str,
    depth_w: int,
    depth_h: int,
    depth_bpr: int,
    conf_path: Optional[str],
    conf_w: int,
    conf_h: int,
    conf_bpr: int,
    confidence_min: int,
    max_depth: float,
):
    """Return a float32 metric depth array with low-confidence pixels set to 0."""
    import numpy as np

    depth = _read_float32_grid(depth_path, depth_w, depth_h, depth_bpr).astype("float32").copy()

    if conf_path and os.path.isfile(conf_path) and confidence_min > 0:
        conf = _read_uint8_grid(conf_path, conf_w, conf_h, conf_bpr)
        if conf.shape == depth.shape:
            depth[conf < confidence_min] = 0.0

    depth[~np.isfinite(depth)] = 0.0
    if max_depth and max_depth > 0:
        depth[depth > max_depth] = 0.0
    return depth


def prepare_lidar_dataset(pack: LidarPack, opts: LidarPrepOptions) -> LidarPrepResult:
    os.makedirs(opts.out_dir, exist_ok=True)

    sparse_dir = find_sparse_dir(pack.colmap_model_dir) if pack.colmap_model_dir else None
    registered = 0
    num_points = 0
    if sparse_dir:
        model = read_model(sparse_dir)
        registered = len(model.images)
        num_points = len(model.points3d)

    depth_dir = None
    depth_written = 0
    manifest_path = None

    try:
        import numpy  # noqa: F401
        have_numpy = True
    except ImportError:
        have_numpy = False

    if have_numpy and pack.sensor_data_dir:
        depth_dir = os.path.join(opts.out_dir, "depths")
        os.makedirs(depth_dir, exist_ok=True)
        manifest = {
            "units": "meters",
            "confidence_min": opts.confidence_min,
            "depth_format": opts.depth_format,
            "entries": [],
        }
        for pair in pack.pairs:
            if not pair.depth_available:
                continue
            depth_aux = pair.aux_of("depth")
            conf_aux = pair.aux_of("depthConfidence")
            if depth_aux is None:
                continue
            depth_path = os.path.join(pack.sensor_data_dir, depth_aux.filename)
            if not os.path.isfile(depth_path):
                continue
            conf_path = (
                os.path.join(pack.sensor_data_dir, conf_aux.filename) if conf_aux else None
            )
            depth = decode_depth_map(
                depth_path, depth_aux.width, depth_aux.height, depth_aux.bytes_per_row,
                conf_path,
                conf_aux.width if conf_aux else 0,
                conf_aux.height if conf_aux else 0,
                conf_aux.bytes_per_row if conf_aux else 0,
                opts.confidence_min, opts.max_depth_meters,
            )

            if opts.upsample_to_image:
                load_lidar_frame_metadata(pack, pair)
                if pair.image_width and pair.image_height:
                    depth = _nn_resize(depth, pair.image_width, pair.image_height)

            base = os.path.splitext(os.path.basename(pair.image_file))[0]
            written = _write_depth(depth, depth_dir, base, opts.depth_format)
            depth_written += 1
            manifest["entries"].append({
                "image": pair.image_file,
                "depth_files": written,
                "quality_score": pair.quality_score,
                "quality_band": pair.quality_band,
            })

        manifest_path = os.path.join(opts.out_dir, "depth_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return LidarPrepResult(
        colmap_model_dir=pack.colmap_model_dir,
        sparse_dir=sparse_dir,
        registered_images=registered,
        num_points3d=num_points,
        depth_dir=depth_dir,
        depth_written=depth_written,
        manifest_path=manifest_path,
    )


def _nn_resize(arr, out_w: int, out_h: int):
    import numpy as np

    h, w = arr.shape
    ys = (np.arange(out_h) * h / out_h).astype("int64").clip(0, h - 1)
    xs = (np.arange(out_w) * w / out_w).astype("int64").clip(0, w - 1)
    return arr[ys][:, xs]


def _write_depth(depth, depth_dir: str, base: str, fmt: str) -> list[str]:
    import numpy as np

    written: list[str] = []
    if fmt in ("npy", "both"):
        p = os.path.join(depth_dir, base + ".npy")
        np.save(p, depth.astype("float32"))
        written.append(os.path.basename(p))
    if fmt in ("png16", "both"):
        p = os.path.join(depth_dir, base + ".png")
        mm = np.clip(depth * 1000.0, 0, 65535).astype("uint16")
        _write_png16(p, mm)
        written.append(os.path.basename(p))
    return written


def _write_png16(path: str, arr) -> None:
    """Write a 16-bit grayscale PNG. Uses OpenCV/Pillow if present, else a
    minimal zlib-based encoder so it works with numpy alone."""
    import numpy as np

    try:
        import cv2  # type: ignore

        cv2.imwrite(path, arr)
        return
    except ImportError:
        pass
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(arr, mode="I;16").save(path)
        return
    except Exception:
        pass

    import zlib

    h, w = arr.shape
    be = arr.astype(">u2").tobytes()
    raw = bytearray()
    stride = w * 2
    for r in range(h):
        raw.append(0)  # filter type 0
        raw.extend(be[r * stride:(r + 1) * stride])

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + typ + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0)  # 16-bit grayscale
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw), 6)))
        f.write(chunk(b"IEND", b""))
