"""Per-camera intrinsics from SplatKing metadata.

This is the crux of the "dual-lens per-camera intrinsic" problem: the GUI
COLMAP plugin exposes a single camera-model dropdown and no ``camera_params``
input, so the known wide (74.6 deg) and ultra (106.2 deg) intrinsics cannot be
injected. Here we derive a rectilinear PINHOLE model per lens directly from the
metadata field-of-view and resolution, ready to be written into a COLMAP model
or a database with two distinct cameras.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .pack import DeviceInfo


@dataclass
class ColmapCamera:
    """A COLMAP camera row: ``CAMERA_ID MODEL WIDTH HEIGHT PARAMS...``."""

    camera_id: int
    model: str  # PINHOLE
    width: int
    height: int
    params: list[float]  # PINHOLE -> [fx, fy, cx, cy]
    source_camera: str = ""  # "wide" | "ultra" (bookkeeping)

    def params_str(self) -> str:
        return " ".join(f"{p:.9g}" for p in self.params)

    def to_line(self) -> str:
        return f"{self.camera_id} {self.model} {self.width} {self.height} {self.params_str()}"


def focal_px_from_fov(pixels: int, fov_deg: float) -> float:
    """Focal length in pixels for an angle of view spanning ``pixels``.

    AVFoundation's ``fieldOfView`` is the horizontal angle of view, so we pass
    the frame width as ``pixels``. ``f = (pixels/2) / tan(fov/2)``.
    """
    if fov_deg <= 0.0 or fov_deg >= 180.0:
        raise ValueError(f"Implausible FOV: {fov_deg}")
    return (pixels / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def pinhole_from_fov(
    width: int, height: int, fov_deg: float
) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) for a square-pixel rectilinear camera.

    The horizontal FOV fixes the focal length; square pixels give fx == fy.
    """
    fx = focal_px_from_fov(width, fov_deg)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def colmap_camera_from_device(
    device: DeviceInfo,
    camera_id: int,
    out_width: Optional[int] = None,
    out_height: Optional[int] = None,
) -> ColmapCamera:
    """Build a PINHOLE COLMAP camera for a lens, honoring an output resize.

    ``out_width``/``out_height`` let you scale intrinsics when frames are
    extracted at a lower resolution than the sensor format (common when working
    within a 3080/10GB VRAM budget).
    """
    src_w, src_h = device.width, device.height
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Missing sensor resolution for {device.camera}")

    # Use the corrected FOV when distortion correction is applied in-stream.
    fov = device.corrected_field_of_view if device.distortion_correction_enabled else device.field_of_view
    if fov <= 0.0:
        fov = device.field_of_view

    fx, fy, cx, cy = pinhole_from_fov(src_w, src_h, fov)

    ow = out_width or src_w
    oh = out_height or src_h
    if (ow, oh) != (src_w, src_h):
        sx = ow / src_w
        sy = oh / src_h
        fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy

    return ColmapCamera(
        camera_id=camera_id,
        model="PINHOLE",
        width=ow,
        height=oh,
        params=[fx, fy, cx, cy],
        source_camera=device.camera,
    )


def fov_from_focal_35mm(focal_35mm: float) -> float:
    """Horizontal FOV (degrees) for a full-frame camera with the given 35mm-eq focal.

    Uses the 36mm full-frame sensor width: ``2 * atan(36 / (2 * f))``.
    """
    if focal_35mm <= 0.0:
        raise ValueError(f"Implausible 35mm focal length: {focal_35mm}")
    return 2.0 * math.degrees(math.atan(36.0 / (2.0 * focal_35mm)))


def device_from_exif_35mm(camera: str, exif: dict) -> "DeviceInfo":
    """Build a DeviceInfo from still EXIF (FocalLenIn35mmFilm + pixel size)."""
    from .pack import DeviceInfo

    width = int(exif.get("PixelXDimension") or exif.get("PixelWidth") or 0)
    height = int(exif.get("PixelYDimension") or exif.get("PixelHeight") or 0)
    f35 = float(exif.get("FocalLenIn35mmFilm") or 0.0)
    fov = fov_from_focal_35mm(f35) if f35 > 0 else 0.0
    # Known iPhone dual-lens fallbacks when EXIF lacks 35mm equivalent.
    if fov <= 0.0:
        fov = 74.6 if camera == "wide" else 106.2 if camera == "ultra" else 70.0
    if width <= 0 or height <= 0:
        width, height = 4224, 2376
    return DeviceInfo(
        camera=camera,
        device_type="exif",
        localized_name=str(exif.get("LensModel") or camera),
        width=width,
        height=height,
        field_of_view=fov,
        corrected_field_of_view=fov,
        distortion_correction_supported=True,
        distortion_correction_enabled=True,
        iso=_exif_iso(exif),
        exposure_seconds=_maybe_float_local(exif.get("ExposureTime")),
    )


def _exif_iso(exif: dict) -> Optional[float]:
    iso = exif.get("ISOSpeedRatings")
    if isinstance(iso, (list, tuple)) and iso:
        return _maybe_float_local(iso[0])
    return _maybe_float_local(iso)


def _maybe_float_local(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def colmap_camera_from_arkit_intrinsics(
    intrinsics_col_major: list[float],
    width: int,
    height: int,
    camera_id: int,
    source_camera: str = "wide",
) -> ColmapCamera:
    """Build a PINHOLE camera from an ARKit 3x3 intrinsics matrix.

    ARKit stores the intrinsics column-major:
    ``[fx, 0, 0,  0, fy, 0,  cx, cy, 1]`` (elements 0,4 = fx,fy; 6,7 = cx,cy).
    Used for the LiDAR path when per-frame intrinsics are preferred over the
    session-median PINHOLE that SplatKing already writes to cameras.txt.
    """
    if len(intrinsics_col_major) < 9:
        raise ValueError("ARKit intrinsics must have 9 elements (3x3 column-major)")
    fx = intrinsics_col_major[0]
    fy = intrinsics_col_major[4]
    cx = intrinsics_col_major[6]
    cy = intrinsics_col_major[7]
    return ColmapCamera(
        camera_id=camera_id,
        model="PINHOLE",
        width=width,
        height=height,
        params=[fx, fy, cx, cy],
        source_camera=source_camera,
    )
