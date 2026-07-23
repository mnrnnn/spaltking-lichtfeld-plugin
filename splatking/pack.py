"""Typed parsing of a SplatKing capture folder ("splatpack").

SplatKing exports three relevant shapes (schema ``splatpack.v2``):

* ``video_dual``          - dual-lens video: ``wide.mov`` + ``ultra.mov``
* ``photo_dual``          - dual-lens stills (no on-device COLMAP)
* ``photo_lidar_single``  - LiDAR series with on-device ``COLMAP_Text_Model``

Plain dataclasses only — no numpy/OpenCV/LichtFeld at import time.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CaptureType(str, Enum):
    VIDEO_DUAL = "video_dual"
    PHOTO_DUAL = "photo_dual"
    PHOTO_LIDAR_SINGLE = "photo_lidar_single"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #
@dataclass
class DeviceInfo:
    """Per-lens camera description taken straight from SplatKing metadata."""

    camera: str  # "wide" | "ultra"
    device_type: str  # e.g. AVCaptureDeviceTypeBuiltInWideAngleCamera
    localized_name: str
    width: int
    height: int
    field_of_view: float  # horizontal FOV in degrees (AVFoundation convention)
    corrected_field_of_view: float
    distortion_correction_supported: bool
    distortion_correction_enabled: bool
    iso: Optional[float] = None
    exposure_seconds: Optional[float] = None
    lens_position: Optional[float] = None

    @property
    def is_undistorted(self) -> bool:
        """True when frames are already rectilinear (safe for a PINHOLE model)."""
        return self.distortion_correction_enabled or not self.distortion_correction_supported

    @classmethod
    def from_metadata(cls, camera: str, d: dict[str, Any]) -> "DeviceInfo":
        fmt = d.get("activeFormat", {})
        return cls(
            camera=camera,
            device_type=d.get("deviceType", ""),
            localized_name=d.get("localizedName", ""),
            width=int(fmt.get("width", 0)),
            height=int(fmt.get("height", 0)),
            field_of_view=float(fmt.get("fieldOfView", 0.0)),
            corrected_field_of_view=float(
                fmt.get("geometricDistortionCorrectedFieldOfView", fmt.get("fieldOfView", 0.0))
            ),
            distortion_correction_supported=bool(d.get("geometricDistortionCorrectionSupported", False)),
            distortion_correction_enabled=bool(d.get("geometricDistortionCorrectionEnabled", False)),
            iso=_maybe_float(d.get("iso")),
            exposure_seconds=_maybe_float(d.get("exposureDurationSeconds")),
            lens_position=_maybe_float(d.get("lensPosition")),
        )


# --------------------------------------------------------------------------- #
# Video (video_dual)
# --------------------------------------------------------------------------- #
@dataclass
class VideoStream:
    camera: str  # "wide" | "ultra"
    raw_video_file: str
    undistorted_video_file: Optional[str]
    device: DeviceInfo
    frame_count: int
    accepted_frame_count: int
    frame_times: list[float] = field(default_factory=list)  # ARKit-clock seconds

    @property
    def video_path_exists(self) -> bool:
        return bool(self.raw_video_file)


@dataclass
class VideoPack:
    root: str
    folder_name: str
    capture_type: CaptureType
    streams: list[VideoStream]
    metadata_file: str = "metadata.json"
    thermal_fps_events: list[dict] = field(default_factory=list)
    recording_parity_valid: bool = True

    def stream(self, camera: str) -> Optional[VideoStream]:
        for s in self.streams:
            if s.camera == camera:
                return s
        return None

    @property
    def cameras(self) -> list[str]:
        return [s.camera for s in self.streams]


# --------------------------------------------------------------------------- #
# Photo dual (still pairs, no on-device COLMAP)
# --------------------------------------------------------------------------- #
@dataclass
class PhotoFrame:
    camera: str
    capture_id: str
    image_file: str
    metadata_file: str
    quality_score: Optional[float] = None
    quality_band: str = ""
    blur_sharpness: Optional[float] = None
    width: int = 0
    height: int = 0
    focal_35mm: Optional[float] = None
    device: Optional[DeviceInfo] = None

    @property
    def basename(self) -> str:
        return os.path.basename(self.image_file)


@dataclass
class PhotoPack:
    root: str
    folder_name: str
    capture_type: CaptureType
    frames: list[PhotoFrame]
    pair_count: int = 0

    def frames_for(self, camera: str) -> list[PhotoFrame]:
        return [f for f in self.frames if f.camera == camera]

    @property
    def cameras(self) -> list[str]:
        seen: list[str] = []
        for f in self.frames:
            if f.camera and f.camera not in seen:
                seen.append(f.camera)
        return seen

    def representative_device(self, camera: str) -> Optional[DeviceInfo]:
        for f in self.frames_for(camera):
            if f.device is not None and f.device.width > 0:
                return f.device
        return None


# --------------------------------------------------------------------------- #
# LiDAR (photo_lidar_single)
# --------------------------------------------------------------------------- #
@dataclass
class AuxOutput:
    type: str  # depth | surfaceConfidence | pointCloudXYZ
    filename: str
    width: int = 0
    height: int = 0
    bytes_per_row: int = 0
    pixel_format: int = 0
    units: str = ""
    coordinate_system: str = ""


@dataclass
class LidarPair:
    camera: str
    image_file: str
    metadata_file: str
    quality_score: Optional[float]
    quality_band: str
    capture_source: str
    depth_available: bool
    aux: list[AuxOutput] = field(default_factory=list)
    intrinsics: Optional[list[float]] = None  # 3x3 column-major (ARKit)
    transform: Optional[list[float]] = None  # 4x4 column-major (ARKit world)
    image_width: int = 0
    image_height: int = 0

    def aux_of(self, kind: str) -> Optional[AuxOutput]:
        for a in self.aux:
            if a.type == kind:
                return a
        return None


@dataclass
class LidarPack:
    root: str
    folder_name: str
    capture_type: CaptureType
    pair_count: int
    pairs: list[LidarPair]
    colmap_model_dir: Optional[str]  # dir containing sparse/0 + images/
    sensor_data_dir: Optional[str]

    @property
    def has_colmap_model(self) -> bool:
        return bool(self.colmap_model_dir)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _maybe_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_out_dir(pack_path: str, capture_type: str | CaptureType) -> str:
    """Default prep folder: ``{pack}/Output/{photo|video|lidar}_prep``."""
    ct = capture_type.value if isinstance(capture_type, CaptureType) else str(capture_type or "")
    kind = {
        CaptureType.VIDEO_DUAL.value: "video_prep",
        CaptureType.PHOTO_DUAL.value: "photo_prep",
        CaptureType.PHOTO_LIDAR_SINGLE.value: "lidar_prep",
    }.get(ct, "prep")
    return os.path.join(pack_path, "Output", kind)


def human_capture_label(capture_type: str | CaptureType) -> str:
    ct = capture_type.value if isinstance(capture_type, CaptureType) else str(capture_type or "")
    return {
        CaptureType.VIDEO_DUAL.value: "Video",
        CaptureType.PHOTO_DUAL.value: "Photo",
        CaptureType.PHOTO_LIDAR_SINGLE.value: "LiDAR",
    }.get(ct, "Unknown")


def detect_capture_type(root: str) -> CaptureType:
    """Detect the capture type without fully parsing the (possibly huge) pack."""
    sp = os.path.join(root, "splatpack.json")
    if os.path.isfile(sp):
        try:
            with open(sp, "r", encoding="utf-8") as f:
                head = f.read(8192)
            if '"video_dual"' in head:
                return CaptureType.VIDEO_DUAL
            if '"photo_lidar_single"' in head:
                return CaptureType.PHOTO_LIDAR_SINGLE
            if '"photo_dual"' in head:
                return CaptureType.PHOTO_DUAL
        except OSError:
            pass

    # Structural fallbacks (order matters).
    if os.path.isfile(os.path.join(root, "metadata.json")) and (
        os.path.isfile(os.path.join(root, "wide.mov"))
        or os.path.isfile(os.path.join(root, "ultra.mov"))
    ):
        return CaptureType.VIDEO_DUAL

    if os.path.isdir(os.path.join(root, "COLMAP_Text_Model")):
        return CaptureType.PHOTO_LIDAR_SINGLE

    started = os.path.join(root, "capture_started.json")
    if os.path.isfile(started):
        try:
            mode = str(_read_json(started).get("mode", "")).lower()
            if mode == "photo":
                return CaptureType.PHOTO_DUAL
            if mode in ("lidar", "photo_lidar"):
                return CaptureType.PHOTO_LIDAR_SINGLE
            if mode == "video":
                return CaptureType.VIDEO_DUAL
        except (OSError, ValueError, TypeError):
            pass

    # Dual stills: wide_*.jpg + ultra_*.jpg without COLMAP model.
    has_jpg_pair = False
    try:
        names = os.listdir(root)
        has_wide = any(n.startswith("wide_") and n.lower().endswith((".jpg", ".jpeg")) for n in names)
        has_ultra = any(n.startswith("ultra_") and n.lower().endswith((".jpg", ".jpeg")) for n in names)
        has_jpg_pair = has_wide and has_ultra
    except OSError:
        pass
    if has_jpg_pair and not os.path.isdir(os.path.join(root, "sensor_data")):
        return CaptureType.PHOTO_DUAL

    # LiDAR without typed splatpack: sensor_data + photo_series.
    if os.path.isdir(os.path.join(root, "sensor_data")) and os.path.isfile(
        os.path.join(root, "photo_series.json")
    ):
        return CaptureType.PHOTO_LIDAR_SINGLE

    return CaptureType.UNKNOWN


def load_pack(root: str):
    """Load a SplatKing pack folder into VideoPack, PhotoPack, or LidarPack."""
    ct = detect_capture_type(root)
    if ct == CaptureType.VIDEO_DUAL:
        return _load_video_pack(root)
    if ct == CaptureType.PHOTO_DUAL:
        return _load_photo_pack(root)
    if ct == CaptureType.PHOTO_LIDAR_SINGLE:
        return _load_lidar_pack(root)
    raise ValueError(f"Not a recognizable SplatKing pack: {root}")


def _load_video_pack(root: str) -> VideoPack:
    splat = _read_json(os.path.join(root, "splatpack.json"))
    meta_path = os.path.join(root, "metadata.json")
    meta = _read_json(meta_path) if os.path.isfile(meta_path) else {}

    streams: list[VideoStream] = []
    for s in splat.get("streams", []):
        cam = s.get("camera", "")
        dev = DeviceInfo.from_metadata(cam, s.get("deviceInfo", meta.get(cam, {})))
        streams.append(
            VideoStream(
                camera=cam,
                raw_video_file=s.get("rawVideoFile", f"{cam}.mov"),
                undistorted_video_file=s.get("undistortedVideoFile"),
                device=dev,
                frame_count=int(s.get("frameCount", 0)),
                accepted_frame_count=int(s.get("acceptedFrameCount", s.get("frameCount", 0))),
                frame_times=[float(t) for t in s.get("frameTimes", [])],
            )
        )

    for st in streams:
        if st.device.width == 0 and st.camera in meta:
            st.device = DeviceInfo.from_metadata(st.camera, meta[st.camera])

    thermal = []
    diag = meta.get("captureDiagnostics", {})
    for ev in diag.get("events", []):
        if ev.get("kind") == "thermal_throttle_gear":
            thermal.append(ev.get("metadata", {}))

    parity = splat.get("recordingParity", {}).get("parityValid", True)

    return VideoPack(
        root=root,
        folder_name=splat.get("folderName", os.path.basename(root)),
        capture_type=CaptureType.VIDEO_DUAL,
        streams=streams,
        metadata_file=splat.get("metadataFile", "metadata.json"),
        thermal_fps_events=thermal,
        recording_parity_valid=bool(parity),
    )


def _exif_block(meta: dict) -> dict:
    """Normalize nested EXIF from sidecar / splatpack stream metadata."""
    if not meta:
        return {}
    if "exif" in meta and isinstance(meta["exif"], dict):
        return meta["exif"]
    inner = meta.get("metadata", {})
    if isinstance(inner, dict):
        if "exif" in inner and isinstance(inner["exif"], dict):
            return inner["exif"]
        # Apple-style {Exif} key inside metadata
        for key in ("{Exif}", "Exif", "exif"):
            if key in inner and isinstance(inner[key], dict):
                return inner[key]
        if "{Exif}" in meta:
            return meta["{Exif}"]
    return {}


def _device_from_photo_stream(camera: str, stream: dict, root: str) -> DeviceInfo:
    """Build DeviceInfo from photo stream EXIF (35mm-equivalent FOV)."""
    from .intrinsics import device_from_exif_35mm

    meta = stream.get("metadata") or {}
    if not meta and stream.get("metadataFile"):
        side = os.path.join(root, stream["metadataFile"])
        if os.path.isfile(side):
            try:
                side_data = _read_json(side)
                meta = side_data.get("metadata", side_data)
            except (OSError, ValueError):
                meta = {}
    exif = _exif_block(meta if isinstance(meta, dict) else {})
    # Also accept top-level Pixel* on Apple {Exif}-less sidecars
    if not exif and isinstance(meta, dict):
        nested = meta.get("metadata", meta)
        if isinstance(nested, dict) and "{Exif}" in nested:
            exif = nested["{Exif}"]
    return device_from_exif_35mm(camera, exif)


def _photo_frame_from_stream(stream: dict, capture_id: str, root: str) -> PhotoFrame:
    extra = stream.get("extraMetadata", {}) or {}
    analysis = stream.get("imageAnalysis", {}) or {}
    cam = stream.get("camera") or extra.get("stream") or "wide"
    image_file = stream.get("imageFile") or stream.get("filename") or ""
    metadata_file = stream.get("metadataFile") or ""
    meta = stream.get("metadata") or {}
    exif = _exif_block(meta if isinstance(meta, dict) else {})
    w = int(exif.get("PixelXDimension") or 0)
    h = int(exif.get("PixelYDimension") or 0)
    # Orientation 6/8 swap display dims — keep EXIF pixel dims for intrinsics.
    f35 = _maybe_float(exif.get("FocalLenIn35mmFilm"))
    device = _device_from_photo_stream(cam, stream, root)
    if w <= 0:
        w = device.width
    if h <= 0:
        h = device.height
    return PhotoFrame(
        camera=cam,
        capture_id=capture_id,
        image_file=image_file,
        metadata_file=metadata_file,
        quality_score=_maybe_float(extra.get("qualityScore")),
        quality_band=str(extra.get("qualityBand", "")),
        blur_sharpness=_maybe_float(analysis.get("blurSharpnessScore")),
        width=w,
        height=h,
        focal_35mm=f35,
        device=device,
    )


def _load_photo_pack(root: str) -> PhotoPack:
    sp_path = os.path.join(root, "splatpack.json")
    ps_path = os.path.join(root, "photo_series.json")
    frames: list[PhotoFrame] = []
    folder_name = os.path.basename(root)
    pair_count = 0

    if os.path.isfile(sp_path):
        sp = _read_json(sp_path)
        folder_name = sp.get("folderName", folder_name)
        pair_count = int(sp.get("pairCount", 0))
        for pr in sp.get("pairs", []):
            cid = str(pr.get("captureId", ""))
            for st in pr.get("streams", []):
                frames.append(_photo_frame_from_stream(st, cid, root))
        if not pair_count:
            # Unique capture ids
            pair_count = len({f.capture_id for f in frames if f.capture_id}) or len(frames)

    if not frames and os.path.isfile(ps_path):
        ps = _read_json(ps_path)
        for cap in ps.get("captures", []):
            extra = cap.get("extraMetadata", {}) or {}
            cid = str(extra.get("captureId", cap.get("filename", "")))
            frames.append(_photo_frame_from_stream(cap, cid, root))
        pair_count = len(frames)

    return PhotoPack(
        root=root,
        folder_name=folder_name,
        capture_type=CaptureType.PHOTO_DUAL,
        frames=frames,
        pair_count=pair_count,
    )


def _load_lidar_pack(root: str) -> LidarPack:
    ps_path = os.path.join(root, "photo_series.json")
    sp_path = os.path.join(root, "splatpack.json")

    pairs: list[LidarPair] = []
    folder_name = os.path.basename(root)
    pair_count = 0

    if os.path.isfile(ps_path):
        ps = _read_json(ps_path)
        for cap in ps.get("captures", []):
            pairs.append(_lidar_pair_from_capture(cap))
        pair_count = len(pairs)
    elif os.path.isfile(sp_path):
        sp = _read_json(sp_path)
        folder_name = sp.get("folderName", folder_name)
        pair_count = int(sp.get("pairCount", 0))
        for pr in sp.get("pairs", []):
            for st in pr.get("streams", []):
                pairs.append(_lidar_pair_from_capture(st))

    if os.path.isfile(sp_path):
        try:
            folder_name = _read_json(sp_path).get("folderName", folder_name)
        except (OSError, ValueError):
            pass

    colmap_dir = os.path.join(root, "COLMAP_Text_Model")
    sensor_dir = os.path.join(root, "sensor_data")

    return LidarPack(
        root=root,
        folder_name=folder_name,
        capture_type=CaptureType.PHOTO_LIDAR_SINGLE,
        pair_count=pair_count or len(pairs),
        pairs=pairs,
        colmap_model_dir=colmap_dir if os.path.isdir(colmap_dir) else None,
        sensor_data_dir=sensor_dir if os.path.isdir(sensor_dir) else None,
    )


def _lidar_pair_from_capture(cap: dict) -> LidarPair:
    extra = cap.get("extraMetadata", {})
    aux = []
    for a in cap.get("auxiliaryOutputs", []):
        aux.append(
            AuxOutput(
                type=a.get("type", ""),
                filename=a.get("filename", ""),
                width=int(a.get("width", 0)),
                height=int(a.get("height", 0)),
                bytes_per_row=int(a.get("bytesPerRow", 0)),
                pixel_format=int(a.get("pixelFormat", 0)),
                units=a.get("units", ""),
                coordinate_system=a.get("coordinateSystem", ""),
            )
        )
    return LidarPair(
        camera=extra.get("stream", cap.get("camera", "wide")),
        image_file=cap.get("filename", cap.get("imageFile", "")),
        metadata_file=cap.get("metadataFile", ""),
        quality_score=_maybe_float(extra.get("qualityScore")),
        quality_band=extra.get("qualityBand", ""),
        capture_source=extra.get("captureSource", ""),
        depth_available=bool(extra.get("depthAvailable", False)),
        aux=aux,
    )


def load_lidar_frame_metadata(pack: LidarPack, pair: LidarPair) -> LidarPair:
    """Lazily fill in intrinsics/transform/resolution from sensor_data JSON."""
    if not pair.metadata_file or not pack.sensor_data_dir:
        return pair
    path = os.path.join(pack.sensor_data_dir, pair.metadata_file)
    if not os.path.isfile(path):
        return pair
    data = _read_json(path)
    cam = data.get("metadata", {}).get("camera", {})
    pair.intrinsics = [float(x) for x in cam.get("intrinsics", [])] or None
    pair.transform = [float(x) for x in cam.get("transform", [])] or None
    res = cam.get("imageResolution", {})
    pair.image_width = int(res.get("width", 0))
    pair.image_height = int(res.get("height", 0))
    return pair


def read_frame_timecodes(root: str) -> list[dict]:
    """Parse frame_timecodes.csv into a list of dict rows (best effort)."""
    path = os.path.join(root, "frame_timecodes.csv")
    if not os.path.isfile(path):
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows
