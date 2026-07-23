"""SplatKing pack parsing and reconstruction-prep core.

This subpackage is intentionally free of any LichtFeld (`lichtfeld`) or heavy
third-party imports at module load time, so it can be unit-tested and run as a
standalone CLI outside the LichtFeld GUI. numpy / OpenCV are imported lazily.
"""

from .pack import (
    CaptureType,
    DeviceInfo,
    VideoStream,
    VideoPack,
    PhotoFrame,
    PhotoPack,
    LidarPair,
    LidarPack,
    load_pack,
    detect_capture_type,
    default_out_dir,
    human_capture_label,
)

__all__ = [
    "CaptureType",
    "DeviceInfo",
    "VideoStream",
    "VideoPack",
    "PhotoFrame",
    "PhotoPack",
    "LidarPair",
    "LidarPack",
    "load_pack",
    "detect_capture_type",
    "default_out_dir",
    "human_capture_label",
]
