"""LichtFeld operators for SplatKing prepare / load / camera-select."""

from __future__ import annotations

import os
from typing import Any

from lfs_plugins.types import Operator
from lfs_plugins.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    BoolProperty,
    EnumProperty,
    PropSubtype,
)

# Module-level state (do NOT import the plugin package by name — install folder
# may be spaltking-lichtfeld-plugin rather than splatking_importer).
_SK_STATE: dict[str, Any] = {
    "pack_path": "",
    "out_dir": "",
    "last_report": "",
    "status": "Idle",
    "capture_type": "",
}


def _ops_state() -> dict[str, Any]:
    return _SK_STATE


def _tool_bins() -> tuple[str, str]:
    """Return (ffmpeg, colmap) from saved prefs with auto-detect fill-in."""
    from splatking.prefs import load_prefs, apply_tool_defaults

    prefs = apply_tool_defaults(load_prefs())
    return prefs.get("ffmpeg_bin") or "ffmpeg", prefs.get("colmap_bin") or "colmap"

class SplatKingPrepareVideoOp(Operator):
    label = "Prepare Video Dataset"
    description = (
        "Extract dual-lens frames, filter blur, inject per-camera PINHOLE "
        "intrinsics, and write COLMAP commands (incl. vocab_tree)."
    )
    options = {"BLOCKING"}

    pack_path = StringProperty(name="Pack", subtype=PropSubtype.DIR_PATH, default="")
    out_dir = StringProperty(name="Output", subtype=PropSubtype.DIR_PATH, default="")
    cameras = StringProperty(name="Cameras", default="wide,ultra")
    stride = IntProperty(name="Stride", default=15, min=1, max=120)
    max_frames = IntProperty(name="Max frames/lens", default=200, min=0)
    resize = IntProperty(name="Resize width", default=1920, min=0)
    blur_percentile = FloatProperty(name="Blur drop %", default=0.15, min=0.0, max=0.9)
    matcher = EnumProperty(
        name="Matcher",
        items=[
            ("sequential", "Sequential", "Good for video sequences"),
            ("exhaustive", "Exhaustive", "O(N²) — small sets only"),
            ("vocab_tree", "Vocab Tree", "Required for thousands of images"),
        ],
        default="sequential",
    )
    vocab_tree = StringProperty(name="Vocab tree", subtype=PropSubtype.FILE_PATH, default="")
    inject_intrinsics = BoolProperty(name="Inject known intrinsics", default=True)
    run_colmap = BoolProperty(name="Run COLMAP now", default=False)
    colmap_bin = StringProperty(name="COLMAP binary", default="")
    ffmpeg_bin = StringProperty(name="ffmpeg binary", default="")

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def execute(self, context) -> set:
        import lichtfeld as lf
        from splatking.pack import load_pack, detect_capture_type, CaptureType
        from splatking.video_pipeline import VideoPrepOptions, prepare_video_dataset

        state = _ops_state()
        pack_path = self.pack_path or state.get("pack_path", "")
        out_dir = self.out_dir or state.get("out_dir", "")
        if not pack_path or not os.path.isdir(pack_path):
            state["status"] = "Error: pick a SplatKing pack folder"
            lf.log.error(state["status"])
            return {"CANCELLED"}
        if not out_dir:
            out_dir = os.path.join(pack_path, "_lichtfeld_prep")
        ct = detect_capture_type(pack_path)
        if ct != CaptureType.VIDEO_DUAL:
            state["status"] = f"Error: expected video_dual, got {ct.value}"
            lf.log.error(state["status"])
            return {"CANCELLED"}

        state["status"] = "Preparing video dataset..."
        lf.log.info(state["status"])
        pack = load_pack(pack_path)
        ff_default, cm_default = _tool_bins()
        opts = VideoPrepOptions(
            out_dir=out_dir,
            cameras=[c.strip() for c in self.cameras.split(",") if c.strip()],
            stride=int(self.stride),
            max_frames_per_lens=int(self.max_frames),
            resize_width=int(self.resize),
            blur_percentile=float(self.blur_percentile),
            matcher=str(self.matcher),
            vocab_tree_path=str(self.vocab_tree),
            inject_intrinsics=bool(self.inject_intrinsics),
            colmap_bin=str(self.colmap_bin) or cm_default,
            ffmpeg_bin=str(self.ffmpeg_bin) or ff_default,
            run_colmap=bool(self.run_colmap),
        )
        try:
            result = prepare_video_dataset(pack, opts)
        except Exception as e:
            state["status"] = f"Error: {e}"
            lf.log.error(state["status"])
            return {"CANCELLED"}

        kept = {k: len(v) for k, v in result.extracted.items()}
        state["out_dir"] = result.out_dir
        state["last_report"] = result.report_path
        state["capture_type"] = "video_dual"
        state["status"] = (
            f"Video ready: kept={kept}; injected {len(result.cameras)} cameras; "
            f"see {result.report_path}"
        )
        lf.log.info(state["status"])
        return {"FINISHED"}


class SplatKingPrepareLidarOp(Operator):
    label = "Prepare LiDAR Depth"
    description = (
        "Decode ARKit depth/confidence bins into metric depth maps for "
        "LichtFeld Depth Loss. COLMAP model is already on-device."
    )
    options = {"BLOCKING"}

    pack_path = StringProperty(name="Pack", subtype=PropSubtype.DIR_PATH, default="")
    out_dir = StringProperty(name="Output", subtype=PropSubtype.DIR_PATH, default="")
    confidence_min = IntProperty(name="Min confidence", default=1, min=0, max=2)
    depth_format = EnumProperty(
        name="Depth format",
        items=[
            ("npy", "NPY (float meters)", "float32 .npy"),
            ("png16", "PNG16 (mm)", "uint16 millimeters"),
            ("both", "Both", "Write both formats"),
        ],
        default="npy",
    )

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def execute(self, context) -> set:
        import lichtfeld as lf
        from splatking.pack import load_pack, detect_capture_type, CaptureType
        from splatking.lidar_pipeline import LidarPrepOptions, prepare_lidar_dataset

        state = _ops_state()
        pack_path = self.pack_path or state.get("pack_path", "")
        out_dir = self.out_dir or state.get("out_dir", "")
        if not pack_path or not os.path.isdir(pack_path):
            state["status"] = "Error: pick a SplatKing pack folder"
            lf.log.error(state["status"])
            return {"CANCELLED"}
        if not out_dir:
            out_dir = os.path.join(pack_path, "_lichtfeld_prep")
        ct = detect_capture_type(pack_path)
        if ct != CaptureType.PHOTO_LIDAR_SINGLE:
            state["status"] = f"Error: expected photo_lidar_single, got {ct.value}"
            lf.log.error(state["status"])
            return {"CANCELLED"}

        pack = load_pack(pack_path)
        opts = LidarPrepOptions(
            out_dir=out_dir,
            confidence_min=int(self.confidence_min),
            depth_format=str(self.depth_format),
        )
        try:
            result = prepare_lidar_dataset(pack, opts)
        except Exception as e:
            state["status"] = f"Error: {e}"
            lf.log.error(state["status"])
            return {"CANCELLED"}

        state["out_dir"] = out_dir
        state["capture_type"] = "photo_lidar_single"
        state["last_report"] = result.manifest_path or ""
        state["status"] = (
            f"LiDAR ready: {result.registered_images} images, "
            f"{result.num_points3d} points, {result.depth_written} depth maps"
        )
        lf.log.info(state["status"])
        return {"FINISHED"}


class SplatKingLoadLidarDatasetOp(Operator):
    label = "Load LiDAR COLMAP into Scene"
    description = "lf.load_file on COLMAP_Text_Model (skips SfM entirely)."
    options = {"BLOCKING"}

    pack_path = StringProperty(name="Pack", subtype=PropSubtype.DIR_PATH, default="")

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def execute(self, context) -> set:
        import lichtfeld as lf
        from splatking.pack import load_pack, detect_capture_type, CaptureType

        state = _ops_state()
        pack_path = self.pack_path or state.get("pack_path", "")
        if not pack_path:
            state["status"] = "Error: pick a SplatKing pack folder"
            return {"CANCELLED"}
        ct = detect_capture_type(pack_path)
        if ct != CaptureType.PHOTO_LIDAR_SINGLE:
            state["status"] = f"Error: expected photo_lidar_single, got {ct.value}"
            return {"CANCELLED"}
        pack = load_pack(pack_path)
        if not pack.colmap_model_dir or not os.path.isdir(pack.colmap_model_dir):
            state["status"] = "Error: COLMAP_Text_Model missing"
            return {"CANCELLED"}
        try:
            lf.load_file(pack.colmap_model_dir, is_dataset=True)
        except Exception as e:
            state["status"] = f"Error loading dataset: {e}"
            lf.log.error(state["status"])
            return {"CANCELLED"}
        state["status"] = f"Loaded dataset: {pack.colmap_model_dir}"
        lf.log.info(state["status"])
        return {"FINISHED"}


class SplatKingSubsampleCamerasOp(Operator):
    label = "Subsample Training Cameras"
    description = "Keep sparse geometry; thin cameras for VRAM (every-N / random %)."
    options = {"BLOCKING"}

    sparse_dir = StringProperty(name="Sparse dir", subtype=PropSubtype.DIR_PATH, default="")
    out_dir = StringProperty(name="Output sparse", subtype=PropSubtype.DIR_PATH, default="")
    mode = EnumProperty(
        name="Mode",
        items=[
            ("every_n", "Every N", "Keep every Nth image"),
            ("random_pct", "Random %", "Keep a random percentage"),
            ("keep_all", "Keep all", "No thinning"),
        ],
        default="every_n",
    )
    every_n = IntProperty(name="Every N", default=2, min=1, max=100)
    random_pct = FloatProperty(name="Random fraction", default=0.5, min=0.05, max=1.0)
    seed = IntProperty(name="Seed", default=42)

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def execute(self, context) -> set:
        import lichtfeld as lf
        from splatking.camera_select import CameraSelectOptions, write_training_subset

        state = _ops_state()
        sparse = self.sparse_dir
        out = self.out_dir
        if not sparse:
            # Try last prep out_dir / sparse/0 or LiDAR COLMAP.
            cand = os.path.join(state.get("out_dir", ""), "sparse", "0")
            if os.path.isdir(cand):
                sparse = cand
        if not sparse or not os.path.isdir(sparse):
            state["status"] = "Error: set sparse_dir (COLMAP sparse/0)"
            return {"CANCELLED"}
        if not out:
            out = os.path.join(os.path.dirname(sparse.rstrip("/\\")), "0_train")
        opts = CameraSelectOptions(
            mode=str(self.mode),
            every_n=int(self.every_n),
            random_pct=float(self.random_pct),
            seed=int(self.seed),
        )
        try:
            summary = write_training_subset(sparse, out, opts)
        except Exception as e:
            state["status"] = f"Error: {e}"
            return {"CANCELLED"}
        state["status"] = (
            f"Training cameras: {summary['source_images']} → "
            f"{summary['training_images']} ({summary['mode']})"
        )
        lf.log.info(state["status"])
        return {"FINISHED"}
