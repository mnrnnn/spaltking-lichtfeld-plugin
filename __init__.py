"""LichtFeld Studio plugin entry for SplatKing importer."""

from __future__ import annotations

import os
import sys

# Allow `import splatking...` (CLI + core) alongside `import splatking_importer...`.
_PLUGIN_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

import lichtfeld as lf

from .panels.main_panel import SplatKingImporterPanel
from .operators.prepare_ops import (
    SplatKingPrepareVideoOp,
    SplatKingPrepareLidarOp,
    SplatKingLoadLidarDatasetOp,
    SplatKingSubsampleCamerasOp,
)
from splatking.prefs import load_prefs, apply_tool_defaults, save_prefs

CLASSES = [
    SplatKingImporterPanel,
    SplatKingPrepareVideoOp,
    SplatKingPrepareLidarOp,
    SplatKingLoadLidarDatasetOp,
    SplatKingSubsampleCamerasOp,
]


def on_load():
    # Auto-detect ffmpeg/COLMAP once so the panel opens with usable paths.
    prefs = apply_tool_defaults(load_prefs())
    save_prefs(prefs)
    for cls in CLASSES:
        lf.register_class(cls)
    lf.log.info(
        f"splatking_importer loaded "
        f"(ffmpeg={prefs.get('ffmpeg_bin') or '?'}, "
        f"colmap={prefs.get('colmap_bin') or 'not set'})"
    )


def on_unload():
    for cls in reversed(CLASSES):
        lf.unregister_class(cls)
    lf.log.info("splatking_importer unloaded")
