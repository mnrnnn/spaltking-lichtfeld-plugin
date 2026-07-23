"""LichtFeld Studio plugin entry for SplatKing importer."""

from __future__ import annotations

import os
import sys
import traceback

# Allow `import splatking...` whether the install folder is splatking_importer
# or the GitHub clone name (spaltking-lichtfeld-plugin).
_PLUGIN_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

import lichtfeld as lf

CLASSES = []


def on_load():
    global CLASSES
    try:
        from splatking.prefs import load_prefs, apply_tool_defaults, save_prefs

        prefs = apply_tool_defaults(load_prefs())
        save_prefs(prefs)
        ff = prefs.get("ffmpeg_bin") or "?"
        cm = prefs.get("colmap_bin") or "not set"
    except Exception as e:
        ff, cm = "?", "not set"
        lf.log.warn(f"splatking_importer: prefs init skipped: {e}")

    from .panels.main_panel import SplatKingImporterPanel
    from .operators.prepare_ops import (
        SplatKingPrepareVideoOp,
        SplatKingPrepareLidarOp,
        SplatKingLoadLidarDatasetOp,
        SplatKingSubsampleCamerasOp,
    )

    CLASSES = [
        SplatKingImporterPanel,
        SplatKingPrepareVideoOp,
        SplatKingPrepareLidarOp,
        SplatKingLoadLidarDatasetOp,
        SplatKingSubsampleCamerasOp,
    ]
    for cls in CLASSES:
        lf.register_class(cls)
    lf.log.info(f"splatking_importer loaded (ffmpeg={ff}, colmap={cm})")


def on_unload():
    for cls in reversed(CLASSES):
        try:
            lf.unregister_class(cls)
        except Exception:
            lf.log.warn(f"splatking_importer: unload failed for {cls}: {traceback.format_exc()}")
    CLASSES.clear()
    lf.log.info("splatking_importer unloaded")
