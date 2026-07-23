"""Persisted plugin preferences."""

from __future__ import annotations

import json
import os
from typing import Any

PLUGIN_NAME = "splatking_importer"

DEFAULTS: dict[str, Any] = {
    "ffmpeg_bin": "",
    "colmap_bin": "",
    "vocab_tree_path": "",
    "video_keep_pct": 0.10,
    "video_resize": 1920,
    "video_blur_percentile": 0.15,
    "video_inject_intrinsics": True,
    "video_run_colmap": False,
    "video_lenses": "both",
    # COLMAP section (shared by after-prepare + Run COLMAP)
    "colmap_matcher": "sequential",
    "colmap_use_gpu": True,
    "colmap_max_image_size": 3200,
    "colmap_max_num_features": 8192,
    "colmap_seq_overlap": 10,
    "colmap_min_num_matches": 15,
    "lidar_confidence_min": 1,
    "cam_mode": "every_n",
    "cam_every_n": 2,
    "cam_random_pct": 0.5,
    "last_pack_path": "",
    "last_out_dir": "",
}


def _json_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, ".splatking_prefs.json")


def _lf_settings():
    try:
        import lichtfeld as lf

        return lf.plugins.settings(PLUGIN_NAME)
    except Exception:
        return None


def load_prefs() -> dict[str, Any]:
    prefs = dict(DEFAULTS)
    path = _json_path()
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                prefs.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
        except (OSError, ValueError):
            pass
    s = _lf_settings()
    if s is not None:
        for key in DEFAULTS:
            try:
                if hasattr(s, "get"):
                    val = s.get(key, None)
                elif hasattr(s, "__getitem__"):
                    val = s[key]
                else:
                    val = getattr(s, key, None)
                if val is not None and val != "":
                    prefs[key] = val
            except Exception:
                continue
    return prefs


def save_prefs(prefs: dict[str, Any]) -> None:
    merged = dict(DEFAULTS)
    merged.update({k: prefs[k] for k in DEFAULTS if k in prefs})
    path = _json_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
    except OSError:
        pass
    s = _lf_settings()
    if s is not None:
        for key, val in merged.items():
            try:
                if hasattr(s, "set"):
                    s.set(key, val)
                elif hasattr(s, "__setitem__"):
                    s[key] = val
                else:
                    setattr(s, key, val)
            except Exception:
                continue


def apply_tool_defaults(prefs: dict[str, Any]) -> dict[str, Any]:
    from .paths import resolve_ffmpeg, resolve_colmap, refresh_process_path

    refresh_process_path()
    out = dict(prefs)
    ff = resolve_ffmpeg(out.get("ffmpeg_bin", ""))
    if ff.found:
        out["ffmpeg_bin"] = ff.path
    cm = resolve_colmap(out.get("colmap_bin", ""))
    if cm.found:
        out["colmap_bin"] = cm.path
    return out
