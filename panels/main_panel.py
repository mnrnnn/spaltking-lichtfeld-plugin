"""Main SplatKing Importer panel — capture-aware UX with Browse + deps install."""

from __future__ import annotations

import os
import re
import threading
import traceback

import lichtfeld as lf

from ..operators.prepare_ops import _ops_state


class SplatKingImporterPanel(lf.ui.Panel):
    id = "splatking_importer.main"
    label = "SplatKing"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 40

    # LichtFeld may construct panels without calling Python __init__.
    def _ensure_state(self):
        if getattr(self, "_sk_ready", False):
            return
        from splatking.prefs import load_prefs, apply_tool_defaults

        prefs = apply_tool_defaults(load_prefs())

        self._pack_path = prefs.get("last_pack_path", "") or ""
        self._out_dir = prefs.get("last_out_dir", "") or ""
        self._capture_type = ""
        self._info_lines = []
        self._estimate_lines = []
        self._warning_lines = []

        self._ffmpeg_bin = prefs.get("ffmpeg_bin", "") or ""
        self._colmap_bin = prefs.get("colmap_bin", "") or ""
        self._vocab_tree = prefs.get("vocab_tree_path", "") or ""
        self._ffmpeg_status = ""
        self._colmap_status = ""
        self._vocab_status = ""
        self._tools_forced_once = False
        self._colmap_prep_override = ""

        # Persist collapsing-header open state; only reset on capture-type change.
        self._sec_open = {
            "tools": False,
            "photo": False,
            "video": False,
            "colmap": False,
            "lidar": False,
            "train": False,
            "results": True,
            "status": True,
        }
        self._last_capture_type = None

        self._keep_pct = float(prefs.get("video_keep_pct", 0.10))
        self._resize = int(prefs.get("video_resize", 1920))
        self._blur_pct = float(prefs.get("video_blur_percentile", 0.15))
        self._lens_items = ["both", "wide", "ultra"]
        lenses = prefs.get("video_lenses", "both")
        self._lens_idx = self._lens_items.index(lenses) if lenses in self._lens_items else 0
        self._inject = bool(prefs.get("video_inject_intrinsics", True))
        self._run_colmap = bool(prefs.get("video_run_colmap", False))
        self._write_colmap_script = bool(prefs.get("write_colmap_script", False))

        self._matcher_items = ["sequential", "exhaustive", "vocab_tree"]
        matcher = prefs.get("colmap_matcher", "sequential")
        self._matcher_idx = (
            self._matcher_items.index(matcher) if matcher in self._matcher_items else 0
        )
        self._colmap_use_gpu = bool(prefs.get("colmap_use_gpu", True))
        self._colmap_max_image_size = int(prefs.get("colmap_max_image_size", 3200))
        self._colmap_max_num_features = int(prefs.get("colmap_max_num_features", 8192))
        self._colmap_seq_overlap = int(prefs.get("colmap_seq_overlap", 10))
        self._colmap_min_num_matches = int(prefs.get("colmap_min_num_matches", 15))
        self._dual_mode = bool(prefs.get("colmap_dual_mode", True))
        self._base_lens_items = ["ultra", "wide"]
        bl = prefs.get("colmap_base_lens", "ultra")
        self._base_lens_idx = (
            self._base_lens_items.index(bl) if bl in self._base_lens_items else 0
        )
        self._dual_method_items = ["auto", "registrator", "rig", "wide_only"]
        dm = prefs.get("colmap_dual_method", "auto")
        self._dual_method_idx = (
            self._dual_method_items.index(dm) if dm in self._dual_method_items else 0
        )

        self._confidence_min = int(prefs.get("lidar_confidence_min", 1))

        self._cam_modes = ["every_n", "random_pct", "keep_all"]
        cam_mode = prefs.get("cam_mode", "every_n")
        self._cam_mode_idx = (
            self._cam_modes.index(cam_mode) if cam_mode in self._cam_modes else 0
        )
        self._every_n = int(prefs.get("cam_every_n", 2))
        self._random_pct = float(prefs.get("cam_random_pct", 0.5))

        self._busy = False
        self._draw_error = ""
        self._gpu_hint = ""
        self._sk_ready = True

        try:
            self._refresh_tool_status()
            self._refresh_gpu_hint()
            if self._pack_path:
                self._refresh_info()
        except Exception as e:
            self._draw_error = f"init: {e}"

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def _colmap_settings(self):
        from splatking.video_pipeline import ColmapSettings

        return ColmapSettings(
            matcher=self._matcher_items[self._matcher_idx],
            vocab_tree_path=self._vocab_tree,
            use_gpu=bool(self._colmap_use_gpu),
            max_image_size=int(self._colmap_max_image_size),
            max_num_features=int(self._colmap_max_num_features),
            sequential_overlap=int(self._colmap_seq_overlap),
            min_num_matches=int(self._colmap_min_num_matches),
            dual_mode=bool(self._dual_mode),
            base_lens=self._base_lens_items[self._base_lens_idx],
            dual_method=self._dual_method_items[self._dual_method_idx],
        )

    # [1/4] COLMAP Feature Extractor (wide) (100/200, 50%)
    _COLMAP_BAR_RE = re.compile(
        r"^\[(\d+)/(\d+)\]\s+(.*?)(?:\s+\((\d+)/(\d+),\s*(\d+)%\))?\s*$"
    )

    @staticmethod
    def _short_progress_label(msg: str, limit: int = 96) -> str:
        """Overlay text for the progress bar."""
        text = (msg or "").strip().replace("\n", " ")
        if not text:
            return ""
        # Preferred COLMAP bar format — show in full (do not path-truncate).
        if text.startswith("[") and "COLMAP" in text:
            return text if len(text) <= 120 else text[:119] + "…"
        if text.startswith("COLMAP ["):
            if ".bat" not in text.lower() and ".exe" not in text.lower() and "--" not in text:
                return text
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    def _append_status_log(self, msg: str, *, limit: int = 48):
        """CLI-style Status log. Coalesce progress ticks for the same [i/n] step."""
        text = (msg or "").strip()
        if not text:
            return
        if len(text) > 220:
            text = text[:219] + "…"
        state = _ops_state()
        log = list(state.get("status_log") or [])
        if log and log[-1] == text:
            state["status"] = text
            return
        # Replace last line when same COLMAP step progress updates (CLI feel).
        bare = text.replace(" done", "").strip()
        m_new = self._COLMAP_BAR_RE.match(bare)
        if m_new and log:
            m_old = self._COLMAP_BAR_RE.match(log[-1].replace(" done", "").strip())
            if (
                m_old
                and m_old.group(1) == m_new.group(1)
                and m_old.group(2) == m_new.group(2)
                and m_old.group(3) == m_new.group(3)
            ):
                log[-1] = text
                state["status_log"] = log
                state["status"] = text
                return
        log.append(text)
        if len(log) > limit:
            log = log[-limit:]
        state["status_log"] = log
        state["status"] = text

    def _append_result_line(self, msg: str, *, limit: int = 32):
        text = (msg or "").strip()
        if not text:
            return
        if len(text) > 260:
            text = text[:259] + "…"
        state = _ops_state()
        lines = list(state.get("result_lines") or [])
        if lines and lines[-1] == text:
            return
        lines.append(text)
        if len(lines) > limit:
            lines = lines[-limit:]
        state["result_lines"] = lines

    def _set_progress(self, frac: float, label: str = "", *, log: bool = True):
        state = _ops_state()
        state["progress"] = min(max(float(frac), 0.0), 1.0)
        if label:
            state["progress_label"] = self._short_progress_label(label)
            if log:
                self._append_status_log(label)

    def _header(self, ui, title: str, key: str) -> bool:
        """Draw collapsing header and keep open state across frames / combo edits."""
        opened = bool(
            ui.collapsing_header(title, default_open=bool(self._sec_open.get(key, False)))
        )
        self._sec_open[key] = opened
        return opened

    def _sync_sec_open_for_type(self):
        ct = self._capture_type or ""
        if ct == self._last_capture_type:
            return
        self._last_capture_type = ct
        self._sec_open["photo"] = ct == "photo_dual"
        self._sec_open["video"] = ct == "video_dual"
        self._sec_open["lidar"] = ct == "photo_lidar_single"

    def _start_bg(self, work, *, label: str = "Working..."):
        """Run blocking Prepare/COLMAP/install on a daemon thread so draw() can update progress."""
        if self._busy:
            return
        self._busy = True
        state = _ops_state()
        state["status_log"] = []
        state["result_lines"] = []
        self._set_progress(0.02, label)

        def runner():
            try:
                work()
            except Exception as e:
                self._set_progress(0.0, "")
                self._set_status(f"Job failed: {e}", error=True)
            finally:
                self._busy = False

        threading.Thread(target=runner, daemon=True).start()

    def _hint(self, ui, text: str):
        fn = getattr(ui, "text_wrapped", None) or getattr(ui, "text_disabled", None)
        if callable(fn):
            fn(text)
        else:
            ui.label(text)

    def _find_tool_in_dir(self, directory: str, names: tuple[str, ...]) -> str:
        if not directory or not os.path.isdir(directory):
            return ""
        for name in names:
            cand = os.path.join(directory, name)
            if os.path.isfile(cand):
                return cand
        for root, dirs, files in os.walk(directory):
            for name in names:
                if name in files:
                    return os.path.join(root, name)
            depth = root[len(directory) :].count(os.sep)
            if depth >= 3:
                dirs.clear()
        return ""

    def _find_bin_in_dir(self, directory: str) -> str:
        if not directory or not os.path.isdir(directory):
            return ""
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(".bin"):
                    return os.path.join(root, f)
            depth = root[len(directory) :].count(os.sep)
            if depth >= 2:
                dirs.clear()
        return ""

    def _browse_tool_binary(self, kind: str) -> str:
        """Prefer file dialog; else folder dialog then search for known binary names."""
        start = {
            "ffmpeg": os.path.dirname(self._ffmpeg_bin) if self._ffmpeg_bin else "",
            "colmap": os.path.dirname(self._colmap_bin) if self._colmap_bin else "",
            "vocab": os.path.dirname(self._vocab_tree) if self._vocab_tree else "",
        }.get(kind, "")
        picked = self._browse_file(start)
        if picked and os.path.isfile(picked):
            return picked
        folder = self._browse_folder(f"Select folder containing {kind}", start)
        if not folder:
            return ""
        if kind == "vocab":
            return self._find_bin_in_dir(folder)
        names = {
            "ffmpeg": ("ffmpeg.exe", "ffmpeg"),
            "colmap": ("COLMAP.bat", "colmap.exe", "colmap"),
        }.get(kind, ())
        return self._find_tool_in_dir(folder, names) if names else ""

    def _current_prefs(self) -> dict:
        return {
            "ffmpeg_bin": self._ffmpeg_bin,
            "colmap_bin": self._colmap_bin,
            "vocab_tree_path": self._vocab_tree,
            "video_keep_pct": self._keep_pct,
            "video_resize": self._resize,
            "video_blur_percentile": self._blur_pct,
            "video_inject_intrinsics": self._inject,
            "video_run_colmap": self._run_colmap,
            "video_lenses": self._lens_items[self._lens_idx],
            "write_colmap_script": self._write_colmap_script,
            "colmap_matcher": self._matcher_items[self._matcher_idx],
            "colmap_use_gpu": self._colmap_use_gpu,
            "colmap_max_image_size": self._colmap_max_image_size,
            "colmap_max_num_features": self._colmap_max_num_features,
            "colmap_seq_overlap": self._colmap_seq_overlap,
            "colmap_min_num_matches": self._colmap_min_num_matches,
            "colmap_dual_mode": self._dual_mode,
            "colmap_base_lens": self._base_lens_items[self._base_lens_idx],
            "colmap_dual_method": self._dual_method_items[self._dual_method_idx],
            "lidar_confidence_min": self._confidence_min,
            "cam_mode": self._cam_modes[self._cam_mode_idx],
            "cam_every_n": self._every_n,
            "cam_random_pct": self._random_pct,
            "last_pack_path": self._pack_path,
            "last_out_dir": self._out_dir,
        }

    def _save_prefs(self):
        from splatking.prefs import save_prefs

        save_prefs(self._current_prefs())
        state = _ops_state()
        state["status"] = "Preferences saved."

    def _refresh_tool_status(self):
        from splatking.paths import resolve_ffmpeg, resolve_colmap, refresh_process_path

        refresh_process_path()
        self._sanitize_tool_paths()
        ff = resolve_ffmpeg(self._ffmpeg_bin)
        cm = resolve_colmap(self._colmap_bin)
        if ff.found:
            self._ffmpeg_bin = ff.path
            self._ffmpeg_status = f"OK ({ff.source})" + (f" — {ff.version}" if ff.version else "")
        else:
            self._ffmpeg_status = "Not found — Install missing or Browse to ffmpeg.exe"
        if cm.found:
            self._colmap_bin = cm.path
            self._colmap_status = f"OK ({cm.source})" + (f" — {cm.version}" if cm.version else "")
        else:
            self._colmap_status = (
                "Not found — Install missing or Browse under Tools"
            )
        if self._vocab_tree and os.path.isfile(self._vocab_tree):
            self._vocab_status = f"OK — {os.path.basename(self._vocab_tree)}"
        else:
            self._vocab_status = "Not set — Download vocab tree or Browse to a .bin"

    def _refresh_gpu_hint(self):
        self._gpu_hint = ""
        try:
            from lfs_plugins.utils import get_gpu_memory

            used = int(get_gpu_memory())
            if used > 0:
                from splatking.paths import format_bytes

                self._gpu_hint = f"GPU memory in use (hint): {format_bytes(used)}"
        except Exception:
            try:
                import lichtfeld as _lf

                if hasattr(_lf, "total_gpu_bytes"):
                    used = int(_lf.total_gpu_bytes())
                    if used > 0:
                        from splatking.paths import format_bytes

                        self._gpu_hint = f"GPU memory in use (hint): {format_bytes(used)}"
            except Exception:
                self._gpu_hint = ""

    def _detect_tools(self):
        from splatking.prefs import apply_tool_defaults

        detected = apply_tool_defaults({
            **self._current_prefs(),
            "ffmpeg_bin": "",
            "colmap_bin": "",
        })
        self._ffmpeg_bin = detected.get("ffmpeg_bin", "") or self._ffmpeg_bin
        self._colmap_bin = detected.get("colmap_bin", "") or self._colmap_bin
        self._refresh_tool_status()
        self._save_prefs()

    def _install_missing(self):
        from splatking.deps import install_missing_tools

        if self._busy:
            return

        def work():
            result = install_missing_tools(
                self._ffmpeg_bin,
                self._colmap_bin,
                on_progress=self._on_progress,
            )
            if result.ffmpeg_path:
                self._ffmpeg_bin = result.ffmpeg_path
            if result.colmap_path:
                self._colmap_bin = result.colmap_path
            self._refresh_tool_status()
            self._save_prefs()
            self._set_progress(1.0 if result.ok else 0.0, result.message)
            self._set_status(result.message, error=not result.ok)

        self._start_bg(work, label="Installing tools...")

    def _download_vocab(self):
        from splatking.deps import download_vocab_tree

        if self._busy:
            return

        def work():
            result = download_vocab_tree(on_progress=self._on_progress)
            if result.ok and result.vocab_path:
                self._vocab_tree = result.vocab_path
                self._save_prefs()
            self._refresh_tool_status()
            self._set_progress(1.0 if result.ok else 0.0, result.message)
            self._set_status(result.message, error=not result.ok)

        self._start_bg(work, label="Downloading vocab tree...")

    def _set_status(self, msg: str, error: bool = False):
        self._append_status_log(msg)
        if error:
            lf.log.error(msg)
        else:
            lf.log.info(msg)

    def _on_progress(
        self,
        msg: str,
        status: bool = True,
        frac: float | None = None,
        result: bool = False,
    ):
        state = _ops_state()
        if result:
            self._append_result_line(msg)
            if frac is not None:
                state["progress"] = min(max(float(frac), 0.0), 1.0)
            return
        if frac is not None:
            self._set_progress(float(frac), msg, log=bool(status))
            return
        if self._busy:
            cur = float(state.get("progress", 0.0))
            if cur < 0.95:
                state["progress"] = min(cur + 0.02, 0.95)
            state["progress_label"] = self._short_progress_label(msg)
        if status:
            self._append_status_log(msg)

    def _stage_progress(
        self,
        msg: str,
        status: bool = True,
        frac: float | None = None,
        result: bool = False,
    ):
        text = (msg or "").strip()
        low = text.lower()

        if result:
            self._append_result_line(text)
            if frac is not None:
                _ops_state()["progress"] = min(max(float(frac), 0.0), 1.0)
            return

        # Preferred: pipeline-supplied overall fraction + bar label.
        if frac is not None and text.startswith("["):
            self._set_progress(float(frac), text, log=bool(status))
            return

        # [1/4] COLMAP Feature Extractor (wide) (100/200, 50%)
        m = self._COLMAP_BAR_RE.match(text.replace(" done", "").strip())
        if m:
            i = max(int(m.group(1)), 1)
            n = max(int(m.group(2)), 1)
            cur_s, tot_s = m.group(4), m.group(5)
            if cur_s and tot_s:
                cur, total = int(cur_s), max(int(tot_s), 1)
                overall = (i - 1) / n + (cur / total) / n
                # CLI Status: coalesce ticks for same step; always update bar.
                self._set_progress(overall, text, log=bool(status))
            else:
                self._set_progress((i - 1) / n, text, log=bool(status))
            return

        # Legacy: COLMAP [i/n] ...
        m2 = re.match(r"COLMAP \[(\d+)/(\d+)\]", text)
        if m2:
            i = max(int(m2.group(1)), 1)
            n = max(int(m2.group(2)), 1)
            self._set_progress((i - 1) / n, text, log=bool(status))
            return

        if "extracting" in low or "ffmpeg" in low or "copying" in low:
            self._set_progress(0.2, text)
        elif "scoring" in low or "sharpness" in low:
            self._set_progress(0.45, text)
        elif "wrote colmap" in low:
            self._set_progress(0.72, text)
        elif "running colmap" in low:
            self._set_progress(0.75, text)
        else:
            self._on_progress(text, status=status, frac=frac)

    def _browse_folder(self, title: str, start: str = "") -> str:
        try:
            path = lf.ui.open_folder_dialog(title=title, start_dir=start or "")
            return path or ""
        except Exception as e:
            self._set_status(f"Folder dialog failed: {e}", error=True)
            return ""

    def _browse_file(self, start: str = "") -> str:
        for name in ("open_file_dialog", "open_json_file_dialog"):
            fn = getattr(lf.ui, name, None)
            if callable(fn):
                try:
                    if name == "open_file_dialog":
                        return fn(title="Select file", start_dir=start or "") or ""
                    return fn() or ""
                except TypeError:
                    try:
                        return fn() or ""
                    except Exception:
                        continue
                except Exception:
                    continue
        return ""

    def _selected_cameras(self) -> list[str]:
        lens = self._lens_items[self._lens_idx]
        if lens == "wide":
            return ["wide"]
        if lens == "ultra":
            return ["ultra"]
        return ["wide", "ultra"]

    def _default_out_preview(self) -> str:
        from splatking.pack import default_out_dir

        if not self._pack_path:
            return ""
        return default_out_dir(self._pack_path, self._capture_type or "prep")

    def _ensure_out_dir(self) -> str:
        from splatking.pack import default_out_dir

        if self._out_dir:
            return self._out_dir
        return default_out_dir(self._pack_path, self._capture_type or "prep")

    def _prep_dir_markers(self, directory: str) -> bool:
        if not directory or not os.path.isdir(directory):
            return False
        for name in ("splatking_prep_report.json", "run_colmap.bat"):
            if os.path.isfile(os.path.join(directory, name)):
                return True
        return False

    def _resolve_prep_dir(self) -> str:
        state = _ops_state()
        if self._colmap_prep_override and os.path.isdir(self._colmap_prep_override):
            return self._colmap_prep_override
        if self._out_dir and self._prep_dir_markers(self._out_dir):
            return self._out_dir
        if self._pack_path:
            from splatking.pack import default_out_dir

            default = default_out_dir(self._pack_path, self._capture_type or "prep")
            if default:
                return default
        return state.get("out_dir", "") or self._out_dir or ""

    def _refresh_info(self):
        from splatking.paths import format_bytes
        from splatking.pack import (
            detect_capture_type,
            load_pack,
            CaptureType,
            human_capture_label,
        )
        from splatking.intrinsics import colmap_camera_from_device

        self._info_lines = []
        self._estimate_lines = []
        self._warning_lines = []
        self._capture_type = ""
        if not self._pack_path or not os.path.isdir(self._pack_path):
            return
        try:
            ct = detect_capture_type(self._pack_path)
            self._capture_type = ct.value
            pack = load_pack(self._pack_path)
            label = human_capture_label(ct)
            self._info_lines.append(f"Detected: {label} ({ct.value})")
            if ct == CaptureType.VIDEO_DUAL:
                self._info_lines.append(f"{pack.folder_name}")
                self._info_lines.append(
                    f"Parity: {'OK' if pack.recording_parity_valid else 'MISMATCH'}"
                )
                for s in pack.streams:
                    cam = colmap_camera_from_device(s.device, 1)
                    vpath = os.path.join(pack.root, s.raw_video_file)
                    try:
                        sz = format_bytes(os.path.getsize(vpath)) if os.path.isfile(vpath) else "?"
                    except OSError:
                        sz = "?"
                    self._info_lines.append(
                        f"{s.camera}: {s.frame_count} frames · "
                        f"{s.device.width}x{s.device.height} · "
                        f"FOV {s.device.field_of_view:.1f} deg · fx={cam.params[0]:.0f} · {sz}"
                    )
                if pack.thermal_fps_events:
                    self._info_lines.append(
                        f"Thermal throttle events: {len(pack.thermal_fps_events)}"
                    )
                self._update_estimate(pack)
            elif ct == CaptureType.PHOTO_DUAL:
                self._info_lines.append(f"{pack.folder_name}")
                self._info_lines.append(
                    f"{pack.pair_count} pairs · {len(pack.frames)} stills"
                )
                for cam in pack.cameras:
                    dev = pack.representative_device(cam)
                    n = len(pack.frames_for(cam))
                    if dev:
                        cc = colmap_camera_from_device(dev, 1)
                        self._info_lines.append(
                            f"{cam}: {n} images · {dev.width}x{dev.height} · "
                            f"FOV {dev.field_of_view:.1f} deg · fx={cc.params[0]:.0f}"
                        )
                    else:
                        self._info_lines.append(f"{cam}: {n} images")
            elif ct == CaptureType.PHOTO_LIDAR_SINGLE:
                self._info_lines.append(f"{pack.folder_name}")
                self._info_lines.append(f"{pack.pair_count} frames (on-device COLMAP)")
                self._info_lines.append(
                    f"COLMAP model: {'ready' if pack.has_colmap_model else 'missing'}"
                )
                depth_n = sum(1 for p in pack.pairs if p.depth_available)
                self._info_lines.append(f"Depth maps available: {depth_n}/{len(pack.pairs)}")
                qs = [p.quality_score for p in pack.pairs if p.quality_score is not None]
                if qs:
                    self._info_lines.append(
                        f"Quality score: {min(qs):.2f}-{max(qs):.2f} (mean {sum(qs)/len(qs):.2f})"
                    )
            else:
                self._info_lines.append("Not a SplatKing pack (need splatpack.json)")
        except Exception as e:
            self._info_lines.append(f"Parse error: {e}")

    def _update_estimate(self, pack=None):
        from splatking.paths import format_bytes
        from splatking.video_pipeline import estimate_extract

        self._estimate_lines = []
        self._warning_lines = []
        if self._capture_type != "video_dual":
            return
        try:
            from splatking.pack import load_pack

            if pack is None:
                if not self._pack_path:
                    return
                pack = load_pack(self._pack_path)
            est = estimate_extract(
                pack, self._selected_cameras(), float(self._keep_pct)
            )
            parts = [
                f"{cam}: {est.source_frames.get(cam, 0)} -> {est.planned_frames.get(cam, 0)}"
                for cam in est.cameras
            ]
            self._estimate_lines.append(
                f"Will extract ~{est.total_planned} frames total ({', '.join(parts)})"
            )
            self._estimate_lines.append(
                f"Video payload: {format_bytes(est.total_video_bytes)} · "
                f"keep={self._keep_pct:.0%} (step={est.step}) · "
                f"resize={self._resize or 'native'}"
            )
            suggested = est.suggested_matcher
            current = self._matcher_items[self._matcher_idx]
            if suggested != current:
                self._estimate_lines.append(
                    f"Suggested matcher: {suggested} (Run COLMAP section: {current})"
                )
            self._warning_lines = list(est.warnings)
        except Exception as e:
            self._warning_lines = [f"Estimate failed: {e}"]

    def _colmap_binary_ok(self) -> bool:
        return "OK" in self._colmap_status

    def _vocab_ok_for_matcher(self) -> bool:
        if self._matcher_items[self._matcher_idx] != "vocab_tree":
            return True
        return bool(self._vocab_tree and os.path.isfile(self._vocab_tree))

    @staticmethod
    def _looks_like_binary(path: str) -> bool:
        if not path or not path.strip():
            return False
        p = path.strip().strip('"')
        low = p.lower()
        if low.endswith((".exe", ".bat", ".cmd", ".bin")):
            return True
        if os.path.sep not in p and ":" not in p:
            return True
        return False

    def _sanitize_tool_paths(self):
        """Drop values that clearly aren't binaries (ImGui ID collisions used to
        write Output/..._prep into the COLMAP field)."""
        for attr in ("_ffmpeg_bin", "_colmap_bin", "_vocab_tree"):
            val = getattr(self, attr, "") or ""
            if not val:
                continue
            if "Output" in val.replace("\\", "/") and val.lower().endswith(
                ("_prep", "video_prep", "photo_prep", "lidar_prep")
            ):
                setattr(self, attr, "")
                continue
            if os.path.isdir(val) and not self._looks_like_binary(val):
                setattr(self, attr, "")

    def _run_prepare_video(self):
        from splatking.pack import load_pack, detect_capture_type, CaptureType
        from splatking.video_pipeline import VideoPrepOptions, prepare_video_dataset
        from splatking.paths import resolve_ffmpeg, resolve_colmap, format_bytes
        from splatking.video_pipeline import estimate_extract

        state = _ops_state()
        if self._busy:
            return
        pack_path = self._pack_path
        if not pack_path or not os.path.isdir(pack_path):
            self._set_status("Browse a SplatKing video pack folder first.", error=True)
            return

        ff = resolve_ffmpeg(self._ffmpeg_bin)
        if not ff.found:
            self._set_status(
                "ffmpeg not found. Use Tools → Install missing or Browse.",
                error=True,
            )
            self._sec_open["tools"] = True
            return
        self._ffmpeg_bin = ff.path

        if self._run_colmap:
            cm = resolve_colmap(self._colmap_bin)
            if not cm.found:
                self._set_status(
                    "COLMAP not found. Uncheck 'Run COLMAP after prepare', or Install missing.",
                    error=True,
                )
                self._sec_open["tools"] = True
                return
            self._colmap_bin = cm.path

        if not self._vocab_ok_for_matcher():
            self._set_status(
                "vocab_tree matcher needs a vocab tree — Download or Browse under Tools.",
                error=True,
            )
            self._sec_open["tools"] = True
            return

        ct = detect_capture_type(pack_path)
        if ct != CaptureType.VIDEO_DUAL:
            self._set_status(f"Expected video_dual pack, got {ct.value}.", error=True)
            return

        out_dir = self._ensure_out_dir()
        state["pack_path"] = pack_path
        state["out_dir"] = out_dir
        self._save_prefs()

        cameras = self._selected_cameras()
        keep_pct = float(self._keep_pct)
        resize = int(self._resize)
        blur = float(self._blur_pct)
        inject = bool(self._inject)
        run_cm = bool(self._run_colmap)
        write_script = bool(self._write_colmap_script)
        colmap_bin = self._colmap_bin or "colmap"
        ffmpeg_bin = self._ffmpeg_bin or "ffmpeg"
        colmap_settings = self._colmap_settings()

        def work():
            pack = load_pack(pack_path)
            est = estimate_extract(pack, cameras, keep_pct)
            self._set_progress(
                0.05,
                f"Decoding ~{est.total_planned} frames from "
                f"{format_bytes(est.total_video_bytes)} of video...",
            )
            result = prepare_video_dataset(
                pack,
                VideoPrepOptions(
                    out_dir=out_dir,
                    cameras=cameras,
                    keep_pct=keep_pct,
                    resize_width=resize,
                    blur_percentile=blur,
                    inject_intrinsics=inject,
                    colmap_bin=colmap_bin,
                    ffmpeg_bin=ffmpeg_bin,
                    run_colmap=run_cm,
                    write_colmap_script=write_script,
                    colmap=colmap_settings,
                ),
                on_progress=self._stage_progress,
            )
            kept = {k: len(v) for k, v in result.extracted.items()}
            dropped = {k: len(v) for k, v in result.rejected.items()}
            state["last_report"] = result.report_path
            state["capture_type"] = "video_dual"
            self._set_progress(1.0, "Video prepare finished.")
            if run_cm:
                next_hint = "COLMAP ran."
            elif write_script:
                next_hint = f"Next: run {os.path.join(result.out_dir, 'run_colmap.bat')}"
            else:
                next_hint = "Next: Run COLMAP section."
            self._set_status(
                f"Done. Kept {kept} (blur-dropped {dropped}). "
                f"Intrinsics injected for {len(result.cameras)} cameras. "
                f"{next_hint}"
            )

        self._start_bg(work, label="Preparing video dataset...")

    def _run_prepare_photo(self):
        from splatking.pack import load_pack, detect_capture_type, CaptureType
        from splatking.photo_pipeline import PhotoPrepOptions, prepare_photo_dataset
        from splatking.paths import resolve_colmap

        state = _ops_state()
        if self._busy:
            return
        pack_path = self._pack_path
        if not pack_path or not os.path.isdir(pack_path):
            self._set_status("Browse a SplatKing photo pack folder first.", error=True)
            return
        ct = detect_capture_type(pack_path)
        if ct != CaptureType.PHOTO_DUAL:
            self._set_status(f"Expected photo_dual pack, got {ct.value}.", error=True)
            return

        if self._run_colmap:
            cm = resolve_colmap(self._colmap_bin)
            if not cm.found:
                self._set_status(
                    "COLMAP not found. Uncheck Run COLMAP, or Install missing under Tools.",
                    error=True,
                )
                self._sec_open["tools"] = True
                return
            self._colmap_bin = cm.path

        if not self._vocab_ok_for_matcher():
            self._set_status(
                "vocab_tree matcher needs a vocab tree — Download or Browse under Tools.",
                error=True,
            )
            self._sec_open["tools"] = True
            return

        out_dir = self._ensure_out_dir()
        state["pack_path"] = pack_path
        state["out_dir"] = out_dir
        self._save_prefs()

        cameras = self._selected_cameras()
        blur = float(self._blur_pct)
        inject = bool(self._inject)
        run_cm = bool(self._run_colmap)
        write_script = bool(self._write_colmap_script)
        colmap_bin = self._colmap_bin or "colmap"
        colmap_settings = self._colmap_settings()

        def work():
            pack = load_pack(pack_path)
            result = prepare_photo_dataset(
                pack,
                PhotoPrepOptions(
                    out_dir=out_dir,
                    cameras=cameras,
                    blur_percentile=blur,
                    inject_intrinsics=inject,
                    colmap_bin=colmap_bin,
                    run_colmap=run_cm,
                    write_colmap_script=write_script,
                    colmap=colmap_settings,
                ),
                on_progress=self._stage_progress,
            )
            kept = {k: len(v) for k, v in result.extracted.items()}
            dropped = {k: len(v) for k, v in result.rejected.items()}
            state["last_report"] = result.report_path
            state["capture_type"] = "photo_dual"
            self._set_progress(1.0, "Photo prepare finished.")
            if run_cm:
                next_hint = "COLMAP ran."
            elif write_script:
                next_hint = f"Next: run {os.path.join(result.out_dir, 'run_colmap.bat')}"
            else:
                next_hint = "Next: Run COLMAP section."
            self._set_status(
                f"Photo ready. Kept {kept} (blur-dropped {dropped}). {next_hint}"
            )

        self._start_bg(work, label="Preparing photo dataset...")

    def _run_colmap_only(self):
        from splatking.video_pipeline import run_colmap_on_prep
        from splatking.paths import resolve_colmap

        if self._busy:
            return
        prep_dir = self._resolve_prep_dir()
        if not prep_dir or not os.path.isdir(prep_dir):
            self._set_status(
                "No prep directory found. Prepare Photo/Video first.",
                error=True,
            )
            return
        if not os.path.isdir(os.path.join(prep_dir, "images")):
            self._set_status(f"No images/ in prep dir: {prep_dir}", error=True)
            return

        cm = resolve_colmap(self._colmap_bin)
        if not cm.found:
            self._set_status(
                "COLMAP not found. Install missing or Browse under Tools.",
                error=True,
            )
            self._sec_open["tools"] = True
            return
        self._colmap_bin = cm.path

        if not self._vocab_ok_for_matcher():
            self._set_status(
                "vocab_tree matcher needs a vocab tree — Download or Browse under Tools.",
                error=True,
            )
            self._sec_open["tools"] = True
            return

        state = _ops_state()
        state["out_dir"] = prep_dir
        self._save_prefs()

        colmap_bin = self._colmap_bin or "colmap"
        colmap_settings = self._colmap_settings()
        cameras = (
            self._selected_cameras()
            if self._capture_type in ("video_dual", "photo_dual")
            else None
        )
        inject = bool(self._inject)
        write_script = bool(self._write_colmap_script)

        def work():
            run_colmap_on_prep(
                prep_dir,
                colmap_bin,
                colmap_settings,
                cameras=cameras,
                inject_intrinsics=inject,
                write_colmap_script=write_script,
                on_progress=self._stage_progress,
            )
            sparse = os.path.join(prep_dir, "sparse", "0")
            self._set_progress(1.0, "COLMAP finished.")
            self._set_status(f"COLMAP finished. Sparse model: {sparse}")

        self._start_bg(work, label="Running COLMAP...")

    def _run_prepare_lidar(self):
        from splatking.pack import load_pack, detect_capture_type, CaptureType
        from splatking.lidar_pipeline import LidarPrepOptions, prepare_lidar_dataset

        if self._busy:
            return
        pack_path = self._pack_path
        if not pack_path or not os.path.isdir(pack_path):
            self._set_status("Browse a SplatKing LiDAR pack folder first.", error=True)
            return
        ct = detect_capture_type(pack_path)
        if ct != CaptureType.PHOTO_LIDAR_SINGLE:
            self._set_status(f"Expected photo_lidar_single, got {ct.value}.", error=True)
            return
        out_dir = self._ensure_out_dir()
        state = _ops_state()
        state["pack_path"] = pack_path
        state["out_dir"] = out_dir
        self._save_prefs()
        confidence_min = int(self._confidence_min)

        def work():
            pack = load_pack(pack_path)
            result = prepare_lidar_dataset(
                pack,
                LidarPrepOptions(
                    out_dir=out_dir,
                    confidence_min=confidence_min,
                ),
            )
            state["capture_type"] = "photo_lidar_single"
            state["last_report"] = result.manifest_path or ""
            self._set_progress(1.0, "LiDAR prepare finished.")
            self._set_status(
                f"LiDAR ready: {result.registered_images} registered images, "
                f"{result.num_points3d:,} points, {result.depth_written} depth maps. "
                "Use Load into Scene next."
            )

        self._start_bg(work, label="Preparing LiDAR depth maps...")

    def _run_load_lidar(self):
        from splatking.pack import load_pack, detect_capture_type, CaptureType

        pack_path = self._pack_path
        if not pack_path:
            self._set_status("Browse a SplatKing LiDAR pack folder first.", error=True)
            return
        ct = detect_capture_type(pack_path)
        if ct != CaptureType.PHOTO_LIDAR_SINGLE:
            self._set_status(f"Expected photo_lidar_single, got {ct.value}.", error=True)
            return
        pack = load_pack(pack_path)
        if not pack.colmap_model_dir:
            self._set_status("COLMAP_Text_Model missing in this pack.", error=True)
            return
        try:
            lf.load_file(pack.colmap_model_dir, is_dataset=True)
            self._set_status(f"Loaded into scene: {pack.colmap_model_dir}")
        except Exception as e:
            self._set_status(f"Load failed: {e}", error=True)

    def _run_subsample(self):
        from splatking.camera_select import CameraSelectOptions, write_training_subset

        state = _ops_state()
        sparse = ""
        if self._capture_type == "photo_lidar_single" and self._pack_path:
            sparse = os.path.join(self._pack_path, "COLMAP_Text_Model", "sparse", "0")
        if not sparse or not os.path.isdir(sparse):
            cand = os.path.join(
                state.get("out_dir", "") or self._out_dir or self._ensure_out_dir(),
                "sparse",
                "0",
            )
            if os.path.isdir(cand):
                sparse = cand
        if not sparse or not os.path.isdir(sparse):
            self._set_status(
                "No sparse/0 found. Prepare Photo/Video COLMAP first, or browse a LiDAR pack.",
                error=True,
            )
            return
        out = os.path.join(os.path.dirname(sparse), "0_train")
        try:
            summary = write_training_subset(
                sparse,
                out,
                CameraSelectOptions(
                    mode=self._cam_modes[self._cam_mode_idx],
                    every_n=int(self._every_n),
                    random_pct=float(self._random_pct),
                ),
            )
            self._save_prefs()
            self._set_status(
                f"Training cameras: {summary['source_images']} -> "
                f"{summary['training_images']} ({summary['mode']}) -> {out}"
            )
        except Exception as e:
            self._set_status(f"Camera subset failed: {e}", error=True)

    def draw(self, ui):
        try:
            self._ensure_state()
            self._draw_body(ui)
        except Exception as e:
            tb = traceback.format_exc()
            try:
                ui.heading("SplatKing")
                ui.text_colored(f"Panel error: {e}", (0.95, 0.45, 0.4, 1.0))
                ui.text_wrapped(tb[-1500:])
                lf.log.error(f"splatking_importer panel draw failed: {tb}")
            except Exception:
                pass

    def _draw_body(self, ui):
        state = _ops_state()
        is_video = self._capture_type == "video_dual"
        is_photo = self._capture_type == "photo_dual"
        is_lidar = self._capture_type == "photo_lidar_single"
        self._sync_sec_open_for_type()

        ui.heading("SplatKing")
        if self._draw_error:
            ui.text_colored(self._draw_error, (0.95, 0.55, 0.35, 1.0))
        ui.text_wrapped(
            "Import SplatKing packs with dual-lens intrinsics, blur filtering, "
            "and LiDAR depth — not just bare images."
        )
        ui.separator()

        self._sanitize_tool_paths()

        changed, path = ui.path_input("Pack folder##sk_pack", self._pack_path, folder_mode=True)
        if changed and path != self._pack_path:
            self._pack_path = path
            state["pack_path"] = path
            self._refresh_info()
            self._save_prefs()
        ui.same_line()
        if ui.small_button("Browse##sk_pack"):
            picked = self._browse_folder("Select SplatKing pack folder", self._pack_path)
            if picked:
                self._pack_path = picked
                state["pack_path"] = picked
                self._refresh_info()
                self._save_prefs()
                self._set_status(f"Pack loaded: {picked}")

        changed, out = ui.path_input(
            "Base Output Folder##sk_out", self._out_dir, folder_mode=True
        )
        if changed:
            self._out_dir = out
            state["out_dir"] = out
            self._save_prefs()
        ui.same_line()
        if ui.small_button("Browse##sk_out"):
            start = self._out_dir or self._pack_path
            picked = self._browse_folder("Select base output folder", start)
            if picked:
                self._out_dir = picked
                state["out_dir"] = picked
                self._save_prefs()
        if not self._out_dir and self._pack_path:
            ui.text_disabled(f"Default: {self._default_out_preview()}")

        if self._info_lines:
            ui.separator()
            for i, line in enumerate(self._info_lines):
                if i == 0 and line.startswith("Detected:"):
                    ui.label(line)
                else:
                    ui.bullet_text(line)

        tools_need_attention = (
            "Not found" in self._ffmpeg_status or "Not found" in self._colmap_status
        )
        if tools_need_attention and not self._tools_forced_once:
            self._tools_forced_once = True
            self._sec_open["tools"] = True

        if self._header(ui, "Tools - ffmpeg / COLMAP / vocab##sk_tools", "tools"):
            ui.text_wrapped(
                "Paths are auto-detected and remembered. Missing tools can be installed "
                "via winget (Windows) or Browse. Vocab tree downloads on demand (~15MB)."
            )

            ui.label("ffmpeg")
            color = (0.4, 0.85, 0.45, 1.0) if "OK" in self._ffmpeg_status else (0.95, 0.55, 0.35, 1.0)
            ui.text_colored(self._ffmpeg_status, color)
            changed, ff_path = ui.path_input(
                "ffmpeg##sk_ff", self._ffmpeg_bin, folder_mode=False
            )
            if changed and (not ff_path or self._looks_like_binary(ff_path) or os.path.isfile(ff_path)):
                self._ffmpeg_bin = ff_path
            ui.same_line()
            if ui.small_button("Browse##sk_ff"):
                picked = self._browse_tool_binary("ffmpeg")
                if picked:
                    self._ffmpeg_bin = picked
                    self._refresh_tool_status()
                    self._save_prefs()

            ui.spacing()
            ui.label("COLMAP")
            color = (0.4, 0.85, 0.45, 1.0) if "OK" in self._colmap_status else (0.95, 0.55, 0.35, 1.0)
            ui.text_colored(self._colmap_status, color)
            changed, cm_path = ui.path_input(
                "COLMAP##sk_cm", self._colmap_bin, folder_mode=False
            )
            if changed:
                if not cm_path or self._looks_like_binary(cm_path) or os.path.isfile(cm_path):
                    self._colmap_bin = cm_path
            ui.same_line()
            if ui.small_button("Browse##sk_cm"):
                picked = self._browse_tool_binary("colmap")
                if picked:
                    self._colmap_bin = picked
                    self._refresh_tool_status()
                    self._save_prefs()
            self._hint(ui, "COLMAP optional for Prepare; required for Run COLMAP / After prepare.")
            self._hint(
                ui,
                "Recommended COLMAP ≥3.13 (4.x OK). GPU flags: FeatureExtraction/FeatureMatching.",
            )

            ui.spacing()
            ui.label("Vocab tree")
            color = (0.4, 0.85, 0.45, 1.0) if "OK" in self._vocab_status else (0.95, 0.55, 0.35, 1.0)
            ui.text_colored(self._vocab_status, color)
            changed, vt_path = ui.path_input(
                "Vocab##sk_vt", self._vocab_tree, folder_mode=False
            )
            if changed and (not vt_path or vt_path.lower().endswith(".bin") or os.path.isfile(vt_path)):
                self._vocab_tree = vt_path
            ui.same_line()
            if ui.small_button("Browse##sk_vt"):
                picked = self._browse_tool_binary("vocab")
                if picked:
                    self._vocab_tree = picked
                    self._refresh_tool_status()
                    self._save_prefs()

            if ui.button("Detect##sk_detect"):
                self._detect_tools()
            ui.same_line()
            if ui.button("Install missing##sk_install"):
                self._install_missing()
            ui.same_line()
            if ui.button("Download vocab tree##sk_vocab"):
                self._download_vocab()
            if ui.button("Save paths##sk_save"):
                self._sanitize_tool_paths()
                self._refresh_tool_status()
                self._save_prefs()
            ui.same_line()
            if ui.small_button("Re-check##sk_recheck"):
                self._refresh_tool_status()

        if self._header(ui, "Photo - stills to COLMAP##sk_photo", "photo"):
            if not is_photo:
                ui.text_disabled("Browse a photo_dual pack to enable this section.")
            else:
                ui.text_wrapped(
                    "Dual-lens stills need SfM. Copies images and injects PINHOLE "
                    "intrinsics from EXIF."
                )
                changed, self._lens_idx = ui.combo(
                    "Lenses##photo", self._lens_idx, self._lens_items
                )
                changed, self._blur_pct = ui.slider_float(
                    "Blur drop##photo", self._blur_pct, 0.0, 0.5
                )
                self._hint(ui, "Drop blurriest fraction of frames by pack sharpness scores.")
                changed, self._inject = ui.checkbox("Inject PINHOLE##photo", self._inject)
                self._hint(ui, "Inject known PINHOLE intrinsics for wide + ultra.")
                changed, self._run_colmap = ui.checkbox("After prepare##photo", self._run_colmap)
                self._hint(ui, "Run COLMAP after prepare (uses Run COLMAP section settings).")
                if self._run_colmap and not self._colmap_binary_ok():
                    ui.text_colored(
                        "COLMAP binary required — Install or Browse under Tools.",
                        (0.95, 0.55, 0.35, 1.0),
                    )
                ui.separator()
                if not self._busy:
                    if ui.button_styled("Prepare Photo Dataset##photo", "primary"):
                        self._run_prepare_photo()
                else:
                    ui.text_colored("Working...", (0.7, 0.8, 1.0, 1.0))

        if self._header(ui, "Video - extract then COLMAP##sk_video", "video"):
            if not is_video:
                ui.text_disabled("Browse a video_dual pack to enable this section.")
            else:
                ui.text_wrapped(
                    "Extract uniform frame samples with ffmpeg, filter blur, inject "
                    "PINHOLE intrinsics, then COLMAP (optional)."
                )
                changed, self._keep_pct = ui.slider_float(
                    "Keep %##video", self._keep_pct, 0.05, 1.0
                )
                self._hint(ui, "Keep frames % — uniform subsample (independent of resize).")
                if changed:
                    self._update_estimate()
                changed, self._resize = ui.slider_int(
                    "Resize##video", self._resize, 0, 3840
                )
                self._hint(ui, "Resize width in pixels (0 = native).")
                changed, self._lens_idx = ui.combo(
                    "Lenses##video", self._lens_idx, self._lens_items
                )
                if changed:
                    self._update_estimate()

                if self._estimate_lines:
                    ui.separator()
                    for line in self._estimate_lines:
                        ui.bullet_text(line)
                for w in self._warning_lines:
                    ui.text_colored(w, (0.95, 0.7, 0.3, 1.0))

                changed, self._blur_pct = ui.slider_float(
                    "Blur drop##video", self._blur_pct, 0.0, 0.5
                )
                self._hint(ui, "Drop blurriest fraction after extract (Laplacian score).")
                changed, self._inject = ui.checkbox("Inject PINHOLE##video", self._inject)
                self._hint(ui, "Inject known PINHOLE intrinsics for wide + ultra.")
                changed, self._run_colmap = ui.checkbox("After prepare##video", self._run_colmap)
                self._hint(ui, "Run COLMAP after prepare (uses Run COLMAP section settings).")
                if self._run_colmap and not self._colmap_binary_ok():
                    ui.text_colored(
                        "COLMAP binary required — Install or Browse under Tools.",
                        (0.95, 0.55, 0.35, 1.0),
                    )

                ui.separator()
                if not self._busy:
                    if ui.button_styled("Prepare Video Dataset##video", "primary"):
                        self._run_prepare_video()
                else:
                    ui.text_colored("Working in background...", (0.7, 0.8, 1.0, 1.0))

        if self._header(ui, "Run COLMAP##sk_colmap", "colmap"):
            prep_dir = self._resolve_prep_dir()
            ui.label("Prep dir")
            ui.text_disabled(prep_dir or "(none — prepare first)")
            ui.same_line()
            if ui.small_button("Browse##sk_prep"):
                start = prep_dir or self._out_dir or self._pack_path
                picked = self._browse_folder("Select COLMAP prep folder", start)
                if picked:
                    self._colmap_prep_override = picked
                    state["out_dir"] = picked
            changed, self._matcher_idx = ui.combo(
                "Matcher##colmap", self._matcher_idx, self._matcher_items
            )
            if self._matcher_items[self._matcher_idx] == "exhaustive":
                ui.text_colored(
                    "Exhaustive is O(N^2) — only for small sets (<~150 frames).",
                    (0.95, 0.55, 0.35, 1.0),
                )
            changed, self._colmap_use_gpu = ui.checkbox("Use GPU##colmap", self._colmap_use_gpu)
            self._hint(ui, "FeatureExtraction / FeatureMatching.use_gpu (COLMAP ≥3.13).")
            changed, self._colmap_max_image_size = ui.slider_int(
                "Max size##colmap",
                self._colmap_max_image_size,
                0,
                6400,
            )
            self._hint(ui, "FeatureExtraction.max_image_size (0 = unlimited).")
            changed, self._colmap_max_num_features = ui.slider_int(
                "Max features##colmap", self._colmap_max_num_features, 512, 32768
            )
            self._hint(ui, "SiftExtraction.max_num_features.")
            if self._matcher_items[self._matcher_idx] == "sequential":
                changed, self._colmap_seq_overlap = ui.slider_int(
                    "Overlap##colmap", self._colmap_seq_overlap, 1, 50
                )
                self._hint(ui, "SequentialMatching.overlap.")
            changed, self._colmap_min_num_matches = ui.slider_int(
                "Min matches##colmap", self._colmap_min_num_matches, 2, 100
            )
            self._hint(ui, "Mapper.min_num_matches.")
            changed, self._dual_mode = ui.checkbox(
                "Dual-lens merge##colmap", self._dual_mode
            )
            self._hint(
                ui,
                "wide+ultra → single sparse (registrator → rig → wide-only). Off = legacy.",
            )
            if self._dual_mode:
                changed, self._base_lens_idx = ui.combo(
                    "Base lens##colmap", self._base_lens_idx, self._base_lens_items
                )
                self._hint(ui, "Skeleton for Path1 (default ultra = wider FOV).")
                changed, self._dual_method_idx = ui.combo(
                    "Dual method##colmap", self._dual_method_idx, self._dual_method_items
                )
                self._hint(ui, "auto tries registrator, then rig, then wide-only.")
            if self._matcher_items[self._matcher_idx] == "vocab_tree":
                ui.text_disabled("Vocab tree path is set under Tools.")
            changed, self._write_colmap_script = ui.checkbox(
                "Write .bat/.sh##colmap", self._write_colmap_script
            )
            self._hint(
                ui,
                "Opt-in: also write run_colmap.bat / .sh (offline recipe; does not resume).",
            )
            if not self._busy:
                if ui.button_styled("Run COLMAP now##colmap", "primary"):
                    self._run_colmap_only()
            else:
                ui.text_colored("Working...", (0.7, 0.8, 1.0, 1.0))

        if self._header(ui, "LiDAR - skip SfM##sk_lidar", "lidar"):
            if not is_lidar:
                ui.text_disabled("Browse a photo_lidar_single pack to enable this section.")
            else:
                ui.text_wrapped(
                    "SplatKing already wrote COLMAP_Text_Model on-device. "
                    "Prepare Depth Maps unlocks native Depth Loss; then Load into Scene."
                )
                changed, self._confidence_min = ui.slider_int(
                    "Confidence##lidar",
                    self._confidence_min,
                    0,
                    2,
                )
                self._hint(ui, "Min depth confidence (0=low 1=med 2=high).")
                if (not self._busy) and is_lidar:
                    if ui.button_styled("1. Prepare Depth Maps##lidar", "primary"):
                        self._run_prepare_lidar()
                    if ui.button("2. Load COLMAP_Text_Model into Scene##lidar"):
                        self._run_load_lidar()
                elif self._busy:
                    ui.text_colored("Working...", (0.7, 0.8, 1.0, 1.0))
                else:
                    ui.text_disabled("Browse a LiDAR pack to enable Prepare / Load")

        if self._header(ui, "Training cameras (thin views for VRAM)##sk_train", "train"):
            ui.text_wrapped(
                "Not a GPU hardware readout — thins training views only while keeping "
                "full sparse geometry (every-N / random %). Useful for 10GB VRAM budgets."
            )
            if self._gpu_hint:
                ui.text_disabled(self._gpu_hint)
            changed, self._cam_mode_idx = ui.combo(
                "Mode##train", self._cam_mode_idx, self._cam_modes
            )
            if self._cam_modes[self._cam_mode_idx] == "every_n":
                changed, self._every_n = ui.slider_int("Every N##train", self._every_n, 1, 20)
            elif self._cam_modes[self._cam_mode_idx] == "random_pct":
                changed, self._random_pct = ui.slider_float(
                    "Keep fraction##train", self._random_pct, 0.1, 1.0
                )
            if ui.button("Write training subset (sparse/0_train)##train"):
                self._run_subsample()

        # Footer: Results (per-step summaries) + Status (CLI live) + progress.
        ui.separator()
        if self._header(ui, "Results##sk_results", "results"):
            results = list(state.get("result_lines") or [])
            if not results:
                ui.text_disabled(
                    "Per-step summaries (keypoints, verified pairs, sparse model) "
                    "appear here when each COLMAP stage finishes."
                )
            else:
                for line in results:
                    if hasattr(ui, "bullet_text"):
                        ui.bullet_text(line)
                    else:
                        ui.text_wrapped(line)

        if self._header(ui, "Status##sk_status", "status"):
            log = list(state.get("status_log") or [])
            status = state.get("status", "Idle") or "Idle"
            if not log:
                if status.lower().startswith("error") or "failed" in status.lower():
                    ui.text_colored(status, (0.95, 0.45, 0.4, 1.0))
                elif status != "Idle":
                    ui.text_colored(status, (0.55, 0.85, 0.55, 1.0))
                else:
                    ui.text_disabled("Idle — Browse a pack to begin")
            else:
                # CLI-style: older lines muted; newest line = current step.
                for line in log[-20:-1]:
                    if hasattr(ui, "text_disabled"):
                        ui.text_disabled(line)
                    else:
                        ui.text_wrapped(line)
                cur_line = log[-1]
                low = cur_line.lower()
                if low.startswith("error") or "failed" in low:
                    ui.text_colored(cur_line, (0.95, 0.45, 0.4, 1.0))
                else:
                    ui.text_colored(cur_line, (0.55, 0.85, 0.55, 1.0))
            if state.get("last_report"):
                ui.text_disabled(state["last_report"])

        progress = float(state.get("progress", 0.0))
        if self._busy or progress > 0:
            overlay = state.get("progress_label", "") or ("Working..." if self._busy else "")
            shown = max(progress, 0.02) if self._busy and progress < 0.02 else progress
            ui.progress_bar(shown, overlay=overlay)
