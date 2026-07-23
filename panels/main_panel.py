"""Main SplatKing Importer panel — capture-aware UX with saved tool paths."""

from __future__ import annotations

import os
import traceback

import lichtfeld as lf

from ..operators.prepare_ops import _ops_state


class SplatKingImporterPanel(lf.ui.Panel):
    id = "splatking_importer.main"
    label = "SplatKing"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 40

    # LichtFeld may construct panels without calling Python __init__.
    # All instance state is created lazily in _ensure_state().
    def _ensure_state(self):
        if getattr(self, "_sk_ready", False):
            return
        from splatking.prefs import load_prefs, apply_tool_defaults
        from splatking.video_pipeline import DENSITY_PRESETS

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
        self._tools_open = False

        self._density_idx = 1
        self._density_labels = [p[0] for p in DENSITY_PRESETS]
        self._stride = int(prefs.get("video_stride", 15))
        self._max_frames = int(prefs.get("video_max_frames", 200))
        self._resize = int(prefs.get("video_resize", 1920))
        self._blur_pct = float(prefs.get("video_blur_percentile", 0.15))
        self._matcher_items = ["sequential", "exhaustive", "vocab_tree"]
        matcher = prefs.get("video_matcher", "sequential")
        self._matcher_idx = (
            self._matcher_items.index(matcher) if matcher in self._matcher_items else 0
        )
        self._lens_items = ["both", "wide", "ultra"]
        lenses = prefs.get("video_lenses", "both")
        self._lens_idx = self._lens_items.index(lenses) if lenses in self._lens_items else 0
        self._inject = bool(prefs.get("video_inject_intrinsics", True))
        self._run_colmap = bool(prefs.get("video_run_colmap", False))

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
        self._sk_ready = True

        try:
            self._refresh_tool_status()
            if self._pack_path:
                self._refresh_info()
        except Exception as e:
            self._draw_error = f"init: {e}"

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def _current_prefs(self) -> dict:
        return {
            "ffmpeg_bin": self._ffmpeg_bin,
            "colmap_bin": self._colmap_bin,
            "vocab_tree_path": self._vocab_tree,
            "video_stride": self._stride,
            "video_max_frames": self._max_frames,
            "video_resize": self._resize,
            "video_blur_percentile": self._blur_pct,
            "video_matcher": self._matcher_items[self._matcher_idx],
            "video_inject_intrinsics": self._inject,
            "video_run_colmap": self._run_colmap,
            "video_lenses": self._lens_items[self._lens_idx],
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
        state["status"] = "Preferences saved (ffmpeg / COLMAP paths remembered)."

    def _refresh_tool_status(self):
        from splatking.paths import resolve_ffmpeg, resolve_colmap

        ff = resolve_ffmpeg(self._ffmpeg_bin)
        cm = resolve_colmap(self._colmap_bin)
        if ff.found:
            self._ffmpeg_bin = ff.path
            self._ffmpeg_status = f"OK ({ff.source})" + (f" — {ff.version}" if ff.version else "")
        else:
            self._ffmpeg_status = "Not found — install ffmpeg or browse to ffmpeg.exe"
        if cm.found:
            self._colmap_bin = cm.path
            self._colmap_status = f"OK ({cm.source})" + (f" — {cm.version}" if cm.version else "")
        else:
            self._colmap_status = (
                "Not found — set path manually (Prepare still writes run_colmap.bat)"
            )

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

    def _set_status(self, msg: str, error: bool = False):
        state = _ops_state()
        state["status"] = msg
        if error:
            lf.log.error(msg)
        else:
            lf.log.info(msg)

    def _on_progress(self, msg: str):
        self._set_status(msg)

    def _selected_cameras(self) -> list[str]:
        lens = self._lens_items[self._lens_idx]
        if lens == "wide":
            return ["wide"]
        if lens == "ultra":
            return ["ultra"]
        return ["wide", "ultra"]

    def _apply_density_preset(self, idx: int):
        from splatking.video_pipeline import DENSITY_PRESETS

        _, stride, max_frames, resize = DENSITY_PRESETS[idx]
        self._density_idx = idx
        self._stride = stride
        self._max_frames = max_frames
        self._resize = resize
        self._update_estimate()

    def _refresh_info(self):
        from splatking.paths import format_bytes

        self._info_lines = []
        self._estimate_lines = []
        self._warning_lines = []
        self._capture_type = ""
        if not self._pack_path or not os.path.isdir(self._pack_path):
            return
        try:
            from splatking.pack import detect_capture_type, load_pack, CaptureType
            from splatking.intrinsics import colmap_camera_from_device

            ct = detect_capture_type(self._pack_path)
            self._capture_type = ct.value
            pack = load_pack(self._pack_path)
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
                pack, self._selected_cameras(), self._stride, self._max_frames
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
                f"resize={self._resize or 'native'} · stride={self._stride}"
            )
            suggested = est.suggested_matcher
            current = self._matcher_items[self._matcher_idx]
            if suggested != current:
                self._estimate_lines.append(
                    f"Suggested matcher: {suggested} (currently {current})"
                )
            self._warning_lines = list(est.warnings)
        except Exception as e:
            self._warning_lines = [f"Estimate failed: {e}"]

    def _ensure_out_dir(self) -> str:
        if self._out_dir:
            return self._out_dir
        return os.path.join(self._pack_path, "_lichtfeld_prep")

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
            self._set_status("Pick a SplatKing video pack folder first.", error=True)
            return

        ff = resolve_ffmpeg(self._ffmpeg_bin)
        if not ff.found:
            self._set_status(
                "ffmpeg not found. Open Tools, Detect or browse to ffmpeg.exe, then Save.",
                error=True,
            )
            self._tools_open = True
            return
        self._ffmpeg_bin = ff.path

        if self._run_colmap:
            cm = resolve_colmap(self._colmap_bin)
            if not cm.found:
                self._set_status(
                    "COLMAP not found. Uncheck 'Run COLMAP after prepare', or set the "
                    "binary path under Tools (Prepare still writes run_colmap.bat).",
                    error=True,
                )
                self._tools_open = True
                return
            self._colmap_bin = cm.path

        if self._matcher_items[self._matcher_idx] == "vocab_tree" and not self._vocab_tree:
            self._set_status(
                "vocab_tree matcher needs a vocab tree file path (Tools section).",
                error=True,
            )
            self._tools_open = True
            return

        ct = detect_capture_type(pack_path)
        if ct != CaptureType.VIDEO_DUAL:
            self._set_status(f"Expected video_dual pack, got {ct.value}.", error=True)
            return

        out_dir = self._ensure_out_dir()
        state["pack_path"] = pack_path
        state["out_dir"] = out_dir
        self._busy = True
        self._save_prefs()
        try:
            pack = load_pack(pack_path)
            est = estimate_extract(
                pack, self._selected_cameras(), self._stride, self._max_frames
            )
            self._set_status(
                f"Decoding ~{est.total_planned} frames from "
                f"{format_bytes(est.total_video_bytes)} of video..."
            )
            result = prepare_video_dataset(
                pack,
                VideoPrepOptions(
                    out_dir=out_dir,
                    cameras=self._selected_cameras(),
                    stride=int(self._stride),
                    max_frames_per_lens=int(self._max_frames),
                    resize_width=int(self._resize),
                    blur_percentile=float(self._blur_pct),
                    matcher=self._matcher_items[self._matcher_idx],
                    vocab_tree_path=self._vocab_tree,
                    inject_intrinsics=bool(self._inject),
                    colmap_bin=self._colmap_bin or "colmap",
                    ffmpeg_bin=self._ffmpeg_bin or "ffmpeg",
                    run_colmap=bool(self._run_colmap),
                ),
                on_progress=self._on_progress,
            )
            kept = {k: len(v) for k, v in result.extracted.items()}
            dropped = {k: len(v) for k, v in result.rejected.items()}
            state["last_report"] = result.report_path
            state["capture_type"] = "video_dual"
            script = os.path.join(result.out_dir, "run_colmap.bat")
            self._set_status(
                f"Done. Kept {kept} (blur-dropped {dropped}). "
                f"Intrinsics injected for {len(result.cameras)} cameras. "
                f"{'COLMAP ran.' if self._run_colmap else f'Next: run {script}'}"
            )
        except Exception as e:
            self._set_status(f"Video prepare failed: {e}", error=True)
        finally:
            self._busy = False

    def _run_prepare_lidar(self):
        from splatking.pack import load_pack, detect_capture_type, CaptureType
        from splatking.lidar_pipeline import LidarPrepOptions, prepare_lidar_dataset

        if self._busy:
            return
        pack_path = self._pack_path
        if not pack_path or not os.path.isdir(pack_path):
            self._set_status("Pick a SplatKing LiDAR pack folder first.", error=True)
            return
        ct = detect_capture_type(pack_path)
        if ct != CaptureType.PHOTO_LIDAR_SINGLE:
            self._set_status(f"Expected photo_lidar_single, got {ct.value}.", error=True)
            return
        out_dir = self._ensure_out_dir()
        state = _ops_state()
        state["pack_path"] = pack_path
        state["out_dir"] = out_dir
        self._busy = True
        self._save_prefs()
        try:
            self._set_status("Decoding LiDAR depth maps...")
            pack = load_pack(pack_path)
            result = prepare_lidar_dataset(
                pack,
                LidarPrepOptions(
                    out_dir=out_dir,
                    confidence_min=int(self._confidence_min),
                ),
            )
            state["capture_type"] = "photo_lidar_single"
            state["last_report"] = result.manifest_path or ""
            self._set_status(
                f"LiDAR ready: {result.registered_images} registered images, "
                f"{result.num_points3d:,} points, {result.depth_written} depth maps. "
                "Use Load into Scene next."
            )
        except Exception as e:
            self._set_status(f"LiDAR prepare failed: {e}", error=True)
        finally:
            self._busy = False

    def _run_load_lidar(self):
        from splatking.pack import load_pack, detect_capture_type, CaptureType

        pack_path = self._pack_path
        if not pack_path:
            self._set_status("Pick a SplatKing LiDAR pack folder first.", error=True)
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
            cand = os.path.join(state.get("out_dir", "") or self._out_dir, "sparse", "0")
            if os.path.isdir(cand):
                sparse = cand
        if not sparse or not os.path.isdir(sparse):
            self._set_status(
                "No sparse/0 found. Prepare video COLMAP first, or pick a LiDAR pack.",
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
        is_lidar = self._capture_type == "photo_lidar_single"

        ui.heading("SplatKing")
        if self._draw_error:
            ui.text_colored(self._draw_error, (0.95, 0.55, 0.35, 1.0))
        ui.text_wrapped(
            "Import SplatKing packs with dual-lens intrinsics, blur filtering, "
            "and LiDAR depth — not just bare images."
        )
        ui.separator()

        changed, path = ui.path_input("Pack folder", self._pack_path, folder_mode=True)
        if changed and path != self._pack_path:
            self._pack_path = path
            state["pack_path"] = path
            self._refresh_info()
        ui.same_line()
        if ui.small_button("Scan"):
            state["pack_path"] = self._pack_path
            self._refresh_info()

        changed, out = ui.path_input(
            "Output (optional)", self._out_dir, folder_mode=True
        )
        if changed:
            self._out_dir = out
            state["out_dir"] = out
        if not self._out_dir and self._pack_path:
            ui.text_disabled(f"Default: {os.path.join(self._pack_path, '_lichtfeld_prep')}")

        if self._info_lines:
            ui.separator()
            if is_video:
                ui.label("Video dual-lens pack")
            elif is_lidar:
                ui.label("LiDAR pack (SfM already done on-device)")
            else:
                ui.label("Pack")
            for line in self._info_lines:
                ui.bullet_text(line)

        tools_need_attention = (
            "Not found" in self._ffmpeg_status or "Not found" in self._colmap_status
        )
        open_tools = self._tools_open or tools_need_attention
        if ui.collapsing_header("Tools — ffmpeg / COLMAP", default_open=open_tools):
            self._tools_open = True
            ui.text_wrapped(
                "Paths are auto-detected once, then remembered. Change them anytime "
                "and press Save."
            )

            ui.label("ffmpeg")
            color = (0.4, 0.85, 0.45, 1.0) if "OK" in self._ffmpeg_status else (0.95, 0.55, 0.35, 1.0)
            ui.text_colored(self._ffmpeg_status, color)
            changed, self._ffmpeg_bin = ui.path_input(
                "ffmpeg binary", self._ffmpeg_bin, folder_mode=False
            )

            ui.spacing()
            ui.label("COLMAP")
            color = (0.4, 0.85, 0.45, 1.0) if "OK" in self._colmap_status else (0.95, 0.55, 0.35, 1.0)
            ui.text_colored(self._colmap_status, color)
            changed, self._colmap_bin = ui.path_input(
                "COLMAP binary", self._colmap_bin, folder_mode=False
            )
            ui.text_disabled(
                "Optional for Prepare: without COLMAP we still write run_colmap.bat"
            )

            changed, self._vocab_tree = ui.path_input(
                "Vocab tree (large sets)", self._vocab_tree, folder_mode=False
            )

            if ui.button("Detect"):
                self._detect_tools()
            ui.same_line()
            if ui.button("Save paths"):
                self._refresh_tool_status()
                self._save_prefs()
            ui.same_line()
            if ui.small_button("Re-check"):
                self._refresh_tool_status()

        show_video = is_video or not self._capture_type
        if show_video and ui.collapsing_header(
            "Video prepare", default_open=is_video or not self._capture_type
        ):
            if not is_video and self._capture_type:
                ui.text_disabled("Scan a video_dual pack to enable this section.")
            else:
                ui.label("Density preset")
                changed, new_idx = ui.combo(
                    "Preset", self._density_idx, self._density_labels
                )
                if changed and new_idx != self._density_idx:
                    self._apply_density_preset(new_idx)

                changed, self._lens_idx = ui.combo(
                    "Lenses", self._lens_idx, self._lens_items
                )
                if changed:
                    self._update_estimate()

                if self._estimate_lines:
                    ui.separator()
                    for line in self._estimate_lines:
                        ui.bullet_text(line)
                for w in self._warning_lines:
                    ui.text_colored(w, (0.95, 0.7, 0.3, 1.0))

                changed, self._matcher_idx = ui.combo(
                    "Matcher", self._matcher_idx, self._matcher_items
                )
                if self._matcher_items[self._matcher_idx] == "exhaustive":
                    ui.text_colored(
                        "Exhaustive is O(N^2) — only for small sets (<~150 frames).",
                        (0.95, 0.55, 0.35, 1.0),
                    )
                elif self._matcher_items[self._matcher_idx] == "vocab_tree":
                    ui.text_disabled("Requires vocab tree path under Tools.")

                changed, self._inject = ui.checkbox(
                    "Inject known PINHOLE intrinsics (wide + ultra)", self._inject
                )
                changed, self._run_colmap = ui.checkbox(
                    "Run COLMAP after prepare (needs COLMAP path)", self._run_colmap
                )

                if ui.collapsing_header("Advanced video options", default_open=False):
                    changed, self._stride = ui.slider_int("Stride", self._stride, 1, 60)
                    if changed:
                        self._update_estimate()
                    changed, self._max_frames = ui.slider_int(
                        "Max frames / lens (0=all)", self._max_frames, 0, 2000
                    )
                    if changed:
                        self._update_estimate()
                    changed, self._resize = ui.slider_int(
                        "Resize width (0=native)", self._resize, 0, 3840
                    )
                    changed, self._blur_pct = ui.slider_float(
                        "Drop blurriest fraction", self._blur_pct, 0.0, 0.5
                    )

                ui.separator()
                can_prep = (not self._busy) and is_video
                if can_prep:
                    if ui.button_styled("Prepare Video Dataset", "primary"):
                        self._run_prepare_video()
                else:
                    ui.text_disabled("Prepare Video Dataset (scan a video pack first)")
                if self._busy:
                    ui.text_colored(
                        "Working... UI may freeze while ffmpeg decodes.",
                        (0.7, 0.8, 1.0, 1.0),
                    )

        show_lidar = is_lidar or not self._capture_type
        if show_lidar and ui.collapsing_header(
            "LiDAR — skip SfM", default_open=is_lidar
        ):
            if not is_lidar and self._capture_type:
                ui.text_disabled("Scan a photo_lidar_single pack to enable this section.")
            else:
                ui.text_wrapped(
                    "SplatKing already wrote COLMAP_Text_Model on-device. "
                    "Prepare Depth Maps unlocks native Depth Loss; then Load into Scene."
                )
                changed, self._confidence_min = ui.slider_int(
                    "Min depth confidence (0=low 1=med 2=high)", self._confidence_min, 0, 2
                )
                if (not self._busy) and is_lidar:
                    if ui.button_styled("1. Prepare Depth Maps", "primary"):
                        self._run_prepare_lidar()
                    if ui.button("2. Load COLMAP_Text_Model into Scene"):
                        self._run_load_lidar()
                else:
                    ui.text_disabled("Scan a LiDAR pack to enable Prepare / Load")

        if ui.collapsing_header("Training cameras (VRAM)", default_open=False):
            ui.text_wrapped(
                "Keep full sparse geometry for structure; thin cameras only for training "
                "(3080 10GB: every-N or ~50%)."
            )
            changed, self._cam_mode_idx = ui.combo(
                "Mode", self._cam_mode_idx, self._cam_modes
            )
            if self._cam_modes[self._cam_mode_idx] == "every_n":
                changed, self._every_n = ui.slider_int("Every N", self._every_n, 1, 20)
            elif self._cam_modes[self._cam_mode_idx] == "random_pct":
                changed, self._random_pct = ui.slider_float(
                    "Keep fraction", self._random_pct, 0.1, 1.0
                )
            if ui.button("Write training subset (sparse/0_train)"):
                self._run_subsample()

        ui.separator()
        ui.label("Status")
        status = state.get("status", "Idle")
        if status.lower().startswith("error") or "failed" in status.lower():
            ui.text_colored(status, (0.95, 0.45, 0.4, 1.0))
        elif status != "Idle":
            ui.text_colored(status, (0.55, 0.85, 0.55, 1.0))
        else:
            ui.text_disabled("Idle — scan a pack to begin")
        if state.get("last_report"):
            ui.text_disabled(state["last_report"])
