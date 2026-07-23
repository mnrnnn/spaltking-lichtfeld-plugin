# SplatKing Importer for LichtFeld Studio

LichtFeld plugin that consumes **SplatKing** capture folders (`splatpack.v2`) end-to-end — not just images or a generic COLMAP dump.

Sample captures (`spaltking_pack/`, multi-GB `.mov` / depth bins) are **not** in this repository and are gitignored. Point the Studio panel or CLI at a local pack folder.

| Blocker (existing plugins) | What this plugin does |
|---|---|
| Single camera-model dropdown; no per-folder intrinsics | Injects **two PINHOLE cameras** from metadata FOV / EXIF 35mm-eq |
| No `camera_params` field | Writes known `fx,fy,cx,cy` into COLMAP feature extraction |
| Blur filter is equirectangular-only / missing | Laplacian / pack quality scores on **flat** wide/ultra frames |
| Matcher = exhaustive/sequential only | Adds **vocab_tree** orchestration (download on demand) |
| LiDAR depth unused | Decodes `*_depth.bin` + confidence → metric depth for Depth Loss |
| VRAM wall at training | Camera Selector (every-N / random %) — thins **views**, not a GPU dashboard |

## Install from GitHub (LichtFeld)

Repo root **is** the plugin (`pyproject.toml` + `__init__.py`), so Studio can install by URL:

```python
import lichtfeld as lf

lf.plugins.install("mnrnnn/spaltking-lichtfeld-plugin")
# or
lf.plugins.install("https://github.com/mnrnnn/spaltking-lichtfeld-plugin")
```

In the Studio UI: **Plugins → Install from Git** → paste the same URL.

Manual clone (folder name should match the plugin `name` in `pyproject.toml`):

```powershell
git clone https://github.com/mnrnnn/spaltking-lichtfeld-plugin.git `
  "$env:USERPROFILE\.lichtfeld\plugins\splatking_importer"
```

```bash
git clone https://github.com/mnrnnn/spaltking-lichtfeld-plugin.git \
  ~/.lichtfeld/plugins/splatking_importer
```

Requirements: `numpy` (from `pyproject.toml`). Optional: `opencv-python-headless` for faster blur scoring. Host needs `ffmpeg` for video; `colmap` optional (Prepare still writes `run_colmap.bat`). On Windows the panel can **Install missing** via winget and **Download vocab tree** on demand (~15MB; not shipped in the repo).

## Pack shapes supported

- **`photo_dual`** — dual stills (`wide_*.jpg` + `ultra_*.jpg`), `splatpack.json` / `photo_series.json` (no on-device COLMAP)
- **`video_dual`** — `wide.mov` + `ultra.mov`, `metadata.json`, `frame_timecodes.csv`, `splatpack.json`
- **`photo_lidar_single`** — `COLMAP_Text_Model/` (poses + sparse), `sensor_data/*_depth.bin`, `photo_series.json`

## CLI (no LichtFeld required)

```bash
# From the plugin root (this repo)
python -m splatking.cli info   /path/to/PhotoSeries_...
python -m splatking.cli info   /path/to/Video_...
python -m splatking.cli info   /path/to/LidarSeries_...

# Photo: copy stills + inject intrinsics + write run_colmap.{sh,bat}
python -m splatking.cli prepare /path/to/PhotoSeries_... --out ./_prep_photo

# Video: extract + inject intrinsics + write run_colmap.{sh,bat}
python -m splatking.cli prepare /path/to/Video_... --out ./_prep_video \
  --stride 15 --blur-percentile 0.15 --matcher sequential

# Large sets:
#   --matcher vocab_tree --vocab-tree /path/to/vocab.bin

# LiDAR: decode depth maps; COLMAP already done on-device
python -m splatking.cli prepare /path/to/LidarSeries_... --out ./_prep_lidar

# Thin training cameras (keep full sparse)
python -m splatking.cli cameras /path/to/LidarSeries_.../COLMAP_Text_Model/sparse/0 \
  --out ./_prep_lidar/sparse/0_train --mode every_n --every-n 2
```

Optional local self-check (pack stays outside git):

```bash
set SPALTKING_PACK=D:\path\to\spaltking_pack
python verify_pack.py
```

## Studio panel

Open the **SplatKing** main-panel tab:

1. **Pack folder** → **Browse** (auto-detects Photo / Video / LiDAR and shows which process to run).
2. **Base Output Folder** → **Browse** (optional). If empty, uses `{pack}/Output/{photo|video|lidar}_prep/`.
3. **Tools** — Detect / **Install missing** (winget on Windows) / **Download vocab tree** / Save paths.
4. Type-specific section only:
   - **Photo — stills to COLMAP**
   - **Video — extract then COLMAP** (density presets Quick / Balanced / Dense / Full)
   - **LiDAR — skip SfM** (Prepare Depth Maps → Load COLMAP_Text_Model)
5. **Training cameras** — thins training views for VRAM (not a live GPU hardware readout).

Default video settings are conservative (stride 15, max 200/lens, resize 1920) so the first run does not decode both ~900MB MOVs at full density.

## Design rule of thumb

- **Photo / small video** — prepare inside the plugin (sequential / exhaustive).
- **Large video** — prepare + `vocab_tree` via CLI COLMAP; train in LichtFeld (+ camera subset).
- **LiDAR** — skip SfM; import on-device COLMAP; enable Depth Loss with decoded depths.
