# SplatKing Importer for LichtFeld Studio

LichtFeld plugin that consumes **SplatKing** capture folders (`splatpack.v2`) end-to-end — not just images or a generic COLMAP dump.

Sample captures (`spaltking_pack/`, multi-GB `.mov` / depth bins) are **not** in this repository and are gitignored. Point the Studio panel or CLI at a local pack folder.

| Blocker (existing plugins) | What this plugin does |
|---|---|
| Single camera-model dropdown; no per-folder intrinsics | Injects **two PINHOLE cameras** from metadata FOV (wide ≈74.6°, ultra ≈106.2°) |
| No `camera_params` field | Writes known `fx,fy,cx,cy` into COLMAP feature extraction |
| Blur filter is equirectangular-only / missing | Laplacian-variance filter on **flat** wide/ultra frames |
| Matcher = exhaustive/sequential only | Adds **vocab_tree** orchestration for thousands of images |
| LiDAR depth unused | Decodes `*_depth.bin` + confidence → metric depth for Depth Loss |
| VRAM wall at training | Camera Selector (every-N / random %) without dropping sparse geometry |

## Install from GitHub (LichtFeld)

Repo root **is** the plugin (`pyproject.toml` + `__init__.py`), so Studio can install by URL:

```python
import lichtfeld as lf

# After you push this repo:
lf.plugins.install("YOUR_GITHUB_USER/spaltking-lichtfeld")
# or
lf.plugins.install("https://github.com/YOUR_GITHUB_USER/spaltking-lichtfeld")
```

In the Studio UI: **Plugins → Install from Git** → paste the same URL.

Manual clone (folder name should match the plugin `name` in `pyproject.toml`):

```powershell
git clone https://github.com/YOUR_GITHUB_USER/spaltking-lichtfeld.git `
  "$env:USERPROFILE\.lichtfeld\plugins\splatking_importer"
```

```bash
git clone https://github.com/YOUR_GITHUB_USER/spaltking-lichtfeld.git \
  ~/.lichtfeld/plugins/splatking_importer
```

Requirements: `numpy` (from `pyproject.toml`). Optional: `opencv-python-headless` for faster blur scoring. Host needs `ffmpeg` for video; `colmap` optional (Prepare still writes `run_colmap.bat`).

## Pack shapes supported

- **`video_dual`** — `wide.mov` + `ultra.mov`, `metadata.json`, `frame_timecodes.csv`, `splatpack.json`
- **`photo_lidar_single`** — `COLMAP_Text_Model/` (poses + sparse), `sensor_data/*_depth.bin`, `photo_series.json`

## CLI (no LichtFeld required)

```bash
# From the plugin root (this repo)
python -m splatking.cli info   /path/to/Video_...
python -m splatking.cli info   /path/to/LidarSeries_...

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

1. **Tools** — ffmpeg / COLMAP paths are auto-detected on load and remembered. Change anytime → **Save paths**.
2. Pick a pack folder → **Scan** (shows FOV, frame counts, video size, depth availability).
3. **Video** — pick a density preset (**Quick / Balanced / Dense / Full**). The panel shows how many frames will be extracted before you commit. Prepare writes `run_colmap.bat` even if COLMAP is not installed.
4. **LiDAR** → Prepare Depth Maps, then **Load COLMAP_Text_Model into Scene**.
5. **Training cameras** → write a thinned sparse model for MCMC / 10GB VRAM budgets.

Default video settings are conservative (stride 15, max 200/lens, resize 1920) so the first run does not decode both ~900MB MOVs at full density.

## Design rule of thumb

- **Small video** — prepare inside the plugin (sequential / exhaustive).
- **Large video** — prepare + `vocab_tree` via CLI COLMAP; train in LichtFeld (+ camera subset).
- **LiDAR** — skip SfM; import on-device COLMAP; enable Depth Loss with decoded depths.
