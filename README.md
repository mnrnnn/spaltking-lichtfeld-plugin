# SplatKing Importer for LichtFeld Studio

LichtFeld plugin that consumes **SplatKing** capture folders (`splatpack.v2`) end-to-end — not just images or a generic COLMAP dump.

Sample captures (`spaltking_pack/`) are **not** in this repository (gitignored).

## Install

```python
import lichtfeld as lf
lf.plugins.install("mnrnnn/spaltking-lichtfeld-plugin")
```

Requirements: `numpy`. Optional: `opencv-python-headless`. Host needs `ffmpeg` for video; **COLMAP ≥3.13** (4.x OK, e.g. 4.1.0) for SfM. Windows: **Install missing** (winget) + **Download vocab tree** (~15MB).

## Pack shapes

- **`photo_dual`** — dual stills, no on-device COLMAP  
- **`video_dual`** — `wide.mov` + `ultra.mov` + metadata  
- **`photo_lidar_single`** — `COLMAP_Text_Model/` + depth bins  

## Studio panel (v0.2.3)

1. **Browse** pack / optional **Base Output** (default `{pack}/Output/{photo|video|lidar}_prep`)  
2. **Tools** — ffmpeg / COLMAP / vocab (+ Browse next to each path)  
3. **Photo** / **Video** — prepare dataset; video uses **Keep %** + **Resize** (independent); jobs run in a **background thread** so the progress bar updates  
4. **Run COLMAP** — matcher + SIFT/mapper params; Prep dir **Browse**; **Run COLMAP now**  
5. Photo/Video **After prepare** uses the **Run COLMAP section** settings  
6. LiDAR / Training cameras / Status + progress bar  

### COLMAP defaults (keep in sync when changing code)

Documented defaults for the **Run COLMAP** section / `ColmapSettings`.

**Recommended COLMAP: 3.13+ (CUDA), 4.x included (e.g. 4.1.0).**  
GPU flags are `FeatureExtraction` / `FeatureMatching`. COLMAP ≤3.12 used `SiftExtraction` / `SiftMatching.use_gpu`; with this plugin (v0.2.3+) upgrade to ≥3.13, or edit generated `run_colmap.bat` back to the old names.

| Parameter | Default | Notes |
|---|---|---|
| `FeatureExtraction.use_gpu` | **ON** (`1`) | Also `FeatureMatching.use_gpu` |
| `FeatureExtraction.max_image_size` | **3200** | `0` = unlimited |
| `SiftExtraction.max_num_features` | **8192** | |
| `SequentialMatching.overlap` | **10** | sequential matcher only |
| `Mapper.min_num_matches` | **15** | |
| Matcher | **sequential** | or `exhaustive` / `vocab_tree` |

Prefs keys: `colmap_use_gpu`, `colmap_max_image_size`, `colmap_max_num_features`, `colmap_seq_overlap`, `colmap_min_num_matches`, `colmap_matcher`.

Video keep default: **10%** frames, resize **1920**. Blur drop default **15%**.

## CLI

```bash
python -m splatking.cli info /path/to/pack
python -m splatking.cli prepare /path/to/Video_... --out ./_prep --keep-pct 0.1 --resize 1920
python -m splatking.cli colmap ./_prep --matcher sequential
python -m splatking.cli cameras ./_prep/sparse/0 --out ./_prep/sparse/0_train --every-n 2
```

```bash
set SPALTKING_PACK=D:\path\to\spaltking_pack
python verify_pack.py
```
