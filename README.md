# SplatKing Importer for LichtFeld Studio

LichtFeld plugin that consumes **SplatKing** capture folders (`splatpack.v2`) end-to-end — not just images or a generic COLMAP dump.

Sample captures (`spaltking_pack/`) are **not** in this repository (gitignored).

## Install

```python
import lichtfeld as lf
lf.plugins.install("mnrnnn/spaltking-lichtfeld-plugin")
```

**Feature branch (v0.3.0 dual-lens):**

```python
lf.plugins.install("github:mnrnnn/spaltking-lichtfeld-plugin@feat-dual-lens-v030")
# or:
# lf.plugins.install("https://github.com/mnrnnn/spaltking-lichtfeld-plugin@feat-dual-lens-v030")
```

Requirements: `numpy`. Optional: `opencv-python-headless`. Host needs `ffmpeg` for video; **COLMAP ≥3.13** (4.x OK, e.g. 4.1.0) for SfM. Windows: **Install missing** (winget) + **Download vocab tree** (~15MB).

## Pack shapes

- **`photo_dual`** — dual stills, no on-device COLMAP  
- **`video_dual`** — `wide.mov` + `ultra.mov` + metadata  
- **`photo_lidar_single`** — `COLMAP_Text_Model/` + depth bins  

## Studio panel (v0.3.0)

1. **Browse** pack / optional **Base Output** (default `{pack}/Output/{photo|video|lidar}_prep`)  
2. **Tools** — ffmpeg / COLMAP / vocab (+ Browse next to each path)  
3. **Photo** / **Video** — prepare dataset; video uses **Keep %** + **Resize**; dual packs **time-pair** wide↔ultra (same basename) + pair-aware blur  
4. **Run COLMAP** — matcher + SIFT/mapper; **Dual-lens merge** (default ON): Path1 `image_registrator` → Path2 rig 2-pass → Path3 wide-only  
5. LiDAR / Training cameras / **Results** + **Status** + progress  

### Dual-lens COLMAP (v0.3.0)

- **Base lens** default `ultra` (wide FOV skeleton); **detail** = the other lens.  
- **Cross-lens matching:** after sequential/exhaustive/vocab, `matches_importer --match_type pairs` for exact basename pairs.  
- **Path1:** Mapper(base list) → `image_registrator` → `point_triangulator` → `bundle_adjuster`.  
- **Path2:** bootstrap mapper → `rig_configurator` → Mapper with `--Mapper.ba_refine_sensor_from_rig 0` (rotation lock; translation may still refine — COLMAP #3569).  
- **GLOMAP not used** on dual/rig paths.  
- FE uses `--ImageReader.single_camera_per_folder 1`. Mapper A uses `--Mapper.image_list_path`.

### COLMAP defaults

**Recommended COLMAP: 3.13+ (CUDA), 4.x included (e.g. 4.1.0).**  
GPU flags are `FeatureExtraction` / `FeatureMatching`. Optional **Write .bat/.sh** (off by default).

| Parameter | Default | Notes |
|---|---|---|
| `FeatureExtraction.use_gpu` | **ON** | Also `FeatureMatching.use_gpu` |
| `FeatureExtraction.max_image_size` | **3200** | `0` = unlimited |
| `SiftExtraction.max_num_features` | **8192** | |
| `SequentialMatching.overlap` | **10** | sequential only |
| `Mapper.min_num_matches` | **15** | |
| Matcher | **sequential** | or `exhaustive` / `vocab_tree` |
| Dual-lens merge | **ON** | `auto` / `registrator` / `rig` / `wide_only` |
| Base lens | **ultra** | |

Video keep default: **10%** frames, resize **1920**. Blur drop default **15%** (pair-aware when dual).

## CLI

```bash
python -m splatking.cli info /path/to/pack
python -m splatking.cli prepare /path/to/Video_... --out ./_prep --keep-pct 0.1 --resize 1920
python -m splatking.cli colmap ./_prep --matcher sequential --dual-method auto
python -m splatking.cli cameras ./_prep/sparse/0 --out ./_prep/sparse/0_train --every-n 2
```

```bash
set SPALTKING_PACK=D:\path\to\spaltking_pack
python verify_pack.py
python tests/test_rig_pairing.py
```
