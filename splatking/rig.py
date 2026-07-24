"""Dual-lens pairing + COLMAP dual reconstruction (registrator / rig / wide-only)."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Callable, Optional


PAIR_TIME_TOLERANCE_S = 0.005  # 5 ms
REGISTRATOR_MIN_RATIO = 0.50
BOOTSTRAP_MIN_RATIO = 0.10


@dataclass(frozen=True)
class TimePair:
    pair_id: int
    wide_idx: int
    ultra_idx: int
    dt_s: float


def pair_by_frame_times(
    wide_times: list[float],
    ultra_times: list[float],
    *,
    tolerance_s: float = PAIR_TIME_TOLERANCE_S,
) -> list[TimePair]:
    """Greedy nearest-time 1:1 pairs within tolerance (wide → ultra)."""
    if not wide_times or not ultra_times:
        return []
    used_u: set[int] = set()
    raw: list[tuple[float, int, int, float]] = []
    for wi, wt in enumerate(wide_times):
        best_j = -1
        best_dt = tolerance_s + 1.0
        for uj, ut in enumerate(ultra_times):
            if uj in used_u:
                continue
            dt = abs(float(wt) - float(ut))
            if dt < best_dt:
                best_dt = dt
                best_j = uj
        if best_j >= 0 and best_dt <= tolerance_s:
            used_u.add(best_j)
            raw.append((float(wt), wi, best_j, best_dt))
    raw.sort(key=lambda t: t[0])
    return [
        TimePair(pair_id=i, wide_idx=wi, ultra_idx=uj, dt_s=dt)
        for i, (_t, wi, uj, dt) in enumerate(raw)
    ]


def select_pairs_keep_pct(pairs: list[TimePair], keep_pct: float) -> list[TimePair]:
    """Uniform stride over the paired list (not per-lens independently)."""
    if not pairs:
        return []
    p = min(max(float(keep_pct), 0.01), 1.0)
    if p >= 0.999:
        return list(pairs)
    step = max(1, int(round(1.0 / p)))
    return list(pairs[::step])


def frame_basename(pair_id: int) -> str:
    return f"frame_{int(pair_id):06d}.jpg"


def write_cross_lens_pairs_file(
    path: str,
    basenames: list[str],
    *,
    cam_a: str = "wide",
    cam_b: str = "ultra",
) -> int:
    """Write matches_importer --match_type pairs list (two names per line)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for name in basenames:
            base = os.path.basename(name)
            f.write(f"{cam_a}/{base} {cam_b}/{base}\n")
            n += 1
    return n


def write_rig_config(
    path: str,
    *,
    ref_sensor: str = "wide",
    other: str = "ultra",
) -> None:
    """COLMAP rig_config.json without cam_from_rig (estimate from bootstrap)."""
    cfg = [
        {
            "cameras": [
                {"image_prefix": f"{ref_sensor}/", "ref_sensor": True},
                {"image_prefix": f"{other}/"},
            ]
        }
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def list_image_basenames(image_root: str, cam: str) -> list[str]:
    cam_dir = os.path.join(image_root, cam)
    if not os.path.isdir(cam_dir):
        return []
    names = [
        n
        for n in os.listdir(cam_dir)
        if n.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    names.sort()
    return names


def shared_basenames(image_root: str, cams: list[str]) -> list[str]:
    if len(cams) < 2:
        return list_image_basenames(image_root, cams[0]) if cams else []
    sets = [set(list_image_basenames(image_root, c)) for c in cams]
    shared = sets[0].intersection(*sets[1:])
    return sorted(shared)


def count_model_images(model_dir: str) -> dict[str, Any]:
    """Best-effort count of registered images (text model preferred)."""
    out: dict[str, Any] = {
        "total": 0,
        "by_prefix": {},
        "model_dir": model_dir,
    }
    if not model_dir or not os.path.isdir(model_dir):
        return out
    images_txt = os.path.join(model_dir, "images.txt")
    images_bin = os.path.join(model_dir, "images.bin")
    n = 0
    by: dict[str, int] = {}
    try:
        if os.path.isfile(images_txt):
            with open(images_txt, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    parts = s.split()
                    if len(parts) >= 10 and parts[0].isdigit():
                        name = parts[-1]
                        n += 1
                        pref = name.split("/", 1)[0] if "/" in name else "_"
                        by[pref] = by.get(pref, 0) + 1
        elif os.path.isfile(images_bin):
            # Unknown without full parser — treat as present but opaque.
            n = -1
    except OSError:
        pass
    out["total"] = n
    out["by_prefix"] = by
    return out


def find_first_sparse_model(sparse_dir: str) -> Optional[str]:
    if not sparse_dir or not os.path.isdir(sparse_dir):
        return None
    subs = []
    for name in os.listdir(sparse_dir):
        path = os.path.join(sparse_dir, name)
        if os.path.isdir(path) and name.isdigit():
            if os.path.isfile(os.path.join(path, "cameras.bin")) or os.path.isfile(
                os.path.join(path, "cameras.txt")
            ):
                subs.append((int(name), path))
    if not subs:
        # model may be written directly into sparse_dir
        if os.path.isfile(os.path.join(sparse_dir, "cameras.bin")) or os.path.isfile(
            os.path.join(sparse_dir, "cameras.txt")
        ):
            return sparse_dir
        return None
    subs.sort()
    return subs[0][1]


def promote_model_to_sparse0(src_model: str, sparse_root: str) -> str:
    """Copy/replace sparse/0 with src model contents."""
    dst = os.path.join(sparse_root, "0")
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src_model):
        s = os.path.join(src_model, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    return dst


def _fe_cmd(
    colmap_bin: str,
    database_path: str,
    image_root: str,
    list_path: str,
    cm,
    *,
    inject_params: str = "",
    dual: bool = True,
) -> list[str]:
    fe = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        database_path,
        "--image_path",
        image_root,
        "--image_list_path",
        list_path,
        "--ImageReader.camera_model",
        "PINHOLE",
        "--FeatureExtraction.use_gpu",
        "1" if cm.use_gpu else "0",
        "--SiftExtraction.max_num_features",
        str(int(cm.max_num_features)),
    ]
    if dual:
        fe += ["--ImageReader.single_camera_per_folder", "1"]
    else:
        fe += ["--ImageReader.single_camera", "1"]
    if cm.max_image_size and cm.max_image_size > 0:
        fe += ["--FeatureExtraction.max_image_size", str(int(cm.max_image_size))]
    if inject_params:
        fe += ["--ImageReader.camera_params", inject_params]
    return fe


def _matcher_cmd(colmap_bin: str, database_path: str, cm) -> list[str]:
    gpu = "1" if cm.use_gpu else "0"
    if cm.matcher == "exhaustive":
        return [
            colmap_bin,
            "exhaustive_matcher",
            "--database_path",
            database_path,
            "--FeatureMatching.use_gpu",
            gpu,
        ]
    if cm.matcher == "vocab_tree":
        m = [
            colmap_bin,
            "vocab_tree_matcher",
            "--database_path",
            database_path,
            "--FeatureMatching.use_gpu",
            gpu,
        ]
        if cm.vocab_tree_path:
            m += ["--VocabTreeMatching.vocab_tree_path", cm.vocab_tree_path]
        return m
    return [
        colmap_bin,
        "sequential_matcher",
        "--database_path",
        database_path,
        "--FeatureMatching.use_gpu",
        gpu,
        "--SequentialMatching.overlap",
        str(int(cm.sequential_overlap)),
    ]


def build_feature_match_commands(
    opts,
    cameras: list[Any],
    image_root: str,
    database_path: str,
    *,
    dual: bool = False,
) -> list[list[str]]:
    """FE (+ optional cross-lens pairs file step is separate) + matcher."""
    from .intrinsics import ColmapCamera

    cm = getattr(opts, "colmap", None)
    colmap_bin = getattr(opts, "colmap_bin", "colmap")
    inject = getattr(opts, "inject_intrinsics", True)
    cam_names = getattr(opts, "cameras", [c.source_camera for c in cameras])
    cam_by_src = {
        getattr(c, "source_camera", ""): c for c in cameras if isinstance(c, ColmapCamera) or hasattr(c, "params")
    }

    cmds: list[list[str]] = []
    for cam in cam_names:
        params = ""
        cc = cam_by_src.get(cam)
        if inject and cc is not None and getattr(cc, "params", None):
            params = ",".join(f"{p:.9g}" for p in cc.params)
        list_path = os.path.join(image_root, f"_list_{cam}.txt")
        cmds.append(
            _fe_cmd(
                colmap_bin,
                database_path,
                image_root,
                list_path,
                cm,
                inject_params=params,
                dual=dual and len(cam_names) > 1,
            )
        )
    cmds.append(_matcher_cmd(colmap_bin, database_path, cm))
    return cmds


def build_cross_lens_importer_cmd(
    colmap_bin: str,
    database_path: str,
    pairs_path: str,
    cm,
) -> list[str]:
    gpu = "1" if getattr(cm, "use_gpu", True) else "0"
    return [
        colmap_bin,
        "matches_importer",
        "--database_path",
        database_path,
        "--match_list_path",
        pairs_path,
        "--match_type",
        "pairs",
        "--FeatureMatching.use_gpu",
        gpu,
    ]


def run_dual_colmap(
    *,
    colmap_bin: str,
    image_root: str,
    database_path: str,
    sparse_dir: str,
    cameras: list[str],
    settings,
    inject_cameras: Optional[list[Any]] = None,
    inject_intrinsics: bool = True,
    on_log: Optional[Callable[..., Any]] = None,
    run_sequence: Optional[Callable] = None,
) -> dict[str, Any]:
    """
    Dual-lens COLMAP with Path1 registrator → Path2 rig → Path3 wide-only.

    ``run_sequence`` should be ``run_colmap_sequence`` from video_pipeline
    (injected to avoid circular imports at module load for tests).
    """
    if run_sequence is None:
        from .video_pipeline import run_colmap_sequence as run_sequence

    def emit(msg: str, **kw):
        if not on_log:
            return
        try:
            on_log(msg, **kw)
        except TypeError:
            on_log(msg)

    cm = settings
    dual_mode = bool(getattr(cm, "dual_mode", True))
    dual_method = str(getattr(cm, "dual_method", "auto") or "auto")
    base_lens = str(getattr(cm, "base_lens", "ultra") or "ultra")
    detail_lens = "wide" if base_lens == "ultra" else "ultra"
    if base_lens not in cameras and cameras:
        base_lens = cameras[0]
        detail_lens = cameras[1] if len(cameras) > 1 else cameras[0]
    min_ratio = float(getattr(cm, "registrator_min_ratio", REGISTRATOR_MIN_RATIO))

    is_dual = dual_mode and len(cameras) >= 2
    report: dict[str, Any] = {
        "dual_mode": is_dual,
        "dual_method_requested": dual_method,
        "base_lens": base_lens,
        "detail_lens": detail_lens,
        "path_used": "legacy",
    }

    class _Opts:
        pass

    o = _Opts()
    o.colmap = cm
    o.colmap_bin = colmap_bin
    o.inject_intrinsics = inject_intrinsics
    o.cameras = cameras

    cams_for_fe = inject_cameras or []
    os.makedirs(sparse_dir, exist_ok=True)

    if not is_dual or dual_method == "wide_only" or (len(cameras) == 1):
        # Single-lens / forced wide-only
        only = ["wide"] if "wide" in cameras else [cameras[0]]
        o.cameras = only
        cmds = build_feature_match_commands(
            o, cams_for_fe, image_root, database_path, dual=False
        )
        cmds.append(
            [
                colmap_bin,
                "mapper",
                "--database_path",
                database_path,
                "--image_path",
                image_root,
                "--output_path",
                sparse_dir,
                "--Mapper.min_num_matches",
                str(int(cm.min_num_matches)),
                "--Mapper.image_list_path",
                os.path.join(image_root, f"_list_{only[0]}.txt"),
            ]
        )
        emit(f"COLMAP dual path: wide-only ({only[0]})")
        run_sequence(cmds, on_log=on_log)
        report["path_used"] = "wide_only"
        report["commands"] = [" ".join(c) for c in cmds]
        return report

    # Shared FE + matcher + cross-lens
    fe_match = build_feature_match_commands(
        o, cams_for_fe, image_root, database_path, dual=True
    )
    emit("COLMAP dual: feature extract + matcher")
    run_sequence(fe_match, on_log=on_log)

    shared = shared_basenames(image_root, cameras)
    pairs_path = os.path.join(image_root, "_cross_lens_pairs.txt")
    n_pairs = write_cross_lens_pairs_file(pairs_path, shared)
    report["cross_lens_pairs"] = n_pairs
    if n_pairs > 0:
        emit(f"[match] cross-lens pairs file ({n_pairs})", result=True)
        cross_cmd = build_cross_lens_importer_cmd(colmap_bin, database_path, pairs_path, cm)
        run_sequence([cross_cmd], on_log=on_log)

    detail_total = len(list_image_basenames(image_root, detail_lens))
    methods = []
    if dual_method == "auto":
        methods = ["registrator", "rig", "wide_only"]
    elif dual_method == "registrator":
        methods = ["registrator", "wide_only"]
    elif dual_method == "rig":
        methods = ["rig", "wide_only"]
    else:
        methods = ["wide_only"]

    for method in methods:
        try:
            if method == "registrator":
                ok = _path_registrator(
                    colmap_bin=colmap_bin,
                    image_root=image_root,
                    database_path=database_path,
                    sparse_dir=sparse_dir,
                    base_lens=base_lens,
                    detail_lens=detail_lens,
                    detail_total=detail_total,
                    min_ratio=min_ratio,
                    cm=cm,
                    on_log=on_log,
                    run_sequence=run_sequence,
                    emit=emit,
                    report=report,
                )
                if ok:
                    report["path_used"] = "registrator"
                    return report
                emit("Path1 registrator below threshold — trying fallback")
            elif method == "rig":
                ok = _path_rig(
                    colmap_bin=colmap_bin,
                    image_root=image_root,
                    database_path=database_path,
                    sparse_dir=sparse_dir,
                    cm=cm,
                    on_log=on_log,
                    run_sequence=run_sequence,
                    emit=emit,
                    report=report,
                )
                if ok:
                    report["path_used"] = "rig"
                    return report
                emit("Path2 rig failed — trying wide-only")
            else:
                only = ["wide"] if "wide" in cameras else [base_lens]
                mapper = [
                    colmap_bin,
                    "mapper",
                    "--database_path",
                    database_path,
                    "--image_path",
                    image_root,
                    "--output_path",
                    sparse_dir,
                    "--Mapper.min_num_matches",
                    str(int(cm.min_num_matches)),
                    "--Mapper.image_list_path",
                    os.path.join(image_root, f"_list_{only[0]}.txt"),
                ]
                emit(f"Path3 wide-only mapper ({only[0]})")
                # Clear previous sparse children carefully — mapper writes new models
                run_sequence([mapper], on_log=on_log)
                report["path_used"] = "wide_only"
                emit("Warning: dual integration fell back to wide-only", result=True)
                return report
        except Exception as e:
            emit(f"Dual path {method} error: {e}")
            continue

    report["path_used"] = "failed"
    raise RuntimeError("All dual COLMAP paths failed")


def _path_registrator(
    *,
    colmap_bin,
    image_root,
    database_path,
    sparse_dir,
    base_lens,
    detail_lens,
    detail_total,
    min_ratio,
    cm,
    on_log,
    run_sequence,
    emit,
    report,
) -> bool:
    base_out = os.path.join(sparse_dir, f"{base_lens}_base")
    merged = os.path.join(sparse_dir, "merged")
    os.makedirs(base_out, exist_ok=True)
    os.makedirs(merged, exist_ok=True)

    mapper_a = [
        colmap_bin,
        "mapper",
        "--database_path",
        database_path,
        "--image_path",
        image_root,
        "--output_path",
        base_out,
        "--Mapper.min_num_matches",
        str(int(cm.min_num_matches)),
        "--Mapper.image_list_path",
        os.path.join(image_root, f"_list_{base_lens}.txt"),
    ]
    emit(f"Path1: Mapper A ({base_lens} only)")
    run_sequence([mapper_a], on_log=on_log)
    base_model = find_first_sparse_model(base_out)
    if not base_model:
        report["registrator_error"] = "no base model"
        return False

    reg = [
        colmap_bin,
        "image_registrator",
        "--database_path",
        database_path,
        "--input_path",
        base_model,
        "--output_path",
        merged,
    ]
    emit(f"Path1: image_registrator ({detail_lens} → {base_lens})")
    run_sequence([reg], on_log=on_log)

    # Prefer writing into merged/; some COLMAP versions write merged/0
    reg_model = find_first_sparse_model(merged) or merged
    if not (
        os.path.isfile(os.path.join(reg_model, "images.bin"))
        or os.path.isfile(os.path.join(reg_model, "images.txt"))
    ):
        # registrator may write flat into --output_path
        if os.path.isfile(os.path.join(merged, "images.bin")) or os.path.isfile(
            os.path.join(merged, "images.txt")
        ):
            reg_model = merged
        else:
            report["registrator_error"] = "registrator produced no model"
            return False

    tri_out = os.path.join(sparse_dir, "merged_tri")
    os.makedirs(tri_out, exist_ok=True)
    tri = [
        colmap_bin,
        "point_triangulator",
        "--database_path",
        database_path,
        "--image_path",
        image_root,
        "--input_path",
        reg_model,
        "--output_path",
        tri_out,
    ]
    ba = [
        colmap_bin,
        "bundle_adjuster",
        "--input_path",
        tri_out,
        "--output_path",
        tri_out,
    ]
    emit("Path1: point_triangulator + bundle_adjuster")
    run_sequence([tri, ba], on_log=on_log)

    counts = count_model_images(tri_out)
    detail_reg = int(counts.get("by_prefix", {}).get(detail_lens, 0))
    ratio = (detail_reg / detail_total) if detail_total > 0 else 0.0
    report["registrator_detail_registered"] = detail_reg
    report["registrator_detail_total"] = detail_total
    report["registrator_ratio"] = ratio
    emit(
        f"[registrator] {detail_lens}→{base_lens}: {detail_reg}/{detail_total} "
        f"({ratio:.0%})",
        result=True,
    )
    if detail_total > 0 and ratio < min_ratio:
        return False

    promote_model_to_sparse0(tri_out, sparse_dir)
    return True


def _path_rig(
    *,
    colmap_bin,
    image_root,
    database_path,
    sparse_dir,
    cm,
    on_log,
    run_sequence,
    emit,
    report,
) -> bool:
    boot = os.path.join(sparse_dir, "_bootstrap")
    os.makedirs(boot, exist_ok=True)
    mapper_a = [
        colmap_bin,
        "mapper",
        "--database_path",
        database_path,
        "--image_path",
        image_root,
        "--output_path",
        boot,
        "--Mapper.min_num_matches",
        str(int(cm.min_num_matches)),
    ]
    emit("Path2: Mapper A bootstrap (no rig)")
    run_sequence([mapper_a], on_log=on_log)
    boot_model = find_first_sparse_model(boot)
    if not boot_model:
        report["rig_error"] = "no bootstrap model"
        return False

    boot_counts = count_model_images(boot_model)
    total_imgs = sum(
        len(list_image_basenames(image_root, c))
        for c in ("wide", "ultra")
        if os.path.isdir(os.path.join(image_root, c))
    )
    reg_n = int(boot_counts.get("total") or 0)
    if total_imgs > 0 and reg_n >= 0 and (reg_n / total_imgs) < BOOTSTRAP_MIN_RATIO:
        report["rig_error"] = f"bootstrap registration too low ({reg_n}/{total_imgs})"
        return False

    rig_cfg = os.path.join(os.path.dirname(database_path), "rig_config.json")
    write_rig_config(rig_cfg, ref_sensor="wide", other="ultra")
    rigged = os.path.join(sparse_dir, "_rigged")
    os.makedirs(rigged, exist_ok=True)
    cfg_cmd = [
        colmap_bin,
        "rig_configurator",
        "--database_path",
        database_path,
        "--rig_config_path",
        rig_cfg,
        "--input_path",
        boot_model,
        "--output_path",
        rigged,
    ]
    emit("Path2: rig_configurator")
    run_sequence([cfg_cmd], on_log=on_log)

    mapper_b_out = os.path.join(sparse_dir, "_rig_map")
    os.makedirs(mapper_b_out, exist_ok=True)
    mapper_b = [
        colmap_bin,
        "mapper",
        "--database_path",
        database_path,
        "--image_path",
        image_root,
        "--output_path",
        mapper_b_out,
        "--Mapper.min_num_matches",
        str(int(cm.min_num_matches)),
        "--Mapper.ba_refine_sensor_from_rig",
        "0",
    ]
    emit("Path2: Mapper B (rig fixed, rotation; see R1)")
    run_sequence([mapper_b], on_log=on_log)
    final = find_first_sparse_model(mapper_b_out)
    if not final:
        report["rig_error"] = "mapper B produced no model"
        return False
    promote_model_to_sparse0(final, sparse_dir)
    emit("[rig] Mapper B → sparse/0", result=True)
    return True
