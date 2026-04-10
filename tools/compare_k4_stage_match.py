#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _parse_boolish(v) -> Optional[int]:
    if isinstance(v, (bool, np.bool_)):
        return 1 if bool(v) else 0
    if isinstance(v, (int, np.integer)):
        return 1 if int(v) != 0 else 0
    if isinstance(v, (float, np.floating)):
        return 1 if float(v) != 0.0 else 0
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("1", "true", "yes", "y", "on"):
            return 1
        if t in ("0", "false", "no", "n", "off", ""):
            return 0
    return None


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_candidates(cu_dir: Path) -> List[Dict]:
    out = []
    for cfg_path in sorted(cu_dir.glob("k4_chunk_*_field_*_config.json")):
        cfg = _load_json(cfg_path)
        stem = cfg_path.name
        if not stem.endswith("_config.json"):
            continue
        prefix = cfg_path.parent / stem[: -len("_config.json")]
        out.append(
            {
                "config_path": cfg_path,
                "prefix": prefix,
                "raw_offset": int(cfg.get("raw_offset", 0)),
                "field_idx": int(cfg.get("field_idx", 0)),
                "file_offset": int(cfg.get("file_offset", 0)),
                "is_first_field": _parse_boolish(cfg.get("is_first_field")),
                "field_phase_id": int(cfg.get("field_phase_id", 0)),
            }
        )
    return out


def _choose_candidate(
    cands: List[Dict],
    req_start: Optional[int],
    py_parity: Optional[int],
    raw_offset: Optional[int],
    field_idx: Optional[int],
    mode: str,
    py_dir: Path,
    distance_window: int,
) -> Tuple[Dict, List[Tuple[int, Dict]]]:
    filtered = cands
    if raw_offset is not None:
        filtered = [c for c in filtered if c["raw_offset"] == raw_offset]
    if field_idx is not None:
        filtered = [c for c in filtered if c["field_idx"] == field_idx]
    if not filtered:
        raise RuntimeError("No cuVHS K4 dump candidates after applying raw/field filters.")

    base_dist = None
    if req_start is not None:
        base_dist = min(abs(int(c["file_offset"]) - int(req_start)) for c in filtered)

    scored = []
    for c in filtered:
        if req_start is not None:
            dist = abs(int(c["file_offset"]) - int(req_start))
        else:
            dist = abs(int(c["file_offset"]))

        if mode == "signal" and base_dist is not None and distance_window >= 0:
            if dist > base_dist + distance_window:
                continue

        if mode == "signal":
            score_corr = -2.0
            py_luma = py_dir / "python_luma_u16.u16"
            py_chroma = py_dir / "python_chroma_u16.u16"
            cu_luma = Path(f"{c['prefix']}_luma_u16.u16")
            cu_chroma = Path(f"{c['prefix']}_chroma_u16.u16")
            if py_luma.exists() and py_chroma.exists() and cu_luma.exists() and cu_chroma.exists():
                a_l = _load_array(py_luma, "u16")
                b_l = _load_array(cu_luma, "u16")
                a_c = _load_array(py_chroma, "u16")
                b_c = _load_array(cu_chroma, "u16")
                n_l = min(a_l.size, b_l.size)
                n_c = min(a_c.size, b_c.size)
                corr_l = _corr(a_l[:n_l], b_l[:n_l]) if n_l > 0 else -1.0
                corr_c = _corr(a_c[:n_c], b_c[:n_c]) if n_c > 0 else -1.0
                score_corr = corr_l + corr_c
                c["match_corr_luma"] = float(corr_l)
                c["match_corr_chroma"] = float(corr_c)
                c["match_corr_sum"] = float(score_corr)
            else:
                c["match_corr_luma"] = None
                c["match_corr_chroma"] = None
                c["match_corr_sum"] = None
            score = (-score_corr, dist, int(c["raw_offset"]), int(c["field_idx"]))
        else:
            parity_mismatch = 0
            if py_parity is not None and c["is_first_field"] is not None and c["is_first_field"] != py_parity:
                parity_mismatch = 1
            score = (parity_mismatch, dist, int(c["raw_offset"]), int(c["field_idx"]))
        scored.append((score, c))

    if not scored:
        raise RuntimeError("No cuVHS K4 dump candidates left after matching constraints.")

    scored.sort(key=lambda x: x[0])
    ranked = [(idx + 1, item[1]) for idx, item in enumerate(scored[:5])]
    return scored[0][1], ranked


def _load_array(path: Path, dtype: str) -> np.ndarray:
    if dtype == "f64":
        return np.fromfile(path, dtype=np.float64)
    if dtype == "u16":
        return np.fromfile(path, dtype=np.uint16).astype(np.float64)
    raise ValueError(f"Unsupported dtype: {dtype}")


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return 1.0
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom == 0.0:
        return 1.0
    return float(np.sum(aa * bb) / denom)


def _metrics(ref: np.ndarray, test: np.ndarray, full_scale: Optional[float]) -> Dict[str, float]:
    n = min(ref.size, test.size)
    if n == 0:
        return {
            "n": 0,
            "rms": 0.0,
            "mae": 0.0,
            "corr": 1.0,
            "rms_pct_ref": 0.0,
            "rms_pct_fs": 0.0,
            "truncated": int(ref.size != test.size),
        }
    refv = ref[:n]
    testv = test[:n]
    diff = refv - testv
    rms = float(np.sqrt(np.mean(diff * diff)))
    diff_centered = diff - np.mean(diff)
    centered_rms = float(np.sqrt(np.mean(diff_centered * diff_centered)))
    mae = float(np.mean(np.abs(diff)))
    corr = _corr(refv, testv)
    ref_rms = float(np.sqrt(np.mean(refv * refv)))
    rms_pct_ref = (100.0 * rms / ref_rms) if ref_rms > 0.0 else 0.0
    centered_rms_pct_ref = (100.0 * centered_rms / ref_rms) if ref_rms > 0.0 else 0.0
    if full_scale is None or full_scale <= 0.0:
        rms_pct_fs = 0.0
    else:
        rms_pct_fs = 100.0 * rms / full_scale
    return {
        "n": int(n),
        "rms": rms,
        "centered_rms": centered_rms,
        "mae": mae,
        "corr": corr,
        "mean_diff": float(np.mean(diff)),
        "rms_pct_ref": rms_pct_ref,
        "centered_rms_pct_ref": centered_rms_pct_ref,
        "rms_pct_fs": rms_pct_fs,
        "truncated": int(ref.size != test.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match a cuVHS K4 dump field to a vhs-decode dump by parity/offset and compare stage arrays."
    )
    parser.add_argument("--python-dir", required=True, help="Directory with dump_k4_vhsdecode.py outputs")
    parser.add_argument("--cuvhs-dir", required=True, help="Directory with CUVHS_K4_DUMP_DIR outputs")
    parser.add_argument("--raw-offset", type=int, default=None, help="Optional exact cuVHS raw_offset to pick")
    parser.add_argument("--field-idx", type=int, default=None, help="Optional exact cuVHS field_idx to pick")
    parser.add_argument(
        "--match-mode",
        choices=("signal", "offset"),
        default="signal",
        help="Field matching mode: signal uses final luma+chroma correlation, offset uses parity+nearest sample.",
    )
    parser.add_argument(
        "--requested-start",
        type=int,
        default=None,
        help="Optional start sample override for nearest-offset candidate selection",
    )
    parser.add_argument(
        "--distance-window",
        type=int,
        default=1000000,
        help="When --match-mode signal is used, only consider candidates within best-offset + this many samples.",
    )
    args = parser.parse_args()

    py_dir = Path(args.python_dir)
    cu_dir = Path(args.cuvhs_dir)
    py_cfg = _load_json(py_dir / "config.json")
    py_parity = _parse_boolish(py_cfg.get("isFirstField"))
    req_start = args.requested_start
    if req_start is None and "requested_start" in py_cfg:
        try:
            req_start = int(py_cfg.get("requested_start"))
        except (TypeError, ValueError):
            req_start = None

    candidates = _load_candidates(cu_dir)
    if not candidates:
        raise RuntimeError(f"No K4 config dumps found in {cu_dir}")
    chosen, ranked = _choose_candidate(
        candidates,
        req_start,
        py_parity,
        args.raw_offset,
        args.field_idx,
        args.match_mode,
        py_dir,
        args.distance_window,
    )
    prefix = chosen["prefix"]

    stage_map = [
        (
            "chroma_source_coords",
            "python_chroma_source_coords.f64",
            [f"{prefix}_source_coords_post.f64"],
            "f64",
            None,
        ),
        (
            "chroma_source_level_adjust",
            "python_chroma_source_level_adjust.f64",
            [f"{prefix}_source_level_adjust_post.f64"],
            "f64",
            None,
        ),
        ("chroma_tbc", "python_chroma_tbc.f64", [f"{prefix}_chroma_tbc.f64"], "f64", None),
        ("chroma_postdeemph", "python_chroma_postdeemph.f64", [f"{prefix}_chroma_postdeemph.f64"], "f64", None),
        ("chroma_uphet_prephase", "python_chroma_uphet_prephase.f64", [f"{prefix}_chroma_uphet_prephase.f64"], "f64", None),
        (
            "chroma_postphase",
            "python_chroma_postphase.f64",
            [f"{prefix}_chroma_postphase.f64", f"{prefix}_chroma_postline.f64"],
            "f64",
            None,
        ),
        ("chroma_postfilter", "python_chroma_postfilter.f64", [f"{prefix}_chroma_postphasefilter.f64"], "f64", None),
        ("chroma_preacc", "python_chroma_preacc.f64", [f"{prefix}_chroma_preacc.f64"], "f64", None),
        ("final_chroma_u16", "python_chroma_u16.u16", [f"{prefix}_chroma_u16.u16"], "u16", 32767.0),
        ("final_luma_u16", "python_luma_u16.u16", [f"{prefix}_luma_u16.u16"], "u16", 32767.0),
    ]

    rows = []
    for stage_name, py_name, cu_names, dtype, fs in stage_map:
        py_path = py_dir / py_name
        cu_path = _first_existing([Path(p) for p in cu_names])
        if not py_path.exists() or cu_path is None or not cu_path.exists():
            rows.append(
                {
                    "stage": stage_name,
                    "status": "missing",
                    "python_path": str(py_path),
                    "cuvhs_path": str(cu_names[0]),
                }
            )
            continue
        a = _load_array(py_path, dtype)
        b = _load_array(cu_path, dtype)
        m = _metrics(a, b, fs)
        rows.append(
            {
                "stage": stage_name,
                "status": "ok",
                "python_path": str(py_path),
                "cuvhs_path": str(cu_path),
                **m,
            }
        )

    summary = {
        "python_meta": {
            "requested_start": req_start,
            "isFirstField": py_cfg.get("isFirstField"),
            "decodefield_offset": py_cfg.get("decodefield_offset"),
            "field_readloc": py_cfg.get("field_readloc"),
        },
        "chosen_cuvhs": {
            "config_path": str(chosen["config_path"]),
            "raw_offset": chosen["raw_offset"],
            "field_idx": chosen["field_idx"],
            "file_offset": chosen["file_offset"],
            "is_first_field": chosen["is_first_field"],
            "field_phase_id": chosen["field_phase_id"],
        },
        "top_candidates": [
            {
                "rank": rank,
                "config_path": str(c["config_path"]),
                "raw_offset": c["raw_offset"],
                "field_idx": c["field_idx"],
                "file_offset": c["file_offset"],
                "is_first_field": c["is_first_field"],
                "field_phase_id": c["field_phase_id"],
                "match_corr_luma": c.get("match_corr_luma"),
                "match_corr_chroma": c.get("match_corr_chroma"),
                "match_corr_sum": c.get("match_corr_sum"),
            }
            for rank, c in ranked
        ],
        "stages": rows,
    }
    print(json.dumps(summary, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
