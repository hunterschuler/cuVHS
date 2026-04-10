#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _load_f64(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float64)


def _best_centered(py: np.ndarray, cu: np.ndarray, max_line_offset: int) -> Tuple[float, int, int]:
    best = None
    for off in range(0, max_line_offset + 1):
        n = min(py.size, cu.size - off)
        if n <= 0:
            continue
        a = py[:n]
        b = cu[off : off + n]
        d = a - b
        d = d - np.mean(d)
        rms = float(np.sqrt(np.mean(d * d)))
        if best is None or rms < best[0]:
            best = (rms, off, n)
    if best is None:
        return 0.0, 0, 0
    return best


def _metric_row(name: str, py_path: Path, cu_path: Path, samples_per_line: float, max_line_offset: int) -> Dict:
    py = _load_f64(py_path)
    cu = _load_f64(cu_path)
    rms, off, n = _best_centered(py, cu, max_line_offset)
    pct_line = (100.0 * rms / samples_per_line) if samples_per_line > 0.0 else 0.0
    return {
        "name": name,
        "python_path": str(py_path),
        "cuvhs_path": str(cu_path),
        "python_len": int(py.size),
        "cuvhs_len": int(cu.size),
        "best_line_offset": int(off),
        "compared_lines": int(n),
        "centered_rms_samples": float(rms),
        "centered_rms_pct_line": float(pct_line),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare K2/K3 line-location dumps with centered RMS and small line-index offset search."
    )
    parser.add_argument("--python-k2", required=True)
    parser.add_argument("--cuvhs-k2", required=True)
    parser.add_argument("--python-k3", required=True)
    parser.add_argument("--cuvhs-k3", required=True)
    parser.add_argument("--samples-per-line", type=float, default=1780.0)
    parser.add_argument("--max-line-offset", type=int, default=8)
    args = parser.parse_args()

    py_k2 = Path(args.python_k2)
    cu_k2 = Path(args.cuvhs_k2)
    py_k3 = Path(args.python_k3)
    cu_k3 = Path(args.cuvhs_k3)

    result = {
        "samples_per_line": args.samples_per_line,
        "max_line_offset": args.max_line_offset,
        "rows": [
            _metric_row(
                "k2_linelocs0",
                py_k2 / "python_linelocs0.f64",
                cu_k2 / "k2_chunk_0_field_0_linelocs0.f64",
                args.samples_per_line,
                args.max_line_offset,
            ),
            _metric_row(
                "k3_linelocs1",
                py_k3 / "python_linelocs1.f64",
                cu_k3 / "k3_chunk_0_field_0_linelocs1.f64",
                args.samples_per_line,
                args.max_line_offset,
            ),
            _metric_row(
                "k3_linelocs2",
                py_k3 / "python_linelocs2.f64",
                cu_k3 / "k3_chunk_0_field_0_linelocs2.f64",
                args.samples_per_line,
                args.max_line_offset,
            ),
        ],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
