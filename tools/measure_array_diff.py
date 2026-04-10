#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args()

    a = np.fromfile(args.a, dtype=np.float64)
    b = np.fromfile(args.b, dtype=np.float64)
    if a.size != b.size:
        raise SystemExit(f"size mismatch: {a.size} vs {b.size}")

    diff = a - b
    a_center = a - np.mean(a)
    b_center = b - np.mean(b)
    denom = np.sqrt(np.sum(a_center * a_center) * np.sum(b_center * b_center))
    corr = float(np.sum(a_center * b_center) / denom) if denom != 0.0 else 1.0

    result = {
        "a": str(Path(args.a)),
        "b": str(Path(args.b)),
        "length": int(a.size),
        "mean_signed_diff": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff * diff))),
        "debiased_rms": float(np.sqrt(np.mean((diff - np.mean(diff)) ** 2))),
        "corr": corr,
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
