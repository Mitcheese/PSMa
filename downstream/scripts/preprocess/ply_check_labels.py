#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_ply_labels.py
Scan .ply files under a folder, validate label values, and write issues to CSV.

Usage:
  python scripts/preprocess/ply_check_labels.py /path/to/folder --out report.csv
Options:
  --pattern "*.ply"        # filename pattern
  --binary-only            # enforce strict binary labels {0,1}
  --recursive              # scan subdirectories
  --eps 1e-6               # float tolerance for comparisons
"""

import os
import sys
import csv
import argparse
import math
from pathlib import Path

from tqdm import tqdm
import numpy as np
from plyfile import PlyData

CANDIDATE_LABEL_KEYS = ["label", "labels", "y", "seg", "mask", "gt"]

def find_label_key(vertex_data) -> str:
    """Find the most likely label field in vertex data (case-insensitive)."""
    names = list(vertex_data.dtype.names or [])
    lower_map = {n.lower(): n for n in names}  # original name map
    for k in CANDIDATE_LABEL_KEYS:
        if k in lower_map:
            return lower_map[k]
    return None

def is_finite_nd(arr: np.ndarray) -> bool:
    return np.isfinite(arr).all()

def check_labels(arr: np.ndarray, eps: float = 1e-6, binary_only: bool = False):
    """
    Return a dict of issue indices:
      - non_finite_idx: rows with NaN/Inf
      - out_of_range_idx: rows <0 or >1 (outside eps tolerance)
      - non_binary_idx: rows not close to 0/1 (only if binary_only=True)
    """
    res = {}

    # Non-finite
    bad_nf = ~np.isfinite(arr)
    if bad_nf.any():
        res["non_finite_idx"] = np.flatnonzero(bad_nf)

    # Out of range (allow eps tolerance)
    bad_oor = (arr < -eps) | (arr > 1.0 + eps)
    if bad_oor.any():
        res["out_of_range_idx"] = np.flatnonzero(bad_oor)

    if binary_only:
        close_to_0 = np.abs(arr - 0.0) <= eps
        close_to_1 = np.abs(arr - 1.0) <= eps
        bad_nb = ~(close_to_0 | close_to_1)
        if bad_nb.any():
            res["non_binary_idx"] = np.flatnonzero(bad_nb)

    return res

def scan_ply(fp: Path, eps: float, binary_only: bool):
    """
    Read a single PLY and return issues (list of dicts).
    Issue format:
      {"file":..., "issue":..., "n_vertices": N, "idx0": i0, "idx1": i1, "value": v}
    Returns empty list if no issues.
    """
    issues = []
    try:
        ply = PlyData.read(str(fp))
    except Exception as e:
        issues.append({
            "file": str(fp),
            "issue": f"read_error:{type(e).__name__}",
            "n_vertices": None,
            "idx0": None,
            "idx1": None,
            "value": None,
        })
        return issues

    if "vertex" not in ply.elements:
        issues.append({
            "file": str(fp),
            "issue": "no_vertex_element",
            "n_vertices": None, "idx0": None, "idx1": None, "value": None
        })
        return issues

    v = ply["vertex"].data
    n = len(v)

    key = find_label_key(v)
    if key is None:
        issues.append({
            "file": str(fp),
            "issue": "missing_label_property",
            "n_vertices": n, "idx0": None, "idx1": None, "value": None
        })
        return issues

    labels = np.asarray(v[key]).astype(np.float64, copy=False)

    # Basic checks
    res = check_labels(labels, eps=eps, binary_only=binary_only)

    def add_rows(tag, idx_array):
        for i in idx_array:
            val = labels[i]
            issues.append({
                "file": str(fp),
                "issue": tag,
                "n_vertices": n,
                "idx0": int(i),           # 0-based
                "idx1": int(i) + 1,       # 1-based
                "value": float(val) if np.isfinite(val) else str(val),
            })

    # Sort by severity: non_finite -> out_of_range -> non_binary
    if "non_finite_idx" in res:
        add_rows("label_non_finite", res["non_finite_idx"])
    if "out_of_range_idx" in res:
        add_rows("label_out_of_[0,1]", res["out_of_range_idx"])
    if binary_only and "non_binary_idx" in res:
        add_rows("label_not_binary_{0,1}", res["non_binary_idx"])

    return issues

def find_ply_files(root: Path, pattern: str, recursive: bool):
    if recursive:
        return sorted(root.rglob(pattern))
    else:
        return sorted(root.glob(pattern))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Root directory to scan")
    ap.add_argument("--out", type=str, default="ply_label_report.csv", help="CSV report output path")
    ap.add_argument("--pattern", type=str, default="*.ply", help="Filename pattern (default: *.ply)")
    ap.add_argument("--recursive", action="store_true", help="Scan subdirectories")
    ap.add_argument("--binary-only", action="store_true", help="Enforce strict binary labels {0,1}")
    ap.add_argument("--eps", type=float, default=1e-6, help="Tolerance for float comparisons")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    files = find_ply_files(root, args.pattern, args.recursive)

    if not files:
        print(f"[INFO] No files matched {args.pattern} under: {root}")
        return 0

    total_files = 0
    bad_files = 0
    total_issues = 0

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "issue", "n_vertices", "idx0", "idx1", "value"])
        w.writeheader()

        for fp in tqdm(files):
            total_files += 1
            issues = scan_ply(fp, eps=args.eps, binary_only=args.binary_only)
            if issues:
                bad_files += 1
                total_issues += len(issues)
                for row in issues:
                    w.writerow(row)

    print(f"[DONE] Scan complete: total files={total_files}, files with issues={bad_files}, total issues={total_issues}")
    print(f"[REPORT] CSV saved to: {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
