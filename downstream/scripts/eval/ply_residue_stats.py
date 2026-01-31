#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Traverse .ply files in a folder (optionally recursive), read residue_id/label/predict,
group by residue_id to compute mean(label), mean(predict), and save CSVs.

Usage:
    python scripts/eval/ply_residue_stats.py INPUT_DIR OUTPUT_DIR [--recursive] [--overwrite]

Dependencies:
    pip install plyfile pandas numpy tqdm
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from plyfile import PlyData
except Exception as e:
    print("Missing plyfile dependency; install with: pip install plyfile", file=sys.stderr)
    raise

REQUIRED_FIELDS = ["residue_id", "label", "predict"]

def read_vertex_table(ply_path: Path) -> pd.DataFrame:
    """Read PLY vertex table and return residue_id/label/predict DataFrame."""
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise ValueError("PLY has no 'vertex' element")

    v = ply["vertex"].data  # numpy structured array

    # Field checks
    missing = [f for f in REQUIRED_FIELDS if f not in v.dtype.names]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Convert to DataFrame
    df = pd.DataFrame({name: v[name] for name in REQUIRED_FIELDS})

    # residue_id may be float (e.g., 148.0); round then cast to int for grouping
    # If residue_id is already integer with no float noise, astype(int) is fine.
    df["residue_id"] = np.rint(df["residue_id"]).astype(np.int64)

    # Cast label/predict to float for averaging.
    df["label"] = df["label"].astype(float)
    df["predict"] = df["predict"].astype(float)

    # Drop missing values
    df = df.dropna(subset=["residue_id", "label", "predict"])

    return df

def group_by_residue(df: pd.DataFrame) -> pd.DataFrame:
    """Group by residue_id and compute mean(label), mean(predict), and counts."""
    grouped = (
        df.groupby("residue_id", as_index=False)
          .agg(mean_label=("label", "mean"),
               mean_predict=("predict", "mean"),
               n_points=("label", "size"))
    )
    # Column order
    grouped = grouped[["residue_id", "mean_label", "mean_predict", "n_points"]]
    return grouped

def process_one_file(ply_path: Path, out_dir: Path, overwrite: bool = False) -> bool:
    """Process one PLY file, write CSV, and return success."""
    try:
        df = read_vertex_table(ply_path)
        result = group_by_residue(df)
    except Exception as e:
        print(f"[SKIP] Read failed: {ply_path} -> {e}", file=sys.stderr)
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / (ply_path.stem + ".csv")

    if out_csv.exists() and not overwrite:
        print(f"[SKIP] CSV exists (use --overwrite to replace): {out_csv}")
        return True

    try:
        # Export only required columns; keep n_points if needed.
        # result[["residue_id", "mean_label", "mean_predict"]].to_csv(out_csv, index=False)
        result.to_csv(out_csv, index=False)  # includes n_points
        return True
    except Exception as e:
        print(f"[FAIL] CSV write failed: {out_csv} -> {e}", file=sys.stderr)
        return False

def collect_ply_files(root: Path, recursive: bool = False):
    if recursive:
        return list(root.rglob("*.ply"))
    else:
        return list(root.glob("*.ply"))

def main():
    parser = argparse.ArgumentParser(description="Aggregate residue-level stats from PLY and export CSV.")
    parser.add_argument("input_dir", type=str, help="Input folder (contains .ply)")
    parser.add_argument("output_dir", type=str, help="Output CSV folder")
    parser.add_argument("--recursive", action="store_true", help="Traverse subdirectories")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV")
    args = parser.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"Input directory does not exist or is not a directory: {in_dir}", file=sys.stderr)
        sys.exit(1)

    ply_files = collect_ply_files(in_dir, recursive=args.recursive)
    if not ply_files:
        print(f"No .ply found in {in_dir} (recursive={args.recursive})", file=sys.stderr)
        sys.exit(1)

    ok, fail = 0, 0
    for ply_path in tqdm(ply_files, desc="Processing PLY", unit="file"):
        if process_one_file(ply_path, out_dir, overwrite=args.overwrite):
            ok += 1
        else:
            fail += 1

    print(f"Done: {ok} succeeded, {fail} failed. Output dir: {out_dir}")

if __name__ == "__main__":
    main()
