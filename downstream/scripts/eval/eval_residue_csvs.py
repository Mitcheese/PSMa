#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate residue-level CSVs and compute ACC, F1, AUROC, AUPRC, and MCC per file.
Each input CSV must have columns: residue_id, mean_label, mean_predict, n_points.

Usage example:
    python scripts/eval/eval_residue_csvs.py \
        --input-dir /path/to/csv_folder \
        --output-dir /path/to/save_results \
        --pred-threshold 0.5 \
        --label-threshold 0.5 \
        --weighted \
        --sort-by auprc  # sort by AUPRC descending (default)

Notes:
- mean_label will be binarized by label-threshold (default 0.5).
- mean_predict is treated as a probability score; for ACC/F1/MCC we threshold by pred-threshold (default 0.5).
- If --weighted is set, all metrics use n_points as sample weights.
- AUROC/AUPRC are undefined if only one class is present; in that case they are set to NaN.
- Sorting: by default descending; add --ascending to sort from small to large.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

REQUIRED_COLS = ["residue_id", "mean_label", "mean_predict", "n_points"]


def binarize_series(series: pd.Series, threshold: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return (s >= threshold).astype(int)


def safe_float_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.astype(float)


def compute_metrics_for_df(
    df: pd.DataFrame, pred_th: float, label_th: float, weighted: bool
) -> dict:
    # Drop rows with missing required values
    df = df.copy()
    df = df[REQUIRED_COLS].dropna()
    if df.empty:
        return {
            "n_rows": 0,
            "pos_rate": np.nan,
            "acc": np.nan,
            "f1": np.nan,
            "auroc": np.nan,
            "auprc": np.nan,
            "mcc": np.nan,
        }

    y_true = binarize_series(df["mean_label"], label_th).to_numpy()
    y_score = safe_float_series(df["mean_predict"]).clip(0.0, 1.0).to_numpy()
    y_pred = (y_score >= pred_th).astype(int)

    sw = None
    if weighted:
        sw = safe_float_series(df["n_points"]).to_numpy()

    # Metrics
    try:
        acc = accuracy_score(y_true, y_pred, sample_weight=sw)
    except Exception:
        acc = np.nan

    try:
        f1 = f1_score(y_true, y_pred, sample_weight=sw)
    except Exception:
        f1 = np.nan

    try:
        auroc = roc_auc_score(y_true, y_score, sample_weight=sw)
    except Exception:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_true, y_score, sample_weight=sw)
    except Exception:
        auprc = np.nan

    try:
        mcc = matthews_corrcoef(y_true, y_pred, sample_weight=sw)
    except Exception:
        mcc = np.nan

    # Positive rate (prevalence)
    try:
        if sw is not None:
            pos_rate = float(np.average(y_true, weights=sw))
        else:
            pos_rate = float(np.mean(y_true))
    except Exception:
        pos_rate = np.nan

    return {
        "n_rows": int(len(y_true)),
        "pos_rate": pos_rate,
        "acc": acc,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "mcc": mcc,
    }


def evaluate_dir(
    input_dir: Path,
    output_dir: Path,
    pred_th: float = 0.5,
    label_th: float = 0.5,
    weighted: bool = False,
    recursive: bool = False,
    output_name: str = "metrics_summary.csv",
    sort_by: Optional[str] = None,
    ascending: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Gather CSV files
    pattern = "**/*.csv" if recursive else "*.csv"
    csv_files = sorted(input_dir.glob(pattern))

    # For global-aggregate metrics, we can concatenate all data
    concat_frames = []

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to read {csv_path}: {e}\n")
            continue

        # Check required columns
        if not all(c in df.columns for c in REQUIRED_COLS):
            sys.stderr.write(
                f"[WARN] {csv_path} missing required columns. Found: {list(df.columns)}\n"
            )
            continue

        m = compute_metrics_for_df(df, pred_th, label_th, weighted)

        results.append(
            {
                "file": csv_path.name,
                "n_rows": m["n_rows"],
                "pos_rate": m["pos_rate"],
                "acc": m["acc"],
                "f1": m["f1"],
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "mcc": m["mcc"],
                "pred_threshold": pred_th,
                "label_threshold": label_th,
                "weighted_by_n_points": weighted,
            }
        )

        # Keep a trimmed copy for global aggregation
        trimmed = df[REQUIRED_COLS].dropna()
        if not trimmed.empty:
            trimmed["__src__"] = csv_path.name  # optional provenance
            concat_frames.append(trimmed)

    # Build summary DataFrame
    summary_df = pd.DataFrame(results)

    # Compute overall metrics across all files (if possible)
    if concat_frames:
        big = pd.concat(concat_frames, ignore_index=True)
        overall = compute_metrics_for_df(big, pred_th, label_th, weighted)
        overall_row = {
            "file": "__ALL__",
            "n_rows": overall["n_rows"],
            "pos_rate": overall["pos_rate"],
            "acc": overall["acc"],
            "f1": overall["f1"],
            "auroc": overall["auroc"],
            "auprc": overall["auprc"],
            "mcc": overall["mcc"],
            "pred_threshold": pred_th,
            "label_threshold": label_th,
            "weighted_by_n_points": weighted,
        }
        summary_df = pd.concat([pd.DataFrame([overall_row]), summary_df], ignore_index=True)

    # Apply sorting if requested (keep __ALL__ on top if present)
    if sort_by is not None:
        if sort_by not in summary_df.columns:
            sys.stderr.write(f"[WARN] sort-by column '{sort_by}' not found. Available: {list(summary_df.columns)}\n")
        else:
            if "file" in summary_df.columns:
                head = summary_df[summary_df["file"] == "__ALL__"]
                rest = summary_df[summary_df["file"] != "__ALL__"].copy()
                rest = rest.sort_values(by=sort_by, ascending=ascending, na_position="last", kind="mergesort")
                summary_df = pd.concat([head, rest], ignore_index=True)
            else:
                summary_df = summary_df.sort_values(by=sort_by, ascending=ascending, na_position="last", kind="mergesort")

    # Numeric formatting (optional: keep raw floats)
    # summary_df = summary_df.round(6)

    out_path = output_dir / output_name
    summary_df.to_csv(out_path, index=False)
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Compute metrics for residue-level CSVs.")
    p.add_argument("--input-dir", required=True, type=Path, help="Folder containing CSV files.")
    p.add_argument("--output-dir", required=True, type=Path, help="Folder to save summary CSV.")
    p.add_argument("--pred-threshold", type=float, default=0.5, help="Threshold for y_pred from mean_predict.")
    p.add_argument("--label-threshold", type=float, default=0.5, help="Threshold to binarize mean_label.")
    p.add_argument("--weighted", action="store_true", help="Use n_points as sample weights.")
    p.add_argument("--recursive", action="store_true", help="Search CSVs recursively.")
    p.add_argument("--output-name", type=str, default="metrics_summary.csv", help="Output CSV filename.")
    p.add_argument("--sort-by", type=str, default=None, help="Column to sort by (e.g., acc, f1, auroc, auprc, mcc, pos_rate, n_rows).")
    p.add_argument("--ascending", action="store_true", help="Sort ascending (default: descending).")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.input_dir.exists():
        print(f"[ERROR] input-dir does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    out_csv = evaluate_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pred_th=args.pred_threshold,
        label_th=args.label_threshold,
        weighted=args.weighted,
        recursive=args.recursive,
        output_name=args.output_name,
        sort_by=args.sort_by,
        ascending=args.ascending,
    )
    print(f"[OK] Saved summary to: {out_csv}")


if __name__ == "__main__":
    main()
