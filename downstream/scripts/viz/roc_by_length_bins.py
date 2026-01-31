#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split files by n_rows into three bins (1–100, 101–200, >200).
For each bin, read and concatenate CSVs pointed by 'file' (label/predict columns),
compute ROC/AUROC, and plot all three curves on one figure.

Dependencies: pandas, numpy, scikit-learn, matplotlib
Install: pip install pandas numpy scikit-learn matplotlib
Usage: python scripts/viz/roc_by_length_bins.py --summary master.csv --out roc_bins.png
Optional: --base-dir for relative file paths (defaults to master.csv directory).
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

LABEL_CANDIDATES = ["label", "y", "gt", "mean_label"]
PRED_CANDIDATES  = ["predict", "prob", "score", "mean_predict", "pred"]

def find_col(df, candidates, name_for_err):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Missing required column: {name_for_err}, candidates={candidates}, columns={list(df.columns)}")

def load_one_file(fp):
    # Read a child CSV and return (y_true, y_score)
    df = pd.read_csv(fp)
    y_col = find_col(df, LABEL_CANDIDATES, "label")
    p_col = find_col(df, PRED_CANDIDATES, "predict/prob/score")
    y = df[y_col].values
    p = df[p_col].values
    # If labels are float, binarize; use 0.5 threshold for non-{0,1} values
    if not set(np.unique(y)).issubset({0, 1}):
        y = (y >= 0.5).astype(int)
    return y, p

def bin_name(n_rows):
    # 1–100, 101–200, >200
    if 1 <= n_rows <= 100:
        return "1–100"
    elif 101 <= n_rows <= 200:
        return "101–200"
    elif n_rows > 200:
        return ">200"
    else:
        return None  # filter n_rows<=0 or missing

def main(args):
    summary = pd.read_csv(args.summary)
    if "file" not in summary.columns or "n_rows" not in summary.columns:
        raise ValueError("master CSV must include columns: file, n_rows")

    # Base directory: default to master.csv directory
    base_dir = args.base_dir or os.path.dirname(os.path.abspath(args.summary))

    # Bin by n_rows
    summary["bin"] = summary["n_rows"].apply(lambda x: bin_name(int(x)) if pd.notna(x) else None)
    summary = summary.dropna(subset=["bin"])

    bins = ["1–100", "101–200", ">200"]
    roc_data = {}  # bin -> (fpr, tpr, auc, n_files, n_samples)

    for b in bins:
        sub = summary[summary["bin"] == b]
        if sub.empty:
            continue

        ys, ps = [], []
        n_files = 0
        for _, row in sub.iterrows():
            f = str(row["file"]).strip()
            fp = f if os.path.isabs(f) else os.path.join(base_dir, f)
            if not os.path.exists(fp):
                print(f"[WARN] File not found, skip: {fp}")
                continue
            try:
                y, p = load_one_file(fp)
                if len(y) > 0 and len(p) > 0:
                    ys.append(y)
                    ps.append(p)
                    n_files += 1
            except Exception as e:
                print(f"[WARN] Read failed, skip: {fp} | reason: {e}")

        if n_files == 0:
            continue

        y_all = np.concatenate(ys, axis=0)
        p_all = np.concatenate(ps, axis=0)

        fpr, tpr, _ = roc_curve(y_all, p_all)
        au = auc(fpr, tpr)
        roc_data[b] = (fpr, tpr, au, n_files, len(y_all))

    # Plot
    if not roc_data:
        raise RuntimeError("No valid data to plot ROC. Check file paths and column names.")

    plt.figure(figsize=(7, 6))
    for b in bins:
        if b in roc_data:
            fpr, tpr, au, n_files, n_samp = roc_data[b]
            plt.plot(fpr, tpr, lw=2, label=f"{b} (AUROC={au:.3f})")
    # Reference diagonal
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC by Length Bins (concat across files per bin)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path = args.out or "roc_by_length_bins.png"
    plt.savefig(out_path, dpi=300)
    print(f"[OK] Saved image: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Master summary CSV (columns: file, n_rows, ...)")
    ap.add_argument("--base-dir", default=None, help="Base dir for relative file paths (default: master CSV dir)")
    ap.add_argument("--out", default=None, help="Output image path (default: roc_by_length_bins.png)")
    args = ap.parse_args()
    main(args)
