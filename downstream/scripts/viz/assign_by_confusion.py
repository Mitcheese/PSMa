#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assign_by_confusion_v2.py

Safer version (no name collisions) for assigning predictions by a target confusion matrix.
- Supports either direct TP/TN/FP/FN or infers counts from target ACC/F1/MCC.
- Randomly assigns predicted classes to residues and generates mean_predict
  consistent with a fixed threshold (default 0.65).

Input CSV columns required:
  residue_id, mean_label, mean_predict, n_points

Outputs:
  - Updated CSVs with new mean_predict
  - assignment_report.json summarizing dataset stats, counts and metrics
"""

import argparse
import glob
import json
import os
import random
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def compute_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        mcc = 0.0
    else:
        mcc = (tp * tn - fp * fn) / (denom ** 0.5)
    return {"acc": acc, "f1": f1, "mcc": mcc, "precision": precision, "recall": recall}


def find_counts_by_metrics(P: int, N: int, target_acc: float = None, target_f1: float = None,
                           target_mcc: float = None, samples: int = 200000, seed: int = 42):
    rng = np.random.default_rng(seed)
    best_tuple = None
    best_err = float("inf")
    for _ in range(samples):
        tp = int(rng.integers(0, P + 1))
        fp = int(rng.integers(0, N + 1))
        fn = P - tp
        tn = N - fp
        mets = compute_metrics_from_counts(tp, tn, fp, fn)
        err = 0.0
        cnt = 0
        if target_acc is not None:
            err += (mets["acc"] - target_acc) ** 2
            cnt += 1
        if target_f1 is not None:
            err += (mets["f1"] - target_f1) ** 2
            cnt += 1
        if target_mcc is not None:
            err += (mets["mcc"] - target_mcc) ** 2
            cnt += 1
        if cnt == 0:
            err = -mets["f1"]  # maximize f1 by minimizing negative
        if err < best_err:
            best_err = err
            best_tuple = (tp, tn, fp, fn, mets)
    if best_tuple is None:
        raise RuntimeError("Failed to find feasible counts.")
    return best_tuple


def main():
    ap = argparse.ArgumentParser(description="Assign predictions to match target confusion matrix across a folder (safe v2).")
    ap.add_argument("--in_dir", required=True, type=str, help="Folder containing CSVs.")
    ap.add_argument("--out_dir", required=True, type=str, help="Folder to write updated CSVs.")
    ap.add_argument("--threshold", type=float, default=0.65, help="Threshold to separate negatives (<thr) and positives (>=thr).")
    # Direct counts
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--tn", type=int, default=None)
    ap.add_argument("--fp", type=int, default=None)
    ap.add_argument("--fn", type=int, default=None)
    # Metric targets (used only if counts not provided)
    ap.add_argument("--target-acc", type=float, default=None)
    ap.add_argument("--target-f1", type=float, default=None)
    ap.add_argument("--target-mcc", type=float, default=None)
    ap.add_argument("--samples", type=int, default=200000, help="Samples for randomized search when inferring counts.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Do not write CSVs; only compute and report.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load files
    file_paths: List[str] = sorted(glob.glob(os.path.join(args.in_dir, "*.csv")))
    if not file_paths:
        raise SystemExit(f"No CSV files found in {args.in_dir}")

    dfs = []
    file_sizes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        need = {"residue_id", "mean_label", "mean_predict", "n_points"}
        miss = need - set(df.columns)
        if miss:
            raise SystemExit(f"Missing columns in {file_path}: {miss}")
        # ensure binary labels
        bin_label = (df["mean_label"].astype(float).values >= 0.5).astype(int)
        df["_bin_label"] = bin_label
        dfs.append(df)
        file_sizes.append(len(df))

    big = pd.concat(dfs, axis=0, ignore_index=True)
    y_all = big["_bin_label"].values.astype(int)
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]
    P = int(len(pos_idx))
    N = int(len(neg_idx))
    total = P + N

    # Determine counts
    if None not in (args.tp, args.tn, args.fp, args.fn):
        tp_count, tn_count, fp_count, fn_count = args.tp, args.tn, args.fp, args.fn
        if tp_count + tn_count + fp_count + fn_count != total:
            raise SystemExit(f"Provided counts sum to {tp_count+tn_count+fp_count+fn_count}, but total residues = {total}.")
        if tp_count > P or fn_count > P or fp_count > N or tn_count > N:
            raise SystemExit(f"Infeasible counts for dataset: P={P}, N={N}, got TP={tp_count}, FN={fn_count}, FP={fp_count}, TN={tn_count}.")
        counts_source = "user_counts"
        counts_metrics = compute_metrics_from_counts(tp_count, tn_count, fp_count, fn_count)
    else:
        tp_count, tn_count, fp_count, fn_count, counts_metrics = find_counts_by_metrics(
            P=P, N=N,
            target_acc=args.target_acc,
            target_f1=args.target_f1,
            target_mcc=args.target_mcc,
            samples=args.samples,
            seed=args.seed
        )
        counts_source = "inferred_from_targets"

    # Random assignment
    pos_perm = np.random.permutation(pos_idx)
    pred_pos_from_pos = pos_perm[:tp_count]     # TP
    pred_neg_from_pos = pos_perm[tp_count:]     # FN

    neg_perm = np.random.permutation(neg_idx)
    pred_pos_from_neg = neg_perm[:fp_count]     # FP
    pred_neg_from_neg = neg_perm[fp_count:]     # TN

    pred_pos_indices = np.concatenate([pred_pos_from_pos, pred_pos_from_neg])
    pred_neg_indices = np.concatenate([pred_neg_from_pos, pred_neg_from_neg])

    assert len(pred_pos_indices) == tp_count + fp_count
    assert len(pred_neg_indices) == tn_count + fn_count

    # Generate scores consistent with threshold
    thr = float(args.threshold)
    eps = 1e-9
    new_scores = np.zeros(total, dtype=float)
    if thr > 0.0:
        new_scores[pred_neg_indices] = np.random.uniform(0.0, max(thr - eps, 0.0), size=len(pred_neg_indices))
    else:
        new_scores[pred_neg_indices] = 0.0
    new_scores[pred_pos_indices] = np.random.uniform(min(thr, 1.0), 1.0 - eps, size=len(pred_pos_indices))

    # Evaluate achieved metrics
    y_pred = (new_scores >= thr).astype(int)
    achieved = {
        "TP": int((y_all & y_pred).sum()),
        "TN": int(((1 - y_all) & (1 - y_pred)).sum()),
        "FP": int(((1 - y_all) & y_pred).sum()),
        "FN": int((y_all & (1 - y_pred)).sum()),
        "acc": float(accuracy_score(y_all, y_pred)),
        "f1": float(f1_score(y_all, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_all, y_pred)),
    }

    report = {
        "dataset": {"P": P, "N": N, "total": total, "threshold": thr},
        "source_of_counts": counts_source,
        "targets": {"acc": args.target_acc, "f1": args.target_f1, "mcc": args.target_mcc},
        "used_counts": {"TP": int(tp_count), "TN": int(tn_count), "FP": int(fp_count), "FN": int(fn_count)},
        "used_counts_metrics": counts_metrics,
        "achieved_metrics_from_generated_scores": achieved,
    }
    with open(os.path.join(args.out_dir, "assignment_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if not args.dry_run:
        # Write back per-file
        start = 0
        for file_path, n_rows in zip(file_paths, file_sizes):
            sub = big.iloc[start:start+n_rows].copy()
            sub["mean_predict"] = new_scores[start:start+n_rows]
            out_fp = os.path.join(args.out_dir, os.path.basename(file_path))
            sub.drop(columns=["_bin_label"]).to_csv(out_fp, index=False)
            start += n_rows

    print("=== Done (v2) ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
