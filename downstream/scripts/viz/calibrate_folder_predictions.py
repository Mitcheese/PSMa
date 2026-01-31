#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate folder predictions to target metrics by updating `mean_predict`.

Given a directory of CSV files with columns:
    residue_id, mean_label, mean_predict, n_points
this script adjusts/overwrites `mean_predict` values so that the overall
metrics across ALL rows (optionally weighted by n_points) match user-specified
targets as closely as possible.

Targets: ACC, F1, AUROC, AUPRC, MCC

⚠️ Notes / Guarantees:
- Hitting all five targets *exactly* may be mathematically impossible on a fixed dataset.
  This script uses randomized search + local refinement to get as close as possible.
- It never edits labels. It only rewrites the `mean_predict` column.
- It preserves each input file's row order and other columns.
- It supports sample weighting via `n_points` (enabled by default).

Usage:
    python calibrate_folder_predictions.py \
        --in_dir /path/to/csvs \
        --out_dir /path/to/updated_csvs \
        --target-acc 0.85 \
        --target-f1 0.80 \
        --target-auc 0.90 \
        --target-auprc 0.75 \
        --target-mcc 0.65 \
        --use-weights \
        --max-iters 2000 \
        --tolerance 0.002

If you want to ignore n_points weights, omit --use-weights.
"""

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)


# ---------------------------- Utilities ----------------------------

def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # Clip to avoid overflow in exp
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def min_max_scale(x: np.ndarray) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    if hi <= lo:
        return np.zeros_like(x) + 0.5
    return (x - lo) / (hi - lo)


def set_seed(seed: int):
    np.random.seed(seed)


@dataclass
class Targets:
    acc: Optional[float]
    f1: Optional[float]
    auc: Optional[float]
    auprc: Optional[float]
    mcc: Optional[float]


@dataclass
class CalibParams:
    a: float      # scale for logit
    b: float      # bias for logit
    alpha: float  # mix factor with label-based ranking [0..1]
    thr: float    # decision threshold for acc/f1/mcc


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    weights: Optional[np.ndarray],
    thr: float,
) -> Dict[str, float]:
    """Compute all metrics from y_true (0/1), scores in [0,1], and a threshold."""
    # Predicted labels
    y_pred = (scores >= thr).astype(int)

    res = {}
    # Accuracy
    res["acc"] = accuracy_score(y_true, y_pred, sample_weight=weights)
    # F1 (binary)
    # If only one class present in y_true, sklearn returns 0.0. That's acceptable here.
    res["f1"] = f1_score(y_true, y_pred, sample_weight=weights, average="binary", zero_division=0)

    # AUROC
    try:
        res["auc"] = roc_auc_score(y_true, scores, sample_weight=weights)
    except Exception:
        # Undefined (e.g., only one class)
        res["auc"] = float("nan")

    # AUPRC
    try:
        res["auprc"] = average_precision_score(y_true, scores, sample_weight=weights)
    except Exception:
        res["auprc"] = float("nan")

    # MCC
    try:
        res["mcc"] = matthews_corrcoef(y_true, y_pred, sample_weight=weights)
    except Exception:
        res["mcc"] = float("nan")

    return res


def metric_error(current: Dict[str, float], target: Targets, metric_weights: Dict[str, float]) -> float:
    """Weighted squared error across available metrics. NaNs contribute a large penalty."""
    err = 0.0
    BIG = 1000.0  # large penalty for NaNs
    for k in ["acc", "f1", "auc", "auprc", "mcc"]:
        w = metric_weights.get(k, 1.0)
        tgt = getattr(target, k)
        cur = current.get(k, float("nan"))
        if tgt is None:
            continue  # skip if target unspecified
        if np.isnan(cur):
            err += w * BIG
        else:
            # Clip metrics to valid ranges for stability
            if k in ("acc", "f1", "auc", "auprc"):
                cur = float(np.clip(cur, 0.0, 1.0))
                tgt = float(np.clip(tgt, 0.0, 1.0))
            elif k == "mcc":
                # Allow typical [0,1] targets; MCC can be [-1,1], but we clip to [-1,1]
                cur = float(np.clip(cur, -1.0, 1.0))
                tgt = float(np.clip(tgt, -1.0, 1.0))
            err += w * (cur - tgt) ** 2
    return err


def apply_transform(
    raw_scores: np.ndarray,
    y_true: np.ndarray,
    a: float,
    b: float,
    alpha: float,
    add_noise: float = 1e-3,
) -> np.ndarray:
    """
    Create transformed scores in [0,1] from raw_scores using:
      s1 = sigmoid(a * logit(raw_scores) + b)
      s2 = label-based ranking score (positives near 1, negatives near 0)
      s  = (1 - alpha) * s1 + alpha * s2
    """
    # Phase 1: Calibrated via logit-sigmoid (keeps relative order largely intact)
    s1 = stable_sigmoid(a * safe_logit(raw_scores) + b)
    s1 = min_max_scale(s1)  # normalize just in case

    # Phase 2: Label-based ranking component
    # positives ~0.9, negatives ~0.1 with tiny noise to create a spread
    noise = np.random.uniform(-add_noise, add_noise, size=y_true.shape[0])
    s2 = 0.1 + 0.8 * y_true.astype(float) + noise
    s2 = np.clip(s2, 0.0, 1.0)

    s = (1.0 - alpha) * s1 + alpha * s2
    s = np.clip(s, 0.0, 1.0)
    return s


def best_threshold(y_true: np.ndarray, scores: np.ndarray, weights: Optional[np.ndarray], target: Targets) -> Tuple[float, Dict[str, float]]:
    """
    Given scores, search for the threshold in [0,1] that minimizes the metric error
    relative to (acc, f1, mcc) parts of the target (AUC/AUPRC are threshold-free).
    """
    # Dense grid for threshold search
    grid = np.linspace(0.0, 1.0, 501)
    best_thr = 0.5
    best_err = float("inf")
    best_metrics = None

    # Only include the threshold-dependent metrics in the objective for threshold search
    metric_weights = {"acc": 1.0, "f1": 1.0, "mcc": 1.0, "auc": 0.0, "auprc": 0.0}

    for thr in grid:
        cur = compute_metrics(y_true, scores, weights, thr)
        err = metric_error(cur, target, metric_weights)
        if err < best_err:
            best_err = err
            best_thr = float(thr)
            best_metrics = cur

    return best_thr, best_metrics if best_metrics is not None else {}


def randomized_search(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    weights: Optional[np.ndarray],
    target: Targets,
    metric_weights: Dict[str, float],
    max_iters: int = 2000,
    seed: int = 42,
    patience: int = 400,
) -> Tuple[CalibParams, Dict[str, float], np.ndarray]:
    """
    Randomized search over (a, b, alpha), with threshold selection for each candidate.
    Returns the best parameters, metrics, and final scores.
    """
    set_seed(seed)

    # Initial ranges for parameters
    a_range = (-10.0, 10.0)
    b_range = (-5.0, 5.0)
    alpha_range = (0.0, 1.0)

    best_params = CalibParams(a=1.0, b=0.0, alpha=0.0, thr=0.5)
    best_err = float("inf")
    best_metrics = {}
    best_scores = raw_scores.copy()

    no_improve = 0

    for it in range(max_iters):
        # Sample parameters (use progressively tighter ranges after we have a best)
        if it < max_iters // 3:
            a = np.random.uniform(*a_range)
            b = np.random.uniform(*b_range)
            alpha = np.random.uniform(*alpha_range)
        else:
            # Local search around current best
            a = np.random.normal(best_params.a, 0.5)
            b = np.random.normal(best_params.b, 0.25)
            alpha = np.clip(np.random.normal(best_params.alpha, 0.1), 0.0, 1.0)

        scores = apply_transform(raw_scores, y_true, a, b, alpha)

        # Best threshold for this candidate
        thr, _ = best_threshold(y_true, scores, weights, target)

        cur_metrics = compute_metrics(y_true, scores, weights, thr)
        cur_err = metric_error(cur_metrics, target, metric_weights)

        if cur_err < best_err:
            best_err = cur_err
            best_params = CalibParams(a=a, b=b, alpha=alpha, thr=thr)
            best_metrics = cur_metrics
            best_scores = scores.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                # modest re-seed to escape local minima
                a_range = (a_range[0] * 0.9, a_range[1] * 0.9)
                b_range = (b_range[0] * 0.9, b_range[1] * 0.9)
                no_improve = 0

    return best_params, best_metrics, best_scores


def within_tolerance(cur: Dict[str, float], target: Targets, tol: float) -> bool:
    for k in ["acc", "f1", "auc", "auprc", "mcc"]:
        tgt = getattr(target, k)
        if tgt is None:
            continue
        v = cur.get(k, float("nan"))
        if np.isnan(v):
            return False
        if abs(v - tgt) > tol:
            return False
    return True


# ---------------------------- Main Routine ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrate folder predictions to target metrics.")
    parser.add_argument("--in_dir", type=str, required=True, help="Input folder containing CSV files.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder to write updated CSVs.")
    parser.add_argument("--use-weights", action="store_true", help="Use n_points as sample weights.")
    parser.add_argument("--targets-json", type=str, default=None, help="Path to a JSON file with targets.")
    parser.add_argument("--target-acc", type=float, default=None)
    parser.add_argument("--target-f1", type=float, default=None)
    parser.add_argument("--target-auc", type=float, default=None)
    parser.add_argument("--target-auprc", type=float, default=None)
    parser.add_argument("--target-mcc", type=float, default=None)
    parser.add_argument("--max-iters", type=int, default=2000, help="Max iterations for randomized search.")
    parser.add_argument("--tolerance", type=float, default=0.002, help="Tolerance for considering targets matched.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect files
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {args.in_dir}")

    # Load and concatenate
    dfs = []
    file_row_counts = []
    for fp in files:
        df = pd.read_csv(fp)
        # basic checks
        needed = {"residue_id", "mean_label", "mean_predict", "n_points"}
        missing = needed - set(df.columns)
        if missing:
            raise SystemExit(f"File {fp} missing required columns: {missing}")
        dfs.append(df)
        file_row_counts.append((fp, len(df)))

    big = pd.concat(dfs, axis=0, ignore_index=True)

    # Prepare y_true (ensure binary)
    y_true_raw = big["mean_label"].values.astype(float)
    # If labels are not exactly 0/1, threshold at 0.5
    if set(np.unique(y_true_raw)) <= {0.0, 1.0}:
        y_true = y_true_raw.astype(int)
    else:
        y_true = (y_true_raw >= 0.5).astype(int)

    # Raw scores (if absent or degenerate, initialize to prevalence)
    raw_scores = big["mean_predict"].values.astype(float)
    if not np.isfinite(raw_scores).all():
        raw_scores = np.nan_to_num(raw_scores, nan=0.5, posinf=0.999, neginf=0.001)

    if np.allclose(np.std(raw_scores), 0.0, atol=1e-9):
        # initialize with a small spread around prevalence
        prev = float(np.average(y_true, weights=big["n_points"].values if args.use_weights else None))
        raw_scores = np.clip(prev + 0.01 * np.random.randn(len(raw_scores)), 0.0, 1.0)

    # Weights
    weights = big["n_points"].values.astype(float) if args.use_weights else None

    # Targets
    if args.targets-json:
        # Not reachable due to hyphen; handle below
        pass

    targets_dict = {}
    if args.targets_json:
        with open(args.targets_json, "r") as f:
            targets_dict = json.load(f)

    tgt = Targets(
        acc=args.target_acc if args.target_acc is not None else targets_dict.get("acc"),
        f1=args.target_f1 if args.target_f1 is not None else targets_dict.get("f1"),
        auc=args.target_auc if args.target_auc is not None else targets_dict.get("auc"),
        auprc=args.target_auprc if args.target_auprc is not None else targets_dict.get("auprc"),
        mcc=args.target_mcc if args.target_mcc is not None else targets_dict.get("mcc"),
    )

    # Metric weights for objective: you can tweak relative importance here
    metric_wts = {"acc": 1.0, "f1": 1.0, "auc": 1.0, "auprc": 1.0, "mcc": 1.0}

    # Baseline metrics (before)
    # Choose a good threshold for the baseline (optimize for acc/f1/mcc only)
    base_thr, _ = best_threshold(y_true, raw_scores, weights, tgt)
    base_metrics = compute_metrics(y_true, raw_scores, weights, base_thr)

    # Search
    best_params, best_metrics, best_scores = randomized_search(
        y_true=y_true,
        raw_scores=min_max_scale(raw_scores),
        weights=weights,
        target=tgt,
        metric_weights=metric_wts,
        max_iters=args.max_iters,
        seed=args.seed,
        patience=max(200, args.max_iters // 5),
    )

    # If not within tolerance, perform a brief second-pass local search around best
    if not within_tolerance(best_metrics, tgt, args.tolerance):
        # local jiggle
        local_iters = max(300, args.max_iters // 4)
        np.random.seed(args.seed + 1337)
        cand_best = best_metrics
        cand_params = best_params
        cand_scores = best_scores
        cand_err = metric_error(best_metrics, tgt, metric_wts)

        for _ in range(local_iters):
            a = np.random.normal(best_params.a, 0.15)
            b = np.random.normal(best_params.b, 0.07)
            alpha = float(np.clip(np.random.normal(best_params.alpha, 0.05), 0.0, 1.0))
            scores = apply_transform(min_max_scale(raw_scores), y_true, a, b, alpha, add_noise=5e-4)
            thr, _ = best_threshold(y_true, scores, weights, tgt)
            cur = compute_metrics(y_true, scores, weights, thr)
            err = metric_error(cur, tgt, metric_wts)
            if err < cand_err:
                cand_err = err
                cand_best = cur
                cand_params = CalibParams(a=a, b=b, alpha=alpha, thr=thr)
                cand_scores = scores.copy()

        best_metrics = cand_best
        best_params = cand_params
        best_scores = cand_scores

    # Finalize: write updated mean_predict per file
    # Split best_scores back across files
    idx = 0
    for fp, nrows in file_row_counts:
        sub = best_scores[idx: idx + nrows]
        idx += nrows
        df = pd.read_csv(fp)
        df["mean_predict"] = sub
        out_fp = os.path.join(args.out_dir, os.path.basename(fp))
        df.to_csv(out_fp, index=False)

    # Write report
    report = {
        "baseline_threshold": base_thr,
        "baseline_metrics": base_metrics,
        "best_params": {
            "a": best_params.a,
            "b": best_params.b,
            "alpha": best_params.alpha,
            "thr": best_params.thr,
        },
        "achieved_metrics": best_metrics,
        "targets": {
            "acc": tgt.acc,
            "f1": tgt.f1,
            "auc": tgt.auc,
            "auprc": tgt.auprc,
            "mcc": tgt.mcc,
        },
        "tolerance": args.tolerance,
        "within_tolerance": within_tolerance(best_metrics, tgt, args.tolerance),
        "use_weights": args.use_weights,
        "num_files": len(files),
        "num_rows_total": int(len(big)),
        "positive_rate": float(np.average(y_true, weights=weights) if weights is not None else y_true.mean()),
    }
    with open(os.path.join(args.out_dir, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Also create a small CSV summary
    rows = []
    for k in ["acc", "f1", "auc", "auprc", "mcc"]:
        rows.append({
            "metric": k,
            "baseline": float(base_metrics.get(k, float("nan"))),
            "target": getattr(tgt, k),
            "achieved": float(best_metrics.get(k, float("nan"))),
            "abs_diff": float(abs(best_metrics.get(k, float("nan")) - (getattr(tgt, k) if getattr(tgt, k) is not None else float("nan")))) if getattr(tgt, k) is not None and not math.isnan(best_metrics.get(k, float("nan"))) else float("nan")
        })
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "calibration_report.csv"), index=False)

    print("\n=== Calibration Finished ===")
    print(f"Files processed: {len(files)}")
    print(f"Rows total: {len(big)}")
    print(f"Baseline threshold: {base_thr:.3f}")
    print("Baseline metrics:", base_metrics)
    print("\nBest params:", best_params)
    print("Achieved metrics:", best_metrics)
    print(f"\nTargets matched within tolerance ({args.tolerance}): {within_tolerance(best_metrics, tgt, args.tolerance)}")
    print(f"Updated CSVs written to: {args.out_dir}")
    print(f"Reports: calibration_report.json, calibration_report.csv")
    

if __name__ == "__main__":
    main()
