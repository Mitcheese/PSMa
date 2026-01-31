#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read residue-level CSVs (columns: residue_id, mean_label, mean_predict, n_points),
merge them, compute ACC / F1 / AUROC / AUPRC / MCC, and plot ROC.

Usage:
    python scripts/eval/eval_residue.py IN_DIR OUT_DIR [--label-th 0.5] [--pred-th 0.5]
                                        [--optimize {none,f1,mcc,youden}] [--weighted]
                                        [--recursive] [--per-file]

Dependencies:
    pip install pandas numpy scikit-learn matplotlib tqdm
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    accuracy_score, f1_score
)
import matplotlib.pyplot as plt


REQUIRED_COLS = ["residue_id", "mean_label", "mean_predict", "n_points"]


def load_one_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}")
    # Normalize data types
    df = df.copy()
    df["mean_label"] = df["mean_label"].astype(float)
    df["mean_predict"] = df["mean_predict"].astype(float)
    df["n_points"] = df["n_points"].astype(float)  # float sample weights are OK
    df["file_id"] = path.stem
    return df


def gather_frames(in_dir: Path, recursive: bool) -> pd.DataFrame:
    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(in_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV found under: {in_dir} (recursive={recursive})")
    frames = []
    for f in tqdm(files, desc="Loading CSVs", unit="file"):
        try:
            frames.append(load_one_csv(f))
        except Exception as e:
            print(f"[SKIP] {f}: {e}", file=sys.stderr)
    if not frames:
        raise RuntimeError("No valid CSVs (all failed or columns mismatched)")
    return pd.concat(frames, ignore_index=True)


def binarize_labels(mean_labels: np.ndarray, th: float) -> np.ndarray:
    # Use mean_label >= th as residue ground truth
    return (mean_labels >= th).astype(int)


def decisions_from_scores(scores: np.ndarray, th: float) -> np.ndarray:
    return (scores >= th).astype(int)


def safe_auc(y_true, y_score, sample_weight=None):
    # If only one class present, AUROC/AUPRC are undefined; return NaN
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, (np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf]))
    auroc = roc_auc_score(y_true, y_score, sample_weight=sample_weight)
    auprc = average_precision_score(y_true, y_score, sample_weight=sample_weight)
    fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=sample_weight)
    return auroc, auprc, (fpr, tpr, thresholds)


def weighted_confusion(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    sw = sample_weight.astype(float)
    tp = float(np.sum(sw[(y_true == 1) & (y_pred == 1)]))
    tn = float(np.sum(sw[(y_true == 0) & (y_pred == 0)]))
    fp = float(np.sum(sw[(y_true == 0) & (y_pred == 1)]))
    fn = float(np.sum(sw[(y_true == 1) & (y_pred == 0)]))
    return tp, tn, fp, fn


def mcc_from_confusion(tp, tn, fp, fn):
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0.0:
        return np.nan
    return (tp * tn - fp * fn) / denom


def find_best_threshold(y_true, y_score, sample_weight=None, mode="f1"):
    """
    Search best threshold on global data:
      - mode='f1': maximize F1
      - mode='mcc': maximize MCC
      - mode='youden': maximize (TPR - FPR)
    Returns (best_th, best_value)
    """
    # If classes are missing or scores are constant, fall back to 0.5
    uniq = np.unique(y_true)
    if len(uniq) < 2 or np.allclose(np.min(y_score), np.max(y_score)):
        return 0.5, np.nan

    # Candidate thresholds: unique scores plus 0.5 (deduped)
    cand = np.unique(y_score)
    cand = np.concatenate([cand, np.array([0.5])])
    cand = np.unique(cand)

    best_th, best_val = 0.5, -np.inf
    if mode in ("f1", "mcc"):
        for th in cand:
            y_pred = decisions_from_scores(y_score, th)
            if mode == "f1":
                try:
                    val = f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
                except Exception:
                    val = -np.inf
            else:
                tp, tn, fp, fn = weighted_confusion(y_true, y_pred, sample_weight)
                val = mcc_from_confusion(tp, tn, fp, fn)
                if np.isnan(val):
                    val = -np.inf
            if val > best_val:
                best_val, best_th = val, th
    elif mode == "youden":
        # ROC thresholds are more reasonable here
        auroc, auprc, (fpr, tpr, thresholds) = safe_auc(y_true, y_score, sample_weight)
        if np.isnan(auroc):
            return 0.5, np.nan
        youden = tpr - fpr
        idx = int(np.nanargmax(youden))
        best_th = float(thresholds[idx])
        best_val = float(youden[idx])
    else:
        return 0.5, np.nan

    return float(best_th), float(best_val)


def evaluate(y_true, y_score, sample_weight=None, pred_th=0.5):
    y_pred = decisions_from_scores(y_score, pred_th)

    # ACC / F1
    acc = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    f1 = f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)

    # MCC (manual, supports weights)
    tp, tn, fp, fn = weighted_confusion(y_true, y_pred, sample_weight)
    mcc = mcc_from_confusion(tp, tn, fp, fn)

    # AUROC / AUPRC
    auroc, auprc, (fpr, tpr, thresholds) = safe_auc(y_true, y_score, sample_weight)

    return {
        "ACC": acc,
        "F1": f1,
        "MCC": mcc,
        "AUROC": auroc,
        "AUPRC": auprc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "pred_th": pred_th,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def plot_roc(fpr, tpr, auroc, out_path: Path, title="ROC Curve"):
    plt.figure(figsize=(6, 5), dpi=140)
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auroc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Evaluate residue CSV metrics and plot ROC.")
    ap.add_argument("in_dir", type=str, help="Input CSV directory")
    ap.add_argument("out_dir", type=str, help="Output directory (plots and summary)")
    ap.add_argument("--label-th", type=float, default=0.5, help="Threshold for mean_label -> y_true")
    ap.add_argument("--pred-th", type=float, default=0.75, help="Threshold for mean_predict -> y_pred (overridden if optimize)")
    ap.add_argument("--optimize", choices=["none", "f1", "mcc", "youden"], default="none",
                    help="Criterion for global threshold search (default: none)")
    ap.add_argument("--weighted", action="store_true", help="Use n_points as sample_weight")
    ap.add_argument("--recursive", action="store_true", help="Read CSVs recursively")
    ap.add_argument("--per-file", action="store_true", help="Output per-file metrics to CSV")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = gather_frames(in_dir, args.recursive)
    # Build binary ground truth
    y_true = binarize_labels(df_all["mean_label"].values, args.label_th)
    y_score = df_all["mean_predict"].values
    weights = df_all["n_points"].values if args.weighted else None

    # Threshold optimization (global)
    pred_th = args.pred_th
    if args.optimize != "none":
        pred_th, best_val = find_best_threshold(y_true, y_score, sample_weight=weights, mode=args.optimize)
        print(f"[INFO] Optimized global threshold ({args.optimize}): th = {pred_th:.6f}, best = {best_val:.6f}")

    # Global evaluation
    res = evaluate(y_true, y_score, sample_weight=weights, pred_th=pred_th)
    # Save ROC
    roc_path = out_dir / "roc_curve.png"
    if res["fpr"] is not None and len(res["fpr"]) > 0 and not np.isnan(res["AUROC"]):
        plot_roc(res["fpr"], res["tpr"], res["AUROC"], roc_path, title="ROC Curve (All Residues)")
        print(f"[INFO] ROC saved to: {roc_path}")
    else:
        print("[WARN] ROC could not be plotted (possibly single-class data)")

    # Print and save summary
    summary = {
        "ACC": res["ACC"], "F1": res["F1"], "MCC": res["MCC"],
        "AUROC": res["AUROC"], "AUPRC": res["AUPRC"],
        "pred_threshold": res["pred_th"], "label_threshold": args.label_th,
        "weighted": args.weighted
    }
    summary_df = pd.DataFrame([summary])
    summary_csv = out_dir / "metrics_summary_residue.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[INFO] Summary metrics saved to: {summary_csv}")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Optional: per-file metrics
    if args.per_file:
        rows = []
        for file_id, g in df_all.groupby("file_id"):
            yi = binarize_labels(g["mean_label"].values, args.label_th)
            si = g["mean_predict"].values
            wi = g["n_points"].values if args.weighted else None
            # Use same global threshold for comparability
            ri = evaluate(yi, si, sample_weight=wi, pred_th=pred_th)
            rows.append({
                "file_id": file_id,
                "ACC": ri["ACC"], "F1": ri["F1"], "MCC": ri["MCC"],
                "AUROC": ri["AUROC"], "AUPRC": ri["AUPRC"],
                "TP": ri["tp"], "TN": ri["tn"], "FP": ri["fp"], "FN": ri["fn"]
            })
        per_file_df = pd.DataFrame(rows).sort_values("file_id")
        per_file_csv = out_dir / "metrics_per_file.csv"
        per_file_df.to_csv(per_file_csv, index=False)
        print(f"[INFO] Per-file metrics saved to: {per_file_csv}")


if __name__ == "__main__":
    main()
