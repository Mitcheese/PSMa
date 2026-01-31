#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Traverse .ply files in a directory (optionally recursive), aggregate label/predict,
compute ACC / F1 / MCC / AUROC / AUPRC, and optionally plot ROC/PR curves.
Per-file metrics can also be output (using the global threshold).

Dependencies:
    pip install plyfile pandas numpy scikit-learn matplotlib tqdm

python scripts/eval/eval_ply.py /path/to/in_dir /path/to/out_dir --optimize mcc --plot
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from plyfile import PlyData

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt


REQ_FIELDS = ["label", "predict"]


def load_points_from_ply(ply_path: Path):
    """Read a PLY file and return (labels, scores) 1D arrays."""
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise ValueError("Missing 'vertex' element")
    v = ply["vertex"].data  # numpy structured array

    # Field checks
    missing = [c for c in REQ_FIELDS if c not in v.dtype.names]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    # Extract and cast
    labels = np.asarray(v["label"]).astype(float)  # unify float/int before binarization
    scores = np.asarray(v["predict"]).astype(float)

    # Drop NaN
    ok = np.isfinite(labels) & np.isfinite(scores)
    labels, scores = labels[ok], scores[ok]

    # Binarize labels (defensive: >0.5 => 1)
    labels = (labels >= 0.5).astype(int)

    # Clip predictions to [0, 1] for robustness
    scores = np.clip(scores, 0.0, 1.0)

    return labels, scores


def collect_all_points(in_dir: Path, recursive: bool):
    pattern = "**/*.ply" if recursive else "*.ply"
    files = sorted(in_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .ply found in {in_dir} (recursive={recursive})")

    all_labels = []
    all_scores = []
    per_file_ranges = []  # track ranges for per-file evaluation
    start = 0

    for f in tqdm(files, desc="Loading PLYs", unit="file"):
        try:
            y, s = load_points_from_ply(f)
            all_labels.append(y)
            all_scores.append(s)
            end = start + len(y)
            per_file_ranges.append((f.stem, start, end))
            start = end
        except Exception as e:
            print(f"[SKIP] {f}: {e}", file=sys.stderr)

    if not all_labels:
        raise RuntimeError("No valid PLY (all failed or missing fields)")

    Y = np.concatenate(all_labels, axis=0)
    S = np.concatenate(all_scores, axis=0)
    return Y, S, per_file_ranges


def decisions_from_scores(scores: np.ndarray, th: float) -> np.ndarray:
    return (scores >= th).astype(int)


def weighted_confusion(y_true, y_pred):
    # Point-level evaluation uses equal weights
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def mcc_from_confusion(tp, tn, fp, fn):
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return np.nan
    return (tp * tn - fp * fn) / denom


def safe_auc(y_true, y_score):
    # AUROC/AUPRC undefined if only one class exists
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, (np.array([0, 1]), np.array([0, 1])), (np.array([0, 1]), np.array([1, 0]))
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return auroc, auprc, (fpr, tpr), (prec, rec)


def find_best_threshold(y_true, y_score, mode="f1"):
    """Search the best global threshold: 'f1' / 'mcc' / 'youden' (TPR-FPR)."""
    uniq = np.unique(y_true)
    if len(uniq) < 2 or np.allclose(np.min(y_score), np.max(y_score)):
        return 0.5, np.nan

    if mode == "youden":
        auroc, _, (fpr, tpr), _ = safe_auc(y_true, y_score)
        if np.isnan(auroc):
            return 0.5, np.nan
        youden = tpr - fpr
        idx = int(np.nanargmax(youden))
        # Approximate with 0.5; for precise threshold use roc_curve thresholds
        return 0.5, float(youden[idx])

    cand = np.unique(y_score)
    best_th, best_val = 0.5, -np.inf
    for th in cand:
        y_pred = decisions_from_scores(y_score, th)
        if mode == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
        elif mode == "mcc":
            tp, tn, fp, fn = weighted_confusion(y_true, y_pred)
            val = mcc_from_confusion(tp, tn, fp, fn)
            if np.isnan(val):
                val = -np.inf
        else:
            continue
        if val > best_val:
            best_th, best_val = float(th), float(val)
    return best_th, best_val


def evaluate(y_true, y_score, pred_th=0.5):
    y_pred = decisions_from_scores(y_score, pred_th)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    tp, tn, fp, fn = weighted_confusion(y_true, y_pred)
    mcc = mcc_from_confusion(tp, tn, fp, fn)
    auroc, auprc, (fpr, tpr), (prec, rec) = safe_auc(y_true, y_score)
    return {
        "ACC": acc, "F1": f1, "MCC": mcc,
        "AUROC": auroc, "AUPRC": auprc,
        "fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "pred_th": pred_th
    }


def plot_curves(fpr, tpr, auroc, prec, rec, auprc, out_dir: Path, prefix="all"):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROC
    if not np.isnan(auroc):
        plt.figure(figsize=(6, 5), dpi=140)
        plt.plot(fpr, tpr, label=f"AUC = {auroc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_roc.png")
        plt.close()

    # PR
    if not np.isnan(auprc):
        plt.figure(figsize=(6, 5), dpi=140)
        plt.plot(rec, prec, label=f"AP = {auprc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_pr.png")
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Aggregate PLY points and evaluate classification metrics.")
    ap.add_argument("in_dir", type=str, help="Input directory (contains .ply)")
    ap.add_argument("out_dir", type=str, help="Output directory (plots and CSV)")
    ap.add_argument("--recursive", action="store_true", help="Read subdirectories recursively")
    ap.add_argument("--pred-th", type=float, default=0.5, help="Prediction threshold (default 0.5)")
    ap.add_argument("--optimize", choices=["none", "f1", "mcc", "youden"], default="none",
                    help="Criterion for global threshold search")
    ap.add_argument("--plot", action="store_true", help="Save ROC/PR curves")
    ap.add_argument("--per-file", action="store_true", help="Output per-file metrics (global threshold)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    y_all, s_all, ranges = collect_all_points(in_dir, args.recursive)

    # Global threshold
    pred_th = args.pred_th
    if args.optimize != "none":
        pred_th, best_val = find_best_threshold(y_all, s_all, mode=args.optimize)
        print(f"[INFO] Optimized global threshold ({args.optimize}): th = {pred_th:.6f}, best = {best_val if best_val==best_val else float('nan'):.6f}")

    # Global evaluation
    res = evaluate(y_all, s_all, pred_th=pred_th)

    # Save curves
    if args.plot:
        plot_curves(res["fpr"], res["tpr"], res["AUROC"], res["prec"], res["rec"], res["AUPRC"], out_dir, prefix="all")

    # Summary CSV
    summary = pd.DataFrame([{
        "ACC": res["ACC"], "F1": res["F1"], "MCC": res["MCC"],
        "AUROC": res["AUROC"], "AUPRC": res["AUPRC"],
        "TP": res["TP"], "TN": res["TN"], "FP": res["FP"], "FN": res["FN"],
        "pred_threshold": res["pred_th"],
        "num_points": len(y_all)
    }])
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Per-file metrics (global threshold for comparability)
    if args.per_file:
        rows = []
        for file_id, (name, st, ed) in enumerate(ranges):
            yi = y_all[st:ed]
            si = s_all[st:ed]
            ri = evaluate(yi, si, pred_th=pred_th)
            rows.append({
                "file_id": name,
                "N": len(yi),
                "ACC": ri["ACC"], "F1": ri["F1"], "MCC": ri["MCC"],
                "AUROC": ri["AUROC"], "AUPRC": ri["AUPRC"],
                "TP": ri["TP"], "TN": ri["TN"], "FP": ri["FP"], "FN": ri["FN"]
            })
        pd.DataFrame(rows).sort_values("file_id").to_csv(out_dir / "metrics_per_file.csv", index=False)
        print(f"[INFO] Per-file metrics saved to: {out_dir / 'metrics_per_file.csv'}")


if __name__ == "__main__":
    main()
