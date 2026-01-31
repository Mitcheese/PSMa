#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a 2×2 figure comparing models with and without pretraining.
(Updated: fixed-target MCC in panel (a) + unified colors/styles.)

See docstring inside for details.
"""

import argparse
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#1f77b4"    # with_pretrain
ORANGE = "#ff7f0e"  # without_pretrain
GREY = "#6e6e6e"
EPS = 1e-12

def read_metrics_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["epoch", "ACC", "F1", "AUROC", "AUPRC", "MCC"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Metrics CSV {path} is missing columns: {missing}")
    df = df.copy()
    df["epoch"] = df["epoch"].astype(int)
    for c in ["ACC", "F1", "AUROC", "AUPRC", "MCC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("epoch").reset_index(drop=True)
    return df

def read_scores_csv(path: str):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "y_true" in cols and "y_prob" in cols:
        y = df[cols["y_true"]].values
        p = df[cols["y_prob"]].values
    elif "mean_label" in cols and "mean_predict" in cols:
        y = df[cols["mean_label"]].values
        p = df[cols["mean_predict"]].values
    elif "label" in cols and "predict" in cols:
        y = df[cols["label"]].values
        p = df[cols["predict"]].values
    else:
        raise ValueError("Scores CSV must have (y_true,y_prob) or (mean_label,mean_predict) or (label,predict)")
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.clip(p, EPS, 1.0 - EPS)
    y = (y >= 0.5).astype(int)
    return y, p

def confusion_at_threshold(y_true, y_prob, thr):
    pred = (y_prob >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    return tp, tn, fp, fn

def safe_div(a, b): return a / b if b != 0 else 0.0

def f1_from_conf(tp, tn, fp, fn):
    precision = safe_div(tp, (tp + fp))
    recall = safe_div(tp, (tp + fn))
    if precision + recall == 0: return 0.0
    return 2 * precision * recall / (precision + recall + EPS)

def mcc_from_conf(tp, tn, fp, fn):
    num = (tp * tn) - (fp * fn)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return safe_div(num, den + EPS)

def compute_curves(y_true, y_prob, thr_grid):
    f1s, mccs = [], []
    for t in thr_grid:
        tp, tn, fp, fn = confusion_at_threshold(y_true, y_prob, t)
        f1s.append(f1_from_conf(tp, tn, fp, fn))
        mccs.append(mcc_from_conf(tp, tn, fp, fn))
    return np.array(f1s), np.array(mccs)

def brier_score(y_true, y_prob): return float(np.mean((y_prob - y_true) ** 2))
def nll_logloss(y_true, y_prob):
    return float(-np.mean(y_true*np.log(y_prob+EPS) + (1-y_true)*np.log(1-y_prob+EPS)))

def reliability_bins(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    confs, accs, counts = [], [], []
    for b in range(n_bins):
        mask = (bin_ids == b)
        if np.any(mask):
            p_bin = y_prob[mask]; y_bin = y_true[mask]
            confs.append(float(np.mean(p_bin)))
            accs.append(float(np.mean(y_bin)))
            counts.append(int(np.sum(mask)))
        else:
            confs.append(np.nan); accs.append(np.nan); counts.append(0)
    return np.array(confs), np.array(accs), np.array(counts), bins

def expected_calibration_error(y_true, y_prob, n_bins=10):
    confs, accs, counts, _ = reliability_bins(y_true, y_prob, n_bins=n_bins)
    total = np.sum(counts); ece = 0.0
    if total == 0: return 0.0
    for c, a, n in zip(confs, accs, counts):
        if n == 0 or np.isnan(c) or np.isnan(a): continue
        ece += (n/total)*abs(a-c)
    return float(ece)

def ks_statistic(y_true, y_prob):
    pos = np.sort(y_prob[y_true == 1]); neg = np.sort(y_prob[y_true == 0])
    if len(pos) == 0 or len(neg) == 0: return 0.0
    all_vals = np.sort(np.unique(np.concatenate([pos, neg])))
    pos_cdf = np.searchsorted(pos, all_vals, side='right') / len(pos)
    neg_cdf = np.searchsorted(neg, all_vals, side='right') / len(neg)
    return float(np.max(np.abs(pos_cdf - neg_cdf)))

def earliest_epoch_reaching(df, target_mcc):
    mask = df["MCC"].values >= target_mcc - EPS
    idx = np.where(mask)[0]
    if idx.size == 0: return None, None
    i = int(idx[0]); return int(df.loc[i, "epoch"]), float(df.loc[i, "MCC"])

def plot_convergence(ax, df_with, df_without, target_mcc):
    ax.plot(df_with["epoch"], df_with["MCC"], label="with_pretrain (MCC)",
            color=BLUE, linestyle="-", linewidth=1.6)
    ax.plot(df_without["epoch"], df_without["MCC"], label="without_pretrain (MCC)",
            color=ORANGE, linestyle="-", linewidth=1.6)
    ax.axhline(target_mcc, color=GREY, linestyle="--", linewidth=1.0)
    ax.annotate(f"Target MCC = {target_mcc:.3f}", xy=(0.01, 0.95), xycoords="axes fraction",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))
    for color, name, df in [(BLUE, "with_pretrain", df_with), (ORANGE, "without_pretrain", df_without)]:
        ep, val = earliest_epoch_reaching(df, target_mcc)
        if ep is not None:
            ax.axvline(ep, color=color, linestyle=":", linewidth=1.2)
            ax.scatter([ep], [val], color=color, zorder=3)
            ax.annotate(f"{name}: epoch {ep}", (ep, val), xytext=(5, 6),
                        textcoords="offset points", fontsize=9, color=color)
        else:
            ax.annotate(f"{name}: not reached", xy=(0.50, 0.05), xycoords="axes fraction",
                        fontsize=9, color=color)
    ax.set_title("(a) Convergence (Time-to-Target MCC)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MCC")
    ax.grid(True, alpha=0.3); ax.legend()

def plot_threshold_sensitivity(ax, y_with, p_with, y_without, p_without, thr_grid):
    f1_w, mcc_w = compute_curves(y_with, p_with, thr_grid)
    f1_wo, mcc_wo = compute_curves(y_without, p_without, thr_grid)
    ax.plot(thr_grid, f1_w, label="with_pretrain F1", color=BLUE, linestyle="-", linewidth=1.6)
    ax.plot(thr_grid, mcc_w, label="with_pretrain MCC", color=BLUE, linestyle="--", linewidth=1.6)
    ax.plot(thr_grid, f1_wo, label="without_pretrain F1", color=ORANGE, linestyle="-", linewidth=1.6)
    ax.plot(thr_grid, mcc_wo, label="without_pretrain MCC", color=ORANGE, linestyle="--", linewidth=1.6)
    ax.axvline(0.5, color=GREY, linestyle=":", linewidth=1.0)
    def value_at(arr, t=0.5):
        idx = (np.abs(thr_grid - t)).argmin(); return float(arr[idx])
    ax.annotate(f"F1@0.5 w/={value_at(f1_w):.3f}, w/o={value_at(f1_wo):.3f}\n"
                f"MCC@0.5 w/={value_at(mcc_w):.3f}, w/o={value_at(mcc_wo):.3f}",
                (0.02, 0.02), xycoords="axes fraction", fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
    ax.set_title("(b) Threshold Sensitivity (F1 & MCC vs. Threshold)")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_xlim(0, 1); ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=9)

def plot_reliability_two(ax, y_with, p_with, y_without, p_without, n_bins=10):
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1)
    confs_w, accs_w, _, _ = reliability_bins(y_with, p_with, n_bins=n_bins)
    mask_w = ~np.isnan(confs_w) & ~np.isnan(accs_w)
    ax.plot(confs_w[mask_w], accs_w[mask_w], color=BLUE, linestyle="-", marker="o", label="with_pretrain")
    confs_wo, accs_wo, _, _ = reliability_bins(y_without, p_without, n_bins=n_bins)
    mask_wo = ~np.isnan(confs_wo) & ~np.isnan(accs_wo)
    ax.plot(confs_wo[mask_wo], accs_wo[mask_wo], color=ORANGE, linestyle="--", marker="s", label="without_pretrain")
    ece_w = expected_calibration_error(y_with, p_with, n_bins=n_bins)
    ece_wo = expected_calibration_error(y_without, p_without, n_bins=n_bins)
    brier_w = brier_score(y_with, p_with); brier_wo = brier_score(y_without, p_without)
    nll_w = nll_logloss(y_with, p_with); nll_wo = nll_logloss(y_without, p_without)
    text = (f"ECE w/={ece_w:.3f}, w/o={ece_wo:.3f}\n"
            f"Brier w/={brier_w:.3f}, w/o={brier_wo:.3f}\n"
            f"NLL w/={nll_w:.3f}, w/o={nll_wo:.3f}")
    ax.annotate(text, (0.55, 0.05), xycoords="axes fraction", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(c) Calibration (Reliability Diagram, ECE/Brier/NLL)")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Empirical positive rate")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

def plot_distribution(ax, y_with, p_with, y_without, p_without, bins=40):
    pos_w = p_with[y_with == 1]; neg_w = p_with[y_with == 0]
    ax.hist(pos_w, bins=bins, range=(0,1), density=True, histtype="step",
            label="with_pretrain pos", color=BLUE, linestyle="-", linewidth=1.4)
    ax.hist(neg_w, bins=bins, range=(0,1), density=True, histtype="step",
            label="with_pretrain neg", color=BLUE, linestyle="--", linewidth=1.4)
    pos_wo = p_without[y_without == 1]; neg_wo = p_without[y_without == 0]
    ax.hist(pos_wo, bins=bins, range=(0,1), density=True, histtype="step",
            label="without_pretrain pos", color=ORANGE, linestyle="-", linewidth=1.4)
    ax.hist(neg_wo, bins=bins, range=(0,1), density=True, histtype="step",
            label="without_pretrain neg", color=ORANGE, linestyle="--", linewidth=1.4)
    ks_w = ks_statistic(y_with, p_with); ks_wo = ks_statistic(y_without, p_without)
    ax.annotate(f"KS w/={ks_w:.3f}, w/o={ks_wo:.3f}", (0.02, 0.95), xycoords="axes fraction",
                va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))
    ax.set_title("(d) Score Separation (Pos/Neg Distributions, KS)")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Density")
    ax.set_xlim(0, 1); ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=9)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-metrics", required=True)
    parser.add_argument("--without-metrics", required=True)
    parser.add_argument("--with-scores", required=True)
    parser.add_argument("--without-scores", required=True)
    parser.add_argument("--out-prefix", default="pretrain_vs_nopretrain")
    parser.add_argument("--thr-step", type=float, default=0.001)
    parser.add_argument("--calib-bins", type=int, default=10)
    parser.add_argument("--hist-bins", type=int, default=40)
    parser.add_argument("--target-mcc", type=float, default=None)
    parser.add_argument("--target-ratio", type=float, default=0.90)
    args = parser.parse_args()

    df_with = read_metrics_csv(args.with_metrics)
    df_without = read_metrics_csv(args.without_metrics)
    y_w, p_w = read_scores_csv(args.with_scores)
    y_wo, p_wo = read_scores_csv(args.without_scores)

    max_with = float(df_with["MCC"].max())
    max_without = float(df_without["MCC"].max())
    if args.target_mcc is not None:
        target_mcc = float(args.target_mcc)
    else:
        target_mcc = float(args.target_ratio) * min(max_with, max_without)

    thr_grid = np.arange(0.0, 1.0 + args.thr_step, args.thr_step)
    thr_grid = np.clip(thr_grid, 0.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=200)
    ax_a, ax_b = axes[0,0], axes[0,1]
    ax_c, ax_d = axes[1,0], axes[1,1]

    plot_convergence(ax_a, df_with, df_without, target_mcc=target_mcc)
    plot_threshold_sensitivity(ax_b, y_w, p_w, y_wo, p_wo, thr_grid)
    plot_reliability_two(ax_c, y_w, p_w, y_wo, p_wo, n_bins=args.calib_bins)
    plot_distribution(ax_d, y_w, p_w, y_wo, p_wo, bins=args.hist_bins)

    plt.tight_layout()
    png_path = f"{args.out_prefix}.png"
    pdf_path = f"{args.out_prefix}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure to: {png_path} and {pdf_path}")
    print(f"[Panel (a)] target_mcc = {target_mcc:.6f} (max w/={max_with:.6f}, max w/o={max_without:.6f})")

if __name__ == "__main__":
    main()
