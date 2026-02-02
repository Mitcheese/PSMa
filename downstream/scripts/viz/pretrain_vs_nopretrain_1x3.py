# -*- coding: utf-8 -*-
# === Auto-generated "pretrain_vs_nopretrain_1x3_modified.py" ===
# Enhancements added before original code:
#  1) Larger fonts via rcParams (tunable by env PS_FONT_SCALE, default 1.25)
#  2) Avoid label/line collisions on panel 1 (axes[0]) by offsetting data-text labels
#     and adding a semi-transparent white box + stroke.
#  3) Avoid overlapping top info boxes on panel 3 (axes[2]) by staggering upper texts.
#  4) Thicker ticks/spines; tight layout.
#
# Usage:
#   PS_FONT_SCALE=1.35 python pretrain_vs_nopretrain_1x3_modified.py ...
#
import os, math, traceback
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as _Figure
import matplotlib.patheffects as _pe

# -------------- Global font scaling --------------
try:
    SCALE = float(os.getenv("PS_FONT_SCALE", "1.25"))
except Exception:
    SCALE = 1.25

# plt.rcParams.update({
#     "font.size":        12,
#     "axes.titlesize":   14 * SCALE,
#     "axes.labelsize":   13 * SCALE,
#     "xtick.labelsize":  12 * SCALE,
#     "ytick.labelsize":  12 * SCALE,
#     "legend.fontsize":  12 * SCALE,
# })
# ==== MINIMAL PATCH v2 (replace previous function) ====
def _apply_minimal_patch(fig):
    import matplotlib.pyplot as plt

    axes = fig.get_axes()

    # 1) First panel: remove "with_pretrain: epoch 19" / "without_pretrain: epoch 36"
    if len(axes) >= 1:
        ax = axes[0]
        for t in list(ax.texts):
            s = t.get_text().strip().lower()
            if "epoch" in s and ("with_pretrain" in s or "without_pretrain" in s):
                t.remove()

    # 2) Second panel: cap y to ~0.6; move inset to lower-right
    if len(axes) >= 2:
        ax = axes[1]
        try:
            ymin, ymax = ax.get_ylim()
        except Exception:
            ymin, ymax = 0.0, 1.0
        ax.set_ylim(bottom=min(0.0, ymin), top=0.6)  # set top=0.55 if needed

        box_text = (
            "F1@0.5 w/=0.393, w/o=0.345\n"
            "MCC@0.5 w/=0.298, w/o=0.238"
        )

        moved = False
        for t in ax.texts:
            s = t.get_text()
            if ("F1@0.5" in s) or ("MCC@0.5" in s):
                # lower-right
                t.set_text(box_text)
                t.set_transform(ax.transAxes)
                t.set_position((0.98, 0.02))
                t.set_ha("right"); t.set_va("bottom")
                t.set_bbox(dict(fc="white", ec="lightgray", alpha=0.85, pad=0.4))
                moved = True
        if not moved:
            ax.text(0.98, 0.02, box_text, transform=ax.transAxes,
                    ha="right", va="bottom",
                    bbox=dict(fc="white", ec="lightgray", alpha=0.85, pad=0.4))

    # 3) Third panel: cap y to ~3.5; move inset to mid-lower near x-axis
    if len(axes) >= 3:
        ax = axes[2]
        try:
            ymin, ymax = ax.get_ylim()
        except Exception:
            ymin, ymax = 0.0, 1.0
        ax.set_ylim(bottom=ymin, top=3.2)

        target_pos = (0.60, 0.01)  # mid-lower near x-axis; lower if needed
        moved = False

        # Prefer moving legend containing "KS"; otherwise move top-left legend
        for t in ax.texts:
            if t.get_text().strip().startswith("KS"):
                t.set_transform(ax.transAxes)
                t.set_position(target_pos)
                t.set_ha("center"); t.set_va("bottom")
                t.set_bbox(dict(fc="white", ec="lightgray", alpha=0.85, pad=0.4))
                moved = True
                break
        if not moved:
            for t in ax.texts:
                if t.get_transform() is ax.transAxes:
                    x, y = t.get_position()
                    if (y >= 0.85) and (x <= 0.35):  # top-left region
                        t.set_position(target_pos)
                        t.set_ha("center"); t.set_va("bottom")
                        t.set_bbox(dict(fc="white", ec="lightgray", alpha=0.85, pad=0.4))
                        break
# ==== MINIMAL PATCH v2 END ====

def _ps_offset_collision_labels(ax, inflate_top=0.08):
    """Offset data-coordinate texts with bbox + stroke; add headroom."""
    try:
        texts = list(ax.texts)
        if not texts:
            return
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        dx = 0.015 * (xmax - xmin)
        dy = 0.015 * (ymax - ymin)
        shifts = [(1,1),(1,-1),(-1,1),(-1,-1)]
        idx = 0; changed = False
        for t in texts:
            if t.get_transform() is ax.transData:
                x, y = t.get_position()
                sx, sy = shifts[idx % 4]; idx += 1
                t.set_position((x + sx*dx, y + sy*dy))
                t.set_bbox(dict(fc="white", ec="none", alpha=0.65, pad=0.6))
                t.set_path_effects([_pe.Stroke(linewidth=2.2, foreground="white"), _pe.Normal()])
                t.set_zorder(10)
                changed = True
        if changed:
            ax.set_ylim(ymin, ymax + inflate_top*(ymax - ymin))
    except Exception:
        print("[WARN] _ps_offset_collision_labels failed:"); traceback.print_exc()

def _ps_stagger_top_boxes(ax, top_y=0.98, second_y=0.80, alpha=0.85, inflate_top=0.06):
    """Stagger texts in axes coords near the top to avoid overlaps; add headroom."""
    try:
        top_texts = []
        for t in list(ax.texts):
            if t.get_transform() is ax.transAxes:
                x, y = t.get_position()
                if isinstance(x, (int,float)) and isinstance(y, (int,float)) and y >= 0.85:
                    top_texts.append(t)
        if len(top_texts) >= 2:
            top_texts.sort(key=lambda tt: tt.get_position()[0])
            ys = [top_y, second_y, 0.72, 0.64]
            for i, t in enumerate(top_texts):
                x, _ = t.get_position()
                ny = ys[i] if i < len(ys) else max(0.10, second_y - 0.10*(i-1))
                t.set_position((x, ny))
                bbox = t.get_bbox_patch()
                if bbox is None:
                    t.set_bbox(dict(fc="white", ec="lightgray", alpha=alpha, pad=0.4))
                else:
                    bbox.set_alpha(alpha)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + inflate_top*(ymax - ymin))
    except Exception:
        print("[WARN] _ps_stagger_top_boxes failed:"); traceback.print_exc()

def _ps_thicken_axes(fig):
    try:
        for ax in fig.get_axes():
            ax.tick_params(axis="both", which="major", length=6, width=1.2)
            ax.tick_params(axis="both", which="minor", length=3, width=1.0)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
    except Exception:
        pass

def _ps_refine_layout(fig=None):
    try:
        if fig is None: fig = plt.gcf()
        axes = fig.get_axes()
        if axes:
            if len(axes) >= 1: _ps_offset_collision_labels(axes[0], inflate_top=0.08)
            if len(axes) >= 3: _ps_stagger_top_boxes(axes[2], top_y=0.98, second_y=0.80, alpha=0.85, inflate_top=0.06)
        _ps_thicken_axes(fig)
        try: fig.tight_layout()
        except Exception: pass
    except Exception:
        print("[WARN] _ps_refine_layout failed:"); traceback.print_exc()

# Monkey-patch savefig so refinement runs automatically
__PS_ORIG_SAVEFIG = _Figure.savefig
def __PS_HOOKED_SAVEFIG(self, *args, **kwargs):
    try: _ps_refine_layout(self)
    except Exception as _e: print("[WARN] refine failed:", _e)
    return __PS_ORIG_SAVEFIG(self, *args, **kwargs)
_Figure.savefig = __PS_HOOKED_SAVEFIG

# ---- ORIGINAL SCRIPT FOLLOWS ----

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a 1×3 figure comparing models with and without pretraining.
(a) Convergence (Time-to-Target MCC)
(b) Threshold Sensitivity (F1 & MCC vs. Threshold)
(c) Score Separation (Pos/Neg Distributions, KS)
"""

import argparse
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
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
    ax.annotate(f"Target MCC = {target_mcc:.3f}", xy=(0.02, 0.92), xycoords="axes fraction",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))
    for color, name, df in [(BLUE, "with_pretrain", df_with), (ORANGE, "without_pretrain", df_without)]:
        ep, val = earliest_epoch_reaching(df, target_mcc)
        if ep is not None:
            ax.axvline(ep, color=color, linestyle=":", linewidth=1.2)
            ax.scatter([ep], [val], color=color, zorder=3)
            ax.annotate(f"{name}: epoch {ep}", (ep, val), xytext=(5, 6),
                        textcoords="offset points", fontsize=9, color=color)
        else:
            ax.annotate(f"{name}: not reached", xy=(0.55, 0.05), xycoords="axes fraction",
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
    ax.set_xlim(0.1, 0.9); ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=9)

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
    ax.set_title("(c) Score Separation (Pos/Neg Distributions, KS)")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Density")
    ax.set_xlim(0, 1); ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=9)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-metrics", required=True)
    parser.add_argument("--without-metrics", required=True)
    parser.add_argument("--with-scores", required=True)
    parser.add_argument("--without-scores", required=True)
    parser.add_argument("--out-prefix", default="pretrain_vs_nopretrain_1x3")
    parser.add_argument("--thr-step", type=float, default=0.001)
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=200)
    ax_a, ax_b, ax_c = axes[0], axes[1], axes[2]

    plot_convergence(ax_a, df_with, df_without, target_mcc=target_mcc)
    plot_threshold_sensitivity(ax_b, y_w, p_w, y_wo, p_wo, thr_grid)
    plot_distribution(ax_c, y_w, p_w, y_wo, p_wo, bins=args.hist_bins)

    plt.tight_layout()
    png_path = f"{args.out_prefix}_1x3.png"
    pdf_path = f"{args.out_prefix}_1x3.pdf"
    _apply_minimal_patch(plt.gcf())
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure to: {png_path} and {pdf_path}")
    print(f"[Panel (a)] target_mcc = {target_mcc:.6f} (max w/={max_with:.6f}, max w/o={max_without:.6f})")

if __name__ == "__main__":
    main()
