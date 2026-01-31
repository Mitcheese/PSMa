#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_2x2.py

Create a 2×2 composite figure comparing two models across per-sequence metrics
(AUROC and AUPRC), following these rules:
- Treat each CSV row as one sequence point.
- Drop rows where `file == "__ALL__"` (aggregate).
- Drop rows where AUROC == 0 or MCC <= 0.
- Pair the two CSVs by the 'file' column (inner join).
- Panel A: AUROC scatter (x: Model A, y: Model B) + y=x line.
- Panel B: AUPRC scatter (x: Model A, y: Model B) + y=x line.
- Panel C: Δ distributions (hist of (B-A) for AUROC & AUPRC), with 0 line,
           and win/tie/loss rates annotated.
- Panel D: Threshold-win curves: for a threshold grid t, plot the proportion of
           sequences with metric > t for both models. Includes AUROC curves and
           AUPRC curves (dashed).

Outputs:
- figure PNG/PDF (by --out_png / --out_pdf)
- summary.csv (win rates, mean/median deltas, optional p-values if SciPy available)

Usage example:
python compare_2x2.py \
  --csv_a metrics_ensem.csv --name_a EnsemPPIS \
  --csv_b metric_pretrain.csv --name_b Pretrain \
  --out_png figure_compare.png
"""

import argparse
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
# Increase global font sizes a bit
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
})


# Optional: Wilcoxon signed-rank test
try:
    from scipy.stats import wilcoxon
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_and_filter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "file" in df.columns:
        df = df[df["file"] != "__ALL__"].copy()
    # Coerce to numeric
    for col in ["auroc", "auprc", "mcc"]:
        if col in df.columns:
            df[col] = _numeric(df[col])
    # Apply user rule: drop AUROC==0 or MCC<=0; also drop NaNs on auroc/auprc/mcc
    need = [c for c in ["auroc", "auprc", "mcc"] if c in df.columns]
    if not set(["auroc","auprc","mcc"]).issubset(df.columns):
        raise SystemExit(f"{path} must contain columns: auroc, auprc, mcc")
    df = df[(df["auroc"] > 0.0) & (df["mcc"] > 0.0)].dropna(subset=["auroc","auprc","mcc"]).copy()
    # Keep helpful optional fields if present
    keep_cols = ["file", "auroc", "auprc", "mcc"]
    for c in ["pos_rate", "n_rows"]:
        if c in df.columns:
            keep_cols.append(c)
    return df[keep_cols].reset_index(drop=True)


def align_on_file(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "file" not in a.columns or "file" not in b.columns:
        raise SystemExit("Both CSVs must have a 'file' column to pair samples.")
    common = sorted(set(a["file"]) & set(b["file"]))
    if len(common) == 0:
        raise SystemExit("No overlapping 'file' entries between the two CSVs after filtering.")
    a2 = a[a["file"].isin(common)].set_index("file").sort_index()
    b2 = b[b["file"].isin(common)].set_index("file").sort_index()
    return a2, b2


def win_tie_loss(deltas: np.ndarray, tie_tol: float = 1e-9) -> Dict[str, float]:
    wins = np.sum(deltas > tie_tol)
    ties = np.sum(np.abs(deltas) <= tie_tol)
    losses = np.sum(deltas < -tie_tol)
    n = deltas.size
    return {
        "wins": int(wins), "ties": int(ties), "losses": int(losses),
        "win_rate": wins / n if n else np.nan,
        "tie_rate": ties / n if n else np.nan,
        "loss_rate": losses / n if n else np.nan
    }


def threshold_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # proportion of values strictly greater than threshold t
    vals = values.astype(float)
    return np.array([(vals > t).mean() for t in grid], dtype=float)


def summarize(d: np.ndarray) -> Dict[str, float]:
    try:
        from scipy.stats import wilcoxon as _wilcoxon
        pval = float(_wilcoxon(d, alternative="two-sided").pvalue) if d.size > 0 and np.any(d != 0) else np.nan
    except Exception:
        pval = np.nan
    return {
        "n": int(d.size),
        "mean": float(np.mean(d)) if d.size else np.nan,
        "median": float(np.median(d)) if d.size else np.nan,
        "std": float(np.std(d, ddof=1)) if d.size >= 2 else np.nan,
        "p_wilcoxon_two_sided": pval,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_a", required=True, help="CSV for model A")
    ap.add_argument("--csv_b", required=True, help="CSV for model B")
    ap.add_argument("--name_a", default="Model A")
    ap.add_argument("--name_b", default="Model B")
    ap.add_argument("--out_png", default="figure_compare.png")
    ap.add_argument("--out_pdf", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--auroc_grid_start", type=float, default=0.5)
    ap.add_argument("--auroc_grid_stop", type=float, default=0.95)
    ap.add_argument("--auroc_grid_step", type=float, default=0.01)
    ap.add_argument("--auprc_grid_start", type=float, default=0.1)
    ap.add_argument("--auprc_grid_stop", type=float, default=0.9)
    ap.add_argument("--auprc_grid_step", type=float, default=0.02)
    ap.add_argument("--tie_tol", type=float, default=1e-9, help="Delta absolute value <= tol is tie")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    a = load_and_filter(args.csv_a)
    b = load_and_filter(args.csv_b)
    a2, b2 = align_on_file(a, b)

    # Arrays
    auroc_a = a2["auroc"].to_numpy()
    auroc_b = b2["auroc"].to_numpy()
    auprc_a = a2["auprc"].to_numpy()
    auprc_b = b2["auprc"].to_numpy()

    d_auc = auroc_b - auroc_a
    d_auprc = auprc_b - auprc_a

    # Summaries
    wtl_auc = win_tie_loss(d_auc, tie_tol=args.tie_tol)
    wtl_auprc = win_tie_loss(d_auprc, tie_tol=args.tie_tol)

    sm_auc = summarize(d_auc)
    sm_auprc = summarize(d_auprc)

    # Grids
    grid_auc = np.arange(args.auroc_grid_start, args.auroc_grid_stop + 1e-12, args.auroc_grid_step)
    grid_auprc = np.arange(args.auprc_grid_start, args.auprc_grid_stop + 1e-12, args.auprc_grid_step)

    curve_auc_a = threshold_curve(auroc_a, grid_auc)
    curve_auc_b = threshold_curve(auroc_b, grid_auc)
    curve_auprc_a = threshold_curve(auprc_a, grid_auprc)
    curve_auprc_b = threshold_curve(auprc_b, grid_auprc)


    # Determine colors for models: Pretrain -> blue; Ensem -> orange-yellow
    def _role_from_name(name: str) -> str:
        n = (name or "").lower()
        # Treat PSMa as the "pretrain" group
        if any(k in n for k in ["psma", "pretrain", "with_pretrain", "pre-train", "w/ pretrain"]):
            return "psma"
        if "ensem" in n:
            return "ensem"
        return ""

    def _pretty_model_name(raw: str) -> str:
        r = _role_from_name(raw)
        if r == "psma":
            return "PSMa"
        if r == "ensem":
            return "Ensem"
        return raw or ""

    COLOR_PSMa  = "#1f77b4"  # blue
    COLOR_ENSEM = "#FFA500"  # orange

    role_a = _role_from_name(args.name_a)
    role_b = _role_from_name(args.name_b)
    col_a  = COLOR_PSMa  if role_a == "psma"  else (COLOR_ENSEM if role_a == "ensem" else None)
    col_b  = COLOR_PSMa  if role_b == "psma"  else (COLOR_ENSEM if role_b == "ensem" else None)
    name_a = _pretty_model_name(args.name_a)
    name_b = _pretty_model_name(args.name_b)

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axA, axB, axC, axD = axes.ravel()

    # Panel A: AUROC scatter
    axA.scatter(auroc_a, auroc_b, s=24, alpha=0.75)
    xy_min = 0.3
    axA.plot([xy_min, 1], [xy_min, 1], ls="--", lw=1, color="gray")
    axA.set_xlim(xy_min, 1); axA.set_ylim(xy_min, 1)
    axA.set_xlabel(f"{args.name_a} AUROC")
    axA.set_ylabel(f"{args.name_b} AUROC")
    axA.set_title("A. Pairwise AUROC")
    axA.text(0.98, 0.02, f"win {wtl_auc['win_rate']*100:.1f}% | tie {wtl_auc['tie_rate']*100:.1f}%\n"
                         f"Δ mean {sm_auc['mean']:.3f}, median {sm_auc['median']:.3f}",
              transform=axA.transAxes, ha="right", va="bottom", fontsize=11,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray"))

    # Panel B: AUPRC scatter
    axB.scatter(auprc_a, auprc_b, s=24, alpha=0.75)
    axB.plot([0, 1], [0, 1], ls="--", lw=1, color="gray")
    axB.set_xlim(0, 1); axB.set_ylim(0, 1)
    axB.set_xlabel(f"{args.name_a} AUPRC")
    axB.set_ylabel(f"{args.name_b} AUPRC")
    axB.set_title("B. Pairwise AUPRC")
    axB.text(0.98, 0.02, f"win {wtl_auprc['win_rate']*100:.1f}% | tie {wtl_auprc['tie_rate']*100:.1f}%\n"
                         f"Δ mean {sm_auprc['mean']:.3f}, median {sm_auprc['median']:.3f}",
              transform=axB.transAxes, ha="right", va="bottom", fontsize=11,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray"))

    # Panel C: Delta distributions
    # q_lo, q_hi = 0.02, 0.98
    # concat = np.concatenate([d_auc, d_auprc])
    # lo, hi = np.quantile(concat, [q_lo, q_hi])
    # pad = 0.02 * (hi - lo)           # add 2% margin on both sides
    # lo, hi = lo - pad, hi + pad
    # num_bins = 30
    # bins = np.linspace(lo, hi, num_bins)
    # axC.hist(d_auc,   bins=bins, alpha=0.85,
    #         color="#8ECAE6", edgecolor="black", linewidth=0.6,
    #         label="ΔAUROC (B−A)")
    # axC.hist(d_auprc, bins=bins, alpha=0.70,
    #         color="#FFD166", edgecolor="black", linewidth=0.6,
    #         label="ΔAUPRC (B−A)")
    # axC.set_xlim(lo, hi)
    # axC.axvline(0, color="k", lw=1)
    # axC.set_xlabel("Delta (Model B − Model A)")
    # axC.set_ylabel("Count")
    # axC.set_title("C. Delta distributions")
    # axC.legend(loc="upper left", fontsize=9)

    # axC.text(0.98, 0.98,
    #          f"AUROC  win {wtl_auc['win_rate']*100:.1f}% | tie {wtl_auc['tie_rate']*100:.1f}%\n"
    #          f"AUPRC  win {wtl_auprc['win_rate']*100:.1f}% | tie {wtl_auprc['tie_rate']*100:.1f}%",
    #          transform=axC.transAxes, ha="right", va="top", fontsize=9,
    #          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray"))
    # Unify bins
    axC.clear()
    data = [d_auc, d_auprc]
    labels = ["ΔAUROC (B−A)", "ΔAUPRC (B−A)"]

    # Violin (lighter fill)
    vp = axC.violinplot(data, showmedians=True, widths=0.5, bw_method=0.85)
    for i, b in enumerate(vp["bodies"]):
        b.set_facecolor(["#C6E6F2", "#FFE39A"][i])
        b.set_edgecolor("black")
        b.set_alpha(0.8)
    vp["cmedians"].set_color("black")

    # Overlay a boxplot (thin edges)
    bp = axC.boxplot(data, widths=0.20, showfliers=False, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("none")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    axC.axhline(0, color="k", lw=1)  # horizontal 0 line (y is data)
    axC.set_xticks([1, 2]); axC.set_xticklabels(labels, rotation=0)
    axC.set_ylabel("Delta (Model B − Model A)")
    axC.set_title("C. Delta distributions")

    # Panel D: Threshold-win curves
    axD.plot(grid_auc, curve_auc_a, label=f"{args.name_a} AUROC", color=(col_a or "#4C78A8"), ls="-", lw=2.0)
    axD.plot(grid_auc, curve_auc_b, label=f"{args.name_b} AUROC", color=(col_b or "#F58518"), ls="-", lw=2.0)
    axD.plot(grid_auprc, curve_auprc_a, ls="--", label=f"{args.name_a} AUPRC", color=(col_a or "#4C78A8"), lw=2.0)
    axD.plot(grid_auprc, curve_auprc_b, ls="--", label=f"{args.name_b} AUPRC", color=(col_b or "#F58518"), lw=2.0)
    axD.set_xlim(min(grid_auc.min(), grid_auprc.min()), max(grid_auc.max(), grid_auprc.max()))
    axD.set_ylim(0, 1)
    axD.set_xlabel("Threshold t")
    axD.set_ylabel("Proportion > t")
    axD.set_title("D. Threshold-win curves")
    handles, labels = axD.get_legend_handles_labels()
    if handles:
        axD.legend(
            handles, labels,
            loc="lower left",                 # lower left
            bbox_to_anchor=(0.02, 0.02),      # slight inset
            frameon=True, framealpha=0.90,    # white background alpha
            borderpad=0.4
        )

    if args.title:
        fig.suptitle(args.title, y=0.995, fontsize=13)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out_png, dpi=args.dpi)
    if args.out_pdf:
        fig.savefig(args.out_pdf)

    # Summary CSV next to the figure
    out_dir = os.path.dirname(os.path.abspath(args.out_png)) or "."
    summary = pd.DataFrame([{
        "n_pairs": int(d_auc.size),
        "win_rate_auroc": wtl_auc["win_rate"],
        "tie_rate_auroc": wtl_auc["tie_rate"],
        "loss_rate_auroc": wtl_auc["loss_rate"],
        "delta_auroc_mean": sm_auc["mean"],
        "delta_auroc_median": sm_auc["median"],
        "wilcoxon_p_auroc": sm_auc["p_wilcoxon_two_sided"],
        "win_rate_auprc": wtl_auprc["win_rate"],
        "tie_rate_auprc": wtl_auprc["tie_rate"],
        "loss_rate_auprc": wtl_auprc["loss_rate"],
        "delta_auprc_mean": sm_auprc["mean"],
        "delta_auprc_median": sm_auprc["median"],
        "wilcoxon_p_auprc": sm_auprc["p_wilcoxon_two_sided"],
        "scipy_available": SCIPY_OK if 'SCIPY_OK' in globals() else False,
        "model_A": args.name_a,
        "model_B": args.name_b
    }])
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print(f"Saved figure to: {args.out_png}")
    if args.out_pdf:
        print(f"Saved figure (PDF) to: {args.out_pdf}")
    print(f"Saved summary to: {os.path.join(out_dir, 'summary.csv')}")
    print(f"Paired samples after filtering: {d_auc.size}")

if __name__ == "__main__":
    main()
