#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_pretrain.py

Load a saved binary segmentation model (state_dict) and run inference on a folder of PLY files.
For each input PLY, write a new PLY that adds `property float predict` per vertex.

Assumptions (match your training):
- Vertex properties include at least: x, y, z, charge, hbond, hphob, iface, nx, ny, nz
- Optional: label (if present you can compute metrics via --report-metrics)
- Optional: residue_id (not used for inference)
- Optionally fuse a global embedding per protein (see --pre-emb-dir, --pre-dim, --pre-fusion)

Usage:
  python scripts/infer_pretrain.py \
      --ckpt /path/to/best.pt \
      --input /path/to/test_dir \
      --output /path/to/out_dir \
      --stats /path/to/stats.json \
      --pre-emb-dir /path/to/emb  --pre-dim 1280 --pre-fusion concat

Notes:
- If --stats is not given, scalars (charge,hbond,hphob,iface) are NOT standardized.
- If a per-protein *.npy is missing, a zero vector is used as fallback.
"""
import os, re, sys, glob, json, math, argparse, warnings
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch-geometric deps
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_max_pool
from torch_cluster import knn_graph
from torch_scatter import scatter_max, scatter_mean

from plyfile import PlyData, PlyElement
import numpy.lib.recfunctions as rfn
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_pretrain import EdgeConvBlock, DGCNNBinarySeg

# ---------------------------- Utils ----------------------------

def extract_protein_id(ply_path: str) -> str:
    """A0A0A0_A.ply -> A0A0A0; 1ABC.ply -> 1ABC"""
    base = os.path.basename(ply_path)
    stem, _ = os.path.splitext(base)
    return re.split(r'[_]', stem)[0]

def load_pre_global(protein_id: str, pre_emb_dir: str, pre_dim: int) -> torch.Tensor:
    """
    Load protein-level global embedding.
    Accepts either shape [D] or [L, D] (then mean-pool on L).
    Returns [1, D] tensor (L2-normalized).
    """
    npy = os.path.join(pre_emb_dir, f"{protein_id}.npy")
    if not os.path.exists(npy):
        # fallback: zeros
        return torch.zeros(1, pre_dim, dtype=torch.float32)
    arr = np.load(npy)
    if arr.ndim == 2:  # [L, D] -> mean pool
        arr = arr.mean(axis=0)
    arr = arr.astype(np.float32)
    norm = np.linalg.norm(arr) + 1e-6
    arr = arr / norm
    return torch.from_numpy(arr).unsqueeze(0)  # [1, D]

def center_scale_pos(pos: np.ndarray) -> np.ndarray:
    """Center positions, scale by mean point norm (robust to units)."""
    pos = pos - pos.mean(0, keepdims=True)
    scale = np.linalg.norm(pos, axis=1).mean()
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    return (pos / scale).astype(np.float32)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_stats(stats_path: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load feature normalization stats.json produced at training.
    Expected keys: "means": [4], "stds": [4].
    Returns (means[4], stds[4]) or None if stats_path is None.
    """
    if not stats_path:
        return None
    with open(stats_path, "r") as f:
        obj = json.load(f)
    means = np.array(obj["means"], dtype=np.float32).reshape(1, 4)
    stds  = np.array(obj["stds"],  dtype=np.float32).reshape(1, 4)
    stds = np.clip(stds, 1e-8, None)
    return means, stds

# ---------------------------- Inference ----------------------------

def read_ply_as_arrays(ply_path: str):
    ply = PlyData.read(ply_path)
    v = ply['vertex'].data
    # Required fields
    for f in ['x','y','z','charge','hbond','hphob','iface','nx','ny','nz']:
        if f not in v.dtype.names:
            raise KeyError(f"[{os.path.basename(ply_path)}] vertex is missing required field '{f}'")
    pos = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)         # (N,3)
    scalars = np.stack([v['charge'], v['hbond'], v['hphob'], v['iface']], axis=1).astype(np.float32)  # (N,4)
    normals = np.stack([v['nx'], v['ny'], v['nz']], axis=1).astype(np.float32)  # (N,3)
    label = v['label'].astype(np.float32) if 'label' in v.dtype.names else None
    return ply, v, pos, scalars, normals, label

@torch.no_grad()
def infer_one(
    model: nn.Module,
    ply_path: str,
    device: torch.device,
    means_stds: Optional[Tuple[np.ndarray, np.ndarray]],
    pre_emb_dir: Optional[str],
    pre_dim: int,
    pre_fusion: str
) -> np.ndarray:
    """Return per-point probabilities in numpy float32 shape (N,)."""
    # 1) Read PLY -> numpy
    ply, v, pos, scalars, normals, _ = read_ply_as_arrays(ply_path)

    # 2) Normalize four scalar features (match training)
    if means_stds is not None:
        means, stds = means_stds  # (1,4), (1,4)
        scalars = (scalars - means) / stds

    # 3) Coordinate normalization (match training: center -> scale)
    pos_n = center_scale_pos(pos)  # (N,3)

    # 4) Training uses x_feat = [pos(3) | scalars(4) | normals(3)], but x uses x_feat[:,3:] = 7 dims
    x_feat = np.concatenate([pos_n, scalars, normals], axis=1).astype(np.float32)  # (N,10)
    x7 = x_feat[:, 3:]  # (N,7)

    # 5) Build Data (field names must match training)
    import os
    N = pos_n.shape[0]
    data = Data(
        pos=torch.from_numpy(pos_n).float().to(device),     # Training uses pos/pos_in/xyz
        x=torch.from_numpy(x7).float().to(device),          # (N,7)
        pos_in=torch.from_numpy(pos_n).float().to(device),  # (N,3)
        xyz=torch.from_numpy(pos_n).float().to(device),     # (N,3)
        name=os.path.basename(ply_path)
    )
    # Single-graph inference: set batch=0
    data.batch = torch.zeros(N, dtype=torch.long, device=device)

    # 6) Fuse pretrained global vector: field name must be pretrain_global
    if pre_emb_dir:
        pid = extract_protein_id(ply_path)                  # Example: A0A0A0_A.ply -> A0A0A0
        g = load_pre_global(pid, pre_emb_dir, pre_dim).to(device)  # [1, D] (L2-normalized)
        data.pretrain_global = g
    else:
        data.pretrain_global = None  # Model uses use_pretrain_global to set head dims

    # 7) Forward inference
    logits = model(data)  # (N,)
    probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)  # (N,)
    return probs

def write_ply_with_predict(ply: PlyData, v_rec, preds: np.ndarray, out_path: str):
    """
    Append or replace 'predict' field and write a new PLY.
    Keeps face elements if present.
    """
    preds = preds.astype(np.float32)
    names = v_rec.dtype.names
    if 'predict' in names:
        # Replace existing field
        v_new = v_rec.copy()
        v_new['predict'] = preds
    else:
        # Append new field at the end
        v_new = rfn.append_fields(v_rec, 'predict', preds, usemask=False, dtypes=np.float32)
        v_new = rfn.repack_fields(v_new)

    # Build PlyElements
    v_el = PlyElement.describe(v_new, 'vertex')
    els = [v_el]
    if 'face' in ply.elements and ply['face'] is not None:
        els.append(ply['face'])  # reuse existing face element

    PlyData(els, text=ply.text).write(out_path)

def compute_metrics_if_possible(all_probs: List[np.ndarray], all_labels: List[np.ndarray]) -> dict:
    # Only compute if every label array is available and matches length
    if len(all_probs) == 0 or len(all_labels) == 0:
        return {}
    try:
        import sklearn.metrics as skm
    except Exception:
        warnings.warn("scikit-learn not installed; skip metrics.")
        return {}
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate([y for y in all_labels if y is not None], axis=0) if all(all_labels) else None
    if labels is None or labels.shape[0] != probs.shape[0]:
        return {}
    preds_bin = (probs >= 0.5).astype(np.int32)
    ACC = skm.accuracy_score(labels, preds_bin)
    F1  = skm.f1_score(labels, preds_bin, zero_division=0)
    try:
        AUROC = skm.roc_auc_score(labels, probs)
    except Exception:
        AUROC = float('nan')
    try:
        AUPRC = skm.average_precision_score(labels, probs)
    except Exception:
        AUPRC = float('nan')
    # MCC
    try:
        MCC = skm.matthews_corrcoef(labels, preds_bin)
    except Exception:
        MCC = float('nan')
    return {"ACC": ACC, "F1": F1, "AUROC": AUROC, "AUPRC": AUPRC, "MCC": MCC}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to model state_dict .pt (or directory containing best.pt)")
    ap.add_argument("--input", required=True, help="Directory containing input .ply files")
    ap.add_argument("--output", required=True, help="Directory to save predicted .ply files")
    ap.add_argument("--stats", default=None, help="Path to stats.json with 'means' and 'stds' for 4 scalar features")
    ap.add_argument("--k", type=int, default=12, help="k for EdgeConv kNN")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    ap.add_argument("--pre-emb-dir", default=None, help="Directory with per-protein *.npy; enable to fuse pre")
    ap.add_argument("--pre-dim", type=int, default=1280, help="Embedding dimensionality")
    ap.add_argument("--pre-fusion", choices=["concat","replace","film"], default="concat", help="How to fuse pre global")
    ap.add_argument("--suffix", default="_pred", help="Suffix for output files (before .ply)")
    ap.add_argument("--report-metrics", action="store_true", help="If labels exist, compute ACC,F1,AUROC,AUPRC,MCC")
    args = ap.parse_args()

    ckpt_path = args.ckpt
    if os.path.isdir(ckpt_path):
        cand = os.path.join(ckpt_path, "best.pt")
        if os.path.exists(cand):
            ckpt_path = cand
    if not os.path.exists(ckpt_path):
        print(f"[ERR] checkpoint not found: {ckpt_path}", file=sys.stderr); sys.exit(2)

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    # Build model and load weights
    model = DGCNNBinarySeg(k=args.k, use_pretrain_global=(args.pre_emb_dir is not None),
                           pretrain_dim=args.pre_dim, pretrain_fusion=args.pre_fusion).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] loaded from {ckpt_path}")
    if missing:
        print("[CKPT] missing keys:", missing)
    if unexpected:
        print("[CKPT] unexpected keys:", unexpected)
    model.eval()

    # Stats
    means_stds = load_stats(args.stats)  # or None

    # Gather files
    files = sorted(glob.glob(os.path.join(args.input, "*.ply")))
    if len(files) == 0:
        print(f"[WARN] no .ply found under {args.input}")
        return

    all_probs, all_labels = [], []
    for i, fp in enumerate(files, 1):
        try:
            ply, v, pos, scalars, normals, label = read_ply_as_arrays(fp)
            probs = infer_one(model, fp, device, means_stds, args.pre_emb_dir, args.pre_dim, args.pre_fusion)
            # Save new PLY
            base = os.path.basename(fp)
            stem, ext = os.path.splitext(base)
            out_fp = os.path.join(args.output, f"{stem}{args.suffix}{ext}")
            write_ply_with_predict(ply, v, probs, out_fp)
            print(f"[{i:04d}/{len(files)}] wrote {out_fp} (N={len(v)})")
            all_probs.append(probs)
            if label is not None:
                all_labels.append(label.astype(np.float32))
        except Exception as e:
            print(f"[ERR] {os.path.basename(fp)} -> {e}", file=sys.stderr)

    if args.report_metrics and len(all_labels) == len(all_probs) and len(all_probs) > 0:
        metrics = compute_metrics_if_possible(all_probs, all_labels)
        if metrics:
            print("[METRICS]", {k: float(v) if v==v else None for k,v in metrics.items()})  # print NaN as None

if __name__ == "__main__":
    main()
