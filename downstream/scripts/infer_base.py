import argparse
import os
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
)
from plyfile import PlyData, PlyElement

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_base import ProteinPLYDataset, DGCNNBinarySeg, Cfg, scan_train_stats, list_plys

# ---------- Utility: load checkpoint (supports multiple key formats + DataParallel) ----------
def _load_state_dict_flex(model: torch.nn.Module, ckpt_path: str, strict: bool = False):
    payload = torch.load(ckpt_path, map_location="cpu")
    state = None
    if isinstance(payload, dict):
        for k in ["state_dict", "model", "model_state", "model_state_dict"]:
            if k in payload and isinstance(payload[k], dict):
                state = payload[k]
                break
        if state is None:
            # Could be state_dict or a mixed payload; try to filter tensors.
            maybe = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
            state = maybe if maybe else payload
    else:
        state = payload

    # Strip DataParallel "module." prefix.
    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    print(f"[CKPT] loaded from {ckpt_path}")
    if missing:
        print(f"[CKPT][missing]: {missing}")
    if unexpected:
        print(f"[CKPT][unexpected]: {unexpected}")

# ---------- Utility: write predict back to a new PLY ----------
def _write_predict_to_ply(in_ply_path: str, probs: np.ndarray, out_dir: str, suffix: str):
    ply = PlyData.read(in_ply_path)
    v = ply["vertex"].data
    N = v.shape[0]

    # Align length (test usually no sampling; fallback if mismatched).
    if probs.shape[0] != N:
        M = min(N, probs.shape[0])
        if probs.shape[0] < N:
            pad = np.zeros(N - probs.shape[0], dtype=np.float32)
            probs = np.concatenate([probs[:M], pad], axis=0)
        else:
            probs = probs[:N]

    new_descr = list(v.dtype.descr) + [("predict", "f4")]
    v_new = np.empty(N, dtype=new_descr)
    for name in v.dtype.names:
        v_new[name] = v[name]
    v_new["predict"] = probs.astype(np.float32)

    vertex_el = PlyElement.describe(v_new, "vertex")

    # Rebuild PlyData; do not mutate tuple in-place.
    elements = []
    replaced = False
    for el in ply.elements:
        if el.name == "vertex" and not replaced:
            elements.append(vertex_el)
            replaced = True
        else:
            elements.append(el)
    if not replaced:
        elements.insert(0, vertex_el)

    ply_new = PlyData(
        elements=elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=list(getattr(ply, "comments", [])),
        obj_info=list(getattr(ply, "obj_info", [])),
    )

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(in_ply_path)
    stem, ext = os.path.splitext(base)
    out_fp = os.path.join(out_dir, f"{stem}{suffix}{ext}")
    ply_new.write(out_fp)
    return out_fp

# ---------- Metrics ----------
def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5):
    y_pred = (y_score >= thr).astype(np.int32)
    metrics = {}
    metrics["ACC"]   = float(accuracy_score(y_true, y_pred))
    metrics["F1"]    = float(f1_score(y_true, y_pred, zero_division=0))
    # AUROC/AUPRC require both classes to be present.
    try:
        metrics["AUROC"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["AUROC"] = float("nan")
    try:
        metrics["AUPRC"] = float(average_precision_score(y_true, y_score))
    except Exception:
        metrics["AUPRC"] = float("nan")
    try:
        metrics["MCC"]   = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        metrics["MCC"] = float("nan")
    return metrics

# ---------- Normalize seq_global (avoid flattening to B*D) ----------
def _norm_seq_global_to_BxD(g_seq: torch.Tensor, data, seq_dim: int) -> torch.Tensor:
    """
    Normalize data.seq_global to [B, seq_dim] without cross-graph averaging.
    Supported shapes:
      - [seq_dim]
      - [B, seq_dim]
      - [1, B*seq_dim]
      - [B*seq_dim]
      - [B, L, seq_dim]  -> mean-pool over L
    """
    if g_seq is None:
        return None
    if hasattr(data, "num_graphs"):
        B = int(data.num_graphs)
    elif hasattr(data, "ptr"):
        B = int(data.ptr.numel() - 1)
    else:
        B = int(data.batch.max().item()) + 1

    if g_seq.dim() == 1:
        if g_seq.numel() == seq_dim:               # [D]
            return g_seq.view(1, seq_dim)
        if g_seq.numel() == B * seq_dim:           # [B*D]
            return g_seq.view(B, seq_dim)
    elif g_seq.dim() == 2:
        if g_seq.shape == (B, seq_dim):
            return g_seq
        if g_seq.shape == (1, B * seq_dim):
            return g_seq.view(B, seq_dim)
        if g_seq.size(1) == seq_dim:               # [?, D]
            out = g_seq.view(-1, seq_dim)
            assert out.size(0) == B, f"seq_global rows {out.size(0)} != B {B}"
            return out
    elif g_seq.dim() == 3 and g_seq.size(0) == B and g_seq.size(-1) == seq_dim:
        return g_seq.mean(dim=1)                   # [B, L, D] -> [B, D]
    raise RuntimeError(f"Unsupported seq_global shape {tuple(g_seq.shape)} (B={B}, D={seq_dim})")

# ========== Core: direct inference ==========
def run_inference_only(
    ckpt_path: str,
    test_files: list,
    stats: dict,
    model_cls,                 # model class (e.g., DGCNNBinarySeg)
    model_kwargs: dict,        # model init kwargs
    out_dir: str,
    suffix: str = "_pred",
    batch_size: int = 1,       # keep 1 to avoid seq_global being flattened in batch
    num_workers: int = 2,
    device: torch.device = None,
    seq_dim: int = 1280,       # ESM embedding dimension
    print_every: int = 20
):
    """
    Load checkpoint, run inference on each PLY in test_files, compute metrics,
    and write predict back into new PLY files.
    Returns: metrics dict.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # 1) Build model and load weights
    model = model_cls(**(model_kwargs or {}))
    model.to(device)
    _load_state_dict_flex(model, ckpt_path, strict=False)
    model.eval()

    # 2) Build dataset and DataLoader (bs=1 is safest)
    ds = ProteinPLYDataset(test_files, stats=stats, use_aug=False, max_points=None)  # no downsampling
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))

    # 3) Inference and write-back
    all_true, all_score = [], []
    os.makedirs(out_dir, exist_ok=True)

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
        for i, data in enumerate(loader):
            # With DataLoader(bs=1), Data is a batched object; use as-is.
            # If your model needs seq_global shape fixes, do it in forward or keep bs=1.
            data = data.to(device)

            # === Optional: fallback seq_global fix if forward does not handle it ===
            if hasattr(data, "seq_global") and isinstance(data.seq_global, torch.Tensor):
                try:
                    data.seq_global = _norm_seq_global_to_BxD(data.seq_global, data, seq_dim)
                except Exception as e:
                    # Unlikely with bs=1; print once as a fallback.
                    print(f"[WARN] seq_global shape fix failed at idx={i}: {e}")

            logits = model(data)                            # (N,)
            probs  = torch.sigmoid(logits).float().cpu().numpy()
            labels = data.y.view(-1).cpu().numpy().astype(np.int32)

            # Collect metrics
            all_true.append(labels)
            all_score.append(probs)

            # Write to new PLY
            fp = test_files[i]  # aligned with shuffle=False + bs=1
            _write_predict_to_ply(fp, probs, out_dir=out_dir, suffix=suffix)

            if (i + 1) % print_every == 0:
                print(f"[INF] {i+1}/{len(loader)} done: {os.path.basename(fp)}")

    # 4) Compute overall metrics
    y_true  = np.concatenate(all_true, axis=0)
    y_score = np.concatenate(all_score, axis=0)
    metrics = _compute_metrics(y_true, y_score, thr=0.5)

    print("[METRICS] ", " | ".join(f"{k}={v:.4f}" if v==v else f"{k}=nan" for k, v in metrics.items()))
    print(f"[DONE] wrote predicted PLYs to: {out_dir}")
    return metrics

def parse_args():
    ap = argparse.ArgumentParser(description="Run inference and write predict to PLY files.")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path")
    ap.add_argument("--input", required=True, help="Directory with input PLY files")
    ap.add_argument("--output", required=True, help="Output directory for predicted PLY files")
    ap.add_argument("--stats", default=None, help="Optional stats.json for normalization")
    ap.add_argument("--suffix", default=Cfg.PRED_SUFFIX, help="Suffix for output PLY filenames")
    ap.add_argument("--k", type=int, default=20, help="kNN neighbors for the model")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size (keep 1 for stability)")
    ap.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    ap.add_argument("--device", default=Cfg.DEVICE, help="Device override (e.g., cuda:0 or cpu)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    test_files = list_plys(args.input)
    if len(test_files) == 0:
        raise SystemExit(f"[ERROR] No .ply files found in {args.input}")

    if args.stats and os.path.exists(args.stats):
        with open(args.stats, "r", encoding="utf-8") as f:
            stats = json.load(f)
    else:
        train_files = list_plys(os.path.join(Cfg.DATA_ROOT, Cfg.TRAIN_DIR))
        stats = scan_train_stats(train_files)
        os.makedirs(os.path.dirname(Cfg.STATS_CACHE), exist_ok=True)
        with open(Cfg.STATS_CACHE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    model_kwargs = dict(k=args.k)
    _ = run_inference_only(
        ckpt_path=args.ckpt,
        test_files=test_files,
        stats=stats,
        model_cls=DGCNNBinarySeg,
        model_kwargs=model_kwargs,
        out_dir=args.output,
        suffix=args.suffix,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seq_dim=1280,
    )
