import os
import glob
import math
import random
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from plyfile import PlyData, PlyElement

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool
from torch_scatter import scatter_max, scatter_mean
from torch_cluster import knn_graph
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ppi.paths import get_data_root, get_outputs_root

# ---------------- Config ----------------
class Cfg:
    DATA_ROOT = str(get_data_root())
    DATASET_TAG = Path(DATA_ROOT).name or "dataset"
    TRAIN_DIR = "labeled_ply_train"
    VAL_DIR   = "labeled_ply_test"
    TEST_DIR  = "labeled_ply_test"

    HOLDOUT_FRACTION = 0.1       # Fraction for splitting val from train.
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    K = 12                       # kNN neighbors.
    EPOCHS = 60
    LR = 1e-3
    WD = 1e-4
    BATCH_TARGET_POINTS = 40_000
    MAX_POINTS_PER_PC = 16_000

    USE_AUG = True
    OUTPUTS_ROOT = get_outputs_root()
    SAVE_DIR = str(
        OUTPUTS_ROOT
        / "runs"
        / "binary_dgcnn"
        / DATASET_TAG
        / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    )
    STATS_CACHE = str(OUTPUTS_ROOT / "runs" / "binary_dgcnn" / DATASET_TAG / "stats.json")
    PRED_PLY_DIR = os.path.join(SAVE_DIR, "test_pred_ply")
    PRED_SUFFIX  = "_pred"

# -------------- Utils & Aug --------------
def fix_seed(s=Cfg.SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def random_rotation_matrix():
    a, b, c = np.random.rand(3) * 2*np.pi
    Rx = np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]], dtype=np.float32)
    Ry = np.array([[math.cos(b),0,math.sin(b)],[0,1,0],[-math.sin(b),0,math.cos(b)]], dtype=np.float32)
    Rz = np.array([[math.cos(c),-math.sin(c),0],[math.sin(c),math.cos(c),0],[0,0,1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)

def center_scale_pos(x):
    x = x - x.mean(0, keepdims=True)
    scale = np.linalg.norm(x, axis=1).mean()
    scale = scale if scale > 1e-6 else 1.0
    return (x / scale).astype(np.float32)

# --------- Dataset & stats scan ----------
def list_plys(root):
    return sorted(glob.glob(os.path.join(root, "*.ply")))

def ensure_dirs():
    os.makedirs(Cfg.SAVE_DIR, exist_ok=True)

def _check_np_arr(name, arr, fn, fp):
    import numpy as np
    if not np.isfinite(arr).all():
        bad = np.sum(~np.isfinite(arr))
        raise RuntimeError(
            f"[DATA-NONFINITE] {name} has {bad} non-finite values in {fn} ({fp}). "
            f"min={np.nanmin(arr)}, max={np.nanmax(arr)}"
        )

def scan_train_stats(train_files, max_files=1500, sample_points=20000, seed=42):
    random.seed(seed); np.random.seed(seed)
    files = random.sample(train_files, min(max_files, len(train_files)))
    sums = np.zeros(4, np.float64); sqs = np.zeros(4, np.float64)
    cnt = pos = tot = 0
    for i, fp in enumerate(files):
        if i % 100 == 0:
            print(f"[stats] {i}/{len(files)}", flush=True)
        v = PlyData.read(fp)['vertex'].data
        N = len(v)
        if N > sample_points:
            idx = np.random.choice(N, sample_points, replace=False)
        else:
            idx = np.arange(N)
        feats = np.stack([v['charge'][idx], v['hbond'][idx], v['hphob'][idx], v['iface'][idx]], 1).astype(np.float32)
        labels = v['label'][idx].astype(np.float32)
        sums += feats.sum(0); sqs += (feats**2).sum(0)
        cnt += feats.shape[0]; pos += (labels > 0.5).sum(); tot += labels.shape[0]
    means = (sums / max(cnt,1)); vars_ = (sqs / max(cnt,1) - means**2).clip(min=1e-8)
    return {"means": means.tolist(), "stds": np.sqrt(vars_).tolist(), "pos_ratio": float(pos)/max(tot,1)}

class ProteinPLYDataset(Dataset):
    def __init__(self, files: List[str], stats: Dict=None, use_aug=False, max_points=None):
        self.files = files
        self.use_aug = use_aug
        self.max_points = max_points
        self.means = np.array(stats["means"], dtype=np.float32) if stats else np.zeros(4, np.float32)
        self.stds  = np.array(stats["stds"],  dtype=np.float32) if stats else np.ones(4,  np.float32)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        ply = PlyData.read(fp)
        v = ply['vertex'].data

        v   = PlyData.read(fp)['vertex'].data
        pos = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float32)
        xft = np.stack([v['charge'], v['hbond'], v['hphob'], v['iface']], 1).astype(np.float32)
        y   = v['label'].astype(np.float32)
        fn  = os.path.basename(fp)

        # Check raw values before normalization.
        _check_np_arr("pos(raw)", pos, fn, fp)
        _check_np_arr("xft(raw)", xft, fn, fp)
        _check_np_arr("y(raw)",   y,   fn, fp)

        # Apply feature normalization if stats.json is provided.
        means = self.means
        stds  = self.stds
        if (stds <= 1e-12).any():
            raise RuntimeError(f"[STATS-ZERO-STD] std<=1e-12 at {np.where(stds<=1e-12)} "
                            f"for file {fn} -> divide by zero risk.")
        xft = (xft - means) / stds

        # Position normalization (if enabled).
        pos = center_scale_pos(pos)

        # Validate label range (report only).
        if (y.min() < 0) or (y.max() > 1):
            raise RuntimeError(f"[LABEL-OUT-OF-RANGE] {fn} y.min={y.min()} y.max={y.max()}")

        # Check values after normalization.
        _check_np_arr("pos(norm)", pos, fn, fp)
        _check_np_arr("xft(norm)", xft, fn, fp)

        pos = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)     # (N,3)
        normals = np.stack([v['nx'], v['ny'], v['nz']], axis=1).astype(np.float32)  # (N,3)
        scalars = np.stack([v['charge'], v['hbond'], v['hphob'], v['iface']], axis=1).astype(np.float32)  # (N,4)
        y = v['label'].astype(np.float32)  # (N,)
        if y.min() < 0 or y.max() > 1:
            y = (np.abs(y) > 0.5).astype(np.float32)
        y = np.clip(y, 0.0, 1.0).astype(np.float32)

        # norm
        pos = center_scale_pos(pos)
        scalars = (scalars - self.means) / self.stds

        # Augmentations (train only).
        if self.use_aug:
            R = random_rotation_matrix()
            pos = (R @ pos.T).T
            normals = (R @ normals.T).T
            pos = pos * np.random.uniform(0.95, 1.05)
            pos = pos + np.random.normal(scale=0.002, size=pos.shape)
            
        # Subsample oversized point clouds.
        N = pos.shape[0]
        if self.max_points and N > self.max_points:
            idxs = np.random.choice(N, self.max_points, replace=False)
            idxs.sort()
            pos, normals, scalars, y = pos[idxs], normals[idxs], scalars[idxs], y[idxs]
            
        # Input to first layer uses [pos | features], 10 dims.
        x_feat = np.concatenate([pos, scalars, normals], axis=1).astype(np.float32)  # (N,10)
        data = Data(
            pos = torch.from_numpy(pos).float(),
            x = torch.from_numpy(x_feat[:, 3:]).float(),
            pos_in = torch.from_numpy(pos).float(),
            xyz = torch.from_numpy(pos).float(),
            y = torch.from_numpy(y).float(),
            name = os.path.basename(fp)
            )
        return data

# --------------- Model -------------------
class EdgeConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_ch*2, out_ch), nn.ReLU(),
            nn.Linear(out_ch, out_ch), nn.ReLU()
        )
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x, pos, batch):
        # x: (N, C), pos: (N,3)
        assert pos.is_cuda, "pos is on CPU; install CUDA wheels for torch-cluster / move data to CUDA first."
        if not torch.isfinite(pos).all():
            raise RuntimeError("[POS-NONFINITE] before KNN; "
                            f"min={pos.nanmean():.3e} anyNaN={(~torch.isfinite(pos)).any().item()}")
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=False)  # (2, E)
        if edge_index.max() >= pos.size(0) or edge_index.min() < 0:
            raise RuntimeError(f"[EDGE-OOB] got index out of range, max={edge_index.max().item()} "
                            f"N={pos.size(0)}")
        E = edge_index.size(1)
        if torch.cuda.is_available() and (random.random() < 0.004):
            m_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[dbg] N={pos.size(0)} k={self.k} E={E} peak={m_used:.2f}GB", flush=True)
        row, col = edge_index  # edges: col -> row (neighbor j -> center i)
        x_i = x[row]
        x_j = x[col]
        m = self.mlp(torch.cat([x_i, x_j - x_i], dim=-1))
        # aggregate max over neighbors per i
        m, _ = scatter_max(m, row, dim=0, dim_size=x.size(0))
        return self.bn(m)

class DGCNNBinarySeg(nn.Module):
    def __init__(self, k=20):
        super().__init__()
        in0 = 10  # Initial input: 10 dims (xyz + 7 features).
        self.proj0 = nn.Linear(in0, 64)
        
        self.ec1 = EdgeConvBlock(64,  96, k)
        self.ec2 = EdgeConvBlock(96,  160, k)
        self.ec3 = EdgeConvBlock(160, 256, k)
        
        self.mlp_local = nn.Sequential(
            nn.Linear(96+160+256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mlp_global = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        
        self.head = nn.Sequential(
            nn.Linear(256+256, 256), nn.ReLU(),
            nn.Linear(256, 1)  # binary logit
        )

    def forward(self, data):
        # Rebuild first-layer input: xyz(3) + x(7).
        x0 = torch.cat([data.xyz, data.x], dim=1)
        x0 = self.proj0(x0)

        h1 = self.ec1(x0, data.pos_in, data.batch)   # (N,96)
        h2 = self.ec2(h1, data.pos_in, data.batch)   # (N,160)
        h3 = self.ec3(h2, data.pos_in, data.batch)   # (N,256)

        h = torch.cat([h1,h2,h3], dim=1)
        h_local = self.mlp_local(h)                  # (N,256)
        
        g = global_max_pool(h_local, data.batch)     # (B,256)
        g = self.mlp_global(g)
        g_exp = g[data.batch]                        # (N,256)
        
        logits = self.head(torch.cat([h_local, g_exp], dim=1))  # (N,1)
        return logits.squeeze(1)

# --------------- DataLoader --------------
def make_loader(files, stats, use_aug, shuffle, target_points, max_points):
    probe = min(16, len(files))
    est = []
    for i in np.linspace(0, len(files)-1, num=probe, dtype=int):
        ply = PlyData.read(files[i])
        est.append(len(ply['vertex'].data))
    mean_n = int(np.mean(est)) if est else 8000
    bs = max(1, target_points // max(1, mean_n))
    ds = ProteinPLYDataset(files, stats=stats, use_aug=use_aug, max_points=max_points)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=4, pin_memory=True)

# --------------- Train/Eval --------------
def compute_metrics(all_probs, all_y, names=None, batches=None):
    y_true = np.concatenate(all_y).astype(np.float32)
    y_prob = np.concatenate(all_probs).astype(np.float32)
    y_pred = (y_prob >= 0.5).astype(np.float32)

    metrics = {}

    # --- AUPRC / AUROC ---
    try:
        metrics["AUPRC"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["AUPRC"] = None
    try:
        metrics["AUROC"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None
    except Exception:
        metrics["AUROC"] = None

    # --- F1 (point-level), ACC, MCC ---
    try:
        # If all one class, sklearn f1 may fail; guard.
        metrics["F1"] = float(f1_score(y_true, y_pred)) if len(np.unique(y_true)) == 2 else None
        metrics["F1_point"] = metrics["F1"]  # Backward-compatible key.
    except Exception:
        metrics["F1"] = None
        metrics["F1_point"] = None

    try:
        metrics["ACC"] = float((y_pred == y_true).mean())
    except Exception:
        metrics["ACC"] = None

    try:
        # sklearn returns 0.0 for degenerate cases; guard again.
        metrics["MCC"] = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) == 2 else None
    except Exception:
        metrics["MCC"] = None

    # --- Optional protein-level F1 aggregation ---
    if names is not None and batches is not None:
        by_prot = defaultdict(lambda: {"y": [], "p": []})
        idx = 0
        for probs, y in zip(all_probs, all_y):
            n = len(y)
            by_prot[names[idx]]["y"].extend(y.tolist())
            by_prot[names[idx]]["p"].extend(probs.tolist())
            idx += 1
        f1s = []
        for k, d in by_prot.items():
            yp = (np.array(d["p"]) >= 0.5).astype(np.float32)
            yt = np.array(d["y"]).astype(np.float32)
            if len(np.unique(yt)) == 2:
                f1s.append(f1_score(yt, yp))
        if f1s:
            metrics["F1_protein_mean"] = float(np.mean(f1s))

    return metrics

def compute_loss(logits, y, pos_weight=None):
    y = y.float()
    # Prevent pos_weight 0/inf.
    if pos_weight is not None:
        if not torch.isfinite(pos_weight):
            pos_weight = None
    loss = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=pos_weight, reduction='mean'
    )
    # Defensive: dump on failure.
    if torch.isnan(loss) or torch.isinf(loss):
        print("[nan/inf loss] logits:", logits.min().item(), logits.max().item(),
              " y.mean:", y.mean().item(), " pos_weight:", 
              (pos_weight.item() if isinstance(pos_weight, torch.Tensor) else pos_weight),
              flush=True)
    return loss

def dump_bad_batch(data, tag="nonfinite"):
    import numpy as np, os, time
    out_dir = Path(Cfg.OUTPUTS_ROOT) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}_{int(time.time())}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out,
        name=str(getattr(data, "name", "unknown")),
        pos=data.pos_in.detach().cpu().numpy(),
        x=data.x.detach().cpu().numpy(),
        y=data.y.detach().cpu().numpy(),
    )
    print(f"[DUMP] saved batch to {out}", flush=True)

def assert_finite(name, t):
    if torch.is_tensor(t):
        ok = torch.isfinite(t).all()
        if not ok:
            tnum = torch.nan_to_num(t)
            raise RuntimeError(
                f"[NONFINITE] {name}: min={tnum.min().item():.3e} "
                f"max={tnum.max().item():.3e} anyNaN={(~torch.isfinite(t)).any().item()}"
            )

def train_epoch(model, loader, opt, pos_weight):
    """
    One training epoch with optional AMP.
    - Accumulates point-weighted average BCE loss (more stable).
    - Moves pos_weight to the correct device automatically.
    - Enables AMP when CUDA is available; otherwise uses FP32.
    """
    model.train()
    device = Cfg.DEVICE

    # Accumulators.
    total_loss = 0.0
    total_points = 0

    # Move pos_weight to correct device/type (float/Tensor/None allowed).
    if isinstance(pos_weight, torch.Tensor):
        pw = pos_weight.to(device=device, dtype=torch.float32)
    elif pos_weight is None:
        pw = None
    else:
        pw = torch.tensor(float(pos_weight), device=device, dtype=torch.float32)

    use_amp = False #torch.cuda.is_available() and (str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for data in loader:
        try:
            # Move batch to device; non_blocking helps when pin_memory=True.
            data = data.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            # Forward + loss (use autocast when AMP is enabled).
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(data)  # (N,)
                assert_finite("logits", logits)
                if data.y.min() < 0 or data.y.max() > 1:
                    print(f"[warn] label out of [0,1] fixed: {data.name} min={data.y.min():.1f} max={data.y.max():.1f}")
                loss = compute_loss(logits, data.y, pw)  # Your stable BCE-with-logits wrapper.
                assert_finite("loss", loss)
                # print(loss)
        except Exception as e:
            print(f"[BATCH-FAIL] sample={getattr(data,'name','?')} N={data.pos_in.size(0)} err={e}")
            dump_bad_batch(data)
            raise

        # Backward + update (AMP/non-AMP).
        if use_amp:
            scaler.scale(loss).backward()
            # Uncomment for grad clipping, e.g., clip=1.0.
            # scaler.unscale_(opt)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        # Weighted accumulation by points in this batch.
        n_points = int(data.y.numel())
        total_loss += float(loss.detach().item()) * n_points
        total_points += n_points

    # Return epoch-average point loss.
    return total_loss / max(total_points, 1)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    total_points = 0
    all_probs, all_y, names = [], [], []
    for data in loader:
        data = data.to(Cfg.DEVICE)
        # logits = model(data)
        # loss = F.binary_cross_entropy_with_logits(logits, data.y)
        with torch.amp.autocast('cuda'):
            logits = model(data)
            loss = compute_loss(logits, data.y)
        # total_loss += loss.item()
        total_loss += loss.item() * data.y.numel()
        total_points += data.y.numel()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_y.append(data.y.detach().cpu().numpy())
        # Collect protein names (batch may contain multiple; simplified).
        names.extend(data.name)
    metrics = compute_metrics(all_probs, all_y, names=names, batches=None)
    # return total_loss / max(1,len(loader)), metrics
    return total_loss / max(total_points, 1), metrics

@torch.no_grad()
def save_predictions_on_test(model, files, stats, out_dir=None, suffix=None):
    """Run per-PLY inference and write `property float predict` into new PLY files."""
    out_dir = out_dir or Cfg.PRED_PLY_DIR
    suffix = suffix or Cfg.PRED_SUFFIX
    os.makedirs(out_dir, exist_ok=True)

    ds = ProteinPLYDataset(files, stats=stats, use_aug=False, max_points=None)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()

    for i, data in enumerate(loader):
        fp = files[i]  # Aligned with shuffle=False + bs=1.
        data = data.to(Cfg.DEVICE)

        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=(Cfg.DEVICE == 'cuda')):
            logits = model(data)

        probs = torch.sigmoid(logits).view(-1).detach().cpu().numpy().astype(np.float32)

        # Read original PLY.
        ply = PlyData.read(fp)
        v = ply['vertex'].data
        N = v.shape[0]

        # Align length (test usually no sampling; fallback if mismatched).
        if probs.shape[0] != N:
            M = min(N, probs.shape[0])
            if probs.shape[0] < N:
                pad = np.zeros(N - probs.shape[0], dtype=np.float32)
                probs = np.concatenate([probs[:M], pad], axis=0)
            else:
                probs = probs[:N]

        # Copy fields and append predict.
        new_descr = list(v.dtype.descr) + [('predict', 'f4')]
        v_new = np.empty(N, dtype=new_descr)
        for name in v.dtype.names:
            v_new[name] = v[name]
        v_new['predict'] = probs

        # Create new vertex element.
        vertex_el = PlyElement.describe(v_new, 'vertex')

        # Rebuild PlyData with new elements list (do not mutate tuple).
        elements = []
        replaced = False
        for el in ply.elements:
            if el.name == 'vertex' and not replaced:
                elements.append(vertex_el)
                replaced = True
            else:
                elements.append(el)
        if not replaced:
            # If no vertex element (rare), insert at the front.
            elements.insert(0, vertex_el)

        # Preserve text/binary, endianness, and comments.
        ply_new = PlyData(
            elements=elements,
            text=ply.text,
            byte_order=ply.byte_order,
            comments=list(getattr(ply, 'comments', [])),
            obj_info=list(getattr(ply, 'obj_info', [])),
        )

        # Write back.
        base = os.path.basename(fp)
        stem, ext = os.path.splitext(base)
        out_fp = os.path.join(out_dir, f"{stem}{suffix}{ext}")
        ply_new.write(out_fp)

def register_nan_hooks(model):
    import types
    def hook(name):
        def _hook(module, inp, out):
            t = out if torch.is_tensor(out) else (out[0] if isinstance(out, (tuple,list)) else None)
            if t is None: return
            if not torch.isfinite(t).all():
                fin = torch.isfinite(t)
                msg = (f"[NAN-HIT] layer={name} type={module.__class__.__name__} "
                       f"shape={tuple(t.shape)} "
                       f"finite_ratio={(fin.float().mean().item() if fin.numel() else 1.0):.4f} "
                       f"min={torch.nan_to_num(t).min().item():.3e} "
                       f"max={torch.nan_to_num(t).max().item():.3e}")
                raise RuntimeError(msg)
        return _hook

    for n, m in model.named_modules():
        # Attach hooks only to risky layers: Linear/BatchNorm/ReLU/EdgeConvBlock.
        if isinstance(m, (torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.ReLU)) \
           or m.__class__.__name__.lower().startswith("edgeconv"):
            m.register_forward_hook(hook(n))

def fnum(x):
    return float(x) if isinstance(x, (float, int)) else float('nan')

def main():
    fix_seed()
    ensure_dirs()
    
    tr_dir = os.path.join(Cfg.DATA_ROOT, Cfg.TRAIN_DIR)
    va_dir = os.path.join(Cfg.DATA_ROOT, Cfg.VAL_DIR)
    te_dir = os.path.join(Cfg.DATA_ROOT, Cfg.TEST_DIR)
    print(Cfg.DEVICE)
    
    train_files = list_plys(tr_dir)
    val_files   = list_plys(va_dir)
    test_files  = list_plys(te_dir)
    
    if len(val_files) == 0 and len(train_files) > 0:
        random.shuffle(train_files)
        n_val = int(len(train_files) * Cfg.HOLDOUT_FRACTION)
        val_files = train_files[:n_val]
        train_files = train_files[n_val:]

    if os.path.exists(Cfg.STATS_CACHE):
        stats = json.load(open(Cfg.STATS_CACHE, "r"))
    else:
        stats = scan_train_stats(train_files)
        os.makedirs(os.path.dirname(Cfg.STATS_CACHE), exist_ok=True)
        json.dump(stats, open(Cfg.STATS_CACHE, "w"), indent=2)
    pos_ratio = stats["pos_ratio"]
    # pos_weight = (#neg / #pos)
    pw = torch.tensor([(1.0 - pos_ratio) / max(pos_ratio, 1e-6)], dtype=torch.float32, device=Cfg.DEVICE)
    
    # DataLoaders
    train_loader = make_loader(train_files, stats, use_aug=Cfg.USE_AUG, shuffle=True,
                               target_points=Cfg.BATCH_TARGET_POINTS, max_points=Cfg.MAX_POINTS_PER_PC)
    val_loader   = make_loader(val_files, stats, use_aug=False, shuffle=False,
                               target_points=Cfg.BATCH_TARGET_POINTS//2, max_points=None)
    
    # Model/opt
    model = DGCNNBinarySeg(k=Cfg.K).to(Cfg.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=Cfg.LR, weight_decay=Cfg.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Cfg.EPOCHS)
    
    best_ap = -1.0
    for ep in range(1, Cfg.EPOCHS+1):
        torch.cuda.reset_peak_memory_stats()
        tr_loss = train_epoch(model, train_loader, opt, pw)
        va_loss, va_metrics = eval_epoch(model, val_loader)
        sched.step()
        
        log = (f"[{ep:03d}] train {tr_loss:.4f} | val {va_loss:.4f} | "
            f"ACC {fnum(va_metrics.get('ACC')):.4f} "
            f"F1 {fnum(va_metrics.get('F1')):.4f} "
            f"AUROC {fnum(va_metrics.get('AUROC')):.4f} "
            f"AUPRC {fnum(va_metrics.get('AUPRC')):.4f} "
            f"MCC {fnum(va_metrics.get('MCC')):.4f}")
        print(log)
        
        cur_ap = va_metrics.get("AUPRC") or -1.0
        if cur_ap > best_ap:
            best_ap = cur_ap
            os.makedirs(Cfg.SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(Cfg.SAVE_DIR, "best.pt"))
            print("  saved:", os.path.join(Cfg.SAVE_DIR, "best.pt"))
            
    if len(test_files) > 0:
        test_loader = make_loader(test_files, stats, use_aug=False, shuffle=False,
                                  target_points=Cfg.BATCH_TARGET_POINTS//2, max_points=None)
        tloss, tmetrics = eval_epoch(model, test_loader)
        print(
            f"[TEST] loss {tloss:.4f} | "
            f"ACC {fnum(tmetrics.get('ACC')):.4f} | "
            f"F1 {fnum(tmetrics.get('F1')):.4f} | "
            f"AUROC {fnum(tmetrics.get('AUROC')):.4f} | "
            f"AUPRC {fnum(tmetrics.get('AUPRC')):.4f} | "
            f"MCC {fnum(tmetrics.get('MCC')):.4f}"
        )
        
        # ---- Save per-point predictions into .ply (property float predict) ----
        save_predictions_on_test(model, test_files, stats, out_dir=Cfg.PRED_PLY_DIR, suffix=Cfg.PRED_SUFFIX)
        print(f"[TEST] wrote predicted .ply files to: {Cfg.PRED_PLY_DIR}")

if __name__ == "__main__":
    main()
