# PPI Binary Segmentation (DGCNN)

This repository trains and evaluates protein-protein interaction (PPI) binary segmentation on surface point clouds, with an optional pretrained global embedding.

## Repository layout
- `scripts/` executable entry points (train / infer / eval / preprocess / viz)
- `src/` shared utilities (path config helpers)
- `configs/` path configuration
- `data/` contains protein sequences

## Path configuration
Default paths are defined in `configs/paths.json`. You can override with environment variables:
- `PPI_DATA_ROOT` (default: `data/DeepPPIPS`)
- `PPI_EMB_DIR` (default: `data/embeddings`)
- `PPI_OUTPUTS_DIR` (default: `outputs`)
- `PPI_VERTEX_RESIDUE_DIR` (default: `data/vertex_residue`)

## Data expectations
`data/DeepPPIPS/` should contain (external, not committed):
- `labeled_ply_train/`
- `labeled_ply_test/`
- (optional) `ply_temp/`, `ply_temp_right_residue/`

If using pretrained embeddings, place `*.npy` in `data/embeddings/` (external, not committed).

## Training
Base (no pretrain):
```
python scripts/train_base.py
```

With pretrained global embedding:
```
python scripts/train_pretrain.py
```

## Inference
Pretrained version:
```
python scripts/infer_pretrain.py \
  --ckpt outputs/runs/binary_dgcnn_pretrain/DeepPPIPS/<run>/best.pt \
  --input data/DeepPPIPS/labeled_ply_test \
  --output outputs/infer_pretrain \
  --stats outputs/runs/binary_dgcnn_pretrain/DeepPPIPS/stats.json \
  --pre-emb-dir data/embeddings \
  --pre-dim 1280 \
  --pre-fusion concat
```

Base version:
```
python scripts/infer_base.py \
  --ckpt outputs/runs/binary_dgcnn/DeepPPIPS/<run>/best.pt \
  --input data/DeepPPIPS/labeled_ply_test \
  --output outputs/infer_base \
  --stats outputs/runs/binary_dgcnn/DeepPPIPS/stats.json
```

## Evaluation
Point-level evaluation:
```
python scripts/eval/eval_ply.py outputs/infer_pretrain outputs/results_eval --plot
```

Residue-level evaluation:
```
python scripts/eval/ply_residue_stats.py outputs/infer_pretrain outputs/residue_csvs
python scripts/eval/eval_residue.py outputs/residue_csvs outputs/residue_eval --plot
```

## Notes
- Outputs are written under `outputs/` by default (external, not committed).
