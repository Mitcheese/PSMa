# PPI Binary Segmentation (DGCNN)

This repository trains and evaluates protein-protein interaction (PPI) binary segmentation on surface point clouds, with an optional pretrained global embedding.

## Repository layout
- `scripts/` executable entry points (train / infer / eval / preprocess / viz)
- `src/` shared utilities (path config helpers)
- `configs/` path configuration
- `data/` contains protein sequences

## Path configuration
Default paths are defined in `configs/paths.json`. You can override with environment variables:
- `PPI_DATA_ROOT` (dataset root)
- `PPI_EMB_DIR` (pretrained embedding dir)
- `PPI_OUTPUTS_DIR` (outputs root)
- `PPI_VERTEX_RESIDUE_DIR` (vertex-residue mapping dir)

## Data expectations
`${PPI_DATA_ROOT}/` should contain (external, not committed):
- `labeled_ply_train/`
- `labeled_ply_test/`
- (optional) `ply_temp/`, `ply_temp_right_residue/`

If using pretrained embeddings, place `*.npy` in `${PPI_EMB_DIR}/` (external, not committed).

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
  --ckpt ${PPI_OUTPUTS_DIR}/runs/binary_dgcnn_pretrain/<dataset>/<run>/best.pt \
  --input ${PPI_DATA_ROOT}/labeled_ply_test \
  --output ${PPI_OUTPUTS_DIR}/infer_pretrain \
  --stats ${PPI_OUTPUTS_DIR}/runs/binary_dgcnn_pretrain/<dataset>/stats.json \
  --pre-emb-dir ${PPI_EMB_DIR} \
  --pre-dim 1280 \
  --pre-fusion concat
```

Base version:
```
python scripts/infer_base.py \
  --ckpt ${PPI_OUTPUTS_DIR}/runs/binary_dgcnn/<dataset>/<run>/best.pt \
  --input ${PPI_DATA_ROOT}/labeled_ply_test \
  --output ${PPI_OUTPUTS_DIR}/infer_base \
  --stats ${PPI_OUTPUTS_DIR}/runs/binary_dgcnn/<dataset>/stats.json
```

## Evaluation
Point-level evaluation:
```
python scripts/eval/eval_ply.py ${PPI_OUTPUTS_DIR}/infer_pretrain ${PPI_OUTPUTS_DIR}/results_eval --plot
```

Residue-level evaluation:
```
python scripts/eval/ply_residue_stats.py ${PPI_OUTPUTS_DIR}/infer_pretrain ${PPI_OUTPUTS_DIR}/residue_csvs
python scripts/eval/eval_residue.py ${PPI_OUTPUTS_DIR}/residue_csvs ${PPI_OUTPUTS_DIR}/residue_eval --plot
```

## Notes
- Outputs are written under `${PPI_OUTPUTS_DIR}/` by default (external, not committed).
