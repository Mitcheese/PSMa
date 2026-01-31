# Index

## Entry points
- Training (base): `scripts/train_base.py`
- Training (pretrain): `scripts/train_pretrain.py`
- Inference (base): `scripts/infer_base.py`
- Inference (pretrain): `scripts/infer_pretrain.py`
- Evaluation:
  - `scripts/eval/eval_ply.py`
  - `scripts/eval/ply_residue_stats.py`
  - `scripts/eval/eval_residue.py`
  - `scripts/eval/eval_residue_csvs.py`
- Preprocessing:
  - `scripts/preprocess/preprocess.py`
  - `scripts/preprocess/ply_add_label.py`
  - `scripts/preprocess/ply_check_labels.py`
  - `scripts/preprocess/ply_check_missing.py`
  - `scripts/preprocess/pdbply_process.py`
  - `scripts/preprocess/pdb_dl.py`
- `scripts/preprocess/check_struct_confidence.py`

## Visualization
- `scripts/viz/*.py`

## Tools
- `scripts/tools/remap_ckpt_keys.py`
- `scripts/tools/res_index2names.py`

## Core library
- `src/ppi/paths.py`: path resolution from config/env

## Data layout
- `data/DeepPPIPS/`: dataset PLYs
- `data/embeddings/`: pretrained global embeddings
- `data/vertex_residue/`: residue mapping JSONs
- `data/interim/preprocess/`: preprocessing intermediates

## Outputs
- `outputs/runs/`: training runs
- `outputs/results/`: evaluation outputs
- `outputs/visualize/`: figures and colored PLYs
- `outputs/debug/`: debug dumps
- `outputs/logs/`: job logs
