# Index

## Primary entry points
- `src/psma_pretrain/cli.py`: pretrain/finetune/test dispatcher (main CLI)
- `src/psma_pretrain/vis.py`: visualization/test wrapper (test-only)
- `src/psma_pretrain/segmentation/main.py`: segmentation training pipeline

## Core modules
- `src/psma_pretrain/tools/`
  - `runner_pretrain.py`: pretraining loop
  - `runner_finetune.py`: finetune/regression loop
  - `runner_vis.py`: visualization/testing
  - `builder.py`: dataset/model/optimizer/scheduler builder
- `src/psma_pretrain/models/`
  - `point_mae.py`: Point-MAE model
  - `point_mae_2.py`: alternative model variant
- `src/psma_pretrain/datasets/`
  - `embedding_dataset.py`: protein embedding dataset
  - `data_process.py`: PLY to embedding generation
  - `generate_few_shot.py`: ModelNet few-shot generator
- `src/psma_pretrain/utils/`: logging, config, DDP helpers, misc utilities

## Configs
- `configs/pretrain_base.yaml`
- `configs/finetune_ogt_base.yaml`
- `configs/dataset_configs/*.yaml` (uses `${PSMA_DATA_ROOT}`)

## Extensions
- `extensions/chamfer_dist`
- `extensions/emd`

## Outputs
- `outputs/experiments/` (checkpoints/logs/tensorboard, runtime-created)
- `outputs/vis/` (visualizations, runtime-created)
