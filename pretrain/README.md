# PSMa_pretrain

Pretraining and finetuning pipeline for protein surface point-embedding models (Point-MAE variants), plus a separate point-cloud segmentation pipeline.

## Quick start

### 1) Install

```bash
python -m pip install -r requirements.txt
```

### 2) Set data root

```bash
export PSMA_DATA_ROOT=/inspurfs/group/gaoshh/chenqy/pro_rna
```

### 2.5) (Optional) Download pretrain dataset

You can use `scripts/mirror_pdb_text.sh` to download the pretraining dataset.
Note: the pretraining dataset is very large, so make sure you have ample disk space and quota before running.

```bash
chmod +x scripts/mirror_pdb_text.sh
./scripts/mirror_pdb_text.sh /path/to/large_disk/pdb_text_archive
```

### 3) Pretrain

```bash
python -m psma_pretrain.cli --config configs/pretrain_base.yaml --exp_name run_pretrain
```

### 4) Finetune

```bash
python -m psma_pretrain.cli --config configs/finetune_ogt_base.yaml --exp_name run_finetune --finetune_model --ckpts /path/to/pretrained.pth
```

### 5) Segmentation (separate pipeline)

```bash
python -m psma_pretrain.segmentation.main --root ${PSMA_DATA_ROOT}/shapenetcore_partanno_segmentation_benchmark_v0_normal
```

## Notes
- CUDA extensions live in `extensions/` and may require local build for Chamfer/EMD.
- Outputs are written under `outputs/experiments/` (created at runtime).
- Configs live in `configs/` and support `${PSMA_DATA_ROOT}` expansion.

## Repo layout
See `docs/INDEX.md` for module-level entry points and dependencies.
