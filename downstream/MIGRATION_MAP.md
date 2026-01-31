# Migration Map

This file records the main path changes after the cleanup.

## Core scripts
- train_protein_binary_seg_pretrain.py -> scripts/train_pretrain.py
- train_protein_binary_seg_0.py -> scripts/train_base.py
- train_protein_binary_seg_4.py -> legacy/train_protein_binary_seg_ddp.py
- ply_save_predict_pretrain.py -> scripts/infer_pretrain.py
- ply_save_predict.py -> scripts/infer_base.py
- res_index2names.py -> scripts/tools/res_index2names.py
- runs/remap_ckpt_keys.py -> scripts/tools/remap_ckpt_keys.py

## Preprocess
- preprocess/*.py -> scripts/preprocess/*.py
- preprocess/zzadd_residue.py -> legacy/zzadd_residue.py
- preprocess/ (non-code artifacts) -> data/interim/preprocess/
- data/DELPHI/check_struct_confidence.py -> scripts/preprocess/check_struct_confidence.py

## Eval
- results/eval_ply.py -> scripts/eval/eval_ply.py
- results/eval_residue.py -> scripts/eval/eval_residue.py
- results/eval_residue_csvs.py -> scripts/eval/eval_residue_csvs.py
- results/ply_residue_stats.py -> scripts/eval/ply_residue_stats.py

## Visualization
- visualize/*.py -> scripts/viz/*.py
- visualize/ (outputs) -> outputs/visualize/

## Outputs
- runs/ -> outputs/runs/
- results/ (outputs only) -> outputs/results/
- debug/ -> outputs/debug/
- *.out/*.err -> outputs/logs/

## Data
- embedding/ -> data/embeddings/
- vertex_residue/ -> data/vertex_residue/

## SLURM jobs
- *.slurm -> scripts/jobs/*.slurm
- data/DELPHI/_CPU.slurm -> scripts/jobs/DELPHI_CPU.slurm
- 1a40_pretrain.slurm -> legacy/jobs/1a40_pretrain.slurm
