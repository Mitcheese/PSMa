# PSMa Index

## Top-level entry points
- Pretrain overview: `pretrain/README.md`
- Downstream overview: `downstream/README.md`

## Pretrain (keep/usable)
- `pretrain/src/psma_pretrain/cli.py`: main CLI for pretrain/finetune/test.
  - Example: `cd pretrain && python -m psma_pretrain.cli --config configs/pretrain_base.yaml --exp_name run_pretrain`
- `pretrain/src/psma_pretrain/vis.py`: visualization/test helper (test-only).
  - Example: `cd pretrain && python -m psma_pretrain.vis --config configs/pretrain_base.yaml --test --ckpts /path/to/ckpt.pth`
- `pretrain/src/psma_pretrain/segmentation/main.py`: separate segmentation pipeline.
  - Example: `cd pretrain && python -m psma_pretrain.segmentation.main`
- `pretrain/tests/smoke_test.py`: minimal model-build smoke test.
  - Example: `cd pretrain && python tests/smoke_test.py --config configs/pretrain_base.yaml`

## Downstream (keep/usable)
- `downstream/scripts/train_base.py`: train baseline binary segmentation.
  - Example: `cd downstream && python scripts/train_base.py`
- `downstream/scripts/train_pretrain.py`: train with pretrained global embedding.
  - Example: `cd downstream && python scripts/train_pretrain.py`
- `downstream/scripts/infer_base.py`: inference for baseline model.
  - Example: `cd downstream && python scripts/infer_base.py --ckpt /path/to/best.pt --input /path/to/ply --output /path/to/out --stats /path/to/stats.json`
- `downstream/scripts/infer_pretrain.py`: inference for pretrained-embedding model.
  - Example: `cd downstream && python scripts/infer_pretrain.py --ckpt /path/to/best.pt --input /path/to/ply --output /path/to/out --stats /path/to/stats.json`
- `downstream/scripts/eval/eval_ply.py`: point-level evaluation from predicted PLY.
  - Example: `cd downstream && python scripts/eval/eval_ply.py /path/to/pred_ply /path/to/eval_out --plot`
- `downstream/scripts/eval/eval_residue.py`: residue-level evaluation from CSVs.
  - Example: `cd downstream && python scripts/eval/eval_residue.py /path/to/csvs /path/to/eval_out`
- `downstream/scripts/eval/eval_residue_csvs.py`: per-file residue CSV summary.
  - Example: `cd downstream && python scripts/eval/eval_residue_csvs.py --input-dir /path/to/csvs --output-dir /path/to/eval_out`
- `downstream/scripts/preprocess/preprocess.py`: preprocessing pipeline.
  - Example: `cd downstream && python scripts/preprocess/preprocess.py`
- `downstream/scripts/preprocess/ply_add_label.py`: add labels to PLYs (data preparation).
  - Example: `cd downstream && python scripts/preprocess/ply_add_label.py`
- `downstream/scripts/preprocess/ply_check_labels.py`: label sanity checks.
  - Example: `cd downstream && python scripts/preprocess/ply_check_labels.py`
- `downstream/scripts/preprocess/ply_check_missing.py`: check missing IDs/files.
  - Example: `cd downstream && python scripts/preprocess/ply_check_missing.py`
- `downstream/scripts/preprocess/pdbply_process.py`: build surface PLYs from structures.
  - Example: `cd downstream && python scripts/preprocess/pdbply_process.py`
- `downstream/scripts/preprocess/pdb_dl.py`: download PDBs for preprocessing.
  - Example: `cd downstream && python scripts/preprocess/pdb_dl.py`
- `downstream/scripts/preprocess/check_struct_confidence.py`: filter structures by confidence.
  - Example: `cd downstream && python scripts/preprocess/check_struct_confidence.py`

## Notes
- Large datasets are external and not tracked in this repo.
- Outputs are written to `outputs/` paths defined in configs or environment variables.
