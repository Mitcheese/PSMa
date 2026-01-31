# PSMa

Monorepo with two components:

- `pretrain/`: pretraining + finetuning pipeline (Point-MAE variants)
- `downstream/`: downstream tasks (PPI)

## Quick navigation

- Pretrain docs: `pretrain/README.md`
- Downstream docs: `downstream/README.md`

## Repo hygiene
- Large datasets are excluded from this repo. Use environment variables or external paths.
- Build artifacts and outputs are ignored by `.gitignore`.
