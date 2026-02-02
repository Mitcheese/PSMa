#!/bin/bash
#SBATCH -J pro_rna
#SBATCH -p critical
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=all
#SBATCH -N 1
#SBATCH --gres=gpu:NVIDIAA40:2
#SBATCH --mem=200000
#SBATCH --time=120:00:00
#SBATCH -A gaoshh-critical
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source activate pro_rna

export PSMA_DATA_ROOT=/path/to/psma_data_root

python -m psma_pretrain.cli --config configs/pretrain_base.yaml --exp_name instant_embed_60_ratio_base
