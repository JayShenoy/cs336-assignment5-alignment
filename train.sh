#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --partition=a5-batch
#SBATCH --qos=a5-batch-qos
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

CONFIG_PATH="$1"

uv run cs336_alignment/train_grpo.py ${CONFIG_PATH}