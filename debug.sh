#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --partition=debug
#SBATCH --qos=debug-qos
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH --time=00:02:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

uv run cs336_alignment/train_grpo.py 1