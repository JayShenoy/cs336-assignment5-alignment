#!/bin/bash
#SBATCH --job-name=run_py
#SBATCH --partition=a5-batch
#SBATCH --qos=a5-batch-qos
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

PYTHON_SCRIPT="$1"

uv run ${PYTHON_SCRIPT}