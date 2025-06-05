#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --partition=debug
#SBATCH --qos=debug-qos
#SBATCH -c 4
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --mem=50G
#SBATCH --time=00:02:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

PYTHON_SCRIPT="$1"

uv run ${PYTHON_SCRIPT}