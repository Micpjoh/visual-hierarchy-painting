#!/bin/bash
#SBATCH --job-name=thesis_fullrun
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/home/mjohn/thesis_models/full-run/logs/run_%j.out
#SBATCH --error=/home/mjohn/thesis_models/full-run/logs/run_%j.err

set -euo pipefail

python ~/thesis_models/full-run/full_run_orchestrator.py