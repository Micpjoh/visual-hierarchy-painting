#!/bin/bash
#SBATCH --job-name=autorun
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/mjohn/thesis_models/auto-run/outputs/logs/autorun_%j.out
#SBATCH --error=/home/mjohn/thesis_models/auto-run/outputs/logs/autorun_%j.err

# Auto-run pipeline: YOLOv7 + SAM3 + DIS + combo_saliency_heavy on ArtDL 2.0.
#
# Stages run inside the orchestrator, which activates the correct virtualenv
# for each model and the shared one for the CPU evaluation stage. Output:
#   ~/thesis_models/auto-run/outputs/results.json   (aggregate metrics)
#   ~/thesis_models/auto-run/outputs/rankings.json  (per-painting rankings + matches)
#   ~/thesis_models/auto-run/outputs/features.json  (raw feature values)
#
# Walltime budget rationale (823 paintings, test_saints.json):
#   - YOLO  ~3 min   (823 images at 640x640)
#   - SAM3  ~1-1.5h  (823 images x ~8 detections each, the bottleneck)
#   - DIS   ~10 min
#   - eval  ~5 min CPU
#   Total ~1.5h; request 2h as a safety margin.

set -euo pipefail

export THESIS_ROOT="/home/mjohn/thesis_models"
mkdir -p "${THESIS_ROOT}/auto-run/outputs/logs"

# Use the orchestrator's "yolo" venv as a base for Python and the orchestrator
# itself - the orchestrator handles per-stage venv switching internally
python "${THESIS_ROOT}/auto-run/auto_run_orchestrator.py"
