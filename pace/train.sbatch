#!/bin/bash
# Train + evaluate one Scoreformer run on a PACE GPU. Submit from the repo root:
#   sbatch pace/train.sbatch                                              # sf_base, defaults
#   sbatch --export=ALL,NAME=sf_large,ARGS="--size large --lr 2e-4" pace/train.sbatch
#   sbatch --export=ALL,NAME=sf_clean,ARGS="--no-noise" pace/train.sbatch
#   sbatch --export=ALL,NAME=s2s,ARGS="--arch score2seq" pace/train.sbatch  # Michael's model, same loop
# Resubmitting the same NAME resumes from outputs/runs/NAME/last.pt (e.g. after the time limit).
# PACE ICE limits: 8 h walltime and 16 GPU hours per job. Any GPU works (bf16 needs A100/H100/L40S
# or newer for full speed); to ask for one type, see `sinfo -o "%P %G"` and e.g. --gres=gpu:h100:1 (lowercase; also h200, rtx_pro_6000_blackwell, a100). Avoid v100 (no bf16) and mi210/r9700 (AMD).
#SBATCH -J scoreformer
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
source pace/common.sh
NAME=${NAME:-sf_base}
ARGS=${ARGS:-}
nvidia-smi --query-gpu=name,memory.total --format=csv
RESUME=""
[ -f "outputs/runs/$NAME/last.pt" ] && RESUME="--resume"
python scripts/train_model.py --name "$NAME" --normal-priority --workers 7 $RESUME $ARGS
python scripts/evaluate_pieces.py --run "$NAME" --normal-priority --musicxml 5
