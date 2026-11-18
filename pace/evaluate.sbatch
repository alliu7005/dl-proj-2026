#!/bin/bash
# Evaluate an existing run on a PACE GPU (no training). Submit from the repo root:
#   sbatch --export=ALL,NAME=s2s pace/evaluate.sbatch                         # all modes, all test sets
#   sbatch --export=ALL,NAME=s2s,ARGS="--modes guarded" pace/evaluate.sbatch  # add one mode
#   sbatch --gres=gpu:h100:1 --export=ALL,NAME=sf_base_reg pace/evaluate.sbatch
# Results are merged into outputs/runs/NAME/eval_pieces_best.json (modes not re-run are kept).
# 4 h covers a full evaluation of the 45M model (~2 h); asking for less walltime matters on PACE,
# where the reserved GPU time of running jobs is capped per user (MaxGRESRunMinsPerUser).
# Never run two evaluations of the same NAME at once: both update the same result files.
#SBATCH -J evaluate
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
source pace/common.sh
: "${NAME:?set NAME, e.g. sbatch --export=ALL,NAME=s2s pace/evaluate.sbatch}"
ARGS=${ARGS:-}
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/evaluate_pieces.py --run "$NAME" --normal-priority --musicxml 5 $ARGS
