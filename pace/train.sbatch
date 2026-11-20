#!/bin/bash
# Train + evaluate one Scoreformer run on a PACE GPU. Submit from the repo root:
#   sbatch pace/train.sbatch                                              # sf_base, defaults
#   NAME=sf_large ARGS="--size large --lr 2e-4" sbatch --export=ALL pace/train.sbatch
#   NAME=sf_clean ARGS="--no-noise" sbatch --export=ALL pace/train.sbatch
#   NAME=s2s ARGS="--arch score2seq" sbatch --export=ALL pace/train.sbatch  # Michael's model, same loop
# (NAME / ARGS as environment variables: --export=ALL,ARGS=... would split ARGS at its commas)
# Resubmitting the same NAME resumes from outputs/runs/NAME/last.pt (e.g. after the time limit).
# PACE ICE limits: 8 h walltime and 16 GPU hours per job.
#SBATCH -J scoreformer
# GPU: any of H100 / H200 / RTX Pro Blackwell, whichever frees up first (node features; see
# `sinfo -h -o "%G | %f" | sort -u`). Not V100 / RTX 6000 (no bf16), not MI210 / R9700 (AMD:
# the CUDA build of torch can't use them); A100 / L40S work but at about half the speed. Override:
#   sbatch --constraint=gpu-a100 ...   (one type)   or   --constraint="gpu-h100|gpu-a100" ...
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-h100|gpu-h200|gpu-rtxpro-blackwell
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
source pace/common.sh
NAME=${NAME:-sf_base}
ARGS=${ARGS:-}
echo "run: $NAME  args: $ARGS"  # which run this job trains (squeue shows every training as scoreformer)
nvidia-smi --query-gpu=name,memory.total --format=csv
RESUME=""
[ -f "outputs/runs/$NAME/last.pt" ] && RESUME="--resume"
python scripts/train_model.py --name "$NAME" --normal-priority --workers 7 $RESUME $ARGS
python scripts/evaluate_pieces.py --run "$NAME" --normal-priority --musicxml 5
