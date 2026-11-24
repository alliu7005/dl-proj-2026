#!/bin/bash
# Train + evaluate one note labeller (v2) on a PACE GPU. Submit from the repo root:
#   NAME=nl_base ARGS="--size base" sbatch --export=ALL pace/train_labeller.sbatch
# (NAME / ARGS as environment variables: --export=ALL,ARGS=... would split ARGS at its commas)
# Resubmitting the same NAME resumes from outputs/runs/NAME/last.pt (e.g. after the time limit);
# a finished run only evaluates (scripts/eval_labeller.py writes every test set it can).
#SBATCH -J labeller
#SBATCH --gres=gpu:1
#SBATCH --constraint=HX00
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
source pace/common.sh
NAME=${NAME:?set NAME, e.g. NAME=nl_base sbatch --export=ALL pace/train_labeller.sbatch}
ARGS=${ARGS:-}
echo "run: $NAME  args: $ARGS"
nvidia-smi --query-gpu=name,memory.total --format=csv
RESUME=""
[ -f "outputs/runs/$NAME/last.pt" ] && RESUME="--resume"
# shellcheck disable=SC2086
python scripts/train_labeller.py --name "$NAME" --normal-priority --workers 7 $RESUME $ARGS
python scripts/eval_labeller.py --run "$NAME" --normal-priority
