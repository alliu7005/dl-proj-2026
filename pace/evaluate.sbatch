#!/bin/bash
# Evaluate an existing run on a PACE GPU (no training). Submit from the repo root:
#   NAME=s2s sbatch --export=ALL pace/evaluate.sbatch                                # all modes, all test sets
#   NAME=s2s ARGS="--modes guarded,meter" sbatch --export=ALL pace/evaluate.sbatch    # some modes, merged
#   NAME=sf_base_reg sbatch --constraint=gpu-h100 --export=ALL pace/evaluate.sbatch   # one GPU type
# (NAME / ARGS as environment variables: --export=ALL,ARGS=... would split ARGS at its commas)
# Results are merged into outputs/runs/NAME/eval_pieces_best.json (modes not re-run are kept).
# 4 h covers a full evaluation of the 45M model (~2 h); asking for less walltime matters on PACE,
# where the reserved GPU time of running jobs is capped per user (MaxGRESRunMinsPerUser).
# Never run two evaluations of the same NAME at once: both update the same result files.
#SBATCH -J evaluate
# GPU: any of H100 / H200 / RTX Pro Blackwell, whichever frees up first (node features; see
# `sinfo -h -o "%G | %f" | sort -u`). Not V100 / RTX 6000 (no bf16), not MI210 / R9700 (AMD:
# the CUDA build of torch can't use them); A100 / L40S work but at about half the speed. Override:
#   sbatch --constraint=gpu-a100 ...   (one node feature: PACE adds &cuda13, and Slurm rejects an OR list with it)
#SBATCH --gres=gpu:1
#SBATCH --constraint=HX00
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
source pace/common.sh
: "${NAME:?set NAME, e.g. NAME=s2s sbatch --export=ALL pace/evaluate.sbatch}"
ARGS=${ARGS:-}
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/evaluate_pieces.py --run "$NAME" --normal-priority --musicxml 5 $ARGS
