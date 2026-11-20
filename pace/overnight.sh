#!/bin/bash
# Submit the post-milestone experiments (run on the login node from the repo root):
#   bash pace/overnight.sh                  # any of H100 / H200 / RTX Pro Blackwell (default)
#   GPUS=gpu-h100 bash pace/overnight.sh    # other GPU types (node features, "a|b" for several)
#
# A short CPU job first checks the context-noise code on this data build; the four training jobs
# (each: train, then evaluate_pieces.py on all test sets) only start if it passes.
#
#   run            what it tests                                    compare with
#   sf_small       time-based encoder at equal size (9.7M vs 10.2M)  s2s
#   s2s_ctx        training with corrupted context (--context-noise)  s2s
#   sf_small_ctx   both                                              sf_small, s2s_ctx
#   sf_base_reg    45M model with more regularization, shorter run   sf_base
set -euo pipefail
GPUS=${GPUS:-gpu-h100|gpu-h200|gpu-rtxpro-blackwell}  # node features: whichever of these GPUs frees up first
G=(--gres=gpu:1 "--constraint=$GPUS")

test_id=$(sbatch --parsable -J ctxtest -c 4 --mem=16G -t 0:30:00 -o slurm-ctxtest-%j.out \
  --wrap="source pace/common.sh; python -m tests.test_context_noise")
echo "context-noise check: job $test_id"

# NAME / ARGS travel as environment variables (--export=ALL): sbatch's --export=ALL,NAME=..,ARGS=..
# form splits on commas, which would break arguments like --modes guarded,meter.
submit() {  # name, args
  local id
  id=$(NAME="$1" ARGS="$2" sbatch --parsable "${G[@]}" --dependency=afterok:$test_id --export=ALL pace/train.sbatch)
  echo "  $1: job $id  ($2)"
}
submit sf_small     "--size small --dec-layers 6"
submit s2s_ctx      "--arch score2seq --context-noise 0.5"
submit sf_small_ctx "--size small --dec-layers 6 --context-noise 0.5"
submit sf_base_reg  "--size base --dropout 0.2 --weight-decay 0.1 --lr 2e-4 --epochs 6"
echo "watch: squeue -u \$USER     logs: slurm-scoreformer-<job>.out"
