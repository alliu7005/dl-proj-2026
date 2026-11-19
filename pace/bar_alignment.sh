#!/bin/bash
# Submit the bar-alignment experiments (run on the login node from the repo root):
#   bash pace/bar_alignment.sh                  # Blackwell GPUs (default)
#   GPU=h100 bash pace/bar_alignment.sh         # another GPU type
#
# The realistic rhythm scores trail the oracle ones (onset F1 ~0.2 vs ~0.56) mostly because the bar
# phase is wrong: onset-shift F1 is twice onset F1. Three fixes, tested on Score2Seq (the best
# model) one at a time and together; every evaluation also reports the guarded and meter modes.
#
# A CPU job first precomputes the metrical labels (scripts/build_metrical.py) and runs the new
# tests on this data build; the training jobs only start if it passes.
#
#   run        what it adds to s2s                                            compare with
#   s2s_c      64 context notes before / 32 after each window                 s2s
#   s2s_a      auxiliary metrical head (downbeat + beat phase per note)       s2s
#   s2s_ca     both                                                           s2s_c, s2s_a
#   s2s_m_ca   both, at 25.9M parameters with the regularization that fixed   s2s_ca
#              the 45M model's overfitting (dropout 0.2, wd 0.1, lr 2e-4)
#   sf_base_reg_ca  both, on the regularized 45M Scoreformer: the best model    sf_base_reg, s2s_ca
#              given the true context, which falls behind without it (bar F1
#              0.42 vs 0.52 with no context); these fixes target that gap
set -euo pipefail
GPU=${GPU:-rtx_pro_6000_blackwell}
G="--gres=gpu:${GPU}:1"

prep=$(sbatch --parsable -J barprep -c 8 --mem=32G -t 1:00:00 -o slurm-barprep-%j.out --wrap="source pace/common.sh; \
python scripts/build_metrical.py --workers 8 && python -m tests.test_metrical && python -m tests.test_context_notes \
&& python -m tests.test_decoding")
echo "labels + tests: job $prep"

# NAME / ARGS travel as environment variables (--export=ALL): sbatch's --export=ALL,NAME=..,ARGS=..
# form splits on commas, which would break arguments like --modes guarded,meter.
submit() {  # name, args
  local id
  id=$(NAME="$1" ARGS="$2" sbatch --parsable $G --dependency=afterok:$prep --export=ALL pace/train.sbatch)
  echo "  $1: job $id  ($2)"
}
CTX="--context-before 64 --context-after 32"
submit s2s_c    "--arch score2seq $CTX"
submit s2s_a    "--arch score2seq --aux-metrical"
submit s2s_ca   "--arch score2seq $CTX --aux-metrical"
submit s2s_m_ca "--arch score2seq --d-model 384 --enc-layers 6 --dec-layers 6 --dropout 0.2 --weight-decay 0.1 --lr 2e-4 $CTX --aux-metrical"
submit sf_base_reg_ca "--size base --dropout 0.2 --weight-decay 0.1 --lr 2e-4 $CTX --aux-metrical"
echo "watch: squeue -u \$USER     logs: slurm-barprep-<job>.out, slurm-scoreformer-<job>.out"
