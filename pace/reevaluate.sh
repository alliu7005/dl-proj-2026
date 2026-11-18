#!/bin/bash
# Re-evaluate finished runs with the current decoding (run on the login node from the repo root):
#   bash pace/reevaluate.sh                          # the six post-milestone comparison runs
#   bash pace/reevaluate.sh s2s sf_base_reg          # specific runs
#   GPU=h100 bash pace/reevaluate.sh                 # another GPU type
#
# Why: decoding now caps each window's output (src/decoding.py max_new_tokens) and adds the guarded
# mode, so oracle / chained / none / guarded are re-run together and every number in a comparison
# comes from the same code. One GPU job per run (pace/evaluate.sbatch), ~1-2 h each.
set -euo pipefail
GPU=${GPU:-rtx_pro_6000_blackwell}
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(s2s s2s_ctx sf_small sf_small_ctx sf_base sf_base_reg)
for r in "${RUNS[@]}"; do
  if [ ! -f "outputs/runs/$r/best.pt" ]; then
    echo "  skip $r: outputs/runs/$r/best.pt not found"
    continue
  fi
  # two evaluations of one run would both rewrite its result files: never queue a second one
  busy=$(squeue -h -u "$USER" -o "%i %j" | awk -v a="eval_$r" -v b="guard_$r" '$2 == a || $2 == b {print $1}')
  if [ -n "$busy" ]; then
    echo "  skip $r: already being evaluated (job $busy)"
    continue
  fi
  id=$(sbatch --parsable --gres=gpu:${GPU}:1 -J "eval_$r" --export=ALL,NAME="$r" pace/evaluate.sbatch)
  echo "  $r: job $id"
done
echo "watch: squeue -u \$USER     logs: slurm-eval_<run>-<job>.out"
