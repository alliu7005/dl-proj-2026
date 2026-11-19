#!/bin/bash
# Re-evaluate finished runs with the current decoding (run on the login node from the repo root):
#   bash pace/reevaluate.sh                          # every finished run (smoke tests excluded)
#   bash pace/reevaluate.sh s2s sf_base_reg          # specific runs
#   GPU=h100 bash pace/reevaluate.sh                 # another GPU type
#   ARGS="--modes meter" bash pace/reevaluate.sh s2s # extra evaluate_pieces.py arguments, e.g. add
#                                                    # one mode (merged into the existing results)
#
# Why: decoding now caps each window's output (src/decoding.py max_new_tokens) and adds the guarded
# and meter modes, so oracle / chained / none / guarded / meter are re-run together and every number in a comparison
# comes from the same code. One GPU job per run (pace/evaluate.sbatch), ~1-2 h each.
set -euo pipefail
GPU=${GPU:-rtx_pro_6000_blackwell}
ARGS=${ARGS:-}
RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then  # every run with a best.pt, except smoke tests
  for f in outputs/runs/*/best.pt; do
    [ -f "$f" ] || continue
    r=$(basename "$(dirname "$f")")
    case "$r" in smoke*) ;; *) RUNS+=("$r") ;; esac
  done
fi
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
  id=$(sbatch --parsable --gres=gpu:${GPU}:1 -J "eval_$r" --export=ALL,NAME="$r",ARGS="$ARGS" pace/evaluate.sbatch)
  echo "  $r: job $id${ARGS:+  ($ARGS)}"
done
echo "watch: squeue -u \$USER     logs: slurm-eval_<run>-<job>.out"
