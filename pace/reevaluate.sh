#!/bin/bash
# Evaluate finished runs: submit one GPU job per run for the results it does not have yet
# (run on the login node from the repo root):
#   bash pace/reevaluate.sh                          # every finished run (smoke tests excluded)
#   bash pace/reevaluate.sh s2s sf_base_reg          # specific runs
#   ARGS="--conditions real" bash pace/reevaluate.sh # extra evaluate_pieces.py arguments: which
#                                                    # conditions / modes / margins are wanted
#   ARGS="--extra-margins -0.05,0.05" bash pace/reevaluate.sh s2s
#   FORCE=1 ARGS=... bash pace/reevaluate.sh         # recompute even results that exist
#   GPUS=gpu-h100 bash pace/reevaluate.sh            # other GPU types (default: any of H100 /
#                                                    # H200 / RTX Pro Blackwell; "a|b" for several)
#   TIME=8:00:00 ARGS=... bash pace/reevaluate.sh    # walltime (default: 2.5 h for models under 15M
#                                                    # parameters, else 4 h: what one evaluation needs,
#                                                    # since PACE caps a user's running walltime x GPUs)
#   AFTER=after:123:124 bash pace/reevaluate.sh      # a Slurm dependency for every job (e.g. start only
#                                                    # once these training jobs have started)
#
# For each run, `python3 -m src.eval.plan` (standard library only) lists what is missing from
# outputs/runs/<run>/eval_pieces_best.json for the requested conditions and modes: a result counts
# only if it was computed with the same guard margin, on all performances, by the current
# evaluation version (src/eval/plan.py EVAL_VERSION, bumped when decoding changes). A run is
# skipped when nothing is missing, when its training has not finished (status.json, written by
# scripts/train_model.py; a training job's best.pt exists long before the run is done), or when an
# evaluation of it is already queued. Submitted jobs get --only-missing, so they compute exactly
# what was listed and merge it into the run's results.
set -euo pipefail
GPUS=${GPUS:-gpu-h100|gpu-h200|gpu-rtxpro-blackwell}  # node features: whichever of these GPUs frees up first
G=(--gres=gpu:1 "--constraint=$GPUS")
ARGS=${ARGS:-}
TIME=${TIME:-}
FORCE=${FORCE:-0}
AFTER=${AFTER:-}
DEP=()
[ -n "$AFTER" ] && DEP=("--dependency=$AFTER")
RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then  # every run with a best.pt, except smoke tests
  for f in outputs/runs/*/best.pt; do
    [ -f "$f" ] || continue
    r=$(basename "$(dirname "$f")")
    case "$r" in smoke*) ;; *) RUNS+=("$r") ;; esac
  done
fi
submitted=0
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
  # what is missing (exit 0), nothing (3), training not finished (4); $ARGS unquoted on purpose:
  # it holds several arguments, exactly as evaluate.sbatch passes it
  set +e
  # shellcheck disable=SC2086
  todo=$(python3 -m src.eval.plan --run "$r" $ARGS 2>&1)
  rc=$?
  set -e
  case "$rc" in
    0) ;;
    3) if [ "$FORCE" != 1 ]; then echo "  skip $r: nothing missing"; continue; fi ;;
    4) echo "  skip $r: $todo"; continue ;;  # even with FORCE: its best.pt is not final yet
    *) echo "  error for $r (src.eval.plan exit $rc):"; echo "$todo" | sed 's/^/    /'; exit 1 ;;
  esac
  t=$TIME
  if [ -z "$t" ]; then  # by model size (config.json "params"; unknown -> the larger limit)
    params=$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("params", 0)))' \
             "outputs/runs/$r/config.json" 2>/dev/null || echo 0)
    if [ "$params" -gt 0 ] && [ "$params" -lt 15000000 ]; then t=2:30:00; else t=4:00:00; fi
  fi
  args="$ARGS"
  [ "$FORCE" = 1 ] || args="$args --only-missing"
  # NAME / ARGS travel as environment variables (--export=ALL): the --export=ALL,ARGS=... form
  # splits on commas, which would break arguments like --modes guarded,meter
  id=$(NAME="$r" ARGS="$args" sbatch --parsable "${G[@]}" ${DEP[@]+"${DEP[@]}"} -t "$t" -J "eval_$r" --export=ALL pace/evaluate.sbatch)
  submitted=$((submitted + 1))
  if [ "$FORCE" = 1 ]; then
    echo "  $r: job $id  ($t; everything requested, FORCE=1${ARGS:+; $ARGS})"
  else
    echo "  $r: job $id  ($t) computes:"
    echo "$todo" | sed 's/^/      /'
  fi
done
echo "$submitted job(s) submitted.  watch: squeue -u \$USER     logs: slurm-eval_<run>-<job>.out"
