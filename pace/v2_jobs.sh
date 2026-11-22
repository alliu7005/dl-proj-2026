#!/bin/bash
# The work of pace/v2.sh's jobs (each job: source pace/common.sh; bash pace/v2_jobs.sh <task> ...).
# Several models share one GPU job, so the batch stays under PACE's per-user job limit:
#
#   meters <run>...        train these meter models side by side (each its own log), up to
#                          --max-hours, resuming from last.pt; then tune + evaluate every finished
#                          one not evaluated yet. Fails unless all are trained and evaluated.
#   labellers <run>...     train these labellers side by side (resuming), then evaluate each on
#                          the true grid. Fails unless all are trained.
#   grids                  the earlier meter models' grids decoded again (GRIDS_REEVAL: the decoder
#                          now keeps the bars of a step longer than a bar), the oracle grids
#                          (eval_meter.py --oracle), then Beat This! on the clean test MIDI rendered
#                          to audio (synth_beats.py)
#   evals <labeller>...    for each: finish its training if it is not done (a labeller trained
#                          here), else evaluate it on the true grid (an earlier labeller: its
#                          results predate the current evaluation); then on every grid: the oracle,
#                          GRIDS_OLD, METERS (a grid not there is skipped with the reason)
#   systems <spec>...      whole pieces (scripts/system_scores.py): <labeller>+<meter> = v2 (only
#                          with both models finished and the meter model evaluated),
#                          <run>:<mode> = seq2seq, <run>:<mode>:sb = seq2seq with the rendered-MIDI
#                          beats on clean MIDI
#
# Run options come from pace/v2.sh (METERS, GRIDS_OLD, NEW_NLS and the flags below).
set -uo pipefail
METERS=${METERS-"meter_v3 meter_v3_s1 meter_v3_s2 meter_v3_large"}
NEW_NLS=${NEW_NLS-"nl_base_gn nl_base_gn_s1"}
GRIDS_OLD=${GRIDS_OLD-"meter_v2_nonote"}
GRIDS_REEVAL=${GRIDS_REEVAL-"meter_v2_nonote meter_v2 meter_v2_s1 meter_v2_asap meter_base"}
MAX_HOURS=${MAX_HOURS:-7.5}
JOB=${SLURM_JOB_ID:-local}

meter_flags() {
  case "$1" in
    meter_v3) echo "--steps 60000 --patience 8" ;;
    meter_v3_s1) echo "--steps 60000 --patience 8 --seed 1" ;;
    meter_v3_s2) echo "--steps 60000 --patience 8 --seed 2" ;;
    meter_v3_large) echo "--steps 60000 --patience 8 --d-model 384 --layers 8" ;;
    *) echo "unknown meter run $1 (add it to meter_flags() in pace/v2_jobs.sh)" >&2; return 1 ;;
  esac
}
nl_flags() {
  case "$1" in
    nl_base_gn) echo "--size base --grid-noise 0.8" ;;
    nl_base_gn_s1) echo "--size base --grid-noise 0.8 --seed 1" ;;
    *) echo "unknown labeller run $1 (add it to nl_flags() in pace/v2_jobs.sh)" >&2; return 1 ;;
  esac
}
done_run() { grep -q '"phase": "done"' "outputs/runs/$1/status.json" 2>/dev/null; }
resume() { [ -f "outputs/runs/$1/last.pt" ] && echo --resume; }
in_list() { case " $2 " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

train_labeller() {  # name -> trains (or resumes) it; logs to slurm-labeller_<name>-<job>.log
  local nl=$1 f
  f=$(nl_flags "$nl") || return 1
  # shellcheck disable=SC2086
  python scripts/train_labeller.py --name "$nl" $f --normal-priority --workers 7 $(resume "$nl") \
    > "slurm-labeller_$nl-$JOB.log" 2>&1
}

task=${1:?task: meters, labellers, grids, evals, systems}
shift
case "$task" in
  meters)
    pids=()
    for m in "$@"; do
      f=$(meter_flags "$m") || exit 1
      if done_run "$m"; then echo "$m: trained already"; continue; fi
      echo "$m: training ($f), log slurm-met_$m-$JOB.log"
      # shellcheck disable=SC2086
      python scripts/train_meter.py --name "$m" $f --workers 6 --normal-priority --max-hours "$MAX_HOURS" \
        $(resume "$m") > "slurm-met_$m-$JOB.log" 2>&1 &
      pids+=("$!")
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "$p"; done
    ok=0
    for m in "$@"; do
      if ! done_run "$m"; then echo "$m: not done yet (the next job resumes it)"; ok=1; continue; fi
      if [ ! "outputs/runs/$m/eval_meter.json" -nt "outputs/runs/$m/best.pt" ]; then
        python scripts/eval_meter.py --run "$m" --tune || ok=1
      fi
      [ -f "outputs/runs/$m/eval_meter.json" ] || ok=1
    done
    exit $ok ;;
  labellers)
    pids=()
    for nl in "$@"; do
      if done_run "$nl"; then echo "$nl: trained already"; continue; fi
      echo "$nl: training ($(nl_flags "$nl")), log slurm-labeller_$nl-$JOB.log"
      train_labeller "$nl" &
      pids+=("$!")
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "$p"; done
    ok=0
    for nl in "$@"; do
      if ! done_run "$nl"; then echo "$nl: not done (its evals job resumes it)"; ok=1; continue; fi
      python scripts/eval_labeller.py --run "$nl" --normal-priority || ok=1
    done
    exit $ok ;;
  grids)
    ok=0
    for m in $GRIDS_REEVAL; do  # their tuned decoder (decode.json) kept
      [ -f "outputs/runs/$m/best.pt" ] || { echo "$m: no best.pt, skipped"; continue; }
      python scripts/eval_meter.py --run "$m" --device cpu || ok=1
    done
    python scripts/eval_meter.py --run meter_oracle --oracle --device cpu || ok=1
    python scripts/synth_beats.py || ok=1
    exit $ok ;;
  evals)
    ok=0
    for nl in "$@"; do
      if in_list "$nl" "$NEW_NLS" && ! done_run "$nl"; then
        echo "$nl: not trained yet, resuming"
        train_labeller "$nl" || { echo "$nl: training failed"; ok=1; continue; }
        python scripts/eval_labeller.py --run "$nl" --normal-priority || ok=1
      elif ! in_list "$nl" "$NEW_NLS"; then
        echo "== $nl on the true grid"
        python scripts/eval_labeller.py --run "$nl" --normal-priority || ok=1
      fi
      for m in meter_oracle $GRIDS_OLD $METERS; do
        echo "== $nl on $m's grids"
        python scripts/eval_labeller.py --run "$nl" --grid "meter:$m" --conditions clean,synthetic,real,periscope \
          --normal-priority || ok=1
      done
    done
    exit $ok ;;
  systems)
    ok=0
    for spec in "$@"; do
      echo "== $spec"
      case "$spec" in
        *+*) nl=${spec%%+*}; m=${spec##*+}
             if ! done_run "$nl" || ! done_run "$m" || [ ! -f "outputs/runs/$m/eval_meter.json" ]; then
               echo "$spec: skipped, $nl or $m not finished (or $m not evaluated)"; ok=1; continue
             fi
             python scripts/system_scores.py --v2 "$spec" || ok=1 ;;
        *:*:sb) r=${spec%%:*}; rest=${spec#*:}; python scripts/system_scores.py --run "$r" --mode "${rest%%:*}" \
                  --midi-beats synth || ok=1 ;;
        *:*) python scripts/system_scores.py --run "${spec%%:*}" --mode "${spec##*:}" || ok=1 ;;
        *) echo "unknown system $spec"; ok=1 ;;
      esac
    done
    exit $ok ;;
  *) echo "unknown task $task (meters, labellers, grids, evals, systems)"; exit 2 ;;
esac
