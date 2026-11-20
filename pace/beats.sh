#!/bin/bash
# Submit the beat-tracking batch (run on the login node from the repo root, after git pull):
#   bash pace/beats.sh                       # any of H100 / H200 / RTX Pro Blackwell (default)
#   GPUS=gpu-h100 bash pace/beats.sh         # other GPU types (node features, "a|b" for several)
#   STAGE=beat-runs bash pace/beats.sh       # only the two beat trainings, from scratch, on the beats
#                                            # and configs/beat_noise.json already there (e.g. after
#                                            # the tracker-error model was refitted)
#
# Why: the best model (sf_base_reg_ca) reaches 57% of its oracle-context onset F1 on real audio, and
# even WITH the true previous bars it misplaces ~40% of onsets, while the beat-snap + XGBoost
# baseline, given the annotated beats, reaches onset F1 0.74. So the model now gets the beats as an
# input (src/beats.py): Beat This! on the audio at inference, ground truth + fitted tracker errors in
# training. These runs also mix in the real-transcription data for the first time (the split now
# exists), so a control run with the same data and no beats separates the two effects.
#
#   job / run          what                                                       depends on
#   beatprep (CPU)     ground-truth beats for every split (ASAP annotations; the  -
#                      score alignment for PERiScoPe), checked against the
#                      annotations, + the beat tests on this data build
#   beattrack (GPU)    Beat This! on the MAESTRO audio (held-out folds), fit of   beatprep
#                      configs/beat_noise.json
#   sf_base_reg_ca_r   the best model's recipe, retrained with the real-          beatprep
#                      transcription data: the control
#   sf_base_reg_cab    + beats                                                    beattrack
#   s2s_m_cab          the runner-up (25.9M Score2Seq) + beats                    beattrack
#   eval_<run>         every finished run: the new `piece` decoding mode          starts after the
#                      (one time signature per piece) via pace/reevaluate.sh      three trainings start
#
# Each training job then evaluates its run on every test set and mode, including the beat-only
# conditions (annotated beats, no beats) for the beat models (src/eval/plan.py BEATS).
set -euo pipefail
GPUS=${GPUS:-gpu-h100|gpu-h200|gpu-rtxpro-blackwell}  # node features: whichever of these GPUs frees up first
G=(--gres=gpu:1 "--constraint=$GPUS")
STAGE=${STAGE:-all}
BEAT_RUNS=(sf_base_reg_cab s2s_m_cab)

# NAME / ARGS travel as environment variables (--export=ALL): sbatch's --export=ALL,NAME=..,ARGS=..
# form splits on commas. Walltime: training (~6 h for the 45M run, which early-stopped at epoch
# 10.9) plus its evaluation; a run that overruns resumes from last.pt, and pace/reevaluate.sh
# evaluates whatever its job did not get to.
submit() {  # name, walltime, dependency (sbatch argument or ""), args
  local id dep=()
  [ -n "$3" ] && dep=("$3")
  id=$(NAME="$1" ARGS="$4" sbatch --parsable "${G[@]}" -t "$2" ${dep[@]+"${dep[@]}"} --export=ALL pace/train.sbatch)
  echo "  $1: job $id  ($2; $4)" >&2
  echo "$id"
}
CTX="--context-before 64 --context-after 32 --aux-metrical"
REG="--dropout 0.2 --weight-decay 0.1 --lr 2e-4"
beat_runs() {  # dependency (sbatch argument or ""); prints the two job ids
  local b c
  b=$(submit sf_base_reg_cab 8:00:00 "$1" "--size base $REG $CTX --beats")
  c=$(submit s2s_m_cab 7:00:00 "$1" "--arch score2seq --d-model 384 --enc-layers 6 --dec-layers 6 $REG $CTX --beats")
  echo "$b $c"
}

case "$STAGE" in
  all) ;;
  beat-runs)
    # a run directory that exists would be resumed, with the tracker errors it started with
    for r in "${BEAT_RUNS[@]}"; do
      if [ -e "outputs/runs/$r" ]; then
        echo "outputs/runs/$r exists: these runs start from scratch. Cancel its job if it is still queued or"
        echo "running (squeue -u \$USER; the training log names the run), then: rm -rf outputs/runs/$r"
        exit 1
      fi
    done
    for s in train val train_transcribed val_transcribed; do
      f=data/processed/$s/beats_$([ "${s%_transcribed}" = "$s" ] && echo gt || echo tracked).npz
      [ -f "$f" ] || { echo "$f missing: run the whole batch (bash pace/beats.sh) first"; exit 1; }
    done
    python3 - <<'EOF' || exit 1
import json, sys
d = json.load(open("configs/beat_noise.json"))
if not d.get("fitted") or "octave_profiles" not in d:
    sys.exit("configs/beat_noise.json is not the current fitted model (per-performance profiles): git pull, "
             "or python scripts/fit_beat_noise.py")
print("tracker errors: configs/beat_noise.json (%d right-tempo + %d wrong-tempo profiles)"
      % (len(d.get("profiles") or []), len(d.get("octave_profiles") or [])))
EOF
    echo "beat runs, from scratch:"
    beat_runs "" > /dev/null
    echo "watch: squeue -u \$USER     logs: slurm-scoreformer-<job>.out"
    exit 0 ;;
  *) echo "unknown STAGE=$STAGE (all, beat-runs)"; exit 1 ;;
esac

prep=$(sbatch --parsable -J beatprep -c 8 --mem=32G -t 2:00:00 -o slurm-beatprep-%j.out --wrap="source pace/common.sh; \
python scripts/build_beats.py --check --workers 8 && python -m tests.test_beats && python -m tests.test_eval_plan \
&& python -m tests.test_pipeline")
echo "ground-truth beats + tests: job $prep"
track=$(sbatch --parsable "${G[@]}" --dependency=afterok:$prep pace/track_beats.sbatch)
echo "Beat This! + noise fit: job $track"
a=$(submit sf_base_reg_ca_r 8:00:00 "--dependency=afterok:$prep" "--size base $REG $CTX")
ids=$(beat_runs "--dependency=afterok:$track")
read -r b c <<< "$ids"

# piece mode for every run evaluated so far, once the trainings have started (so they get the
# GPU-time cap first); runs in training or without best.pt are skipped by reevaluate.sh
echo "piece mode for the finished runs:"
AFTER="after:$a:$b:$c" ARGS="--modes piece" bash pace/reevaluate.sh
echo "watch: squeue -u \$USER     logs: slurm-beatprep-<job>.out, slurm-beattrack-<job>.out, slurm-scoreformer-<job>.out"
