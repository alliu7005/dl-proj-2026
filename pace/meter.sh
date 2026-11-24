#!/bin/bash
# Submit the meter model's second round, v2 Phases 2 + 3 (on the login node from the repo root, after git pull;
# the third round, and the whole v2 system: pace/v2.sh):
#   bash pace/meter.sh
#   SKIP_PREP=1 bash pace/meter.sh                 # data/processed/meter already has the PERiScoPe performances
#   METERS="meter_v2" LABELLERS="nl_base" LARGE_ON="" bash pace/meter.sh   # a subset
#
# Phase 2, the meter model (src/meter.py): from the tracked beats and the notes of a whole piece to
# the score's grid (notated beats, bar lines, a time signature per bar). Phase 3, the real v2
# system: the note labeller on that grid, its bars, time signatures and an estimated key signature,
# so nothing comes from the score (scripts/eval_labeller.py --grid meter:<run>).
#
# The first meter model (meter_base) got the time signature of a third of the test bars right. This
# round: 6x the pieces (PERiScoPe's 971 training works, their grid from the score alignment), a note
# encoder (every note of a beat embedded and pooled, not only hand-made beat features), pitch-class
# profiles, whole-piece-length training chunks, piece-balanced sampling, real transcriptions without
# synthetic noise on top, model selection on bar lines + time signatures, and the decoder's weights
# tuned on validation (scripts/eval_meter.py --tune). The runs separate what each part gives:
#
#   job                 what                                                         waits for
#   metprep             whole performances: ASAP (+ transcriptions, Beat This!'s     -
#                       beats) and PERiScoPe (train / val / test, grids from the
#                       alignment), the alignment grids checked against ASAP's
#                       annotations (data/processed/meter/check.json), the tests
#   met_meter_v2        the new model on everything; train, tune the decoder, eval   metprep
#   met_meter_v2_s1     the same, seed 1 (how much is seed noise)                    metprep
#   met_meter_v2_nonote no note encoder (what the note encoder gives)                metprep
#   met_meter_v2_asap   no PERiScoPe (what the data gives)                           metprep
#   metev_meter_base    the first meter model: decoder tuned, + the PERiScoPe test   metprep
#   v2_<nl>@<meter>     labeller <nl> on that meter model's grids (clean,            that meter job, and
#                       synthetic, real, periscope): LABELLERS on every meter        labeller jobs still
#                       model, nl_large also on LARGE_ON                             queued or running
set -euo pipefail
GPUS=${GPUS:-HX00}
G=(--gres=gpu:1 "--constraint=$GPUS")
METERS=${METERS:-"meter_v2 meter_v2_s1 meter_v2_nonote meter_v2_asap"}
LABELLERS=${LABELLERS:-"nl_base"}
LARGE_ON=${LARGE_ON-"meter_v2 meter_base"}  # empty = nl_large on none

flags() {  # train_meter.py options of each meter run (no note encoder is the default since round 3)
  case "$1" in
    meter_v2) echo "--note-encoder" ;;
    meter_v2_s1) echo "--note-encoder --seed 1" ;;
    meter_v2_nonote) echo "--no-note-encoder" ;;
    meter_v2_asap) echo "--note-encoder --no-periscope" ;;
    *) echo "unknown meter run $1 (add it to flags() in pace/meter.sh)" >&2; return 1 ;;
  esac
}

dep_prep=""
if [ -z "${SKIP_PREP:-}" ]; then
  prep=$(sbatch --parsable -J metprep -c 8 --mem=64G -t 2:00:00 -o slurm-metprep-%j.out --wrap="source pace/common.sh; \
python scripts/build_meter_data.py --workers 8 --check && python -m tests.test_meter")
  echo "whole performances (ASAP + PERiScoPe) + check + tests: job $prep"
  dep_prep="--dependency=afterok:$prep"
fi
declare -A job
for m in $METERS; do
  f=$(flags "$m")
  job[$m]=$(sbatch --parsable "${G[@]}" -J "met_$m" -c 8 --mem=64G -t 8:00:00 $dep_prep -o "slurm-met_$m-%j.out" \
    --wrap="source pace/common.sh; python scripts/train_meter.py --name $m $f --workers 6 --normal-priority \
\$( [ -f outputs/runs/$m/last.pt ] && echo --resume ) && python scripts/eval_meter.py --run $m --tune")
  echo "meter model $m ($([ -n "$f" ] && echo "$f" || echo "everything")): job ${job[$m]}"
done
job[meter_base]=$(sbatch --parsable "${G[@]}" -J metev_meter_base -c 8 --mem=64G -t 3:00:00 $dep_prep \
  -o slurm-metev_meter_base-%j.out --wrap="source pace/common.sh; python scripts/eval_meter.py --run meter_base --tune")
echo "first meter model re-evaluated (tuned decoder, + PERiScoPe test): job ${job[meter_base]}"
# labeller jobs still queued or running: Phase 3 waits for them too (afterany: a failed one is skipped)
pending=$(squeue -h -u "$USER" -n labeller -o "%i" | tr '\n' ':' | sed 's/:$//')
for m in $METERS meter_base; do
  nls="$LABELLERS"
  case " $LARGE_ON " in *" $m "*) nls="$nls nl_large" ;; esac
  for nl in $nls; do
    dep="afterok:${job[$m]}"
    [ -n "$pending" ] && dep="$dep,afterany:$pending"
    id=$(sbatch --parsable "${G[@]}" -J "v2_$nl@$m" -c 8 --mem=32G -t 4:00:00 --dependency="$dep" \
         -o "slurm-v2_$nl@$m-%j.out" --wrap="source pace/common.sh; python scripts/eval_labeller.py --run $nl \
--grid meter:$m --conditions clean,synthetic,real,periscope --normal-priority")
    echo "  $nl on $m's grids: job $id (waits for $dep)"
  done
done
echo "watch: squeue -u \$USER     then: python3 scripts/compare_runs.py   ('Meter models (v2)', 'Note labeller (v2)')"
