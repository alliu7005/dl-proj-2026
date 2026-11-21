#!/bin/bash
# Submit Phase 2 + 3 of v2 (on the login node from the repo root, after git pull):
#   bash pace/meter.sh
#   LABELLERS="nl_base" bash pace/meter.sh       # which note labellers get the meter's grids
#
# Phase 2, the meter model (src/meter.py): from the tracked beats and the notes of a whole piece to
# the score's grid (notated beats, bar lines, a time signature per bar). Phase 3, the real v2
# system: the note labeller on that grid, its bars, time signatures and an estimated key signature,
# so nothing comes from the score (scripts/eval_labeller.py --grid meter:<run>).
#
#   job              what                                                          waits for
#   metprep          whole performances: notes + annotated grids (+ the cut Transkun -
#                    transcriptions and Beat This!'s beats), then the meter tests
#   meter_base       train the meter model, evaluate it on the clean / real test     metprep
#                    performances and save its grids (scripts/eval_meter.py)
#   v2_<labeller>    every labeller in LABELLERS on the meter model's grids           meter_base, and any
#                                                                                   labeller job still
#                                                                                   queued or running
set -euo pipefail
GPUS=${GPUS:-gpu-h100|gpu-h200|gpu-rtxpro-blackwell}
G=(--gres=gpu:1 "--constraint=$GPUS")
LABELLERS=${LABELLERS:-"nl_base nl_large"}
METER=meter_base
prep=$(sbatch --parsable -J metprep -c 8 --mem=32G -t 1:00:00 -o slurm-metprep-%j.out --wrap="source pace/common.sh; \
python scripts/build_meter_data.py && python -m tests.test_meter")
echo "whole performances + tests: job $prep"
met=$(sbatch --parsable "${G[@]}" -J "$METER" -c 8 --mem=32G -t 4:00:00 --dependency=afterok:$prep \
      -o slurm-$METER-%j.out --wrap="source pace/common.sh; \
python scripts/train_meter.py --name $METER --steps 20000 --batch 32 --chunk 512 --normal-priority \
\$( [ -f outputs/runs/$METER/last.pt ] && echo --resume ) && python scripts/eval_meter.py --run $METER")
echo "meter model: job $met"
# labeller jobs still queued or running: Phase 3 waits for them too (afterany: a failed one is skipped)
pending=$(squeue -h -u "$USER" -n labeller -o "%i" | tr '\n' ':' | sed 's/:$//')
dep="afterok:$met"
[ -n "$pending" ] && dep="$dep,afterany:$pending"
for nl in $LABELLERS; do
  id=$(sbatch --parsable "${G[@]}" -J "v2_$nl" -c 8 --mem=32G -t 2:00:00 --dependency="$dep" -o slurm-v2_$nl-%j.out \
       --wrap="source pace/common.sh; python scripts/eval_labeller.py --run $nl --grid meter:$METER --normal-priority")
  echo "  $nl on the meter grids: job $id (waits for $dep)"
done
echo "watch: squeue -u \$USER     then: python3 scripts/compare_runs.py   ('Note labeller (v2)': true grid vs meter model)"
