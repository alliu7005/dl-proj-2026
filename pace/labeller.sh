#!/bin/bash
# Submit Phase 1 of v2, the note labeller with the true grid (on the login node from the repo root,
# after git pull):
#   bash pace/labeller.sh
#   GPUS=gpu-h100 bash pace/labeller.sh
#
# The labeller writes the score note by note: every performed note gets its bar, position, written
# length, staff and flags (or "delete") in one pass, from where a metrical grid puts it (README,
# "v2: grid first, then label every note"). Phase 1 gives it the true grid, as the beat-snap
# baselines are given the annotated beats: the number to beat is Baseline 2's onset F1 0.718 (clean)
# / 0.738 (real), and the seq2seq model's 0.58 with the true previous bars.
#
#   job          what                                                             waits for
#   labelprep    labels + true grids for every split (scripts/build_notelabels.py), -
#                then the label tests on this data build (CPU)
#   nl_base      8-layer labeller (~26M parameters), train + evaluate on every test labelprep
#                set (pace/train_labeller.sbatch)
#   nl_large     12-layer, d_model 768 (~85M): does capacity help a labeller       labelprep
set -euo pipefail
GPUS=${GPUS:-HX00}
G=(--gres=gpu:1 "--constraint=$GPUS")
prep=$(sbatch --parsable -J labelprep -c 8 --mem=48G -t 2:00:00 -o slurm-labelprep-%j.out --wrap="source pace/common.sh; \
python scripts/build_notelabels.py --workers 8 && python -m tests.test_notelabels")
echo "labels + true grids + tests: job $prep"
submit() {  # name, args
  local id
  id=$(NAME="$1" ARGS="$2" sbatch --parsable "${G[@]}" --dependency=afterok:$prep --export=ALL pace/train_labeller.sbatch)
  echo "  $1: job $id  ($2)"
}
submit nl_base "--size base"
submit nl_large "--size large --dropout 0.15"
echo "watch: squeue -u \$USER     logs: slurm-labelprep-<job>.out, slurm-labeller-<job>.out"
echo "then: python3 scripts/compare_runs.py   (outputs/compare/summary.md: 'Note labeller (v2) per test set')"
