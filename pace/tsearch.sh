#!/bin/bash
# Submit the time-signature batch (run on the login node from the repo root, after git pull):
#   bash pace/tsearch.sh                     # any of H100 / H200 / RTX Pro Blackwell (default)
#   GPUS=gpu-h100 bash pace/tsearch.sh       # other GPU types (node features, "a|b" for several)
#
# Why: the best model (sf_base_reg_cab) writes the right notes (note F1 0.91) but puts only 35% of
# them at the right place in the bar on real audio, against 58% when it is given the true previous
# bars. Most of that gap is the time signature: without the true context it gets 33-41% of them
# right, even with the annotated beats and downbeats, because it picks a window's time signature in
# one greedy step before writing a note. So:
#
#   job / run                 what                                                      starts
#   eval_<run>@<condition>    the new decoding modes on the two best models, one job   at once
#                             per test set: ts_true (the true time signature forced,
#                             the most any choice can give), tsearch (each piece
#                             decoded under its 3 likeliest time signatures, keeping
#                             the most probable), tsearch_beat (keeping the one whose
#                             beats agree with the tracked beats)
#   sf_base_reg_cab_s1        the best recipe again with another seed: how much one     after those
#                             run differs from the next (every comparison so far was    (queued first)
#                             one run each)
#   sf_base_reg_cabx          the best recipe with exact beats everywhere in training   "
#                             (--beat-noise none --transcribed-beats gt): can the
#                             model use beats when they are right? (real_gt_beats)
#   sf_base_reg_cab_c128      the best recipe with twice the context notes (128 before  "
#                             / 64 after): more bars to read the meter from
#   eval_<run>@<condition>    the remaining test sets for the two best models, and      after the
#                             tsearch on the best model without beats (sf_base_reg_ca)  trainings start
#
# The trainings evaluate themselves afterwards (every mode, now with the time-signature ones; what
# their 8 h does not reach: PER_CONDITION=1 bash pace/reevaluate.sh <run>). A run that stops at the
# time limit resumes when it is submitted again with the same NAME / ARGS (pace/train.sbatch).
set -euo pipefail
GPUS=${GPUS:-HX00}  # node feature: an H100 or an H200 (one feature: see pace/README.md, "Which GPU")
G=(--gres=gpu:1 "--constraint=$GPUS")
export GPUS
TS_MODES="--modes ts_true,tsearch,tsearch_beat"
FIRST="real,clean,periscope,real_gt_beats"      # the numbers that decide: the demo's input, MIDI,
REST="synthetic,clean_gt_beats,real_no_beats"   # other transcriptions, the beat ceiling; then the rest

echo "time-signature modes, first test sets:"
PER_CONDITION=1 ARGS="$TS_MODES --conditions $FIRST" bash pace/reevaluate.sh sf_base_reg_cab s2s_m_cab

# NAME / ARGS travel as environment variables (--export=ALL): sbatch's --export=ALL,NAME=..,ARGS=..
# form splits on commas
submit() {  # name, walltime, args
  local id
  id=$(NAME="$1" ARGS="$3" sbatch --parsable "${G[@]}" -t "$2" --export=ALL pace/train.sbatch)
  echo "  $1: job $id  ($2; $3)" >&2
  echo "$id"
}
CTX="--context-before 64 --context-after 32 --aux-metrical"
REG="--dropout 0.2 --weight-decay 0.1 --lr 2e-4"
echo "trainings:"
a=$(submit sf_base_reg_cab_s1 8:00:00 "--size base $REG $CTX --beats --seed 1")
b=$(submit sf_base_reg_cabx 8:00:00 "--size base $REG $CTX --beats --beat-noise none --transcribed-beats gt")
c=$(submit sf_base_reg_cab_c128 8:00:00 "--size base $REG --context-before 128 --context-after 64 --aux-metrical --beats")

echo "time-signature modes, remaining test sets (once the trainings have started):"
AFTER="after:$a:$b:$c" PER_CONDITION=1 ARGS="$TS_MODES --conditions $REST" bash pace/reevaluate.sh sf_base_reg_cab s2s_m_cab
AFTER="after:$a:$b:$c" PER_CONDITION=1 ARGS="--modes ts_true,tsearch --conditions $FIRST,synthetic" \
  bash pace/reevaluate.sh sf_base_reg_ca
echo "watch: squeue -u \$USER     logs: slurm-eval_<run>@<condition>-<job>.out, slurm-scoreformer-<job>.out"
echo "then: python3 scripts/compare_runs.py   (outputs/compare/summary.md: 'Time-signature search per test set')"
