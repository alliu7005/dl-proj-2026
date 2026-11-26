#!/bin/bash
# v2: the readability rules, the short-bar fix and the beat-level timing check measured on whole pieces (on the login node from
# the repo root, after git pull, with room in the queue: 4 jobs).
#   bash pace/v2_readable.sh
#
# src/readable.py, from the user's reading of the demo's scores: fix_jumps shortens a bar the meter
# model's decoder stretched with a jump (the song's bars 47 and 51, under a second each, had been
# written over a whole bar of long notes and rests; now 2/4 each); readable_lengths closes rests of
# a 16th or less and trims overlaps of a 16th or less (when the new length is one note to write);
# stem_voices sets stems by voice and hides the other voices' rests. src/timing.py check_beats (tb)
# writes straight a whole beat whose timing fits straight sixteenths better than the labeller's
# triplets (the song's bars 64-71: triplets 20% -> 0%). Closing rests lowered duration
# accuracy against ASAP's scores before (src/pipeline.py tidy_lengths), so each is its own system,
# on top of the timing check (_tc, the demo's default; pace/v2_timing.sh or v2_pm2s2.sh measured it,
# and the `missing` runs here fill it in if not).
#
#   job       what                                                                waits for
#   rdtests   the test suites (CPU)                                                -
#   rdasap    nl_gq + meter_v4_gq:tidy,d95,tc with tb / rd / jf / tb,rd,jf on ASAP  rdtests (ok)
#   rdgq      the same on Gangqinpu's whole pieces                                 rdtests (ok)
#   rdfinal   Beyer & Dai's metrics for every system (ASAP, Gangqinpu), paired.md  rdasap, rdgq (any)
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
NL=${NL:-nl_gq}
BD=${BD:-$HOME/scratch/beyer_dai}
for f in "outputs/runs/$NL/best.pt" outputs/runs/meter_v4_gq/best.pt outputs/system_eval/manifest.json \
         outputs/system_eval_gangqinpu/manifest.json; do
  [ -e "$f" ] || { echo "no $f: run pace/v2_gqlabel.sh (and the rounds before) first"; exit 1; }
done
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: run bash pace/bd_setup.sh first"; exit 1; }
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
JOBS="bash pace/v2_jobs.sh"
GQOUT=outputs/system_eval_gangqinpu
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
M="$NL+meter_v4_gq:tidy,d95,tc"
SPECS="$M,missing $M,tb $M,rd $M,jf $M,tb,rd,jf"
n=0
ID=""
submit() {
  local name=$1
  shift
  local kill=()
  case " $* " in *" --dependency=afterok:"*) kill=(--kill-on-invalid-dep=yes) ;; esac
  ID=$(sbatch --parsable -J "$name" -o "slurm-$name-%j.out" ${kill[@]+"${kill[@]}"} "$@")
  n=$((n + 1))
}
submit rdtests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; python -m tests.test_readable && python -m tests.test_timing && python -m tests.test_v2_system && python -m tests.test_system_eval && python -m tests.test_compare_runs"
tests=$ID
echo "tests: job $tests"
submit rdasap "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $JOBS systems $SPECS"
asap=$ID
echo "$NL + meter_v4_gq with the timing check, + tb / rd / jf / all three, ASAP: job $asap"
submit rdgq "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $GQSYS $JOBS systems $SPECS"
gq=$ID
echo "the same on Gangqinpu: job $gq"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
submit rdfinal -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''"
echo "Beyer & Dai's metrics + paired comparisons: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
