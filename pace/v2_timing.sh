#!/bin/bash
# v2: the timing check measured on whole pieces (on the login node from the repo root, after git
# pull, with the queue empty or nearly: 3 jobs). pace/v2_pm2s2.sh measures the same systems in its
# p2sys job; this is for when that batch was submitted before the timing check existed.
#   bash pace/v2_timing.sh
#
# The timing check (src/timing.py): a note the labeller wrote on a triplet position but played on a
# straight eighth or sixteenth is written there. On the user's song it moved 89 notes (bars 64-71:
# triplets 40% -> 20%, eighths 7% -> 23%). Here: nl_gq + meter_v4_gq (the demo's default) with it
# (_tc) against without, on ASAP and Gangqinpu, rated with Beyer & Dai's metrics and paired.
#
#   job       what                                                                waits for
#   tctests   the test suites (CPU)                                                -
#   tcsys     nl_gq + meter_v4_gq:tidy,d95,tc on ASAP and Gangqinpu's whole pieces tctests (ok)
#   tcfinal   Beyer & Dai's metrics for every system (ASAP, Gangqinpu), paired.md  tcsys (any)
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
submit tctests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; python -m tests.test_timing && python -m tests.test_v2_system && python -m tests.test_system_eval && python -m tests.test_compare_runs"
tests=$ID
echo "tests: job $tests"
submit tcsys "${G[@]}" -c 8 --mem=48G -t 6:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $JOBS systems $NL+meter_v4_gq:tidy,d95,tc; $GQSYS $JOBS systems $NL+meter_v4_gq:tidy,d95,tc"
sys=$ID
echo "$NL + meter_v4_gq with the timing check, ASAP and Gangqinpu: job $sys"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
submit tcfinal -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$sys" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''"
echo "Beyer & Dai's metrics + paired comparisons: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
