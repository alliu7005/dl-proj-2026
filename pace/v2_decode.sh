#!/bin/bash
# v2 tenth round: the deep-learning side of the user's rhythm problems, and the rules measured
# against it (on the login node from the repo root, after git pull, with the queue empty: 8 jobs).
#   bash pace/v2_decode.sh
#
# The user's song showed two kinds of error (2026-11-26) that rules now patch (src/timing.py,
# src/readable.py). This round tests the model-side fixes and measures every rule on its own:
#   * the labeller writes straight 16th syncopation as triplets, 67,000 to 1 sure. Beat-level
#     decoding (src/beat_decode.py: one split family per beat, the model's probabilities fused with
#     the timing) is tuned on validation (ASAP + Gangqinpu); on ASAP alone it gained 0.1 point and
#     could not move the song. So the labels are audited (scripts/audit_labels.py: beats written as
#     triplets but played straight, against ASAP's 6-8%), and nl_gq is retrained with those notes'
#     positions left out of the loss (nl_gq_mc, --mask-contradicted).
#   * the meter decoder stretches a short bar into a jump. A local tempo consistency term
#     (src/meter.py local_tempo_logprior) is tuned on validation as meter_v4_gq_lt; on the song it
#     removed the jumps but put four bar lines off the downbeats, so it is measured, not assumed.
# Whole pieces, ASAP and Gangqinpu: the timing check alone (_tc) against each rule (_tb, _rd, _jf),
# the beat decoding (_bd), the retrained labeller, the local tempo meter, the rules together and the
# demo's default (all of them); `missing`: a system already scored (by an earlier batch) is kept.
#
#   job       what                                                                    waits for
#   dctests   the test suites (CPU)                                                    -
#   dcaudit   label audit: train, train_gangqinpu, val, val_gangqinpu (CPU)            dctests (ok)
#   dctune    nl_gq's beat decoding tuned + evaluated + published; meter_v4_gq_lt      dctests (ok)
#   dctrain   nl_gq_mc trained (contradicted labels masked), evaluated, published      dctests (ok)
#   dceval    nl_gq_mc on meter_v4_gq's grids; its beat decoding tuned                 dctrain (ok)
#   dcasap    whole pieces on ASAP: 9 systems                                          dctune, dceval (any)
#   dcgq      the same on Gangqinpu's whole pieces                                     dctune, dceval (any)
#   dcfinal   Beyer & Dai's metrics for every system (ASAP, Gangqinpu), paired.md      dcasap, dcgq (any)
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
for f in outputs/runs/nl_gq/best.pt outputs/runs/meter_v4_gq/best.pt outputs/runs/meter_v4_gq/decode.json \
         outputs/system_eval/manifest.json outputs/system_eval_gangqinpu/manifest.json; do
  [ -e "$f" ] || { echo "no $f: run pace/v2_gqlabel.sh (and the rounds before) first"; exit 1; }
done
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: run bash pace/bd_setup.sh first"; exit 1; }
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
JOBS="bash pace/v2_jobs.sh"
GQOUT=outputs/system_eval_gangqinpu
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
M="nl_gq+meter_v4_gq:tidy,d95,tc"
SPECS="$M,missing $M,tb,missing $M,rd,missing $M,jf,missing $M,bd,missing $M,tb,rd,jf,missing $M,tb,rd,jf,bd,missing"
SPECS="$SPECS nl_gq_mc+meter_v4_gq:tidy,d95,tc,missing nl_gq+meter_v4_gq_lt:tidy,d95,tc,missing"
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
submit dctests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; python -m tests.test_beat_decode && python -m tests.test_readable && python -m tests.test_timing && python -m tests.test_v2_system && python -m tests.test_system_eval && python -m tests.test_compare_runs && python -m tests.test_publish"
tests=$ID
echo "tests: job $tests"
submit dcaudit -c 4 --mem=48G -t 4:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; python scripts/audit_labels.py --splits train,train_gangqinpu,val,val_gangqinpu --windows 20000"
echo "label audit: job $ID"
submit dctune "${G[@]}" -c 8 --mem=64G -t 4:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $JOBS beatdecode nl_gq; $JOBS localtempo meter_v4_gq"
tune=$ID
echo "nl_gq's beat decoding, meter_v4_gq_lt: job $tune"
submit dctrain "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $JOBS labellers nl_gq_mc"
train=$ID
echo "nl_gq_mc (contradicted labels masked): job $train"
submit dceval "${G[@]}" -c 8 --mem=64G -t 4:00:00 "--dependency=afterok:$train" --wrap="source pace/common.sh; METERS=meter_v4_gq GRIDS_OLD= NEW_NLS=nl_gq_mc $JOBS evals nl_gq_mc; $JOBS beatdecode nl_gq_mc"
evl=$ID
echo "nl_gq_mc on meter_v4_gq's grids, its beat decoding: job $evl"
submit dcasap "${G[@]}" -c 8 --mem=48G -t 16:00:00 "--dependency=afterany:$tune:$evl" --wrap="source pace/common.sh; $JOBS systems $SPECS"
asap=$ID
echo "whole pieces, ASAP (9 systems): job $asap"
submit dcgq "${G[@]}" -c 8 --mem=48G -t 16:00:00 "--dependency=afterany:$tune:$evl" --wrap="source pace/common.sh; $GQSYS $JOBS systems $SPECS"
gq=$ID
echo "whole pieces, Gangqinpu: job $gq"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
submit dcfinal -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''"
echo "Beyer & Dai's metrics + paired comparisons: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
