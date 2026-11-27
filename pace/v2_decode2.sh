#!/bin/bash
# v2 tenth round, follow-up: the correction rules judged on top of the deep-learning configuration,
# for both labellers, and the verdict published so the demo turns off what the models made
# unnecessary (on the login node from the repo root, after git pull; 4 jobs, alongside or after
# pace/v2_decode.sh: its jobs still queued are waited for, found by name).
#   bash pace/v2_decode2.sh
#
# The deep-learning configuration is a labeller + its tuned beat decoding (src/beat_decode.py) on
# meter_v4_gq. pace/v2_decode.sh measured the rules on nl_gq on top of the timing check; this
# measures each correction rule (tc, tb, jf) added to the models alone, for nl_gq and for nl_gq_mc
# (retrained without the contradicted labels), plus nl_gq_mc without anything and with every rule.
# scripts/rule_verdict.py then decides, for the labeller the release serves: the beat decoding on
# unless it hurts; a correction on only if it makes the system significantly better on real
# recordings (ASAP or Gangqinpu) and worse on neither; publish_model.py --settings puts that in
# the release manifest, which the demo reads (PUBLISH=0: decided, not published).
#
#   job       what                                                                    waits for
#   d2tests   the test suites (CPU)                                                    -
#   d2asap    whole pieces on ASAP: nl_gq + bd (+ tb / jf); nl_gq_mc plain, + bd,     d2tests (ok); dctune, dceval
#             + tc / tb / jf, everything                                               (any, if still queued)
#   d2gq      the same on Gangqinpu's whole pieces                                     the same
#   d2final   Beyer & Dai's metrics (ASAP, Gangqinpu), paired.md, the rule verdict,    d2asap, d2gq; dcfinal (any)
#             published to the release
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
PUBLISH=${PUBLISH:-1}
for f in outputs/runs/nl_gq/best.pt outputs/runs/meter_v4_gq/best.pt outputs/system_eval/manifest.json \
         outputs/system_eval_gangqinpu/manifest.json; do
  [ -e "$f" ] || { echo "no $f: run pace/v2_gqlabel.sh (and the rounds before) first"; exit 1; }
done
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: run bash pace/bd_setup.sh first"; exit 1; }
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
JOBS="bash pace/v2_jobs.sh"
GQOUT=outputs/system_eval_gangqinpu
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
NL="nl_gq+meter_v4_gq:tidy,d95"
MC="nl_gq_mc+meter_v4_gq:tidy,d95"
# nl_gq's models-only and timing-check systems come from earlier rounds / pace/v2_decode.sh
SPECS="$NL,bd,missing $NL,tb,bd,missing $NL,jf,bd,missing"
SPECS="$SPECS $MC,missing $MC,bd,missing $MC,tc,bd,missing $MC,tb,bd,missing $MC,jf,bd,missing $MC,tc,tb,rd,jf,bd,missing"
queued() {  # the IDs of this user's queued / running jobs with these names, joined by ':' (empty: none)
  local ids=""
  for name in "$@"; do
    for id in $(squeue -u "$USER" -h -n "$name" -o %i 2>/dev/null); do ids="${ids:+$ids:}$id"; done
  done
  echo "$ids"
}
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
submit d2tests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; python -m tests.test_rule_verdict && python -m tests.test_beat_decode && python -m tests.test_publish && python -m tests.test_release && python -m tests.test_compare_runs && python -m tests.test_system_eval"
tests=$ID
echo "tests: job $tests"
before=$(queued dctune dceval)
dep="afterok:$tests${before:+,afterany:$before}"
echo "the systems wait for: $dep"
submit d2asap "${G[@]}" -c 8 --mem=48G -t 14:00:00 "--dependency=$dep" --wrap="source pace/common.sh; $JOBS systems $SPECS"
asap=$ID
echo "whole pieces, ASAP (9 systems): job $asap"
submit d2gq "${G[@]}" -c 8 --mem=48G -t 14:00:00 "--dependency=$dep" --wrap="source pace/common.sh; $GQSYS $JOBS systems $SPECS"
gq=$ID
echo "whole pieces, Gangqinpu (9 systems): job $gq"
others=$(queued dcasap dcgq dcfinal)
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
PUB="echo 'PUBLISH=0: the verdict is not published'"
[ "$PUBLISH" != 0 ] && PUB="python scripts/publish_model.py --settings outputs/rule_verdict.json"
submit d2final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq${others:+:$others}" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''; source pace/common.sh; python scripts/rule_verdict.py --labeller auto && $PUB"
echo "Beyer & Dai's metrics, paired comparisons, the rule verdict (published: $PUBLISH): job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
