#!/bin/bash
# v2 pilot, part 2: the 2/4-against-4/4 problem, three ways, at small scale (on the login node from
# the repo root, after git pull; its jobs wait for the first pilot's (pace/v2_pilot.sh) still queued,
# found by name, so both fit PACE's queue limit; 6 jobs):
#   bash pace/v2_pilot2.sh
#
# The demo writes one piece in two bar lengths: Lemon 45 bars of 4/4 and 23 of 2/2, Maple Leaf Rag
# 46 of 2/4 and 19 of 4/4. The decoder may change time signature at any bar line, the scores the
# meter model learned from write the same music either way, and nothing in its inputs says how many
# beats a bar repeats at.
#   F  one meter per piece after decoding (src/meter.py consistent_meter, DecodeConfig.one_meter):
#      2/2 = 4/4 written as the piece's more common one; 2/4 bars paired into 4/4 (the pairing by the
#      bar head) or 4/4 split, when that reading holds at least 0.5 / 0.6 of the piece's duple time
#      (meter_v4_gq_om5, _om6: the release's checkpoint and decoder): no training
#   G  the repetition features (rep_features: how similar each beat is to the ones 1-8 beats back):
#      meter_v4_gq_rep_p12k against meter_v4_gq_p12k (the first pilot's, 12k steps)
#   H  bar-length-tolerant training (TS_EQUIV: 2/4 = 4/4 = 2/2 and 6/8 = 12/8 for the time-signature
#      head, the middle of a 4/4 bar not counted against a bar line): meter_v4_gq_tol_p12k
# Every meter run reports ts_acc_equiv (2/4 = 4/4 = 2/2) and ts_changes_per_100_bars next to the
# usual scores. outputs/pilot/report.md (scripts/pilot_report.py) gets sections F-H with a go / no go.
#
#   job       what                                                                waits for
#   r2tests   the test suites (CPU)                                                -
#   r2cons    F: meter_v4_gq_om5, _om6 made and evaluated                          r2tests (ok); q1prep (any)
#   r2train   G, H: the two 12k-step models (and the baseline if the first pilot   r2tests (ok); q1short (any)
#             did not train it), tuned, evaluated
#   r2asap    F on whole pieces, ASAP                                              r2cons (ok); q1asap (any)
#   r2gq      F on whole pieces, Gangqinpu                                         r2cons (ok); q1gq (any)
#   r2final   Beyer & Dai's metrics, paired.md, the pilot report                   all of the above, q1final (any)
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
GQOUT=outputs/system_eval_gangqinpu
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: bash pace/bd_setup.sh first"; exit 1; }
export PUBLISH=0
JOBS="PUBLISH=0 bash pace/v2_jobs.sh"
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
queued() {  # IDs of this user's queued / running jobs with these names, joined by ':' (empty: none)
  local ids=""
  for name in "$@"; do
    for id in $(squeue -u "$USER" -h -n "$name" -o %i 2>/dev/null); do ids="${ids:+$ids:}$id"; done
  done
  echo "$ids"
}
after() {  # "afterok:<ok>" plus ",afterany:<the first pilot's jobs with these names>" when any are queued
  local ok=$1; shift
  local q; q=$(queued "$@")
  echo "afterok:$ok${q:+,afterany:$q}"
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
submit r2tests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; python -m tests.test_bar_length && python -m tests.test_meter && python -m tests.test_ensemble"
tests=$ID
echo "tests: job $tests"
submit r2cons "${G[@]}" -c 8 --mem=48G -t 4:00:00 "--dependency=$(after "$tests" q1prep)" --wrap="source pace/common.sh; $JOBS onemeter meter_v4_gq 0.5 meter_v4_gq_om5 && $JOBS onemeter meter_v4_gq 0.6 meter_v4_gq_om6"
cons=$ID
echo "F: one meter per piece (meter_v4_gq_om5, _om6): job $cons"
submit r2train "${G[@]}" -c 16 --mem=96G -t 5:00:00 "--dependency=$(after "$tests" q1short)" --wrap="source pace/common.sh; MAX_HOURS=4.5 $JOBS meters meter_v4_gq_p12k meter_v4_gq_rep_p12k meter_v4_gq_tol_p12k && $JOBS trackers meter_v4_gq_p12k meter_v4_gq_rep_p12k meter_v4_gq_tol_p12k"
train=$ID
echo "G, H: meter_v4_gq_rep_p12k, meter_v4_gq_tol_p12k (12k steps): job $train"
SPECS="nl_gq+meter_v4_gq:tidy,d95,missing nl_gq+meter_v4_gq_om5:tidy,d95,missing nl_gq+meter_v4_gq_om6:tidy,d95,missing"
submit r2asap "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=$(after "$cons" q1asap)" --wrap="source pace/common.sh; $JOBS systems $SPECS"
asap=$ID
echo "F on whole pieces, ASAP: job $asap"
submit r2gq "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=$(after "$cons" q1gq)" --wrap="source pace/common.sh; $GQSYS $JOBS systems $SPECS"
gq=$ID
echo "F on whole pieces, Gangqinpu: job $gq"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
prev=$(queued q1final)
submit r2final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq:$train:$cons${prev:+:$prev}" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''; python3 scripts/pilot_report.py"
echo "Beyer & Dai's metrics, paired comparisons, outputs/pilot/report.md (sections F-H): job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
