#!/bin/bash
# v2 eleventh round: compound meters in the meter model's training (on the login node from the repo
# root, after git pull; 6 jobs, alongside the tenth round's batches: their scoring jobs still queued
# are waited for, found by name).
#   bash pace/v2_compound.sh
#
# The user's "unlasting" (6/8) came out 3/4 with stretches at three different written tempi: Beat
# This! switched between a dotted quarter, two-thirds of one (the bar counted in three) and the
# eighths, and the meter model followed it. On Gangqinpu's real recordings meter_v4_gq gets 6/8
# right on 49% of the songs (12/8 on none); the training's simulated tracker had no 2/3 level and
# one octave error a piece at most. Two fixes, measured apart and together (meter_v4_gq's recipe
# otherwise):
#   meter_v5_lv     the simulated tracker's level per stretch fitted on the real tracker's beats
#                   (src/beats.py level_runs: ASAP's recordings -> configs/beat_noise_levels.json,
#                   Gangqinpu's -> configs/beat_noise_bt_gangqinpu_levels.json), by meter kind, and
#                   replayed: several switches a piece, the 2/3 level
#   meter_v5_cw     compound meters drawn 3 times as often (--compound-weight 3)
#   meter_v5_lvcw   both
#
#   job        what                                                                 waits for
#   cmfit      the tests; both level fits (CPU)                                     -
#   cmtrain    the three meters, side by side on one GPU                            cmfit (ok)
#   cmtrain2   resumes what is not finished; tuned, evaluated (published if         cmtrain (any)
#              better: v2_jobs.sh meters); on the real recordings (trackers)
#   cmasap     whole pieces on ASAP: nl_gq + each new meter                         cmtrain2 (any)
#   cmgq       the same on Gangqinpu's whole pieces                                 the same
#   cmfinal    Beyer & Dai's metrics, both paired.md, outputs/meter_ts.md (per      cmasap, cmgq; the earlier
#              true time signature)                                                  batches' scoring (any)
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
for f in outputs/runs/nl_gq/best.pt outputs/runs/meter_v4_gq/eval_meter.json data/processed/meter/train.pkl \
         data/processed/meter/train_gangqinpu.pkl outputs/system_eval/manifest.json \
         outputs/system_eval_gangqinpu/manifest.json; do
  [ -e "$f" ] || { echo "no $f: run pace/v2_gqlabel.sh (and the rounds before) first"; exit 1; }
done
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: run bash pace/bd_setup.sh first"; exit 1; }
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
JOBS="bash pace/v2_jobs.sh"
GQOUT=outputs/system_eval_gangqinpu
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
RUNS="meter_v5_lv meter_v5_cw meter_v5_lvcw"
SPECS=""
for m in $RUNS; do SPECS="$SPECS nl_gq+$m:tidy,d95,missing"; done
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
submit cmfit -c 8 --mem=64G -t 3:00:00 --wrap="source pace/common.sh; python -m tests.test_beat_levels && python -m tests.test_meter && python -m tests.test_piece_tempo && python scripts/fit_beat_noise.py --levels && python scripts/fit_pm2s_noise.py --from-stored beat_this --levels && ls -la configs/beat_noise_levels.json configs/beat_noise_bt_gangqinpu_levels.json"
fit=$ID
echo "tests and the level fits: job $fit"
submit cmtrain "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$fit" --wrap="source pace/common.sh; $JOBS meters $RUNS"
train=$ID
echo "the three meters: job $train"
submit cmtrain2 "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterany:$train" --wrap="source pace/common.sh; $JOBS meters $RUNS; $JOBS trackers $RUNS"
train2=$ID
echo "resumed if needed, evaluated, on the real recordings: job $train2"
submit cmasap "${G[@]}" -c 8 --mem=48G -t 10:00:00 "--dependency=afterany:$train2" --wrap="source pace/common.sh; $JOBS systems$SPECS"
asap=$ID
echo "whole pieces, ASAP (3 systems): job $asap"
submit cmgq "${G[@]}" -c 8 --mem=48G -t 10:00:00 "--dependency=afterany:$train2" --wrap="source pace/common.sh; $GQSYS $JOBS systems$SPECS"
gq=$ID
echo "whole pieces, Gangqinpu (3 systems): job $gq"
others=$(queued dcfinal d2final ptfinal)
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
submit cmfinal -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq${others:+:$others}" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''; python3 scripts/meter_ts_report.py --runs meter_v4_gq,$(echo $RUNS | tr ' ' ',')"
echo "Beyer & Dai's metrics, paired comparisons, the time-signature report: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
