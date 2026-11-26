#!/bin/bash
# v2, ninth round: PM2S retrained with stable optimisation (on the login node from the repo root,
# after git pull, with the queue empty: 8 jobs):
#   bash pace/v2_pm2s2.sh
#
# Why: the seventh round's PM2S without our test pieces (pm2s_ho) trained unstably with PM2S's own
# recipe (AdamW 1e-3, betas 0.8 / 0.8, no clipping): its training loss rose from step 8,000 to
# 38,000 and validation fell with it (best at 10,000). Even under-trained it made v2 for MIDI input
# level with seq2seq and Beyer & Dai on clean MIDI (meter_v4_gq_pmho: 11.85 against 11.74 / 11.05),
# while the demo renders a MIDI upload for Beat This!, about 3 points worse there. This round:
#   - PM2S trained twice with stable settings (AdamW betas 0.9 / 0.999, eps 1e-8, clipping 1.0,
#     warmup 1000, cosine decay): learning rate 3e-4 (pm2s_ho2) and 1e-4 (pm2s_ho3); the better by
#     its own validation score is linked as pm2s_hos (scripts/pick_run.py) and used with tag hos;
#   - its beats against the released PM2S, the first retraining and Beat This!; the release with it;
#   - meter_v4_gq_pmhs, the meter model trained for it; v2 with it and nl_gq (the default labeller)
#     on whole pieces, ASAP and Gangqinpu; on Gangqinpu also the default with Beat This! on the
#     MIDI rendered to audio (_rsb), which is what a MIDI upload gets today.
#
#   job         what                                                              waits for
#   p2tests     the test suites, on PACE's Python (CPU)                            -
#   p2train_a   pm2s_ho2 (lr 3e-4) trained, up to 7.5 h                            p2tests (ok)
#   p2train_b   pm2s_ho3 (lr 1e-4) trained, up to 7.5 h                            p2tests (ok)
#   p2track     the better one linked as pm2s_hos; its beats stored for training   both (any)
#               and testing, its errors fitted (configs/beat_noise_pm2s_hos.json),
#               its beats on the ASAP test inputs (beats_compare_hos.md)
#   p2sys       the release with its beats (_pm_rpm_hos), meter_v4_gq_pm with      p2track (ok)
#               them; on Gangqinpu, nl_gq + meter_v4_gq on the rendered MIDI (_rsb)
#   p2meter     meter_v4_gq_pmhs trained, tuned and evaluated                      p2track (ok)
#   p2eval      meter_v4_gq_pmhs with its beats on every test set, nl_gq on its    p2meter (any)
#               grids, whole pieces (nl_gq + meter_v4_gq_pmhs:tidy,d95,pm,rpm),
#               ASAP and Gangqinpu
#   p2final     outputs/compare, Beyer & Dai's metrics for every system on both    p2sys, p2eval (any)
#               test sets, the paired comparisons
#
# Again after a stop (finished runs are kept):
#   STAGE=meter bash pace/v2_pm2s2.sh   # meter_v4_gq_pmhs not done: p2meter2, p2eval, p2final
#   STAGE=eval bash pace/v2_pm2s2.sh    # only p2eval and p2final
# If a training job stopped before the end: bash pace/v2_pm2s2.sh again (both resume from last.pt).
set -euo pipefail
GPUS=${GPUS:-cuda13}  # any NVIDIA GPU with bf16 (pace/README.md, "Which GPU")
G=(--gres=gpu:1 "--constraint=$GPUS")
PM_RUN=${PM_RUN:-pm2s_hos}                 # the name the better retraining is linked under
PM_TAG=${PM_TAG:-hos}
METER=${METER:-meter_v4_gq_pmhs}
NL=${NL:-nl_gq}
BASE_NL=${BASE_NL:-nl_base_gn}
GQ=${GQ:-data/raw/gangqinpu}
BD=${BD:-$HOME/scratch/beyer_dai}
STAGE=${STAGE:-all}   # all; meter: meter training resumed, then p2eval + p2final; eval: p2eval + p2final
case "$STAGE" in all|meter|eval) ;; *) echo "STAGE must be all, meter or eval, not $STAGE"; exit 1 ;; esac
for f in data/processed/meter/train_gangqinpu.pkl data/processed/meter/val_gangqinpu.pkl \
         outputs/runs/meter_v3_tp/decode.json outputs/runs/meter_v4_gq/best.pt outputs/runs/meter_v4_gq_pm/best.pt \
         "outputs/runs/$NL/best.pt" "outputs/runs/$BASE_NL/best.pt" outputs/system_eval/manifest.json \
         outputs/system_eval_gangqinpu/manifest.json outputs/system_eval_gangqinpu/beats_gangqinpu.npz; do
  [ -e "$f" ] || { echo "no $f: run pace/v2_gangqinpu.sh, pace/v2_gqlabel.sh (and the rounds before) first"; exit 1; }
done
[ -n "$(ls "$GQ"/data/*.parquet 2>/dev/null)" ] || { echo "no $GQ/data/*.parquet: run pace/v2_gangqinpu.sh first"; exit 1; }
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: run bash pace/bd_setup.sh first"; exit 1; }
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
# every job that runs PM2S gets the picked weights and their tag
PMENV="source pace/common.sh; export PM2S_WEIGHTS=\"\$SLURM_SUBMIT_DIR/outputs/runs/$PM_RUN/best.pt\" PM2S_TAG=$PM_TAG"
JOBS="bash pace/v2_jobs.sh"
GQOUT=outputs/system_eval_gangqinpu
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
STABLE="--betas 0.9,0.999 --eps 1e-8 --clip 1.0 --schedule cosine --warmup 1000 --steps 60000"
n=0
ID=""
submit() {  # name, then sbatch options and --wrap: the job id in $ID, counted in $n
  local name=$1
  shift
  local kill=()
  case " $* " in *" --dependency=afterok:"*) kill=(--kill-on-invalid-dep=yes) ;; esac
  ID=$(sbatch --parsable -J "$name" -o "slurm-$name-%j.out" ${kill[@]+"${kill[@]}"} "$@")
  n=$((n + 1))
}

sys=""
meter2=""
if [ "$STAGE" = all ]; then
submit p2tests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; python -m tests.test_pm2s_train && python -m tests.test_pm2s && python -m tests.test_tracked && python -m tests.test_meter && python -m tests.test_v2_system && python -m tests.test_system_eval && python -m tests.test_compare_runs && python -m tests.test_glissando"
tests=$ID
echo "tests: job $tests"
trains=""
for spec in "pm2s_ho2 3e-4 a" "pm2s_ho3 1e-4 b"; do
  set -- $spec
  resume=""
  [ -f "outputs/runs/$1/last.pt" ] && resume="--resume"
  submit "p2train_$3" "${G[@]}" -c 8 --mem=64G -t 8:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; python scripts/train_pm2s.py --name $1 --lr $2 $STABLE --normal-priority --max-hours 7.5 $resume"
  trains="$trains:$ID"
  echo "$1 (lr $2, stable settings) trained: job $ID"
done
submit p2track "${G[@]}" -c 8 --mem=64G -t 8:00:00 "--dependency=afterany$trains" --wrap="$PMENV; python scripts/pick_run.py --runs pm2s_ho2 pm2s_ho3 --as $PM_RUN && python scripts/track_meter_data.py --release $GQ --trackers pm2s && python scripts/fit_pm2s_noise.py && python scripts/compare_beats.py"
track=$ID
echo "the better one picked as $PM_RUN (tag $PM_TAG), its beats stored, its errors fitted, its beats on the ASAP test inputs: job $track"
submit p2sys "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$track" --wrap="$PMENV; $JOBS systems $BASE_NL+meter_v3_tp:tidy,d95,pm,rpm; PM2S_RUNS=\"meter_v4_gq_pm\" $JOBS trackers meter_v4_gq_pm; $GQSYS $JOBS systems $NL+meter_v4_gq:tidy,d95,rsb"
sys=$ID
echo "the release with its beats, meter_v4_gq_pm with them, the default on Gangqinpu's rendered MIDI: job $sys"
submit p2meter "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$track" --wrap="$PMENV; $JOBS meters $METER"
meter2=$ID
echo "$METER trained, tuned and evaluated: job $meter2"
elif [ "$STAGE" = meter ]; then
submit p2meter2 "${G[@]}" -c 16 --mem=96G -t 8:00:00 --wrap="$PMENV; $JOBS meters $METER"
meter2=$ID
echo "$METER (resumed), tuned and evaluated: job $meter2"
fi
dep=()
[ -n "$meter2" ] && dep=("--dependency=afterany:$meter2")
submit p2eval "${G[@]}" -c 8 --mem=48G -t 8:00:00 ${dep[@]+"${dep[@]}"} --wrap="$PMENV; PM2S_RUNS=\"$METER\" $JOBS trackers $METER; METERS=\"$METER\" GRIDS_OLD=\"\" NEW_NLS=\"\" EVAL_CONDITIONS=\"clean,real,periscope\" $JOBS evals $NL; $JOBS systems $NL+$METER:tidy,d95,pm,rpm; $GQSYS $JOBS systems $NL+$METER:tidy,d95,pm,rpm"
evl=$ID
echo "$METER with its beats, $NL on its grids, whole pieces (ASAP, Gangqinpu): job $evl"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
submit p2final -c 16 --mem=64G -t 6:00:00 "--dependency=afterany:${sys:+$sys:}$evl" --wrap="source pace/common.sh; python3 scripts/compare_runs.py --quiet; $BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''"
echo "outputs/compare, Beyer & Dai's metrics (ASAP, Gangqinpu), paired comparisons: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
