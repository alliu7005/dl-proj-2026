#!/bin/bash
# v2, seventh round: PM2S retrained without our test pieces (on the login node from the repo root,
# after git pull and after pace/v2_gangqinpu.sh):
#   bash pace/v2_pm2s.sh
#
# Why: with PM2S's beats (beats from the notes) v2 gained the most of any change so far (sixth
# round: the release -2.10 MUSTER on clean MIDI, -0.74 on real recordings; meter_v4_gq_pm with them
# within chance of seq2seq and Beyer & Dai). But PM2S's released weights were trained on all of
# ASAP, our 14 test pieces included. scripts/train_pm2s.py trains its model again, from scratch, by
# its own recipe, on our training splits only (ASAP's training pieces, their transcriptions,
# PERiScoPe, Gangqinpu): pm2s_ho ("held out"). Everything made with it carries the tag ho
# (PM2S_WEIGHTS / PM2S_TAG; src/pm2s_beats.py weights_tag), next to the released weights' results:
#   - its beats against the annotations, with the released PM2S's and Beat This!'s
#     (beats_compare_ho.md on the ASAP test inputs, tracked_report_ho.json on PERiScoPe and
#     Gangqinpu, against tracked_report.json);
#   - the release and meter_v4_gq_pm with its beats on whole pieces (_pm_rpm_ho): the sixth round's
#     gain, without the leak;
#   - meter_v4_gq_pmho, meter_v4_gq_pm trained again for its beats (their errors fitted on
#     Gangqinpu, its real beats in half of the draws), on whole pieces with them.
#
#   job         what                                                              waits for
#   phtests     the test suites, on PACE's Python (CPU)                            -
#   phtrain     pm2s_ho trained (scripts/train_pm2s.py, up to 7.5 h; about 2 h     phtests (ok)
#               expected); fails unless done
#   phtrack     PM2S (pm2s_ho) on Gangqinpu's and PERiScoPe's notes, stored for    phtrain (ok)
#               training and testing (track_meter_data.py --trackers pm2s); its
#               errors fitted (fit_pm2s_noise.py -> configs/beat_noise_pm2s_ho.json);
#               its beats on the ASAP test inputs (compare_beats.py)
#   phsys       the release and meter_v4_gq_pm with pm2s_ho's beats: whole pieces  phtrain (ok)
#               (:tidy,d95,pm,rpm) and meter_v4_gq_pm at the meter level
#               (<condition>_pm2s_ho)
#   phmeter,    meter_v4_gq_pmho trained (resuming), tuned and evaluated           phtrack (ok);
#   phmeter2                                                                       phmeter (any)
#   pheval      meter_v4_gq_pmho with pm2s_ho's beats on every test set,           phmeter2 (any)
#               nl_base_gn on its grids, whole pieces (:tidy,d95,pm,rpm)
#   phfinal     outputs/compare, every system rated with Beyer & Dai's metrics,    phsys, pheval (any)
#               the paired comparisons (outputs/system_eval/paired.md)
#
# Again after a stop (finished runs are kept):
#   STAGE=meter bash pace/v2_pm2s.sh   # meter_v4_gq_pmho not done after phmeter2: phmeter3, pheval, phfinal
#   STAGE=eval bash pace/v2_pm2s.sh    # only pheval and phfinal again
# If phtrain stopped before the end: bash pace/v2_pm2s.sh again (it resumes from last.pt).
set -euo pipefail
GPUS=${GPUS:-cuda13}  # any NVIDIA GPU with bf16 (pace/README.md, "Which GPU")
G=(--gres=gpu:1 "--constraint=$GPUS")
PM_RUN=${PM_RUN:-pm2s_ho}
PM_TAG=${PM_TAG:-ho}
METER=${METER:-meter_v4_gq_pmho}
BASE=${BASE-"meter_v3_tp meter_v4_gq_pm"}   # meter models already trained, tried with the new beats
LABELLER=${LABELLER:-nl_base_gn}
GQ=${GQ:-data/raw/gangqinpu}
BD=${BD:-$HOME/scratch/beyer_dai}
STAGE=${STAGE:-all}   # all; meter: meter training resumed, then pheval + phfinal; eval: pheval + phfinal
case "$STAGE" in all|meter|eval) ;; *) echo "STAGE must be all, meter or eval, not $STAGE"; exit 1 ;; esac
for f in data/processed/meter/train_gangqinpu.pkl data/processed/meter/val_gangqinpu.pkl \
         outputs/runs/meter_v3_tp/decode.json outputs/runs/meter_v4_gq_pm/best.pt \
         "outputs/runs/$LABELLER/best.pt" outputs/system_eval/manifest.json; do
  [ -e "$f" ] || { echo "no $f: run pace/v2_gangqinpu.sh (and the rounds before it) first"; exit 1; }
done
[ -n "$(ls "$GQ"/data/*.parquet 2>/dev/null)" ] || { echo "no $GQ/data/*.parquet: run pace/v2_gangqinpu.sh first"; exit 1; }
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: run bash pace/bd_setup.sh first"; exit 1; }
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
# every job that runs PM2S gets the new weights and their tag
PMENV="source pace/common.sh; export PM2S_WEIGHTS=\"\$SLURM_SUBMIT_DIR/outputs/runs/$PM_RUN/best.pt\" PM2S_TAG=$PM_TAG"
JOBS="bash pace/v2_jobs.sh"
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
submit phtests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; python -m tests.test_pm2s_train && python -m tests.test_pm2s && python -m tests.test_tracked && python -m tests.test_meter && python -m tests.test_v2_system && python -m tests.test_system_eval && python -m tests.test_compare_runs && python -m tests.test_notelabels"
tests=$ID
echo "tests: job $tests"
resume=""
[ -f "outputs/runs/$PM_RUN/last.pt" ] && resume="--resume"
submit phtrain "${G[@]}" -c 8 --mem=64G -t 8:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; python scripts/train_pm2s.py --name $PM_RUN --normal-priority --max-hours 7.5 $resume && grep -q '\"phase\": \"done\"' outputs/runs/$PM_RUN/status.json"
train=$ID
echo "$PM_RUN trained (scripts/train_pm2s.py): job $train"
submit phtrack "${G[@]}" -c 8 --mem=64G -t 8:00:00 "--dependency=afterok:$train" --wrap="$PMENV; python scripts/track_meter_data.py --release $GQ --trackers pm2s && python scripts/fit_pm2s_noise.py && python scripts/compare_beats.py"
track=$ID
echo "$PM_RUN's beats stored (tracked_*_pm2s_$PM_TAG.npz), its errors fitted (configs/beat_noise_pm2s_$PM_TAG.json), its beats on the ASAP test inputs: job $track"
specs=""
for m in $BASE; do specs="$specs $LABELLER+$m:tidy,d95,pm,rpm"; done
submit phsys "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$train" --wrap="$PMENV; $JOBS systems$specs; PM2S_RUNS=\"meter_v4_gq_pm\" $JOBS trackers meter_v4_gq_pm"
sys=$ID
echo "whole pieces with $PM_RUN's beats ($specs ), meter_v4_gq_pm with them: job $sys"
submit phmeter "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$track" --wrap="$PMENV; $JOBS meters $METER"
meter=$ID
submit phmeter2 "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterany:$meter" --wrap="$PMENV; $JOBS meters $METER"
meter2=$ID
echo "$METER trained, tuned and evaluated: jobs $meter, then $meter2 (resumes)"
elif [ "$STAGE" = meter ]; then
submit phmeter3 "${G[@]}" -c 16 --mem=96G -t 8:00:00 --wrap="$PMENV; $JOBS meters $METER"
meter2=$ID
echo "$METER (resumed), tuned and evaluated: job $meter2"
fi
dep=()
[ -n "$meter2" ] && dep=("--dependency=afterany:$meter2")
submit pheval "${G[@]}" -c 8 --mem=48G -t 8:00:00 ${dep[@]+"${dep[@]}"} --wrap="$PMENV; PM2S_RUNS=\"$METER\" $JOBS trackers $METER; METERS=\"$METER\" GRIDS_OLD=\"\" NEW_NLS=\"\" $JOBS evals $LABELLER; $JOBS systems $LABELLER+$METER:tidy,d95,pm,rpm"
evl=$ID
echo "$METER with $PM_RUN's beats, $LABELLER on its grids, whole pieces: job $evl"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16"
submit phfinal -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:${sys:+$sys:}$evl" --wrap="source pace/common.sh; python3 scripts/compare_runs.py --quiet; $BDENV; $SCORE --systems all; python3 scripts/compare_systems.py"
echo "outputs/compare + every system rated with Beyer & Dai's metrics + paired comparisons: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
