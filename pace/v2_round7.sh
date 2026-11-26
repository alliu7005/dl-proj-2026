#!/bin/bash
# v2, round 7: is round 6's best system's gain real, and can it be pushed further? (on the login node
# from the repo root, after git pull, once pace/v2_gangqinpu.sh has finished):
#   bash pace/v2_round7.sh
#   TRAIN_CPUS=24 bash pace/v2_round7.sh     # more cores for the meter trainings
#
# Round 6's best whole-piece system was nl_base_gn + meter_v4_gq_pm with PM2S's beats (MUSTER 11.05
# real / 11.56 clean against the release's 12.31 / 14.53). From one seed of each model, without
# paired intervals, and with the release's own Beat This! row missing from this account's results.
# Round 7 measures it properly and tries the three changes the results point to:
#   - meter_v4_gq_pm_s1: its second seed (seeds of meter_v3 differ by up to 0.05 downbeat F / 0.06
#     time signature): is the gain more than chance?
#   - meter_v4_gq_pm_g30: Gangqinpu 0.3 of every batch instead of 0.5 (its 81% 4/4 pulled the
#     classical time signatures towards 4/4: meter_v4_gq lost 0.03-0.08 there)
#   - nl_base_gn_pm: the labeller trained on PM2S's timing errors instead of Beat This!'s (the system
#     reads PM2S's beats; matching the labeller's grid noise to the tracker was round 3's largest gain)
# and fills the gaps: the release with Beat This!'s beats on whole pieces, meter_v4_gq_posh on whole
# pieces and with Beat This!'s real beats of Gangqinpu's recordings. The last job writes
# outputs/round7/report.md (scripts/analyze_round7.py): every meter model per condition, Gangqinpu
# by genre, the time-signature mistakes (how many wrong bars became 4/4), the labellers on every
# grid, every whole-piece system, and round 7's paired comparisons with bootstrap intervals.
#
#   job         what                                                              waits for
#   r7tests     the test suites, on PACE's Python (CPU)                            -
#   r7base      whole pieces: the release (Beat This!), meter_v4_gq_posh; posh     r7tests (ok)
#               with Beat This!'s real beats of Gangqinpu's recordings
#   r7meters    meter_v4_gq_pm_s1 and meter_v4_gq_pm_g30 side by side, up to 7.5 h  r7tests (ok)
#   r7meters2   resumed where they stopped, tuned and evaluated                    r7meters (any)
#   r7nl        nl_base_gn_pm trained, evaluated on the true grid                  r7tests (ok)
#   r7eval      the new meters with PM2S's beats; both labellers on every new      r7meters2, r7nl (any)
#               grid; whole pieces: the new meters, nl_base_gn_pm on meter_v4_gq_pm
#               and on the release's meter model
#   r7final     every system rated with Beyer & Dai's metrics, outputs/compare,    r7base, r7eval (any)
#               outputs/round7/report.md
#
# Again after a stop (finished runs and systems are kept):
#   STAGE=eval bash pace/v2_round7.sh    # only r7eval + r7final (trainings finished or not)
set -euo pipefail
GPUS=${GPUS:-cuda13}  # any NVIDIA GPU with bf16 (pace/README.md, "Which GPU")
G=(--gres=gpu:1 "--constraint=$GPUS")
TRAIN_CPUS=${TRAIN_CPUS:-16}   # r7meters*: 2 meter runs side by side, 6 sample workers each
BD=${BD:-$HOME/scratch/beyer_dai}
STAGE=${STAGE:-all}
case "$STAGE" in all|eval) ;; *) echo "STAGE must be all or eval, not $STAGE"; exit 1 ;; esac
NEW_METERS="meter_v4_gq_pm_s1 meter_v4_gq_pm_g30"
NL=nl_base_gn_pm
for r in meter_v3_tp meter_v4_gq meter_v4_gq_pm meter_v4_gq_posh nl_base_gn; do
  [ -f "outputs/runs/$r/best.pt" ] || { echo "outputs/runs/$r/best.pt missing: run pace/v2_gangqinpu.sh first"; exit 1; }
done
for f in outputs/system_eval/manifest.json configs/beat_noise_pm2s.json "$BD/MIDI2ScoreTF.ckpt" \
         data/processed/meter/tracked_train_gangqinpu_pm2s.npz; do
  [ -s "$f" ] || { echo "$f missing: run pace/v2_gangqinpu.sh (and pace/bd_setup.sh) first"; exit 1; }
done
JOBS="source pace/common.sh; bash pace/v2_jobs.sh"
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
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

base=""
deps_eval=""
if [ "$STAGE" = all ]; then
submit r7tests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; python -m tests.test_meter && python -m tests.test_v2_system && python -m tests.test_system_eval && python -m tests.test_gangqinpu && python -m tests.test_compare_runs"
tests=$ID
echo "tests: job $tests"
submit r7base "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$JOBS systems nl_base_gn+meter_v3_tp:tidy,d95 nl_base_gn+meter_v4_gq_posh:tidy,d95; PM2S_RUNS=\"\" bash pace/v2_jobs.sh trackers meter_v4_gq_posh"
base=$ID
echo "whole pieces: the release (Beat This!) + meter_v4_gq_posh; posh with Beat This!'s real beats: job $base"
submit r7meters "${G[@]}" -c "$TRAIN_CPUS" --mem=96G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$JOBS meters $NEW_METERS"
m1=$ID
submit r7meters2 "${G[@]}" -c "$TRAIN_CPUS" --mem=96G -t 8:00:00 "--dependency=afterany:$m1" --wrap="$JOBS meters $NEW_METERS"
m2=$ID
echo "train + tune + evaluate $NEW_METERS: jobs $m1, then $m2 (resumes)"
submit r7nl "${G[@]}" -c 8 --mem=64G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$JOBS labellers $NL"
nl=$ID
echo "labeller $NL (PM2S's timing errors) trained + evaluated on the true grid: job $nl"
deps_eval="--dependency=afterany:$m2:$nl"
fi
specs="nl_base_gn+meter_v4_gq_pm_s1:tidy,d95,pm,rpm nl_base_gn+meter_v4_gq_pm_g30:tidy,d95,pm,rpm $NL+meter_v4_gq_pm:tidy,d95,pm,rpm $NL+meter_v3_tp:tidy,d95,pm,rpm"
wrap="source pace/common.sh; PM2S_RUNS=\"$NEW_METERS\" bash pace/v2_jobs.sh trackers $NEW_METERS; \
METERS=\"meter_v4_gq_pm $NEW_METERS\" GRIDS_OLD=\"\" NEW_NLS=\"$NL\" bash pace/v2_jobs.sh evals $NL; \
METERS=\"$NEW_METERS meter_v4_gq_posh\" GRIDS_OLD=\"\" NEW_NLS=\"\" bash pace/v2_jobs.sh evals nl_base_gn; \
$JOBS systems $specs"
dep=()
[ -n "$deps_eval" ] && dep=("$deps_eval")
submit r7eval "${G[@]}" -c 8 --mem=48G -t 8:00:00 ${dep[@]+"${dep[@]}"} --wrap="$wrap"
evl=$ID
echo "the new meters with PM2S's beats, both labellers on the new grids, whole pieces ($specs): job $evl"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16"
submit r7final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:${base:+$base:}$evl" --wrap="source pace/common.sh; python3 scripts/compare_runs.py --quiet; $BDENV; $SCORE --systems all; conda activate dl-proj; python scripts/analyze_round7.py"
echo "Beyer & Dai's metrics for every system + outputs/round7/report.md: job $ID ($n jobs)"
echo "watch: squeue -u \$USER     then: less outputs/round7/report.md"
