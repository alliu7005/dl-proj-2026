#!/bin/bash
# v2, sixth round: Gangqinpu as meter-model data, and the grid's beat times taken from the notes
# (on the login node from the repo root, after git pull; after pace/v2_level.sh or alongside it):
#   bash pace/v2_gangqinpu.sh
#
# Once per account, before the first run: accept the dataset's licence at
# https://huggingface.co/datasets/Wayne42/gangqinpu, then log in with the cache the jobs use:
#   HF_HOME=$HOME/scratch/.cache/huggingface hf auth login
#
# 1. Gangqinpu (src/preprocessing/gangqinpu.py): ~18k web piano recordings of many genres, Transkun
#    V2 transcriptions aligned to their scores, the score's beats timed through the alignment. The
#    meter model looks data-limited (the larger model overfit, seeds differ by 0.07 in time
#    signature) and its largest error, the note-value level, depends on the piece. Two runs:
#    meter_v4_gq (meter_v3's recipe, Gangqinpu half of every batch) and meter_v3_gqft (meter_v3
#    fine-tuned on that mix); both decoded like the release (tempo prior), on every test set and
#    Gangqinpu's own (music outside the classical benchmark), meter_v3_tp scored there too.
# 2. Beat times from the notes (src/meter.py refine_grid): the anchors snapped onto chord onsets
#    (SNAP s) and the beats the tracker missed added at chord onsets (FILL beats). The oracle grid
#    loses 0.16 onset F1 to the tracker's timing (0.857 -> 0.697); this shows how much the notes
#    give back, on the oracle and on the release's meter model, with the fill alone too.
# 3. PM2S (src/pm2s_beats.py): beats from the notes instead of Beat This!. Its beats scored against
#    the annotations next to Beat This!'s (scripts/compare_beats.py), then the release's v2 on whole
#    pieces with them, on clean MIDI and on the real transcriptions (:pm,rpm). Its released weights
#    were trained on all of ASAP: optimistic until it is retrained without our test pieces.
# 4. Beat This!, PM2S, or both: three meter models alike but for the tracker(s) they read:
#    meter_v4_gq (Beat This!), meter_v4_gq_pm (PM2S), meter_v4_gq_both (Beat This!, with PM2S's
#    beats as evidence on them: train_meter.py --aux pm2s). Every one mixes its trackers' real beats
#    into training (scripts/track_meter_data.py: Beat This! on Gangqinpu's recordings, downloaded
#    here by scripts/fetch_gangqinpu_audio.py; PM2S on Gangqinpu's and PERiScoPe's notes; half of
#    the draws that have them) with their simulated errors in the rest, fitted, not guessed:
#    Beat This!'s on ASAP (configs/beat_noise.json) and, for Gangqinpu, on its recordings
#    (configs/beat_noise_bt_gangqinpu.json, fitted in gqtrack: its off-beat rate on this music
#    included); PM2S's on Gangqinpu (configs/beat_noise_pm2s.json). Each is tested with its own
#    tracker(s): at the meter level (Gangqinpu's test recordings with Beat This!'s real beats,
#    gangqinpu_real; every test set with PM2S's, <condition>_pm2s) and on whole pieces
#    (:tidy,d95; :tidy,d95,pm,rpm for PM2S). tracked_report.json compares the two trackers' beats
#    on the same recordings.
#
# The recordings belong to the source site and its uploaders (the dataset licence covers its
# annotations): fetched politely (1 request/s), for non-commercial research, never redistributed.
# AUDIO_TRAIN training recordings are fetched (with every validation and test one); 0 = none.
#
#   job         what                                                              waits for
#   gqtests      the test suites, on PACE's Python (CPU)                            -
#   gqdata       Gangqinpu downloaded (scripts/fetch_gangqinpu.py), its meter       gqtests (ok)
#                data built (build_meter_data.py --gangqinpu-only), PM2S's errors
#                fitted on it (configs/beat_noise_pm2s.json), meter_v3_tp evaluated
#                on its test set (the baseline there); then the recordings
#                downloaded (fetch_gangqinpu_audio.py; a failed file is skipped)
#   gqtrack      Beat This! on the recordings, PM2S on the notes: their beats       gqdata (ok)
#                stored for training and testing (track_meter_data.py); Beat This!'s
#                errors on Gangqinpu fitted from them (fit_pm2s_noise.py --from-stored)
#   gqrefine     meter_v3_tp_rf (snap + fill), meter_v3_tp_fill (fill only) and     gqtests (ok)
#                meter_oracle_rf; nl_base_gn on their grids; v2 on whole pieces
#                with the two refined decoders (:tidy,d95, as the release); PM2S's
#                beats against Beat This!'s, and the release with PM2S's beats
#   gqtrain      meter_v4_gq, meter_v4_gq_pm and meter_v4_gq_both side by side, up  gqtrack (ok)
#                to 7.5 h; then
#   gqtrain2     resumed where they stopped, tuned and evaluated                    gqtrain (any)
#   gqeval       the new meter models and meter_v3_tp with their real trackers      gqtrain2 (any)
#                (v2_jobs.sh trackers); nl_base_gn on the new grids; v2 on whole
#                pieces with each new model and its tracker(s)
#   gqfinal      outputs/compare, every system rated with Beyer & Dai's metrics,    gqrefine, gqeval (any)
#                the paired comparisons (outputs/system_eval/paired.md)
#
# Again after a stop (finished runs and downloads are kept either way):
#   STAGE=train bash pace/v2_gangqinpu.sh   # training not done after gqtrain2: gqtrain3 (resumes), gqeval, gqfinal
#   STAGE=eval bash pace/v2_gangqinpu.sh    # only the evaluation and the final scoring again
#
# Data built elsewhere and copied in (Gangqinpu's meter data, both trackers' stored beats, their
# fitted profiles: what gqdata + gqtrack make), so neither runs again and nothing is downloaded:
#   STAGE=prebuilt bash pace/v2_gangqinpu.sh   # gqtests, gqrefine (+ BASE on Gangqinpu's test set), gqtrain, ...
set -euo pipefail
GPUS=${GPUS:-cuda13}  # any NVIDIA GPU with bf16 (pace/README.md, "Which GPU")
G=(--gres=gpu:1 "--constraint=$GPUS")
RUNS=${RUNS-"meter_v4_gq meter_v4_gq_pm meter_v4_gq_both"}   # Beat This!, PM2S, both
PM2S_RUNS=${PM2S_RUNS-"meter_v4_gq_pm"}                      # the ones that read PM2S alone
BASE=${BASE-"meter_v3_tp"}        # the release's meter model and decoder
LABELLER=${LABELLER-"nl_base_gn"}
SNAP=${SNAP:-0.05}
FILL=${FILL:-0.25}
GQ=${GQ:-data/raw/gangqinpu}
AUDIO_TRAIN=${AUDIO_TRAIN:-3000}  # training recordings to download (with all ~1,000 validation and test ones)
BD=${BD:-$HOME/scratch/beyer_dai}
STAGE=${STAGE:-all}   # all; train: training resumed (one more job), then gqeval + gqfinal; eval: gqeval + gqfinal
case "$STAGE" in all|prebuilt|train|eval) ;; *) echo "STAGE must be all, prebuilt, train or eval, not $STAGE"; exit 1 ;; esac
if [ "$STAGE" = prebuilt ]; then   # what gqdata + gqtrack would have made
  for f in data/processed/meter/train_gangqinpu.pkl data/processed/meter/tracked_train_gangqinpu_beat_this.npz \
           data/processed/meter/tracked_train_gangqinpu_pm2s.npz configs/beat_noise_pm2s.json \
           configs/beat_noise_bt_gangqinpu.json; do
    [ -f "$f" ] || { echo "STAGE=prebuilt: $f missing (copy the prebuilt data in, or use STAGE=all)"; exit 1; }
  done
fi
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
JOBS="source pace/common.sh; bash pace/v2_jobs.sh"
REFINED="${BASE}_rf ${BASE}_fill meter_oracle_rf"
if [ ! -f "outputs/runs/meter_v3/best.pt" ] || [ ! -f "outputs/runs/$BASE/decode.json" ]; then
  echo "no outputs/runs/meter_v3/best.pt or $BASE/decode.json: run pace/v2.sh and pace/v2_round4.sh first"
  exit 1
fi
if [ ! -f outputs/system_eval/manifest.json ] || [ ! -s "$BD/MIDI2ScoreTF.ckpt" ]; then
  echo "no outputs/system_eval/manifest.json or $BD/MIDI2ScoreTF.ckpt: run bash pace/bd_setup.sh and bash pace/system_eval.sh first"
  exit 1
fi
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

refine=""
train2=""
if [ "$STAGE" = all ] || [ "$STAGE" = prebuilt ]; then
submit gqtests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; python -m tests.test_gangqinpu && python -m tests.test_pm2s && python -m tests.test_tracked && python -m tests.test_beats && python -m tests.test_meter && python -m tests.test_pipeline && python -m tests.test_v2_system && python -m tests.test_system_eval"
tests=$ID
echo "tests: job $tests"
gqbase=""
if [ "$STAGE" = prebuilt ]; then
track=$tests   # the data and the trackers' beats are there already: training waits only for the tests
gqbase="$JOBS gqbase $BASE; "   # gqdata's baseline: the release's meter model on Gangqinpu's test set
echo "prebuilt data: no gqdata / gqtrack jobs"
else
submit gqdata -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; python scripts/fetch_gangqinpu.py --out $GQ && python scripts/build_meter_data.py --gangqinpu $GQ --gangqinpu-only && python scripts/fit_pm2s_noise.py && bash pace/v2_jobs.sh gqbase $BASE && \
{ python scripts/fetch_gangqinpu_audio.py --release $GQ --out $GQ/audio --train $AUDIO_TRAIN || \
  echo 'WARNING: the recordings could not all be fetched: the batch goes on (Beat This! simulated where missing)'; }"
data=$ID
echo "Gangqinpu download + meter data + PM2S's errors fitted + $BASE on its test set + recordings: job $data"
submit gqtrack "${G[@]}" -c 8 --mem=64G -t 8:00:00 "--dependency=afterok:$data" --wrap="source pace/common.sh; python scripts/track_meter_data.py --release $GQ --audio $GQ/audio && python scripts/fit_pm2s_noise.py --from-stored beat_this"
track=$ID
echo "Beat This! on the recordings, PM2S on the notes, stored for training; Beat This!'s errors on them fitted: job $track"
fi
wrap="$gqbase$JOBS refine $BASE ${BASE}_rf $SNAP $FILL; $JOBS refine $BASE ${BASE}_fill 0 $FILL; $JOBS refine oracle meter_oracle_rf $SNAP $FILL; METERS=\"$REFINED\" GRIDS_OLD=\"\" NEW_NLS=\"\" bash pace/v2_jobs.sh evals $LABELLER; $JOBS systems $LABELLER+${BASE}_rf:tidy,d95 $LABELLER+${BASE}_fill:tidy,d95; python scripts/compare_beats.py; $JOBS systems $LABELLER+$BASE:tidy,d95,pm,rpm"
submit gqrefine "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$wrap"
refine=$ID
echo "beat times from the notes (snap $SNAP s, fill $FILL beats): $REFINED, $LABELLER on them, whole pieces: job $refine"
submit gqtrain "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$track" --wrap="$JOBS meters $RUNS"
train=$ID
submit gqtrain2 "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterany:$train" --wrap="$JOBS meters $RUNS"
train2=$ID
echo "train + tune + evaluate $RUNS: jobs $train, then $train2 (resumes)"
elif [ "$STAGE" = train ]; then   # training not finished after gqtrain2: one more job, resuming
submit gqtrain3 "${G[@]}" -c 16 --mem=96G -t 8:00:00 --wrap="$JOBS meters $RUNS"
train2=$ID
echo "train (resumed) + tune + evaluate $RUNS: job $train2"
fi
sys_specs=""
for m in $RUNS; do   # each with its own tracker(s); a model that reads PM2S as evidence gets it in the pipeline
  case " $PM2S_RUNS " in *" $m "*) sys_specs="$sys_specs $LABELLER+$m:tidy,d95,pm,rpm" ;;
                         *) sys_specs="$sys_specs $LABELLER+$m:tidy,d95" ;; esac
done
wrap="source pace/common.sh; PM2S_RUNS=\"$PM2S_RUNS\" bash pace/v2_jobs.sh trackers $RUNS $BASE; METERS=\"$RUNS\" GRIDS_OLD=\"\" NEW_NLS=\"\" bash pace/v2_jobs.sh evals $LABELLER; $JOBS systems$sys_specs"
dep=()
[ -n "$train2" ] && dep=("--dependency=afterany:$train2")
submit gqeval "${G[@]}" -c 8 --mem=48G -t 8:00:00 ${dep[@]+"${dep[@]}"} --wrap="$wrap"
evl=$ID
echo "the real trackers on $RUNS $BASE, $LABELLER on the new grids, whole pieces ($sys_specs ): job $evl"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16"
submit gqfinal -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:${refine:+$refine:}$evl"   --wrap="source pace/common.sh; python3 scripts/compare_runs.py --quiet; $BDENV; $SCORE --systems all; python3 scripts/compare_systems.py"
echo "outputs/compare + every system rated with Beyer & Dai's metrics + paired comparisons: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
