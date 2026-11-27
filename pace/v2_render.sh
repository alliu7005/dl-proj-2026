#!/bin/bash
# v2, rendered beats: Beat This! on every training performance with no recording, rendered to
# audio with a sampled piano, in place of the simulated tracker (on the login node from the repo
# root, after git pull, with the queue empty: 8 jobs).
#   bash pace/v2_render.sh
#
# Why: the meter model trains on a tracker's beats, but only Gangqinpu's 3,000 downloaded recordings
# (and ASAP's real transcriptions) have real ones; everything else gets errors simulated at fitted
# rates (src/beats.py BeatNoise), not where the music makes the tracker go wrong. Renders of the
# same performances' MIDI with FluidSynth and MuseScore's piano track like the recordings
# (10 ASAP performances, held-out fold models: beat F 0.784 vs 0.793, downbeat F 0.618 vs 0.606,
# the same tempo-octave errors; src/render.py), so the renders stand in for recordings:
#   - rrcheck checks that here on more performances and every SoundFont (scripts/compare_render.py);
#   - every training performance (ASAP: its own MIDI with pedal, by its held-out fold model;
#     PERiScoPe, Gangqinpu: each segment's notes, final0) rendered and tracked, in 3 shards
#     (scripts/track_meter_data.py --trackers beat_this_render), then joined;
#   - two meter models on them: meter_v4_gq_rb (the recordings' beats where there are some, else the
#     renders', never simulated) and meter_v4_gq_rbm (those in half the draws, simulated in the rest);
#   - against meter_v4_gq (simulated where there was no recording) on whole pieces, with the
#     release's labeller and its beat decoding (no correction rules: the models alone).
#
# Before (once, on the login node):
#   module load anaconda3 && conda activate dl-proj && conda install -y -c conda-forge fluidsynth
#   mkdir -p data/raw/soundfonts    # then copy the SoundFonts there (scp), e.g. MuseScore's
#                                   # "MS Basic.sf3" (MuseScore 4/sound/) and Salamander's .sf2
#
#   job        what                                                                   waits for
#   rrtests    FluidSynth and every SoundFont render; the test suites; the renders     -
#              against the recordings (outputs/render_check/render_check.md)
#   rrtrack0-2 a third each of the training performances rendered and tracked,       rrtests (ok)
#              saved as they go (a shard run again goes on where it stopped)
#   rrmeter    the parts joined; meter_v4_gq_rb and meter_v4_gq_rbm trained, tuned,  rrtrack* (any)
#              evaluated (the join fails if a shard is unfinished: STAGE=track)
#   rrmeter2   either one not finished: resumed                                       rrmeter (any)
#   rreval     both with Beat This!'s real beats of Gangqinpu's test recordings;      rrmeter2 (any)
#              whole pieces, ASAP and Gangqinpu: $NL + each, tidy, d95, bd
#   rrfinal    outputs/compare, Beyer & Dai's metrics (ASAP, Gangqinpu), paired.md    rreval (any)
#
# Again after a stop (what is done is kept):
#   STAGE=track bash pace/v2_render.sh   # a shard did not finish: the shards again (they resume), then on
#   STAGE=meter bash pace/v2_render.sh   # the tracking is done: from the join on
#   STAGE=eval bash pace/v2_render.sh    # only rreval and rrfinal
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
NL=${NL:-nl_gq_mc}                         # the release's labeller
BASE=${BASE:-meter_v4_gq}                  # the meter model it is compared with
METERS=${METERS:-"meter_v4_gq_rb meter_v4_gq_rbm"}
SHARDS=${SHARDS:-3}
WORKERS=${WORKERS:-20}                     # FluidSynth processes per tracking job
CHECK_N=${CHECK_N:-40}                     # performances in the render check
BD=${BD:-$HOME/scratch/beyer_dai}
STAGE=${STAGE:-all}
case "$STAGE" in all|track|meter|eval) ;; *) echo "STAGE must be all, track, meter or eval, not $STAGE"; exit 1 ;; esac
# the SoundFonts: $SOUNDFONTS, else every .sf2 / .sf3 in data/raw/soundfonts
if [ -z "${SOUNDFONTS:-}" ]; then
  shopt -s nullglob
  fonts=(data/raw/soundfonts/*.sf2 data/raw/soundfonts/*.sf3)
  shopt -u nullglob
  [ ${#fonts[@]} -gt 0 ] || { echo "no SoundFont in data/raw/soundfonts (or set SOUNDFONTS): see the top of this script"; exit 1; }
  SOUNDFONTS=$(IFS=:; echo "${fonts[*]}")
fi
IFS=: read -r -a check_fonts <<< "$SOUNDFONTS"
for f in "${check_fonts[@]}"; do [ -s "$f" ] || { echo "no SoundFont $f"; exit 1; }; done
for f in data/processed/meter/train.pkl data/processed/meter/train_gangqinpu.pkl data/processed/meter/val_gangqinpu.pkl \
         data/raw/asap_bd/metadata.csv data/raw/beat_this/asap_8-folds.split configs/beat_noise_bt_gangqinpu.json \
         "outputs/runs/$BASE/best.pt" "outputs/runs/$NL/best.pt" outputs/system_eval/manifest.json \
         outputs/system_eval_gangqinpu/manifest.json; do
  [ -e "$f" ] || { echo "no $f: run the earlier rounds (pace/v2_gangqinpu.sh, pace/v2_gqlabel.sh, pace/v2_decode.sh) first"; exit 1; }
done
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: run bash pace/bd_setup.sh first"; exit 1; }
echo "SoundFonts: $SOUNDFONTS"
ENV="source pace/common.sh; export SOUNDFONTS=\"$SOUNDFONTS\""
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
JOBS="bash pace/v2_jobs.sh"
GQOUT=outputs/system_eval_gangqinpu
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
TRACK="python scripts/track_meter_data.py --trackers beat_this_render --splits train"
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

tracks=""
if [ "$STAGE" = all ] || [ "$STAGE" = track ]; then
  dep=()
  if [ "$STAGE" = all ]; then
    # FluidSynth there, every SoundFont renders, the tests; then the renders against the recordings
    # (informative: it does not hold the rest back, read render_check.md before trusting the result)
    SMOKE="python -c \"import sys; sys.path.insert(0, '.'); import numpy as np; from src.render import Renderer, find_soundfonts; n = np.array([[60 + k, 0.5 * k + 1, 0.5 * k + 1.4, 80] for k in range(8)]); [print('renders:', Renderer([f])(n, f'/tmp/rr_smoke_{k}.wav', 'x')) for k, f in enumerate(find_soundfonts())]\""
    submit rrtests "${G[@]}" -c 8 --mem=48G -t 4:00:00 --wrap="$ENV; command -v fluidsynth || { echo 'no fluidsynth in dl-proj: conda install -c conda-forge fluidsynth'; exit 1; }; $SMOKE && python -m tests.test_render && python -m tests.test_tracked && python -m tests.test_meter && { python scripts/compare_render.py --n $CHECK_N || echo 'render check failed (see above)'; }"
    dep=("--dependency=afterok:$ID")
    echo "FluidSynth, the SoundFonts, the tests; the renders against the recordings: job $ID"
  fi
  for ((k = 0; k < SHARDS; k++)); do
    submit "rrtrack$k" "${G[@]}" -c $((WORKERS + 4)) --mem=64G -t 8:00:00 ${dep[@]+"${dep[@]}"} \
      --wrap="$ENV; $TRACK --shard $k/$SHARDS --workers $WORKERS"
    tracks="$tracks:$ID"
    echo "shard $k of $SHARDS rendered and tracked: job $ID"
  done
fi
meter=""
if [ "$STAGE" != eval ]; then
  dep=()
  [ -n "$tracks" ] && dep=("--dependency=afterany$tracks")
  # shellcheck disable=SC2086
  submit rrmeter "${G[@]}" -c 16 --mem=96G -t 8:00:00 ${dep[@]+"${dep[@]}"} \
    --wrap="$ENV; $TRACK --merge $SHARDS && $JOBS meters $METERS"
  echo "the shards joined; $METERS trained, tuned and evaluated: job $ID"
  submit rrmeter2 "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterany:$ID" \
    --wrap="$ENV; $JOBS meters $METERS"
  meter=$ID
  echo "either not finished: resumed: job $meter"
fi
dep=()
[ -n "$meter" ] && dep=("--dependency=afterany:$meter")
specs=""
for m in $METERS; do specs="$specs $NL+$m:tidy,d95,bd"; done
# shellcheck disable=SC2086
submit rreval "${G[@]}" -c 8 --mem=48G -t 8:00:00 ${dep[@]+"${dep[@]}"} \
  --wrap="$ENV; $JOBS trackers $METERS; $JOBS systems $NL+$BASE:tidy,d95,bd,missing $specs; $GQSYS $JOBS systems $NL+$BASE:tidy,d95,bd,missing $specs"
evl=$ID
echo "with the real beats of Gangqinpu's recordings; whole pieces, ASAP and Gangqinpu ($NL + $BASE / $METERS): job $evl"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
submit rrfinal -c 16 --mem=64G -t 6:00:00 "--dependency=afterany:$evl" --wrap="source pace/common.sh; python3 scripts/compare_runs.py --quiet; $BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''"
echo "outputs/compare, Beyer & Dai's metrics (ASAP, Gangqinpu), paired comparisons: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
