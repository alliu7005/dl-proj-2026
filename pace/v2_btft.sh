#!/bin/bash
# Our Beat This!: retrained from scratch without ASAP, fine-tuned on piano (classical and modern,
# recordings and renders), in two versions (plain, and multi-task with a time-signature head), each
# cross-fitted; then the meter model retrained on each one's beats (on the login node from the repo
# root, after git pull, with the queue empty). Run AFTER pace/v2_data.sh's build stage (this exports
# the v6 data, its downloaded recordings included); see pace/README.md, "Run order".
#   bash pace/v2_btft.sh                    # stage 1, train: the data and the base model (4 jobs)
#   STAGE=finetune bash pace/v2_btft.sh     # stage 2, after stage 1: the 10 fine-tunes (8 jobs)
#   STAGE=use bash pace/v2_btft.sh          # stage 3, after stage 2: beats, meter models, whole pieces (8 jobs)
#
# Why: the released Beat This! trained on ASAP (our benchmark pieces among them) and barely on piano
# otherwise (it dropped ASAP's rubato pieces; no pop piano): on Gangqinpu it tracks the wrong tempo
# octave on 34% of pieces. Fine-tuned on piano it should track piano better. Nothing leaks:
#   - the base model never heard ASAP (its published data without it: scripts/fetch_beat_this_data.py);
#   - the fine-tuning data (scripts/export_beat_this.py) never holds a test piece; validation pieces
#     only validate;
#   - every training piece has one of K folds (src/bt_models.py); K models are fine-tuned each
#     leaving one fold out, and the meter model's training beats of a performance come from the one
#     that never heard its piece (a tracker tracks what it trained on too well, and a meter model
#     trained on that trusts beats too much);
#   - the model fine-tuned on every fold tracks the test recordings (and would serve the demo).
# Fine-tuning data (the v6 data, scripts/export_beat_this.py): ASAP (renders of its MIDI, its MAESTRO
# recordings that are here), PianoCoRe (PERiScoPe and Aria-MIDI renders; ATEPP / GiantMIDI-Piano's
# downloaded recordings), Gangqinpu (its recordings, and renders), classical / modern x recordings /
# renders weighed alike.
# Two versions, everything else equal: bt_ft (plain) and bt_ftm (multi-task: also learning each
# frame's time signature from the score alignment, src/bt_multitask.py; the head is dropped after
# training). Each gets its meter model (meter_v6_ft, meter_v6_ftm: the v6 recipe, every training
# performance's beats from that tracker). The labeller (nl_v6, pace/v2_data.sh) needs no retraining:
# it never reads tracked beats (it trains on score grids with simulated tracker timing).
#
# Before (once, on the login node): FluidSynth and the SoundFonts as for pace/v2_render.sh, and
#   module load anaconda3 && conda activate dl-proj && pip install pytorch-lightning
# Space: about 108 GB for Beat This!'s data without ASAP (EXCLUDE="asap,harmonix": 63 GB) plus
# ~25 GB for ours; check pace-quota first. Compute nodes without internet: fetch on the login node
# first (in tmux: python scripts/fetch_beat_this_data.py --exclude asap) and run with FETCHED=1.
#
# Stage 1 (train)                                                                        waits for
#   btfetch     the tests; Beat This!'s published data without ASAP (CPU)                 -
#   btexport    the v6 data in its format, K folds, the four cells balanced (CPU)         -
#   btbase      Beat This! from scratch on its data without ASAP, its recipe (bt_noasap)  btfetch (ok)
#   btbase2     resumed if not finished                                                   btbase (any)
# Stage 2 (finetune)
#   ftfull      bt_ft_full (every fold); ftmfull: bt_ftm_full                              -
#   ft01, ft23  bt_ft_fold0-3 (fold k left out); ftm01, ftm23: bt_ftm_fold0-3              -
#   ftres       the plain ones not finished: resumed; ftmres: the multi-task ones          their jobs (any)
# Stage 3 (use)
#   btprep      both model sets (models.json); Gangqinpu's recordings tracked by both      -
#               (training ones by their fold's model); the evaluation's recordings by both
#               full models (tags ft, ftm); renders against recordings for the released
#               tracker and both new ones (outputs/render_check_{released,ft,ftm})
#   bttrack0-2  every training performance rendered once and tracked by both sets          btprep (ok)
#   btmeter     joined; meter_v6_ft and meter_v6_ftm trained, tuned, evaluated              the shards (any)
#   btmeter2    resumed if not finished                                                    btmeter (any)
#   bteval      each meter model with its tracker's beats of Gangqinpu's test recordings;  btmeter2 (any)
#               whole pieces (ASAP real, Gangqinpu): nl_v6 + meter_v6_ft on bt_ft's beats
#               (_btft), nl_v6 + meter_v6_ftm on bt_ftm's (_btftm); and the tracker alone:
#               the release's nl_gq_mc + meter_v4_gq on each one's beats
#   btfinal     Beyer & Dai's metrics, paired.md (ASAP, Gangqinpu)                         bteval (any)
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
STAGE=${STAGE:-train}
K=${K:-4}
BASE=${BASE:-bt_noasap}
EXCLUDE=${EXCLUDE:-asap}
NL=${NL:-nl_v6}                 # the v6 labeller (pace/v2_data.sh stage 3)
REL=${REL:-nl_gq_mc+meter_v4_gq}   # the release: the tracker-alone comparison
SHARDS=${SHARDS:-3}
WORKERS=${WORKERS:-20}
BD=${BD:-$HOME/scratch/beyer_dai}
case "$STAGE" in train|finetune|use) ;; *) echo "STAGE must be train, finetune or use, not $STAGE"; exit 1 ;; esac
case ",$EXCLUDE," in *,asap,*) ;; *) echo "EXCLUDE must keep asap out of the base model"; exit 1 ;; esac
if [ -z "${SOUNDFONTS:-}" ]; then
  shopt -s nullglob
  fonts=(data/raw/soundfonts/*.sf2 data/raw/soundfonts/*.sf3)
  shopt -u nullglob
  [ ${#fonts[@]} -gt 0 ] || { echo "no SoundFont in data/raw/soundfonts (or set SOUNDFONTS): see pace/v2_render.sh"; exit 1; }
  SOUNDFONTS=$(IFS=:; echo "${fonts[*]}")
fi
# the run order: the v6 data must be built (pace/v2_data.sh STAGE=build), or the export would take the old data
[ -d data/processed_pre_v6 ] || { echo "the v6 data is not built yet: pace/v2_data.sh (STAGE=fetch, then STAGE=build) first"; exit 1; }
[ -n "$(ls data/raw/pianocore/atepp_tk/rows-*.parquet 2>/dev/null)" ] || echo "note: no re-transcribed recordings (data/raw/pianocore/atepp_tk): the export has no downloaded classical recordings"
for f in data/processed/meter/train.pkl data/processed/meter/train_gangqinpu.pkl data/processed/meter/train_periscope.pkl \
         data/raw/asap_bd/metadata.csv configs/beat_noise_bt_gangqinpu.json outputs/system_eval/manifest.json \
         outputs/system_eval_gangqinpu/manifest.json; do
  [ -e "$f" ] || { echo "no $f: run the earlier rounds first"; exit 1; }
done
ENV="source pace/common.sh; export SOUNDFONTS=\"$SOUNDFONTS\""
JOBS="bash pace/v2_jobs.sh"
TRAIN="python scripts/train_beat_this.py"
FT="--data data/beat_this_ft --init outputs/beat_this/$BASE/final.ckpt --finetune --num-workers 12"
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

if [ "$STAGE" = train ]; then
  get="python scripts/fetch_beat_this_data.py --exclude $EXCLUDE"
  [ "${FETCHED:-0}" = 1 ] && get="ls data/beat_this_orig/audio/spectrograms/*.npz"
  submit btfetch -c 4 --mem=16G -t 8:00:00 --wrap="$ENV; python -c 'import pytorch_lightning' || { echo 'pip install pytorch-lightning'; exit 1; }; python -m tests.test_bt && python -m tests.test_bt_multitask && python -m tests.test_render && $get"
  fetch=$ID
  echo "the tests; Beat This!'s data without $EXCLUDE: job $fetch"
  submit btexport -c $((WORKERS + 4)) --mem=64G -t 8:00:00 --wrap="$ENV; command -v fluidsynth >/dev/null || { echo 'no fluidsynth: conda install -c conda-forge fluidsynth'; exit 1; }; python scripts/export_beat_this.py --k $K --augment --workers $WORKERS"
  echo "the v6 data in its format ($K folds; data/beat_this_ft/export_report.json: the mix): job $ID"
  submit btbase "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$fetch" \
    --wrap="$ENV; $TRAIN --name $BASE --data data/beat_this_orig --exclude $EXCLUDE --no-val --num-workers 14 --max-hours 7.6"
  submit btbase2 "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterany:$ID" \
    --wrap="$ENV; $TRAIN --name $BASE --data data/beat_this_orig --exclude $EXCLUDE --no-val --num-workers 14 --max-hours 7.6"
  echo "Beat This! from scratch without $EXCLUDE ($BASE; resumed if needed): job $ID ($n jobs)"
  echo "when outputs/beat_this/$BASE/final.ckpt and data/beat_this_ft/pieces.json exist: STAGE=finetune bash pace/v2_btft.sh"
  exit 0
fi

if [ "$STAGE" = finetune ]; then
  [ -f "outputs/beat_this/$BASE/final.ckpt" ] || { echo "outputs/beat_this/$BASE not finished: stage 1 (train) first"; exit 1; }
  [ -f data/beat_this_ft/pieces.json ] || { echo "no data/beat_this_ft/pieces.json: stage 1's export first"; exit 1; }
  run() {  # <set: bt_ft or bt_ftm> <full or a fold> <hours>: one fine-tune
    local mt=""
    [ "$1" = bt_ftm ] && mt="--multitask"
    if [ "$2" = full ]; then echo "$TRAIN --name ${1}_full $FT $mt --max-hours $3;"
    else echo "$TRAIN --name ${1}_fold$2 $FT $mt --fold $2 --max-hours $3;"; fi
  }
  for set in bt_ft bt_ftm; do
    jobs=""
    tag=${set#bt_}
    submit "${tag}full" "${G[@]}" -c 16 --mem=96G -t 8:00:00 --wrap="$ENV; $(run $set full 7.5)"
    jobs="$jobs:$ID"
    echo "${set}_full (every fold): job $ID"
    for pair in "0 1" "2 3"; do
      set -- $pair
      submit "${tag}$1$2" "${G[@]}" -c 16 --mem=96G -t 8:00:00 --wrap="$ENV; $(run $set $1 3.7) $(run $set $2 3.7)"
      jobs="$jobs:$ID"
      echo "${set}_fold$1 and _fold$2 (each fold left out): job $ID"
    done
    resume="$(run $set full 1.4)"
    for ((f = 0; f < K; f++)); do resume="$resume $(run $set $f 1.4)"; done
    submit "${tag}res" "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterany$jobs" --wrap="$ENV; $resume"
    echo "any of $set's not finished: resumed: job $ID"
  done
  echo "($n jobs) when all ten have final.ckpt: STAGE=use bash pace/v2_btft.sh"
  exit 0
fi

# stage 3: using them
for set in bt_ft bt_ftm; do
  for r in "${set}_full" $(for ((f = 0; f < K; f++)); do echo "${set}_fold$f"; done); do
    [ -f "outputs/beat_this/$r/final.ckpt" ] || { echo "outputs/beat_this/$r not finished: stage 2 (finetune) first"; exit 1; }
  done
done
done_run() { grep -q '"phase": "done"' "outputs/runs/$1/status.json" 2>/dev/null; }   # as pace/v2_jobs.sh
done_run "$NL" || echo "note: $NL (pace/v2_data.sh stage 3) is not finished: its whole-piece systems will be skipped until it is"
folds() { for ((f = 0; f < K; f++)); do printf "%s" "${1}_fold$f"; [ $f -lt $((K - 1)) ] && printf ","; done; }
SETS="outputs/beat_this/bt_ft,outputs/beat_this/bt_ftm"
TAGS="ft,ftm"
ASSEMBLE="$TRAIN --assemble bt_ft --full bt_ft_full --fold-runs $(folds bt_ft) --pieces data/beat_this_ft/pieces.json && \
$TRAIN --assemble bt_ftm --full bt_ftm_full --fold-runs $(folds bt_ftm) --pieces data/beat_this_ft/pieces.json"
# do renders still track like recordings after fine-tuning? (each piece by the model that never heard it;
# the meter model's training beats of PERiScoPe, clean ASAP and Aria-MIDI come from renders)
CHECK="python scripts/compare_render.py --n 40"
submit btprep "${G[@]}" -c 8 --mem=48G -t 8:00:00 --wrap="$ENV; $ASSEMBLE && \
python scripts/track_meter_data.py --trackers beat_this --models $SETS --tag $TAGS && \
python scripts/track_eval_beats.py --models outputs/beat_this/bt_ft --tag ft && \
python scripts/track_eval_beats.py --models outputs/beat_this/bt_ftm --tag ftm; rc=\$?; $CHECK --out outputs/render_check_released; $CHECK --models outputs/beat_this/bt_ft --out outputs/render_check_ft; $CHECK --models outputs/beat_this/bt_ftm --out outputs/render_check_ftm; exit \$rc"
prep=$ID
echo "both model sets; Gangqinpu's recordings and the evaluation's tracked by both: job $prep"
RTRACK="python scripts/track_meter_data.py --trackers beat_this_render --models $SETS --tag $TAGS --splits train"
shards=""
for ((k = 0; k < SHARDS; k++)); do
  submit "bttrack$k" "${G[@]}" -c $((WORKERS + 4)) --mem=64G -t 8:00:00 "--dependency=afterok:$prep" \
    --wrap="$ENV; $RTRACK --shard $k/$SHARDS --workers $WORKERS"
  shards="$shards:$ID"
  echo "shard $k: rendered once, tracked by both sets (each piece by its fold's models): job $ID"
done
submit btmeter "${G[@]}" -c 16 --mem=128G -t 8:00:00 "--dependency=afterany$shards" \
  --wrap="$ENV; $RTRACK --merge $SHARDS && $JOBS meters meter_v6_ft meter_v6_ftm"
submit btmeter2 "${G[@]}" -c 16 --mem=128G -t 8:00:00 "--dependency=afterany:$ID" --wrap="$ENV; $JOBS meters meter_v6_ft meter_v6_ftm"
meter=$ID
echo "joined; meter_v6_ft and meter_v6_ftm trained, tuned, evaluated (resumed if needed): job $meter"
GQOUT=outputs/system_eval_gangqinpu
sys_on() {  # <tag> <specs>: whole pieces with that tracker's beats of the recordings, ASAP and Gangqinpu
  echo "SYS_ARGS=\"--beats-tag $1 --conditions real\" $JOBS systems $2; SYS_ARGS=\"--beats-tag $1 --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\" $JOBS systems $2"
}
EVAL="python scripts/eval_meter.py --run meter_v6_ft --real-tracked --real-tracker beat_this_ft --conditions gangqinpu; \
python scripts/eval_meter.py --run meter_v6_ftm --real-tracked --real-tracker beat_this_ftm --conditions gangqinpu; \
$(sys_on ft "$NL+meter_v6_ft:tidy,d95,bd $REL:tidy,d95,bd"); $(sys_on ftm "$NL+meter_v6_ftm:tidy,d95,bd $REL:tidy,d95,bd")"
submit bteval "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterany:$meter" --wrap="$ENV; $EVAL"
evl=$ID
echo "each meter model on its tracker's beats; whole pieces (_btft, _btftm), the tracker alone too: job $evl"
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
submit btfinal -c 16 --mem=64G -t 6:00:00 "--dependency=afterany:$evl" --wrap="source pace/common.sh; python3 scripts/compare_runs.py --quiet; $BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''"
echo "Beyer & Dai's metrics, paired comparisons: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
