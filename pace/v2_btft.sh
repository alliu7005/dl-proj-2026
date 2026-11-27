#!/bin/bash
# Our Beat This!: retrained from scratch without ASAP, fine-tuned on piano (classical and pop,
# recordings and renders), cross-fitted, then used for the meter model's training beats and the
# evaluation (on the login node from the repo root, after git pull, with the queue empty).
#   bash pace/v2_btft.sh                 # stage 1: the trackers (7 jobs)
#   STAGE=use bash pace/v2_btft.sh       # stage 2, after stage 1 finished: using them (8 jobs)
#
# Why: the released Beat This! trained on ASAP (our benchmark pieces among them) and barely on piano
# otherwise (it dropped ASAP's rubato pieces; no pop piano): on Gangqinpu it tracks the wrong tempo
# octave on 34% of pieces. Fine-tuned on our piano data it should track piano better. Done so that
# nothing leaks:
#   - the base model never heard ASAP (its published data without it: scripts/fetch_beat_this_data.py);
#   - the fine-tuning data (scripts/export_beat_this.py) never holds a test piece; validation pieces
#     only validate;
#   - every training piece has one of K folds (src/bt_models.py); K models are fine-tuned each
#     leaving one fold out, and the meter model's training beats of a performance come from the one
#     that never heard its piece, so they are as good as beats of unseen music are (a tracker
#     tracks what it trained on too well, and a meter model trained on that trusts beats too much);
#   - the model fine-tuned on every fold tracks the test recordings (and would serve the demo).
# Fine-tuning data: ASAP (renders of its MIDI, and its MAESTRO recordings that are here), PERiScoPe
# (renders), Gangqinpu (its downloaded recordings, and renders): recordings because the tracker meets
# recordings at test time, renders because it tracks renders for the meter model's data.
#
# Before (once, on the login node): FluidSynth and the SoundFonts as for pace/v2_render.sh, and
#   module load anaconda3 && conda activate dl-proj && pip install pytorch-lightning
# Space: about 108 GB for Beat This!'s data without ASAP (EXCLUDE="asap,harmonix": 63 GB) plus
# ~25 GB for ours; check pace-quota first.
#
# Stage 1                                                                              waits for
#   btfetch     the tests; Beat This!'s published data without ASAP (CPU)               -
#   btexport    our piano data in its format, K folds (CPU, FluidSynth)                 -
#   btbase      Beat This! from scratch on its data without ASAP, its recipe            btfetch (ok)
#   btbase2     resumed if not finished                                                 btbase (any)
#   btft_a      fine-tuned from it: on every fold (bt_ft_full), fold 0 and 1 left out    btbase2, btexport (ok)
#   btft_b      fold 2 and 3 left out                                                   btbase2, btexport (ok)
#   btft_c      any of the five not finished: resumed                                   btft_a, btft_b (any)
# Stage 2 (STAGE=use)
#   btprep      the models as one set (outputs/beat_this/bt_ft/models.json); Gangqinpu's  -
#               recordings tracked by it (train: cross-fitted; val, test: the full model);
#               the evaluation's recordings (ASAP, Gangqinpu) tracked by the full model
#   bttrack0-2  every training performance rendered and tracked by its fold's model      btprep (ok)
#   btmeter     joined; meter_v4_gq_ft trained (beat_this_ft, beat_this_render_ft only)   the shards (any)
#   btmeter2    resumed if not finished                                                 btmeter (any)
#   bteval      meter_v4_gq_ft with the full model's beats of Gangqinpu's test           btmeter2 (any)
#               recordings; whole pieces (ASAP real, Gangqinpu) with our tracker's beats:
#               $NL + meter_v4_gq_ft, and $NL + meter_v4_gq (the tracker alone)
#   btfinal     Beyer & Dai's metrics, paired.md (ASAP, Gangqinpu)                      bteval (any)
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
STAGE=${STAGE:-train}
K=${K:-4}
BASE=${BASE:-bt_noasap}
SET=${SET:-bt_ft}
TAG=${TAG:-ft}
EXCLUDE=${EXCLUDE:-asap}
NL=${NL:-nl_gq_mc}
METER=${METER:-meter_v4_gq_ft}
SHARDS=${SHARDS:-3}
WORKERS=${WORKERS:-20}
BD=${BD:-$HOME/scratch/beyer_dai}
case "$STAGE" in train|use) ;; *) echo "STAGE must be train or use, not $STAGE"; exit 1 ;; esac
case ",$EXCLUDE," in *,asap,*) ;; *) echo "EXCLUDE must keep asap out of the base model"; exit 1 ;; esac
if [ -z "${SOUNDFONTS:-}" ]; then
  shopt -s nullglob
  fonts=(data/raw/soundfonts/*.sf2 data/raw/soundfonts/*.sf3)
  shopt -u nullglob
  [ ${#fonts[@]} -gt 0 ] || { echo "no SoundFont in data/raw/soundfonts (or set SOUNDFONTS): see pace/v2_render.sh"; exit 1; }
  SOUNDFONTS=$(IFS=:; echo "${fonts[*]}")
fi
for f in data/processed/meter/train.pkl data/processed/meter/train_gangqinpu.pkl data/processed/meter/train_periscope.pkl \
         data/raw/asap_bd/metadata.csv configs/beat_noise_bt_gangqinpu.json "outputs/runs/$NL/best.pt" \
         outputs/runs/meter_v4_gq/best.pt outputs/system_eval/manifest.json outputs/system_eval_gangqinpu/manifest.json; do
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
  # FETCHED=1: the data was fetched already (e.g. on the login node, in tmux, if compute nodes cannot
  # reach Zenodo: python scripts/fetch_beat_this_data.py --exclude asap); the job then only tests
  get="python scripts/fetch_beat_this_data.py --exclude $EXCLUDE"
  [ "${FETCHED:-0}" = 1 ] && get="ls data/beat_this_orig/audio/spectrograms/*.npz"
  submit btfetch -c 4 --mem=16G -t 8:00:00 --wrap="$ENV; python -c 'import pytorch_lightning' || { echo 'pip install pytorch-lightning'; exit 1; }; python -m tests.test_bt && python -m tests.test_render && $get"
  fetch=$ID
  echo "the tests; Beat This!'s data without $EXCLUDE: job $fetch"
  submit btexport -c $((WORKERS + 4)) --mem=64G -t 8:00:00 --wrap="$ENV; command -v fluidsynth >/dev/null || { echo 'no fluidsynth: conda install -c conda-forge fluidsynth'; exit 1; }; python scripts/export_beat_this.py --k $K --augment --workers $WORKERS"
  export_=$ID
  echo "our piano data in its format ($K folds): job $export_"
  submit btbase "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$fetch" \
    --wrap="$ENV; $TRAIN --name $BASE --data data/beat_this_orig --exclude $EXCLUDE --no-val --num-workers 14 --max-hours 7.6"
  submit btbase2 "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterany:$ID" \
    --wrap="$ENV; $TRAIN --name $BASE --data data/beat_this_orig --exclude $EXCLUDE --no-val --num-workers 14 --max-hours 7.6"
  base=$ID
  echo "Beat This! from scratch without $EXCLUDE ($BASE; resumed if needed): job $base"
  ftjobs=""
  for spec in "a full 0 1" "b 2 3"; do
    set -- $spec
    letter=$1
    shift
    cmds=""
    for x in "$@"; do
      if [ "$x" = full ]; then cmds="$cmds $TRAIN --name ${SET}_full $FT --max-hours 2.3;"
      else cmds="$cmds $TRAIN --name ${SET}_fold$x $FT --fold $x --max-hours 2.3;"; fi
    done
    submit "btft_$letter" "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$base:$export_" \
      --wrap="$ENV; [ -f outputs/beat_this/$BASE/final.ckpt ] || { echo 'the base model did not finish'; exit 1; }; $cmds"
    ftjobs="$ftjobs:$ID"
    echo "fine-tuned ($*): job $ID"
  done
  all=""
  for ((f = 0; f < K; f++)); do all="$all $TRAIN --name ${SET}_fold$f $FT --fold $f --max-hours 1.8;"; done
  submit btft_c "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterany$ftjobs" \
    --wrap="$ENV; $TRAIN --name ${SET}_full $FT --max-hours 1.8; $all"
  echo "any not finished: resumed: job $ID ($n jobs). Then: STAGE=use bash pace/v2_btft.sh"
  echo "watch: squeue -u \$USER"
  exit 0
fi

# stage 2: using them
for r in "${SET}_full" $(for ((f = 0; f < K; f++)); do echo "${SET}_fold$f"; done); do
  [ -f "outputs/beat_this/$r/final.ckpt" ] || { echo "outputs/beat_this/$r not finished: stage 1 (btft_c) first"; exit 1; }
done
folds=$(for ((f = 0; f < K; f++)); do printf "%s" "${SET}_fold$f"; [ $f -lt $((K - 1)) ] && printf ","; done)
MODELS="outputs/beat_this/$SET"
submit btprep "${G[@]}" -c 8 --mem=48G -t 6:00:00 --wrap="$ENV; $TRAIN --assemble $SET --full ${SET}_full --fold-runs $folds --pieces data/beat_this_ft/pieces.json && python scripts/track_meter_data.py --trackers beat_this --models $MODELS --tag $TAG && python scripts/track_eval_beats.py --models $MODELS --tag $TAG"
prep=$ID
echo "the model set; Gangqinpu's recordings and the evaluation's tracked by it: job $prep"
RTRACK="python scripts/track_meter_data.py --trackers beat_this_render --models $MODELS --tag $TAG --splits train"
shards=""
for ((k = 0; k < SHARDS; k++)); do
  submit "bttrack$k" "${G[@]}" -c $((WORKERS + 4)) --mem=64G -t 8:00:00 "--dependency=afterok:$prep" \
    --wrap="$ENV; $RTRACK --shard $k/$SHARDS --workers $WORKERS"
  shards="$shards:$ID"
  echo "shard $k: rendered and tracked by each piece's fold model: job $ID"
done
submit btmeter "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterany$shards" \
  --wrap="$ENV; $RTRACK --merge $SHARDS && $JOBS meters $METER"
submit btmeter2 "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterany:$ID" --wrap="$ENV; $JOBS meters $METER"
meter=$ID
echo "joined; $METER trained, tuned, evaluated (resumed if needed): job $meter"
GQOUT=outputs/system_eval_gangqinpu
SPECS="$NL+$METER:tidy,d95,bd $NL+meter_v4_gq:tidy,d95,bd"
submit bteval "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterany:$meter" \
  --wrap="$ENV; python scripts/eval_meter.py --run $METER --real-tracked --real-tracker beat_this_$TAG --conditions gangqinpu; SYS_ARGS=\"--beats-tag $TAG --conditions real\" $JOBS systems $SPECS; SYS_ARGS=\"--beats-tag $TAG --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\" $JOBS systems $SPECS"
evl=$ID
echo "$METER on our tracker's beats of the test recordings; whole pieces: job $evl"
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
submit btfinal -c 16 --mem=64G -t 6:00:00 "--dependency=afterany:$evl" --wrap="source pace/common.sh; python3 scripts/compare_runs.py --quiet; $BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''"
echo "Beyer & Dai's metrics, paired comparisons: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
