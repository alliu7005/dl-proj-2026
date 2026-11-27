#!/bin/bash
# v6 data: a larger, balanced classical side, then the models retrained on it, larger ones included
# (on the login node from the repo root, after git pull, with the queue empty).
#   bash pace/v2_data.sh                  # stage 1, fetch (7 jobs: PianoCoRe rows, YouTube, Gangqinpu)
#   STAGE=build bash pace/v2_data.sh      # stage 2, after stage 1: re-transcription, the data rebuilt (4 jobs)
#   STAGE=train bash pace/v2_data.sh      # stage 3, after stage 2: the v6 models and whole pieces (8 jobs)
# Then pace/v2_btft.sh fine-tunes Beat This! on it (its export takes the downloaded classical
# recordings as real classical audio).
#
# Why: the classical half of every batch came from ~1,100 works and ~550 h (ASAP, 10 PERiScoPe
# performances per work) against ~1,000 h of Gangqinpu, so it was repeated about twice as often, and
# larger models overfit. Now (PianoCoRe metadata, best-aligned, never a Beyer & Dai test work):
#   - PERiScoPe up to 30 performances per work (~10,400 more, 776 h; Transkun V2 already);
#   - ATEPP and GiantMIDI-Piano (~4,700 within the same cap, 390 h, 794 works): their recordings
#     fetched from YouTube (their metadata links them) and re-transcribed with Transkun V2, so every
#     input carries the test-time transcriber's errors; a recording that does not match is dropped;
#   - Aria-MIDI's works none of those has (~220 works, ~1,000 performances at 10 per work, ~63 h):
#     its own transcriptions as they are (no recordings are linked), source aria_real, never given
#     the Transkun-like noise; the runs *_noaria train without them, to see what they bring;
#   - every Gangqinpu training recording (17.6k; 3,000 so far): real tracked beats and real pop audio.
# Not overfitting by work or by performance:
#   - splits stay by work, so a work's performances from every source share its split; the new
#     sources never add to test_periscope (comparable with every earlier result);
#   - at most 30 performances per work over every source, different pianists first;
#   - training weighs every work as one performance (meter --balance piece, labeller --piece-balance),
#     half of every batch classical, half Gangqinpu.
#
# Before (once, on the login node):
#   module load anaconda3 && conda activate dl-proj && pip install yt-dlp
# Space: ~22 GB of ATEPP / GiantMIDI audio, ~50 GB of Gangqinpu recordings, a few GB of PianoCoRe
# rows; the current data/processed is kept as data/processed_pre_v6. Check pace-quota first.
# YouTube may ask PACE's addresses to confirm they are not a bot: the fetch then stops and says so;
# export a cookies.txt from a logged-in browser, copy it here and run with COOKIES=<its path>.
#
# Stage 1 (fetch)                                                                         waits for
#   dtests   the tests (CPU)                                                               -
#   dpc      PianoCoRe rows: periscope_more (PERiScoPe to 30 per work), atepp              dtests (ok)
#            (ATEPP, GiantMIDI-Piano; to 30 per work over all three), aria (Aria-MIDI's new works)
#   dyt1-3   their recordings from YouTube, 3 s apart (three chained jobs; they resume)    dpc (ok), then each other
#   dgq1-2   every Gangqinpu training recording (two chained jobs; they resume)             dtests (ok)
# Stage 2 (build)
#   dtr1-2   the recordings re-transcribed with Transkun V2 (atepp -> atepp_tk; resume)    -
#   drebuild data/processed rebuilt with the extra sources (the old kept), the Gangqinpu    dtr2 (any)
#            data carried over; the data tests
#   dtrack   Beat This! on every Gangqinpu recording (its real beats for the meter model)   drebuild (ok)
# Stage 3 (train)
#   mtests   the tests                                                                      -
#   mnl      nl_v6, nl_v6_noaria (base); mnll: nl_v6_large                                  mtests (ok)
#   mmeter   meter_v6, meter_v6_s1, meter_v6_ne, meter_v6_noaria; mmeterl: meter_v6_large, mtests (ok)
#            meter_v6_large_s1
#   mresume  any not finished: resumed                                                     the four (any)
#   meval    whole pieces, ASAP and Gangqinpu: each v6 labeller on each v6 meter, and     mresume (any)
#            nl_gq_mc + meter_v4_gq (the release) to compare
#   mfinal   Beyer & Dai's metrics, paired.md                                               meval (any)
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
STAGE=${STAGE:-fetch}
PER_WORK=${PER_WORK:-30}
ARIA_PER_WORK=${ARIA_PER_WORK:-10}       # Aria-MIDI's new works: performances per work
PC=data/raw/pianocore
OLD=${OLD:-data/processed_pre_v6}
BD=${BD:-$HOME/scratch/beyer_dai}
COOKIES=${COOKIES:-}
case "$STAGE" in fetch|build|train) ;; *) echo "STAGE must be fetch, build or train, not $STAGE"; exit 1 ;; esac
for f in "$PC/metadata.parquet" "$PC/periscope/selection.parquet" data/processed/vocab.json \
         data/processed/meter/train_gangqinpu.pkl data/raw/gangqinpu; do
  [ -e "$f" ] || { echo "no $f: run the earlier rounds first"; exit 1; }
done
ENV="source pace/common.sh"
JOBS="bash pace/v2_jobs.sh"
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

if [ "$STAGE" = fetch ]; then
  submit dtests -c 8 --mem=32G -t 2:00:00 --wrap="$ENV; python -c 'import yt_dlp' || { echo 'pip install yt-dlp'; exit 1; }; python -m tests.test_data_v6 && python -m tests.test_bt && python -m tests.test_preprocessing"
  tests=$ID
  echo "the tests: job $tests"
  submit dpc -c 4 --mem=32G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$ENV; \
python scripts/fetch_pianocore.py --out-name periscope_more --have periscope --per-work 1000 --total-per-work $PER_WORK && \
python scripts/fetch_pianocore.py --source ATEPP,GiantMIDI-Piano --out-name atepp --have periscope,periscope_more --per-work 1000 --total-per-work $PER_WORK && python scripts/fetch_pianocore.py --source Aria-MIDI --out-name aria --have periscope,periscope_more,atepp --new-works-only --per-work $ARIA_PER_WORK"
  pc=$ID
  echo "PianoCoRe rows (PERiScoPe to $PER_WORK per work; ATEPP, GiantMIDI-Piano; Aria-MIDI's new works): job $pc"
  YT="python scripts/fetch_pianocore_audio.py --dir $PC/atepp ${COOKIES:+--cookies $COOKIES}"
  dep="--dependency=afterok:$pc"
  for k in 1 2 3; do
    submit "dyt$k" -c 2 --mem=8G -t 8:00:00 "$dep" --wrap="$ENV; timeout 7.8h $YT || true"
    dep="--dependency=afterany:$ID"
    echo "their recordings from YouTube ($k of 3, resuming): job $ID"
  done
  GQ="python scripts/fetch_gangqinpu_audio.py --train 100000"
  dep="--dependency=afterok:$tests"
  for k in 1 2; do
    submit "dgq$k" -c 2 --mem=8G -t 8:00:00 "$dep" --wrap="$ENV; timeout 7.8h $GQ || true"
    dep="--dependency=afterany:$ID"
    echo "every Gangqinpu training recording ($k of 2, resuming): job $ID"
  done
  echo "($n jobs) then: STAGE=build bash pace/v2_data.sh   (check $PC/atepp/audio/fetch_report.json first)"
  exit 0
fi

if [ "$STAGE" = build ]; then
  [ -f "$PC/atepp/audio/fetch_report.json" ] || { echo "no recordings fetched: stage 1 first"; exit 1; }
  [ -e "$OLD" ] && { echo "$OLD exists (rebuilt before?): move it away or set OLD=<another folder>"; exit 1; }
  RE="python scripts/retranscribe_pianocore.py --dir $PC/atepp --out $PC/atepp_tk"
  submit dtr1 "${G[@]}" -c 8 --mem=48G -t 8:00:00 --wrap="$ENV; timeout 7.8h $RE || true"
  submit dtr2 "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterany:$ID" --wrap="$ENV; $RE"
  tr=$ID
  echo "re-transcribed with Transkun V2 (two chained jobs, resuming): job $tr"
  # as pace/rebuild.sh, with the extra PianoCoRe sources (configs/preprocess.yaml extra_dirs); the
  # Gangqinpu data (built on its own) carried over
  submit drebuild -c 16 --mem=96G -t 8:00:00 "--dependency=afterany:$tr" --wrap="$ENV; set -euo pipefail; \
mv data/processed $OLD; \
python scripts/build_dataset.py --vocab $OLD/vocab.json; \
cp $OLD/feature_stats.json data/processed/feature_stats.json; \
python scripts/build_transcribed.py --asap-split test; \
python scripts/build_transcribed.py --asap-split train --maestro-splits test,validation; \
python scripts/build_transcribed.py --asap-split val --maestro-splits test,validation; \
for s in train_transcribed val_transcribed test_transcribed; do \
  if [ -d $OLD/\$s ] && [ -d data/processed/\$s ]; then cp $OLD/\$s/beats_tracked* data/processed/\$s/ 2>/dev/null || true; fi; \
done; \
for s in train_gangqinpu val_gangqinpu test_gangqinpu; do [ -d $OLD/\$s ] && cp -r $OLD/\$s data/processed/; done; \
python scripts/build_metrical.py --workers 16; \
python scripts/build_beats.py --workers 16 --check; \
python scripts/build_notelabels.py --workers 16; \
python scripts/build_meter_data.py --workers 16 --check; \
cp $OLD/meter/*gangqinpu* $OLD/meter/tracked_* data/processed/meter/ 2>/dev/null || true; \
python -m tests.test_integrity; python -m tests.test_preprocessing; \
python -c \"import json; r = json.load(open('data/processed/report.json')); print(json.dumps(r.get('sizes', r), indent=1)[:3000])\"; \
du -sh data/processed $OLD"
  rb=$ID
  echo "data/processed rebuilt with the extra sources ($OLD kept): job $rb"
  submit dtrack "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$rb" \
    --wrap="$ENV; python scripts/track_meter_data.py --trackers beat_this --splits train"
  echo "Beat This! on every Gangqinpu training recording: job $ID ($n jobs). Then: STAGE=train bash pace/v2_data.sh"
  exit 0
fi

# stage 3: the v6 models
[ -e "$OLD" ] || { echo "no $OLD: stage 2 has not rebuilt the data"; exit 1; }
submit mtests -c 8 --mem=64G -t 2:00:00 --wrap="$ENV; python -m tests.test_data_v6 && python -m tests.test_meter && python -m tests.test_v2_system"
tests=$ID
submit mnl "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$ENV; PUBLISH=0 $JOBS labellers nl_v6 nl_v6_noaria"
a=$ID
submit mnll "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$ENV; PUBLISH=0 $JOBS labellers nl_v6_large"
b=$ID
submit mmeter "${G[@]}" -c 32 --mem=160G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$ENV; PUBLISH=0 $JOBS meters meter_v6 meter_v6_s1 meter_v6_ne meter_v6_noaria"
c=$ID
submit mmeterl "${G[@]}" -c 16 --mem=128G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$ENV; PUBLISH=0 $JOBS meters meter_v6_large meter_v6_large_s1"
d=$ID
echo "nl_v6 / _noaria (job $a), nl_v6_large (job $b), meter_v6 / _s1 / _ne / _noaria (job $c), meter_v6_large / _s1 (job $d)"
submit mresume "${G[@]}" -c 24 --mem=128G -t 8:00:00 "--dependency=afterany:$a:$b:$c:$d" \
  --wrap="$ENV; PUBLISH=0 $JOBS labellers nl_v6 nl_v6_noaria nl_v6_large; PUBLISH=0 $JOBS meters meter_v6 meter_v6_s1 meter_v6_ne meter_v6_noaria meter_v6_large meter_v6_large_s1"
res=$ID
echo "any not finished: resumed: job $res"
GQOUT=outputs/system_eval_gangqinpu
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
SPECS="nl_gq_mc+meter_v4_gq:tidy,d95,bd,missing"
for nl in nl_v6 nl_v6_large; do
  for m in meter_v4_gq meter_v6 meter_v6_s1 meter_v6_ne meter_v6_large meter_v6_large_s1; do SPECS="$SPECS $nl+$m:tidy,d95,bd"; done
done
SPECS="$SPECS nl_v6_noaria+meter_v6_noaria:tidy,d95,bd"   # the same without Aria-MIDI's works
submit meval "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterany:$res" \
  --wrap="$ENV; $JOBS trackers meter_v6 meter_v6_s1 meter_v6_ne meter_v6_noaria meter_v6_large meter_v6_large_s1; $JOBS systems $SPECS; $GQSYS $JOBS systems $SPECS"
evl=$ID
echo "whole pieces, ASAP and Gangqinpu: job $evl"
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
submit mfinal -c 16 --mem=64G -t 6:00:00 "--dependency=afterany:$evl" --wrap="source pace/common.sh; python3 scripts/compare_runs.py --quiet; $BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''"
echo "Beyer & Dai's metrics, paired comparisons: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
