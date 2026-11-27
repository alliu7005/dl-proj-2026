#!/bin/bash
# v2, twelfth round: everything the pilots said to scale, one stage after another (on the login node
# from the repo root, after git pull; submits every stage now, each waiting for the one before):
#   bash pace/v2_round12.sh
#   STAGES="chooser seq2seq" bash pace/v2_round12.sh    # only some stages, in this order
#
# Stages (each waits for every job of the stage before it, finished or not: afterany):
#   ensemble  round 11 (pace/v2_ensemble.sh, 7 jobs): two more seeds of the release's meter model,
#             the seeds averaged (meter_v4_gq_ens3, _ens3t), whole pieces. The pilot's A: averaging
#             3 seeds beat one seed by 0.5-0.7 MUSTER on real ASAP with the PM2S meters; this is the
#             same for the release's (Beat This!) meter, which the pilot did not test.
#   chooser   the duration chooser (src/dur_chooser.py) trained for nl_gq on the validation windows
#             (scripts/train_dur_chooser.py), measured on the test windows; whole pieces with it (_dc)
#             and with the new defaults: one meter per piece (meter_v4_gq_om6, the pilot's F) and
#             swing (_sw, the pilot's I), at d95 and at d80 (MIDI input's setting, the pilot's D);
#             5 jobs. The headroom test: the right length is in the top 4 for 95% of notes, and a
#             perfect pick would raise rendered duration accuracy 0.87 -> 0.97.
#   seq2seq   the seq2seq model's recipe (sf_base_reg_cab_c128) retrained on the rebuilt data with
#             30% of its windows from Gangqinpu (train_model.py --gangqinpu-share 0.3; nl_gq's
#             Gangqinpu data was v2's biggest single gain, -1.93 MUSTER on pop), up to 3 x 8 h,
#             evaluated, whole pieces on ASAP and Gangqinpu; 7 jobs
#   pm2s      round 9 (pace/v2_pm2s2.sh, 8 jobs): PM2S retrained with stable settings (the pilot's C)
# outputs/round11/report.md (ensemble), outputs/round12/report.md (chooser, seq2seq), and round 9's
# outputs/compare + paired.md. Nothing is published (PUBLISH=0).
set -euo pipefail
STAGES=${STAGES:-"ensemble chooser seq2seq pm2s"}
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
GQOUT=outputs/system_eval_gangqinpu
SEQ_RUN=${SEQ_RUN:-sf_c128_gq30}
SEQ_ARGS="--size base --dropout 0.2 --weight-decay 0.1 --lr 2e-4 --context-before 128 --context-after 64 --aux-metrical --beats --gangqinpu-share 0.3"
for s in $STAGES; do
  case "$s" in ensemble|chooser|seq2seq|pm2s) ;; *) echo "unknown stage $s (ensemble, chooser, seq2seq, pm2s)"; exit 1 ;; esac
done
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: bash pace/bd_setup.sh first"; exit 1; }
GQ=${GQ:-data/raw/gangqinpu}
case " $STAGES " in *" seq2seq "*)   # s1data builds Gangqinpu's windows and the beats where they are missing
  [ -n "$(ls "$GQ"/data/*.parquet 2>/dev/null)" ] || { echo "no $GQ/data/*.parquet: run pace/v2_gangqinpu.sh first"; exit 1; }
  for f in data/processed/train/index.npy data/processed/val/index.npy data/processed/vocab.json; do
    [ -e "$f" ] || { echo "no $f: the data build first (pace/data.sbatch)"; exit 1; }
  done ;;
esac
export PUBLISH=0
JOBS="PUBLISH=0 bash pace/v2_jobs.sh"
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
FINAL="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''; python3 scripts/round12_report.py"

JOBLOG=$(mktemp)
export JOBLOG
AFTER=""
n=0
ID=""
submit() {   # as the other batches' (AFTER: a job with no dependency of its own waits for the stage before)
  local name=$1
  shift
  local kill=() wait=()
  case " $* " in *" --dependency=afterok:"*) kill=(--kill-on-invalid-dep=yes) ;; esac
  case " $* " in *" --dependency="*) ;; *) [ -n "$AFTER" ] && wait=("--dependency=afterany:$AFTER") ;; esac
  ID=$(sbatch --parsable -J "$name" -o "slurm-$name-%j.out" ${kill[@]+"${kill[@]}"} ${wait[@]+"${wait[@]}"} "$@")
  echo "$ID" >> "$JOBLOG"
  n=$((n + 1))
}
next_stage() {   # every job submitted so far: the next stage waits for them
  AFTER=$(paste -sd: "$JOBLOG")
  export AFTER
}

for stage in $STAGES; do
  echo "== $stage"
  case "$stage" in
    ensemble)
      AFTER="$AFTER" bash pace/v2_ensemble.sh ;;
    pm2s)
      AFTER="$AFTER" bash pace/v2_pm2s2.sh ;;
    chooser)
      submit c1tests -c 8 --mem=32G -t 1:00:00 --wrap="source pace/common.sh; python -m tests.test_dur_chooser && python -m tests.test_release && python -m tests.test_system_eval"
      tests=$ID
      PREP="[ -f outputs/runs/meter_v4_gq_om6/decode.json ] || $JOBS onemeter meter_v4_gq 0.6 meter_v4_gq_om6 || exit 1; python scripts/train_dur_chooser.py --run nl_gq"
      submit c1train "${G[@]}" -c 8 --mem=48G -t 4:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $PREP"
      train=$ID
      echo "the chooser trained and measured on the test windows (+ meter_v4_gq_om6 if missing): job $train"
      SPECS="nl_gq+meter_v4_gq:tidy,d95,dc,missing nl_gq+meter_v4_gq_om6:tidy,d95,sw,missing nl_gq+meter_v4_gq_om6:tidy,d95,sw,dc,missing nl_gq+meter_v4_gq_om6:tidy,d80,sw,dc,missing"
      submit c1asap "${G[@]}" -c 8 --mem=48G -t 10:00:00 "--dependency=afterok:$train" --wrap="source pace/common.sh; $JOBS systems $SPECS"
      asap=$ID
      submit c1gq "${G[@]}" -c 8 --mem=48G -t 10:00:00 "--dependency=afterok:$train" --wrap="source pace/common.sh; $GQSYS $JOBS systems $SPECS"
      gq=$ID
      echo "whole pieces with the chooser and the new defaults: jobs $asap (ASAP), $gq (Gangqinpu)"
      submit c1final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq" --wrap="$FINAL"
      echo "Beyer & Dai's metrics, outputs/round12/report.md: job $ID" ;;
    seq2seq)
      DATA="for s in train val; do [ -s data/processed/\${s}_gangqinpu/index.npy ] || python scripts/build_gangqinpu_windows.py --release $GQ --splits \$s || exit 1; done; "
      DATA+="B=train_gangqinpu,val_gangqinpu; for s in train val; do [ -f data/processed/\$s/beats_gt.npz ] || B=\$B,\$s; done; "
      DATA+="python scripts/build_beats.py --splits \$B --workers 16 && python scripts/build_metrical.py --splits train_gangqinpu,val_gangqinpu --workers 16"
      submit s1data -c 16 --mem=64G -t 6:00:00 --wrap="source pace/common.sh; $DATA"
      data=$ID
      echo "Gangqinpu's windows (if missing), the beats and metrical labels for seq2seq: job $data"
      prev="afterok:$data"
      for k in a b c; do   # one run over up to three 8-hour jobs: each resumes from last.pt, a finished one evaluates and exits
        submit "s1train_$k" "${G[@]}" -c 8 --mem=64G -t 8:00:00 "--dependency=$prev" --wrap="source pace/common.sh; $JOBS seq2seq $SEQ_RUN $SEQ_ARGS"
        prev="afterany:$ID"
      done
      echo "$SEQ_RUN trained (up to 3 x 8 h) and evaluated: last job $ID"
      submit s1asap "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=$prev" --wrap="source pace/common.sh; $JOBS systems $SEQ_RUN:meter:sb"
      asap=$ID
      submit s1gq "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=$prev" --wrap="source pace/common.sh; $GQSYS $JOBS systems $SEQ_RUN:meter:sb"
      gq=$ID
      echo "whole pieces with $SEQ_RUN: jobs $asap (ASAP), $gq (Gangqinpu)"
      submit s1final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq" --wrap="$FINAL"
      echo "Beyer & Dai's metrics, outputs/round12/report.md: job $ID" ;;
  esac
  next_stage
done
echo "$(wc -l < "$JOBLOG" | tr -d ' ') jobs in all (stages: $STAGES); watch: squeue -u \$USER"
rm -f "$JOBLOG"
