#!/bin/bash
# The rhythm prior pilot: fewer stray triplets (on the login node from the repo root, after git pull;
# any time, 6 jobs):
#   bash pace/v2_rhythm.sh
#
# Why: v2 writes triplets the playing does not support (Lemon: 62% of bars with a rare rhythm, the
# off-grid notes of bars 52-58 / 81-89 nearer straight 16ths), because the labeller places each
# note on its own; a global penalty on rare rhythms cost 0.3-0.7 MUSTER on classical, where the
# triplets are real. The beat decoding (src/beat_decode.py) already takes one family of splits per
# beat; here it decodes the beats' families as a sequence with a rhythm prior learned from the
# written scores (scripts/fit_rhythm_prior.py): how rare a triplet beat is, and that triplets come
# in runs. Its weight is tuned on the validation windows (ASAP and Gangqinpu) with everything else,
# so classical's real triplets weigh as much as pop's straight beats.
#
#   rptests  the tests                                                             -
#   rpfit    the prior from the training scores (train, train_gangqinpu; CPU)      rptests (ok)
#   rptune   nl_gq's beat decoding tuned twice on the same windows: without the    rpfit (ok)
#            prior (nl_gq_bd0) and with it (nl_gq_rp); each an alias of nl_gq with
#            its own beat_decode.json
#   rpasap   whole pieces, round 15's candidate (meter_v4_gq_pmhs, PM2S's beats,    rptune (ok)
#   rpgq     _jf) with each decoding (_bd)
#   rpfinal  Beyer & Dai's metrics, outputs/rhythm/report.md                        both (any)
# Go: rare rhythms down on Gangqinpu, MUSTER not worse beyond chance anywhere (against _bd without
# the prior and against the candidate without beat decoding).
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
GQOUT=outputs/system_eval_gangqinpu
PM_RUN=${PM_RUN:-pm2s_hos}
PM_TAG=${PM_TAG:-hos}
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: bash pace/bd_setup.sh first"; exit 1; }
for f in outputs/runs/nl_gq/best.pt outputs/runs/nl_gq/status.json "outputs/runs/$PM_RUN/best.pt" \
         outputs/runs/meter_v4_gq_pmhs/eval_meter.json data/processed/train/notelabels.npz data/processed/val/notelabels.npz \
         "$GQOUT/manifest.json" outputs/system_eval/manifest.json; do
  [ -e "$f" ] || { echo "no $f: an earlier round makes it"; exit 1; }
done
export PUBLISH=0
JOBS="PUBLISH=0 bash pace/v2_jobs.sh"
PMENV="source pace/common.sh; export PM2S_WEIGHTS=\"\$SLURM_SUBMIT_DIR/outputs/runs/$PM_RUN/best.pt\" PM2S_TAG=$PM_TAG"
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
PRIOR=outputs/rhythm_prior/prior.json
# nl_gq under another name with its own beat_decode.json (nl_gq's own is left as it is)
ALIAS='for a in nl_gq_bd0 nl_gq_rp; do mkdir -p outputs/runs/$a; ln -sf ../nl_gq/best.pt outputs/runs/$a/best.pt; for f in config.json status.json vocab.json; do [ -e outputs/runs/nl_gq/$f ] && cp outputs/runs/nl_gq/$f outputs/runs/$a/$f; done; done'
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
submit rptests -c 8 --mem=32G -t 1:00:00 --wrap="source pace/common.sh; python -m tests.test_rhythm_prior"
tests=$ID
submit rpfit -c 8 --mem=48G -t 4:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; python scripts/fit_rhythm_prior.py --out $PRIOR"
fit=$ID
submit rptune "${G[@]}" -c 8 --mem=48G -t 6:00:00 "--dependency=afterok:$fit" --wrap="source pace/common.sh; $ALIAS; python scripts/tune_beat_decode.py --run nl_gq_bd0 --normal-priority && python scripts/tune_beat_decode.py --run nl_gq_rp --prior-file $PRIOR --normal-priority"
tune=$ID
SPECS="nl_gq_bd0+meter_v4_gq_pmhs:tidy,d95,jf,bd,pm,rpm,missing nl_gq_rp+meter_v4_gq_pmhs:tidy,d95,jf,bd,pm,rpm,missing"
submit rpasap "${G[@]}" -c 8 --mem=48G -t 10:00:00 "--dependency=afterok:$tune" --wrap="$PMENV; $JOBS systems $SPECS"
asap=$ID
submit rpgq "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tune" --wrap="$PMENV; $GQSYS $JOBS systems $SPECS"
gq=$ID
others=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '$2 == "r5final" || $2 == "x5final" {print $1}' | paste -sd: -)
submit rpfinal -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq${others:+:$others}" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; source pace/common.sh; python scripts/rhythm_report.py"
echo "the rhythm prior: fit $fit, tuned $tune, whole pieces $asap $gq; outputs/rhythm/report.md: job $ID ($n jobs)"
