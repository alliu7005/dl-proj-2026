#!/bin/bash
# v2, fifteenth round: the release candidate built on PM2S's beats (on the login node from the repo
# root, after git pull; 14 jobs):
#   bash pace/v2_round15.sh
#
# Why: round 9's stably retrained PM2S (pm2s_hos) with its meter model (meter_v4_gq_pmhs) beat the
# release by 2.33 MUSTER on real ASAP, 4.29 on clean MIDI and 0.33 / 0.81 on Gangqinpu, the largest
# gain of any change: its beats have far fewer tempo-octave errors (0.12 against Beat This!'s 0.33
# on real), the costliest beat error (outputs/beat_errors/report.md). Round 12's seq2seq with
# Gangqinpu (sf_c128_gq30) is now the system to beat. Four experiments side by side:
#   A  the candidate, no training: nl_gq + meter_v4_gq_pmhs, tidy, d95, stretched bars shortened (_jf,
#      better everywhere in the simplicity round), PM2S's beats; at d80 too (MIDI input's setting,
#      read clean)
#   B  its meter from the Aria-MIDI pretraining (round 14: the pretrained ensemble beat the release on
#      both Gangqinpu conditions): meter_v4_gq_pmhs_ptr, three seeds, their ensemble, and the four
#      with meter_v4_gq_pmhs; each in the candidate
#   C  seq2seq given PM2S's beats of the notes (sf_c128_gq30, _spm) instead of Beat This!'s
#   D  v2 on the seq2seq score's grid, the new seq2seq's (_s2g_sf_c128_gq30; round 13's _s2g failed)
# outputs/round15/report.md after r5final. Nothing is published (PUBLISH=0).
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
GQOUT=outputs/system_eval_gangqinpu
PM_RUN=${PM_RUN:-pm2s_hos}
PM_TAG=${PM_TAG:-hos}
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: bash pace/bd_setup.sh first"; exit 1; }
for f in "outputs/runs/$PM_RUN/best.pt" outputs/runs/meter_v4_gq_pmhs/eval_meter.json configs/beat_noise_pm2s_hos.json \
         outputs/runs/meter_pt/best.pt outputs/runs/sf_c128_gq30/best.pt outputs/runs/nl_gq/config.json \
         data/processed/vocab.json "$GQOUT/manifest.json" outputs/system_eval/manifest.json; do
  [ -e "$f" ] || { echo "no $f: an earlier round makes it (round 9 PM2S, 12 seq2seq, 13 pretraining)"; exit 1; }
done
export PUBLISH=0
JOBS="PUBLISH=0 bash pace/v2_jobs.sh"
# every job that runs PM2S gets the stably retrained weights and their tag (as pace/v2_pm2s2.sh)
PMENV="source pace/common.sh; export PM2S_WEIGHTS=\"\$SLURM_SUBMIT_DIR/outputs/runs/$PM_RUN/best.pt\" PM2S_TAG=$PM_TAG"
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
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
submit r5tests -c 8 --mem=32G -t 1:00:00 --wrap="source pace/common.sh; python -m tests.test_round15 && python -m tests.test_round14"
tests=$ID
ends=""

# A: the candidate (no training)
CAND="nl_gq+meter_v4_gq_pmhs:tidy,d95,jf,pm,rpm,missing"
submit c5asap "${G[@]}" -c 8 --mem=48G -t 10:00:00 "--dependency=afterok:$tests" --wrap="$PMENV; $JOBS systems $CAND && SYS_ARGS=\"--conditions clean\" $JOBS systems nl_gq+meter_v4_gq_pmhs:tidy,d80,jf,pm,rpm,missing"
ends="$ends:$ID"
submit c5gq "${G[@]}" -c 8 --mem=48G -t 6:00:00 "--dependency=afterok:$tests" --wrap="$PMENV; $GQSYS $JOBS systems $CAND"
ends="$ends:$ID"
echo "A, the candidate:${ends//:/ }"

# B: its meter pretrained, three seeds, ensembled
SEEDS="meter_v4_gq_pmhs_ptr meter_v4_gq_pmhs_ptr_s1 meter_v4_gq_pmhs_ptr_s2"
submit m5train "${G[@]}" -c 16 --mem=96G -t 10:00:00 "--dependency=afterok:$tests" --wrap="$PMENV; $JOBS meters $SEEDS && $JOBS trackers $SEEDS"
mtrain=$ID
ENS="ENS_KIND=pm2s ENS_NOISE=configs/beat_noise_pm2s_hos.json"
submit m5ens "${G[@]}" -c 8 --mem=48G -t 6:00:00 "--dependency=afterok:$mtrain" --wrap="$PMENV; $ENS $JOBS ensemble meter_v4_gq_pmhs_ptr_ens3 none $SEEDS && $ENS $JOBS ensemble meter_v4_gq_pmhs_ens4 none meter_v4_gq_pmhs $SEEDS"
mens=$ID
SPECS=""
for m in $SEEDS meter_v4_gq_pmhs_ptr_ens3 meter_v4_gq_pmhs_ens4; do SPECS="$SPECS nl_gq+$m:tidy,d95,jf,pm,rpm,missing"; done
submit m5asap "${G[@]}" -c 8 --mem=48G -t 16:00:00 "--dependency=afterany:$mens" --wrap="$PMENV; $JOBS systems $SPECS"
ends="$ends:$ID"
submit m5gq "${G[@]}" -c 8 --mem=48G -t 10:00:00 "--dependency=afterany:$mens" --wrap="$PMENV; $GQSYS $JOBS systems $SPECS"
ends="$ends:$ID"
echo "B, pretrained meters for PM2S: train $mtrain, ensembles $mens"

# C: seq2seq with PM2S's beats
submit q5asap "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$PMENV; $JOBS systems sf_c128_gq30:meter:spm"
ends="$ends:$ID"
submit q5gq "${G[@]}" -c 8 --mem=48G -t 6:00:00 "--dependency=afterok:$tests" --wrap="$PMENV; $GQSYS $JOBS systems sf_c128_gq30:meter:spm"
ends="$ends:$ID"
echo "C, seq2seq with PM2S's beats:${ends##*:}"

# D: v2 on the new seq2seq's grid
S2G="nl_gq+meter_v4_gq:tidy,d95,s2gq,missing"
submit g5asap "${G[@]}" -c 8 --mem=48G -t 10:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $JOBS systems $S2G"
ends="$ends:$ID"
submit g5gq "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $GQSYS $JOBS systems $S2G"
ends="$ends:$ID"
echo "D, v2 on seq2seq's grid: $ID"

submit r5final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany${ends}" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; source pace/common.sh; python3 scripts/round15_report.py"
echo "Beyer & Dai's metrics, outputs/round15/report.md: job $ID ($n jobs); watch: squeue -u \$USER"
