#!/bin/bash
# v2, fourteenth round: four experiments side by side, none waiting for another (on the login node
# from the repo root, after git pull; 19 jobs):
#   bash pace/v2_round14.sh
#
#   A  pretrained meters: meter_v4_gq's full recipe (60k steps) from the Aria-MIDI pretraining
#      (outputs/runs/meter_pt, round 13: its 12k pilot matched the release on real ASAP), three seeds
#      side by side, their ensemble (and with the 5 views), whole pieces against the release and
#      against round 11's ensemble
#   B  more pretraining data: Aria-MIDI's deduped subset (~150k of its 371k files), PM2S beats in 4
#      shards, the masked-beat pretraining on them (meter_pt150), the pilot's 12k recipe from it
#      (meter_v4_gq_pt150_p12k) against round 13's from 32k files
#   C  which beat-tracking errors cost the most (scripts/beat_errors.py; CPU; waits for round 13's
#      g1final if still queued, whose scores it reads)
#   D  a pretrained labeller: masked-note pretraining on Aria-MIDI (scripts/pretrain_labeller.py), then
#      nl_gq's recipe for 20k steps from it (nl_gq_ptl_p20k) against the same from scratch (nl_gq_p20k,
#      the articulation pilot's), test windows and whole pieces
# outputs/round14/report.md (A, B, D) after r4final; outputs/beat_errors/report.md (C).
# Nothing is published (PUBLISH=0).
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
GQOUT=outputs/system_eval_gangqinpu
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: bash pace/bd_setup.sh first"; exit 1; }
for f in outputs/runs/meter_pt/best.pt data/raw/aria_midi/unique/fetch.json outputs/runs/nl_gq/config.json \
         data/processed/vocab.json "$GQOUT/manifest.json" outputs/system_eval/manifest.json; do
  [ -e "$f" ] || { echo "no $f: an earlier round makes it (round 13 for the pretraining)"; exit 1; }
done
export PUBLISH=0
JOBS="PUBLISH=0 bash pace/v2_jobs.sh"
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
submit r4tests -c 8 --mem=32G -t 1:00:00 --wrap="source pace/common.sh; python -m tests.test_round14 && python -m tests.test_grid_sources"
tests=$ID
ends=""

# A: pretrained meters, three seeds, their ensembles
SEEDS="meter_v4_gq_ptr meter_v4_gq_ptr_s1 meter_v4_gq_ptr_s2"
submit m4train "${G[@]}" -c 16 --mem=96G -t 10:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $JOBS meters $SEEDS && $JOBS trackers $SEEDS"
mtrain=$ID
submit m4ens "${G[@]}" -c 8 --mem=48G -t 6:00:00 "--dependency=afterok:$mtrain" --wrap="source pace/common.sh; $JOBS ensemble meter_v4_gq_ptr_ens3 none $SEEDS && $JOBS ensemble meter_v4_gq_ptr_ens3t tta $SEEDS"
mens=$ID
SPECS=""
# (round 11's meter_v4_gq_ens3t has its own whole-piece jobs, e1asap and e1gq)
for m in $SEEDS meter_v4_gq_ptr_ens3 meter_v4_gq_ptr_ens3t; do SPECS="$SPECS nl_gq+$m:tidy,d95,missing"; done
submit m4asap "${G[@]}" -c 8 --mem=48G -t 10:00:00 "--dependency=afterany:$mens" --wrap="source pace/common.sh; $JOBS systems $SPECS"
ends="$ends:$ID"
submit m4gq "${G[@]}" -c 8 --mem=48G -t 6:00:00 "--dependency=afterany:$mens" --wrap="source pace/common.sh; $GQSYS $JOBS systems $SPECS"
ends="$ends:$ID"
echo "A, pretrained meters: train $mtrain, ensembles $mens, whole pieces${ends//:/ }"

# B: more pretraining data
submit d4fetch -c 4 --mem=16G -t 4:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; python scripts/fetch_aria_midi.py --subset deduped --max-files 150000"
fetch=$ID
shards="" stores=""
for i in 0 1 2 3; do
  submit "d4prep$i" "${G[@]}" -c 8 --mem=64G -t 8:00:00 "--dependency=afterok:$fetch" --wrap="source pace/common.sh; python scripts/pretrain_meter.py prep --midi data/raw/aria_midi/deduped --shard $i/4 --out data/processed/meter/pretrain_dd$i.pkl"
  shards="$shards:$ID"
  stores="${stores:+$stores,}data/processed/meter/pretrain_dd$i.pkl"
done
submit d4train "${G[@]}" -c 8 --mem=96G -t 8:00:00 "--dependency=afterok${shards}" --wrap="source pace/common.sh; python scripts/pretrain_meter.py train --name meter_pt150 --data $stores --steps 30000"
dtrain=$ID
submit d4ft "${G[@]}" -c 16 --mem=96G -t 6:00:00 "--dependency=afterok:$dtrain" --wrap="source pace/common.sh; $JOBS meters meter_v4_gq_pt150_p12k && $JOBS trackers meter_v4_gq_pt150_p12k"
ends="$ends:$ID"
echo "B, more pretraining data: fetch $fetch, prep${shards//:/ }, pretrain $dtrain, fine-tune $ID"

# C: beat errors (round 13's scores; after its g1final if still queued)
g1=$(squeue -u "$USER" -h -n g1final -o %i 2>/dev/null | paste -sd: -)
submit b4errors -c 4 --mem=16G -t 1:00:00 "--dependency=afterok:$tests${g1:+,afterany:$g1}" --wrap="source pace/common.sh; python scripts/beat_errors.py"
echo "C, beat errors: job $ID -> outputs/beat_errors/report.md"

# D: a pretrained labeller
submit l4prep -c 8 --mem=64G -t 6:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; python scripts/pretrain_labeller.py prep --midi data/raw/aria_midi/unique"
lprep=$ID
submit l4train "${G[@]}" -c 8 --mem=96G -t 10:00:00 "--dependency=afterok:$lprep" --wrap="source pace/common.sh; python scripts/pretrain_labeller.py train --name nl_pt --steps 30000"
ltrain=$ID
submit l4ft "${G[@]}" -c 16 --mem=96G -t 16:00:00 "--dependency=afterok:$ltrain" --wrap="source pace/common.sh; $JOBS labellers nl_gq_ptl_p20k"
lft=$ID
LSPECS="nl_gq_ptl_p20k+meter_v4_gq:tidy,d95,missing"
submit l4asap "${G[@]}" -c 8 --mem=48G -t 6:00:00 "--dependency=afterok:$lft" --wrap="source pace/common.sh; $JOBS systems $LSPECS"
ends="$ends:$ID"
submit l4gq "${G[@]}" -c 8 --mem=48G -t 4:00:00 "--dependency=afterok:$lft" --wrap="source pace/common.sh; $GQSYS $JOBS systems $LSPECS"
ends="$ends:$ID"
echo "D, pretrained labeller: prep $lprep, pretrain $ltrain, fine-tune $lft"

submit r4final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany${ends}" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; source pace/common.sh; python3 scripts/round14_report.py"
echo "Beyer & Dai's metrics, outputs/round14/report.md: job $ID ($n jobs); watch: squeue -u \$USER"
