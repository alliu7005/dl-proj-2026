#!/bin/bash
# v2, thirteenth round: what a better grid is worth, where one could come from, and meter
# pretraining (on the login node from the repo root, after git pull):
#   bash pace/v2_round13.sh
#   PRETRAIN_AFTER=now bash pace/v2_round13.sh     # pretraining without waiting for the queue
#
# The grid is v2's bottleneck (onset F1 0.78-0.86 on the true grid, 0.26-0.49 on the meter models'),
# and every meter idea so far moved it little.
#   grid      (starts now; ~4 GPU hours) whole pieces, ASAP and Gangqinpu, of the release with:
#             _og   a perfect meter model on its tracked beats where the ground truth reaches (the
#                   ceiling of every meter fix), _ogt with the annotated beats too (beats + meter);
#             _s2g  the seq2seq system's score aligned to the performance as the grid (the seq2seq
#                   system is ~2 MUSTER better on ASAP and writes its bar lines from the notes);
#             _rf   the labeller's bar lines fed back to the meter decoder, labelled again
#             (src/grid_sources.py; scripts/system_scores.py --grid / --refine).
#   pretrain  (waits for every job queued now, i.e. round 12, unless PRETRAIN_AFTER=now) Aria-MIDI's
#             unique subset (32,522 transcriptions, CC-BY-NC-SA) fetched, PM2S beats and beat
#             features made, the meter model pretrained to rebuild masked beats (scripts/
#             pretrain_meter.py, 30k steps), then fine-tuned with the pilot's 12k-step recipe
#             (meter_v4_gq_pt_p12k) against the same recipe from scratch (meter_v4_gq_p12k).
#   job       what                                                          waits for
#   g1tests   the test suites (CPU)                                          -
#   g1asap    _og, _ogt, _s2g, _rf on ASAP                                   g1tests (ok)
#   g1gq      the same on Gangqinpu                                          g1tests (ok)
#   g1final   Beyer & Dai's metrics, outputs/round13/report.md               g1asap, g1gq (any)
#   p1fetch   Aria-MIDI's unique subset (CPU)                                the queue (any)
#   p1prep    PM2S beats + beat features of every file                      p1fetch (ok)
#   p1train   the masked-beat pretraining                                   p1prep (ok)
#   p1ft      meter_v4_gq_pt_p12k fine-tuned, tuned, evaluated              p1train (ok)
#   p1final   outputs/round13/report.md again, with section 2               p1ft, g1final (any)
# Nothing is published (PUBLISH=0).
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
GQOUT=outputs/system_eval_gangqinpu
PRETRAIN_AFTER=${PRETRAIN_AFTER:-queue}
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: bash pace/bd_setup.sh first"; exit 1; }
for f in data/processed/meter/test.pkl data/processed/meter/test_gangqinpu.pkl outputs/runs/sf_base_reg_cab_c128/best.pt \
         "$GQOUT/manifest.json" outputs/system_eval/manifest.json; do
  [ -e "$f" ] || { echo "no $f: an earlier round makes it (pace/README.md)"; exit 1; }
done
export PUBLISH=0
JOBS="PUBLISH=0 bash pace/v2_jobs.sh"
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
FINAL="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''; python3 scripts/round13_report.py"
queued=""
if [ "$PRETRAIN_AFTER" = queue ]; then
  queued=$(squeue -u "$USER" -h -o %i 2>/dev/null | paste -sd: -)
fi
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

submit g1tests -c 8 --mem=32G -t 1:00:00 --wrap="source pace/common.sh; python -m tests.test_grid_sources && python -m tests.test_dur_chooser"
tests=$ID
SPECS="nl_gq+meter_v4_gq:tidy,d95,og,missing nl_gq+meter_v4_gq:tidy,d95,ogt,missing nl_gq+meter_v4_gq:tidy,d95,s2g,missing nl_gq+meter_v4_gq:tidy,d95,rf,missing"
submit g1asap "${G[@]}" -c 8 --mem=48G -t 10:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $JOBS systems $SPECS"
asap=$ID
submit g1gq "${G[@]}" -c 8 --mem=48G -t 10:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $GQSYS $JOBS systems $SPECS"
gq=$ID
echo "the grid's worth (_og, _ogt, _s2g, _rf): jobs $asap (ASAP), $gq (Gangqinpu)"
submit g1final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq" --wrap="$FINAL"
gfinal=$ID
echo "Beyer & Dai's metrics, outputs/round13/report.md: job $gfinal"

fetch_dep=()
[ -n "$queued" ] && fetch_dep=("--dependency=afterany:$queued")
submit p1fetch -c 4 --mem=16G -t 6:00:00 ${fetch_dep[@]+"${fetch_dep[@]}"} --wrap="source pace/common.sh; python scripts/fetch_aria_midi.py --subset unique --max-files 40000"
fetch=$ID
echo "Aria-MIDI's unique subset: job $fetch${queued:+ (after the $(echo "$queued" | tr ':' '\n' | wc -l | tr -d ' ') jobs queued now)}"
submit p1prep "${G[@]}" -c 8 --mem=64G -t 8:00:00 "--dependency=afterok:$fetch" --wrap="source pace/common.sh; python scripts/pretrain_meter.py prep --midi data/raw/aria_midi/unique"
prep=$ID
submit p1train "${G[@]}" -c 8 --mem=64G -t 8:00:00 "--dependency=afterok:$prep" --wrap="source pace/common.sh; python scripts/pretrain_meter.py train --name meter_pt --steps 30000"
train=$ID
submit p1ft "${G[@]}" -c 16 --mem=96G -t 6:00:00 "--dependency=afterok:$train" --wrap="source pace/common.sh; $JOBS meters meter_v4_gq_pt_p12k && $JOBS trackers meter_v4_gq_pt_p12k"
ft=$ID
echo "beats + features, pretraining, fine-tuning: jobs $prep, $train, $ft"
submit p1final -c 4 --mem=16G -t 1:00:00 "--dependency=afterany:$ft:$gfinal" --wrap="source pace/common.sh; python3 scripts/round13_report.py"
echo "outputs/round13/report.md with the pretraining: job $ID ($n jobs); watch: squeue -u \$USER"
