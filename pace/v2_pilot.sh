#!/bin/bash
# v2 pilot: every idea for beating the release, at small scale, before any full-scale run (on the
# login node from the repo root, after git pull, with the queue empty: 8 jobs, about 4-6 hours):
#   python scripts/fetch_pdmx.py --max 20000     # once, on the login node (for B; skipped without it)
#   bash pace/v2_pilot.sh
#
#   A  meter ensembles from models this account has trained already (no training): the release's
#      meter model on 5 views of each performance (meter_v4_gq_tta); round 7's three PM2S meter
#      models averaged (meter_v4_gq_pm_ens3) and meter_v4_gq_pm on 5 views (_pm_tta)
#   B  synthetic performances of written scores (src/synthetic.py; 3,000 PDMX scores, 2 renders):
#      meter_v4_gq's recipe for 12k steps without and with them (a quarter of every batch)
#   C  PM2S retrained with the ninth round's stable optimisation, 16k steps at lr 3e-4 and 1e-4:
#      does the loss stay down this time
#   D  the deletion threshold at 0.8 on the release (nl_gq + meter_v4_gq)
#   E  the ceiling of choosing seq2seq or v2 per piece (from the scores, no job)
# outputs/pilot/report.md gives each a go / no go for its full-scale run (pace/v2_ensemble.sh for
# A; B, C: their full runs). Nothing is published (PUBLISH=0).
#
#   job       what                                                               waits for
#   q1tests   the test suites (CPU)                                               -
#   q1prep    nl_gq / meter_v4_gq from the model repo where missing, Gangqinpu's  q1tests (ok)
#             whole test pieces if missing; A's ensembles made, tuned, evaluated
#   q1synth   B's synthetic performances built (CPU)                              q1tests (ok)
#   q1short   B's two 12k-step meter models, tuned, evaluated                     q1synth (ok)
#   q1pm2s    C's two PM2S trainings side by side                                 q1tests (ok)
#   q1asap    whole pieces on ASAP: A's ensembles and their members, D             q1prep (any)
#   q1gq      the same on Gangqinpu (+ seq2seq there, for E)                      q1prep (any)
#   q1final   Beyer & Dai's metrics, paired.md, outputs/pilot/report.md           q1asap, q1gq, q1short, q1pm2s (any)
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
GQ=${GQ:-data/raw/gangqinpu}
PDMX=${PDMX:-data/raw/pdmx/mxl}
N_SCORES=${N_SCORES:-3000}
GQOUT=outputs/system_eval_gangqinpu

missing=0
need() { [ -e "$1" ] || { echo "missing: $1 ($2)"; missing=1; }; }
need data/processed/vocab.json "the data build: pace/data.sbatch"
need data/processed/meter/train_gangqinpu.pkl "Gangqinpu's meter data: pace/v2_gangqinpu.sh"
need data/processed/meter/tracked_train_gangqinpu_beat_this.npz "Beat This! on Gangqinpu's recordings: pace/v2_gangqinpu.sh"
need data/processed/meter/tracked_test_gangqinpu_beat_this.npz "the same"
need configs/beat_noise_bt_gangqinpu.json "git pull"
need outputs/system_eval/manifest.json "ASAP's whole-piece inputs"
need "$BD/MIDI2ScoreTF.ckpt" "Beyer & Dai's setup: bash pace/bd_setup.sh"
if [ ! -e "$GQOUT/manifest.json" ]; then
  [ -n "$(ls "$GQ"/data/*.parquet 2>/dev/null)" ] || { echo "missing: $GQ/data/*.parquet"; missing=1; }
  [ -n "$(ls "$GQ"/audio/test/*.mp3 2>/dev/null)" ] || { echo "missing: $GQ/audio/test/*.mp3"; missing=1; }
fi
[ "$missing" = 0 ] || { echo "not submitted: make the files above first"; exit 1; }
PM_MEMBERS="meter_v4_gq_pm meter_v4_gq_pm_s1 meter_v4_gq_pm_g30"
have_pm=1
for m in $PM_MEMBERS; do [ -f "outputs/runs/$m/best.pt" ] || { echo "no outputs/runs/$m: A's PM2S ensemble skipped"; have_pm=0; }; done
have_synth=1
[ -n "$(ls "$PDMX" 2>/dev/null | head -1)" ] || { echo "no scores in $PDMX: B skipped (python scripts/fetch_pdmx.py --max 20000 on the login node, then again)"; have_synth=0; }
queued=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
[ "$queued" -eq 0 ] || echo "warning: $queued jobs queued already (PACE allows about 8; this adds up to 8)"

export PUBLISH=0
JOBS="PUBLISH=0 bash pace/v2_jobs.sh"
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""

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

submit q1tests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; python -m tests.test_ensemble && python -m tests.test_synthetic && python -m tests.test_meter"
tests=$ID
echo "tests: job $tests"

PREP='python scripts/install_run.py --run nl_gq || exit 1; '
PREP+='[ -f outputs/runs/meter_v4_gq/best.pt ] || python scripts/install_run.py --run meter_v4_gq || exit 1; '
PREP+='[ -f outputs/runs/meter_v4_gq/decode.json ] || python scripts/eval_meter.py --run meter_v4_gq --tune --tempo-prior || exit 1; '
PREP+="[ -f $GQOUT/manifest.json ] || python scripts/gangqinpu_inputs.py --release $GQ --n 60 || exit 1; "
PREP+='grep -q "\"gangqinpu_real\"" outputs/runs/meter_v4_gq/eval_meter.json 2>/dev/null || python scripts/eval_meter.py --run meter_v4_gq --real-tracked --conditions gangqinpu || exit 1; '
PREP+="$JOBS ensemble meter_v4_gq_tta tta meter_v4_gq || exit 1"
if [ "$have_pm" = 1 ]; then
  PREP+="; ENS_KIND=pm2s $JOBS ensemble meter_v4_gq_pm_ens3 none $PM_MEMBERS || exit 1; ENS_KIND=pm2s $JOBS ensemble meter_v4_gq_pm_tta tta meter_v4_gq_pm || exit 1"
fi
submit q1prep "${G[@]}" -c 8 --mem=48G -t 5:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $PREP"
prep=$ID
echo "A: models from the repo where missing, the ensembles: job $prep"

after="$prep"
if [ "$have_synth" = 1 ]; then
  submit q1synth -c 16 --mem=64G -t 3:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; python scripts/build_synthetic_meter.py --scores $PDMX --max $N_SCORES --workers 16"
  synth=$ID
  echo "B: synthetic performances: job $synth"
  submit q1short "${G[@]}" -c 16 --mem=96G -t 4:00:00 "--dependency=afterok:$synth" --wrap="source pace/common.sh; MAX_HOURS=3.5 $JOBS meters meter_v4_gq_p12k meter_v4_gq_syn_p12k && $JOBS trackers meter_v4_gq_p12k meter_v4_gq_syn_p12k"
  after="$after:$ID"
  echo "B: meter_v4_gq_p12k, meter_v4_gq_syn_p12k: job $ID"
fi

STABLE="--betas 0.9,0.999 --eps 1e-8 --clip 1.0 --schedule cosine --warmup 1000 --steps 16000 --eval-every 2000 --max-hours 3"
submit q1pm2s "${G[@]}" -c 16 --mem=96G -t 4:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; python scripts/train_pm2s.py --name pm2s_pilot3e4 --lr 3e-4 $STABLE --normal-priority > slurm-pm2s_pilot3e4-\$SLURM_JOB_ID.log 2>&1 & python scripts/train_pm2s.py --name pm2s_pilot1e4 --lr 1e-4 $STABLE --normal-priority > slurm-pm2s_pilot1e4-\$SLURM_JOB_ID.log 2>&1 & wait"
after="$after:$ID"
echo "C: PM2S, stable settings, 16k steps at two rates: job $ID"

BT="nl_gq+meter_v4_gq:tidy,d95,missing nl_gq+meter_v4_gq_tta:tidy,d95,missing nl_gq+meter_v4_gq:tidy,d80,missing"
PMS=""
if [ "$have_pm" = 1 ]; then
  for nl in nl_base_gn nl_gq; do
    for m in $PM_MEMBERS meter_v4_gq_pm_ens3 meter_v4_gq_pm_tta; do PMS="$PMS $nl+$m:tidy,d95,pm,rpm,missing"; done
  done
fi
SEQ="sf_base_reg_cab_c128:meter:sb"
submit q1asap "${G[@]}" -c 8 --mem=48G -t 12:00:00 "--dependency=afterany:$prep" --wrap="source pace/common.sh; $JOBS systems $BT $PMS"
asap=$ID
echo "whole pieces, ASAP: job $asap"
submit q1gq "${G[@]}" -c 8 --mem=48G -t 12:00:00 "--dependency=afterany:$prep" --wrap="source pace/common.sh; $GQSYS $JOBS systems $BT $PMS $SEQ"
gq=$ID
echo "whole pieces, Gangqinpu: job $gq"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
submit q1final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq:$after" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''; python3 scripts/pilot_report.py"
echo "Beyer & Dai's metrics, paired comparisons, outputs/pilot/report.md: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
