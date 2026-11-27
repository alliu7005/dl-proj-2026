#!/bin/bash
# v2, eleventh round: meter ensembles (on the login node from the repo root, after git pull, with
# room in the queue for 7 jobs):
#   bash pace/v2_ensemble.sh
#
# Why: the grid is v2's bottleneck (the labeller's window onset F1 is 0.78-0.86 on the true grid,
# 0.26-0.49 on the meter models'), and the meter model is noisy from seed to seed: up to 0.09 apart
# in time-signature accuracy (third round) and ~0.9 MUSTER on whole pieces (eighth round), more
# than most changes measured. A larger model overfits. The release's meter_v4_gq is one draw.
# Averaging several seeds' probabilities before the decoder (src/meter.py MeterEnsemble) is the
# standard answer to that variance and costs no new recipe. Measured, each with its own decoder
# tuned on validation, as the release's was:
#   meter_v4_gq_s1, _s2   meter_v4_gq's recipe, seeds 1 and 2 (how far chance alone moves it)
#   meter_v4_gq_tta       the release's model on 5 views of each performance (as is, -3 / +3
#                         semitones, x0.9 / x1.1 time): no training
#   meter_v4_gq_ens3      the three seeds averaged
#   meter_v4_gq_ens3t     the three seeds on the 5 views each
#   meter_v4_gq_ens4      + the published meter_v4_gq (meter_v4_gq_hf), when this account's
#                         meter_v4_gq is a different training of the same recipe (made only then)
# Whole pieces with nl_gq at the release's settings (tidy, d95) on ASAP and Gangqinpu, and the
# deletion threshold at 0.8 (round 7 on the michael branch: small but sure) on the release and on
# ens3t. outputs/round11/report.md says which beat the release (scripts/ensemble_report.py: better
# beyond chance on a condition, worse on none).
#
#   job       what                                                                  waits for
#   e1tests   the test suites (CPU)                                                  -
#   e1prep    nl_gq / meter_v4_gq from the model repo where this account lacks them  e1tests (ok)
#             (scripts/install_run.py); Gangqinpu's whole test pieces if missing
#             (gangqinpu_inputs.py); the release's model on Gangqinpu's real
#             recordings; meter_v4_gq_tta made, tuned, evaluated
#   e1train   meter_v4_gq_s1, meter_v4_gq_s2 trained side by side (~3 h), tuned,      e1tests (ok)
#             evaluated (+ gangqinpu_real); never published (PUBLISH=0)
#   e1ens     meter_v4_gq_ens3, _ens3t (and _ens4) made, tuned, evaluated            e1train, e1prep (ok)
#   e1asap    whole pieces on ASAP (real, clean)                                     e1ens (any), e1prep (ok)
#   e1gq      whole pieces on Gangqinpu (gangqinpu_real, gangqinpu)                  e1ens (any), e1prep (ok)
#   e1final   Beyer & Dai's metrics (ASAP, Gangqinpu), paired.md, the round report   e1asap, e1gq (any)
#
# Again after a stop (finished runs, ensembles and scores are kept):
#   STAGE=eval bash pace/v2_ensemble.sh   # e1ens, e1asap, e1gq, e1final only
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
GQ=${GQ:-data/raw/gangqinpu}
STAGE=${STAGE:-all}
case "$STAGE" in all|eval) ;; *) echo "STAGE must be all or eval, not $STAGE"; exit 1 ;; esac
GQOUT=outputs/system_eval_gangqinpu

# ---- what has to be here already (earlier rounds on this account) ----
missing=0
need() { [ -e "$1" ] || { echo "missing: $1 ($2)"; missing=1; }; }
need data/processed/vocab.json "the data build: pace/data.sbatch"
need data/processed/meter/train_gangqinpu.pkl "Gangqinpu's meter data: pace/v2_gangqinpu.sh (gqdata)"
need data/processed/meter/val_gangqinpu.pkl "the same"
need data/processed/meter/test_gangqinpu.pkl "the same"
need data/processed/meter/tracked_train_gangqinpu_beat_this.npz "Beat This! on Gangqinpu's recordings: pace/v2_gangqinpu.sh (gqtrack)"
need data/processed/meter/tracked_test_gangqinpu_beat_this.npz "the same"
need configs/beat_noise_bt_gangqinpu.json "git pull"
need outputs/system_eval/manifest.json "ASAP's whole-piece inputs: pace/v2_round3.sh (system_inputs.py)"
need "$BD/MIDI2ScoreTF.ckpt" "Beyer & Dai's setup: bash pace/bd_setup.sh"
if [ ! -e "$GQOUT/manifest.json" ]; then   # e1prep makes it from these
  [ -n "$(ls "$GQ"/data/*.parquet 2>/dev/null)" ] || { echo "missing: $GQ/data/*.parquet (pace/v2_gangqinpu.sh)"; missing=1; }
  [ -n "$(ls "$GQ"/audio/test/*.mp3 2>/dev/null)" ] || { echo "missing: $GQ/audio/test/*.mp3 (pace/v2_gangqinpu.sh downloads them)"; missing=1; }
fi
[ "$missing" = 0 ] || { echo "not submitted: make the files above first"; exit 1; }
for r in nl_gq meter_v4_gq; do
  [ -f "outputs/runs/$r/best.pt" ] && echo "$r: this account's" || echo "$r: not here, e1prep installs the published one"
done
queued=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
want=7; [ "$STAGE" = eval ] && want=4
[ $((queued + want)) -le 8 ] || echo "warning: $queued jobs queued already and this adds $want (PACE allows about 8)"

export PUBLISH=0   # nothing from this round goes to the model repo by itself
JOBS="PUBLISH=0 bash pace/v2_jobs.sh"
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
SEEDS="meter_v4_gq meter_v4_gq_s1 meter_v4_gq_s2"
# the systems, decided when the job runs (meter_v4_gq_hf / _ens4 exist only when e1prep made them)
SPECS='S=""; for m in meter_v4_gq meter_v4_gq_s1 meter_v4_gq_s2 meter_v4_gq_tta meter_v4_gq_ens3 meter_v4_gq_ens3t meter_v4_gq_hf meter_v4_gq_ens4; do [ -f outputs/runs/$m/eval_meter.json ] && S="$S nl_gq+$m:tidy,d95,missing"; done; S="$S nl_gq+meter_v4_gq:tidy,d80,missing nl_gq+meter_v4_gq_ens3t:tidy,d80,missing"'

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

PREP='python scripts/install_run.py --run nl_gq || exit 1; '
PREP+='if [ -f outputs/runs/meter_v4_gq/best.pt ]; then python scripts/install_run.py --run meter_v4_gq --as meter_v4_gq_hf --unless-same meter_v4_gq || echo "the published meter_v4_gq: not fetched (no fourth member)"; '
PREP+='else python scripts/install_run.py --run meter_v4_gq || exit 1; fi; '
PREP+='[ -f outputs/runs/meter_v4_gq/decode.json ] || python scripts/eval_meter.py --run meter_v4_gq --tune --tempo-prior || exit 1; '
PREP+="[ -f $GQOUT/manifest.json ] || python scripts/gangqinpu_inputs.py --release $GQ --n 60 || exit 1; "
PREP+='for m in meter_v4_gq meter_v4_gq_hf; do [ -f outputs/runs/$m/best.pt ] || continue; '
PREP+='grep -q "\"gangqinpu_real\"" outputs/runs/$m/eval_meter.json 2>/dev/null || python scripts/eval_meter.py --run $m --real-tracked --conditions gangqinpu || exit 1; done; '
PREP+="$JOBS ensemble meter_v4_gq_tta tta meter_v4_gq"

if [ "$STAGE" = all ]; then
  submit e1tests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; python -m tests.test_ensemble && python -m tests.test_meter"
  tests=$ID
  echo "tests: job $tests"
  submit e1prep "${G[@]}" -c 8 --mem=48G -t 4:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $PREP"
  prep=$ID
  echo "models from the repo where missing, Gangqinpu's inputs if missing, meter_v4_gq_tta: job $prep"
  submit e1train "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $JOBS meters meter_v4_gq_s1 meter_v4_gq_s2 && $JOBS trackers meter_v4_gq_s1 meter_v4_gq_s2"
  train=$ID
  echo "meter_v4_gq_s1, meter_v4_gq_s2: job $train"
  dep_ens="--dependency=afterok:$train:$prep"
else
  dep_ens=""
fi
ENS="$JOBS ensemble meter_v4_gq_ens3 none $SEEDS && $JOBS ensemble meter_v4_gq_ens3t tta $SEEDS || exit 1; if [ -f outputs/runs/meter_v4_gq_hf/best.pt ]; then $JOBS ensemble meter_v4_gq_ens4 none $SEEDS meter_v4_gq_hf || exit 1; fi"
submit e1ens "${G[@]}" -c 8 --mem=48G -t 6:00:00 ${dep_ens:+"$dep_ens"} --wrap="source pace/common.sh; $ENS"
ens=$ID
echo "the ensembles: job $ens"
dep_sys="--dependency=afterany:$ens"
[ "$STAGE" = all ] && dep_sys="--dependency=afterany:$ens,afterok:$prep"
submit e1asap "${G[@]}" -c 8 --mem=48G -t 12:00:00 "$dep_sys" --wrap="source pace/common.sh; $SPECS; $JOBS systems \$S"
asap=$ID
echo "whole pieces, ASAP: job $asap"
submit e1gq "${G[@]}" -c 8 --mem=48G -t 12:00:00 "$dep_sys" --wrap="source pace/common.sh; $SPECS; $GQSYS $JOBS systems \$S"
gq=$ID
echo "whole pieces, Gangqinpu: job $gq"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
REPORT="python3 scripts/ensemble_report.py --extra v2_nl_gq_meter_v4_gq_d80_tidy v2_nl_gq_meter_v4_gq_d95_tidy --extra v2_nl_gq_meter_v4_gq_ens3t_d80_tidy v2_nl_gq_meter_v4_gq_ens3t_d95_tidy"
submit e1final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''; $REPORT"
echo "Beyer & Dai's metrics, paired comparisons, outputs/round11/report.md: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
