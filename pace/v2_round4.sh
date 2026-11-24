#!/bin/bash
# v2, fourth round: the rhythm and the written score (on the login node from the repo root, after
# git pull; after pace/v2.sh):
#   bash pace/v2_round4.sh
#
# What the third round left (docs/experiments.md, "Third round: results"; "Fourth round"):
#   * the meter model's note value: 3/8 written for 3/4, 2/4 for 2/2 costs ~0.15 of the 0.21 onset
#     F1 the meter model loses on PERiScoPe. meter_v3_tp: meter_v3's decoder tuned again with a prior
#     on the written tempo each step implies (src/meter.py tempo_logprior; weight 0 = as before);
#   * the labeller's checkpoint: chosen on validation loss, which rose while the decoded onset F1
#     still improved. nl_base_gn_f1: kept and stopped on validation onset F1 (--select onset_f1);
#     nl_base_gn1_f1: the same with every training window on the tracker-timed grid (--grid-noise 1);
#   * written lengths that run just past the next note (extra voices): :tidy (tidy_lengths);
#   * notes left out: the labeller's "delete" covers notes really played; :d95 / :keepall;
#   * what the rendered-MIDI beats cost v2 on clean MIDI: the beats of the real recordings instead
#     (ablation, 24 performances).
# At most 8 jobs (PACE's per-user limit):
#
#   job          what                                                              waits for
#   r4tests      the test suites, on PACE's Python (CPU)                            -
#   r4lab        nl_base_gn_f1 and nl_base_gn1_f1 side by side, then each on the    r4tests (ok)
#                true grid
#   r4tempo      meter_v3_tp: the decoder tuned with the tempo prior, evaluated,     r4tests (ok)
#                its grids saved (CPU)
#   r4evals      the new labellers and nl_base_gn on the oracle grid, meter_v3 and   r4lab, r4tempo (any)
#                meter_v3_tp, every test set
#   r4sys_a/b    v2 on whole pieces with the new models (SYS_A, SYS_B)                r4lab, r4tempo (any)
#   r4sys_c      the round-3 system tidied, at d95 and keeping every note, and the   r4tests (ok)
#                beat ablation (SYS_C, ABLATION)
#   r4final      outputs/compare; every system rated with Beyer & Dai's metrics,     all of the above (any)
#                and the ablation on its shared pieces
set -euo pipefail
GPUS=${GPUS:-cuda13}  # any NVIDIA GPU with bf16 (A100, A40, L40S, H100, H200): these jobs are short
                      # (pace/README.md, "Which GPU"); H100 / H200 only: GPUS=HX00
G=(--gres=gpu:1 "--constraint=$GPUS")
NEW_LABS=${NEW_LABS-"nl_base_gn_f1 nl_base_gn1_f1"}
TEMPO_FROM=${TEMPO_FROM-"meter_v3"}
TEMPO_RUN=${TEMPO_RUN-"meter_v3_tp"}
export METERS="$TEMPO_FROM $TEMPO_RUN"   # the grids every labeller is evaluated on (with the oracle)
export NEW_NLS="$NEW_LABS"
export GRIDS_OLD=""
SYS_A=${SYS_A-"nl_base_gn+meter_v3_tp nl_base_gn_f1+meter_v3 nl_base_gn_f1+meter_v3_tp nl_base_gn_f1+meter_v3_tp:tidy,d95"}
SYS_B=${SYS_B-"nl_base_gn1_f1+meter_v3 nl_base_gn1_f1+meter_v3_tp nl_base_gn1_f1+meter_v3_tp:tidy,d95"}
SYS_C=${SYS_C-"nl_base_gn+meter_v3:tidy nl_base_gn+meter_v3:d95,missing nl_base_gn+meter_v3:keepall,missing"}
ABLATION=${ABLATION-"nl_base_gn+meter_v3"}
BD=${BD:-$HOME/scratch/beyer_dai}
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
JOBS="source pace/common.sh; bash pace/v2_jobs.sh"
if [ ! -f outputs/system_eval/manifest.json ] || [ ! -s "$BD/MIDI2ScoreTF.ckpt" ]; then
  echo "no outputs/system_eval/manifest.json or $BD/MIDI2ScoreTF.ckpt: run bash pace/bd_setup.sh and bash pace/system_eval.sh first"
  exit 1
fi
n=0
ID=""
submit() {  # name, then sbatch options and --wrap: the job id in $ID, counted in $n
  local name=$1
  shift
  local kill=()
  case " $* " in *" --dependency=afterok:"*) kill=(--kill-on-invalid-dep=yes) ;; esac
  ID=$(sbatch --parsable -J "$name" -o "slurm-$name-%j.out" ${kill[@]+"${kill[@]}"} "$@")
  n=$((n + 1))
}

submit r4tests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; \
python -m tests.test_meter && python -m tests.test_notelabels && python -m tests.test_pipeline && \
python -m tests.test_v2_system && python -m tests.test_system_eval && python -m tests.test_compare_runs && \
python -m tests.test_preprocessing && python -m tests.test_context_notes && python -m tests.test_training_stop"
tests=$ID
echo "tests: job $tests"
all=("$tests")
models=()
if [ -n "$NEW_LABS" ]; then
  submit r4lab "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$JOBS labellers $NEW_LABS"
  models+=("$ID"); all+=("$ID")
  echo "labellers $NEW_LABS: job $ID"
fi
if [ -n "$TEMPO_RUN" ]; then
  submit r4tempo -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$JOBS tempo $TEMPO_FROM $TEMPO_RUN"
  models+=("$ID"); all+=("$ID")
  echo "$TEMPO_RUN ($TEMPO_FROM, decoder tuned with the tempo prior): job $ID"
fi
mdep=()
[ ${#models[@]} -gt 0 ] && mdep=("--dependency=afterany:$(IFS=:; echo "${models[*]}")")
submit r4evals "${G[@]}" -c 8 --mem=48G -t 8:00:00 ${mdep[@]+"${mdep[@]}"} --wrap="$JOBS evals $NEW_LABS nl_base_gn"
all+=("$ID")
echo "labellers on every grid: job $ID"
for part in a b; do
  if [ $part = a ]; then specs=$SYS_A; else specs=$SYS_B; fi
  [ -n "$specs" ] || continue
  submit "r4sys_$part" "${G[@]}" -c 8 --mem=48G -t 8:00:00 ${mdep[@]+"${mdep[@]}"} --wrap="$JOBS systems $specs"
  all+=("$ID")
  echo "whole pieces ($specs): job $ID"
done
wrap=""
[ -n "$SYS_C" ] && wrap="$JOBS systems $SYS_C"
[ -n "$ABLATION" ] && wrap="${wrap:+$wrap; }$JOBS ablation $ABLATION"
if [ -n "$wrap" ]; then
  submit r4sys_c "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$wrap"
  all+=("$ID")
  echo "whole pieces ($SYS_C${ABLATION:+ + ablation $ABLATION}): job $ID"
fi
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16"
submit r4final -c 16 --mem=64G -t 8:00:00 "--dependency=afterany:$(IFS=:; echo "${all[*]}")" \
  --wrap="source pace/common.sh; python3 scripts/compare_runs.py --quiet; $BDENV; $SCORE --systems all; \
if [ -d outputs/system_eval_ablation ]; then $SCORE --systems all --conditions clean --out outputs/system_eval_ablation; fi"
echo "outputs/compare + every system rated with Beyer & Dai's metrics + the ablation: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
