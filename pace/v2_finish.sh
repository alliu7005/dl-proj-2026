#!/bin/bash
# The rest of the third round (on the login node from the repo root, after git pull; after pace/v2.sh):
#   bash pace/v2_finish.sh
#
# What pace/v2.sh's results left open (README, "Third round: results"):
#   * the best seq2seq recipe (sf_base_reg_cab_c128) trained on the rebuilt data, so it is compared
#     with v2 on the same targets: RETRAIN, three chained 8-hour jobs (training resumes from last.pt,
#     then the evaluation: only what is missing), then its whole-piece scores;
#   * three seq2seq systems scored before but never written again with the current export (their
#     earlier run hit the time limit before its summary): OLD_SYSTEMS;
#   * what the rendered MIDI costs v2's beats on clean MIDI: v2 on the clean test MIDI with the beats
#     Beat This! tracked in the real recordings (the 24 performances that have one), next to the same
#     system with the rendered-MIDI beats (outputs/system_eval_ablation): ABLATION.
# At most 8 jobs (PACE's per-user limit):
#
#   job               what                                                           waits for
#   seq_<run> (x3)    train RETRAIN (resuming), then evaluate it (SEQ_MODES on the      the job before (any)
#                     four test sets)
#   sysold_a/b        OLD_SYSTEMS on whole pieces (the second also the ablation)      -
#   sysnew            RETRAIN on whole pieces, with and without the rendered-MIDI      the last seq_ job (any)
#                     beats on clean MIDI (skipped if it is not finished)
#   v2final           the results scored before the rebuild re-scored on the current   all of the above (any)
#                     targets (scripts/rescore.py: onset-scale F1 for the seq2seq runs);
#                     outputs/compare; every system rated with Beyer & Dai's metrics
#                     (results.md), and the ablation on its shared pieces
set -euo pipefail
GPUS=${GPUS:-gpu-h100|gpu-h200|gpu-rtxpro-blackwell}
G=(--gres=gpu:1 "--constraint=$GPUS")
RETRAIN=${RETRAIN-"sf_base_reg_cab_c128_r"}
RETRAIN_ARGS=${RETRAIN_ARGS-"--size base --dropout 0.2 --weight-decay 0.1 --lr 2e-4 --context-before 128 --context-after 64 --aux-metrical --beats"}
OLD_SYSTEMS=${OLD_SYSTEMS-"sf_base_reg_cab:meter sf_base_reg_cab_s1:meter sf_base_reg_cab_c128:meter"}
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

all=()
last=""
if [ -n "$RETRAIN" ]; then
  for k in 1 2 3; do
    dep=()
    [ -n "$last" ] && dep=("--dependency=afterany:$last")
    submit "seq_$RETRAIN" "${G[@]}" -c 8 --mem=48G -t 8:00:00 ${dep[@]+"${dep[@]}"} \
      --wrap="$JOBS seq2seq $RETRAIN $RETRAIN_ARGS"
    last=$ID
    all+=("$ID")
  done
  echo "$RETRAIN ($RETRAIN_ARGS): jobs ${all[*]} (each resumes the one before)"
fi
read -r -a old <<< "$OLD_SYSTEMS"
half=$(( (${#old[@]} + 1) / 2 ))
for part in a b; do
  if [ $part = a ]; then specs=("${old[@]:0:$half}"); else specs=("${old[@]:$half}"); fi
  wrap=""
  [ ${#specs[@]} -gt 0 ] && wrap="$JOBS systems ${specs[*]}"
  if [ $part = b ] && [ -n "$ABLATION" ]; then
    wrap="${wrap:+$wrap; }$JOBS ablation $ABLATION"
  fi
  [ -n "$wrap" ] || continue
  submit "sysold_$part" "${G[@]}" -c 8 --mem=48G -t 8:00:00 --wrap="$wrap"
  all+=("$ID")
  echo "whole pieces (${specs[*]:-}${ABLATION:+ + ablation $ABLATION}): job $ID"
done
if [ -n "$RETRAIN" ]; then
  submit sysnew "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterany:$last" \
    --wrap="$JOBS systems $RETRAIN:meter:sb $RETRAIN:meter"
  all+=("$ID")
  echo "$RETRAIN on whole pieces: job $ID"
fi
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16"
submit v2final -c 16 --mem=64G -t 8:00:00 "--dependency=afterany:$(IFS=:; echo "${all[*]}")" \
  --wrap="source pace/common.sh; python scripts/rescore.py --workers 16; python3 scripts/compare_runs.py --quiet; \
$BDENV; $SCORE --systems all; \
if [ -d outputs/system_eval_ablation ]; then $SCORE --systems all --conditions clean --out outputs/system_eval_ablation; fi"
echo "outputs/compare + every system rated with Beyer & Dai's metrics + the ablation: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
