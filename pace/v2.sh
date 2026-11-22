#!/bin/bash
# v2, third round: the whole system, measured end to end (on the login node from the repo root,
# after git pull; needs pace/meter.sh's data and runs, and pace/system_eval.sh run once):
#   bash pace/v2.sh
#   METERS="meter_v3" NEW_NLS="nl_base_gn" SYSTEMS="nl_base_gn+meter_v3" SEQ_SB="" bash pace/v2.sh   # a subset
#   SKIP_SYSTEM=1 bash pace/v2.sh       # no whole-piece comparison with Beyer & Dai
#
# The second round's results (README, "Meter model, second round: results"): PERiScoPe helps, the
# note encoder overfits (best at 4k steps) while the beat features alone still improved at 26k of
# 30k steps, and the labeller on a meter grid keeps only ~0.3 onset F1 of its 0.85 with the true
# grid. This round: longer meter training without the note encoder (3 seeds, one larger model);
# the oracle grid (each meter head certain of the truth on the same tracked beats), which splits
# what the tracked beats cost from what the meter model's errors cost; the labeller trained with
# the tracker's timing on its grid (GridNoise); every labeller on every grid; the windows' onset
# F1 at halved / doubled note values (onset_scaled_f1) for every run's saved predictions; and v2
# on whole pieces next to Beyer & Dai and the seq2seq models, with their metrics.
#
#   job                what                                                          waits for
#   v2tests            the test suites of this round, on PACE's Python (CPU)           -
#   met_<m> a / b      meter model <m>: two 8-hour jobs, the second resuming the       v2tests (b: a)
#                      first (--max-hours), then the decoder tuned on validation and
#                      every test set (eval_meter.py --tune)
#   metev_meter_oracle the oracle grids (no model, CPU)                                v2tests
#   <nl>               labeller trained with GridNoise, then evaluated on the true     v2tests
#                      grid (pace/train_labeller.sbatch)
#   v2_<nl>@<grid>     labeller <nl> on <grid>'s grids, every test set                 that grid's job and
#                      (NLS x the oracle, GRIDS_OLD and METERS; pairs already           <nl>'s job
#                      evaluated are skipped)
#   synthbeats         Beat This! on the clean test MIDI rendered to audio             v2tests
#   sys_v2_<nl>_<m>    v2 on whole pieces (scripts/system_scores.py --v2), SYSTEMS     synthbeats, <m>, <nl>
#   sys_<run>_sb       a seq2seq model with the same rendered-MIDI beats on clean      synthbeats
#   sysfix_<run>       the pieces a scored system has no score for, again (the export's   v2tests
#                      fixes; --only-missing)
#   sysscore           every system in outputs/system_eval rated with Beyer & Dai's    all sys_ jobs (any)
#                      metrics -> outputs/system_eval/results.md
#   rescore            onset_scaled_f1 added to every run's saved predictions, then    all v2_ jobs (any)
#                      outputs/compare (scripts/rescore.py, compare_runs.py)
set -euo pipefail
GPUS=${GPUS:-gpu-h100|gpu-h200|gpu-rtxpro-blackwell}
G=(--gres=gpu:1 "--constraint=$GPUS")
METERS=${METERS-"meter_v3 meter_v3_s1 meter_v3_s2 meter_v3_large"}
NEW_NLS=${NEW_NLS-"nl_base_gn nl_base_gn_s1"}       # labellers trained here
NLS=${NLS-"nl_base $NEW_NLS"}                        # labellers put on every grid
GRIDS_OLD=${GRIDS_OLD-"meter_v2_nonote"}             # earlier meter models whose grids they also get
SYSTEMS=${SYSTEMS-"nl_base+meter_v3 nl_base_gn+meter_v3 nl_base_gn_s1+meter_v3 nl_base_gn+meter_v3_large nl_base+meter_v2_nonote nl_base_gn+meter_v2_nonote"}
SEQ_SB=${SEQ_SB-"sf_base_reg_cab sf_base_reg_cab_c128"}  # seq2seq models also given the rendered-MIDI beats
SEQ_FIX=${SEQ_FIX-"sf_base_reg_cab:meter sf_base_reg_cab_s1:meter sf_base_reg_cab_c128:meter s2s_m_cab:meter"}  # their missing scores
BD=${BD:-$HOME/scratch/beyer_dai}
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""

meter_flags() {  # train_meter.py options of each meter run
  case "$1" in
    meter_v3) echo "--steps 60000 --patience 8" ;;
    meter_v3_s1) echo "--steps 60000 --patience 8 --seed 1" ;;
    meter_v3_s2) echo "--steps 60000 --patience 8 --seed 2" ;;
    meter_v3_large) echo "--steps 60000 --patience 8 --d-model 384 --layers 8" ;;
    *) echo "unknown meter run $1 (add it to meter_flags() in pace/v2.sh)" >&2; return 1 ;;
  esac
}
nl_flags() {  # train_labeller.py options of each labeller trained here
  case "$1" in
    nl_base_gn) echo "--size base --grid-noise 0.8" ;;
    nl_base_gn_s1) echo "--size base --grid-noise 0.8 --seed 1" ;;
    *) echo "unknown labeller run $1 (add it to nl_flags() in pace/v2.sh)" >&2; return 1 ;;
  esac
}
in_list() { case " $2 " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

tests=$(sbatch --parsable -J v2tests -c 8 --mem=64G -t 2:00:00 -o slurm-v2tests-%j.out --wrap="source pace/common.sh; \
python -m tests.test_meter && python -m tests.test_notelabels && python -m tests.test_pipeline && \
python -m tests.test_v2_system && python -m tests.test_system_eval && python -m tests.test_compare_runs")
echo "tests: job $tests"
declare -A job
for m in $METERS; do
  f=$(meter_flags "$m")
  train="source pace/common.sh; python scripts/train_meter.py --name $m $f --workers 6 --normal-priority --max-hours 7.5 \
\$( [ -f outputs/runs/$m/last.pt ] && echo --resume ) && if grep -q '\"phase\": \"done\"' outputs/runs/$m/status.json \
&& [ ! outputs/runs/$m/eval_meter.json -nt outputs/runs/$m/best.pt ]; then python scripts/eval_meter.py --run $m --tune; fi"
  a=$(sbatch --parsable "${G[@]}" -J "met_$m" -c 8 --mem=64G -t 8:00:00 --dependency=afterok:$tests \
      -o "slurm-met_$m-%j.out" --wrap="$train")
  # the second job fails (so nothing waits on an unfinished model) unless training and evaluation are done
  b=$(sbatch --parsable "${G[@]}" -J "met_$m" -c 8 --mem=64G -t 8:00:00 --dependency=afterany:$a \
      -o "slurm-met_$m-%j.out" --wrap="$train; grep -q '\"phase\": \"done\"' outputs/runs/$m/status.json \
&& [ -f outputs/runs/$m/eval_meter.json ]")
  job[$m]=$b
  echo "meter model $m ($f): jobs $a, then $b"
done
job[meter_oracle]=$(sbatch --parsable -J metev_meter_oracle -c 4 --mem=32G -t 2:00:00 --dependency=afterok:$tests \
  -o slurm-metev_meter_oracle-%j.out --wrap="source pace/common.sh; python scripts/eval_meter.py --run meter_oracle \
--oracle --device cpu")
echo "oracle grids: job ${job[meter_oracle]}"
for nl in $NEW_NLS; do
  f=$(nl_flags "$nl")
  job[$nl]=$(NAME="$nl" ARGS="$f" sbatch --parsable "${G[@]}" --dependency=afterok:$tests --export=ALL \
             pace/train_labeller.sbatch)
  echo "labeller $nl ($f): job ${job[$nl]}"
done
evals=()
for nl in $NLS; do
  for m in meter_oracle $GRIDS_OLD $METERS; do
    if ! in_list "$nl" "$NEW_NLS" && ! in_list "$m" "meter_oracle $METERS" \
       && grep -q "\"grid_meter_$m\"" "outputs/runs/$nl/eval_pieces_best.json" 2>/dev/null; then
      echo "  $nl on $m's grids: evaluated before (rescore adds the new metric)"
      continue
    fi
    deps=()
    [ -n "${job[$m]:-}" ] && deps+=("${job[$m]}")
    [ -n "${job[$nl]:-}" ] && deps+=("${job[$nl]}")
    dep=()
    [ ${#deps[@]} -gt 0 ] && dep=("--dependency=afterok:$(IFS=:; echo "${deps[*]}")")
    id=$(sbatch --parsable "${G[@]}" -J "v2_$nl@$m" -c 8 --mem=32G -t 4:00:00 ${dep[@]+"${dep[@]}"} \
         -o "slurm-v2_$nl@$m-%j.out" --wrap="source pace/common.sh; python scripts/eval_labeller.py --run $nl \
--grid meter:$m --conditions clean,synthetic,real,periscope --normal-priority")
    evals+=("$id")
    echo "  $nl on $m's grids: job $id"
  done
done
rdep=()
[ ${#evals[@]} -gt 0 ] && rdep=("--dependency=afterany:$(IFS=:; echo "${evals[*]}")")
re=$(sbatch --parsable -J rescore -c 16 --mem=64G -t 4:00:00 ${rdep[@]+"${rdep[@]}"} -o slurm-rescore-%j.out \
     --wrap="source pace/common.sh; python scripts/rescore.py --workers 16 && python3 scripts/compare_runs.py --quiet")
echo "new metric on every run's saved predictions, then outputs/compare: job $re"
if [ -n "${SKIP_SYSTEM:-}" ]; then
  echo "SKIP_SYSTEM: no whole-piece comparison"
elif [ ! -f outputs/system_eval/manifest.json ] || [ ! -s "$BD/MIDI2ScoreTF.ckpt" ]; then
  echo "no outputs/system_eval/manifest.json or $BD/MIDI2ScoreTF.ckpt: run bash pace/bd_setup.sh and bash pace/system_eval.sh first"
else
  sb=$(sbatch --parsable "${G[@]}" -J synthbeats -c 8 --mem=32G -t 2:00:00 --dependency=afterok:$tests \
       -o slurm-synthbeats-%j.out --wrap="source pace/common.sh; python scripts/synth_beats.py")
  echo "Beat This! on the rendered clean MIDI: job $sb"
  sys=()
  for spec in $SYSTEMS; do
    nl=${spec%%+*}; m=${spec##*+}
    deps=("$sb")
    [ -n "${job[$m]:-}" ] && deps+=("${job[$m]}")
    [ -n "${job[$nl]:-}" ] && deps+=("${job[$nl]}")
    id=$(sbatch --parsable "${G[@]}" -J "sys_v2_${nl}_$m" -c 8 --mem=48G -t 3:00:00 \
         "--dependency=afterok:$(IFS=:; echo "${deps[*]}")" -o "slurm-sys_v2_${nl}_$m-%j.out" \
         --wrap="source pace/common.sh; python scripts/system_scores.py --v2 $spec")
    sys+=("$id")
    echo "  v2 $nl + $m on whole pieces: job $id"
  done
  for r in $SEQ_SB; do
    id=$(sbatch --parsable "${G[@]}" -J "sys_${r}_sb" -c 8 --mem=48G -t 3:00:00 --dependency=afterok:$sb \
         -o "slurm-sys_${r}_sb-%j.out" --wrap="source pace/common.sh; python scripts/system_scores.py --run $r \
--midi-beats synth")
    sys+=("$id")
    echo "  $r with the rendered-MIDI beats on whole pieces: job $id"
  done
  for spec in $SEQ_FIX; do
    r=${spec%%:*}; mode=${spec##*:}
    [ -f "outputs/system_eval/${r}_$mode/summary.json" ] || continue
    id=$(sbatch --parsable "${G[@]}" -J "sysfix_$r" -c 8 --mem=48G -t 2:00:00 --dependency=afterok:$tests          -o "slurm-sysfix_$r-%j.out" --wrap="source pace/common.sh; python scripts/system_scores.py --run $r --mode $mode --only-missing")
    sys+=("$id")
    echo "  ${r}_$mode: its missing scores again: job $id"
  done
  sc=$(sbatch --parsable -J sysscore -c 16 --mem=64G -t 6:00:00 "--dependency=afterany:$(IFS=:; echo "${sys[*]}")" \
       -o slurm-sysscore-%j.out --wrap="$BDENV; python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer \
--systems all --jobs 16")
  echo "every system rated with Beyer & Dai's metrics: job $sc (outputs/system_eval/results.md)"
fi
echo "watch: squeue -u \$USER     then: python3 scripts/compare_runs.py   (outputs/compare/summary.md)"
