#!/bin/bash
# v2, third round: the whole system, measured end to end (on the login node from the repo root,
# after git pull; needs pace/meter.sh's data and runs, and pace/system_eval.sh run once):
#   bash pace/v2.sh                 # wave 1; its last job submits wave 2 when the rest has finished
#   bash pace/v2.sh wave2           # wave 2 by hand (if that job could not submit it)
#   SKIP_SYSTEM=1 bash pace/v2.sh   # without the whole-piece comparison with Beyer & Dai
#   AFTER=<job> bash pace/v2.sh     # wave 1 starts once that job has succeeded (pace/rebuild.sh)
#
# PACE limits how many jobs a user may have submitted at once, so several models share one GPU job
# (pace/v2_jobs.sh does the work) and the batch comes in two waves of at most 8 jobs each.
#
# The second round's results (docs/experiments.md, "Meter model, second round: results"): PERiScoPe helps, the
# note encoder overfits, the labeller on a meter grid keeps only ~0.35 onset F1 of its 0.85 with the
# true grid. This round: longer meter training without the note encoder (3 seeds, one larger
# model); the oracle grid (every meter head certain of the truth on the same tracked beats), which
# splits what the tracked beats cost from what the meter model's errors cost; the labeller trained
# with the tracker's timing on its grid (GridNoise); every labeller on every grid; the windows'
# onset F1 at halved / doubled note values for every run's saved predictions; and v2 on whole
# pieces next to Beyer & Dai and the seq2seq models, with their metrics.
#
# wave 1                                                                          waits for
#   v2tests           the test suites, on PACE's Python (CPU)                       -
#   met_<a>+<b> (x2)  two meter models side by side in one GPU job, up to 7.5 h;     v2tests
#                     a second job resumes them, then tunes the decoder and           (2nd: the 1st)
#                     evaluates each (pace/v2_jobs.sh meters). METER_GROUPS: 2 pairs
#   labellers         nl_base_gn and nl_base_gn_s1 (--grid-noise 0.8) side by side,  v2tests
#                     then each evaluated on the true grid
#   v2grids           the earlier meter models' grids decoded again (fixed decoder),  v2tests
#                     the oracle grids, then Beat This! on the clean test MIDI
#                     rendered to audio (CPU)
#   v2next            submits wave 2                                                 all of wave 1 (any)
# wave 2
#   evals_<nl> (x3)   labeller <nl> on the true grid and every grid: oracle,          -
#                     GRIDS_OLD, METERS (the first job also NLS_EXTRA: nl_large)
#   sys_v2            v2 on whole pieces: SYSTEMS (finished models only)              the evals (any)
#   sys_seq_a/b       seq2seq: the two best with the rendered-MIDI beats on clean    -
#                     (SEQ_SB), and those scored before written again with the
#                     current export, so spelling and missing pieces compare (SEQ_RERUN)
#   v2final           onset_scaled_f1 for every run's saved predictions and          all of the above (any)
#                     outputs/compare; then every system in outputs/system_eval
#                     rated with Beyer & Dai's metrics -> outputs/system_eval/results.md
set -euo pipefail
WAVE=${1:-wave1}
GPUS=${GPUS:-HX00}
G=(--gres=gpu:1 "--constraint=$GPUS")
export METERS=${METERS-"meter_v3 meter_v3_s1 meter_v3_s2 meter_v3_large"}
export NEW_NLS=${NEW_NLS-"nl_base_gn nl_base_gn_s1"}
export GRIDS_OLD=${GRIDS_OLD-"meter_v2_nonote"}
NLS=${NLS-"nl_base $NEW_NLS"}
NLS_EXTRA=${NLS_EXTRA-"nl_large"}   # evaluated again in the first labeller's job (its results predate this evaluation)
METER_GROUPS=${METER_GROUPS-"meter_v3,meter_v3_s1 meter_v3_s2,meter_v3_large"}   # one GPU job per group
SYSTEMS=${SYSTEMS-"nl_base+meter_v3 nl_base_gn+meter_v3 nl_base_gn_s1+meter_v3 nl_base_gn+meter_v3_large nl_base+meter_v2_nonote nl_base_gn+meter_v2_nonote"}
SEQ_SB=${SEQ_SB-"sf_base_reg_cab:meter sf_base_reg_cab_c128:meter"}
SEQ_RERUN=${SEQ_RERUN-"sf_base_reg_cab:meter sf_base_reg_cab_s1:meter sf_base_reg_cab_c128:meter s2s_m_cab:meter"}
BD=${BD:-$HOME/scratch/beyer_dai}
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
JOBS="source pace/common.sh; bash pace/v2_jobs.sh"
n=0
ID=""
submit() {  # name, then sbatch options and --wrap: the job id in $ID, counted in $n
  local name=$1
  shift
  # a job waiting on another's success is cancelled if that fails, instead of pending forever
  # (which would also hold back every afterany job behind it)
  local kill=()
  case " $* " in *" --dependency=afterok:"*) kill=(--kill-on-invalid-dep=yes) ;; esac
  ID=$(sbatch --parsable -J "$name" -o "slurm-$name-%j.out" ${kill[@]+"${kill[@]}"} "$@")
  n=$((n + 1))
}

if [ "$WAVE" = wave1 ]; then
  after=()  # AFTER=<job id>: start once that job (e.g. pace/rebuild.sh's) has succeeded
  [ -n "${AFTER:-}" ] && after=("--dependency=afterok:$AFTER")
  submit v2tests -c 8 --mem=64G -t 2:00:00 ${after[@]+"${after[@]}"} --wrap="source pace/common.sh; \
python -m tests.test_meter && python -m tests.test_notelabels && python -m tests.test_pipeline && \
python -m tests.test_v2_system && python -m tests.test_system_eval && python -m tests.test_compare_runs && \
python -m tests.test_preprocessing && python -m tests.test_context_notes && python -m tests.test_training_stop"
  tests=$ID
  echo "tests: job $tests"
  ids=("$tests")
  for grp in $METER_GROUPS; do
    runs=${grp//,/ }
    submit "met_${grp//,/+}" "${G[@]}" -c 16 --mem=120G -t 8:00:00 --dependency=afterok:$tests \
        --wrap="$JOBS meters $runs"
    a=$ID
    # the second job: once the first has ended (however), and only if the tests passed
    submit "met_${grp//,/+}" "${G[@]}" -c 16 --mem=120G -t 8:00:00 --dependency=afterok:$tests,afterany:$a \
        --wrap="$JOBS meters $runs"
    b=$ID
    ids+=("$a" "$b")
    echo "meter models $runs: jobs $a, then $b (resumes)"
  done
  if [ -n "$NEW_NLS" ]; then
    submit labellers "${G[@]}" -c 16 --mem=96G -t 8:00:00 --dependency=afterok:$tests \
          --wrap="$JOBS labellers $NEW_NLS"
    lab=$ID
    ids+=("$lab")
    echo "labellers $NEW_NLS: job $lab"
  fi
  submit v2grids -c 8 --mem=48G -t 8:00:00 --dependency=afterok:$tests --wrap="$JOBS grids"
  grids=$ID
  ids+=("$grids")
  echo "oracle grids + rendered-MIDI beats: job $grids"
  # wave 2 only after the tests passed (else this job is cancelled with the rest), and once every
  # other job of wave 1 has ended, however it ended
  others=("${ids[@]:1}")
  submit v2next -c 1 --mem=2G -t 0:20:00 "--dependency=afterok:$tests,afterany:$(IFS=:; echo "${others[*]}")" \
       --wrap="source pace/common.sh; bash pace/v2.sh wave2"
  nx=$ID
  echo "wave 2 is submitted by job $nx when all of these have finished ($n jobs now)"
  echo "watch: squeue -u \$USER"
  exit 0
fi
[ "$WAVE" = wave2 ] || { echo "unknown wave $WAVE (wave1, wave2)"; exit 2; }

evals=()
first=1
for nl in $NLS; do
  extra=""
  [ $first = 1 ] && extra=" $NLS_EXTRA"
  first=0
  submit "evals_$nl" "${G[@]}" -c 8 --mem=48G -t 8:00:00 --wrap="$JOBS evals $nl$extra"
  evals+=("$ID")
  echo "$nl$extra on every grid: job $ID"
done
FINAL="source pace/common.sh; python scripts/rescore.py --workers 16; python3 scripts/compare_runs.py --quiet"
systems_on=1
if [ -n "${SKIP_SYSTEM:-}" ]; then
  echo "SKIP_SYSTEM: no whole-piece comparison"
  systems_on=""
elif [ ! -f outputs/system_eval/manifest.json ] || [ ! -s "$BD/MIDI2ScoreTF.ckpt" ]; then
  echo "no outputs/system_eval/manifest.json or $BD/MIDI2ScoreTF.ckpt: run bash pace/bd_setup.sh and bash pace/system_eval.sh first"
  systems_on=""
fi
if [ -z "$systems_on" ]; then
  fdep=()
  [ ${#evals[@]} -gt 0 ] && fdep=("--dependency=afterany:$(IFS=:; echo "${evals[*]}")")
  submit v2final -c 16 --mem=64G -t 4:00:00 ${fdep[@]+"${fdep[@]}"} --wrap="$FINAL"
  echo "new metric on every run's saved predictions, then outputs/compare: job $ID (wave 2: $n jobs)"
  exit 0
fi
seq=()
for s in $SEQ_SB; do seq+=("$s:sb"); done
for s in $SEQ_RERUN; do  # only systems scored before
  [ -f "outputs/system_eval/${s%%:*}_${s##*:}/summary.json" ] && seq+=("$s")
done
half=$(( (${#seq[@]} + 1) / 2 ))
sys=()
if [ -n "$SYSTEMS" ]; then
  vdep=()
  [ ${#evals[@]} -gt 0 ] && vdep=("--dependency=afterany:$(IFS=:; echo "${evals[*]}")")
  # shellcheck disable=SC2086
  submit sys_v2 "${G[@]}" -c 8 --mem=48G -t 8:00:00 ${vdep[@]+"${vdep[@]}"} --wrap="$JOBS systems $SYSTEMS"
  id=$ID
  sys+=("$id")
  echo "v2 on whole pieces ($SYSTEMS): job $id"
fi
for part in a b; do
  if [ $part = a ]; then specs=("${seq[@]:0:$half}"); else specs=("${seq[@]:$half}"); fi
  [ ${#specs[@]} -gt 0 ] || continue
  submit "sys_seq_$part" "${G[@]}" -c 8 --mem=48G -t 8:00:00 --wrap="$JOBS systems ${specs[*]}"
  id=$ID
  sys+=("$id")
  echo "seq2seq on whole pieces (${specs[*]}): job $id"
done
all=(${evals[@]+"${evals[@]}"} ${sys[@]+"${sys[@]}"})
submit v2final -c 16 --mem=64G -t 8:00:00 "--dependency=afterany:$(IFS=:; echo "${all[*]}")" \
     --wrap="$FINAL; $BDENV; python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --systems all --jobs 16"
echo "new metric + outputs/compare, then every system rated with Beyer & Dai's metrics: job $ID"
echo "wave 2: $n jobs. then: python3 scripts/compare_runs.py (outputs/compare/summary.md), outputs/system_eval/results.md"
