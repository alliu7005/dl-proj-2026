#!/bin/bash
# v2, fifth round: the note-value level of a whole piece (on the login node from the repo root,
# after git pull; after pace/v2_round4.sh; alongside pace/v2_release.sh: together 7 jobs):
#   bash pace/v2_level.sh
#   AFTER=<relfinal's job id> bash pace/v2_level.sh   # the final scoring after the release's
#
# The meter model writes whole pieces at half or double note values (3/4 as 3/8; 5 of the 14 ASAP
# test pieces), the largest part of what it costs (docs/experiments.md, "Fifth round"). The decoder
# cannot weigh that choice with its own heads, and on ASAP neither a written-tempo / density prior
# nor the labeller's confidence picks the right level better than the decoder does
# (scripts/level_experiments.py; here on every validation and test set). What is left is a learned
# estimate of this piece's written tempo:
#
#   job          what                                                              waits for
#   lvtests      the test suites, on PACE's Python (CPU)                            -
#   lvtrain      meter_v3_th (a tempo head on meter_v3's frozen trunk) and          lvtests (ok)
#                meter_v3_ft (meter_v3 fine-tuned with it) side by side; each then
#                tuned with the level choice and evaluated (pace/v2_jobs.sh meters)
#   lvsys        the level experiments; nl_base_gn on both models' grids (every     lvtrain (any)
#                test set); v2 on whole pieces with each, in the release's
#                configuration (:tidy,d95)
#   lvfinal      outputs/compare, every system rated with Beyer & Dai's metrics,    lvsys (any), AFTER
#                the paired comparisons (outputs/system_eval/paired.md)
set -euo pipefail
GPUS=${GPUS:-cuda13}  # any NVIDIA GPU with bf16 (pace/README.md, "Which GPU")
G=(--gres=gpu:1 "--constraint=$GPUS")
RUNS=${RUNS-"meter_v3_th meter_v3_ft"}
LABELLER=${LABELLER-"nl_base_gn"}
SYS=${SYS-"nl_base_gn+meter_v3_th:tidy,d95 nl_base_gn+meter_v3_ft:tidy,d95"}
LEVELEXP=${LEVELEXP-"meter_v3_tp"}
AFTER=${AFTER:-}
BD=${BD:-$HOME/scratch/beyer_dai}
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
export METERS="$RUNS"    # the grids the labeller is evaluated on (with the oracle)
export NEW_NLS=""
export GRIDS_OLD=""
JOBS="source pace/common.sh; bash pace/v2_jobs.sh"
if [ ! -f outputs/runs/meter_v3/best.pt ] || [ ! -f outputs/runs/meter_v3_tp/decode.json ]; then
  echo "no outputs/runs/meter_v3/best.pt or meter_v3_tp/decode.json: run pace/v2.sh and pace/v2_round4.sh first"
  exit 1
fi
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

submit lvtests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; \
python -m tests.test_meter && python -m tests.test_release && python -m tests.test_compare_runs && \
python -m tests.test_pipeline && python -m tests.test_v2_system"
tests=$ID
echo "tests: job $tests"
submit lvtrain "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$JOBS meters $RUNS"
train=$ID
echo "train + tune + evaluate $RUNS: job $train"
wrap="$JOBS levelexp $LEVELEXP; $JOBS evals $LABELLER; $JOBS systems $SYS"
submit lvsys "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterany:$train" --wrap="$wrap"
sys=$ID
echo "level experiments, $LABELLER on the new grids, whole pieces ($SYS): job $sys"
deps="$sys${AFTER:+:$AFTER}"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16"
submit lvfinal -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$deps" \
  --wrap="source pace/common.sh; python3 scripts/compare_runs.py --quiet; $BDENV; $SCORE --systems all; python3 scripts/compare_systems.py"
echo "outputs/compare + every system rated with Beyer & Dai's metrics + paired comparisons: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
