#!/bin/bash
# The release (on the login node from the repo root, after git pull; after pace/v2_round4.sh):
#   bash pace/v2_release.sh
#
# Round 4 measured each change on its own with the round-3 labeller (docs/experiments.md, "Fourth
# round: results"): the tempo-prior decoder, the length tidying and keeping notes (d80 / d95) each
# help; the new labellers do not. The release combines them, and this scores exactly that, plus
# the same at d80 and with the labeller's second seed (how much a training run varies), then makes
# the release (scripts/export_models.py -> models/, uploaded from there: docs/handoff.md, "Models").
#
#   job          what                                                              waits for
#   reltests     the test suites, on PACE's Python (CPU)                            -
#   relsys       v2 on whole pieces: the release (SYS), every test performance      reltests (ok)
#   relfinal     the release in models/ (export_models.py: checked by loading it),  relsys (any)
#                every system rated with Beyer & Dai's metrics, the paired
#                comparisons (compare_systems.py -> outputs/system_eval/paired.md)
set -euo pipefail
GPUS=${GPUS:-cuda13}  # any NVIDIA GPU with bf16: a short job (pace/README.md, "Which GPU")
G=(--gres=gpu:1 "--constraint=$GPUS")
SYS=${SYS-"nl_base_gn+meter_v3_tp:tidy,d95 nl_base_gn+meter_v3_tp:tidy,d80 nl_base_gn_s1+meter_v3_tp:tidy,d95"}
EXPORT=${EXPORT-"python scripts/export_models.py"}
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

submit reltests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; \
python -m tests.test_release && python -m tests.test_compare_runs && python -m tests.test_meter && python -m tests.test_pipeline && \
python -m tests.test_system_eval && python -m tests.test_v2_system"
tests=$ID
echo "tests: job $tests"
submit relsys "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tests" --wrap="$JOBS systems $SYS"
sys=$ID
echo "whole pieces ($SYS): job $sys"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16"
submit relfinal -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$sys" \
  --wrap="source pace/common.sh; $EXPORT; $BDENV; $SCORE --systems all; python3 scripts/compare_systems.py"
echo "the release in models/ + every system rated with Beyer & Dai's metrics: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
