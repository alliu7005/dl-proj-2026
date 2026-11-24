#!/bin/bash
# Submit pace/real_asap.sbatch without disturbing training jobs already queued (login node, repo root):
#   bash pace/real_asap.sh                 # any of H100 / H200 / RTX Pro Blackwell (default)
#   GPUS=gpu-h100 bash pace/real_asap.sh   # other GPU types (node features, "a|b" for several)
#
# Training runs mix train_transcribed in as soon as it exists. So that every run of a batch already
# queued trains on the same data, the build waits until all of your queued / running training jobs
# (job name "scoreformer") have STARTED (they read their data at start); runs submitted afterwards
# use the new splits. When everything has finished, add the "real" test condition to every run:
#   ARGS="--conditions real" bash pace/reevaluate.sh
set -euo pipefail
GPUS=${GPUS:-HX00}  # node feature: an H100 or an H200 (one feature: see pace/README.md, "Which GPU")
G=(--gres=gpu:1 "--constraint=$GPUS")
jobs=$(squeue -h -u "$USER" -n scoreformer -o %i | paste -sd: -)
dep=""
if [ -n "$jobs" ]; then
  dep="--dependency=after:$jobs"
  echo "waiting for these training jobs to start first: ${jobs//:/ }"
fi
id=$(sbatch --parsable "${G[@]}" $dep pace/real_asap.sbatch)
echo "real ASAP transcriptions: job $id   (log: slurm-realasap-$id.out)"
echo "when every job has finished:  ARGS=\"--conditions real\" bash pace/reevaluate.sh"
