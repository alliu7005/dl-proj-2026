#!/bin/bash
# Headroom tests (scripts/headroom.py): how much each proposed idea could gain at most, with the
# released models as they are, before anyone trains anything (on the login node from the repo root,
# after git pull; 4 jobs; they wait for the first pilot's q1prep if it is still queued, which
# installs nl_gq where this account lacks it):
#   bash pace/v2_headroom.sh
#
#   h1tests   the test suite (CPU)                                                  -
#   h1lab     a learned duration / position chooser: nl_gq's heads with the best of   h1tests (ok); q1prep (any)
#             their top 2 / 4 candidates (true grid), the windows rendered and scored
#   h1met     local meter repair: meter_v4_gq's decoded grid against the best of a     h1tests (ok)
#             few alternatives per piece and per 8 bars (at most 400 performances per set)
#   h1report  outputs/headroom/report.md                                             h1lab, h1met (any)
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
q=$(squeue -u "$USER" -h -n q1prep -o %i 2>/dev/null | paste -sd: -)
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
submit h1tests -c 4 --mem=16G -t 1:00:00 --wrap="source pace/common.sh; python -m tests.test_headroom"
tests=$ID
echo "tests: job $tests"
submit h1lab "${G[@]}" -c 8 --mem=48G -t 4:00:00 "--dependency=afterok:$tests${q:+,afterany:$q}" --wrap="source pace/common.sh; python scripts/headroom.py labeller --run nl_gq"
lab=$ID
echo "the labeller's top-k oracle: job $lab"
submit h1met "${G[@]}" -c 8 --mem=48G -t 4:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; python scripts/headroom.py meter --run meter_v4_gq --max-perfs 400"
met=$ID
echo "the meter's alternatives oracle: job $met"
submit h1report -c 2 --mem=4G -t 0:20:00 "--dependency=afterany:$lab:$met" --wrap="source pace/common.sh; python scripts/headroom.py report"
echo "outputs/headroom/report.md: job $ID ($n jobs)"
