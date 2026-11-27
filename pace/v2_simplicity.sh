#!/bin/bash
# Simpler rhythms: v2 writes the rhythm it hears (a detached off-beat eighth as a 1/6-beat note and
# rests, a rushed run as 32nds or a quintuplet), where an arrangement writes plain eighths and
# sixteenths. On the login node from the repo root, after git pull (4 jobs, starts at once):
#   bash pace/v2_simplicity.sh
#
# Whole pieces, ASAP and Gangqinpu, of the release (nl_gq + meter_v4_gq, tidy, d95) with:
#   _sp10, _sp20   rarer rhythms cost 1 / 2 x their complexity in the labeller's choice of position
#                  in the beat and written length (src/labeller.py rhythm_complexity)
#   _sl            rare written lengths made plain (src/readable.py simplify_lengths)
#   _sp20_sl       both (the Lemon demo: bars with a rare rhythm 28 -> 8 of 69)
#   _jf            bars the decoder stretched with a jump shortened: the demo's setting, and where
#                  Lemon's 3/4 and 2/4 bars came from (all 69 bars 4/4 without it)
# outputs/simplicity/report.md: how often each writes a rare rhythm against the reference scores,
# MUSTER (onset / offset parts), score similarity's duration error, MUSTER paired against the release.
#   job       what                                                        waits for
#   x1tests   the test suites (CPU)                                        -
#   x1asap    the five systems on ASAP                                     x1tests (ok)
#   x1gq      the same on Gangqinpu                                        x1tests (ok)
#   x1final   Beyer & Dai's metrics, outputs/simplicity/report.md          x1asap, x1gq (any)
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
GQOUT=outputs/system_eval_gangqinpu
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: bash pace/bd_setup.sh first"; exit 1; }
export PUBLISH=0
JOBS="PUBLISH=0 bash pace/v2_jobs.sh"
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
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
submit x1tests -c 8 --mem=32G -t 1:00:00 --wrap="source pace/common.sh; python -m tests.test_simplicity && python -m tests.test_readable"
tests=$ID
SPECS=""
for o in sp10 sp20 sl sp20,sl jf; do SPECS="$SPECS nl_gq+meter_v4_gq:tidy,d95,$o,missing"; done
submit x1asap "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $JOBS systems $SPECS"
asap=$ID
submit x1gq "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $GQSYS $JOBS systems $SPECS"
gq=$ID
echo "simpler rhythms on whole pieces: jobs $asap (ASAP), $gq (Gangqinpu)"
submit x1final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; source pace/common.sh; python scripts/simplicity_report.py"
echo "Beyer & Dai's metrics, outputs/simplicity/report.md: job $ID ($n jobs); watch: squeue -u \$USER"
