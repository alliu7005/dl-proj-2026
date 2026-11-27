#!/bin/bash
# v2 tenth round, second follow-up: the meter decoder's piece-tempo prior, and the two-voice score
# writer, measured on whole pieces (on the login node from the repo root, after git pull; 5 jobs,
# alongside or after pace/v2_decode.sh and pace/v2_decode2.sh: their jobs still queued are waited
# for, found by name).
#   bash pace/v2_tempo.sh
#
# The piece-tempo prior (src/meter.py piece_tempo_logprior): Beat This! switches tempo octaves
# within a piece and the meter model follows it, so a stretch is written at double or half the
# note values of the rest (the user's "unlasting" and "blue bird"). The decoder runs twice: the
# piece's most common written tempo from the first pass, then a log-normal around it on every step.
# Tuned on validation (eval_meter.py --piece-tempo: meter_v4_gq_pt, the same checkpoint), then
# measured as its own meter system. The two-voice writer (src/preprocessing/detokenize.py
# _two_voices, _tv): two voices to a staff at most, ordered by pitch, held notes tied; engraving,
# measured in the demo's configuration.
#
#   job       what                                                                  waits for
#   pttests   the test suites (CPU)                                                  -
#   pttune    meter_v4_gq_pt: the prior tuned on validation, evaluated              pttests (ok)
#   ptasap    whole pieces on ASAP: nl_gq + meter_v4_gq_pt alone, + bd; the demo's  pttune (ok); dctune (any,
#             configuration with two voices, on meter_v4_gq and meter_v4_gq_pt        if still queued)
#   ptgq      the same on Gangqinpu's whole pieces                                   the same
#   ptfinal   Beyer & Dai's metrics (ASAP, Gangqinpu), both paired.md               ptasap, ptgq; dcfinal,
#                                                                                    d2final (any)
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
for f in outputs/runs/nl_gq/best.pt outputs/runs/meter_v4_gq/best.pt outputs/runs/meter_v4_gq/decode.json \
         outputs/system_eval/manifest.json outputs/system_eval_gangqinpu/manifest.json; do
  [ -e "$f" ] || { echo "no $f: run pace/v2_gqlabel.sh (and the rounds before) first"; exit 1; }
done
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: run bash pace/bd_setup.sh first"; exit 1; }
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
JOBS="bash pace/v2_jobs.sh"
GQOUT=outputs/system_eval_gangqinpu
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
NL="nl_gq+meter_v4_gq:tidy,d95"
PT="nl_gq+meter_v4_gq_pt:tidy,d95"
# compared with nl_gq + meter_v4_gq alone (earlier rounds), + bd (pace/v2_decode2.sh) and the demo's
# configuration (pace/v2_decode.sh)
SPECS="$PT,missing $PT,bd,missing $NL,tc,tb,rd,jf,bd,tv,missing $PT,tc,tb,rd,jf,bd,tv,missing"
queued() {  # the IDs of this user's queued / running jobs with these names, joined by ':' (empty: none)
  local ids=""
  for name in "$@"; do
    for id in $(squeue -u "$USER" -h -n "$name" -o %i 2>/dev/null); do ids="${ids:+$ids:}$id"; done
  done
  echo "$ids"
}
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
submit pttests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; python -m tests.test_piece_tempo && python -m tests.test_readable && python -m tests.test_system_eval && python -m tests.test_beat_decode && python -m tests.test_compare_runs"
tests=$ID
echo "tests: job $tests"
submit pttune "${G[@]}" -c 8 --mem=64G -t 4:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $JOBS piecetempo meter_v4_gq"
tune=$ID
echo "the piece-tempo prior tuned (meter_v4_gq_pt): job $tune"
before=$(queued dctune)
dep="afterok:$tune${before:+,afterany:$before}"
echo "the systems wait for: $dep"
submit ptasap "${G[@]}" -c 8 --mem=48G -t 12:00:00 "--dependency=$dep" --wrap="source pace/common.sh; $JOBS systems $SPECS"
asap=$ID
echo "whole pieces, ASAP (4 systems): job $asap"
submit ptgq "${G[@]}" -c 8 --mem=48G -t 12:00:00 "--dependency=$dep" --wrap="source pace/common.sh; $GQSYS $JOBS systems $SPECS"
gq=$ID
echo "whole pieces, Gangqinpu (4 systems): job $gq"
others=$(queued dcfinal d2final)
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
submit ptfinal -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq${others:+:$others}" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''"
echo "Beyer & Dai's metrics, paired comparisons: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
