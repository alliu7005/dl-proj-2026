#!/bin/bash
# Articulation pilot: written lengths that do not follow played ones (on the login node from the repo
# root, after git pull; 6 jobs, starting at once):
#   bash pace/v2_articulation.sh
#
# Why: v2 writes a detached off-beat eighth as a 1/6-beat note and rests (the Lemon demo: 124 rare
# written lengths, 28 of 69 bars with a rare rhythm). Training varies played lengths by only ~15%
# (configs/noise_stats.json offset_log_std), while real playing is detached (30-60% of the written
# length), legato (up to the next note) or held under the pedal: the labeller never learned that the
# written length does not follow the played one. src/dataset.py articulate plays half the training
# windows another way, stretch by stretch and hand by hand, with the targets unchanged.
#   job       what                                                                  waits for
#   a1tests   the test suites (CPU)                                                  -
#   a1data    Gangqinpu's windows and labels where missing (CPU)                     round 12's s1data (any)
#   a1train   nl_gq_p20k and nl_gq_art_p20k (nl_gq's recipe, 20k steps, without /    a1tests, a1data (ok)
#             with --articulation 0.5) side by side, evaluated on the test windows
#   a1asap    both with meter_v4_gq (tidy, d95) on ASAP's whole pieces               a1train (ok)
#   a1gq      the same on Gangqinpu                                                  a1train (ok)
#   a1final   Beyer & Dai's metrics, outputs/articulation/report.md                  a1asap, a1gq (any)
# Nothing is published (PUBLISH=0).
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
GQOUT=outputs/system_eval_gangqinpu
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: bash pace/bd_setup.sh first"; exit 1; }
GQ=${GQ:-data/raw/gangqinpu}
for f in data/processed/vocab.json outputs/runs/meter_v4_gq/best.pt "$GQOUT/manifest.json" outputs/system_eval/manifest.json; do
  [ -e "$f" ] || { echo "no $f: an earlier round makes it (pace/README.md)"; exit 1; }
done
[ -n "$(ls "$GQ"/data/*.parquet 2>/dev/null)" ] || { echo "no $GQ/data/*.parquet: run pace/v2_gangqinpu.sh first"; exit 1; }
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
submit a1tests -c 8 --mem=32G -t 1:00:00 --wrap="source pace/common.sh; python -m tests.test_articulation"
tests=$ID
# Gangqinpu's windows and their labels where this account lacks them (nl_gq's recipe trains on them),
# after round 12's builder of the same windows if it is still queued (s1data: never two at once)
builder=$(squeue -u "$USER" -h -n s1data -o %i 2>/dev/null | paste -sd: -)
DATA="for s in train val test; do [ -s data/processed/\${s}_gangqinpu/index.npy ] || python scripts/build_gangqinpu_windows.py --release $GQ --splits \$s || exit 1; done; "
DATA+="L=; for s in train val test; do [ -f data/processed/\${s}_gangqinpu/notelabels.npz ] || L=\${L:+\$L,}\${s}_gangqinpu; done; "
DATA+="[ -z \"\$L\" ] || python scripts/build_notelabels.py --splits \$L --workers 16"
submit a1data -c 16 --mem=64G -t 6:00:00 ${builder:+"--dependency=afterany:$builder"} --wrap="source pace/common.sh; $DATA"
data=$ID
echo "Gangqinpu's windows and labels where missing: job $data${builder:+ (after the round-12 builder s1data, job $builder)}"
submit a1train "${G[@]}" -c 16 --mem=96G -t 16:00:00 "--dependency=afterok:$tests:$data" --wrap="source pace/common.sh; $JOBS labellers nl_gq_p20k nl_gq_art_p20k"
train=$ID
echo "nl_gq_p20k, nl_gq_art_p20k trained side by side and evaluated: job $train"
SPECS="nl_gq_p20k+meter_v4_gq:tidy,d95,missing nl_gq_art_p20k+meter_v4_gq:tidy,d95,missing"
submit a1asap "${G[@]}" -c 8 --mem=48G -t 6:00:00 "--dependency=afterok:$train" --wrap="source pace/common.sh; $JOBS systems $SPECS"
asap=$ID
submit a1gq "${G[@]}" -c 8 --mem=48G -t 6:00:00 "--dependency=afterok:$train" --wrap="source pace/common.sh; $GQSYS $JOBS systems $SPECS"
gq=$ID
echo "whole pieces: jobs $asap (ASAP), $gq (Gangqinpu)"
submit a1final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$asap:$gq" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; source pace/common.sh; python scripts/articulation_report.py"
echo "Beyer & Dai's metrics, outputs/articulation/report.md: job $ID ($n jobs); watch: squeue -u \$USER"
