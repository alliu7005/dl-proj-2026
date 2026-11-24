#!/bin/bash
# System-level comparison with Beyer & Dai (Baseline 3), on the login node from the repo root, after
# pace/bd_setup.sh (once):
#   bash pace/system_eval.sh
#   RUNS="sf_base_reg_cab:meter sf_base_reg_cab:tsearch_beat" bash pace/system_eval.sh   # our systems
#   ONLY=bd bash pace/system_eval.sh       # only Beyer & Dai's predictions, then the scoring again
#   ONLY=score bash pace/system_eval.sh    # only the scoring (every prediction already there)
#
# Every system turns each whole test performance into one MusicXML score, scored against the ASAP
# reference with Beyer & Dai's own metrics (MUSTER, score similarity), so the comparison does not
# depend on our window metrics (scripts/system_inputs.py, scripts/beyer_dai.py):
#
#   job          what                                                              waits for
#   sysinputs    manifest: the 59 clean test performances + their Transkun          -
#                transcriptions cut to each piece (CPU)
#   sys_<run>    our pipeline, as the demo runs it, on every input (GPU, dl-proj)   sysinputs
#   bdpredict    Beyer & Dai's pretrained model on every input (GPU, bd311)         sysinputs
#   sysscore     both rated with their metrics -> outputs/system_eval/results.md    all of the above
#                (CPU, bd311; their paper's row printed next to the clean results)
set -euo pipefail
GPUS=${GPUS:-HX00}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
RUNS=${RUNS:-"sf_base_reg_cab:meter sf_base_reg_cab:tsearch_beat s2s_m_cab:meter"}
ONLY=${ONLY:-}
case "$ONLY" in ""|bd|score) ;; *) echo "unknown ONLY=$ONLY (bd, score)"; exit 1 ;; esac
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
if [ ! -s "$BD/MIDI2ScoreTF.ckpt" ] || [ ! -d "$BD/MIDI2ScoreTransformer" ]; then
  echo "$BD has no MIDI2ScoreTransformer checkout / checkpoint: run bash pace/bd_setup.sh first"
  exit 1
fi
if [ -n "$ONLY" ] && [ ! -f outputs/system_eval/manifest.json ]; then
  echo "outputs/system_eval/manifest.json missing: run the whole thing first (bash pace/system_eval.sh)"
  exit 1
fi
inp=""
afterinp=()
if [ -z "$ONLY" ]; then
  inp=$(sbatch --parsable -J sysinputs -c 4 --mem=16G -t 0:30:00 -o slurm-sysinputs-%j.out \
        --wrap="source pace/common.sh; python scripts/system_inputs.py")
  echo "inputs: job $inp"
  afterinp=("--dependency=afterok:$inp")
fi
deps=()
systems=beyer_dai
for spec in $RUNS; do
  run=${spec%%:*}; mode=${spec##*:}
  if [ -n "$ONLY" ]; then  # predictions already made: only named for the scoring
    systems="$systems,${run}_${mode}"
    continue
  fi
  [ -f "outputs/runs/$run/best.pt" ] || { echo "  skip $spec: outputs/runs/$run/best.pt not found"; continue; }
  t=3:00:00
  case "$mode" in tsearch*) t=8:00:00 ;; esac  # three decodings of every piece
  id=$(sbatch --parsable "${G[@]}" -J "sys_$run" -c 8 --mem=48G -t "$t" "${afterinp[@]}" \
       -o slurm-sys_$run-%j.out --wrap="source pace/common.sh; python scripts/system_scores.py --run $run --mode $mode")
  echo "  ours $run ($mode): job $id ($t)"
  deps+=("$id")
  systems="$systems,${run}_${mode}"
done
if [ "$ONLY" != score ]; then
  bd=$(sbatch --parsable "${G[@]}" -J bdpredict -c 8 --mem=48G -t 3:00:00 ${afterinp[@]+"${afterinp[@]}"} \
       -o slurm-bdpredict-%j.out \
       --wrap="$BDENV; python scripts/beyer_dai.py predict --repo $BD/MIDI2ScoreTransformer --ckpt $BD/MIDI2ScoreTF.ckpt")
  echo "  Beyer & Dai: job $bd"
  deps+=("$bd")
fi
# afterany: score whatever finished, even if one system's job failed (its pieces show as not scored)
scdep=()
if [ ${#deps[@]} -gt 0 ]; then scdep=("--dependency=afterany:$(IFS=:; echo "${deps[*]}")"); fi
sc=$(sbatch --parsable -J sysscore -c 16 --mem=64G -t 3:00:00 ${scdep[@]+"${scdep[@]}"} -o slurm-sysscore-%j.out \
     --wrap="$BDENV; python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --systems $systems --jobs 16")
echo "  scoring ($systems): job $sc"
echo "watch: squeue -u \$USER     result: outputs/system_eval/results.md (slurm-sysscore-$sc.out)"
