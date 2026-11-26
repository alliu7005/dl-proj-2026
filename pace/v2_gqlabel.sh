#!/bin/bash
# v2, eighth round: the note labeller trained with Gangqinpu's written scores, and whole pieces
# outside classical piano (on the login node from the repo root, after git pull, with the queue
# empty: 8 jobs):
#   bash pace/v2_gqlabel.sh
#
# Why: the labeller has only seen classical scores (ASAP, PERiScoPe). Gangqinpu (sixth round)
# made the meter model general (time signatures right on 0.79 of Gangqinpu's real recordings,
# against 0.54), and its 17,773 training scores have what the labeller learns too: every note's
# measure, position, length, staff and grace flag, aligned to the recording's Transkun notes
# (scripts/build_gangqinpu_windows.py). And "more general" has only been measured on the meter
# model: whole pieces from Gangqinpu's test split (scripts/gangqinpu_inputs.py) measure it for
# every system, with the same metrics as the ASAP benchmark. Measured:
#   - nl_gq: nl_base_gn's recipe, half of the training windows from Gangqinpu; on every window test
#     set (Gangqinpu's included) with the true grid and the meter models' grids;
#   - whole pieces, ASAP and Gangqinpu (outputs/system_eval_gangqinpu): the release, with nl_gq,
#     with meter_v4_gq, with both, PM2S's beats with meter_v4_gq_pm, seq2seq, Beyer & Dai;
#   - meter_v4_gql: meter_v4_gq trained longer (100k steps; its best was at 56k of 60k);
#   - the release's meter model's second seed (meter_v3_s1 with the tempo-prior decoder:
#     meter_v3_s1_tp), so the report can give v2's spread.
#
#   job        what                                                                  waits for
#   lbtests    the test suites, on PACE's Python (CPU)                                -
#   lbdata     Gangqinpu's scores as windows (train / val / test_gangqinpu) and       lbtests (ok)
#              their labels (CPU); meter_v3_s1_tp tuned and evaluated
#   lbinputs   Gangqinpu's whole test pieces (60, one per score): references, the     lbtests (ok)
#              aligned notes, Transkun on the recordings, Beat This!'s beats
#   lbtrain    nl_base_gn on Gangqinpu's windows first (so publishing nl_gq compares   lbdata (ok)
#              them on pop too); nl_gq trained and evaluated (true grid); nl_gq and nl_base_gn on the
#              meter models' grids, Gangqinpu's windows included
#   lbmeter    meter_v4_gql trained, tuned and evaluated                              lbtests (ok)
#   lbsys      whole pieces: nl_gq (ASAP, Gangqinpu), the second seed (ASAP)          lbtrain, lbinputs (any)
#   lbsys2     whole pieces on Gangqinpu: the release, meter_v4_gq, meter_v4_gq_pm    lbinputs (ok)
#              with PM2S, seq2seq; Beyer & Dai's predictions
#   lbfinal    meter_v4_gql on whole pieces (if trained); outputs/compare; Beyer &    lbsys, lbsys2, lbmeter (any)
#              Dai's metrics for every system on both test sets; paired comparisons
#
# Again after a stop (finished runs are kept):
#   STAGE=eval bash pace/v2_gqlabel.sh    # only lbsys, lbsys2 and lbfinal again
set -euo pipefail
GPUS=${GPUS:-cuda13}  # any NVIDIA GPU with bf16 (pace/README.md, "Which GPU")
G=(--gres=gpu:1 "--constraint=$GPUS")
NL=${NL:-nl_gq}
BASE_NL=${BASE_NL:-nl_base_gn}
LONG=${LONG:-meter_v4_gql}
GQ=${GQ:-data/raw/gangqinpu}
N_PIECES=${N_PIECES:-60}
BD=${BD:-$HOME/scratch/beyer_dai}
STAGE=${STAGE:-all}   # all; eval: the whole-piece jobs and the scoring only
case "$STAGE" in all|eval) ;; *) echo "STAGE must be all or eval, not $STAGE"; exit 1 ;; esac
for f in outputs/runs/meter_v3_tp/decode.json outputs/runs/meter_v4_gq/best.pt outputs/runs/meter_v4_gq_pm/best.pt \
         outputs/runs/meter_v3_s1/best.pt "outputs/runs/$BASE_NL/best.pt" outputs/system_eval/manifest.json \
         data/processed/vocab.json data/processed/train/notelabels.npz data/processed/meter/train_gangqinpu.pkl \
         configs/beat_noise_bt_gangqinpu.json; do
  [ -e "$f" ] || { echo "no $f: run pace/v2_gangqinpu.sh (and the rounds before it) first"; exit 1; }
done
[ -n "$(ls "$GQ"/data/*.parquet 2>/dev/null)" ] || { echo "no $GQ/data/*.parquet: run pace/v2_gangqinpu.sh first"; exit 1; }
[ -n "$(ls "$GQ"/audio/test/*.mp3 2>/dev/null)" ] || { echo "no $GQ/audio/test/*.mp3: run pace/v2_gangqinpu.sh first (it downloads them)"; exit 1; }
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: run bash pace/bd_setup.sh first"; exit 1; }
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
JOBS="bash pace/v2_jobs.sh"
GQOUT=outputs/system_eval_gangqinpu
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
SEQ=sf_base_reg_cab_c128:meter:sb
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

train="" inputs="" meter="" dep_sys=() dep_sys2=()
if [ "$STAGE" = all ]; then
submit lbtests -c 8 --mem=64G -t 2:00:00 --wrap="source pace/common.sh; python -m tests.test_gangqinpu_scores && python -m tests.test_publish && python -m tests.test_gangqinpu && python -m tests.test_notelabels && python -m tests.test_v2_system && python -m tests.test_system_eval && python -m tests.test_compare_runs && python -m tests.test_preprocessing"
tests=$ID
echo "tests: job $tests"
submit lbdata -c 16 --mem=96G -t 4:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; python scripts/build_gangqinpu_windows.py --release $GQ --workers 16 && python scripts/build_notelabels.py --splits train_gangqinpu,val_gangqinpu,test_gangqinpu --workers 16 && $JOBS tempo meter_v3_s1 meter_v3_s1_tp"
data=$ID
echo "Gangqinpu's scores as windows + labels (data/processed/*_gangqinpu), meter_v3_s1_tp: job $data"
submit lbinputs "${G[@]}" -c 8 --mem=48G -t 3:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; python scripts/gangqinpu_inputs.py --release $GQ --n $N_PIECES"
inputs=$ID
echo "Gangqinpu's whole test pieces ($GQOUT/manifest.json): job $inputs"
submit lbtrain "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$data" --wrap="source pace/common.sh; python scripts/eval_labeller.py --run $BASE_NL --conditions gangqinpu --normal-priority; $JOBS labellers $NL; METERS=\"meter_v3_tp meter_v4_gq\" GRIDS_OLD=\"\" NEW_NLS=\"$NL\" $JOBS evals $NL $BASE_NL"
train=$ID
echo "$NL trained and evaluated, $NL and $BASE_NL on the meter models' grids: job $train"
submit lbmeter "${G[@]}" -c 16 --mem=96G -t 8:00:00 "--dependency=afterok:$tests" --wrap="source pace/common.sh; $JOBS meters $LONG"
meter=$ID
echo "$LONG trained, tuned and evaluated: job $meter"
dep_sys=("--dependency=afterany:$train:$inputs")
dep_sys2=("--dependency=afterok:$inputs")
fi
# lbsys: the new labeller everywhere, the second seed on ASAP
submit lbsys "${G[@]}" -c 8 --mem=48G -t 8:00:00 ${dep_sys[@]+"${dep_sys[@]}"} --wrap="source pace/common.sh; $JOBS systems $NL+meter_v3_tp:tidy,d95 $NL+meter_v4_gq:tidy,d95 $BASE_NL+meter_v3_s1_tp:tidy,d95; $GQSYS $JOBS systems $NL+meter_v4_gq:tidy,d95 $NL+meter_v3_tp:tidy,d95"
sys=$ID
echo "whole pieces with $NL (ASAP, Gangqinpu) and the second seed (ASAP): job $sys"
# lbsys2: the systems we have, on Gangqinpu's whole pieces, and Beyer & Dai's predictions there
submit lbsys2 "${G[@]}" -c 8 --mem=48G -t 8:00:00 ${dep_sys2[@]+"${dep_sys2[@]}"} --wrap="source pace/common.sh; $GQSYS $JOBS systems $BASE_NL+meter_v3_tp:tidy,d95 $BASE_NL+meter_v4_gq:tidy,d95 $BASE_NL+meter_v4_gq_pm:tidy,d95,pm,rpm $SEQ; $BDENV; python scripts/beyer_dai.py predict --repo $BD/MIDI2ScoreTransformer --ckpt $BD/MIDI2ScoreTF.ckpt --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real"
sys2=$ID
echo "whole pieces on Gangqinpu: the release, meter_v4_gq, meter_v4_gq_pm with PM2S, seq2seq, Beyer & Dai: job $sys2"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
LONGSYS="$JOBS systems $BASE_NL+$LONG:tidy,d95; $GQSYS $JOBS systems $BASE_NL+$LONG:tidy,d95"
submit lbfinal "${G[@]}" -c 16 --mem=64G -t 8:00:00 "--dependency=afterany:$sys:$sys2${meter:+:$meter}" --wrap="source pace/common.sh; $LONGSYS; python3 scripts/compare_runs.py --quiet; $BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''"
echo "$LONG on whole pieces, outputs/compare, every system rated with Beyer & Dai's metrics (ASAP, Gangqinpu), paired comparisons: job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
