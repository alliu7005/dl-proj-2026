#!/bin/bash
# v2 pilot, part 3: the score-level rules, measured on the whole-piece benchmark (on the login node
# from the repo root, after git pull; its jobs wait for the first two pilots' still queued, found by
# name; 4 jobs):
#   bash pace/v2_pilot3.sh
#
#   _ps  PKSpell (src/pkspell.py; Foscarin et al., ISMIR 2021, MIT): the key signature and every
#        note's spelling from a model, not from one key signature (v2's spelling error ~12% against
#        Beyer & Dai's 8.6%; Chopin's Mazurka op. 6 no. 2 came out in four flats instead of four sharps)
#   _eg  engraving (src/readable.py engrave_score): clefs from each staff's range, every one-voice
#        note's stem by the engraver's rule (v2's stem error ~51% against Beyer & Dai's 25%: v2 set
#        stems only where a staff has two voices)
#   _sw  swing (src/swing.py): swung pairs written straight under a swing marking (Lemon: 78% -> 96%
#        of the sheet's onsets in bars 2-7)
# _ps and _eg change only the written score, so they are applied to the release's scores as written
# already (scripts/postprocess_scores.py); _sw changes the labelled notes, so that system runs again,
# and once with all three through the pipeline itself (the path the demo would take).
# outputs/pilot/report.md gets section I. PKSpell's weights are fetched here on the login node
# (compute nodes may have no internet). Nothing is published.
#
#   job       what                                                                 waits for
#   p3tests   the test suites (CPU)                                                 -
#   p3post    the release's scores with _ps, _eg, _ps_eg, ASAP and Gangqinpu (CPU)  p3tests (ok); q1asap, q1gq, r2asap, r2gq (any)
#   p3swing   the release with _sw, and with _ps_eg_sw, ASAP and Gangqinpu          p3tests (ok); q1prep (any)
#   p3final   Beyer & Dai's metrics, paired.md, the pilot report                    p3post, p3swing; q1final, r2final (any)
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
GQOUT=outputs/system_eval_gangqinpu
REL=v2_nl_gq_meter_v4_gq_d95_tidy
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: bash pace/bd_setup.sh first"; exit 1; }
export TORCH_HOME="${TORCH_HOME:-$HOME/scratch/.cache/torch}"   # as pace/common.sh sets it for the jobs
python3 -c "import sys; sys.path.insert(0, '.'); from src.pkspell import fetch_weights; print('PKSpell weights:', fetch_weights())" \
  || { echo "PKSpell's weights could not be downloaded here (module load anaconda3 && conda activate dl-proj first?)"; exit 1; }
export PUBLISH=0
JOBS="PUBLISH=0 bash pace/v2_jobs.sh"
GQSYS="SYS_ARGS=\"--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real\""
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
queued() {
  local ids=""
  for name in "$@"; do
    for id in $(squeue -u "$USER" -h -n "$name" -o %i 2>/dev/null); do ids="${ids:+$ids:}$id"; done
  done
  echo "$ids"
}
after() {
  local ok=$1; shift
  local q; q=$(queued "$@")
  echo "afterok:$ok${q:+,afterany:$q}"
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
submit p3tests -c 8 --mem=32G -t 1:00:00 --wrap="source pace/common.sh; python -m tests.test_score_rules && python -m tests.test_readable && python -m tests.test_timing"
tests=$ID
echo "tests: job $tests"
POST=""
for out in outputs/system_eval "$GQOUT"; do
  for f in --pkspell --engraving "--pkspell --engraving"; do
    POST="$POST python scripts/postprocess_scores.py --system $REL $f --out $out;"
  done
done
submit p3post -c 8 --mem=32G -t 4:00:00 "--dependency=$(after "$tests" q1asap q1gq r2asap r2gq)" --wrap="source pace/common.sh; $POST"
post=$ID
echo "_ps, _eg, _ps_eg on the release's scores: job $post"
SPECS="nl_gq+meter_v4_gq:tidy,d95,sw,missing nl_gq+meter_v4_gq:tidy,d95,ps,eg,sw,missing"
submit p3swing "${G[@]}" -c 8 --mem=48G -t 10:00:00 "--dependency=$(after "$tests" q1prep)" --wrap="source pace/common.sh; $JOBS systems $SPECS; $GQSYS $JOBS systems $SPECS"
swing=$ID
echo "_sw and _ps_eg_sw through the pipeline, ASAP and Gangqinpu: job $swing"
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
prev=$(queued q1final r2final)
submit p3final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany:$post:$swing${prev:+:$prev}" --wrap="$BDENV; $SCORE; $SCORE --manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real; python3 scripts/compare_systems.py; python3 scripts/compare_systems.py --results $GQOUT/results.json --out $GQOUT/paired.md --ablation ''; python3 scripts/pilot_report.py"
echo "Beyer & Dai's metrics, paired comparisons, outputs/pilot/report.md (section I): job $ID ($n jobs)"
echo "watch: squeue -u \$USER"
