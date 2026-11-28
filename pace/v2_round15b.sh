#!/bin/bash
# Round 15, part b: the team's published release and model's beat decoding on our candidate (on the
# login node from the repo root, after git pull and after bash pace/v2_round15.sh; 5 jobs):
#   bash pace/v2_round15b.sh
#
#   T  the release on Hugging Face (dat-dev/scoreformer, 2026-11-27): nl_gq_mc + meter_v4_gq with its
#      tuned beat decoding and stretched bars shortened (_jf_bd), scored on our test pieces next to
#      round 15's candidate: the same pieces, the same scoring
#   B  round 15's candidate with model's beat decoding (nl_gq's tuned beat_decode.json from the same
#      release): nl_gq + meter_v4_gq_pmhs, tidy, d95, _jf, _bd, PM2S's beats. Does it stack on PM2S?
# Their files are fetched here, on the login node (compute nodes may have no internet):
# outputs/runs/nl_gq_mc_hf/ (best.pt, config.json, vocab.json, beat_decode.json), and nl_gq's
# beat_decode.json only where outputs/runs/nl_gq has none. x5final scores everything again (after
# round 15's r5final if it is still queued) and writes outputs/round15/report.md with T and B.
set -euo pipefail
GPUS=${GPUS:-cuda13}
G=(--gres=gpu:1 "--constraint=$GPUS")
BD=${BD:-$HOME/scratch/beyer_dai}
GQOUT=outputs/system_eval_gangqinpu
PM_RUN=${PM_RUN:-pm2s_hos}
PM_TAG=${PM_TAG:-hos}
HF=${HF:-dat-dev/scoreformer}
TEAM=outputs/runs/nl_gq_mc_hf
[ -s "$BD/MIDI2ScoreTF.ckpt" ] || { echo "no $BD/MIDI2ScoreTF.ckpt: bash pace/bd_setup.sh first"; exit 1; }
for f in "outputs/runs/$PM_RUN/best.pt" outputs/runs/meter_v4_gq_pmhs/eval_meter.json outputs/runs/meter_v4_gq/best.pt \
         outputs/runs/nl_gq/best.pt "$GQOUT/manifest.json" outputs/system_eval/manifest.json; do
  [ -e "$f" ] || { echo "no $f: an earlier round makes it"; exit 1; }
done
if [ "${FETCHED:-0}" != 1 ]; then   # the release's files, here on the login node
  # pace/common.sh is for jobs (it cds to $SLURM_SUBMIT_DIR, and conda's hooks trip set -u): a subshell
  (set +eu; export SLURM_SUBMIT_DIR="$PWD"; source pace/common.sh; set -e
  python - "$HF" "$TEAM" <<'EOF'
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, team = sys.argv[1], Path(sys.argv[2])
team.mkdir(parents=True, exist_ok=True)
for src, dst in (("runs/nl_gq_mc/model.pt", "best.pt"), ("runs/nl_gq_mc/config.json", "config.json"),
                 ("runs/nl_gq_mc/vocab.json", "vocab.json"), ("runs/nl_gq_mc/beat_decode.json", "beat_decode.json")):
    shutil.copy(hf_hub_download(repo, src), team / dst)
nl = Path("outputs/runs/nl_gq/beat_decode.json")
if not nl.exists():
    shutil.copy(hf_hub_download(repo, "runs/nl_gq/beat_decode.json"), nl)
    print(f"{nl}: the release's (tuned for nl_gq by model's round 10)")
print(f"{team}: the release's labeller and its beat decoding")
EOF
  )
fi
for f in "$TEAM/best.pt" "$TEAM/beat_decode.json" outputs/runs/nl_gq/beat_decode.json; do
  [ -s "$f" ] || { echo "no $f: the fetch failed (FETCHED=1 skips it once the files are there)"; exit 1; }
done
PMENV="source pace/common.sh; export PM2S_WEIGHTS=\"\$SLURM_SUBMIT_DIR/outputs/runs/$PM_RUN/best.pt\" PM2S_TAG=$PM_TAG"
GQARGS="--manifest $GQOUT/manifest.json --out $GQOUT --conditions gangqinpu,gangqinpu_real"
BDENV="module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; export PYTHONUNBUFFERED=1; cd \"\$SLURM_SUBMIT_DIR\""
SCORE="python scripts/beyer_dai.py score --repo $BD/MIDI2ScoreTransformer --jobs 16 --systems all"
TEAMSYS="python scripts/system_scores.py --v2 nl_gq_mc_hf+meter_v4_gq --delete-above 0.95 --tidy --fix-jumps --beat-decode --only-missing"
CANDBD="python scripts/system_scores.py --v2 nl_gq+meter_v4_gq_pmhs --delete-above 0.95 --tidy --fix-jumps --beat-decode --clean-beats pm2s --real-beats pm2s --only-missing"
n=0
ID=""
submit() {
  local name=$1
  shift
  ID=$(sbatch --parsable -J "$name" -o "slurm-$name-%j.out" "$@")
  n=$((n + 1))
}
submit x5tests -c 8 --mem=32G -t 1:00:00 --wrap="source pace/common.sh; python -m tests.test_round15"
tests=$ID
ends=""
submit t5asap "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tests" --kill-on-invalid-dep=yes --wrap="source pace/common.sh; $TEAMSYS"
ends="$ends:$ID"
submit t5gq "${G[@]}" -c 8 --mem=48G -t 6:00:00 "--dependency=afterok:$tests" --kill-on-invalid-dep=yes --wrap="$PMENV; $TEAMSYS $GQARGS && $CANDBD $GQARGS"
ends="$ends:$ID"
submit b5asap "${G[@]}" -c 8 --mem=48G -t 8:00:00 "--dependency=afterok:$tests" --kill-on-invalid-dep=yes --wrap="$PMENV; $CANDBD"
ends="$ends:$ID"
r5=$(squeue -u "$USER" -h -n r5final -o %i 2>/dev/null | paste -sd: -)
submit x5final -c 16 --mem=64G -t 4:00:00 "--dependency=afterany${ends}${r5:+:$r5}" --wrap="$BDENV; $SCORE; $SCORE $GQARGS; source pace/common.sh; python scripts/round15_report.py"
echo "the team's release (T) and the candidate with beat decoding (B):${ends//:/ }; scored with round 15's report: job $ID ($n jobs)"
