#!/bin/bash
# One-time setup of Beyer & Dai's model (Baseline 3) in its own environment, on the login node:
#   bash pace/bd_setup.sh
#   COMMIT=fc21531 bash pace/bd_setup.sh     # their release-tag commit instead of main (see below)
#
# Their code (github.com/TimFelixBeyer/MIDI2ScoreTransformer) is Python 3.11 with three forks they
# patched (music21 0ed70bb, score_transformer 934a228, muster), so it gets a conda env of its own
# (bd311) and never touches dl-proj. muster compiles C++ evaluation programs at install time (g++).
# The checkpoint (MIDI2ScoreTF.ckpt, 390 MB) is their GitHub release v0.0.1. The default is their
# main branch (it fixes the KV cache their inference uses); the release tag is fc21531. If the
# reproduction check (scripts/beyer_dai.py score vs their paper's Table 3) is far off, try the tag.
# Everything goes to $BD (default ~/scratch/beyer_dai); rerunning skips what is already there.
set -euo pipefail
BD=${BD:-$HOME/scratch/beyer_dai}
COMMIT=${COMMIT:-main}
REPO_URL=https://github.com/TimFelixBeyer/MIDI2ScoreTransformer.git
CKPT_URL=https://github.com/TimFelixBeyer/MIDI2ScoreTransformer/releases/download/v0.0.1/MIDI2ScoreTF.ckpt
ROOT=$(pwd)
module load anaconda3
eval "$(conda shell.bash hook)"
command -v g++ >/dev/null || module load gcc
if ! conda env list | awk '{print $1}' | grep -qx bd311; then
  conda create -y -n bd311 python=3.11
fi
conda activate bd311
mkdir -p "$BD"
[ -d "$BD/MIDI2ScoreTransformer/.git" ] || git clone "$REPO_URL" "$BD/MIDI2ScoreTransformer"
git -C "$BD/MIDI2ScoreTransformer" fetch --quiet origin
git -C "$BD/MIDI2ScoreTransformer" checkout --quiet "$COMMIT"
[ "$COMMIT" = main ] && git -C "$BD/MIDI2ScoreTransformer" pull --quiet --ff-only origin main
echo "MIDI2ScoreTransformer at $(git -C "$BD/MIDI2ScoreTransformer" rev-parse --short HEAD)"
pip install --quiet -r "$BD/MIDI2ScoreTransformer/requirements.txt"
# imported but not declared anywhere pip reads: score_transformer parses MusicXML with BeautifulSoup
# ('lxml-xml' parser: lxml); their models/model.py imports pytorch_lightning while requirements.txt
# lists only lightning (pinned to the same version so pip leaves torch alone); tqdm (dataset.py,
# run_eval.py); joblib runs scripts/beyer_dai.py score in parallel
pip install --quiet joblib beautifulsoup4 lxml tqdm \
  "pytorch-lightning==$(python -c 'import lightning; print(lightning.__version__)')"
# muster runs its evaluation programs through a bash script: make sure they are executable
python - <<'EOF'
import os, stat, muster
d = os.path.dirname(muster.__file__)
paths = [os.path.join(d, "evaluate_XML_voicePlus.sh")] + [os.path.join(d, "Programs", f)
                                                         for f in os.listdir(os.path.join(d, "Programs"))]
for p in paths:
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
print(f"muster: {len(paths)} executables in {d}")
EOF
if [ ! -s "$BD/MIDI2ScoreTF.ckpt" ]; then
  curl -L --fail -o "$BD/MIDI2ScoreTF.ckpt.part" "$CKPT_URL"
  mv "$BD/MIDI2ScoreTF.ckpt.part" "$BD/MIDI2ScoreTF.ckpt"
fi
ls -l "$BD/MIDI2ScoreTF.ckpt"
# smoke test: their code imports, and MUSTER rates a reference score against itself as error-free
cd "$ROOT"
python - "$BD/MIDI2ScoreTransformer" <<'EOF'
import glob, sys, warnings
sys.path.insert(0, sys.argv[1] + "/midi2scoretransformer")
warnings.simplefilter("ignore")
from utils import quantize_path  # noqa: F401  (imports muster, score_transformer, their tokenizer)
from models.roformer import Roformer  # noqa: F401
from muster import muster
ref = sorted(glob.glob("data/raw/asap_bd/Bach/Prelude/bwv_846/xml_score.musicxml"))
if ref:
    r = muster(ref[0], ref[0])
    print("MUSTER of a score against itself:", {k: r[k] for k in ("PitchER", "OnsetER", "MeanER")})
    assert r["MeanER"] is not None and r["MeanER"] < 1e-6, r
print("Beyer & Dai environment ready")
EOF
echo "next: bash pace/system_eval.sh"
