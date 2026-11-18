#!/bin/bash
# Build data/processed on a CPU node: PianoCoRe/PERiScoPe download + ASAP/PERiScoPe windows + checks.
#   sbatch pace/data.sbatch
# Not needed if you copied a finished data/processed from Michael (see pace/README.md).
#SBATCH -J pianoscore-data
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
source pace/common.sh
if ! curl -sfI https://huggingface.co >/dev/null; then
    echo "This compute node has no internet. Run the two fetch commands below on the login node"
    echo "(they only stream data), then resubmit this job:"
    echo "  python scripts/fetch_pianocore_meta.py && python scripts/fetch_pianocore.py"
fi
[ -f data/raw/pianocore/metadata.parquet ] || python scripts/fetch_pianocore_meta.py
python scripts/fetch_pianocore.py            # resumes; skips row groups already downloaded
python scripts/build_dataset.py
python -m tests.test_integrity
python -m tests.test_scoreformer
du -sh data/processed
