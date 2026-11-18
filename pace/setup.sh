#!/bin/bash
# One-time setup on the PACE ICE login node (light work only: conda env + git clones).
# Run from the repo root, which should live in scratch (home is only 30 GB):
#   cd ~/scratch && git clone -b model https://github.com/alliu7005/dl-proj-2026.git
#   cd dl-proj-2026 && bash pace/setup.sh
set -euo pipefail
case "$PWD" in */scratch/*) ;; *) echo "warning: repo is not under ~/scratch (home quota is 30 GB)";; esac

# conda environments in scratch, as PACE recommends (~/.conda -> ~/scratch/.conda)
if [ ! -e ~/.conda ]; then
    mkdir -p ~/scratch/.conda && ln -s ~/scratch/.conda ~/.conda && echo "linked ~/.conda -> ~/scratch/.conda"
elif [ ! -L ~/.conda ]; then
    echo "note: ~/.conda is a real folder in home; the env (~6 GB) will count against the 30 GB home quota."
    echo "      To move it: see PACE's conda symlink tutorial, then rerun this script."
fi

module load anaconda3
if conda env list | grep -q "^dl-proj "; then
    echo "conda env dl-proj exists (update with: conda env update -f environment.yml)"
else
    conda env create -f environment.yml
fi

# ASAP: Beyer & Dai's fork (pinned) + original as a fallback for one score
mkdir -p data/raw
[ -d data/raw/asap_bd ] || git clone https://github.com/TimFelixBeyer/asap-dataset.git data/raw/asap_bd
git -C data/raw/asap_bd checkout -q 8cba199e15931975542010a7ea2ff94a6fc9cbee
[ -d data/raw/asap ] || git clone --depth 1 https://github.com/fosfrancesco/asap-dataset.git data/raw/asap
mkdir -p outputs
echo "setup done. Next: sbatch pace/data.sbatch  (or copy Michael's data/processed, see pace/README.md)"
