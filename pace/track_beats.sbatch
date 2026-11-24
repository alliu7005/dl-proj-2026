#!/bin/bash
# Beat This! on the MAESTRO audio of the real-transcription splits, then the tracker-error fit
# (one GPU job, well under an hour; submitted by pace/beats.sh after the ground-truth beats):
#   sbatch pace/track_beats.sbatch
#
#   1. scripts/track_beats.py: every ASAP performance with audio (test / train / val _transcribed),
#      each tracked by the Beat This! cross-validation model that never trained on it
#      -> data/processed/<split>/beats_tracked.npz (+ .json: checkpoint and beat F per performance)
#   2. scripts/fit_beat_noise.py: tracked vs ground-truth beats on train / val _transcribed (never
#      test) -> configs/beat_noise.json, the errors training and the simulated-tracker test
#      condition apply. Copy it back and commit it (like configs/noise_stats.json).
# Needs internet (the fold split file and the Beat This! checkpoints, ~80 MB each, cached in
# $TORCH_HOME on scratch) and the MAESTRO audio pace/real_asap.sbatch downloaded.
#SBATCH -J beattrack
#SBATCH --gres=gpu:1
#SBATCH --constraint=HX00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
source pace/common.sh
export TORCH_HOME="$HOME/scratch/.cache/torch"
nvidia-smi --query-gpu=name,memory.total --format=csv
# Beat This! (MIT); torch is already there, so pip keeps the CUDA build
python -c "import beat_this, rotary_embedding_torch, einops" 2>/dev/null || pip install beat-this==1.1.0 rotary-embedding-torch einops
python -c "from beat_this.inference import File2Beats; print('beat_this ready')"
python scripts/track_beats.py
python scripts/fit_beat_noise.py
echo "tracked beats:"; ls -l data/processed/*/beats_tracked.npz
