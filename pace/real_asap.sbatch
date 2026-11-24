#!/bin/bash
# Build the real-transcription splits of ASAP on PACE (one GPU job, ~2-4 h; resumable):
#   sbatch pace/real_asap.sbatch          (or: bash pace/real_asap.sh, which orders it after training jobs)
#
# Some ASAP performances also exist as audio in MAESTRO. Transkun V2 transcribes that audio, and
# scripts/build_transcribed.py pairs each transcription with the ASAP score, using the SAME windows,
# targets and vocab as the clean splits (checkpoints stay compatible):
#   test_transcribed   every ASAP test performance with audio (24); meta "maestro_split" marks the
#                      21 whose audio Transkun was trained on, so "real_unseen_audio" reports the rest
#   train_transcribed  only audio Transkun never trained on (MAESTRO test / validation): 85 performances
#   val_transcribed    same rule: 22 performances
# Training runs that start afterwards mix train_transcribed in automatically (scripts/train_model.py),
# and evaluations add the "real" condition: the baselines' real-transcription numbers (docs/baselines.md) are on
# these same test windows.
#SBATCH -J realasap
# GPU: any of H100 / H200 / RTX Pro Blackwell, whichever frees up first (node features; see
# `sinfo -h -o "%G | %f" | sort -u`). Not V100 / RTX 6000 (no bf16), not MI210 / R9700 (AMD:
# the CUDA build of torch can't use them); A100 / L40S work but at about half the speed. Override:
#   sbatch --constraint=gpu-a100 ...   (one node feature: PACE adds &cuda13, and Slurm rejects an OR list with it)
#SBATCH --gres=gpu:1
#SBATCH --constraint=HX00
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
source pace/common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv

python - <<'EOF' || { echo "this node has no internet: MAESTRO audio cannot be downloaded here"; exit 1; }
import urllib.request
urllib.request.urlopen("https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv", timeout=30)
EOF
# Transkun V2 is installed without its dependencies (they are in environment.yml; see its note)
python -c "import transkun" 2>/dev/null || pip install --no-deps transkun==2.0.1
python -c "import transkun, moduleconf, librosa, soxr, torchaudio; print('transkun ready')"

# 1. audio: test = every ASAP test performance with MAESTRO audio; train / val = only audio the
#    transcriber never saw in training (MAESTRO test / validation), so its errors are realistic
python scripts/fetch_maestro_audio.py --asap-split test
python scripts/fetch_maestro_audio.py --asap-split train,val --maestro-splits test,validation
# 2. Transkun V2 on the GPU (3 processes on disjoint recordings; a re-run skips finished ones)
python scripts/transcribe.py --model transkun --jobs 3 --normal-priority
# 3. pair with the scores: same windows, targets and vocab as the clean splits
python scripts/build_transcribed.py --asap-split test
python scripts/build_transcribed.py --asap-split train --maestro-splits test,validation
python scripts/build_transcribed.py --asap-split val --maestro-splits test,validation
# 4. metrical labels for the new splits (--aux-metrical training reads them), then the data checks:
#    targets and prompts identical to the clean splits, no transcriber-training audio in train / val
python scripts/build_metrical.py --splits train_transcribed,val_transcribed,test_transcribed --workers 8
python -m tests.test_integrity
echo "real-transcription splits ready:"
ls -d data/processed/*_transcribed
