#!/bin/bash
# ~30-45 min GPU sanity run before the long training: CUDA check, model tests, 1,500 training
# steps of the base model (eval every 500), then a quick test-set evaluation (10 pieces).
#   sbatch pace/smoke.sbatch
#SBATCH -J pianoscore-smoke
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1:30:00
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
source pace/common.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), "| built for CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("torch can't see the GPU: the driver above is too old for this torch build. "
                     "Install a torch wheel for an older CUDA (see pace/README.md).")
PY
python -m tests.test_scoreformer
rm -rf outputs/runs/smoke_base
python scripts/train_model.py --name smoke_base --size base --max-steps 1500 --warmup 300 \
    --eval-every 500 --eval-loss-windows 200 --eval-decode-windows 32 --workers 7 --normal-priority
python scripts/evaluate_pieces.py --run smoke_base --ckpt last.pt --limit-perfs 10 \
    --modes oracle,chained --musicxml 2 --normal-priority
