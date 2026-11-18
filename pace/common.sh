# Sourced by the pace/*.sbatch jobs: conda env + caches on scratch.
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate dl-proj
export HF_HOME="$HOME/scratch/.cache/huggingface"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
