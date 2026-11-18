# Running on PACE ICE

Limits that matter: home 30 GB and scratch 300 GB (check with `pace-quota`); a job gets at most
8 h walltime and 16 GPU hours. Never run training or dataset builds on the login node.

## 0. Get the code onto GitHub

The `model` branch has to be pushed first (commit and push from your own machine).

## 1. One-time setup (login node)

Connect to the GT VPN (GlobalProtect), then `ssh <gtid>@login-ice.pace.gatech.edu` (or use the
dashboard: Files → Open in Terminal).

```bash
cd ~/scratch
git clone -b model https://github.com/alliu7005/dl-proj-2026.git
cd dl-proj-2026
bash pace/setup.sh          # conda env in scratch (~10 min), ASAP clones
```

## 2. Data

**Option A: copy Michael's finished `data/processed` (recommended).** His build includes the
real-transcription splits (`train_transcribed`, `val_transcribed`, `test_transcribed`, ...), which
need the MAESTRO audio and a Transkun run to rebuild. From his machine:

```bash
scp -r data/processed <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/data/
```

**Option B: build it on PACE** (ASAP + PERiScoPe, no real-transcription splits; the training script
skips those):

```bash
sbatch pace/data.sbatch     # CPU job, ~1 h: PianoCoRe download, build_dataset, integrity checks
```

If the job log says the compute node has no internet, run
`python scripts/fetch_pianocore_meta.py && python scripts/fetch_pianocore.py` on the login node
(after `module load anaconda3 && conda activate dl-proj`) and resubmit.

## 3. Smoke run (GPU, ~30–45 min)

```bash
sbatch --gres=gpu:h100:1 pace/smoke.sbatch
squeue -u $USER                     # job status
tail -f slurm-pianoscore-smoke-*.out
```

It checks that torch sees the GPU, runs `tests/test_scoreformer.py`, trains the base model for
1,500 steps (validation every 500), and evaluates 10 test pieces in oracle and chained mode.
What to look for in the log: validation loss falling at each eval, and `note_f1` / `onset_f1`
above zero by step 1,500.

If it stops with "torch can't see the GPU", the node's driver is older than the CUDA 12.8 torch
build in `environment.yml`. Check `driver_version` at the top of the log, then install a matching
build into the env, e.g. `pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126`.

## 4. Full runs (GPU, up to 8 h each)

```bash
sbatch pace/train.sbatch                                                 # sf_base, 20 epochs
sbatch --export=ALL,NAME=sf_clean,ARGS="--no-noise" pace/train.sbatch    # clean-input ablation
sbatch --export=ALL,NAME=s2s,ARGS="--arch score2seq" pace/train.sbatch   # Michael's 10M model, same loop
```

Each job trains, then runs `scripts/evaluate_pieces.py` (all conditions, oracle/chained/none,
MusicXML for 5 pieces). A run that hits the 8 h limit is resumed by submitting the same command
again (it picks up `outputs/runs/<NAME>/last.pt`).

Pick the GPU at submit time, e.g. `sbatch --gres=gpu:h100:1 pace/train.sbatch`. Names are
lowercase (`sinfo -o "%G" | sort -u` lists them): `h100`, `h200`, `rtx_pro_6000_blackwell` (16 cards,
often the shortest queue), `a100`, `l40s`. Avoid `v100` (no bf16, the scripts fall back to slower
fp32) and `mi210` / `r9700` (AMD GPUs, the CUDA build of torch can't use them).

## 4b. Post-milestone experiments

```bash
bash pace/overnight.sh          # 1 short CPU check + 4 GPU runs (Blackwell); GPU=h100 bash ... for another type
```

| run | what it tests | compare with |
|---|---|---|
| `sf_small` | time-based encoder at equal size (9.7M vs 10.2M), same recipe | `s2s` |
| `s2s_ctx` | training with corrupted context (`--context-noise 0.5`) | `s2s` |
| `sf_small_ctx` | both | `sf_small`, `s2s_ctx` |
| `sf_base_reg` | 45M model: dropout 0.2, weight decay 0.1, lr 2e-4, 6 epochs | `sf_base` |

`sf_base` overfit: its mean validation loss bottomed out at epoch 3-5 of 20 (0.275) and rose to
0.317. Context noise targets the other milestone finding: in chained evaluation every model scored
below decoding with no context at all, because training only ever showed it correct context.

## 5. Bring results back

```bash
scp -r <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/outputs/runs/sf_base ./outputs/runs/
```

Skip the `*.pt` files if you only need the logs and metrics (`log.csv`, `eval_pieces_*.json`,
`musicxml/`). The checkpoints are what goes to Hugging Face.
