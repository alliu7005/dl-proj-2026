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

**Build it on PACE** (ASAP + PERiScoPe):

```bash
sbatch pace/data.sbatch     # CPU job, ~1 h: PianoCoRe download, build_dataset, integrity checks
```

Then the real ASAP transcriptions (section 4e): `bash pace/real_asap.sh`. Copying another build
of `data/processed` from someone else's machine is not recommended: a build made from different
data can have a different vocabulary, which no existing checkpoint can read.

If the job log says the compute node has no internet, run
`python scripts/fetch_pianocore_meta.py && python scripts/fetch_pianocore.py` on the login node
(after `module load anaconda3 && conda activate dl-proj`) and resubmit.

## 3. Smoke run (GPU, ~30–45 min)

```bash
sbatch pace/smoke.sbatch
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
NAME=sf_clean ARGS="--no-noise" sbatch --export=ALL pace/train.sbatch    # clean-input ablation
NAME=s2s ARGS="--arch score2seq" sbatch --export=ALL pace/train.sbatch   # Michael's 10M model, same loop
```

`NAME` and `ARGS` are passed as environment variables with `--export=ALL` (the
`--export=ALL,NAME=..,ARGS=..` form splits at commas, which breaks arguments such as
`--modes guarded,meter`). Each job trains, then runs `scripts/evaluate_pieces.py` (all conditions, oracle/chained/none,
MusicXML for 5 pieces). A run that hits the 8 h limit is resumed by submitting the same command
again (it picks up `outputs/runs/<NAME>/last.pt`).

**Which GPU.** Every GPU job (the `.sbatch` files and the submit scripts) asks for one GPU of any
of these types, whichever frees up first: `--constraint="gpu-h100|gpu-h200|gpu-rtxpro-blackwell"`
(node features; `sinfo -h -o "%G | %f" | sort -u` lists them, and `sinfo -h -o "%G %t %D" | grep -i gpu`
shows how many nodes of each are idle / mixed / allocated). Asking for one exact type left jobs
waiting for Blackwell cards while H100 / H200 nodes had free GPUs. Left out on purpose: `gpu-v100`
and `gpu-rtx6000` (no bf16: slower fp32), `gpu-mi210` / `gpu-r9700` (AMD: the CUDA build of torch
can't use them); `gpu-a100` / `gpu-l40s` / `gpu-a40` work at about half the speed, which can push a
45M training plus its evaluation past the 8 h limit. To choose: `GPUS=gpu-h100 bash pace/...sh`, or
`sbatch --constraint=gpu-a100 ...` (several: `"gpu-h100|gpu-a100"`).

## 4b. Post-milestone experiments

```bash
bash pace/overnight.sh          # 1 short CPU check + 4 GPU runs (H100 / H200 / Blackwell)
```

| run | what it tests | compare with |
|---|---|---|
| `sf_small` | time-based encoder at equal size (9.7M vs 10.2M), same recipe | `s2s` |
| `s2s_ctx` | training with corrupted context (`--context-noise 0.5`) | `s2s` |
| `sf_small_ctx` | both | `sf_small`, `s2s_ctx` |
| `sf_base_reg` | 45M model: dropout 0.2, weight decay 0.1, lr 2e-4, 6 epochs | `sf_base` |

`sf_base` overfit: its mean validation loss bottomed out at step 20,000, in the 3rd of 20 epochs
(0.275), and rose to 0.317 by the end. Context noise targets the other milestone finding: in
chained evaluation every model scored below decoding with no context at all, because training only
ever showed it correct context.

Since then, every training run stops early (no validation improvement for 5 epochs, then a 1-epoch
learning-rate cooldown from the best checkpoint; see the main README), and every evaluation also
runs the `guarded` mode, so new `pace/train.sbatch` jobs need no extra flags.

## 4c. Re-evaluate and test the regularization

```bash
bash pace/reevaluate.sh         # every finished run: whatever it lacks of all modes on all test sets
sbatch -J simguard -c 4 --mem=16G -t 0:45:00 -o slurm-simguard-%j.out --wrap="source pace/common.sh; python scripts/simulate_guard.py"
```

`reevaluate.sh` submits one `pace/evaluate.sbatch` job per run (evaluation only, merged into the
run's existing results), and only for what the run is missing: `python3 -m src.eval.plan` (standard
library, runs on the login node) compares the requested conditions and modes with the run's
`eval_pieces_best.json`. A result counts only if it has the same guard margin, all performances and
the current `EVAL_VERSION` (`src/eval/plan.py`; bumped whenever decoding changes, which makes every
older result missing, so all modes of a comparison always come from the same code). It skips:

| skipped | why |
|---|---|
| `nothing missing` | every requested result exists (`FORCE=1` recomputes anyway) |
| `training not finished (phase ...)` | `status.json` says `train` / `cooldown`: its `best.pt` is not final (not even with `FORCE=1`) |
| `already being evaluated (job ...)` | an `eval_<run>` job is queued: two would rewrite the same files |

Each submitted job gets `--only-missing` and prints `to compute: ...` at the top of its log. To check
what a run lacks without submitting anything:

```bash
python3 -m src.eval.plan --run s2s --conditions real
```

Two runs separate the causes of `sf_base_reg`'s improvement (it changed dropout, weight decay,
learning rate and run length at once, and stopped one epoch after its best, too early to show
whether its validation loss would climb):

```bash
NAME=sf_base_e6 ARGS="--size base --epochs 6" sbatch --export=ALL pace/train.sbatch
NAME=sf_base_reg20 ARGS="--size base --dropout 0.2 --weight-decay 0.1 --lr 2e-4 --epochs 20" sbatch --export=ALL pace/train.sbatch
```

`sf_base_e6` (original settings, 6 epochs) vs `sf_base_reg`: is the gain just the shorter schedule?
`sf_base_reg20` (regularized, up to 20 epochs): does its validation loss stay near its minimum where
`sf_base`'s climbed? With early stopping it ends 5 epochs after its best plus the cooldown, which
answers the question either way.

## 4d. Bar alignment: context notes, auxiliary metrical head, meter consistency

```bash
bash pace/bar_alignment.sh       # labels + tests on CPU, then 5 GPU runs (H100 / H200 / Blackwell)
```

A CPU job first builds the metrical labels for this data build (`scripts/build_metrical.py`, a few
minutes) and runs the new tests on it; the five training jobs only start if that passes. Every run
trains with early stopping and is then evaluated in all five modes (oracle, chained, none, guarded,
meter).

| run | what it adds to s2s | compare with |
|---|---|---|
| `s2s_c` | 64 context notes before / 32 after each window | `s2s` |
| `s2s_a` | auxiliary metrical head (downbeat + beat phase of every note) | `s2s` |
| `s2s_ca` | both | `s2s_c`, `s2s_a` |
| `s2s_m_ca` | both, 25.9M parameters, dropout 0.2, weight decay 0.1, lr 2e-4 | `s2s_ca` |
| `sf_base_reg_ca` | both, on the regularized 45M Scoreformer (best with the true context, behind without it: bar F1 0.42 vs 0.52 with no context) | `sf_base_reg`, `s2s_ca` |

Every finished run gets the new `meter` mode without retraining (only that mode is decoded and merged
into its results), so all of them can be compared in every mode:

```bash
ARGS="--modes meter" bash pace/reevaluate.sh
```

## 4e. Real transcriptions of the ASAP performances

```bash
bash pace/real_asap.sh           # one GPU job, ~2-4 h; waits until queued training jobs have started
```

Some ASAP performances also exist as audio in MAESTRO. `pace/real_asap.sbatch` downloads only that
audio (range requests, not the 108 GB archive), transcribes it with Transkun V2, and pairs each
transcription with its ASAP score using the same windows, targets and vocabulary as the clean
splits (`scripts/build_transcribed.py`), so every existing checkpoint can use it:

| split | performances | what for |
|---|---|---|
| `test_transcribed` | 24 (21 with audio Transkun was trained on) | the "real" test condition, on the same windows as the baselines' real-transcription numbers; `real_unseen_audio` reports the 3 unseen ones |
| `train_transcribed` | 85, audio Transkun never trained on | real errors paired with ASAP scores, mixed into training |
| `val_transcribed` | 22, same rule | an extra validation set |

This adds about 1% to the real-transcription training data (PERiScoPe has 6,714 performances); its
main value is the real test condition comparable with the baselines. Training runs that start
after it exists mix `train_transcribed` in and validate on `val_transcribed` too (each run's
`config.json` records its data under `"data"`), which is why the script waits until every queued
or running training job has started: a batch already in the queue keeps training on the same data.
When every job has finished, add the real condition to every run that does not have it yet
(evaluations that ran after the split existed already include it, and are skipped):

```bash
ARGS="--conditions real" bash pace/reevaluate.sh
```

## 4f. Beats as an input

```bash
bash pace/beats.sh                   # ground-truth beats + tests (CPU), Beat This! + noise fit (GPU), 3 runs
STAGE=beat-runs bash pace/beats.sh   # only the two beat runs, from scratch, on the beats and the
                                     # configs/beat_noise.json already there (after a refit)
```

| job / run | what | waits for |
|---|---|---|
| `beatprep` (CPU) | `scripts/build_beats.py --check`: ground-truth beats for every split, compared with the ASAP annotations; the beat, plan and pipeline tests on this data build | - |
| `beattrack` (GPU, `pace/track_beats.sbatch`) | installs Beat This!, tracks the MAESTRO audio of test / train / val _transcribed with the held-out fold models, fits `configs/beat_noise.json` | `beatprep` |
| `sf_base_reg_ca_r` | the best model's recipe with the real-transcription data (new in these runs): the control | `beatprep` |
| `sf_base_reg_cab` | the same + `--beats` | `beattrack` |
| `s2s_m_cab` | the runner-up (25.9M Score2Seq) + `--beats` | `beattrack` |
| `eval_<run>` | the new `piece` mode for every finished run (`pace/reevaluate.sh`) | the three trainings have started |

`sf_base_reg_cab` vs `sf_base_reg_ca_r` is the effect of the beats alone (same data, same recipe);
`sf_base_reg_ca_r` vs `sf_base_reg_ca` the effect of the real-transcription data. Compare everything
with `python3 scripts/compare_runs.py` (it shows the beat-only conditions too).

`configs/beat_noise.json` is committed (fitted from the first beattrack run's beats). A refit after
a change to the error model needs no GPU: copy the beats back
(`tar czf ~/scratch/beats_files.tgz data/processed/*_transcribed/beats_*` on PACE, scp, `tar xzf`),
run `python scripts/fit_beat_noise.py` and commit the file; on PACE, `git pull` (remove the
untracked `configs/beat_noise.json` beattrack wrote first), then restart the beat runs with
`STAGE=beat-runs bash pace/beats.sh`: it refuses while their run directories exist (they would
resume with the errors they started with), so cancel their jobs and `rm -rf` them first. The
training log's first line names the run (`run: <name>`).

## 4g. Time-signature search

```bash
bash pace/tsearch.sh               # the new decoding modes on the best models + 3 runs
```

| job / run | what | starts |
|---|---|---|
| `eval_<run>@<condition>` | `ts_true`, `tsearch`, `tsearch_beat` (README: time-signature search) for `sf_base_reg_cab` and `s2s_m_cab` on real, clean, PERiScoPe, real_gt_beats: one job per test set (`PER_CONDITION=1 bash pace/reevaluate.sh`, 3 h each) | at once |
| `sf_base_reg_cab_s1` | the best recipe, seed 1: how much one run differs from the next | queued after those |
| `sf_base_reg_cabx` | the best recipe with exact beats in training, the real transcriptions too | " |
| `sf_base_reg_cab_c128` | the best recipe with 128 / 64 context notes | " |
| `eval_<run>@<condition>` | the remaining test sets of both, and `ts_true` / `tsearch` for `sf_base_reg_ca` (no beats) | once the trainings have started |

Each training evaluates itself afterwards with every mode; whatever its 8 h does not reach, and a
run that has to be resubmitted to finish training, the next morning:

```bash
PER_CONDITION=1 bash pace/reevaluate.sh sf_base_reg_cab_s1 sf_base_reg_cabx sf_base_reg_cab_c128
```

(A training that stopped at the time limit: resubmit it with the same NAME / ARGS as in
`pace/tsearch.sh`; it resumes from `last.pt`.) Compare with `python3 scripts/compare_runs.py`: the new
"Time-signature search per test set" tables put meter, ts_true, tsearch and tsearch_beat side by side.

## 4h. Beyer & Dai (Baseline 3) and the system-level comparison

```bash
bash pace/bd_setup.sh        # once, on the login node (~10 min): env bd311, their code, checkpoint
bash pace/system_eval.sh     # sysinputs (CPU) -> sys_<run> + bdpredict (GPU) -> sysscore (CPU)
```

`bd_setup.sh` ends with a smoke test (their code imports; MUSTER of a score against itself is 0).
`system_eval.sh` runs our systems (`RUNS="run:mode ..."`, default `sf_base_reg_cab:meter
sf_base_reg_cab:tsearch_beat s2s_m_cab:meter`) and theirs on the same whole-piece inputs and scores
everything with their metrics: `outputs/system_eval/results.md`, with their paper's Table 3 row next to
the clean results to check the reproduction. Bring it back with
`scp -r <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/outputs/system_eval/results.md outputs/system_eval/`.

## 4i. v2 Phase 1: the note labeller with the true grid

```bash
bash pace/labeller.sh        # labelprep (CPU: labels + true grids for every split, then the label
                             # tests) -> nl_base + nl_large (GPU: train, then evaluate every test set)
```

`pace/train_labeller.sbatch` trains one labeller and evaluates it (`scripts/eval_labeller.py`, mode
`grid_true`); resubmitting the same NAME / ARGS resumes from `last.pt`, and a finished run only
evaluates. The comparison: `python3 scripts/compare_runs.py`, section "Note labeller (v2) per test set".

## 4j. v2 Phases 2 + 3: the meter model, and the labeller on its grids

```bash
bash pace/meter.sh           # metprep (CPU: whole performances, ASAP + PERiScoPe, the alignment check,
                             # tests) -> met_<meter> (GPU: train, tune the decoder, evaluate, save grids)
                             # + metev_meter_base -> v2_<labeller>@<meter> (GPU: labellers on the grids)
SKIP_PREP=1 bash pace/meter.sh                # the data is built already
METERS="meter_v2" LABELLERS="nl_base" LARGE_ON="" bash pace/meter.sh   # a subset
```

The second round (the first, `meter_base`, got a third of the test bars' time signatures right):
`meter_v2` (PERiScoPe's 971 works added, the note encoder, augmentation, longer chunks),
`meter_v2_s1` (seed 1), `meter_v2_nonote` (no note encoder), `meter_v2_asap` (no PERiScoPe), each
trained for up to 30k steps (early stopping on validation downbeat F + time signature), then
`eval_meter.py --tune`; `metev_meter_base` re-evaluates the first model with a tuned decoder and on
the PERiScoPe test. `nl_base` labels every meter model's grids, `nl_large` those of `meter_v2` and
`meter_base` (`LABELLERS`, `LARGE_ON`), on clean, synthetic, real and periscope.

The `v2_*` jobs also wait for any `labeller` job still queued or running, so submit this whenever.
`metprep` prints how well the alignment-derived ground truth matches ASAP's annotations
(`data/processed/meter/check.json`). `outputs/runs/<meter>/eval_meter.json` has each grid's beat /
downbeat F and time-signature accuracy next to the tracker's own, with the untuned decoder's and
per-piece numbers, the time-signature confusions (`meter_scores_<condition>.json`: per performance);
`decode.json` the tuned decoder. `python3 scripts/compare_runs.py` tables the meter models ("Meter
models (v2)") and puts each labeller's true-grid
and meter-grid results side by side ("Note labeller (v2)").

## 4j2. Rebuild the data with the 2026-11-22 fixes

The sweep of 2026-11-22 (README, "Repo sweep") fixed three errors in how scores become training
data: ties in chords, pickups, and one piece in both train and val. They take effect once the data
is rebuilt:

```bash
bash pace/rebuild.sh                  # one CPU job (up to 8 h); the data in use is kept as data/processed_pre1122
AFTER=<its job id> bash pace/v2.sh    # 4k on the rebuilt data, starting when the rebuild has succeeded
```

The rebuild keeps the vocabulary (same token ids: every checkpoint reads the new data) and the
feature scaling, reuses the transcriptions and Beat This!'s tracked beats, and builds everything
that depends on the scores again (windows and targets, real-transcription splits, metrical labels,
annotated beats, note labels, meter data), then runs the data tests. To go back:
`rm -rf data/processed && mv data/processed_pre1122 data/processed`.

## 4k. v2, third round: the whole system end to end

After 4j (its data and runs) and 4h (`pace/bd_setup.sh`, `pace/system_eval.sh` once: the manifest and
Beyer & Dai's predictions):

```bash
bash pace/v2.sh                       # wave 1 (8 jobs); its last job submits wave 2 when the rest has finished
bash pace/v2.sh wave2                 # wave 2 by hand, if that job could not submit it (see its slurm-v2next log)
SKIP_SYSTEM=1 bash pace/v2.sh         # without the whole-piece comparison
METERS="meter_v3" METER_GROUPS="meter_v3" NEW_NLS="nl_base_gn" SYSTEMS="nl_base_gn+meter_v3" SEQ_SB="" bash pace/v2.sh   # a subset
```

PACE limits how many jobs a user may have submitted at once (a submission over the limit fails with
`QOSMaxSubmitJobPerUserLimit`), so models share GPU jobs and the batch comes in two waves of at most
8 jobs, each wave submitted when the one before has finished. `pace/v2_jobs.sh` does each job's work.
A job waiting on another's success is cancelled if that one fails (`--kill-on-invalid-dep=yes`), so
the waves never wait forever.

| wave | job | what | waits for |
|---|---|---|---|
| 1 | `v2tests` | the test suites, on PACE's Python (CPU) | - |
| 1 | `met_<a>+<b>` (x2 per group) | two meter models side by side on one GPU (`METER_GROUPS`: `meter_v3`+`_s1`, `_s2`+`_large`): no note encoder, up to 60k steps, 7.5 h per job; the second job resumes the first, then tunes the decoder and evaluates each model. It fails unless both are trained and evaluated | `v2tests` (2nd: the 1st) |
| 1 | `labellers` | `nl_base_gn` and `nl_base_gn_s1` (`--grid-noise 0.8`) side by side, then each on the true grid | `v2tests` |
| 1 | `v2grids` | the earlier meter models' grids decoded again (the fixed decoder), the oracle grids, then Beat This! on the clean test MIDI rendered to audio, each piece with the fold model that never heard it -> `outputs/system_eval/beats_synth.npz` (+ `.json`: its beat F next to the real audio's) (CPU) | `v2tests` |
| 1 | `v2next` | submits wave 2 | all of wave 1 (finished or not) |
| 2 | `evals_<nl>` (x3) | each labeller (`nl_base`, then also `nl_large`; the two new ones) on the true grid and every grid (the oracle, `meter_v2_nonote`, the new meter models), every test set | - |
| 2 | `sys_v2` | v2 on the whole test pieces (`SYSTEMS`; a system whose models are not finished is skipped) | the `evals_*` jobs |
| 2 | `sys_seq_a`, `sys_seq_b` | `sf_base_reg_cab` and `sf_base_reg_cab_c128` with the rendered-MIDI beats on clean MIDI, and the seq2seq systems scored before written again with the current export (spelling, overflow, per-bar fallbacks; the 4 real pieces they had no score for) | - |
| 2 | `v2final` | the onset-scale F1 for every run's saved predictions and `outputs/compare`, then every system in `outputs/system_eval` rated with Beyer & Dai's metrics -> `results.md` | all of wave 2 (finished or not) |

The meter runs take the longest (up to 16 h each; each model logs to `slurm-met_<run>-<job>.log`).
A meter pair not finished after its two jobs is continued with
`sbatch --gres=gpu:1 -c 16 --mem=120G -t 8:00:00 --wrap="source pace/common.sh; bash pace/v2_jobs.sh meters <a> <b>"`
(it resumes from `last.pt`), then `bash pace/v2.sh wave2`. A piece whose score cannot be written keeps
`<slug>.tokens.txt` and `<slug>.error.txt` in `outputs/system_eval/<system>/<condition>/`.

To bring everything back (PACE, then Windows):

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_round3.tgz outputs/compare outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/beats_synth.json outputs/system_eval/*/summary.json outputs/system_eval/*/*/*.error.txt outputs/system_eval/*/*/*.tokens.txt outputs/runs/meter_*/*.json outputs/runs/meter_*/log.csv outputs/runs/nl_*/*.json outputs/runs/nl_*/log.csv 2>/dev/null; ls -la v2_round3.tgz
```

```powershell
scp <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/v2_round3.tgz .
```

```powershell
tar xzf v2_round3.tgz
```

## 4l. The rest of the third round

After 4k (README, "Third round: results"):

```bash
bash pace/v2_finish.sh                # 7 jobs
RETRAIN="" bash pace/v2_finish.sh     # without the seq2seq retraining
```

| job | what | waits for |
|---|---|---|
| `seq_sf_base_reg_cab_c128_r` (x3) | the best seq2seq recipe (`RETRAIN_ARGS`: the flags `sf_base_reg_cab_c128` trained with) on the rebuilt data, as `sf_base_reg_cab_c128_r`. Each job resumes from `last.pt`; once training is done, the evaluation (`SEQ_MODES` on `SEQ_CONDITIONS`, only what is missing). A job fails while training is not done, and the next one continues it | the job before (finished or not) |
| `sysold_a`, `sysold_b` | `OLD_SYSTEMS` on whole pieces with the current export (the three seq2seq systems whose earlier export hit the time limit); `sysold_b` then runs the beat ablation: `ABLATION` on the clean test MIDI with the beats of the performances' real recordings (`system_scores.py --clean-beats audio`, 24 performances) into `outputs/system_eval_ablation`, next to a link to the same system's rendered-MIDI version | - |
| `sysnew` | the retrained model on whole pieces, with (`:sb`) and without the rendered-MIDI beats on clean MIDI (refused if training is not finished) | the last `seq_` job (finished or not) |
| `v2final` | `scripts/rescore.py` (onset-scale F1, and the results on the old targets re-scored on the current ones), `outputs/compare`, every system rated with Beyer & Dai's metrics (`outputs/system_eval/results.md`), then the ablation on its shared pieces (`outputs/system_eval_ablation/results.md`) | all of the above (finished or not) |

The retraining takes up to 24 h, then up to 8 h for `sysnew` and `v2final`. A run still not trained
after its three jobs is continued with `sbatch --gres=gpu:1 -c 8 --mem=48G -t 8:00:00 --wrap="source
pace/common.sh; bash pace/v2_jobs.sh seq2seq sf_base_reg_cab_c128_r <RETRAIN_ARGS>"` (the same
flags), then `bash pace/v2_jobs.sh systems ...` as in `sysnew`.

To bring everything back (PACE, then Windows):

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_finish.tgz outputs/compare outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/*/summary.json outputs/system_eval/*/*/*.error.txt outputs/system_eval/*/*/*.tokens.txt outputs/system_eval_ablation/results.md outputs/system_eval_ablation/results.json outputs/system_eval_ablation/*/summary.json outputs/runs/*/eval_pieces_*.json outputs/runs/*/status.json outputs/runs/*/config.json outputs/runs/sf_base_reg_cab_c128_r/log.csv slurm-v2final-*.out 2>/dev/null; ls -la v2_finish.tgz
```

```powershell
scp <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/v2_finish.tgz .
```

```powershell
tar xzf v2_finish.tgz
```

## 5. Compare the runs and bring results back

```bash
python3 scripts/compare_runs.py      # every evaluated run -> outputs/compare/ (login node, seconds)
```

It writes three files to `outputs/compare/` and prints the first:

| file | what |
|---|---|
| `summary.md` | the runs (size, context notes, aux head, regularization, best validation loss, epochs, how training ended, whether it trained on real transcriptions), then per test set the system output (meter mode, what the demo runs) and onset / bar F1 in every context mode; best value per column in bold |
| `margins.md` | the guard-margin sweeps, per run and test set, with chained / none as the two extremes |
| `results.csv` | every number, one row per run, test set and mode |

Copy the folder to the Windows machine (run in PowerShell from the repo folder) and open the `.md`
files in VS Code with Ctrl+Shift+V to see the tables rendered:

```powershell
scp -r <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/outputs/compare ./outputs/
```

```bash
scp -r <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/outputs/runs/sf_base ./outputs/runs/
```

Skip the `*.pt` files if you only need the logs and metrics (`log.csv`, `eval_pieces_*.json`,
`musicxml/`). The checkpoints are what goes to Hugging Face.
