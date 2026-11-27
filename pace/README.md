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

**Which GPU.** Every GPU job (the `.sbatch` files and the submit scripts) asks for one H100 or
H200, whichever frees up first: `--constraint=HX00`, a node feature both carry (`sinfo -h -o "%G | %f"
| sort -u` lists the features, and `sinfo -h -o "%G %t %D" | grep -i gpu` shows how many nodes of
each are idle / mixed / allocated). It has to be one feature: since November 2026 PACE adds
`&cuda13` to every GPU job's constraint, and Slurm leaves a job whose constraint mixes an OR list
with it (`gpu-h100|gpu-h200&cuda13`) pending forever as `BadConstraints` (`scontrol show job <id> |
grep Features` shows what the job ended up with). Asking for one exact GPU type left jobs waiting
while other fast nodes had free GPUs. Left out on purpose: `gpu-v100` and `gpu-rtx6000` (no bf16:
slower fp32; also only `cuda12`), `gpu-mi210` / `gpu-r9700` (AMD: the CUDA build of torch can't use
them); `gpu-a100` / `gpu-l40s` / `gpu-a40` work at about half the speed, which can push a 45M
training plus its evaluation past the 8 h limit. To choose another (one feature): `GPUS=gpu-a100
bash pace/...sh`, or `sbatch --constraint=gpu-a100 ...`. A job already pending with a constraint
that cannot be met takes a new one in place, keeping its dependencies: `scontrol update
JobId=<id> Features=HX00`.

Short GPU jobs (`pace/v2_round4.sh`: the labellers, about an hour each on an H100, the evaluations and
whole-piece systems) ask for `cuda13` instead, a feature every NVIDIA GPU with bf16 carries (A100,
A40, L40S, H100, H200; not V100 / RTX 6000, not the AMD cards). On 2026-11-23 the H100 / H200
queue was two days deep (`sbatch --test-only ...` gave a start on Nov 26), `cuda13` started the
next morning on an A100. To compare before submitting (nothing is submitted):
`for c in HX00 cuda13; do sbatch --test-only --gres=gpu:1 --constraint=$c -c 8 --mem=48G -t 4:00:00 --wrap=hostname; done`.

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
learning-rate cooldown from the best checkpoint; see docs/experiments.md), and every evaluation also
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
| `eval_<run>@<condition>` | `ts_true`, `tsearch`, `tsearch_beat` (docs/experiments.md: time-signature search) for `sf_base_reg_cab` and `s2s_m_cab` on real, clean, PERiScoPe, real_gt_beats: one job per test set (`PER_CONDITION=1 bash pace/reevaluate.sh`, 3 h each) | at once |
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

The sweep of 2026-11-22 (docs/experiments.md, "Repo sweep") fixed three errors in how scores become training
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

## 4l. v2, fourth round: rhythm and the written score

After 4k (docs/experiments.md, "Fourth round"):

```bash
bash pace/v2_round4.sh                # 8 jobs
```

| job | what | waits for |
|---|---|---|
| `r4tests` | the test suites, on PACE's Python (CPU) | - |
| `r4lab` | `nl_base_gn_f1` (checkpoint kept and training stopped on validation onset F1, `--select onset_f1`) and `nl_base_gn1_f1` (the same, every training window on the tracker-timed grid, `--grid-noise 1`) side by side, then each on the true grid | `r4tests` (cancelled if they fail) |
| `r4tempo` | `meter_v3_tp`: `meter_v3`'s checkpoint (linked) with its decoder tuned again, the written-tempo prior included (`eval_meter.py --tune --tempo-prior`), evaluated, its grids saved (CPU) | `r4tests` (cancelled if they fail) |
| `r4evals` | the new labellers and `nl_base_gn` on the true grid, the oracle grid, `meter_v3` and `meter_v3_tp`, every test set | `r4lab`, `r4tempo` (finished or not) |
| `r4sys_a`, `r4sys_b` | v2 on whole pieces with the new models (`SYS_A`, `SYS_B`), the last of each with every change (`:tidy,d95`) | `r4lab`, `r4tempo` (finished or not) |
| `r4sys_c` | the round-3 system tidied (`:tidy`), at `:d95` and `:keepall` (`:missing`: what the threshold job wrote is kept), then the beat ablation (`ABLATION` on the clean test MIDI with the real recordings' beats, `outputs/system_eval_ablation`) | `r4tests` (cancelled if they fail) |
| `r4final` | `outputs/compare`, every system rated with Beyer & Dai's metrics (`outputs/system_eval/results.md`), the ablation on its shared pieces (`outputs/system_eval_ablation/results.md`) | all of the above (finished or not) |

A v2 system spec takes options after a colon, comma-separated (`pace/v2_jobs.sh systems`): `tidy`,
`d<percent>` (e.g. `d95`), `keepall`, `missing`. The labellers take about an hour each, the tempo
tuning about one on CPU, each whole-piece system about 25 minutes: the batch takes about half a day.

To bring everything back (PACE, then Windows):

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_round4.tgz outputs/compare outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/*/summary.json outputs/system_eval/*/*/*.error.txt outputs/system_eval/*/*/*.tokens.txt outputs/system_eval_ablation/results.md outputs/system_eval_ablation/results.json outputs/system_eval_ablation/*/summary.json outputs/runs/meter_v3_tp/*.json outputs/runs/nl_*/*.json outputs/runs/nl_*/log.csv slurm-r4*.out slurm-labeller_*.log 2>/dev/null; ls -la v2_round4.tgz
```

```powershell
scp <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/v2_round4.tgz .
```

```powershell
tar xzf v2_round4.tgz
```

## 4m. The release

After 4l (docs/experiments.md, "Fourth round: results"):

```bash
bash pace/v2_release.sh               # 3 jobs
```

| job | what | waits for |
|---|---|---|
| `reltests` | the test suites, including the release's (CPU) | - |
| `relsys` | v2 on whole pieces in the released configuration (`nl_base_gn+meter_v3_tp:tidy,d95`), the same at `d80`, and with the labeller's second seed | `reltests` (cancelled if they fail) |
| `relfinal` | `models/` (`scripts/export_models.py`: each model without its optimizer state, its training vocabulary, the tempo-prior decoder, the model card, a manifest with sha256 sums; loaded once to check it), every system rated with Beyer & Dai's metrics, the paired comparisons (`outputs/system_eval/paired.md`) | `relsys` (finished or not) |

About 1.5 hours. `relfinal`'s log lists each model's size and ends with `wrote models (... MB)`;
if it shows a traceback instead, `models/` is left as it was (the export builds in
`models.partial/` and only replaces `models/` once the release loads).

**Upload** (from PACE: the models are already there). Once, a Hugging Face account and a token
with write access (huggingface.co, Settings, Access Tokens):

```bash
cd ~/scratch/dl-proj-2026 && module load anaconda3 && conda activate dl-proj && hf auth login
```

```bash
cd ~/scratch/dl-proj-2026 && module load anaconda3 && conda activate dl-proj && python scripts/download_models.py --check && python scripts/upload_models.py --repo dat-dev/scoreformer
```

`--private` keeps the repo to you (and an organization's members). The upload is tagged with the
release's name (`v2-round4`), so `download_models.py --revision v2-round4` always gets this
version. `HF_REPO` in `src/release.py` names this repo, so anyone gets the models with
`python scripts/download_models.py`.

**Publishing as training finishes (since 2026-11-26).** The `hf auth login` above, once, is all
it needs. Every run a job of `pace/v2_jobs.sh` finishes (meter models once evaluated, labellers,
seq2seq) goes to the repo through `scripts/publish_model.py`: kept under `runs/<run>/`, and made
the default of its folder when it scores at least as well as the one there on their shared test
sets. The demo then pulls it. A failed upload never fails a job. `PUBLISH=0 bash pace/<batch>.sh`
publishes nothing. By hand, for a run finished before:

```bash
cd ~/scratch/dl-proj-2026 && module load anaconda3 && conda activate dl-proj && python scripts/publish_model.py --run outputs/runs/meter_v4_gq
```

To bring the results back (PACE, then Windows):

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_release.tgz outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/paired.md outputs/system_eval/*/summary.json models/manifest.json slurm-rel*.out 2>/dev/null; ls -la v2_release.tgz
```

```powershell
scp <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/v2_release.tgz .
```

```powershell
tar xzf v2_release.tgz
```

## 4n. v2, fifth round: the note-value level (overnight)

After 4l (docs/experiments.md, "Fifth round"). With 4m: run the release first, then this with its
last job id, so the final scoring comes after both (3 + 4 = 7 jobs):

```bash
bash pace/v2_release.sh                       # prints 3 job ids; the last is relfinal
```

```bash
AFTER=<relfinal's job id> bash pace/v2_level.sh
```

| job | what | waits for |
|---|---|---|
| `lvtests` | the test suites (CPU) | - |
| `lvtrain` | `meter_v3_th` (a written-tempo head on `meter_v3`'s frozen trunk) and `meter_v3_ft` (`meter_v3` fine-tuned with it) side by side, then each tuned with the level choice and evaluated | `lvtests` (cancelled if they fail) |
| `lvsys` | `scripts/level_experiments.py` on `meter_v3_tp` (the priors and the labeller, every validation and test set), `nl_base_gn` on both new grids, v2 on whole pieces with each (`:tidy,d95`) | `lvtrain` (finished or not) |
| `lvfinal` | `outputs/compare`, every system rated with Beyer & Dai's metrics, `outputs/system_eval/paired.md` | `lvsys` and `AFTER` (finished or not) |

The training takes about 3 hours, the evaluations and systems about 2, the scoring about 1. To
bring it back (PACE, then Windows):

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_level.tgz outputs/compare outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/paired.md outputs/system_eval/*/summary.json outputs/level_experiments outputs/runs/meter_v3_th/*.json outputs/runs/meter_v3_th/log.csv outputs/runs/meter_v3_ft/*.json outputs/runs/meter_v3_ft/log.csv outputs/runs/nl_base_gn/eval_pieces_best.json slurm-lv*.out slurm-met_meter_v3_*.log 2>/dev/null; ls -la v2_level.tgz
```

```powershell
scp <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/v2_level.tgz .
```

```powershell
tar xzf v2_level.tgz
```

## 4o. v2, sixth round: Gangqinpu, Beat This! vs PM2S vs both, beat times from the notes

After 4n (docs/experiments.md, "Sixth round"). What it does, in short:
- **Gangqinpu** (~18k web piano recordings of many genres, Transkun transcriptions aligned to scores)
  becomes half of every meter-model training batch.
- **Three meter models** alike but for the beat tracker(s) they read: `meter_v4_gq` (Beat This!),
  `meter_v4_gq_pm` (PM2S, beats from the notes) and `meter_v4_gq_both` (Beat This! with PM2S's beats
  as evidence). Each trains on its trackers' real beats in half of the draws that have them, and on
  their simulated errors (fitted on real output, not guessed) in the rest.
- **Beat times from the notes** (snap / fill after decoding) on the release's meter model.

**Before you start** (once; on the login node, from `~/scratch/dl-proj-2026`):
1. `git pull` (the `model` branch).
2. Earlier rounds' outputs must be there: `outputs/runs/meter_v3/best.pt`, `outputs/runs/meter_v3_tp/decode.json`,
   `outputs/system_eval/manifest.json` and Beyer & Dai's setup (`~/scratch/beyer_dai`, section 4h). The script
   stops with a message if one is missing.
3. Accept the dataset's licence at https://huggingface.co/datasets/Wayne42/gangqinpu (CC BY-NC-SA 4.0,
   non-commercial; tick both boxes), with the Hugging Face account whose token you use below.
4. Log in with the cache the jobs use (`hf` comes from the project environment; a token with read
   access, from huggingface.co -> Settings -> Access Tokens):
   ```bash
   module load anaconda3; eval "$(conda shell.bash hook)"; conda activate dl-proj
   HF_HOME=$HOME/scratch/.cache/huggingface hf auth login      # older huggingface_hub: huggingface-cli login
   ```
5. The recordings (not in the dataset; `scripts/fetch_gangqinpu_audio.py` fetches them from the URLs
   it lists, on gangqinpu.com's file host) belong to that site and its uploaders: look at its terms
   first. They are fetched at one request a second, for this project only, never redistributed.
   `AUDIO_TRAIN=0` fetches no training recordings (validation and test only), `AUDIO_TRAIN=<n>` n.

**Space and time.** About 13 GB on scratch (~660 MB of Parquet, ~12 GB of MP3s at the default
`AUDIO_TRAIN=3000`, plus the built data). End to end about 1.5-2 days if jobs start promptly:
`gqdata` 3-6 h (mostly the recordings), `gqtrack` 2-5 h, training 2 x 8 h, `gqeval` 5-7 h,
`gqfinal` ~1 h; `gqrefine` runs alongside from the start and is back within ~8 h.

```bash
bash pace/v2_gangqinpu.sh                    # 8 jobs (PACE's limit per batch); SNAP=0.05 FILL=0.25
AUDIO_TRAIN=0 bash pace/v2_gangqinpu.sh      # without training recordings (validation and test still)
```

| job | what | waits for |
|---|---|---|
| `gqtests` | the test suites (CPU) | - |
| `gqdata` | Gangqinpu's Parquet downloaded (`scripts/fetch_gangqinpu.py`, ~660 MB, pinned revision), its meter data built (`build_meter_data.py --gangqinpu ... --gangqinpu-only`: `train/val/test_gangqinpu.pkl`, `gangqinpu_report.json` says what the filters dropped), PM2S's errors fitted on it (`scripts/fit_pm2s_noise.py` -> `configs/beat_noise_pm2s.json`), `meter_v3_tp` evaluated on its test set; then the recordings downloaded (`scripts/fetch_gangqinpu_audio.py`: every validation and test one and `AUDIO_TRAIN` = 3,000 training ones, ~12 GB, 1 request/s, a few hours; resumable, a failed file skipped) | `gqtests` (cancelled if they fail) |
| `gqtrack` | Beat This! on the recordings and PM2S on the notes (Gangqinpu, and PM2S on PERiScoPe): their beats stored per performance for training and testing (`scripts/track_meter_data.py` -> `data/processed/meter/tracked_*.npz`); `tracked_report.json` compares the two trackers on the same recordings; Beat This!'s errors on Gangqinpu fitted from its real beats (`fit_pm2s_noise.py --from-stored beat_this` -> `configs/beat_noise_bt_gangqinpu.json`: its off-beat rate on this music, measured) | `gqdata` (cancelled if it fails) |
| `gqrefine` | `meter_v3_tp_rf` (snap + fill), `meter_v3_tp_fill` (fill only), `meter_oracle_rf`: the grids' beat times taken from the notes (`v2_jobs.sh refine`), `nl_base_gn` on them, v2 on whole pieces with the two refined decoders (`:tidy,d95`); PM2S's beats against Beat This!'s (`scripts/compare_beats.py` -> `outputs/system_eval/beats_compare.md`), the release with PM2S's beats on clean MIDI and the transcriptions (`:tidy,d95,pm,rpm`) | `gqtests` (cancelled if they fail) |
| `gqtrain`, `gqtrain2` | Three meter models alike but for the tracker(s) they read: `meter_v4_gq` (Beat This!), `meter_v4_gq_pm` (PM2S) and `meter_v4_gq_both` (Beat This!, with PM2S's beats as evidence: `--aux pm2s`). `meter_v3`'s recipe, Gangqinpu half of every batch; every one draws its tracker(s)' real beats in half of the draws that have them (`--tracked`, `--tracked-share 0.5`) and their fitted errors in the rest (Beat This!'s on Gangqinpu: `configs/beat_noise_bt_gangqinpu.json`; PM2S's: `configs/beat_noise_pm2s.json`). The second job resumes the first; then each is tuned with the tempo prior (`meter_v4_gq_pm` on PM2S's errors) and evaluated | `gqtrack` (cancelled if it fails); `gqtrain` (finished or not) |
| `gqeval` | Each new meter model with its own tracker(s) (`v2_jobs.sh trackers`): `meter_v4_gq_pm` with PM2S's beats on every test set (`<condition>_pm2s`), the others with Beat This!'s real beats of Gangqinpu's test recordings (`gangqinpu_real`; `meter_v4_gq_both` with PM2S's as evidence), `meter_v3_tp` there too; `nl_base_gn` on the new grids; v2 on whole pieces with each (`:tidy,d95`; `:tidy,d95,pm,rpm` for PM2S). Beat This! against PM2S against both | `gqtrain2` (finished or not) |
| `gqfinal` | `outputs/compare`, every system rated with Beyer & Dai's metrics, `outputs/system_eval/paired.md` | `gqrefine`, `gqeval` (finished or not) |

**What to check as it runs** (`squeue -u $USER`; each job's log is `slurm-<job>-<id>.out`):
- after `gqdata`: `data/processed/meter/gangqinpu_report.json` (how many recordings gave grids and
  what the filters dropped: about 98% of the recordings, 81% of the beats on validation) and
  `data/raw/gangqinpu/audio/fetch_summary.json` (fetched / failed; failures per file in
  `download_log.jsonl`);
- after `gqtrack`: `data/processed/meter/tracked_report.json` (how many performances each tracker
  has beats for, and Beat This! against PM2S on the same recordings) and
  `configs/beat_noise_bt_gangqinpu.json` (`p_offbeat`, `p_double`, `p_half`: Beat This!'s measured
  errors on this music; missing if fewer than 30 recordings were tracked);
- in each new run's `outputs/runs/<run>/config.json`: `tracked_by_source` / `tracked_aux_by_source`
  (how many training performances have real beats; 0 = trained on the simulated tracker only).

**If something goes wrong:**
- *The recordings cannot be fetched* (all failed in `fetch_summary.json`): the batch goes on. The
  Beat This! models train on its simulated errors only (a warning in their logs; Gangqinpu then
  uses the ASAP-fitted profile), PM2S needs no audio, and `gangqinpu_real` is skipped at test.
  `python scripts/fetch_gangqinpu_audio.py --release data/raw/gangqinpu --out data/raw/gangqinpu/audio`
  resumes later (only what is missing).
- *A job failed*: its log says why. `gqtests` failing cancels everything after it: fix, pull and
  run the batch again (downloads and finished training runs are kept; the data build, tracking and
  evaluations run again).
- *Training not finished after `gqtrain2`* (`gqeval`'s log says "not done yet"): one more training
  job (resuming), then the evaluation and the final scoring again:
  ```bash
  STAGE=train bash pace/v2_gangqinpu.sh      # gqtrain3 -> gqeval -> gqfinal
  ```
- *Only the evaluation again* (e.g. after a fix in it): `STAGE=eval bash pace/v2_gangqinpu.sh`.

The refined grids need no training and come back first (a few hours). To bring everything back
(PACE, then Windows):

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_gq.tgz outputs/compare outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/paired.md outputs/system_eval/*/summary.json outputs/system_eval/beats_compare.* data/processed/meter/gangqinpu_report.json outputs/runs/meter_v4_gq/*.json outputs/runs/meter_v4_gq/log.csv outputs/runs/meter_v4_gq_both/*.json outputs/runs/meter_v4_gq_both/log.csv configs/beat_noise_bt_gangqinpu.json outputs/runs/meter_v4_gq_pm/*.json outputs/runs/meter_v4_gq_pm/log.csv configs/beat_noise_pm2s.json data/processed/meter/tracked_report.json data/raw/gangqinpu/audio/fetch_summary.json outputs/runs/meter_v3_tp/eval_meter.json outputs/runs/*_rf/*.json outputs/runs/meter_v3_tp_fill/*.json outputs/runs/nl_base_gn/eval_pieces_best.json slurm-gq*.out slurm-met_meter_v*_gq*.log 2>/dev/null; ls -la v2_gq.tgz
```

```powershell
scp <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/v2_gq.tgz .
```

```powershell
tar xzf v2_gq.tgz
```

## 4p. v2, seventh round: PM2S retrained without our test pieces

After 4o (docs/experiments.md, "Seventh round"). With PM2S's beats, v2 gained the most of any change
so far, but PM2S's released weights were trained on all of ASAP, our test pieces included.
`scripts/train_pm2s.py` trains its model again from scratch, by its own recipe, on our training
splits only: `pm2s_ho`. Every output made with it carries the tag `ho` (`PM2S_WEIGHTS` /
`PM2S_TAG`, `src/pm2s_beats.py weights_tag`), so the released PM2S's results stay as they are.

**Before you start:** `git pull`; the queue empty (8 jobs, PACE's limit); 4o's outputs there (the
script stops with a message if one is missing).

```bash
bash pace/v2_pm2s.sh
```

| job | what | waits for |
|---|---|---|
| `phtests` | the test suites (CPU) | - |
| `phtrain` | `pm2s_ho` trained (`scripts/train_pm2s.py`: ASAP's training pieces, their transcriptions, PERiScoPe and Gangqinpu, validation on their val splits; up to 7.5 h, about 2 h expected); fails unless finished | `phtests` (cancelled if they fail) |
| `phtrack` | its beats of Gangqinpu's and PERiScoPe's notes stored (`tracked_*_pm2s_ho.npz`, `tracked_report_ho.json`), its errors fitted (`configs/beat_noise_pm2s_ho.json`), its beats on the ASAP test inputs (`outputs/system_eval/beats_compare_ho.md`) | `phtrain` (cancelled if it fails) |
| `phsys` | the release and `meter_v4_gq_pm` on whole pieces with its beats (`..._pm_rpm_ho`); `meter_v4_gq_pm` at the meter level with them (`<condition>_pm2s_ho`) | `phtrain` (cancelled if it fails) |
| `phmeter`, `phmeter2` | `meter_v4_gq_pmho`: `meter_v4_gq_pm`'s recipe with `pm2s_ho`'s stored beats and fitted errors; tuned and evaluated; the second job resumes | `phtrack`; `phmeter` (finished or not) |
| `pheval` | `meter_v4_gq_pmho` with `pm2s_ho`'s beats on every test set, `nl_base_gn` on its grids, whole pieces (`:tidy,d95,pm,rpm`) | `phmeter2` (finished or not) |
| `phfinal` | `outputs/compare`, every system rated with Beyer & Dai's metrics, `outputs/system_eval/paired.md` | `phsys`, `pheval` (finished or not) |

**What to check:** after `phtrain`, `outputs/runs/pm2s_ho/log.csv` (validation beat / downbeat F
per source) and `status.json`; after `phtrack`, `outputs/system_eval/beats_compare_ho.md` next to
`beats_compare.md` (the released PM2S, same inputs) and `data/processed/meter/tracked_report_ho.json`
next to `tracked_report.json`.

**If something goes wrong:**
- *`phtrain` stopped before the end:* `bash pace/v2_pm2s.sh` again (it resumes from `last.pt`).
- *`meter_v4_gq_pmho` not done after `phmeter2`* (`pheval`'s log says "not done yet"):
  `STAGE=meter bash pace/v2_pm2s.sh`.
- *Only the evaluation again:* `STAGE=eval bash pace/v2_pm2s.sh`.

To bring the results back (PACE, then Windows, from the repo folder):

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_pm2s.tgz outputs/compare outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/paired.md outputs/system_eval/beats_compare_ho.* data/processed/meter/tracked_report_ho.json configs/beat_noise_pm2s_ho.json outputs/runs/pm2s_ho/*.json outputs/runs/pm2s_ho/log.csv outputs/runs/meter_v4_gq_pmho/*.json outputs/runs/meter_v4_gq_pmho/log.csv outputs/runs/meter_v4_gq_pm/eval_meter.json outputs/runs/nl_base_gn/eval_pieces_best.json slurm-ph*.out slurm-met_meter_v4_gq_pmho*.log 2>/dev/null; ls -la v2_pm2s.tgz
```

```powershell
scp <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/v2_pm2s.tgz .
```

```powershell
tar xzf v2_pm2s.tgz
```

## 4q. v2, eighth round: the labeller on Gangqinpu's scores, whole pieces outside classical piano

After 4p (docs/experiments.md, "Eighth round"). The labeller has only seen classical scores, and
"more general" has only been measured on the meter model. This round trains the labeller with
Gangqinpu's written scores (`nl_gq`) and scores every system on 60 of Gangqinpu's test pieces as
whole pieces, with the ASAP benchmark's metrics (`outputs/system_eval_gangqinpu`). It also trains
`meter_v4_gq` longer (`meter_v4_gql`) and puts the release meter model's second seed through
whole pieces (`meter_v3_s1_tp`).

**Before you start:**
- `git pull`;
- the queue empty: 8 jobs, PACE's limit, so 4p's batch must have finished;
- 4o's outputs there: the Gangqinpu release, its test recordings, `meter_v4_gq`, `meter_v4_gq_pm`.
  The script stops with a message if one is missing.

```bash
bash pace/v2_gqlabel.sh
```

| job | what | waits for |
|---|---|---|
| `lbtests` | the test suites (CPU) | - |
| `lbdata` | Gangqinpu's scores as windows (`scripts/build_gangqinpu_windows.py` -> `data/processed/{train,val,test}_gangqinpu`, each with a `report.json`) and their labels (CPU); `meter_v3_s1` with the tempo-prior decoder (`meter_v3_s1_tp`) | `lbtests` |
| `lbinputs` | 60 whole test pieces, one per score (`scripts/gangqinpu_inputs.py`): the written score as reference, the aligned notes (`gangqinpu`), Transkun on the recording (`gangqinpu_real`), Beat This!'s beats of the recording | `lbtests` |
| `lbtrain` | `nl_base_gn` scored on Gangqinpu's windows first, so the gate that publishes `nl_gq` compares them on pop too; `nl_gq` trained (`nl_base_gn`'s recipe, Gangqinpu half of the windows) and evaluated on the true grid; `nl_gq` and `nl_base_gn` on `meter_v3_tp`'s and `meter_v4_gq`'s grids, Gangqinpu's windows included | `lbdata` (cancelled if it fails) |
| `lbmeter` | `meter_v4_gql` (`meter_v4_gq` with 100k steps) trained, tuned and evaluated | `lbtests` |
| `lbsys` | whole pieces: `nl_gq` with `meter_v3_tp` and `meter_v4_gq` (ASAP and Gangqinpu), `nl_base_gn` with `meter_v3_s1_tp` (ASAP) | `lbtrain`, `lbinputs` (finished or not) |
| `lbsys2` | whole pieces on Gangqinpu: the release, `meter_v4_gq`, `meter_v4_gq_pm` with PM2S, seq2seq; Beyer & Dai's predictions there | `lbinputs` (cancelled if it fails) |
| `lbfinal` | `meter_v4_gql` on whole pieces (if trained); `outputs/compare`; Beyer & Dai's metrics for every system on both test sets; `outputs/system_eval/paired.md`, `outputs/system_eval_gangqinpu/paired.md` | `lbsys`, `lbsys2`, `lbmeter` (finished or not) |

**What to check:**
- after `lbdata`: `data/processed/train_gangqinpu/report.json`. It should show about 98% of the
  performances kept, no tokens outside the vocabulary, and few windows dropped.
- after `lbinputs`: `outputs/system_eval_gangqinpu/inputs_report.json` (what was skipped). If
  Transkun is not installed in the environment, there is no `gangqinpu_real` and its log says so.
- after `lbtrain`: `outputs/runs/nl_gq/log.csv`. `val_gq_onset_f1` is the labeller on Gangqinpu's
  validation windows; `val_onset_f1` and `val_real_onset_f1` should stay near `nl_base_gn`'s.

**If something goes wrong:**
- *`nl_gq` not done after `lbtrain`:* `bash pace/v2_jobs.sh labellers nl_gq` in a GPU job resumes
  it, then `STAGE=eval bash pace/v2_gqlabel.sh`.
- *Only the whole pieces and the scoring again:* `STAGE=eval bash pace/v2_gqlabel.sh`.

To bring the results back (PACE, then Windows, from the repo folder):

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_gqlabel.tgz outputs/compare outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/paired.md outputs/system_eval_gangqinpu/results.md outputs/system_eval_gangqinpu/results.json outputs/system_eval_gangqinpu/paired.md outputs/system_eval_gangqinpu/manifest.json outputs/system_eval_gangqinpu/inputs_report.json data/processed/*_gangqinpu/report.json outputs/runs/nl_gq/*.json outputs/runs/nl_gq/log.csv outputs/runs/nl_base_gn/eval_pieces_best.json outputs/runs/meter_v4_gql/*.json outputs/runs/meter_v4_gql/log.csv outputs/runs/meter_v3_s1_tp/*.json slurm-lb*.out slurm-labeller_nl_gq*.log slurm-met_meter_v4_gql*.log 2>/dev/null; ls -la v2_gqlabel.tgz
```

```powershell
scp <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/v2_gqlabel.tgz .
```

```powershell
tar xzf v2_gqlabel.tgz
```

## 4r. v2, ninth round: PM2S retrained with stable optimisation

After 4q (docs/experiments.md, "Ninth round"). The seventh round's PM2S without our test pieces
trained unstably and was best at step 10,000 of 60,000. Even so, it made v2 for MIDI input level
with seq2seq and Beyer & Dai on clean MIDI. This round trains it twice with stable settings
(learning rate 3e-4 and 1e-4) and picks the better one by its validation score, linked as
`pm2s_hos` (tag `hos`). Then it trains the meter model for it (`meter_v4_gq_pmhs`) and scores v2
with it and `nl_gq` on ASAP and Gangqinpu, against today's default and against Beat This! on the
rendered MIDI (what a MIDI upload gets now).

**Before you start:**
- `git pull`. PACE has `configs/beat_noise_pm2s_ho.json` untracked, so move it aside first (the
  command is below).
- The queue empty: 8 jobs.
- 4q's outputs there. The script stops with a message if one is missing.

```bash
cd ~/scratch/dl-proj-2026 && [ -f configs/beat_noise_pm2s_ho.json ] && mv configs/beat_noise_pm2s_ho.json configs/beat_noise_pm2s_ho.json.pace; git pull && squeue -u $USER
```

```bash
bash pace/v2_pm2s2.sh
```

| job | what | waits for |
|---|---|---|
| `p2tests` | the test suites (CPU) | - |
| `p2train_a`, `p2train_b` | `pm2s_ho2` (lr 3e-4) and `pm2s_ho3` (lr 1e-4): AdamW betas 0.9 / 0.999, eps 1e-8, clipping 1.0, warmup 1000, cosine; up to 7.5 h each (about 3 h expected) | `p2tests` |
| `p2track` | the better one picked (`scripts/pick_run.py` -> `outputs/runs/pm2s_hos`, `chosen.json`); its beats stored (`tracked_*_pm2s_hos.npz`), its errors fitted (`configs/beat_noise_pm2s_hos.json`), its beats on the ASAP test inputs (`beats_compare_hos.md`) | both trainings (finished or not) |
| `p2sys` | the release with its beats (`_pm_rpm_hos`); `meter_v4_gq_pm` at the meter level with them; on Gangqinpu, `nl_gq` + `meter_v4_gq` with Beat This! on the rendered MIDI (`_rsb`); `nl_gq` + `meter_v4_gq` with the timing check (`_tc`, `src/timing.py`) on ASAP and Gangqinpu | `p2track` |
| `p2meter` | `meter_v4_gq_pmhs` trained, tuned and evaluated | `p2track` |
| `p2eval` | `meter_v4_gq_pmhs` with its beats on every test set, `nl_gq` on its grids, whole pieces (`nl_gq+meter_v4_gq_pmhs:tidy,d95,pm,rpm`) on ASAP and Gangqinpu | `p2meter` (finished or not) |
| `p2final` | `outputs/compare`, Beyer & Dai's metrics for every system on both test sets, both `paired.md` | `p2sys`, `p2eval` (finished or not) |

**What to check:**
- after the trainings: `outputs/runs/pm2s_ho2/log.csv` and `pm2s_ho3/log.csv`. `loss_beat` should
  fall and stay down (pm2s_ho's rose after step 8,000), with `grad_norm` steady.
- after `p2track`: `outputs/runs/pm2s_hos/chosen.json` (which one won, both scores), then
  `outputs/system_eval/beats_compare_hos.md` next to `beats_compare_ho.md` and `beats_compare.md`.

**If something goes wrong:**
- *A training stopped before the end:* `bash pace/v2_pm2s2.sh` again (it resumes from `last.pt`).
- *`meter_v4_gq_pmhs` not done after `p2meter`:* `STAGE=meter bash pace/v2_pm2s2.sh`.
- *Only the evaluation again:* `STAGE=eval bash pace/v2_pm2s2.sh`.

To bring the results back (PACE, then Windows, from the repo folder):

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_pm2s2.tgz outputs/compare outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/paired.md outputs/system_eval/beats_compare_hos.* outputs/system_eval/*/summary.json outputs/system_eval_gangqinpu/results.md outputs/system_eval_gangqinpu/results.json outputs/system_eval_gangqinpu/paired.md outputs/system_eval_gangqinpu/*/summary.json data/processed/meter/tracked_report_hos.json configs/beat_noise_pm2s_hos.json outputs/runs/pm2s_ho2/*.json outputs/runs/pm2s_ho2/log.csv outputs/runs/pm2s_ho3/*.json outputs/runs/pm2s_ho3/log.csv outputs/runs/pm2s_hos/chosen.json outputs/runs/meter_v4_gq_pmhs/*.json outputs/runs/meter_v4_gq_pmhs/log.csv outputs/runs/meter_v4_gq_pm/eval_meter.json outputs/runs/nl_gq/eval_pieces_best.json slurm-p2*.out slurm-met_meter_v4_gq_pmhs*.log 2>/dev/null; ls -la v2_pm2s2.tgz
```

```powershell
scp <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/v2_pm2s2.tgz .
```

```powershell
tar xzf v2_pm2s2.tgz
```

## 4s. The timing check on whole pieces (if 4r was submitted before it)

`pace/v2_pm2s2.sh` measures the timing check in its `p2sys` job. If that batch was submitted before
the timing check existed, `pace/v2_timing.sh` does the same in 3 jobs. Submit it once the queue has
room, since PACE allows about 8 jobs:

```bash
bash pace/v2_timing.sh
```

| job | what | waits for |
|---|---|---|
| `tctests` | the test suites (CPU) | - |
| `tcsys` | `nl_gq` + `meter_v4_gq:tidy,d95,tc` on ASAP and Gangqinpu's whole pieces | `tctests` |
| `tcfinal` | Beyer & Dai's metrics for every system (ASAP, Gangqinpu), both `paired.md` | `tcsys` (finished or not) |

To bring the results back:

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_timing.tgz outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/paired.md outputs/system_eval/*/summary.json outputs/system_eval_gangqinpu/results.md outputs/system_eval_gangqinpu/results.json outputs/system_eval_gangqinpu/paired.md outputs/system_eval_gangqinpu/*/summary.json slurm-tc*.out 2>/dev/null; ls -la v2_timing.tgz
```

## 4t. Tenth round: the model-side fixes, and every rule measured

`pace/v2_decode.sh` tests the deep-learning side of the rhythm problems the demo's rules patch, and
scores each rule on its own (whole pieces, ASAP and Gangqinpu). It needs the queue empty (8 jobs):

```bash
bash pace/v2_decode.sh
```

| job | what | waits for |
|---|---|---|
| `dctests` | the test suites (CPU) | - |
| `dcaudit` | the label audit (`scripts/audit_labels.py`): how often Gangqinpu's and ASAP's scores write a beat as triplets that was played straight | `dctests` |
| `dctune` | `nl_gq`'s beat decoding tuned on validation (`beat_decode.json`), the labeller evaluated with it, published (the demo pulls it); `meter_v4_gq_lt`: the meter decoder's local tempo term tuned and evaluated | `dctests` |
| `dctrain` | `nl_gq_mc`: `nl_gq` retrained with the contradicted labels left out of the loss (`--mask-contradicted`), evaluated, published if better | `dctests` |
| `dceval` | `nl_gq_mc` on `meter_v4_gq`'s grids; its beat decoding tuned | `dctrain` |
| `dcasap` | whole pieces on ASAP: `nl_gq` + `meter_v4_gq:tidy,d95,tc` alone and with `tb`, `rd`, `jf`, `bd`, the rules together, all of them; `nl_gq_mc`; `meter_v4_gq_lt` (systems already scored are kept) | `dctune`, `dceval` (finished or not) |
| `dcgq` | the same on Gangqinpu's whole pieces | `dctune`, `dceval` (finished or not) |
| `dcfinal` | Beyer & Dai's metrics for every system (ASAP, Gangqinpu), both `paired.md` | `dcasap`, `dcgq` (finished or not) |

To bring the results back:

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_decode.tgz outputs/label_audit.json outputs/runs/nl_gq/beat_decode.json outputs/runs/nl_gq/eval_pieces_best.json outputs/runs/nl_gq_mc/config.json outputs/runs/nl_gq_mc/status.json outputs/runs/nl_gq_mc/log.csv outputs/runs/nl_gq_mc/eval_pieces_best.json outputs/runs/nl_gq_mc/beat_decode.json outputs/runs/meter_v4_gq_lt/decode.json outputs/runs/meter_v4_gq_lt/eval_meter.json outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/paired.md outputs/system_eval/*/summary.json outputs/system_eval_gangqinpu/results.md outputs/system_eval_gangqinpu/results.json outputs/system_eval_gangqinpu/paired.md outputs/system_eval_gangqinpu/*/summary.json slurm-dc*.out slurm-labeller_nl_gq_mc-*.log 2>/dev/null; ls -la v2_decode.tgz
```

## 4u. The rule verdict (after or alongside 4t)

`pace/v2_decode2.sh` measures each correction rule added to the models alone (labeller + its
tuned beat decoding), for `nl_gq` and the retrained `nl_gq_mc`, and lets the benchmark decide which
rules the demo keeps (`scripts/rule_verdict.py`). It waits for 4t's jobs that are still queued
(found by name), so it can be submitted while they run (4 jobs):

```bash
bash pace/v2_decode2.sh
```

| job | what | waits for |
|---|---|---|
| `d2tests` | the test suites (CPU) | - |
| `d2asap` | whole pieces on ASAP: `nl_gq` + decoding (+ `tb` / `jf`); `nl_gq_mc` alone, + decoding, + `tc` / `tb` / `jf`, every rule | `d2tests`; `dctune`, `dceval` if queued |
| `d2gq` | the same on Gangqinpu's whole pieces | the same |
| `d2final` | Beyer & Dai's metrics, both `paired.md`, the rule verdict (`outputs/rule_verdict.md`), published to the release | `d2asap`, `d2gq`; `dcfinal` if queued |

The verdict, for the labeller the release serves: the beat decoding stays on unless it makes the
system significantly worse; a correction rule (`timing_check`, `timing_beats`, `fix_jumps`) stays
on only if it makes it significantly better on real recordings (ASAP or Gangqinpu) and worse on
neither. `publish_model.py --settings` puts the decisions in the release manifest; the demo reads
them at its next check (`PUBLISH=0 bash pace/v2_decode2.sh`: decided, not published).

To bring the results back:

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_decode2.tgz outputs/rule_verdict.json outputs/rule_verdict.md outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/paired.md outputs/system_eval_gangqinpu/results.md outputs/system_eval_gangqinpu/results.json outputs/system_eval_gangqinpu/paired.md slurm-d2*.out 2>/dev/null; ls -la v2_decode2.tgz
```

## 4w. The pilot: every idea at small scale (before 4v)

Before any full-scale run, `pace/v2_pilot.sh` tries each idea for beating the release cheaply, and
`outputs/pilot/report.md` gives each a go / no go (`scripts/pilot_report.py`):

| | idea | at pilot scale |
|---|---|---|
| A | meter ensembles | no training: the release's meter model on 5 views of each performance; round 7's three PM2S meter models averaged |
| B | synthetic performances of written scores (PDMX) | 3,000 scores; meter_v4_gq's recipe for 12k steps without and with them |
| C | PM2S retrained with stable optimisation | 16k steps at two rates: does the loss stay down |
| D | deleting above p 0.8 | on the release |
| E | choosing seq2seq or v2 per piece | its ceiling, from the scores |

**Before you start:** `git fetch && git checkout michael && git pull`; the queue empty (8 jobs). For
B, the scores once, on the login node (compute nodes may have no internet; about 2 GB):

```bash
python scripts/fetch_pdmx.py --list          # the Zenodo record's files
python scripts/fetch_pdmx.py --max 20000     # the scores' archive -> data/raw/pdmx/mxl/
bash pace/v2_pilot.sh
```

Without the scores, B is skipped (the script says so) and the rest runs. Nothing is published.
About 4-6 hours; `q1final` writes the report.

To bring the results back:

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_pilot.tgz outputs/pilot outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/paired.md outputs/system_eval_gangqinpu/results.md outputs/system_eval_gangqinpu/results.json outputs/system_eval_gangqinpu/paired.md outputs/runs/meter_v4_gq*/eval_meter.json outputs/runs/meter_v4_gq*/decode.json outputs/runs/meter_v4_gq_*p12k/log.csv outputs/runs/pm2s_pilot*/log.csv slurm-r2*.out slurm-p3*.out data/processed/meter/synthetic_report.json slurm-q1*.out slurm-met_meter_v4_gq_*p12k*.log slurm-pm2s_pilot*.log 2>/dev/null; ls -la v2_pilot.tgz
```

```bash
scp <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/v2_pilot.tgz . && tar xzf v2_pilot.tgz
```

## 4x. The pilot, part 2: 2/4 against 4/4 (alongside 4w)

The demo writes one piece in two bar lengths (Lemon: 45 bars of 4/4 and 23 of 2/2; Maple Leaf Rag:
46 of 2/4 and 19 of 4/4). `pace/v2_pilot2.sh` tries three fixes at small scale; its jobs wait for
4w's still queued, so both fit the queue:

| | fix | at pilot scale |
|---|---|---|
| F | one meter per piece after decoding (`src/meter.py consistent_meter`, `decode.json` `one_meter`) | no training: the release's meter model with the rule at 0.5 and 0.6 (`meter_v4_gq_om5`, `_om6`), meter scores and whole pieces |
| G | the repetition features (`rep_features`, `train_meter.py --rep-features`) | 12k steps against 4w's `meter_v4_gq_p12k` |
| H | bar-length-tolerant training (`--tolerant`) | 12k steps, judged on the time signature with 2/4 = 4/4 = 2/2 |

```bash
git pull && bash pace/v2_pilot2.sh
```

Every meter evaluation now also reports `ts_acc_equiv` (2/4 = 4/4 = 2/2, 6/8 = 12/8) and
`ts_changes_per_100_bars`. `outputs/pilot/report.md` gets sections F-H. Bring the results back with
4w's command (it includes them).

## 4y. The pilot, part 3: spelling, stems and swing (alongside 4w, 4x)

Three rules on the written score, each its own system against the release (`nl_gq` + `meter_v4_gq`,
`tidy,d95`). `pace/v2_pilot3.sh` waits for 4w's and 4x's jobs still queued:

| suffix | rule | why |
|---|---|---|
| `_ps` | PKSpell's key signature and spellings (`src/pkspell.py`; Foscarin et al., ISMIR 2021, MIT) | spelling error ~12% against Beyer & Dai's 8.6%; on music21's corpus it spells 97.5% of notes right against v2's 84.9% |
| `_eg` | clefs from each staff's range, one-voice stems by the engraver's rule (`src/readable.py engrave_score`) | stem error ~51% against 25%: v2 set stems only where a staff has two voices |
| `_sw` | swung pairs written straight under a swing marking (`src/swing.py`) | Lemon: 78% -> 96% of the sheet's right-hand onsets in bars 2-7 |

```bash
module load anaconda3 && conda activate dl-proj    # the login node fetches PKSpell's weights
git pull && bash pace/v2_pilot3.sh
```

`_ps` and `_eg` are applied to the release's scores as written already
(`scripts/postprocess_scores.py`); `_sw` and all three together run through the pipeline. Section I
of `outputs/pilot/report.md` has the MUSTER and the score-similarity spelling, stem and staff
errors, with a go / no go. PKSpell was trained on ASAP's scores, our ASAP test pieces included:
Gangqinpu is its honest test.

## 4z. Headroom tests: what an idea could gain at most (no training)

Before building a proposed improvement, `pace/v2_headroom.sh` (`scripts/headroom.py`) measures its
ceiling with the released models:
- **A learned duration / position chooser** (picking each note's written length from the
  surrounding notes among the labeller's own top candidates): `nl_gq`'s heads with the best of
  their top 2 / 4, the windows rendered and scored. If even that oracle barely lifts duration
  accuracy, the chooser is not worth training.
- **Local meter repair** (a ranker over alternative grids where the decoder is unsure):
  `meter_v4_gq`'s grid against the best of the decoded one, bar lines shifted by 1-3 beats, note
  values halved or doubled, each of the 3 likeliest time signatures, and one meter per piece. Per
  piece and per 8 bars.

```bash
git pull && bash pace/v2_headroom.sh       # 4 jobs, about an hour or two
```

`outputs/headroom/report.md`; bring it back with `scp` (it is small).

## 4aa. Twelfth round: everything the pilots said to scale, one stage after another

The pilot (4w-4y) and the headroom tests (4z) left four things worth a full run. `pace/v2_round12.sh`
submits them all at once, each stage waiting for every job of the one before (`afterany`), so they
run in turn and never compete for the queue:
1. **ensemble**: round 11 (4v, `pace/v2_ensemble.sh`). Averaging three seeds beat one seed by
   0.5-0.7 MUSTER on real ASAP with the PM2S meters (the pilot's A); this is the same for the
   release's Beat This! meter, which the pilot did not test.
2. **chooser**: the duration chooser (`src/dur_chooser.py`): each note's written length picked among
   the labeller's top 4 from the notes placed around it (`scripts/train_dur_chooser.py`, trained on
   the validation windows, measured on the test windows). The headroom test found the right
   length in the top 4 for 95% of notes, and a perfect pick would raise rendered duration accuracy
   from 0.87 to 0.97. Whole pieces with it (`_dc`) and with the new defaults: one meter per piece
   (`meter_v4_gq_om6`), swing (`_sw`), and d80, MIDI input's setting.
3. **seq2seq**: the seq2seq recipe (`sf_base_reg_cab_c128`) retrained on the rebuilt data with 30% of
   its windows from Gangqinpu (`train_model.py --gangqinpu-share 0.3`, run `sf_c128_gq30`; Gangqinpu
   data was v2's biggest single gain, -1.93 MUSTER on pop). Up to 3 x 8 h jobs, each resuming the
   last; whole pieces on ASAP and Gangqinpu.
4. **pm2s**: round 9 (`pace/v2_pm2s2.sh`), PM2S retrained with stable settings (the pilot's C).

```bash
git pull && bash pace/v2_round12.sh                       # all four stages, ~27 jobs, 2-3 days
STAGES="chooser seq2seq" bash pace/v2_round12.sh          # only some, in this order
```

Reports: `outputs/round11/report.md` (ensembles), `outputs/round12/report.md` (the chooser on the
windows and on whole pieces, the new defaults, seq2seq with Gangqinpu against seq2seq before and
against v2), and round 9's `outputs/compare` and `paired.md`. If `sf_c128_gq30` is not finished after
its three jobs, `STAGES=seq2seq bash pace/v2_round12.sh` resumes it. Nothing is published (`PUBLISH=0`).

## 4v. Eleventh round: meter ensembles (any time after 4q)

The grid is v2's bottleneck: the labeller's window onset F1 is 0.78-0.86 on the true grid and
0.26-0.49 on the meter models' grids. And meter models of one recipe differ a lot from seed to seed
(time-signature accuracy up to 0.09 apart; ~0.9 MUSTER on whole pieces), more than most changes
measured, while a larger model overfits. This round averages several meter models' probabilities
before the decoder (`src/meter.py` MeterEnsemble, `scripts/make_ensemble.py`), each ensemble with its
own decoder tuned on validation, and measures them on whole pieces against the release's
`nl_gq` + `meter_v4_gq`. Nothing is published (`PUBLISH=0`).

**Before you start:** `git fetch && git checkout michael && git pull` (this round's code is on `michael`), and the pilot (4w) first;
room for 7 jobs in the queue. The script lists any earlier round's file it needs and stops if one is
missing. `nl_gq` and `meter_v4_gq` come from the model repo (`scripts/install_run.py`) when this
account has not trained them; when this account's `meter_v4_gq` is a different training of the same
recipe, the published one joins as a fourth member (`meter_v4_gq_ens4`).

```bash
bash pace/v2_ensemble.sh
```

| job | what | waits for |
|---|---|---|
| `e1tests` | the test suites (CPU) | - |
| `e1prep` | models from the repo where missing; Gangqinpu's whole test pieces if missing; `meter_v4_gq_tta` (the release's model on 5 views: as is, -3 / +3 semitones, x0.9 / x1.1 time) | `e1tests` |
| `e1train` | `meter_v4_gq_s1`, `meter_v4_gq_s2` (the release's recipe, seeds 1 and 2), about 3 h | `e1tests` |
| `e1ens` | `meter_v4_gq_ens3` (3 seeds), `meter_v4_gq_ens3t` (3 seeds x 5 views), `meter_v4_gq_ens4` (if made) | `e1train`, `e1prep` |
| `e1asap`, `e1gq` | whole pieces with `nl_gq` at the release's settings (`tidy,d95`), every meter model above; and `d80` on the release and on `ens3t` | `e1ens` (finished or not), `e1prep` |
| `e1final` | Beyer & Dai's metrics on both test sets, `paired.md`, `outputs/round11/report.md` | `e1asap`, `e1gq` (finished or not) |

**What to check:** `outputs/round11/report.md`: the meter scores of every seed and ensemble, the
whole pieces paired against the release, and the verdict (better beyond chance on a condition and
worse on none). The single seeds against the release show how far chance alone moves it.

**If something goes wrong:**
- *A seed did not finish in 7.5 h* (`e1ens` and after are cancelled): `bash pace/v2_ensemble.sh` again
  (it resumes from `last.pt`; what finished is kept).
- *Only the ensembles and the evaluation again:* `STAGE=eval bash pace/v2_ensemble.sh`.
- *`e1prep` failed on the download* (a private repo needs `hf auth login` once): log in and run again.

To bring the results back (PACE, then your machine, from the repo folder):

```bash
cd ~/scratch/dl-proj-2026 && tar czf v2_ensemble.tgz outputs/round11 outputs/system_eval/results.md outputs/system_eval/results.json outputs/system_eval/paired.md outputs/system_eval_gangqinpu/results.md outputs/system_eval_gangqinpu/results.json outputs/system_eval_gangqinpu/paired.md outputs/runs/meter_v4_gq*/eval_meter.json outputs/runs/meter_v4_gq*/decode.json outputs/runs/meter_v4_gq_*/config.json outputs/runs/meter_v4_gq_s*/log.csv slurm-e1*.out slurm-met_meter_v4_gq_s*.log 2>/dev/null; ls -la v2_ensemble.tgz
```

```bash
scp <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/v2_ensemble.tgz . && tar xzf v2_ensemble.tgz
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
