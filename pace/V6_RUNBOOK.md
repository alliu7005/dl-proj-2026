# Runbook: v6 data, our Beat This!, the models retrained

Exactly what to run on PACE, in order. Each step: the command, how long it takes, and how to know it
is done before starting the next. The steps build on each other (the models must not train on old
data or an old tracker's beats); the scripts refuse to start a step whose inputs are missing, but
check anyway. Background and the job tables: [README.md](README.md), sections "Run order", 4y, 4z.

All commands from the repo folder on the login node:

```bash
ssh <gtid>@login-ice.pace.gatech.edu        # on the GT VPN
cd ~/scratch/dl-proj-2026
```

**Between steps:** the jobs of the steps a step needs must have finished (gone from `squeue -u $USER`),
then run their "done when" checks. Some run side by side (PACE allows 50 queued jobs): the base Beat
This! with step 1, 3a with 3b, step 4 while 3a still trains. One after another the steps take 7-8
days; side by side about 5. A job's log is `slurm-<job name>-<job id>.out` in the repo folder.

---

## Step 0: once, before anything

**On your own machine**, push the code (the work is on `main`):

```bash
git push origin main
```

**On PACE:**

```bash
git fetch origin && git checkout main && git pull
module load anaconda3 && conda activate dl-proj
pip install yt-dlp pytorch-lightning
conda install -y -c conda-forge fluidsynth
fluidsynth --version | head -1 && python -c "import yt_dlp, pytorch_lightning; print('ok')"
pace-quota                                  # scratch must have ~200 GB free (see Space below)
mkdir -p data/raw/soundfonts
```

**SoundFonts:** copy at least one piano SoundFont into `data/raw/soundfonts/`, from your machine. For
example MuseScore 4's (Windows: `C:\Program Files\MuseScore 4\sound\MS Basic.sf3`):

```bash
scp "MS Basic.sf3" <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/data/raw/soundfonts/
```

More pianos (`.sf2` / `.sf3`, e.g. Salamander Grand Piano, YDP Grand) can go there too: each
performance gets one of them. Check each license allows making training data.

**Space:** ~22 GB ATEPP / GiantMIDI-Piano recordings, ~50 GB Gangqinpu recordings, ~108 GB Beat
This!'s data (`EXCLUDE="asap,harmonix"` in steps 3b-5 makes it ~63 GB), ~25 GB our Beat This! data,
plus the old `data/processed` kept as `data/processed_pre_v6`.

---

## Step 1: fetch the new data (7 jobs, about a day: the YouTube downloads are slow on purpose), and the base Beat This! (3 jobs)

```bash
bash pace/v2_data.sh
STAGE=base bash pace/v2_btft.sh            # Beat This! from scratch without ASAP: needs no v6 data
```

The base model's fetch can fail on a compute node without internet (see step 3b): then fetch on the
login node and submit `FETCHED=1 STAGE=base bash pace/v2_btft.sh`.

**If YouTube asks to confirm you are not a bot** (`grep -l "not a bot" slurm-dyt*.out` finds it), export
a `cookies.txt` from a browser logged in to YouTube (a "cookies.txt" browser extension), copy it to
the repo folder, and fetch the rest on the login node (in `tmux`, so it survives a disconnect):

```bash
tmux new -s yt
module load anaconda3 && conda activate dl-proj
python scripts/fetch_pianocore_audio.py --dir data/raw/pianocore/atepp --cookies cookies.txt
# Ctrl-b d to leave it running; tmux attach -t yt to come back
```

**Done when** (`squeue -u $USER` empty, then):

```bash
ls data/raw/pianocore/{periscope_more,atepp,aria}/selection.parquet          # all three exist
python -c "import json, collections; r = json.load(open('data/raw/pianocore/atepp/audio/fetch_report.json')); print(collections.Counter(v.split(':')[0] for v in r.values()))"
ls data/raw/gangqinpu/audio/train | wc -l                                     # ~17,000 (3,000 before)
```

A few hundred ATEPP recordings `unavailable` or `error` is normal. If most are `error`, run the
fetch again (it resumes; `--retry-failed` retries the errors).

---

## Step 2: build the v6 data (4 jobs, ~1 day: re-transcription and the rebuild)

```bash
STAGE=build bash pace/v2_data.sh
```

**Done when:**

```bash
ls -d data/processed_pre_v6                                                   # the old data, kept
python -c "import json; print(json.load(open('data/raw/pianocore/atepp_tk/retranscribe_report.json'))['counts'])"
tail -3 slurm-drebuild-*.out                                                  # sizes, no Traceback
ls data/processed/meter/train*.pkl data/processed/meter/tracked_train_gangqinpu_beat_this.npz
```

If `drebuild` failed: read its log, fix, then move the half-built folder away and restore the old
one before running the stage again (`mv data/processed data/processed_failed && mv data/processed_pre_v6 data/processed`).

---

## Step 3a: the v6 models on the released Beat This! (8 jobs, 1-2 days)

```bash
STAGE=train bash pace/v2_data.sh
```

**Done when:** `nl_v6` (the labeller every later step uses) and the meter models finished:

```bash
for r in nl_v6 nl_v6_large nl_v6_noaria meter_v6 meter_v6_s1 meter_v6_ne meter_v6_noaria meter_v6_large meter_v6_large_s1; do
  printf "%-20s " $r; grep -o '"phase": *"[a-z]*"' outputs/runs/$r/status.json 2>/dev/null || echo missing; done
```

Every line should say `"phase": "done"`. One that does not after its jobs: `STAGE=train bash pace/v2_data.sh`
again resumes it (finished runs are skipped).

---

## Step 3b: our Beat This!, its data (1 job, a few hours; with 3a, right after it)

```bash
bash pace/v2_btft.sh                       # the export; the base model from step 1 is kept (finished or queued)
```

If step 1 did not start the base model, this submits it too (4 jobs, ~1 day; `FETCHED=1 bash
pace/v2_btft.sh` if you fetched by hand, below).

If compute nodes cannot reach Zenodo (`slurm-btfetch-*.out` shows connection errors), fetch on the
login node in `tmux`, then submit with `FETCHED=1`:

```bash
python scripts/fetch_beat_this_data.py --exclude asap
```

**Done when:**

```bash
ls outputs/beat_this/bt_noasap/final.ckpt data/beat_this_ft/pieces.json
python -c "import json; r = json.load(open('data/beat_this_ft/export_report.json')); print(json.dumps(r['mix'], indent=1)); print([k for k, v in r['folds'].items() if 0 in v.values()])"
```

The mix should show all four cells (classical / pop x real / render); the last line should print `[]`
(every fold model trains on every cell). If `bt_noasap` has no `final.ckpt` after both base jobs,
resume it: `sbatch --gres=gpu:1 --constraint=cuda13 -c 16 --mem=96G -t 8:00:00 --wrap="source pace/common.sh; python scripts/train_beat_this.py --name bt_noasap --data data/beat_this_orig --exclude asap --no-val --num-workers 14 --max-hours 7.6"`.

---

## Step 4: the ten fine-tunes (8 jobs, ~1 day; as soon as 3b and the base model are done, 3a may still run)

```bash
STAGE=finetune bash pace/v2_btft.sh
```

**Done when** all ten have `final.ckpt`:

```bash
ls outputs/beat_this/bt_ft{,m}_{full,fold0,fold1,fold2,fold3}/final.ckpt | wc -l      # 10
```

Fewer than 10: `STAGE=finetune bash pace/v2_btft.sh` again (finished ones are skipped, the others resume).

---

## Step 5: beats, meter models, whole pieces (8 jobs, ~2 days; after step 4 and 3a's `nl_v6`)

```bash
STAGE=use bash pace/v2_btft.sh
```

**Done when** `squeue -u $USER` is empty and these exist:

```bash
cat outputs/render_check_{released,ft,ftm}/render_check.md      # renders vs recordings per tracker
for r in meter_v6_ft meter_v6_ftm; do printf "%-14s " $r; grep -o '"phase": *"[a-z]*"' outputs/runs/$r/status.json; done   # "done"
grep -E "btft|btftm|nl_v6" outputs/system_eval/paired.md | head -20
```

**Read the render check first:** for `ft` and `ftm`, the renders' beat F, downbeat F and wrong-octave
count should be close to the recordings' (as for `released`). If renders track clearly better than
recordings after fine-tuning, the meter models trained on those beats (`meter_v6_ft*`) learned from
beats cleaner than test-time ones: report that with their results.

A meter run not `"done"`: `STAGE=use bash pace/v2_btft.sh` again only after checking which jobs failed
(it resubmits the whole stage; finished work is skipped or resumed).

---

## Bring the results back

On PACE:

```bash
cd ~/scratch/dl-proj-2026 && tar czf v6_results.tgz data/raw/pianocore/*/audio/fetch_report.json data/raw/pianocore/*/retranscribe_report.json data/processed/report.json data/processed/meter/*report*.json data/beat_this_ft/export_report.json data/beat_this_ft/pieces.json outputs/render_check* outputs/beat_this/*/status.json outputs/beat_this/*/metrics.csv outputs/beat_this/bt_ft*/models.json outputs/beat_this/bt_ft*/eval_beats_*.json outputs/runs/*_v6*/config.json outputs/runs/*_v6*/log.csv outputs/runs/*_v6*/eval_*.json outputs/system_eval/results.* outputs/system_eval/paired.md outputs/system_eval_gangqinpu/results.* outputs/system_eval_gangqinpu/paired.md slurm-d*.out slurm-m*.out slurm-bt*.out slurm-ft*.out 2>/dev/null; ls -la v6_results.tgz
```

On your machine:

```bash
scp <gtid>@login-ice.pace.gatech.edu:~/scratch/dl-proj-2026/v6_results.tgz . && tar xzf v6_results.tgz
```

## What to compare

In `outputs/system_eval/paired.md` (ASAP) and `outputs/system_eval_gangqinpu/paired.md` (Gangqinpu):

| question | systems |
|---|---|
| does the v6 data help? | `nl_gq_mc + meter_v4_gq` (the release) vs `nl_v6 + meter_v6` |
| do larger models help now? | `meter_v6` vs `meter_v6_large` (two seeds each), `nl_v6` vs `nl_v6_large` |
| what does Aria-MIDI bring? | `nl_v6 + meter_v6` vs `nl_v6_noaria + meter_v6_noaria` |
| does our Beat This! help? | `nl_v6 + meter_v6` vs `nl_v6 + meter_v6_ft` (`_btft`) |
| does the multi-task version help? | `nl_v6 + meter_v6_ft` (`_btft`) vs `nl_v6 + meter_v6_ftm` (`_btftm`) |
| the tracker alone | `nl_gq_mc + meter_v4_gq` on the released, `_btft` and `_btftm` beats |

The meter model's seed spread is about 0.9 points on whole pieces: a difference smaller than that
between single runs is noise.
