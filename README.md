# dl-proj-2026: Noise-Robust Piano Score Reconstruction

A piano recording in, a written score (MusicXML) out. **Transkun V2** transcribes the audio into
performance MIDI, with the timing, wrong and missing notes of a real transcription. Then one of two
systems writes the score:

- **v2** decides the metrical grid of the whole piece first: Beat This! tracks the beats, and a
  meter model turns them into bars and time signatures. Then a note labeller places every
  performed note in that grid: position, written length, staff, grace and ornament flags.
- **seq2seq**, the original plan: a transformer (the Scoreformer) writes score tokens window by
  window, 128 notes at a time.

Both are compared with Beyer & Dai's system (ISMIR 2024), on their test set and with their
metrics, on clean MIDI and on real recordings.

**Status (2026-11-24, after round 4):** the full system runs end to end in a web demo, and both
systems' models are on Hugging Face as [dat-dev/scoreformer](https://huggingface.co/dat-dev/scoreformer)
([model card](docs/model_card.md)). On the headline metric the seq2seq system is still ahead of v2, and
v2's remaining gap is its rhythm. [docs/handoff.md](docs/handoff.md) covers what works, what
doesn't, and the options for what to do next.

## Results

Whole pieces, Beyer & Dai's metrics: error rates in %, lower is better.
- "real": the 24 test performances transcribed from their audio by Transkun.
- "clean": the 59 test performances as clean MIDI.
- MUSTER mean averages the pitch, missing, extra, onset and offset errors.
- Spelling and insertion are score-similarity errors per reference note.
- Full tables: `outputs/system_eval/results.md`. Differences with intervals:
  `scripts/compare_systems.py`.

| system | MUSTER mean real / clean | onset | missing | extra | pitch | spelling | insertion |
|---|---|---|---|---|---|---|---|
| Beyer & Dai (their checkpoint) | 11.14 / **11.05** | 13.60 | **7.10** | **7.75** | 2.38 | **2.41** | **8.62** |
| seq2seq `sf_base_reg_cab_c128` | **10.65** / 11.74 | **9.90** | 7.57 | 13.48 | 3.06 | 10.92 | 15.08 |
| **v2, the release** (`nl_base_gn` + `meter_v3`, tempo-prior decoder, notes kept unless p(delete) > 0.95, lengths tidied) | 12.31 / 14.53 | 13.82 | 8.90 | 11.80 | **2.26** | 6.58 | 11.98 |
| v2 round 3 (`nl_base_gn` + `meter_v3`) | 12.83 / 15.07 | 13.81 | 11.96 | 10.73 | 2.34 | 6.30 | 10.62 |

Onset through insertion are on real audio.

- **On real audio**, seq2seq leads on onsets and offsets. v2 leads on pitch, extra notes,
  insertion and spelling.
- **v2's gap to seq2seq is outside chance:** 1.66 points on real audio and 2.79 on clean. That
  is on classical piano (ASAP). Outside it, v2 holds up better. On an anime piano arrangement in
  4/4 throughout, seq2seq switched time signature 6 times and wrote 37 extra bars, while v2 kept
  4/4 ([handoff](docs/handoff.md)).
- **Most of v2's loss is the meter model:** half and double note values, like 3/8 written for 3/4.
  The labeller's onset F1 on PERiScoPe windows is 0.857 with the true grid, 0.697 with an oracle
  grid on the tracked beats, and 0.488 with the meter model's grid. Choosing the level per piece
  afterwards was tried in round 5 (priors, the labeller's confidence, a learned written-tempo
  head): none beats the decoder.
- **On clean MIDI, v2 also loses to its beats.** They come from the MIDI rendered to audio. Using
  the real recordings' beats instead lowers its error by 2.8 points (24 performances).
- **The round-by-round story is in [docs/experiments.md](docs/experiments.md)**, including what
  each change measured.

## Use it

```bash
conda env create -f environment.yml && conda activate dl-proj
pip install --no-deps transkun==2.0.1      # the transcriber (its other dependencies are in environment.yml)
python scripts/check_env.py                # everything loads? (on Windows: names any file Smart App Control blocks)
python scripts/download_models.py          # the released models (dat-dev/scoreformer) -> models/, 300 MB
python demo/app.py                         # then open http://127.0.0.1:7860 and drop in a recording
```

On Windows with Smart App Control on, Windows can refuse some of the packages' libraries ("An
Application Control policy has blocked this file"). The way around it that changes nothing in
Windows' security is to run it in WSL: [docs/demo.md](docs/demo.md#windows-files-blocked-by-application-control).

**The demo** runs everything in one process, on the GPU if there is one, else the CPU: Transkun,
Beat This!, and the system chosen on the page.
- **What the page shows:** the engraved score (OpenSheetMusicDisplay), players for the recording,
  the transcription and the score, and a bar that follows the music. Downloads: MusicXML, the
  score's MIDI, the transcription and the tokens.
- **v2 options:** how sure the labeller must be to leave a note out as a transcription error
  ("only when sure", p > 0.95, is the default).
- **seq2seq options:** the decoding mode (meter, guarded, chained, none, time-signature search).
- **MIDI uploads** skip the transcription.
- **From the command line**, the same pipeline:
  ```bash
  python scripts/audio_to_score.py recording.wav --v2
  ```
  Without `--v2` it runs seq2seq. Either way it writes to `outputs/demo/<name>/`.
- **Details** (staves, grace notes, leaving notes out, the playback bar):
  [docs/demo.md](docs/demo.md).

**The models** (`models/`, from [the model card](docs/model_card.md)):
- `v2_labeller`: the note labeller, 25.6M parameters.
- `v2_meter`: the meter model, 4.8M, with its tuned decoder.
- `seq2seq`: the Scoreformer, 44.9M.

Each one has the vocabulary it was trained with beside it. Without `models/`, the demo and
`audio_to_score.py` use the training runs in `outputs/runs/`.

## Reproduce it

1. **Data** ([docs/data.md](docs/data.md)):
   - ASAP (Beyer & Dai's fork) and PianoCoRe's PERiScoPe subset.
   - Windows of performed notes with their score tokens.
   - Synthetic transcription noise fitted on Transkun's errors.
   - Real Transkun transcriptions of the ASAP performances whose audio Transkun never trained on.
   ```bash
   pip install --no-deps transkun==2.0.1      # its ncls dep needs a C compiler on Windows; training-only
   git clone https://github.com/TimFelixBeyer/asap-dataset.git data/raw/asap_bd
   git -C data/raw/asap_bd checkout 8cba199e15931975542010a7ea2ff94a6fc9cbee
   git clone --depth 1 https://github.com/fosfrancesco/asap-dataset.git data/raw/asap   # one fallback score
   python scripts/fetch_pianocore_meta.py && python scripts/fetch_pianocore.py         # ~6 GB streamed, ~300 MB kept
   python scripts/build_dataset.py                                                     # ~25 min
   ```
2. **Training and evaluation** ran on Georgia Tech's PACE cluster. Each round is one submit
   script ([pace/README.md](pace/README.md)); the experiments are in
   [docs/experiments.md](docs/experiments.md).
   - seq2seq: `scripts/train_model.py`.
   - v2: `scripts/train_meter.py` and `scripts/train_labeller.py`.
   - Window-level tables of every run: `scripts/compare_runs.py`.
   - Whole pieces: `scripts/system_scores.py`, then `scripts/beyer_dai.py score` (Beyer & Dai's
     metrics), then `scripts/compare_systems.py`.
3. **The release:**
   - `bash pace/v2_release.sh` scores the released configuration, then runs
     `scripts/export_models.py`. That builds `models/`, checked by loading it.
   - Uploading it is covered in [docs/handoff.md](docs/handoff.md) ("Models").
4. **Tests:** `python -m tests.<name>`. Each module tests one part, from data integrity to the PACE
   job graphs. The model tests use tiny untrained checkpoints, so they run on a laptop. PACE runs
   them before every batch.

## Docs

| file | what |
|---|---|
| [docs/handoff.md](docs/handoff.md) | where things stand and the options for next steps: start here |
| [docs/experiments.md](docs/experiments.md) | every round: what it tried, the numbers, what they mean |
| [docs/data.md](docs/data.md) | data sources, preprocessing, model inputs and outputs, noise model, transcriptions |
| [docs/demo.md](docs/demo.md) | how the demo and `audio_to_score.py` work |
| [docs/baselines.md](docs/baselines.md) | beat-snap and XGBoost baselines, the note-level error detector |
| [docs/model_card.md](docs/model_card.md) | the released models (the Hugging Face page) |
| [docs/repo_layout.md](docs/repo_layout.md) | what every file does |
| [pace/README.md](pace/README.md) | running on PACE: setup, every batch, GPUs, job limits, bringing results back |

## Repo map

```
src/preprocessing/   scores and performances -> training data (windows, tokens, noise, vocabulary)
src/scoreformer.py   the seq2seq model;  src/decoding.py: whole-piece decoding modes
src/meter.py         v2 meter model + consistent Viterbi decoding;  src/beats.py: beats and tracker errors
src/labeller.py      v2 note labeller;  src/notelabels.py: per-note labels, rendering to tokens
src/pipeline.py      audio / MIDI -> MusicXML (ScorePipeline: seq2seq, V2Pipeline: v2)
src/release.py       the released models and where the demo finds them
demo/                web demo (FastAPI + one page);  scripts/audio_to_score.py: the CLI
scripts/             data building, training, evaluation, comparison, release
pace/                PACE batch scripts (one per round);  tests/: one module per part
docs/                the documentation above
```

## Credits

- **Transkun V2** (Yan & Duan, ISMIR 2024), the transcriber.
- **Beat This!** (Foscarin et al., ISMIR 2024), the beat tracker.
- **ASAP** (Foscarin et al., ISMIR 2020) and Beyer & Dai's corrected fork.
- **PianoCoRe / PERiScoPe**: aligned Transkun transcriptions.
- **Beyer & Dai**, "End-to-end piano performance-MIDI to score conversion with transformers"
  (ISMIR 2024): their model, test split and metrics (MUSTER, score similarity).
- **Kong et al. (2021)**: the second transcriber, for robustness checks.
