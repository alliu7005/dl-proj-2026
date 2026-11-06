# dl-proj-2026: Noise-Robust Piano Score Reconstruction

Audio → transcriber → noisy performance MIDI → seq2seq transformer → score tokens → MusicXML.

Two transcribers are supported (`scripts/transcribe.py --model`): **Kong et al. (2021)**, the one in
the proposal, which all current noise statistics, real-transcription splits and baselines use; and
**Transkun V2** (Yan & Duan, ISMIR 2024), the one PianoCoRe's PERiScoPe subset was made with.

Training data:
- **ASAP**: clean Disklavier MIDI with hand-checked beat annotations, plus synthetic
  transcription noise fitted on real transcription errors (Kong; see [Noise model](#noise-model)),
  plus real transcriptions of ASAP performances whose audio the transcriber never saw.
- **PianoCoRe's PERiScoPe subset**: real Transkun V2 transcriptions aligned to scores, so the
  model sees the inference transcriber's own errors.

## Setup

```bash
conda env create -f environment.yml && conda activate dl-proj
pip install --no-deps transkun==2.0.1      # its ncls dep needs a C compiler on Windows; training-only

# Beyer & Dai's ASAP fork (fixes ties/duplicate notes in 49 scores), pinned for reproducibility
git clone https://github.com/TimFelixBeyer/asap-dataset.git data/raw/asap_bd
git -C data/raw/asap_bd checkout 8cba199e15931975542010a7ea2ff94a6fc9cbee
# original ASAP, only used as a fallback for one score the fork can't parse (Ravel/Pavane)
git clone --depth 1 https://github.com/fosfrancesco/asap-dataset.git data/raw/asap

# PianoCoRe: metadata (~35 MB streamed), then the PERiScoPe subset (~6 GB streamed, ~300 MB kept;
# add --max-row-groups 20 for a quick test download)
python scripts/fetch_pianocore_meta.py
python scripts/fetch_pianocore.py
```

## Quick start

| step | command | time |
|---|---|---|
| 1. build train/val/test/test_periscope segments | `python scripts/build_dataset.py` | ~3 min ASAP; PianoCoRe (if downloaded) ~1–2 h with 6 workers (raise `workers` on a bigger machine) |
| 2. check the data: leakage, pairing, arrays, loaders; then tokens → MusicXML → tokens | `python -m tests.test_integrity` and `python -m tests.test_roundtrip --split val` | ~3 min |
| 2b. check window cutting (stitching, prompts) | `python -m tests.test_windows --n 40` | ~3 min |
| 3. *(already done for Kong: `configs/noise_stats.json`)* fit the noise model on real transcriptions | see [Noise model](#noise-model) | ~30–60 min with a GPU |
| 4. *(needs transcriptions)* build the real-transcription ASAP splits | `python scripts/build_transcribed.py --asap-split train --maestro-splits test,validation` (same with `val`), then `--asap-split test` | ~2 min |
| 5. baselines | `python -m scripts.train_xgb_quantizer` then `python scripts/run_baselines.py --methods beat_snap,xgb_quantizer` | ~15 min |

`data/` is git-ignored. The fitted noise model `configs/noise_stats.json` is committed, so step 3 is only needed to refit it.

## Model inputs and outputs

One training example = one **window** of ~48–192 performed notes (see [Windows](#windows-no-bar-lines-needed-at-inference))
→ the score for exactly those notes, as tokens. `src.dataset.collate` returns a batch dict:

| key | dtype, shape | content |
|---|---|---|
| `notes` | float32 `[B, N, 8]` | one row per performed note, sorted by onset, zero-padded (features below) |
| `note_mask` | bool `[B, N]` | True = real note, False = padding |
| `target` | int64 `[B, T]` | `<bos> [prompt <sep>] target tokens <eos>`, padded with 0 (`<pad>`) |
| `loss_mask` | bool `[B, T]` | True on the tokens to learn: the target tokens and `<eos>`. Not on `<bos>`, the prompt, `<sep>` or padding |

Limits: N ≤ 768 notes (val median ~124), T ≤ 1536 tokens (val median ~520, max ~950).

### Encoder input: 8 features per note

| col | feature | values | how to feed it |
|---|---|---|---|
| 0 | pitch | MIDI 21–108 (integer stored as float) | embedding: `pitch.long() - 21` → 88 classes |
| 1 | onset | seconds from window start; < 0 = carried over | standardized (train mean/std); use for a time (sinusoidal) position encoding |
| 2 | inter-onset gap | seconds since previous note | standardized |
| 3 | log duration | log(seconds) | standardized |
| 4 | velocity | /127 | standardized |
| 5 | pedal down at onset | 0/1 | as is |
| 6 | pedal down at release | 0/1 | as is |
| 7 | carried over | 0/1: struck before the window, still held into it | as is |

Columns 1–4 are standardized with `data/processed/feature_stats.json`, written by
`build_dataset.py` and loaded automatically by the Dataset. Pedal timing is only visible through
columns 5–6. The raw pedal intervals are in `pedal.npy` if the model wants them as separate events.

### Decoder output: score tokens

- **Vocabulary:** `data/processed/vocab.json`, 621 tokens. Fixed ids: `<pad>`=0, `<bos>`=1,
  `<eos>`=2, `<unk>`=3, `<sep>`=4. Grammar: [Target tokens](#target-tokens).
- **Training (teacher forcing):**
  - decoder input `target[:, :-1]`, labels `target[:, 1:]`;
  - cross-entropy only where `loss_mask[:, 1:]`;
  - per-token weights from `vocab.loss_weights(ornament_weight=5.0)` (rare ornament and pedal
    tokens up-weighted).
- **Inference, per window:**
  - start the decoder with `<bos> prompt <sep>`, where prompt =
    `segment.prompt_from_output(all tokens decoded so far)` (empty for the first window → just `<bos>`);
  - decode until `<eos>`.
  - Concatenate the windows' outputs in order and drop a window's opening `ped_on` if the pedal
    was already down.
  - `detokenize.tokens_to_musicxml(tokens, "out.musicxml")` writes the score.

### Splits and sources

| split | contents | inputs | use |
|---|---|---|---|
| `train` | ASAP + PERiScoPe | ASAP: clean + synthetic noise (Dataset adds it); PERiScoPe: real Transkun output | training |
| `val` | ASAP + PERiScoPe | clean / real Transkun | model selection |
| `test` | Beyer & Dai's ASAP test set (14 pieces, 59 performances) | clean Disklavier MIDI | comparison with Beyer & Dai |
| `test_periscope` | 10% of PERiScoPe works, none seen in training | real Transkun output | **main test: realistic inference input** |
| `train_transcribed` / `val_transcribed` | ASAP train/val performances with real transcriptions of MAESTRO audio the transcriber never saw | real transcriptions (Kong) | training with real errors / model selection |
| `test_transcribed` | ASAP test with real transcriptions of MAESTRO audio | real transcriptions (Kong) | small real-audio check; most of this audio was in the transcriber's training data |

Every item's `meta.jsonl` line has `dataset` (`asap`/`pianocore`) and `source` (`clean` /
`kong_real` / `transkun_real`); the Dataset adds synthetic noise only to `source == "clean"` items.

### Loading

```python
from torch.utils.data import DataLoader
from src.dataset import AugmentConfig, MixedTrainDataset, ScoreSegmentDataset, collate
from src.preprocessing.noise import NoiseModel

V = "data/processed/vocab.json"
# synthetic noise (+ augmentation) on clean ASAP windows, 30% of samples from REAL transcriptions;
# PERiScoPe windows in `train` are real Transkun output and are never noised (source != "clean")
train = MixedTrainDataset("data/processed/train", "data/processed/train_transcribed", V,
                          noise=NoiseModel.from_json("configs/noise_stats.json"),
                          augment=AugmentConfig(), real_fraction=0.3)
dl = DataLoader(train, batch_size=32, sampler=train.sampler(), collate_fn=collate, num_workers=4)
for epoch in range(E):
    train.set_epoch(epoch)                   # fresh (reproducible) noise every epoch
    for b in dl: ...                         # b["notes"], b["note_mask"], b["target"], b["loss_mask"]

loss_w = train.vocab.loss_weights(ornament_weight=5.0)    # -> nn.CrossEntropyLoss(weight=..., reduction="none")
val_clean = ScoreSegmentDataset("data/processed/val", V)               # no noise, no prompt dropout
val_real = ScoreSegmentDataset("data/processed/val_transcribed", V)    # real transcriptions
test_real = ScoreSegmentDataset("data/processed/test_transcribed", V)
test_periscope = ScoreSegmentDataset("data/processed/test_periscope", V)  # if PianoCoRe was built
```

- **Training mode:** passing `noise=` or `augment=` turns on synthetic noise for clean items and
  10% prompt dropout. Leave both out for val/test.
- **Synthetic-only training** (e.g. the Beyer & Dai-style clean baseline):
  `ScoreSegmentDataset("data/processed/train", V, noise=..., augment=AugmentConfig())`, or pass
  `noise=None` for clean inputs.
- **Oversampling:** `sampler()` draws windows with grace notes, arpeggios or ornaments 3× as often.
- **Augmentation** (`AugmentConfig`, training only): applied to the clean window before noise.
  - Transposition by −6…+6 semitones: input pitches and the target's and prompt's `p_*` tokens
    shift together, and `key_*` is re-spelled. Only shifts that keep every note on the keyboard
    are used.
  - Tempo stretch ×0.85–1.18 (the carried-note threshold `carry_min_sec` is stretched with it).
  - Velocity scale ×0.8–1.2.
- **Model selection:** validation is composer-balanced; report the macro-average over composers
  too. `val_real` is the best predictor of performance on real audio. Its audio was never used to
  fit the noise model.

## On-disk format

`data/processed/` contains:

```
vocab.json                 621 target tokens incl. <sep> (+ training counts)
feature_stats.json         mean/std of the continuous note features (train)
splits.json                {"asap": {folder: split}, "pianocore": {work: split}}
report.json                per-performance stats, skips, and dropped windows
train/ val/ test/ test_periscope/   one directory per split, each holding:
  notes.npy   float32 [total_notes, 4]   pitch, onset (s, rel. to window start; < 0 = carried over), duration (s), velocity
  pedal.npy   float32 [total_pedal, 2]   sustain pedal (on, off) in seconds, rel. to window start
  tokens.npy  int16   [total_tokens]     prompt + target token ids (no <bos>/<sep>/<eos>)
  index.npy   int64   [n_windows, 7]     note_lo, note_hi, pedal_lo, pedal_hi, tok_lo, tok_hi, tgt_lo
  meta.jsonl  one line per window        dataset, source, piece, performance, win, s_lo, s_hi, start_pos, ...
train_transcribed/  val_transcribed/  test_transcribed/
                           same layout; inputs are REAL transcriptions (Kong by default) of the same
                           ASAP performances, cut at the clean windows' exact cut points; targets
                           and prompts copied from the matching clean window
```

Window `i`'s notes are `notes[index[i,0]:index[i,1]]`, and likewise for pedal. Its prompt is
`tokens[tok_lo:tgt_lo]` and its target `tokens[tgt_lo:tok_hi]` (`SegmentStore.get` / `get_prompt`).
The arrays are memory-mapped, so nothing loads into RAM until it's used. Read through the Dataset,
not the raw arrays: it also drops released carried-over candidates (below) and scales the features.

Current build (ASAP only, PianoCoRe not downloaded; `report.json` has the exact counts):

| split | windows | performances | compositions |
|---|---|---|---|
| train | 36,172 (48–192 notes, 50% overlap) | 743 | 153 |
| val | 2,093 (~128 notes, no overlap) | 102 | 29 |
| test | 1,725 (~128 notes, no overlap) | 59 | 14 |
| train_transcribed | 3,841 | 87 (all audio unseen by the transcriber) | |
| val_transcribed | 394 | 20 (all audio unseen by the transcriber) | |
| test_transcribed | 611 | 24 (21 seen by the transcriber, 3 unseen) | 11 |

Real transcriptions cover 99–99.5% of their performances' clean windows (a few windows end up
empty or over the length cap). They can contain more notes per window than clean MIDI: p99 251,
max ~1,530, vs. a clean max of 761. Cap or truncate the input length in the model.

Expected PERiScoPe additions with the full download (estimates from the download selection):

| split | PERiScoPe |
|---|---|
| train | ~85% of 1,200 works, ~5.7k performances → roughly 250k windows |
| val | ~5% of works |
| test | none (Beyer & Dai test works are excluded from PERiScoPe) |
| test_periscope | ~10% of works, ~670 performances |

The per-performance alignment checks (`tests/test_windows.py --source pianocore`) passed on all 25
sampled PERiScoPe performances, with a median window input-vs-target pitch F1 of 0.975.

### Note details

- **Carried-over notes:** notes struck before a window but still held at least 1/4 quarter into it
  are included with a negative onset, so a target note tied over the window's first bar line
  (`... tie`) has input evidence. Pedal presses that started before the window likewise keep a
  negative start. The noise model never moves a note or pedal across t = 0.
  - `notes.npy` also stores *pedal-sustained* candidates (key released, pedal still down). The
    Dataset drops them after noise unless the key is still down at the window's `carry_min_sec`,
    so clean inputs only show key-held notes, while pedal lengthening in the noise model
    produces carried notes the way real transcriptions can. Anything reading `notes.npy`
    directly should apply `src.dataset.drop_released_carried` too.
  - About half of the ties into a window's first beat get a carried note. Most of the rest are
    tied notes the pianist released early and held with the pedal (about 25%), or released early
    with no pedal, so the model still has to infer some ties from the music. Carrying all
    pedal-sustained notes would cover 72% of ties, but 94% of what it adds are not ties.

### Data audit

`python scripts/audit_data.py` prints a data-quality report. Re-run it after changing the pipeline.
It covers:

- token mix and pedal flutter;
- target notes with no input onset (ties);
- target diversity and composer balance per split;
- key coverage and input timing;
- `<unk>` rate, and what the build dropped.

### Windows: no bar lines needed at inference

At inference on raw audio we don't know where the bar lines are, and beat trackers are unreliable
on expressive classical piano (Beat This!, the 2024 state of the art, gets ~66 downbeat F1 on
classical recordings). So segments are **not cut at bar lines** (`segment.mode: windows`; the old
4-bar mode is still available as `mode: bars`):

- **Cut anywhere.** Each performance is cut into windows of ~48–192 notes (train, 50% overlap) or
  ~128 notes (val/test, no overlap), at the largest timing gap near the target size, so chords and
  ornaments are never split.
- **The model places the bar lines.** A window usually starts mid-bar. Its target then opens
  without `bar`, with the time/key signature and a position saying where in the bar it starts:
  `ts_3/4 key_0 pos_1 sub_1/2 p_60 ...` = "starts on beat 2.5 of a 3/4 bar". Every later bar line
  is a `bar` token the model has to predict.
- **Context prompt.** Each window also stores the score just before it (the start of the current
  bar, or the whole previous bar if the window starts on a bar line). The Dataset feeds
  `<bos> prompt <sep> target <eos>`, with the loss only on the target. At inference the prompt is
  `segment.prompt_from_output(tokens decoded so far)`, which `tests/test_windows.py` checks gives
  exactly the training prompts. It carries the meter, position in the bar and open ties from one
  window to the next. 10% of training items (synthetic or real) drop the prompt, so the first window
  works without one.
- **Stitching.** Concatenating consecutive windows' outputs gives one valid token stream. The only
  cleanup needed is dropping a window's opening `ped_on` if the pedal was already down.

### Target tokens

Example of one bar:

```
bar ts_3/4 key_-2  pos_0 ped_on staff_1 p_46 d_2 p_53 d_1  pos_1 p_65 d_1/2 tie  pos_1 sub_1/2 p_67 grace p_65 d_1/2 arp orn_trill ...
```

- **Time signature / key** (`ts_*`, `key_*`): emitted on the first bar of a segment and whenever
  they change.
- **Position** = `pos_<whole quarter>` plus an optional `sub_<fraction>`, emitted when it changes.
- **Staff** (`staff_k`): emitted when it changes; resets to 0 at each bar.
- **Each note** is `p_<midi>` followed by `d_<quarter length>`, or by `grace` for grace notes.
  Optional flags after it: `tie` (continues a tied note), `arp`, `orn_{trill,mordent,turn,tremolo,fermata}`.
- **Pedal** (`ped_on`, `ped_off`): taken from the performance's sustain-pedal data, as the proposal
  specifies.
- **Note order** within a position is by staff, then pitch (grace notes keep their written order).
- **Vocabulary:** a fixed grammar covering every snapped position and duration, so val/test never produce `<unk>`. Rare durations seen in training are added on top.

Positions and durations are exact fractions, snapped to common denominators (2, 3, 4, …, 16, 24,
32, 48).

To go back from tokens to MusicXML (for MUSTER/MV2H evaluation):
`src.preprocessing.detokenize.tokens_to_musicxml(tokens, "out.musicxml")`.

## Pipeline details

1. **Split by composition, before segmenting.**
   - **Test** is Beyer & Dai's exact 14 pieces / 59 performances (`configs/splits/beyer_dai.json`,
     regenerated by `scripts/make_bd_split.py`). We also exclude their list of 69 badly aligned
     performances.
   - **Validation is composer-stratified** (`val_strategy: composer_stratified`). Beyer & Dai's val
     (`piece_id % 10 == 0`) was 77% Beethoven. Ours takes about 10% of each composer's
     performances, choosing whole compositions and skipping any piece so large it would drain that
     composer from train. The result mirrors train's composer mix (Beethoven 31% val / 36% train).
   - The test set is identical either way. `val_strategy: fixed` restores Beyer & Dai's val.
   - Repeat variants (`X` and `X_no_repeat`) always share one composition ID.
2. **Parse the score** (`src/preprocessing/score.py`, music21). Each measure index gets its notes
   with staff, duration, grace, tie, arpeggio and ornaments.
3. **Align bars.** ASAP's `downbeats_score_map` maps every performed bar to MusicXML measure
   indices, which handles repeats (`'8-0'` = a bar split across a repeat) and pickup bars.
   Performance onsets are mapped into score time by piecewise-linear interpolation over the beat
   annotations.
4. **Cut into note windows** (see [Windows](#windows-no-bar-lines-needed-at-inference)). Each cut
   between two performed notes is mapped to score time (midpoint of their mapped onsets) and from
   there to a bar and a position, which decides exactly which score notes the target contains.
   The older 4-bar mode (`segment.mode: bars`) cuts at annotated bar lines instead.
5. **Quality filters.** These dropped 201 of about 40k windows:
   - performed pitches vs. score pitches below 0.6 F1 (local beat-annotation errors);
   - bars longer than 16 quarters (unbarred cadenzas);
   - segments too long for the model.
6. **Other skipped data:**
   - performances with no downbeat map (50);
   - performances ASAP marks as not aligned (31);
   - Liszt's Mephisto Waltz (13 performances), whose score annotations have 3 more beats than every
     performance, so beats can't be paired.

**Alignment sanity checks:**
- Performed vs. score pitch content per segment has a median F1 of 0.97.
- Segments whose note counts differ by more than 20% are mostly trills and tremolos, where one
  written note is many played notes.
- Tokens → MusicXML → tokens keeps 99.6% of notes on val windows (`tests/test_roundtrip.py`, 0 failures).
- Stitching all windows of a performance reproduces the whole piece exactly, every performed note is
  in exactly one window, and every prompt matches `prompt_from_output` (`tests/test_windows.py`,
  40 performances, 914 windows).

## Noise model

`src/preprocessing/noise.py` injects transcription errors on the fly during training:

- missing notes, more likely in dense chords;
- hallucinated notes;
- onset jitter;
- duration noise and sustain-pedal lengthening;
- velocity noise;
- pedal timing errors and missed pedal presses.

10% of segments are left clean.

Everything is **fitted on real Kong transcriptions** of 95 recordings (425k notes) from the
MAESTRO test and validation splits.
- Kong's model was trained on MAESTRO train, so only test/validation audio gives realistic errors.
- Audio of ASAP **val and test** performances is excluded from the fit, so `val_transcribed` and
  `test_transcribed` remain independent checks.

The model is **empirical**: `configs/noise_stats.json` holds three fitted pieces, plus a few scalar
parameters for timing, velocity and pedal.

- **Missing notes:** a miss-rate table indexed by chord size × velocity bin.
- **Hallucination rate:** hallucinations spawned per real note, by the source note's velocity.
- **Error bank:** 7,739 real hallucinations, each stored relative to its most likely source. The
  source is a recent, loud note that is still sounding, counting the sustain pedal. Each entry
  records interval, delay, velocity ratio and duration.

During training, loud notes spawn hallucinations by replaying sampled bank entries.

| error | measured |
|---|---|
| missing notes | mostly a function of loudness. Lone notes: 29% if quiet (velocity < 40), 0.5% if loud. 8-note chords: 57% for quiet notes, 6% for loud ones |
| hallucinated notes | 1.8% of notes. Loud notes spawn them 5.5× more often than quiet ones. Most common: an octave below or above the source (−12 / +12), then thirds and sixths, then two octaves |
| onset jitter | 6.1 ms std |
| notes lengthened to pedal release | 68% of notes released under sustain |
| missed pedal presses | 8%, with ~75 ms timing error |

`configs/noise_stats_parametric.json` is the first, hand-parametrized version. Keep it for ablations.

**Validation with a detector** (`scripts/train_xgb_detector.py`, see [XGBoost baseline](#xgboost-baseline-note-level-error-detector)).
An error detector trained only on synthetic noise and tested on **real** transcriptions of unseen
audio scores:

| noise model | AUC | AP |
|---|---|---|
| parametric | 0.660 | 0.010 |
| empirical | 0.904 | 0.090 |
| trained on real errors (upper bound) | 0.987 | 0.435 |

Random guessing gives AP 0.005. The empirical noise is much closer to real errors, but a gap
remains. On 8-second windows, synthetic noise gives pitch F1 0.964 vs. 0.974 for real
transcriptions, i.e. slightly harsher than real.

To reproduce the fit:

```bash
uv venv --python 3.12 .venv-transcribe       # Kong's package needs Python <= 3.12 + CUDA torch
uv pip install --python .venv-transcribe/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-transcribe/Scripts/python.exe piano_transcription_inference audioread librosa pretty_midi
# checkpoint (the package's own download uses wget, which Windows lacks):
curl -L -o ~/piano_transcription_inference_data/"note_F1=0.9677_pedal_F1=0.9186.pth" \
  "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"

# range-downloads single recordings from the 108 GB zip (~10 GB total for all three)
python scripts/fetch_maestro_audio.py --split test --limit 40
python scripts/fetch_maestro_audio.py --asap-split train,val --maestro-splits test,validation
python scripts/fetch_maestro_audio.py --asap-split test
.venv-transcribe/Scripts/python scripts/transcribe.py            # ~40x real time on an RTX 5070 Ti
python scripts/fit_noise.py --pairs data/transcribed/pairs.csv --out configs/noise_stats.json
```

### Transkun V2 errors (measured by Alex, not yet committed)

Transkun V2, the transcriber PERiScoPe was made with, makes very different errors from Kong, as
measured on 40 MAESTRO-test recordings (170k notes):

| error | Transkun V2 | Kong 2021 |
|---|---|---|
| missing notes | 3.7% for single notes → 15% in 8-note chords | 4.5% → 20% |
| extra (hallucinated) notes | 0.23% of notes, mostly ±1–4 semitones | 2.3% (10× more) |
| onset jitter | 5.9 ms std | 6.5 ms |
| notes lengthened to pedal release | 12% of notes released under sustain | 65% |
| missed pedal presses | 12%, ~80 ms timing error | 9% |

So the synthetic noise must match the transcriber used at inference. The committed
`configs/noise_stats.json` is **Kong**. To train for Transkun, fit and commit a Transkun model:

```bash
python scripts/transcribe.py --model transkun        # -> data/transcribed_transkun (~5x real time)
python scripts/fit_noise.py --pairs data/transcribed_transkun/pairs.csv --out configs/noise_stats_transkun.json

```

`transcribe.py` gives the same output as the package's `PianoTranscription.transcribe`: about 0.3%
of notes shift by at most ~1 ms, from GPU rounding differences between batch sizes. It's about 20×
faster:
- batched GPU inference (the package hard-codes batch size 1);
- audio decoding prefetched in background threads;
- note/pedal decoding in worker processes.

Other GPU-heavy programs (e.g. games) slow it down noticeably.

## Baselines

The milestone asks for 1–3 baselines evaluated "with the same data, metrics, and protocol" as our
method. All methods are scored by `src/eval/metrics.py` on the same test windows. Each method
outputs tokens in the target grammar for every window, and these are compared with the
ground-truth target. A perfect prediction scores 1.0 on every metric (checked).

```bash
python -m scripts.train_xgb_quantizer          # ~10 min: trains Baseline 2's classifiers
python scripts/run_baselines.py --methods beat_snap,xgb_quantizer   # ~5 min
```

| baseline | category | what it does |
|---|---|---|
| **1. Beat-snap** (`src/baselines/quantize.py`) | classical heuristic | Maps each note to score time through the **annotated (oracle) beats**, snaps onsets to the nearest 16th / 8th-triplet and durations to the nearest common value, and splits staffs at middle C. No grace notes, ties or ornaments. |
| **2. Beat-snap + XGBoost** (`src/baselines/xgb_quantizer.py`) | lightweight supervised | Baseline 1, plus four per-note XGBoost classifiers (keep/delete, staff, written duration, onset shift). Trained on synthetic-noise and real-transcription training windows (1.17M notes). |
| 3. Beyer & Dai (2024), pretrained | prior published | Not run yet: needs their code and checkpoint (external code; see below). |

Conditions (same windows, same targets):
- **clean:** clean MIDI.
- **synthetic:** clean MIDI plus our fitted noise.
- **clean_paired:** clean MIDI on exactly the windows that have a real transcription.
- **real:** Kong transcriptions.
- **real_unseen:** real transcriptions of audio Kong never saw (3 performances, 62 windows).

| metric | B1 clean | B1 real | B2 clean | B2 synthetic | B2 clean_paired | B2 real | B2 real_unseen |
|---|---|---|---|---|---|---|---|
| note F1 | 0.929 | 0.926 | 0.930 | 0.906 | 0.935 | 0.930 | 0.914 |
| onset F1 (rhythm) | 0.631 | 0.646 | 0.714 | 0.688 | 0.732 | 0.732 | 0.791 |
| duration acc | 0.341 | 0.289 | 0.538 | 0.517 | 0.491 | 0.517 | 0.545 |
| staff acc | 0.743 | 0.786 | 0.827 | 0.828 | 0.865 | 0.868 | 0.913 |
| tie / grace F1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

**Reading the table:**
- **Not meaningful for these baselines:** bar F1 (≈1.0) and pedal F1 (1.0 on clean). Both
  baselines are given the true beats, bar lines and pedal, so those metrics are trivially high.
- **What the transformer has to beat:** rhythm (onset F1 ~0.71–0.79), written durations
  (~0.5), staff (~0.83–0.91), and ties/grace notes, which the baselines can't produce at all.
- **Which columns to compare:** "clean" and "real" cover different performances; compare
  `clean_paired` with `real`. The real column is mostly audio the transcriber was trained on,
  hence barely any drop. Synthetic noise costs Baseline 2 about 2.6 points of onset F1.

**Baseline 3 (Beyer & Dai) needs a decision.** Their pretrained checkpoint and their MUSTER /
score-similarity metrics need their code: a patched music21 fork plus the `score_transformer` and
`muster` packages, installed from GitHub into a separate Python 3.11 environment. That's external
code, so it needs explicit approval before it's installed and run.

## XGBoost baseline: note-level error detector

```bash
pip install xgboost scikit-learn
python scripts/train_xgb_detector.py      # ~3 min on GPU; writes outputs/xgb_*.json
```

The model classifies each transcribed note as real vs. error (hallucinated or wrong pitch). It
uses 56 local-context features (`src/baselines/note_features.py`), such as:

- what starts at the same time;
- whether a louder note sits an overtone interval away;
- what is still sounding;
- local key fit and loudness;
- repeated notes.

Two detectors are trained:

- **On synthetic noise** (ASAP train). Labels are the notes the noise injector added.
- **On real Kong errors** (95 MAESTRO test/validation transcriptions, with ASAP val/test
  recordings excluded). Labels come from matching each note against the ground truth within 50 ms.

| detector | synthetic test AUC / AP | real test AUC / AP | real, unseen audio AUC / AP |
|---|---|---|---|
| flag quietest notes | 0.658 / 0.027 | 0.689 / 0.031 | 0.695 / 0.017 |
| trained on synthetic (empirical noise) | 0.961 / 0.589 | 0.540 / 0.030 | 0.904 / 0.090 |
| trained on real errors | 0.818 / 0.062 | 0.620 / 0.117 | 0.987 / 0.435 |

**Takeaways:**

- **What real errors look like.** The strongest features are a louder note an octave below that is
  still sounding, being the lowest note of a chord, and being quiet relative to the chord. So
  hallucinations are mostly overtones.
- **The unseen-audio column is small:** only 3 performances, about 40 error notes.
- **"Real test" is mostly audio Kong was trained on**, where the remaining errors are rare and
  atypical.
- **Deleting flagged notes doesn't help yet** (precision about 0.2). Use the detector score as an
  input feature rather than as a hard filter.

## Real-transcription splits

- **`train_transcribed` / `val_transcribed`:** only audio Kong never trained on (MAESTRO
  test/validation). 87 training and 20 validation performances. Against the score, pitch agreement
  drops from 0.961 for clean inputs to 0.946 for real transcriptions (train; val: 0.967 to 0.955).
- **`test_transcribed`:** see the caveat below.

### Caveat: test_transcribed

Only 25 of the 59 ASAP test performances have MAESTRO audio (segment counts: rebuild with
`build_transcribed_eval.py`):

| MAESTRO split of the audio | performances | segments | meaning |
|---|---|---|---|
| train | 21 | 549 | the transcriber (Kong) was trained on this audio, so transcriptions are optimistic |
| validation | 2 | 42 | unseen by the transcriber |
| test | 1 | 20 | unseen by the transcriber |

A 25th performance with audio (MAESTRO train) is left out because it's excluded from the clean test
split, so there's nothing to pair it with.

Each segment's `meta.jsonl` has a `maestro_split` field, so report the unseen subset separately.

## Repo layout

```
configs/preprocess.yaml          all preprocessing settings
configs/splits/beyer_dai.json    Beyer & Dai split + exclusions
src/preprocessing/
  performance.py   MIDI notes + sustain pedal, beat-grid time maps
  score.py         MusicXML -> measures -> target tokens
  segment.py       bar alignment, windowing, quality filters
  split.py         composition-level splits
  vocab.py         grammar + data vocabulary, loss weights
  noise.py         transcription-error injection
  store.py         memory-mapped on-disk format
  detokenize.py    tokens -> MusicXML
src/baselines/note_features.py   per-note context features (XGBoost detector / quantizer)
src/baselines/quantize.py        Baseline 1: beat-snap quantizer
src/baselines/xgb_quantizer.py   Baseline 2: beat-snap + per-note XGBoost
src/eval/metrics.py              shared window-level metrics
src/dataset.py     PyTorch Dataset / sampler / collate
scripts/           build_dataset, make_bd_split, fetch_maestro_audio, transcribe, fit_noise,
                   build_transcribed, train_xgb_detector, audit_data
tests/test_integrity.py   leakage, pairing, array and loader checks
tests/test_roundtrip.py   tokens -> MusicXML -> tokens
```

## PianoCoRe (PERiScoPe subset)

[PianoCoRe](https://github.com/ilya16/PianoCoRe) ([paper](https://doi.org/10.5334/tismir.333))
combines several piano datasets. Its PERiScoPe performances were transcribed from audio with
Transkun V2, the same transcriber used here at inference, and aligned note by note to scores.

- **Selection** (`scripts/fetch_pianocore.py`):
  - tier A* (best alignments), PERiScoPe only, no Beyer & Dai test works;
  - 10 random performances per work: 6,714 performances, 1,200 works (1,003 not in ASAP), 481 h.
  - Only the needed parquet columns are streamed. ASAP's own performances are not taken from
    PianoCoRe (ASAP is used directly).
- **Inputs are the raw transcriptions**, so real Transkun errors stay in (`source: transkun_real`).
  The Dataset never adds synthetic noise to them.
- **Bars without beat annotations.**
  - PianoCoRe's score MIDI plays repeats and puts pickups at the end of a full bar, so the
    MusicXML is unfolded (music21 `expandRepeats`) and a constant offset is fitted
    (`score_midi_shift`). On 28 sampled scores the median onset+pitch agreement was 0.989.
    Scores below 0.9 are skipped.
  - The refined alignment then gives (performance time, score quarter) pairs per note, which
    `segment.timeline_from_alignment` turns into the same `Timeline` that ASAP's beat annotations
    produce.
- **Clock offset.** PianoCoRe's refined performance is a trimmed copy of the raw transcription
  (median 0.5 s offset, up to 6 s), so the raw notes are shifted by `estimate_offset`. Notes more
  than 1 s outside the aligned span (lead-in/out the alignment trimmed) are dropped.
- **Splits.**
  - Works also in ASAP inherit ASAP's split.
  - Other works are hashed 85/5/10 into train / val / `test_periscope`.
  - `test` stays Beyer & Dai's ASAP test set.
- **Known label noise:**
  - pedal targets for these rows come from Transkun's transcribed pedal;
  - bar positions come from automatic alignment, not hand annotation.
