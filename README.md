# dl-proj-2026: Noise-Robust Piano Score Reconstruction

Audio → Kong et al. (2021) transcriber → noisy performance MIDI → seq2seq transformer → score tokens → MusicXML.

This branch (`michael`) contains the complete data preprocessing pipeline.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate         # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Beyer & Dai's ASAP fork (fixes ties/duplicate notes in 49 scores), pinned for reproducibility
git clone https://github.com/TimFelixBeyer/asap-dataset.git data/raw/asap_bd
git -C data/raw/asap_bd checkout 8cba199e15931975542010a7ea2ff94a6fc9cbee
# original ASAP, only used as a fallback for one score the fork can't parse (Ravel/Pavane)
git clone --depth 1 https://github.com/fosfrancesco/asap-dataset.git data/raw/asap
```

## Quick start

| step | command | time |
|---|---|---|
| 1. build train/val/test segments | `python scripts/build_dataset.py` | ~3 min |
| 2. check the data: leakage, pairing, arrays, loaders; then tokens → MusicXML → tokens | `python -m tests.test_integrity` and `python -m tests.test_roundtrip --split val` | ~3 min |
| 2b. check window cutting (stitching, prompts) | `python -m tests.test_windows --n 40` | ~3 min |
| 3. *(already done; output in `configs/noise_stats.json`)* fit the noise model on real transcriptions | see [Noise model](#noise-model) | ~1 h with a GPU |
| 4. *(needs transcriptions, see [Noise model](#noise-model))* build the real-transcription splits | `python scripts/build_transcribed.py --asap-split train --maestro-splits test,validation` (and the same with `val`), then `python scripts/build_transcribed.py --asap-split test` | ~2 min |

`data/` is git-ignored. The fitted noise model `configs/noise_stats.json` is committed, so step 3 is only needed to refit it.

## Output format

`data/processed/` contains:

```
vocab.json                 621 target tokens incl. <sep> (+ training counts)
splits.json                piece -> train/val/test
report.json                per-performance stats, skips, and dropped segments
train/  val/  test/        one directory per split, each holding:
  notes.npy   float32 [total_notes, 4]   pitch, onset (s, rel. to segment start; < 0 = carried over), duration (s), velocity
  pedal.npy   float32 [total_pedal, 2]   sustain pedal (on, off) in seconds, rel. to segment start
  tokens.npy  int16   [total_tokens]     prompt + target token ids (no <bos>/<sep>/<eos>)
  index.npy   int64   [n_segments, 7]    note_lo, note_hi, pedal_lo, pedal_hi, tok_lo, tok_hi, tgt_lo
  meta.jsonl  one line per segment       piece, performance, win, s_lo, s_hi, start_pos, bar_start, ...
train_transcribed/  val_transcribed/  test_transcribed/
                           same layout; inputs are REAL Kong transcriptions of the same performances,
                           cut at the clean windows' exact cut points; targets and prompts copied
                           from the matching clean window
```

Segment `i`'s notes are `notes[index[i,0]:index[i,1]]`, and likewise for pedal. Its prompt is
`tokens[tok_lo:tgt_lo]` and its target `tokens[tgt_lo:tok_hi]` (`SegmentStore.get` / `get_prompt`).
The arrays are memory-mapped, so nothing loads into RAM until it's used.

| split | windows | performances | compositions |
|---|---|---|---|
| train | 36,172 (48–192 notes, 50% overlap) | 756 | 153 |
| val | 2,093 (~128 notes, no overlap) | 102 | 29 |
| test | 1,725 (~128 notes, no overlap) | 59 | 14 |
| train_transcribed | 3,841 | 87 (all audio unseen by Kong) | |
| val_transcribed | 394 | 20 (all audio unseen by Kong) | |
| test_transcribed | 611 | 24 (21 seen by Kong, 3 unseen) | 11 |

Real transcriptions cover 99–99.5% of their performances' clean windows (a few windows end up
empty or over the length cap). They can also contain more notes per window than clean MIDI:
p99 251, max ~1,530, vs. a clean max of 761. Cap or truncate the input length in the model.

### Loading for training

```python
from torch.utils.data import DataLoader
from src.dataset import AugmentConfig, MixedTrainDataset, ScoreSegmentDataset, collate
from src.preprocessing.noise import NoiseModel

V = "data/processed/vocab.json"
# synthetic noise on clean train segments + 30% of samples from REAL Kong transcriptions
train = MixedTrainDataset("data/processed/train", "data/processed/train_transcribed", V,
                          noise=NoiseModel.from_json("configs/noise_stats.json"),
                          augment=AugmentConfig(), real_fraction=0.3)
dl = DataLoader(train, batch_size=32, sampler=train.sampler(), collate_fn=collate, num_workers=4)
for epoch in range(E):
    train.set_epoch(epoch)                   # fresh (reproducible) noise every epoch
    for b in dl: ...                         # notes [B,N,8], note_mask [B,N], target [B,T] (pad = 0),
                                             # loss_mask [B,T]: train only on target tokens, not the prompt

loss_w = train.vocab.loss_weights(ornament_weight=5.0)   # -> nn.CrossEntropyLoss(weight=...)
val_clean = ScoreSegmentDataset("data/processed/val", V)               # clean inputs
val_real = ScoreSegmentDataset("data/processed/val_transcribed", V)    # real transcriptions
test_real = ScoreSegmentDataset("data/processed/test_transcribed", V)
```

For synthetic-only training (e.g. the Beyer & Dai-style clean baseline), use
`ScoreSegmentDataset("data/processed/train", V, noise=..., augment=AugmentConfig())`, or pass
`noise=None` for clean inputs.

**Model selection.** Validation is composer-balanced; report the macro-average over composers too.
`val_real` is the best predictor of performance on real audio. Its audio was never used to fit the
noise model.

- **Note features (8 per note):** pitch, onset, inter-onset gap, log duration, velocity/127,
  whether the pedal is down at onset and at offset, and whether the note is carried over.
- **Carried-over notes:** notes struck before a segment but still held at least 1/4 quarter into it
  are included with a negative onset, so a target note tied over the window's first bar line
  (`... tie`) has input evidence. Pedal presses that started before the segment likewise keep a
  negative start. The noise model never moves a note or pedal across t = 0.
  - `notes.npy` also stores *pedal-sustained* candidates (key released, pedal still down). The
    Dataset drops them after noise unless the key is still down at the segment's `carry_min_sec`,
    so clean inputs only show key-held notes, while Kong-style pedal lengthening in the noise model
    produces carried notes the way real Kong transcriptions do. Anything reading `notes.npy`
    directly should apply `src.dataset.drop_released_carried` too.
  - About half of the ties into a window's first beat get a carried note. Most of the rest are
    tied notes the pianist released early and held with the pedal (about 25%), or released early
    with no pedal, so the model still has to infer some ties from the music. Carrying all
    pedal-sustained notes would cover 72% of ties, but 94% of what it adds are not ties.
- **Oversampling:** `sampler()` draws segments with grace notes, arpeggios or ornaments 3× as often.
- **Augmentation** (`AugmentConfig`, training only): applied to the clean segment before noise.
  - Transposition by −6…+6 semitones: input pitches and the target's and prompt's `p_*` tokens
    shift together, and `key_*` is re-spelled. Only shifts that keep every note on the keyboard are
    used.
  - Tempo stretch ×0.85–1.18 (the carried-note threshold `carry_min_sec` is stretched with it).
  - Velocity scale ×0.8–1.2.

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

`transcribe.py` gives the same output as the package's `PianoTranscription.transcribe`: about 0.3%
of notes shift by at most ~1 ms, from GPU rounding differences between batch sizes. It's about 20×
faster:
- batched GPU inference (the package hard-codes batch size 1);
- audio decoding prefetched in background threads;
- note/pedal decoding in worker processes.

Other GPU-heavy programs (e.g. games) slow it down noticeably.

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

Only 25 of the 59 test performances have MAESTRO audio:

| MAESTRO split of the audio | performances | segments | meaning |
|---|---|---|---|
| train | 21 | 549 | Kong's model was trained on this audio, so transcriptions are optimistic |
| validation | 2 | 42 | unseen by Kong's model |
| test | 1 | 20 | unseen by Kong's model |

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
src/baselines/note_features.py   per-note context features (XGBoost baseline)
src/dataset.py     PyTorch Dataset / sampler / collate
scripts/           build_dataset, make_bd_split, fetch_maestro_audio, transcribe, fit_noise,
                   build_transcribed, train_xgb_detector, audit_data
tests/test_integrity.py   leakage, pairing, array and loader checks
tests/test_roundtrip.py   tokens -> MusicXML -> tokens
```

## Not done yet: PianoCoRe

PianoCoRe ([repo](https://github.com/ilya16/PianoCoRe), [paper](https://doi.org/10.5334/tismir.333))
already contains ASAP. When adding it:

- send its scores through the same `load_score` / `tokenize_bars`;
- map its pieces to the same composition IDs, so no Beyer & Dai test piece enters training;
- de-duplicate performances that also appear in ASAP;
- sample per score (it has about 100 performances per score).

The shard → vocab → memory-mapped flow in `build_dataset.py` already scales to it.
