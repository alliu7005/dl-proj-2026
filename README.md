# dl-proj-2026: Noise-Robust Piano Score Reconstruction

Audio → transcriber → noisy performance MIDI → seq2seq transformer → score tokens → MusicXML.

The transcriber is **Transkun V2** (Yan & Duan, ISMIR 2024), the one PianoCoRe's PERiScoPe subset
was made with: training on PERiScoPe and testing on Transkun output use the same kind of errors.
**Kong et al. (2021)**, the transcriber in the proposal, is kept as a second transcriber
(`scripts/transcribe.py --model kong`, `*_transcribed_kong` splits, `configs/noise_stats_kong.json`)
to test robustness across transcribers.

Training data:
- **ASAP**: clean Disklavier MIDI with hand-checked beat annotations, plus synthetic
  transcription noise fitted on real Transkun errors (see [Noise model](#noise-model)), plus real
  Transkun transcriptions of ASAP performances whose audio Transkun never saw.
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
| 1. build train/val/test/test_periscope segments | `python scripts/build_dataset.py` | ~3 min ASAP only; ~25 min with PERiScoPe (4 workers) |
| 2. check the data: leakage, pairing, arrays, loaders; then tokens → MusicXML → tokens | `python -m tests.test_integrity` and `python -m tests.test_roundtrip --split val` | ~3 min |
| 2b. check window cutting (stitching, prompts) | `python -m tests.test_windows --n 40` | ~3 min |
| 3. *(already done: `configs/noise_stats.json`)* transcribe MAESTRO audio and fit the noise model | see [Noise model](#noise-model) | ~1–2 h with a GPU |
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
| `train_transcribed` / `val_transcribed` | ASAP train/val performances with Transkun transcriptions of MAESTRO audio Transkun never saw | real Transkun output | training with real errors / model selection |
| `test_transcribed` | ASAP test with Transkun transcriptions of MAESTRO audio | real Transkun output | small real-audio check; most of this audio was in Transkun's training data |
| `*_transcribed_kong` | the same three, transcribed with Kong | real Kong output | robustness to a different transcriber |

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
                           same layout; inputs are REAL Transkun transcriptions of the same ASAP
                           performances, cut at the clean windows' exact cut points; targets and
                           prompts copied from the matching clean window
train_transcribed_kong/ ...  the same with Kong transcriptions
```

Window `i`'s notes are `notes[index[i,0]:index[i,1]]`, and likewise for pedal. Its prompt is
`tokens[tok_lo:tgt_lo]` and its target `tokens[tgt_lo:tok_hi]` (`SegmentStore.get` / `get_prompt`).
The arrays are memory-mapped, so nothing loads into RAM until it's used. Read through the Dataset,
not the raw arrays: it also drops released carried-over candidates (below) and scales the features.

Current build (ASAP + PERiScoPe; `report.json` has the exact counts):

| split | windows | ASAP performances (compositions) | PERiScoPe performances (works) |
|---|---|---|---|
| train | 205,089 | 739 (153) | 5,260 (971) |
| val | 10,252 | 106 (29) | 434 (78) |
| test | 1,725 | 59 (14), Beyer & Dai's test set | none |
| test_periscope | 6,755 | none | 490 (104) |
| train_transcribed | 3,733 | 85, audio unseen by Transkun | |
| val_transcribed | 445 | 22, audio unseen by Transkun | |
| test_transcribed | 611 | 24 (21 seen by Transkun, 3 unseen) | |

The `*_transcribed_kong` splits have the same windows.

- **Coverage:** real transcriptions cover 99–99.5% of their performances' clean windows.
- **Input length:** real transcriptions can have more notes per window than clean MIDI (p99 235,
  max ~1,530, vs. a clean max of 761). Cap or truncate the input length in the model.
- **Target length:** p99 904, max 1,489 tokens.
- **PERiScoPe drops:** 3,714 windows were dropped by the input-vs-score pitch filter (<0.6 F1).
  On real transcriptions this filter also removes some genuinely noisy windows; lower
  `min_pitch_f1` if that matters.
- **Unusual scores:** one PERiScoPe score (Liszt, *Mephisto Waltz No. 3*) has 5 staves. Staves
  beyond the 4th fold into `staff_3`.

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

Everything is **fitted on real Transkun V2 transcriptions** of 93 recordings (419k notes) from
the MAESTRO test and validation splits.
- Transkun was trained on MAESTRO train, so only test/validation audio gives realistic errors.
- Audio of ASAP **val and test** performances is excluded from the fit, so `val_transcribed` and
  `test_transcribed` remain independent checks.

The model is **empirical**: `configs/noise_stats.json` holds three fitted pieces, plus a few scalar
parameters for timing, velocity and pedal.

- **Missing notes:** a miss-rate table indexed by chord size × velocity bin.
- **Hallucination rate:** hallucinations spawned per real note, by the source note's velocity.
- **Error bank:** real hallucinations, each stored relative to its most likely source. The source
  is a recent, loud note that is still sounding, counting the sustain pedal. Each entry records
  interval, delay, velocity ratio and duration.

During training, loud notes spawn hallucinations by replaying sampled bank entries.

The Kong fit (`configs/noise_stats_kong.json`) uses exactly the same procedure. The two
transcribers make very different errors:

| error | Transkun V2 (`noise_stats.json`) | Kong 2021 (`noise_stats_kong.json`) |
|---|---|---|
| hallucinated notes | **0.19%**, mostly ±1 octave, then sixths and thirds | 1.83% (~10× more) |
| notes lengthened to pedal release | **13%** of notes released under sustain | 68% |
| missed lone notes, quiet (vel < 40) / loud | 25% / 0.5% | 29% / 0.5% |
| missed notes in 8-note chords, quiet / loud | 56% / 8.5% | 57% / 5.7% |
| missed pedal presses | 11% | 8% |
| onset jitter | 5.5 ms | 6.1 ms |

**What this means for the model:** with Transkun, extra notes are rare. The errors the model has to
fix are mostly **missing notes** (quiet notes and dense chords) and **rhythm/duration
quantization**, not hallucinations.

`configs/noise_stats_parametric.json` is the first, hand-parametrized version, kept for ablations.

**Validation with a detector:** see [the error detector](#xgboost-baseline-note-level-error-detector).
With Kong noise, a detector trained only on synthetic errors reached AUC 0.90 on real errors in
unseen audio, vs. 0.66 for the parametric noise.

To reproduce the fit:

```bash
# range-downloads single recordings from the 108 GB MAESTRO zip (~10 GB for all three)
python scripts/fetch_maestro_audio.py --split test --limit 40
python scripts/fetch_maestro_audio.py --asap-split train,val --maestro-splits test,validation
python scripts/fetch_maestro_audio.py --asap-split test

# Transkun V2 (default). One process by default; --jobs N runs N workers (~4 GB GPU memory each).
# Ctrl+C stops all of them cleanly; running the same command again resumes.
python scripts/transcribe.py
python scripts/fit_noise.py --pairs data/transcribed_transkun/pairs.csv --out configs/noise_stats.json

# Kong, for comparison (env: Python <= 3.12, pip install piano_transcription_inference audioread;
# get the checkpoint with curl, since the package's own download uses wget):
#   curl -L -o ~/piano_transcription_inference_data/"note_F1=0.9677_pedal_F1=0.9186.pth" \
#     "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
python scripts/transcribe.py --model kong
python scripts/fit_noise.py --pairs data/transcribed/pairs.csv --out configs/noise_stats_kong.json
```

**Transcriber speed on an RTX 5070 Ti:**
- **Transkun:** ~12× real time per process. The in-process mode loads the model once; `--isolate`
  restores a fresh process per recording, for small GPUs.
- **Kong:** ~40× real time.

For Kong, `transcribe.py` gives the same output as the package's `PianoTranscription.transcribe`:
about 0.3% of notes shift by at most ~1 ms, from GPU rounding differences between batch sizes.
It's about 20× faster:
- batched GPU inference (the package hard-codes batch size 1);
- audio decoding prefetched in background threads;
- note/pedal decoding in worker processes.

Other GPU-heavy programs (e.g. games) slow it down noticeably.

## Training the transformer

```bash
python scripts/train.py --name base             # start; Ctrl+C saves and stops
python scripts/train.py --name base --resume    # continue
python scripts/evaluate.py --run base           # test sets, same metrics/windows as the baselines
```

**Model** (`src/model.py`): encoder-decoder transformer, about 10M parameters by default.
- **Encoder:** the performed notes. Each note gets a pitch embedding, its continuous features, a
  Fourier encoding of its onset time, and a note-index position.
- **Decoder:** writes the score tokens autoregressively after `<bos> prompt <sep>`. Its output
  embeddings are tied to the input embeddings.
- **Size flags:** `--d-model`, `--enc-layers`, `--dec-layers`.

**Training** (`scripts/train.py`):
- **Data:** all of `train` (real PERiScoPe transcriptions, plus clean ASAP with synthetic Transkun
  noise and augmentation) and `train_transcribed`, in their natural proportion.
- **Optimization:** bf16 mixed precision, AdamW, 2,000 warmup steps then cosine decay, label
  smoothing 0.1, rare ornament/pedal tokens weighted 5× in the loss.
- **Batching:** windows are grouped by length, and each batch is filled up to a token budget
  (`--batch-tokens 16384`, `--batch-notes 8192`) instead of a fixed number of windows, so little
  GPU time goes to padding.
- **Decoder attention:** uses PyTorch's fused flash-attention kernel (pure causal mask during
  training). Generation uses a key/value cache, which gives identical output to step-by-step
  decoding but runs ~5× faster.
- **Speed:** ~350k target tokens/s on an RTX 5070 Ti, about **5 min per epoch** (3.5× faster
  than the first version). The default 10 epochs takes under an hour plus evaluations.
- **CPU workers:** 6 processes prepare batches (`--workers`); the GPU never waits for data.
  Lower this if the PC lags. Everything runs at below-normal CPU priority.
- **Every `--eval-every` steps:** teacher-forced loss and token accuracy on fixed val subsets, plus
  greedy decoding of a few windows from `val` (ASAP), `val` (PERiScoPe) and `val_transcribed`,
  scored with the baseline metrics. Logged to `outputs/runs/<name>/log.csv`.
- **Checkpoints:** `last.pt` at every eval and on Ctrl+C; `best.pt` has the best `val_real` loss.

**Clean-input baseline** (Beyer & Dai-style, for H1): `--no-noise` trains on clean ASAP windows
only, with augmentation but no synthetic noise and no real transcriptions.

**Evaluation** (`scripts/evaluate.py`): greedy decoding of every test window under the same
conditions as `run_baselines.py`, plus `periscope` (`test_periscope`). It uses the ground-truth
prompt before each window, the same window-level protocol as the baselines.

## Scoreformer (main model)

`src/scoreformer.py` is the main model. Score2Seq (`src/model.py`, ~10M parameters) was a
first small-scale check. The Scoreformer uses the same data, windows, noise and token grammar, and
the same `forward` / `generate` interface, so both are trained and scored by the same scripts.

```bash
python scripts/train_model.py --name sf_base                      # base size, ~45M parameters
python scripts/train_model.py --name sf_base --resume             # continue after Ctrl+C / time limit
python scripts/evaluate_pieces.py --run sf_base --musicxml 5      # test sets, 3 prompt modes
python -m tests.test_scoreformer                                  # model + chained-prompt checks
sbatch pace/train.sbatch                                          # train + evaluate on a PACE GPU
```

**What differs from Score2Seq:**

| | Score2Seq | Scoreformer |
|---|---|---|
| size | 10M (d=256, 4+6 layers) | `--size small/base/large`: 8M / 45M / 133M |
| note timing | Fourier features of the *standardized* onset (finest period ~0.4 s, since onset std is 6.3 s) | **rotary attention on onset seconds**, periods 20 ms to 60 s: attention depends on how far apart two notes were played |
| note features | one linear layer on the standardized values | log-binned embeddings of gap and duration, binned velocity, plus the linear layer |
| decoder positions | learned table (fixed max length) | rotary over token index |
| feed-forward | GELU | SwiGLU |
| best.pt | lowest `val_real` loss | lowest mean unweighted loss over all validation sets |

**Training** (`scripts/train_model.py`): the same loop and helpers as `scripts/train.py` (token-budget
batches, bf16, warmup + cosine, label smoothing, 5× weight on ornament/pedal tokens). Additions:
`--arch score2seq` trains Michael's model under the identical loop (a fair architecture comparison),
`--no-noise` is the clean-input ablation, `--grad-accum` / `--compile` for big runs, and missing
splits (`train_transcribed`, PERiScoPe) are skipped instead of crashing. 20 epochs by default.

**Evaluation** (`scripts/evaluate_pieces.py`), works for checkpoints of either script:

| prompt mode | the decoder's context before each window | use |
|---|---|---|
| `oracle` | the ground-truth score before the window (= `scripts/evaluate.py`) | upper bound |
| `chained` | the model's **own** output so far, window by window through each piece | **the system's real performance** |
| `none` | nothing | robustness check |

All three score each window against its ground truth with `src/eval/metrics.py`, so they compare
directly with the baselines. `--musicxml N` writes the stitched prediction and ground truth of the
first N pieces as MusicXML (for figures and for MUSTER / MV2H).

**Prompt fix** (`segment.prompt_from_output`): when a position in the bar held only a pedal token,
removing the pedal left an empty `pos_k sub_x` behind, so 43% of inference-time prompts differed
from the training prompts (the old test compared them by meaning, not token by token). Empty
positions are now dropped; `tests/test_scoreformer.py` checks all 3,631 val/test prompts are
identical token for token.

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
| **2. Beat-snap + XGBoost** (`src/baselines/xgb_quantizer.py`) | lightweight supervised | Baseline 1, plus four per-note XGBoost classifiers (keep/delete, staff, written duration, onset shift). Trained on synthetic-noise (Transkun) and real-Transkun training windows (1.16M notes). |
| 3. Beyer & Dai (2024), pretrained | prior published | Not run yet: needs their code and checkpoint (external code; see below). |

Conditions (same windows, same targets):
- **clean:** clean MIDI.
- **synthetic:** clean MIDI plus our fitted (Transkun) noise.
- **clean_paired:** clean MIDI on exactly the windows that have a real transcription.
- **real:** Transkun transcriptions.
- **real_unseen:** real transcriptions of audio Transkun never saw (3 performances, 62 windows).
- **real_kong:** the same windows transcribed by Kong.

| metric | B1 clean | B1 real | B2 clean | B2 synthetic | B2 clean_paired | B2 real | B2 real_unseen | B2 real_kong |
|---|---|---|---|---|---|---|---|---|
| note F1 | 0.929 | 0.927 | 0.932 | 0.914 | 0.935 | 0.932 | 0.918 | 0.930 |
| onset F1 (rhythm) | 0.631 | 0.645 | 0.718 | 0.700 | 0.738 | 0.738 | 0.801 | 0.740 |
| duration acc | 0.341 | 0.242 | 0.557 | 0.546 | 0.530 | 0.535 | 0.537 | 0.541 |
| staff acc | 0.743 | 0.786 | 0.829 | 0.830 | 0.871 | 0.871 | 0.925 | 0.869 |
| tie / grace F1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

**Reading the table:**
- **Not meaningful for these baselines:** bar F1 (≈1.0) and pedal F1 (1.0 on clean). Both
  baselines are given the true beats, bar lines and pedal, so those metrics are trivially high.
- **What the transformer has to beat:** rhythm (onset F1 ~0.71–0.79), written durations
  (~0.5), staff (~0.83–0.91), and ties/grace notes, which the baselines can't produce at all.
- **Which columns to compare:** "clean" and "real" cover different performances; compare
  `clean_paired` with `real`. The real column is mostly audio Transkun was trained on, hence no
  drop.
- **Synthetic noise** costs Baseline 2 about 1.8 points of onset F1.
- **Transkun vs Kong inputs:** almost the same for these baselines.
- **test_periscope:** the baselines need ASAP's beat annotations (oracle beats), so they can't
  run on `test_periscope`. Only the transformer is evaluated there.

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
- **On real errors** (MAESTRO test/validation transcriptions, with ASAP val/test recordings
  excluded). Labels come from matching each note against the ground truth within 50 ms.

| detector | synthetic test AUC / AP | real test AUC / AP | real, unseen audio AUC / AP |
|---|---|---|---|
| **Transkun** (current) | | | |
| flag quietest notes | 0.722 / 0.004 | 0.652 / 0.011 | 0.849 / 0.033 |
| trained on synthetic noise | 0.983 / 0.430 | 0.202 / 0.005 | 0.924 / 0.005 |
| trained on real errors | 0.914 / 0.019 | 0.315 / 0.006 | 0.994 / 0.098 |
| **Kong** (earlier run) | | | |
| trained on synthetic noise | 0.961 / 0.589 | 0.540 / 0.030 | 0.904 / 0.090 |
| trained on real errors | 0.818 / 0.062 | 0.620 / 0.117 | 0.987 / 0.435 |

Error rates (= AP of random guessing): Transkun synthetic 0.002, real 0.008, unseen 0.001; Kong
about 10× higher.

**Takeaways:**

- **With Kong, the detector works.** Hallucinations are 1.8% of notes, mostly quiet octave
  overtones of a louder note that is still sounding, and they are detectable (AUC 0.90–0.99 on
  unseen audio).
- **With Transkun, there's almost nothing to detect.** Hallucinations are 0.19% of notes; on
  unseen audio only ~0.1% of notes are errors (a handful of notes), so the AP numbers are noise.
- **Transkun "real test" AUC is below 0.5.** The few mismatches there are mostly timing differences
  on audio Transkun was trained on, not hallucinations, and they look like confident real notes.
- **Conclusion:** with Transkun, a delete-the-extra-notes step isn't worth it. The model's work is
  recovering missing notes and quantizing rhythm and durations. The detector score is unlikely to
  help as an input feature; keep it only as an ablation.

## Real-transcription splits

- **`train_transcribed` / `val_transcribed`:** only audio Transkun never trained on (MAESTRO
  test/validation): 85 training and 22 validation performances. Against the score, input pitch
  agreement drops from 0.962 (clean) to 0.955 (Transkun) / 0.948 (Kong) on train; on val from
  0.957 to 0.952 / 0.945.
- **`test_transcribed`:** see the caveat below.

### Caveat: test_transcribed

Only 25 of the 59 ASAP test performances have MAESTRO audio :

| MAESTRO split of the audio | performances | segments | meaning |
|---|---|---|---|
| train | 21 | 549 | Transkun (and Kong) were trained on this audio, so transcriptions are optimistic |
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
src/model.py                     Score2Seq: first small encoder-decoder transformer
src/scoreformer.py               Scoreformer: main model (time-rotary encoder), checkpoint loading
src/dataset.py     PyTorch Dataset / sampler / collate
scripts/train_model.py           train either model; scripts/evaluate_pieces.py: oracle/chained/none
pace/train.sbatch                PACE GPU job: train + evaluate
tests/test_scoreformer.py        model checks + chained prompts == training prompts
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
