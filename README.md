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
  Both are written to a temporary file and renamed, so a job killed mid-save (a time limit) keeps the
  previous complete checkpoint and the resubmitted job resumes. `status.json` holds the training
  phase (`train` / `cooldown` / `done`), step and best loss from the first step on, readable without
  torch: `pace/reevaluate.sh` leaves a run alone until it says `done`.

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
python scripts/evaluate_pieces.py --run sf_base --musicxml 5      # test sets, 5 context modes
python scripts/build_metrical.py                                  # labels for --aux-metrical (once)
python -m tests.test_scoreformer                                  # model + chained-prompt checks
python -m tests.test_decoding                                     # chained / guarded / meter decoding
python -m tests.test_pipeline                                     # demo pipeline end to end
python -m tests.test_training_stop                                # LR schedule, early stop, cooldown
python -m tests.test_context_notes                                # context notes, exact neighbours
python -m tests.test_metrical                                     # metrical labels and the aux head
python -m tests.test_eval_plan                                    # what an evaluation still has to compute
python -m tests.test_reevaluate                                   # pace/reevaluate.sh submissions (bash)
python -m tests.test_compare_runs                                 # the comparison tables
python -m tests.test_beats                                        # beats: ground truth, features, tracker noise
python scripts/build_beats.py --check                             # ground-truth beats -> <split>/beats_gt.npz
python scripts/compare_runs.py                                    # all evaluated runs -> outputs/compare/
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
splits (`train_transcribed`, PERiScoPe) are skipped instead of crashing.

**Early stopping with a cooldown.** `--epochs` (default 20) is the maximum and the length of the
cosine learning-rate schedule. `sf_base` showed why a fixed 20 epochs is wrong for big models: its
mean validation loss bottomed out at step 20,000, in the 3rd of 20 epochs (0.275), and rose to
0.317 by the end. Now, once
the mean validation loss has not improved for `--patience` epochs (default 5; `0` = off),
training reloads the best checkpoint (weights and optimizer state) and runs a `--cooldown`
(default 1 epoch) in which the learning rate decays linearly from its value at the best step to the
schedule's minimum. An early-stopped run therefore still gets the low-learning-rate phase that the
cosine schedule would have given it at the end, and `best.pt` keeps improving if the cooldown
helps. The phase (`train` / `cooldown` / `done`) is stored in every checkpoint: a job killed
mid-cooldown resumes the cooldown, and resubmitting a finished run trains nothing.
`tests/test_training_stop.py` checks the schedule values and runs real short trainings for every
path (stop + cooldown at the predicted steps, resume after finishing, resume mid-cooldown,
`--cooldown 0`, `--patience 0`, context notes + auxiliary head kept on resume).

**Bar alignment.** The realistic scores trail the oracle ones mostly in rhythm (guarded onset F1
~0.2 vs ~0.56), and onset-shift F1 is about twice onset F1: windows often get the rhythm right but
place it at the wrong point in the bar. A window sees ~128 notes, a few bars, and has to work out
the meter and where the bar starts on its own. Three fixes, each optional and recorded in the
checkpoint (so evaluation and the demo follow it automatically):

- **Context notes** (`--context-before N --context-after M`): the encoder also sees the N performed
  notes before the window and the M after it, flagged by two extra input features (10 instead of
  8), so it can read the rhythm and meter around the window. The target is unchanged: the model
  still writes only the window's notes. The notes come from a pool of every stored window of the
  same performance in absolute time (no data rebuild), get the same augmentation as the window and
  their own synthetic noise, and are assembled by one function (`dataset.assemble_notes`) for
  training and for the inference pipeline. With both set to 0 the data is identical to before.
- **Auxiliary metrical head** (`--aux-metrical`, `--aux-weight 0.5`): a linear head on the
  encoder predicts, for every performed note of the window, whether it starts a bar and its
  position within the beat (on the beat, an eighth, a triplet, a sixteenth, finer), so the encoder
  has to learn the bar phase from the timing. Labels (`src/metrical.py`) come from aligning each
  window's performed notes to its score: a rank-based first time map that follows any rubato, then
  order-preserving per-pitch matching and two refits through the matches. On performances rendered
  from real windows' scores (drifting tempo, chord asynchrony, 3% dropped and 3% extra notes) the
  labels are 99% correct and cover 97% of the notes; on the real data they cover 88% (test) and
  91% (train) of performed notes (the rest are mostly ornaments and grace notes the score writes
  differently). `scripts/build_metrical.py` precomputes them into `<split>/metrical.npy` once per
  data build (45 s on 6 cores here); validation logs the head's downbeat F1 and phase accuracy. The
  head is used only in training; decoding never touches it.
- **Meter consistency** (decoding mode `meter`, below): fixes isolated windows whose time signature
  disagrees with the rest of the piece.

**Evaluation** (`scripts/evaluate_pieces.py`), works for checkpoints of either script:

| context mode | the decoder's context before each window | use |
|---|---|---|
| `oracle` | the ground-truth score before the window (= `scripts/evaluate.py`) | upper bound |
| `chained` | the model's **own** output so far, window by window through each piece | what a real recording gets, unguarded |
| `none` | nothing | robustness check |
| `guarded` | chained, with degenerate windows redone without context (below) | **the system's real performance** (the demo's default) |
| `meter` | guarded, plus consistent time signatures along each piece (below) | guarded, with the meter fixed |
| `piece` | meter, with one time signature per piece unless it changes for a long stretch | |
| `ts_true` | guarded with each window's true time signature forced (below) | the most a time-signature choice can give |
| `tsearch` | each piece decoded under its 3 likeliest time signatures, keeping the most probable (below) | |
| `tsearch_beat` | the same candidates, keeping the one whose beats agree with the tracked beats (below) | |

All modes score each window against its ground truth with `src/eval/metrics.py`, so they compare
directly with the baselines. Results are merged into `eval_pieces_best.json` after every test set
(under a lock: `PER_CONDITION=1 bash pace/reevaluate.sh <run>` runs one job per test set, and a job cut
off by its time limit keeps the test sets it finished). `--musicxml N` writes the stitched prediction and ground truth of the
first N pieces as MusicXML (for figures and for MUSTER / MV2H). Running the script with a subset of
`--modes` adds those modes to an earlier evaluation's files instead of replacing them.

Chained, guarded and meter decoding live in `src/decoding.py`, which the demo pipeline uses too, so
the demo and the evaluation run identical decoding. Each window's output is capped at
8 tokens per input note + 64 (`decoding.max_new_tokens`): real targets average 3.4 tokens per note
and the longest train / val / test target is 80% of the cap, so it never truncates a real output;
it only keeps a looping window from running on to the model's 2,048-token limit.

**Prompt fix** (`segment.prompt_from_output`): when a position in the bar held only a pedal token,
removing the pedal left an empty `pos_k sub_x` behind, so 43% of inference-time prompts differed
from the training prompts (the old test compared them by meaning, not token by token). Empty
positions are now dropped; `tests/test_scoreformer.py` checks all 3,631 val/test prompts are
identical token for token.

**Context noise** (`--context-noise P`, `src.dataset.ContextNoise`): in chained evaluation every
model scored below decoding each window with no context at all (Score2Seq, clean: onset F1 0.14
chained vs 0.19 without context), because training only ever shows the correct context. With
`--context-noise 0.5`, half of the training windows that have a context get it corrupted with the
mistakes a model's own previous output makes: a wrong time signature, all positions shifted by
1-2 quarter notes, the first positions missing (bar line placed late), deleted notes, wrong
durations, flipped staff. Targets and note inputs are never changed, and with the flag off the data
is identical to before (`python -m tests.test_context_noise`).

**Guarded chained decoding** (`--modes guarded`, on by default). `scripts/analyze_chained.py`
rebuilds every chained context from saved predictions and showed that the chained losses come from
cascades, not from long contexts as such. One degenerate window (no notes, or a loop more than 3x
its target's length) wrecks the next window's context, and pitch F1 after such a window drops to
0.12-0.40, often degenerating again:

| model, synthetic input | degenerate windows | pitch F1 after one | pitch F1 chained / none |
|---|---|---|---|
| s2s | 0.6% | 0.77 | 0.87 / 0.88 |
| s2s_ctx | 20.9% | 0.16 | 0.70 / 0.88 |
| sf_small_ctx | 71.2% | 0.12 | 0.24 / 0.86 |
| sf_base | 56.9% | 0.21 | 0.36 / 0.81 |
| sf_base_reg | 45.4% | 0.30 | 0.49 / 0.86 |

The model cannot see the true score at inference, but it does know the notes it was given: a
healthy output contains about the same pitches as its input, a degenerate one does not. Guarded mode
also decodes every window without context and keeps that output when its pitch F1 against the
window's input notes (`decoding.input_agreement`) beats the chained output's by more than
`--guard-margin` (default 0; ties keep the chained output). The kept output becomes the history the
next context is built from, so one bad window can no longer take the rest of the piece down with it.
`tests/test_decoding.py` checks the logic with a stand-in model.

**Choosing the margin.** `scripts/simulate_guard.py` estimates guarded decoding offline from saved
chained / none predictions for a range of margins (conservatively: it cannot show the better
contexts that follow a replaced window). On all six post-milestone models, a smaller margin was
better down to 0, on note, onset, duration and bar F1 in nearly every model and condition:

| s2s, clean (offline) | windows replaced | note F1 | onset F1 | duration acc | bar F1 |
|---|---|---|---|---|---|
| margin 0.2 | 2% | 0.890 | 0.138 | 0.572 | 0.433 |
| margin 0.05 (first default) | 16% | 0.900 | 0.148 | 0.594 | 0.457 |
| margin 0 | 56% | 0.907 | 0.170 | 0.642 | 0.494 |

At margin 0 the no-context output still had the better rhythm on its own (onset F1 0.188), so the
simulation now also tries negative margins (the no-context output wins unless it agrees clearly
worse), and the real decoding can compare margins in one job: `--extra-margins -0.05,-0.02` adds
modes `guarded@-0.05`, `guarded@-0.02` (and `meter@...` if meter is run) next to the default, each
with its `guard_margin` recorded in the results.

**Meter consistency** (`--modes meter`, on by default). A wrong time signature mis-sizes every bar
of its window, so every later position shifts. Every window opens with `[bar] ts_X`, so after a
guarded pass the time signatures along a piece can be read off and checked. `smooth_meters` works on
runs of windows with equal time signatures: a run of at most 2 windows with the same time signature
on both sides takes it (`A A B A A`, `A B B A A A`); at the start or end of a piece a short run takes
its neighbour's only if that run is at least twice as long (`B A A`); a genuine change of meter
(`A A A B B B`) has no short run between equal ones and is kept. Each corrected window is decoded
again with the new time signature forced as its first token(s), and generating any other time
signature in that window is blocked (`generate(banned=...)`: without the block a real model writes
its own `ts_4/4` right after a forced `ts_3/4`). The whole piece is then decoded again from there,
so the windows after a corrected one see the corrected context. Meter mode reuses guarded mode's
outputs as its first pass and costs nothing when no window needs fixing; the evaluation reports the
share of windows corrected (`meter_fix_rate`) and `ts_acc`.

**Time-signature search** (`--modes ts_true,tsearch,tsearch_beat`, on by default; `pace/tsearch.sh`).
Without the true context the best model gets only 33-41% of its windows' time signatures right,
even when it is given the annotated beats and downbeats, and with the true context 98%: onset F1
then roughly doubles (real: 0.319 with no context, 0.577 with the true one). The model picks a
window's time signature as its first token, by argmax, before it has written a single note, and a
wrong one mis-sizes every bar after it. Meter mode only fixes isolated windows; a piece whose
windows mostly guess wrong stays wrong (`piece` mode does not help: the majority is wrong too).

- `tsearch`: each piece is decoded once per candidate time signature, forced in every window
  (guarded decoding, so degenerate windows still fall back), and the candidate the model finds most
  probable is kept: the mean log-probability of the tokens it generated over the whole piece
  (`generate(return_logp=True)`, `decoding.Scored`). The candidates are the piece's 3 most frequent
  time signatures in its guarded and no-context outputs, topped up with the commonest ones of the
  training targets (`decoding.FALLBACK_TS`: 4/4, 3/4, 6/8, 2/4, 12/8, 2/2). All pieces' candidate
  ranks share the batches, so it costs 3 guarded decodings.
- `tsearch_beat`: the same candidates, keeping the one whose written beats agree best with the
  condition's beats (`src/beats.py grid_agreement`): the window's notes are aligned to the
  performed ones (`window_grid`), which times its written beats (`beat_layout`: a 6/8 reading puts
  them on dotted quarters, a 3/4 one on quarters) and bar lines; score = beat F-measure against the
  tracked beats + 0.5 x downbeat F-measure (bar lines against the tracked downbeats), averaged over
  the piece's windows. The model's probability breaks ties and decides where there are no beats
  (models trained without them, MIDI input).
- `ts_true` (diagnostic): guarded with each window's true time signature forced, the most any choice
  of time signature can give.

Results report `ts_switch_rate`, the share of windows whose time signature the mode changed from
guarded decoding. The demo offers both searches ("Meter search", "Meter search by beats");
`tests/test_decoding.py` checks the search with a stand-in model that scores its outputs.

**Tried and dropped: regularizing the tracker's downbeats.** A metrical HMM over Beat This!'s beats
(states: bar length 2 / 3 / 4 and position in the bar; emissions: the fitted downbeat rates per
position; transitions allowing missed / extra beats and meter changes) makes its downbeats regular,
but never more accurate: downbeat F on train / val _transcribed 0.646 raw, 0.54-0.64 over 12
settings. Beat This! marks 38% of half-bar beats as downbeats, so the regular reading it settles on
is often the half bar.

### Beats as an input (`--beats`)

Even given the true previous bars, the best model puts ~40% of onsets in the wrong place, while the
beat-snap + XGBoost baseline, which is given the annotated beats, reaches onset F1 0.74. So the
model can also be given the beats (`src/beats.py`): 7 more features per note, its phase between the
surrounding beats and its position in the bar (each on the unit circle), the local beat period,
the beats per bar, and a "has beats" flag (all zero where there are no beats).

| where the beats come from | ASAP | PERiScoPe (no annotations, no audio) |
|---|---|---|
| ground truth (`scripts/build_beats.py` -> `<split>/beats_gt.npz`) | the human annotations (`asap_annotations.json`) | the score's beats timed by the note alignment of every window (`window_grid`), joined by bar number |
| a real tracker (`scripts/track_beats.py` -> `beats_tracked.npz`) | Beat This! on the MAESTRO audio of the real-transcription splits | - |

- **Beat counting follows ASAP** (`beat_layout`, checked on all 1,067 annotated performances):
  dotted beats when the numerator is a multiple of 3 above 3 (6/8 -> 2, 12/8 -> 4, 6/4 -> 2), the
  numerator otherwise (4/4 -> 4, 3/8 -> 3).
- **The alignment-derived grid is checked against the human annotations** on ASAP (`build_beats.py
  --check`): median beat F-measure 0.92 (train), 0.90 (val), 0.85 (test); where it is right it is
  exact (median error 1 ms), and it is weakest around pickup bars (written left-aligned in our
  tokens) and on beats where nothing is played. That is better than the tracker it stands in for
  (below: beat F 0.81 / 0.71), and training adds tracker errors on top.
- **No leakage from the tracker.** Beat This! was trained on ASAP, so each performance is tracked by
  the authors' cross-validation model that never saw it (their `asap/8-folds.split`: fold k ->
  checkpoint `fold<k>`; tracks outside the split -> `final0`). Its beats are shifted onto the ASAP
  timeline by the same offset as the transcription.
- **How good Beat This! is on our audio** (`scripts/fit_beat_noise.py`, held-out fold models, scored
  within each piece's annotated span: 101 of the 418 MAESTRO recordings hold more than one ASAP
  piece, e.g. a prelude and its fugue, and the tracker hears the whole recording):

  | | beat F | downbeat F | wrong tempo octave |
  |---|---|---|---|
  | train / val _transcribed (107 performances, fitted on) | 0.81 | 0.65 | 23% (19 too fast, 6 too slow) |
  | test_transcribed (24, never fitted on) | 0.71 | 0.60 | 33% (8 too fast) |
  | the Beat This! paper, ASAP | 0.76 | 0.61 | |

  At the right tempo it misses 12% of the beats and adds 13% spurious ones (mostly mid-beat: local
  subdivision), 7 ms late on average with 18 ms jitter. Downbeats: without its DBN (`dbn=False`,
  what we run) Beat This! imposes no bar length, every downbeat is its own peak: it marks 87% of
  the true downbeats, but also 38% of the half-bar beats (beat 3 of 4/4), 22% of the other beats
  and 34% of its spurious ones, and how often varies a lot between pieces (false downbeats on
  "other" beats: 9% to 40% between the quartiles). Too-fast tracking is often partial (a stretch of
  the piece) and sometimes by 3 (the eighth notes of a compound meter).
- **Tracker errors are fitted, not guessed** (`scripts/fit_beat_noise.py` -> `configs/beat_noise.json`):
  Beat This! vs the annotations on train / val _transcribed (never test), one performance at a
  time (`compare`). `BeatNoise` replays them on the ground truth: each beat list draws one fitted
  performance profile (from the same bar length; its miss / extra rates, timing, and downbeat rates
  by the beat's place in the bar), and with the fitted probability a tempo octave error replayed
  from one of the real wrong-tempo performances (its kind, factor, the stretch of the piece it
  covered, its downbeat rates). Replayed on the same ground truth it tracks like Beat This!: beat F
  0.82 vs 0.81 (right tempo 0.87 vs 0.87, too fast 0.64 vs 0.62, too slow 0.70 vs 0.64), downbeat F
  0.60 vs 0.65 (a little pessimistic, the wrong-tempo lists most); on test, never fitted on,
  downbeat F 0.60 vs 0.60
  (`tests/test_beats.py test_noise_matches_tracker`). Training applies it (`--beat-noise`; a run
  records the values it started with in config.json and resumes with them), and `--beat-dropout`
  (0.15) of the windows get no beats, so the model also works on MIDI input without a tracker. On
  the real transcriptions it trains on the tracker's own beats.

Beat models are evaluated with the beats each condition would really have (`src/eval/plan.py BEATS`):

| condition | beats | what for |
|---|---|---|
| clean, synthetic, periscope | ground truth + the fitted tracker errors (fixed seed) | the usual test sets, a simulated tracker |
| real | Beat This! on the audio (held-out folds) | the system as the demo runs it |
| clean_gt_beats, real_gt_beats | the annotated beats | the upper bound; what the beat-snap baselines are given |
| real_no_beats | none | MIDI input without a tracker |

Models without beats skip the three beat-only conditions. **`piece` mode** (new, for every model)
is meter mode with no limit on how long a deviating run of windows may be: a piece keeps one time
signature unless a different one holds for a stretch as long as the rest (`PIECE_MAX_BLIP`).

### Post-milestone results

All runs trained with the same loop on the same PACE data (ASAP + PERiScoPe; the runs in this
first table not on the real-transcription split, which was built after they started; the beat batch
below also on it), each evaluated with `best.pt` on
four test sets: the 59 ASAP test performances as clean MIDI and with the fixed synthetic noise draw,
the PERiScoPe test transcriptions, and `real`, 24 ASAP test performances transcribed from their
MAESTRO audio by Transkun V2 (closest to the demo). Tables: `python3 scripts/compare_runs.py` ->
`outputs/compare/` (every run, test set, mode and metric; `margins.md` for the margin sweeps).
Numbers below are **meter mode** unless stated: the system as the demo runs it.

**The bar-alignment fixes** (context notes: 64 performed notes before / 32 after each window;
auxiliary metrical head: per-note downbeat and beat phase, `src/metrical.py`), onset F1 / bar F1:

| run | params | best val loss | clean | synthetic | PERiScoPe | real |
|---|---|---|---|---|---|---|
| s2s | 10.2M | 0.2781 | 0.202 / 0.507 | 0.192 / 0.504 | 0.246 / 0.521 | 0.223 / 0.484 |
| s2s_c (context notes) | 10.2M | 0.2694 | 0.198 / 0.521 | 0.181 / 0.501 | 0.242 / 0.520 | 0.234 / 0.509 |
| s2s_a (metrical head) | 10.2M | 0.2800 | 0.211 / 0.501 | 0.201 / 0.480 | 0.354 / **0.660** | 0.257 / 0.567 |
| s2s_ca (both) | 10.2M | 0.2779 | 0.267 / 0.553 | 0.255 / 0.557 | 0.332 / 0.601 | 0.319 / 0.568 |
| s2s_m_ca (both, regularized) | 25.9M | 0.2563 | 0.280 / 0.604 | 0.271 / 0.599 | **0.367** / 0.653 | 0.309 / 0.590 |
| sf_base_reg | 44.6M | 0.2414 | 0.154 / 0.424 | 0.145 / 0.421 | 0.200 / 0.430 | 0.177 / 0.460 |
| **sf_base_reg_ca** (both) | 44.6M | **0.2384** | **0.296 / 0.615** | **0.272 / 0.608** | 0.359 / 0.643 | **0.323 / 0.616** |

Of these, `sf_base_reg_ca` (the regularized 45M Scoreformer with both fixes) was best: best onset
and bar F1 on clean, synthetic and real input, and the lowest validation loss; `s2s_m_ca` a close
second (best on PERiScoPe, 0.367 vs 0.359 onset F1). Its full meter-mode numbers on `real`: note F1
0.909, onset F1 0.323, onset-shift F1 0.503, duration 0.762, staff 0.877, bar F1 0.616, time
signature 0.280. The beat batch below improves on it.

What this shows:
- **The two fixes work together, not alone.** Context notes alone give the best 10M validation loss
  and the best oracle-context scores (clean oracle onset F1 0.578) but no gain once the model writes
  its own context (meter onset F1 0.198 vs 0.202). The metrical head alone helps on transcribed
  input (PERiScoPe bar F1 +0.14, real +0.08) but little on ASAP MIDI. Both together improve onset F1 on
  every test set (+32% on clean, +43% on real for the 10M model): the head learns where beats fall,
  and the extra notes give it enough of the surrounding beat to place them.
- **They turn the 45M model from the worst into the best.** Without them it leaned on the context to
  find the bar and fell behind (`sf_base_reg` onset F1 0.154 on clean); with them its onset F1
  nearly doubles on every test set (0.296 clean, 0.323 real), and bar F1 rises 0.42-0.46 -> 0.61-0.64.
- **The bar-phase gap has narrowed but not closed.** Onset-shift F1 / onset F1 is 1.65 for
  sf_base_reg_ca on clean input (was 2.2 for sf_base_reg), and meter-mode onset F1 is 53% of the
  oracle-context value (0.296 vs 0.562; was 28%). Time signature accuracy (0.28-0.41) is now the
  weakest metric.
- **Validation loss does not rank the systems.** It is teacher-forced with the true context, so it
  measures the oracle mode: `sf_base_reg` has a much lower loss than `s2s` (0.241 vs 0.278) and
  decodes worse in every realistic mode. Compare runs on the decoded test sets.
- **Regularization, not the shorter schedule, fixed most of the 45M model's overfitting**: with
  the original settings, 6 epochs reach the same best loss as 20 (`sf_base_e6` 0.2742 vs
  `sf_base` 0.2754); dropout 0.2, weight decay 0.1 and lr 2e-4 bring it to 0.241. Early stopping
  (the default since) keeps that checkpoint: `sf_base_reg_ca` stopped at epoch 10.9.
- **Training on noise is essential:** `sf_clean` (clean input only) is the worst run on every test
  set (onset F1 0.130 clean, 0.133 real).
- **H1 is not supported at equal size.** `sf_small` (9.7M) has a lower validation loss than `s2s`
  (10.2M) but decodes worse on every test set (meter onset F1 0.177 vs 0.202 on clean).
- **Context noise is mixed:** `s2s_ctx` improves `s2s` on clean input (onset F1 0.223 vs 0.202),
  about equal elsewhere; it hurt `sf_small`.

**Decoding modes.** Guarded decoding beats chained on onset F1 for all 14 models on all four test
sets (56 / 56) and on bar F1 in 50 / 56 (the model's own context, unguarded, carries errors from
window to window: `sf_base_e6` chained bar F1 on real is 0.213, guarded 0.489). The 6 exceptions
are all models with the metrical head, whose own context is good enough that chained bar F1 comes
close or ahead (`s2s_a` on real: 0.595 chained vs 0.553 guarded). Guarded mode replaces 39-52% of
windows with their no-context output. Meter mode adds to it for the models with both fixes: for `sf_base_reg_ca`,
+0.001 to +0.007 onset F1 and +0.005 to +0.014 bar F1 over guarded on every full test set, with
the time signature corrected in 7-9% of windows. Meter is therefore the default in the demo, in
`scripts/audio_to_score.py` and in `scripts/export_musicxml.py --mode auto`.

**Guard margin: 0 is kept.** Swept on `s2s` and `sf_base_reg` (`--extra-margins -0.05,-0.02,0.05`,
`outputs/compare/margins.md`): in meter mode margin 0 has the best onset F1 on all four test sets
for `s2s` and on three of four for `sf_base_reg` (clean: 0.154 vs 0.157 at +0.05). Negative margins
fall back on 73-92% of windows and lower onset F1 everywhere (s2s real: 0.200 at -0.05 vs 0.223);
bar F1 differences between margins are at most 0.023, in both directions.

**Beats as an input (the beat batch, `pace/beats.sh`).** Three runs on the best recipe (45M
Scoreformer, regularized, context notes + metrical head), all three also trained on the
real-transcription split: `sf_base_reg_ca_r` without beats (the control), `sf_base_reg_cab` with
beats, and `s2s_m_cab`, the runner-up's recipe (25.9M Score2Seq) with beats. Meter mode, onset F1 /
bar F1:

| run | beats | clean | synthetic | PERiScoPe | real | real: time sig |
|---|---|---|---|---|---|---|
| sf_base_reg_ca (before) | - | 0.296 / 0.615 | 0.272 / 0.608 | 0.359 / 0.643 | 0.323 / 0.616 | 0.280 |
| sf_base_reg_ca_r (control) | - | 0.293 / 0.603 | 0.267 / 0.597 | 0.396 / 0.651 | 0.307 / 0.572 | 0.229 |
| **sf_base_reg_cab** | yes | 0.319 / **0.635** | 0.305 / **0.623** | **0.441** / 0.695 | 0.350 / **0.639** | **0.352** |
| s2s_m_ca (before) | - | 0.280 / 0.604 | 0.271 / 0.599 | 0.367 / 0.653 | 0.309 / 0.590 | 0.280 |
| s2s_m_cab | yes | **0.333** / 0.621 | **0.314** / 0.620 | 0.436 / **0.713** | **0.363** / 0.600 | 0.309 |

(Their validation losses, 0.2253-0.2346, also average over `val_transcribed`, so they are not
comparable with the runs above.)

- **The beats help on every test set.** Same data and recipe, `sf_base_reg_cab` vs the control:
  onset F1 +0.026 to +0.045, bar F1 +0.027 to +0.067, time signature +0.05 to +0.12 (real: +0.123),
  duration +0.008 to +0.032 on the four full test sets; note F1 unchanged, staff accuracy lower
  (-0.008 to -0.023). The Score2Seq recipe gains as much (`s2s_m_cab` vs `s2s_m_ca`: onset F1 +0.043
  to +0.069), though that comparison also includes the real-transcription data.
- **The real-transcription data alone did not help on ASAP.** The control vs `sf_base_reg_ca` (one
  run each, so differences of a few hundredths can be noise): about equal on clean and synthetic,
  better on PERiScoPe (onset F1 +0.037), worse on real (onset -0.016, bar -0.044).
- **New best model: `sf_base_reg_cab`**, now the demo's default. On `real`, against `sf_base_reg_ca`:
  onset F1 0.350 (+0.027), bar F1 0.639 (+0.023), time signature 0.352 (+0.072), duration 0.785
  (+0.023); onset-shift F1 0.485 (-0.017) and staff 0.849 (-0.028) are lower. Its meter-mode onset
  F1 is 61% of the oracle-context value on real (was 57%). `s2s_m_cab` has the best onset F1 (0.363
  on real) but lower bar F1 (0.600) and time signature accuracy (0.309).
- **The tracker's errors cost nothing; the beats themselves are worth +0.04 onset F1.** On the real
  transcriptions, for `sf_base_reg_cab` (meter mode, onset / bar / time signature):

  | beats | onset F1 | bar F1 | time sig |
  |---|---|---|---|
  | annotated (`real_gt_beats`) | 0.324 | 0.634 | 0.365 |
  | Beat This! (`real`, what the demo gets) | 0.350 | 0.639 | 0.352 |
  | none (`real_no_beats`) | 0.310 | 0.610 | 0.293 |

  Beat This!'s beats are as good as the annotations for this model (`s2s_m_cab`: 0.363 vs 0.355
  onset F1); on real transcriptions it was trained with the tracker's own beats. Better beat
  tracking (Beat This!'s DBN, say) is therefore not the next lever. Without beats, the beat model is
  still as good as the control (onset 0.310 vs 0.307, bar 0.610 vs 0.572), so MIDI input loses
  nothing.
- **The model still uses the beats far less than the beat-snap baseline does.** Even with the
  annotated beats and the true previous bars (oracle mode), `sf_base_reg_cab` reaches onset F1 0.586
  on clean input and 0.585 on the real transcriptions, against the beat-snap + XGBoost baseline's
  0.718 and 0.738 with the same beats (B2, below). The beats are an input feature
  the model weighs, not a grid its onsets are placed on; that gap is where the remaining onset
  errors are.
- **`piece` mode does not beat `meter`.** Over all 17 runs on the four full test sets it is better on
  onset F1 in 22 / 68 cases (mean -0.007) and on bar F1 in 27 / 68 (mean -0.003); time signature
  accuracy is a wash (38 / 68, mean +0.009). Meter stays the default.

**Next: the time-signature batch (`pace/tsearch.sh`).** The decoding modes above on the two best
models (one job per test set), and three runs on the best recipe: `sf_base_reg_cab_s1` (another
seed: how much a single run varies, which every comparison above needs), `sf_base_reg_cabx` (exact
beats everywhere in training, `--beat-noise none --transcribed-beats gt`: can the model use beats
when they are right? its `real_gt_beats` against `sf_base_reg_cab`'s) and `sf_base_reg_cab_c128`
(128 / 64 context notes: more bars to read the meter from).

`real_unseen_audio` (the 3 `real` performances whose audio Transkun never trained on, 62 windows) is
reported in `outputs/compare/` but too small to decide anything (time signature accuracy is 0.000 for
6 runs, and 0.000 to 0.500 between runs).

## v2: grid first, then label every note

The seq2seq models write the score token by token, one 128-note window at a time. The evaluations
above show where that loses: forcing the true time signature lifts onset F1 only from 0.350 to 0.424
on real audio, while forgiving each window a constant shift lifts it to 0.549, close to the 0.577 it
gets with the true previous bars. Most of the remaining loss is each window guessing where in the
bar it starts. And given the true meter, the per-note XGBoost baseline places notes better (onset F1
0.718 clean / 0.738 real) than the seq2seq model does (0.586), while the seq2seq model writes better
durations (0.79-0.87 vs 0.53). v2 combines the two: decide the metrical grid once for the whole
piece, then label every performed note against it.

1. **Grid** (Phase 1: the true one; Phase 2: Beat This! + a meter model): beats with bar numbers and
   time signatures. Each note's position in the bars is read off it (`src/notelabels.py`
   `grid_features`: the bar and position it falls at, Fourier features of that position, the local
   tempo).
2. **Note labeller** (`src/labeller.py`): the Scoreformer encoder (time rotary attention over onsets
   in seconds) with per-note heads instead of a decoder: keep or delete (a transcription error, an
   extra note of a trill), the bar correction (-1 / 0 / +1: notes played just before a bar line), the
   position as the target grammar writes it (`pos_k` + `sub_x`), the written length with ties, staff,
   and grace / arpeggio / ornament flags. One parallel pass per window: no generation, so nothing
   loops, and no window guesses where it starts.
3. **Rendering** (`src/notelabels.py render`): labelled notes back to tokens; a note held across a
   bar line becomes the note + its tied continuation, as scores write it. The same metrics then
   score v2 and every other system.

**Labels** (`scripts/build_notelabels.py` -> `<split>/notelabels.npz`): each window's performed notes
are aligned to its target (the note alignment of `src/metrical.py`, plus grace notes by pitch and
time); paired notes get their score labels, the others "delete". Synthetic noise carries the labels
along (`transfer_rows`: a note the noise adds is a delete), tempo augmentation stretches the grid.

**The label scheme's ceiling** (`scripts/label_ceiling.py`): every window rendered from its own TRUE
labels, i.e. a perfect labeller. On ASAP test windows: onset F1 0.949, duration 0.998, staff 1.000,
time signature 1.000. What it cannot reach is score notes no performed note pairs with (9% of the
sounding notes: mostly notes really not played as written, since 77% of them have no same-pitch
leftover note within 0.3 s), so a better aligner would add about 0.01. 13% of clean-MIDI performed
notes are deletes (trill and ornament notes the score writes once).

**Phase 1** (`pace/labeller.sh`): the labeller with the true grid, the setting the beat-snap
baselines are evaluated in. `nl_base` (8 layers, 25.6M parameters) and `nl_large` (12 layers, d 768,
85.6M). A 3.3M-parameter labeller trained for 60 steps on a laptop CPU already reaches onset F1 0.630
on validation, the beat-snap baseline's level. Results: `scripts/eval_labeller.py` -> mode
`grid_true` in each run's results, tabled by `scripts/compare_runs.py` ("Note labeller (v2)").

## Demo: audio in, score out

A small web app runs the whole system on one recording: Transkun V2 transcribes the audio, the
score model writes the score window by window in meter mode, the best in the evaluation (guarded
decoding: each window's context is the model's own previous output, and degenerate windows are
redone without it; plus time signatures kept consistent along the piece; the page can also switch
to guarded only, plain chained or no context), and the page shows the engraved score with playback
and downloads. The default checkpoint is the best model, `sf_base_reg_cab`, which also takes the
beats Beat This! finds in the audio. The page also offers the time-signature searches ("Meter
search": 3 decodings of the piece, so about 3 times slower).

```bash
pip install fastapi uvicorn python-multipart        # once (also listed in environment.yml)
python demo/app.py --ckpt outputs/runs/sf_base_reg_cab/best.pt
# open http://127.0.0.1:7860 and drop in a WAV / MP3 / FLAC (or a .mid to skip transcription)
```

Everything runs in that Python process, on the GPU if there is one, otherwise the CPU (a
30-second clip: about 10 s of Transkun on a laptop CPU, plus a few seconds per 128-note window for
the score model). The page only uploads and displays: the score is drawn with OpenSheetMusicDisplay
and the MIDI players use html-midi-player, both loaded from a CDN. Each job's files go to
`outputs/demo/<job id>/`: `transcription.mid`, `tokens.txt`, `score.musicxml`, `score.mid`,
`result.json`. The same pipeline from the command line:

```bash
python scripts/audio_to_score.py recording.wav --ckpt outputs/runs/sf_base_reg_cab/best.pt
```

For a model trained with `--beats`, the demo tracks the uploaded audio with Beat This! (`final0`;
`pip install beat-this rotary-embedding-torch einops`, in environment.yml) after transcription and
shows how many beats it found; MIDI input, or no beat-this package, runs it without beats (the
"no beats" input it was also trained on), and the page says so.

The demo needs `vocab.json` (and, for Score2Seq checkpoints, `feature_stats.json`) next to the
checkpoint or in `data/processed/`. Copy both next to `best.pt` when sharing a checkpoint.

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
| 3. Beyer & Dai (2024), pretrained | prior published (ISMIR 2024, the published state of the art for performance MIDI -> score) | Their end-to-end transformer and released checkpoint, run on our test inputs and scored with their metrics, alongside our systems (below). |

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

**Baseline 3: Beyer & Dai, and the system-level comparison** (`pace/bd_setup.sh`, `pace/system_eval.sh`).
Their model writes whole scores, not our windows, and their paper reports MUSTER and score-similarity
error rates, so the comparison is made at the level of whole pieces, with their metrics, for every
system alike:

```bash
bash pace/bd_setup.sh        # once, login node: conda env bd311 (Python 3.11), their code, its forks
                             # (music21 0ed70bb, score_transformer 934a228, muster: compiles C++),
                             # the checkpoint (release v0.0.1, 390 MB); smoke test: MUSTER of a score
                             # against itself is 0
bash pace/system_eval.sh     # inputs, ours + theirs on every input, scoring -> outputs/system_eval/results.md
```

- **Inputs** (`scripts/system_inputs.py`): the 59 ASAP test performances' MIDI (their test set) and
  the same performances transcribed by Transkun V2, shifted onto the ASAP timeline and cut to their
  piece (a MAESTRO recording can hold a prelude and its fugue), each with its ASAP reference score.
- **Ours** (`scripts/system_scores.py`): the demo pipeline on each file, exactly as the demo runs a MIDI
  file (windows cut from the performance alone); a beat model gets the tracked beats on the real
  transcriptions (what the demo gets from audio) and none on clean MIDI (like any MIDI input).
- **Theirs** (`scripts/beyer_dai.py predict`): their `quantize_path` (512-note chunks, 64 overlap), the
  score written with `makeNotation=False` as their code says.
- **Scoring** (`scripts/beyer_dai.py score`): every predicted score is read back and cleaned with their
  `postprocess_score`, then rated with `muster` (pitch / missing / extra / onset / offset / mean error
  %) and `score_similarity_normalized` (deletion / insertion / duration / staff / stem / spelling, per
  reference note), averaged over performances without weighting as their `run_eval.py` does. Tables
  over all performances a system scored and over those every system scored ("common"); their paper's
  Table 3 row sits next to the clean results (MUSTER onset 15.55, mean 11.30; duration 51.86, staff
  6.62), which checks that their model runs here as published (if not: `COMMIT=fc21531 bash
  pace/bd_setup.sh`, their release-tag code).

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
- **Building them on PACE:** `bash pace/real_asap.sh` runs the whole chain in one GPU job
  (`pace/real_asap.sbatch`: fetch the MAESTRO audio of these performances, Transkun V2,
  `build_transcribed.py`, metrical labels, `tests/test_integrity.py`). The splits reuse the
  existing vocabulary and the clean splits' targets, so existing checkpoints work on them
  unchanged.

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
src/dataset.py     PyTorch Dataset / sampler / collate, context noise (ContextNoise)
src/decoding.py                  chained / guarded / meter decoding of whole pieces (evaluation and demo)
src/metrical.py                  per-note downbeat / beat-phase labels (alignment) and the aux loss
scripts/build_metrical.py        precompute the metrical labels of every split (once per data build)
src/pipeline.py                  audio / MIDI -> windows -> chained decoding -> MusicXML
demo/app.py, demo/static/        web demo (FastAPI + one HTML page); scripts/audio_to_score.py: CLI
scripts/train_model.py           train either model; scripts/evaluate_pieces.py: oracle/chained/none
src/eval/plan.py                 evaluation options and mode names; what a run's results still lack
src/beats.py                     beats: ground-truth grid, per-note features, tracker errors (BeatNoise)
scripts/build_beats.py           ground-truth beats per performance -> <split>/beats_gt.npz
scripts/track_beats.py           Beat This! on the MAESTRO audio (held-out folds) -> beats_tracked.npz
scripts/fit_beat_noise.py        tracker vs ground truth -> configs/beat_noise.json
scripts/compare_runs.py          tables of every evaluated run -> outputs/compare/{summary,margins}.md, results.csv
scripts/system_inputs.py         whole-piece test inputs (clean MIDI, cut Transkun transcriptions) + references
scripts/system_scores.py         our pipeline on them -> one MusicXML per performance
scripts/beyer_dai.py             Baseline 3: Beyer & Dai's model (predict) and their metrics for every system (score)
src/notelabels.py                v2: per-note score labels, tie-aware rendering back to tokens, grid features
src/labeller.py                  v2: the note labeller (Scoreformer encoder + per-note heads), prediction -> tokens
scripts/build_notelabels.py      v2: labels + true grids per split -> <split>/notelabels.npz
scripts/label_ceiling.py         v2: every window rendered from its true labels (the labeller's ceiling)
scripts/train_labeller.py        v2: train the note labeller
scripts/eval_labeller.py         v2: evaluate it on the test sets (mode grid_true)
pace/train.sbatch                PACE GPU job: train + evaluate
pace/reevaluate.sh               evaluate finished runs: what is missing, one job per run or per test set
pace/beats.sh                    the beat batch (ground-truth beats, Beat This!, 3 beat runs)
pace/tsearch.sh                  the time-signature batch (search modes on the best models, 3 runs)
pace/bd_setup.sh                 Beyer & Dai's environment, code, forks and checkpoint (once)
pace/system_eval.sh              system-level comparison: ours vs Beyer & Dai on whole pieces
pace/labeller.sh                 v2 Phase 1: labels for every split, then two note labellers (true grid)
pace/train_labeller.sbatch       one note labeller: train + evaluate (resumes from last.pt)
tests/test_scoreformer.py        model checks + chained prompts == training prompts
tests/test_decoding.py           chained / guarded / meter / time-signature search with a stand-in model
tests/test_pipeline.py           demo pipeline end to end: windowing, every mode, output budget, banned tokens
tests/test_context_notes.py      context notes: exact neighbours, unchanged window notes, both models
tests/test_metrical.py           metrical labels: synthetic precision / recall, coverage, the aux head
tests/test_training_stop.py      LR schedule, early stopping, cooldown, resume, status.json, atomic saves
tests/test_eval_plan.py          src/eval/plan.py rules + evaluate_pieces.py --only-missing end to end
tests/test_reevaluate.py         pace/reevaluate.sh (per run / per test set), pace/beats.sh, pace/tsearch.sh: what they submit
tests/test_system_eval.py        system-level comparison: inputs, our pipeline on them, aggregation, job graph
tests/test_notelabels.py         v2: labels, ties, rendering, grid features, dataset, labeller training / evaluation
tests/test_compare_runs.py       the comparison tables: bold best, dashes, alignment, margin order, CSV
tests/test_beats.py              beats vs ASAP annotations, features, tracker errors (bar classes, octave stretches, profiles, fit, vs Beat This!), tracker folds, train / eval / demo
tests/test_context_noise.py      context corruption: off = unchanged, on = only the context changes
scripts/analyze_chained.py       why chained decoding loses notes (from saved predictions)
scripts/simulate_guard.py        guarded decoding estimated offline, per guard margin
scripts/export_musicxml.py       MusicXML from saved predictions (no GPU)
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
