# dl-proj-2026: Noise-Robust Piano Score Reconstruction

Audio → **Transkun V2** transcriber (Yan & Duan, ISMIR 2024) → noisy performance MIDI →
seq2seq transformer → score tokens → MusicXML.

Training data:
- **ASAP**: clean Disklavier MIDI with hand-checked beat annotations, plus synthetic
  transcription noise fitted on Transkun's real errors.
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
| 1. build train/val/test/test_periscope segments | `python scripts/build_dataset.py` | ~3 min ASAP; PianoCoRe ~1–2 h with 6 workers (raise `workers` on a bigger machine) |
| 2. check targets survive tokens → MusicXML → tokens | `python -m tests.test_roundtrip --split val` | ~2 min |
| 2b. check window cutting (stitching, prompts) | `python -m tests.test_windows --n 40` | ~3 min |
| 3. fit the noise model on real Transkun transcriptions | see [Noise model](#noise-model) | ~30 min with a GPU |
| 4. *(optional)* build the real-transcription ASAP eval split | `python scripts/build_transcribed_eval.py --asap-split test` | ~1 min |

`data/` is git-ignored. Share `data/noise_stats.json` with the team (it's small), or re-run step 3.

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

- **Vocabulary:** `data/processed/vocab.json`, 343 tokens. Fixed ids: `<pad>`=0, `<bos>`=1,
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
| `test_transcribed` | ASAP test with Transkun transcriptions of MAESTRO audio (optional build) | real Transkun output | small real-audio check; most of this audio was in Transkun's training data |

Every item's `meta.jsonl` line has `dataset` (`asap`/`pianocore`) and `source` (`clean` /
`transkun_real`); the Dataset adds synthetic noise only to `source == "clean"` items.

### Loading

```python
from torch.utils.data import DataLoader
from src.dataset import ScoreSegmentDataset, collate
from src.preprocessing.noise import NoiseModel

V = "data/processed/vocab.json"
train = ScoreSegmentDataset("data/processed/train", V, noise=NoiseModel.from_json("data/noise_stats.json"))
dl = DataLoader(train, batch_size=32, sampler=train.sampler(), collate_fn=collate, num_workers=4)
for epoch in range(E):
    train.set_epoch(epoch)                   # fresh (reproducible) noise every epoch
    for b in dl: ...                         # b["notes"], b["note_mask"], b["target"], b["loss_mask"]

loss_w = train.vocab.loss_weights(ornament_weight=5.0)    # -> nn.CrossEntropyLoss(weight=..., reduction="none")
val = ScoreSegmentDataset("data/processed/val", V)                       # no noise, no prompt dropout
test = ScoreSegmentDataset("data/processed/test_periscope", V)
```

- **Noise:** `noise=` turns on training mode: synthetic noise on clean (ASAP) items and 10% prompt
  dropout. Leave it out for val/test.
- **Oversampling:** `sampler()` draws windows with grace notes, arpeggios or ornaments 3× as often.

## On-disk format

`data/processed/` contains:

```
vocab.json                 343 target tokens incl. <sep> (+ training counts)
feature_stats.json         mean/std of the continuous note features (train)
splits.json                composition -> split, for ASAP folders and PianoCoRe works
report.json                per-performance stats, skips, and dropped windows
train/ val/ test/ test_periscope/   one directory per split, each holding:
  notes.npy   float32 [total_notes, 4]   pitch, onset (s, rel. to window start; < 0 = carried over), duration (s), velocity
  pedal.npy   float32 [total_pedal, 2]   sustain pedal (on, off) in seconds, rel. to window start
  tokens.npy  int16   [total_tokens]     prompt + target token ids (no <bos>/<sep>/<eos>)
  index.npy   int64   [n_windows, 7]     note_lo, note_hi, pedal_lo, pedal_hi, tok_lo, tok_hi, tgt_lo
  meta.jsonl  one line per window        dataset, source, piece, performance, win, s_lo, s_hi, start_pos, ...
test_transcribed/          same layout; inputs are REAL Transkun transcriptions, targets = clean test targets
```

Window `i`'s notes are `notes[index[i,0]:index[i,1]]`, and likewise for pedal. Its prompt is
`tokens[tok_lo:tgt_lo]` and its target `tokens[tgt_lo:tok_hi]` (`SegmentStore.get` / `get_prompt`).
The arrays are memory-mapped, so nothing loads into RAM until it's used. Read through the Dataset,
not the raw arrays: it also drops released carried-over candidates (below) and scales the features.

Expected sizes (full PERiScoPe download; `report.json` has the exact counts after a build):

| split | ASAP | PERiScoPe |
|---|---|---|
| train | 779 performances, 166 compositions → 35,873 windows (48–192 notes, 50% overlap) | ~85% of 1,200 works, ~5.7k performances → roughly 250k windows (estimate) |
| val | 79 performances, 16 compositions → 2,203 windows (~128 notes) | ~5% of works |
| test | Beyer & Dai: 59 performances, 14 pieces → 1,725 windows | none (Beyer & Dai test works are excluded from PERiScoPe) |
| test_periscope | none | ~10% of works, ~670 performances |

The ASAP counts are from a full build. The PERiScoPe rows are estimates from the download
selection; the per-performance alignment checks (`tests/test_windows.py --source pianocore`) passed
on all 25 sampled performances, with a median window input-vs-target pitch F1 of 0.975.

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
    with no pedal, so the model still has to infer some ties from the music.

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
  window to the next. 10% of training items drop the prompt, so the first window works without one.
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
- **Vocabulary:** structural tokens are a fixed grammar. Only `sub_*` and `d_*` come from training data.

Positions and durations are exact fractions, snapped to common denominators (2, 3, 4, …, 16, 24,
32, 48).

To go back from tokens to MusicXML (for MUSTER/MV2H evaluation):
`src.preprocessing.detokenize.tokens_to_musicxml(tokens, "out.musicxml")`.

## Pipeline details

1. **Split by composition, before segmenting.** We use Beyer & Dai's exact split
   (`configs/splits/beyer_dai.json`, regenerated by `scripts/make_bd_split.py`).
   - Their 14 test pieces / 59 test performances are reproduced exactly, along with their list of
     69 badly aligned performances, which we exclude.
   - Pieces not in their list are hash-split into train or val only.
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
5. **Quality filters.** These dropped 216 of about 40k windows:
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

Parameters are **fitted on real Transkun V2 transcriptions** of 40 MAESTRO-*test* recordings
(170k notes). Transkun was trained on MAESTRO train, so only test-split audio gives realistic errors.
The noise is applied only to clean (ASAP) training items; PERiScoPe items are real Transkun output.

| error | Transkun V2 (measured) | Kong 2021, for reference |
|---|---|---|
| missing notes | 3.7% for single notes → 15% in 8-note chords | 4.5% → 20% |
| extra (hallucinated) notes | 0.23% of notes, mostly ±1–4 semitones | 2.3% |
| onset jitter | 5.9 ms std | 6.5 ms |
| notes lengthened to pedal release | 12% of notes released under sustain | 65% |
| missed pedal presses | 12%, with ~80 ms timing error | 9% |

To reproduce the fit (≈30 min on an RTX 4050; each recording runs in its own process and falls
back to the CPU if the GPU runs out of memory):

```bash
python scripts/fetch_maestro_audio.py --split test --limit 40    # range-downloads from the 108 GB zip (~3 GB)
python scripts/fetch_maestro_audio.py --asap-split test          # audio for the real-transcription eval
python scripts/transcribe.py                                     # Transkun V2, ~5x faster than real time
python scripts/fit_noise.py --pairs data/transcribed/pairs.csv --out data/noise_stats.json
```

## Real-transcription eval split: caveat

Only 25 of the 59 ASAP test performances have MAESTRO audio (segment counts: rebuild with
`build_transcribed_eval.py`):

| MAESTRO split of the audio | performances | segments | meaning |
|---|---|---|---|
| train | 21 | – | Transkun was trained on this audio, so transcriptions are optimistic |
| validation | 2 | – | unseen by Transkun |
| test | 1 | – | unseen by Transkun |

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
src/dataset.py     PyTorch Dataset / sampler / collate
scripts/           build_dataset, make_bd_split, fetch_maestro_audio, transcribe, fit_noise,
                   build_transcribed_eval
tests/test_roundtrip.py
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
