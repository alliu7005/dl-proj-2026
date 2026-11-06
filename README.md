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
| 2. check targets survive tokens → MusicXML → tokens | `python -m tests.test_roundtrip --split val` | ~2 min |
| 2b. check window cutting (stitching, prompts) | `python -m tests.test_windows --n 40` | ~3 min |
| 3. *(already done; output in `data/noise_stats.json`)* fit the noise model on real transcriptions | see [Noise model](#noise-model) | ~1 h with a GPU |
| 4. *(optional)* build the real-transcription eval split | `python scripts/build_transcribed_eval.py --asap-split test` | ~1 min |

`data/` is git-ignored. Share `data/noise_stats.json` with the team (it's small), or re-run step 3.

## Output format

`data/processed/` contains:

```
vocab.json                 343 target tokens incl. <sep> (+ training counts)
splits.json                piece -> train/val/test
report.json                per-performance stats, skips, and dropped segments
train/  val/  test/        one directory per split, each holding:
  notes.npy   float32 [total_notes, 4]   pitch, onset (s, rel. to segment start; < 0 = carried over), duration (s), velocity
  pedal.npy   float32 [total_pedal, 2]   sustain pedal (on, off) in seconds, rel. to segment start
  tokens.npy  int16   [total_tokens]     prompt + target token ids (no <bos>/<sep>/<eos>)
  index.npy   int64   [n_segments, 7]    note_lo, note_hi, pedal_lo, pedal_hi, tok_lo, tok_hi, tgt_lo
  meta.jsonl  one line per segment       piece, performance, win, s_lo, s_hi, start_pos, bar_start, ...
test_transcribed/          same layout; inputs are REAL Kong transcriptions, targets = clean test targets
```

Segment `i`'s notes are `notes[index[i,0]:index[i,1]]`, and likewise for pedal. Its prompt is
`tokens[tok_lo:tgt_lo]` and its target `tokens[tgt_lo:tok_hi]` (`SegmentStore.get` / `get_prompt`).
The arrays are memory-mapped, so nothing loads into RAM until it's used.

| split | windows | performances | compositions |
|---|---|---|---|
| train | 35,873 (48–192 notes, 50% overlap) | 779 | 166 |
| val | 2,203 (~128 notes, no overlap) | 79 | 16 |
| test | 1,725 (~128 notes, no overlap) | 59 | 14 |
| test_transcribed | not rebuilt yet for windows (needs data/transcribed) | | |

### Loading for training

```python
from torch.utils.data import DataLoader
from src.dataset import ScoreSegmentDataset, collate
from src.preprocessing.noise import NoiseModel

train = ScoreSegmentDataset("data/processed/train", "data/processed/vocab.json",
                            noise=NoiseModel.from_json("data/noise_stats.json"))
dl = DataLoader(train, batch_size=32, sampler=train.sampler(), collate_fn=collate, num_workers=4)
for epoch in range(E):
    train.set_epoch(epoch)                   # fresh (reproducible) noise every epoch
    for b in dl: ...                         # notes [B,N,8], note_mask [B,N], target [B,T] (pad = 0),
                                             # loss_mask [B,T]: train only on target tokens, not the prompt

loss_w = train.vocab.loss_weights(ornament_weight=5.0)   # -> nn.CrossEntropyLoss(weight=...)
val = ScoreSegmentDataset("data/processed/val", "data/processed/vocab.json")          # clean inputs
real = ScoreSegmentDataset("data/processed/test_transcribed", "data/processed/vocab.json")
```

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

Parameters are **fitted on real Kong transcriptions** of 40 MAESTRO-*test* recordings (172k notes).
Kong's model was trained on MAESTRO train, so only test-split audio gives realistic errors.

| error | measured |
|---|---|
| missing notes | 4.5% for single notes → 20% in 8-note chords |
| hallucinated notes | 2.3% of notes; mostly ±3–4 and ±1–2 semitones (partly wrong-pitch substitutions), then octaves; as loud as real notes |
| onset jitter | 6.5 ms std |
| notes lengthened to pedal release | 65% of notes released under sustain |
| missed pedal presses | 9%, with ~80 ms timing error |

On 8-second windows of those recordings, synthetic noise gives pitch F1 0.961 vs. 0.970 for the real
transcriptions, with 96.2% vs. 95.8% of notes kept. The synthetic noise is therefore slightly harsher
than real.

To reproduce the fit:

```bash
uv venv --python 3.12 .venv-transcribe       # Kong's package needs Python <= 3.12 + CUDA torch
uv pip install --python .venv-transcribe/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-transcribe/Scripts/python.exe piano_transcription_inference audioread librosa pretty_midi
# checkpoint (the package's own download uses wget, which Windows lacks):
curl -L -o ~/piano_transcription_inference_data/"note_F1=0.9677_pedal_F1=0.9186.pth" \
  "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"

python scripts/fetch_maestro_audio.py --split test --limit 40    # range-downloads from the 108 GB zip (~3 GB)
python scripts/fetch_maestro_audio.py --asap-split test          # audio for the real-transcription eval
.venv-transcribe/Scripts/python scripts/transcribe.py            # ~1.5 recordings/min on an RTX 5070 Ti
python scripts/fit_noise.py --pairs data/transcribed/pairs.csv --out data/noise_stats.json
```

## Real-transcription eval split: caveat

Only 25 of the 59 test performances have MAESTRO audio:

| MAESTRO split of the audio | performances | segments | meaning |
|---|---|---|---|
| train | 21 | 901 | Kong's model was trained on this audio, so transcriptions are optimistic |
| validation | 2 | 126 | unseen by Kong's model |
| test | 1 | 51 | unseen by Kong's model |

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

## Not done yet: PianoCoRe

PianoCoRe ([repo](https://github.com/ilya16/PianoCoRe), [paper](https://doi.org/10.5334/tismir.333))
already contains ASAP. When adding it:

- send its scores through the same `load_score` / `tokenize_bars`;
- map its pieces to the same composition IDs, so no Beyer & Dai test piece enters training;
- de-duplicate performances that also appear in ASAP;
- sample per score (it has about 100 performances per score).

The shard → vocab → memory-mapped flow in `build_dataset.py` already scales to it.
