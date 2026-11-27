---
tags:
- music
- piano
- music-transcription
- sheet-music
- musicxml
- symbolic-music
library_name: pytorch
pipeline_tag: other
---

# Piano score models (dl-proj-2026)

These models turn a piano performance into a written score. The performance comes as MIDI, for
example Transkun V2's transcription of a recording; the score comes out as MusicXML. They run with
the project's code. The repo has a web demo (`demo/app.py`) and a command-line tool
(`scripts/audio_to_score.py`).

```bash
git clone <the project's repo> && cd dl-proj-2026
conda env create -f environment.yml && conda activate dl-proj
python scripts/download_models.py                  # dat-dev/scoreformer -> models/, checked against manifest.json
python demo/app.py                                 # http://127.0.0.1:7860: a recording in, a score out
                                                   # (it pulls a newer default by itself: at start, then hourly)
python scripts/audio_to_score.py recording.wav --v2
```

## The models

| folder | what | parameters | trained run |
|---|---|---|---|
| `v2_labeller/` | **v2 note labeller.** A transformer encoder over a window of performed notes, with per-note heads: keep or delete, bar correction, position in the bar, written length, staff, grace / arpeggio / ornament flags. | 25.6M | `nl_gq` (step 40,000): half of its training windows from Gangqinpu's pop, folk, jazz and classical scores |
| `v2_meter/` | **v2 meter model.** A transformer over the piece's tracked beats (Beat This!) predicting bar starts, beat lengths and time signatures. A Viterbi search then decodes one consistent grid for the whole piece. `decode.json` holds its tuned decoder, including a prior on the written tempo. | 4.8M | `meter_v4_gq` (step 56,000): trained with Gangqinpu's pop, folk and jazz performances as half of every batch |
| `seq2seq/` | **The seq2seq score model (Scoreformer).** An encoder-decoder that writes score tokens window by window, with beats as an input. | 44.9M | `sf_base_reg_cab_c128` |

- **The folders change as training goes on.** `manifest.json` says which run each folder holds
  (`models.<folder>.source_run`, its test scores, and `history`). A newly trained run replaces
  one when it scores at least as well on the test sets both were scored on
  (`scripts/publish_model.py`). Every published run stays under `runs/<run>/`, and every new
  default is tagged `<date>-<folder>-<run>`.
- **v2** = Beat This! -> `v2_meter` (the grid of the whole piece) -> `v2_labeller` (every note
  placed in it). It runs with: notes left out only when the labeller's p(delete) > 0.95, and
  written lengths tidied (`manifest.json` `settings`).
- **Each folder has the model's vocabulary** (`vocab.json`), the one it was trained with. A
  vocabulary from another data build gives other token ids, and the code refuses it.
- **`manifest.json`** lists every file with its sha256.
- **The checkpoints** hold the weights and the model config only (no optimizer state).

## Training data

- **ASAP**, from Beyer & Dai's corrected fork: performances aligned to their scores, with beat
  annotations. The clean MIDI gets synthetic transcription noise fitted on Transkun V2's real
  errors.
- **Real Transkun V2 transcriptions** of the ASAP performances whose audio Transkun never trained
  on (MAESTRO's test and validation audio).
- **PianoCoRe's PERiScoPe subset**: Transkun V2 transcriptions aligned note by note to scores.
- **Gangqinpu** (Wayne42/gangqinpu, CC BY-NC-SA 4.0, the meter model): web piano recordings of
  pop, folk, jazz and classical arrangements, Transkun V2 transcriptions aligned to their scores.
  Its training split only; its test split holds every score close to one of our test pieces.
- **Test split:** Beyer & Dai's, 14 pieces / 59 performances. 24 of the performances have a real
  recording.

## Evaluation

Whole pieces, scored with Beyer & Dai's metrics (MUSTER and score similarity). Error rates in %,
lower is better. MUSTER mean on real audio (24 performances, transcribed by Transkun) / clean MIDI
(59):

| system | MUSTER mean real / clean |
|---|---|
| Beyer & Dai (ISMIR 2024, their checkpoint) | 11.14 / 11.05 |
| seq2seq (`seq2seq/`, meter decoding) | 10.65 / 11.74 |
| v2 (`v2_meter` = `meter_v4_gq` + `v2_labeller`, at these settings) | 12.70 / 15.18 |
| v2 with the classical-only meter model (`meter_v3`, release v2-round4) | 12.31 / 14.53 |

- **Outside classical piano** (58 Gangqinpu test pieces, mostly pop, transcribed from their
  recordings; MUSTER mean): this v2 (`nl_gq` + `meter_v4_gq`) 4.00, seq2seq 7.69, Beyer & Dai
  8.07, v2 with the classical-only models 6.18. On ASAP, v2 with these models: 12.71 / 14.86.
- **Why the default meter model is the Gangqinpu one:** people bring songs, not only classical
  pieces. On Gangqinpu's real recordings, `meter_v4_gq` gets the time signature right 0.79 of the
  time against 0.54 for `meter_v3`, and the note-value level 0.75 against 0.65. On the classical
  meter scores the two are equal. On whole classical pieces it is a little worse: +0.39 on real
  recordings (within chance), +0.65 on clean MIDI (outside chance). `meter_v3` stays available as
  release tag `v2-round4`.

- **Comparing the two systems on real audio** (v2 with `meter_v3`):
  - v2 is better on pitch (2.26 vs 3.06), extra notes (11.80 vs 13.48) and spelling (6.58 vs
    10.92).
  - The seq2seq model is better on onsets (9.90 vs 13.82), offsets and missing notes.
- **Outside classical piano:** on an anime piano arrangement in 4/4 throughout, seq2seq switched
  time signature 6 times and wrote 37 extra bars, while v2 kept 4/4. One piece, judged by ear.
- **Window level (PERiScoPe), v2:** onset F1 0.488. With the true metrical grid it would be
  0.857, so most of v2's error is the meter model's grid.
- The project's `docs/experiments.md` has every round.

## Limitations

- **Note values at the wrong metrical level.** The meter model often writes a whole piece at half
  or double values: 3/8 for 3/4, 2/4 for 2/2. It gets the level right for 64% of test
  performances, and time signature accuracy per bar is 0.44 on real recordings and 0.64 on
  PERiScoPe. Choosing the level afterwards did not help (round 5).
- **MIDI input is tracked through a rendering.** The beats then come from the MIDI rendered to
  audio, which tracks worse than a recording: v2's error on clean MIDI is 2.8 points higher than
  with the recording's beats.
- **Glissandi and very fast runs** come out as many short notes.
- **Repertoire:** solo piano, mostly classical. The training scores are ASAP's composers and
  PERiScoPe's pieces.
- **Not checked:** other instruments, other transcribers (beyond a Kong et al. robustness test),
  and pop or jazz notation.

## License

No license has been chosen for these weights yet. The training data's own terms apply: MAESTRO
is CC BY-NC-SA 4.0, and ASAP and PianoCoRe have their own terms. Check those before any use
beyond research.
