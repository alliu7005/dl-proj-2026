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

# Piano score models (dl-proj-2026, release v2-round4)

These models turn a piano performance into a written score. The performance comes as MIDI, for
example Transkun V2's transcription of a recording; the score comes out as MusicXML. They run with
the project's code. The repo has a web demo (`demo/app.py`) and a command-line tool
(`scripts/audio_to_score.py`).

```bash
git clone <the project's repo> && cd dl-proj-2026
conda env create -f environment.yml && conda activate dl-proj
python scripts/download_models.py                  # this repo -> models/, checked against manifest.json
python demo/app.py                                 # http://127.0.0.1:7860: a recording in, a score out
python scripts/audio_to_score.py recording.wav --v2
```

## The models

| folder | what | parameters | trained run |
|---|---|---|---|
| `v2_labeller/` | **v2 note labeller.** A transformer encoder over a window of performed notes, with per-note heads: keep or delete, bar correction, position in the bar, written length, staff, grace / arpeggio / ornament flags. | 25.6M | `nl_base_gn` (step 18,000; training timing jittered like the beat tracker's) |
| `v2_meter/` | **v2 meter model.** A transformer over the piece's tracked beats (Beat This!) predicting bar starts, beat lengths and time signatures. A Viterbi search then decodes one consistent grid for the whole piece. `decode.json` holds its tuned decoder, including a prior on the written tempo. | 4.8M | `meter_v3` (step 38,000), decoder of `meter_v3_tp` |
| `seq2seq/` | **The seq2seq score model (Scoreformer).** An encoder-decoder that writes score tokens window by window, with beats as an input. | 44.9M | `sf_base_reg_cab_c128` |

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
| v2 (`v2_meter` with its tempo-prior decoder + `v2_labeller`) | 12.62 / 14.60 |

- **Comparing the two systems on real audio:**
  - v2 is better on pitch (2.22 vs 3.06), extra notes (10.66 vs 13.48) and spelling (6.37 vs
    10.92).
  - The seq2seq model is better on onsets (9.90 vs 13.45) and offsets.
- **Window level (PERiScoPe), v2:** onset F1 0.488. With the true metrical grid it would be
  0.857, so most of v2's error is the meter model's grid.
- The project's `docs/experiments.md` has every round.

## Limitations

- **Note values at the wrong metrical level.** The meter model often writes half or double
  values: 3/8 for 3/4, 2/4 for 2/2. Time signature accuracy per bar is 0.44 on real recordings
  and 0.64 on PERiScoPe.
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
