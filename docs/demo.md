# Demo: the details

How the web demo and `scripts/audio_to_score.py` work, and why they leave notes out the way they do. Moved from the main README on 2026-11-24; the README has how to run it.

## Demo: audio in, score out

A small web app runs the whole system on one recording. Transkun V2 transcribes the audio, a
score system writes the score, and the page shows the engraved score with playback and downloads.
Two systems, chosen per upload on the page (each is loaded when its checkpoints exist). The
checkpoints: `models/` (the release, `python scripts/download_models.py`), else the training runs
in `outputs/runs/` (`src/release.py default_path`; `--labeller`, `--meter`, `--ckpt` override).
At start the demo brings `models/` to the Hugging Face repo's current default. It checks again
every hour while running (`--update-every <minutes>`; `--no-update` never checks) and swaps a new
default in between jobs. When it is offline, it runs on what is there.

- **v2 (default):** Beat This! tracks the beats, the meter model decodes the bars and time
  signatures of the whole piece from them, and the note labeller places every note in them
  (`src/pipeline.py V2Pipeline`; `--labeller`, `--meter`). A MIDI upload is rendered to audio for
  Beat This! (`src/synth.py`); `audio_to_score.py --v2 --beats pm2s` takes the beats from the notes
  with PM2S instead (`src/pm2s_beats.py`; the web page still uses Beat This!). The page shows the main time signature, the key signature and how
  many notes it left out as transcription errors. A third or fourth staff is written only when at
  least 2% of the piece's notes are on it (`MIN_EXTRA_STAFF`; in the target scores such staves hold
  28-100% of a piece, or one short passage). Below that the staff is a stray prediction, and its
  notes go on the likeliest of the staves kept: one note of 1,221 had given a demo piece two empty
  staves for all 73 bars.
- **Leaving notes out.** The labeller's keep head marks notes as transcription errors, but its
  "delete" label means the alignment paired no score note with the note. That also covers notes
  really played: extra trill notes, and notes the alignment missed. On clean MIDI, where every
  note was played, 12.7% of notes carry the label. At the head's own threshold (p(delete) > 0.5)
  it leaves out 6.9% of the notes, and 34% of those had a score note. Above 0.8 it leaves out
  1.8% (16% of them score notes), and above 0.95 it leaves out 0.22% (4%). Transkun adds a note
  that was not played for only 0.19% of notes (`configs/noise_stats.json` `halluc_rate`; its main
  error is missing notes, which deletion cannot fix), yet at 0.5 v2 left out 4.6% of the real
  recordings' notes. On whole real pieces, v2's missing notes were its deletions plus the floor
  every system has. The page offers "only when sure" (> 0.95, the default: about as many as
  Transkun adds), "as trained" (> 0.5, as evaluated) and "keep every note" (`demo/app.py DELETE`,
  `audio_to_score.py --delete-above`). Each setting can be scored on whole pieces (`system_scores.py --v2 ...
  --delete-above 0.95`: system `..._d95`, `_keepall` for 1).
- **Written lengths** are tidied (`src/pipeline.py tidy_lengths`, `--no-tidy` to keep them): a
  length running at most a 32nd past the next onset of its staff stops there. On whole pieces this
  lowered MUSTER's error on 41 of 59 clean and 16 of 24 real performances (docs/experiments.md,
  "Fourth round: results").
- **seq2seq:** the Scoreformer writes the score window by window (`--ckpt`; the release's
  `sf_base_reg_cab_c128`, else `outputs/runs/sf_base_reg_cab`). It runs in meter mode by default, the best in its evaluation: guarded
  decoding (each window's context is the model's own previous output, and degenerate windows are
  redone without it), plus time signatures kept consistent along the piece. The page can also
  switch to guarded only, plain chained, no context, or the time-signature searches ("Meter
  search": 3 decodings of the piece, so about 3 times slower).

```bash
pip install fastapi uvicorn python-multipart        # once (also listed in environment.yml)
python demo/app.py                                   # v2 (nl_base_gn + meter_v3) and seq2seq, if present
python demo/app.py --no-tidy                         # v2's written lengths as the labeller gives them
python demo/app.py --labeller outputs/runs/nl_base/best.pt --meter outputs/runs/meter_v2_nonote/best.pt
# open http://127.0.0.1:7860 and drop in a WAV / MP3 / FLAC (or a .mid to skip transcription)
```

Everything runs in that Python process, on the GPU if there is one, otherwise the CPU (a
30-second clip: about 10 s of Transkun on a laptop CPU, plus a few seconds per 128-note window for
the score model). The page only uploads and displays: the score is drawn with OpenSheetMusicDisplay
and the MIDI players use html-midi-player, both loaded from a CDN. A bar moves over the score where the
music is playing. With the score MIDI it follows the score directly, since that plays at one
tempo. With the recording or the transcription (v2), it follows the performer's tempo through the
beats of the grid: `result.json` `follow` maps the performance's seconds to quarters from the first
written note. Each job's files go to
`outputs/demo/<job id>/`: `transcription.mid`, `tokens.txt`, `score.musicxml`, `score.mid`,
`result.json`. The same pipeline from the command line:

```bash
python scripts/audio_to_score.py recording.wav --v2      # v2 (--labeller, --meter)
python scripts/audio_to_score.py recording.wav --ckpt outputs/runs/sf_base_reg_cab/best.pt
```

For a model trained with `--beats`, the demo tracks the uploaded audio with Beat This! (`final0`;
`pip install beat-this rotary-embedding-torch einops`, in environment.yml) after transcription and
shows how many beats it found; MIDI input, or no beat-this package, runs it without beats (the
"no beats" input it was also trained on), and the page says so.

The demo needs `vocab.json` (and, for Score2Seq checkpoints, `feature_stats.json`) next to the
checkpoint or in `data/processed/`. The release has them beside each model
(`scripts/export_models.py` picks the vocabulary each was trained with); to share a checkpoint
by hand, copy both next to `best.pt`.
`train_model.py` and `train_labeller.py` now keep the vocabulary they train with in the run folder
(`src/preprocessing/vocab.py pin_vocab`; a resume on data with other tokens is refused). For a run
trained before that, copy the `data/processed/vocab.json` it trained on (PACE's) next to its
`best.pt`: a local `data/processed/` from another build has other tokens, and the labeller refuses it
(its length classes would mean other note values).
