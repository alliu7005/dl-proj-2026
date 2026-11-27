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
  many notes it left out as transcription errors, and where it wrote glissandos. A third or fourth staff is written only when at
  least 2% of the piece's notes are on it (`MIN_EXTRA_STAFF`; in the target scores such staves hold
  28-100% of a piece, or one short passage). Below that the staff is a stray prediction, and its
  notes go on the likeliest of the staves kept: one note of 1,221 had given a demo piece two empty
  staves for all 73 bars.
- **Glissandos** (`src/glissando.py`, on by default, `--no-glissando` to turn off): a fast one-way
  run on the white keys (or the black keys) is written as its first and last note joined by a
  "gliss." line, both where the beat grid puts them (to the sixteenth). The labeller still gets
  every note and the run's middle is taken out of what it writes, so the rest of the piece is
  written exactly as with glissandos off: on the user's song, only the two glissandos' bars differ.
  An end with more than three ledger lines is written under 8va / 15ma (8vb / 15mb under the bass).
  A glissando is one hand's, so its last note goes on the first note's staff when it reads there
  within two ledger lines (down to A3 under the treble, up to E4 over the bass). A wider sweep keeps
  its line across the staves. The MusicXML has a `<glissando>`, which MuseScore draws.
  The page's viewer (OpenSheetMusicDisplay 1.9) draws none, so the page labels it "gliss." and draws
  a line when both ends are on one staff. The score's MIDI plays the sweep.
- **The timing check** (`src/timing.py`, on by default, `--no-timing-check` to turn off): a note
  the labeller wrote on a triplet position but played on a straight eighth or sixteenth is written
  there.
  - It applies only in meters whose beat is a quarter, and only when the timing is clear: within
    0.06 of a beat of the straight position, at least 0.1 of a beat from the triplet one.
  - Swung notes (between 2/3 and 3/4 of the beat) stay as the labeller wrote them.
  - On the user's song it moved 89 notes. In bars 64-71, where a window had read straight eighths
    as triplets, triplets went from 40% to 20% and eighths from 7% to 23%.
- **Which rules run is decided by the benchmark.** The rules that correct a model's output (the
  timing check, the whole-beat check, the short-bar fix) and the beat-level decoding are on by
  default (`src/release.py`), and the release manifest can turn them off:
  - `scripts/rule_verdict.py` (`pace/v2_decode2.sh`) keeps a correction rule only if it makes the
    whole system significantly better on real recordings and worse nowhere, and keeps the decoding
    (the model side) unless it makes it worse.
  - `publish_model.py --settings` writes the decisions into the manifest. The demo reads it at
    start and at its hourly check (a settings change needs no download) and logs the rules it runs.
  - `--no-timing-check`, `--no-timing-beats`, `--no-fix-jumps` and `--no-beat-decode` still turn a
    rule off whatever the manifest says. The engraving rules (lengths, stems, glissandos, grace
    notes) are not the models' job and are not decided this way.
- **Beat-level decoding of the labeller** (`src/beat_decode.py`, on by default,
  `--no-beat-decode` to turn off): instead of taking each note's likeliest split of the quarter on
  its own, the labeller's own probabilities are combined with when the note was played, and
  optionally one family of splits (straight or triplet) is kept per beat.
  - The settings are tuned on validation data for each labeller (`scripts/tune_beat_decode.py`,
    `beat_decode.json` next to the checkpoint, published with it). Without that file the labeller
    decides as before, so the demo gets it once the tenth round has tuned and published it.
  - On ASAP's validation set alone it gained 0.1 point of position accuracy. It could not fix the
    song's bars 64-71: the labeller writes those notes as triplets with a median confidence of
    67,000 to 1, so the whole-beat rule below still does that work.
- **Whole beats checked** (`src/timing.py` check_beats, on by default, `--no-timing-beats` to turn
  off): the timing check judges one note at a time, which missed the song's bars 64-71.
  - Each bar there repeats a syncopation played at 1.5, 1.71, 2.24 and 2.73: the straight 16th
    figure (16th, 8th, 8th, 16th), as bar 14 writes it. The labeller wrote it as triplets.
  - No single note was clear enough: 2.24 is 0.09 from 2 1/3, and 1.71 is halfway between the two
    grids.
  - So each beat where the labeller wrote a triplet is judged as a whole: are the notes played in
    it nearer the straight 16ths or the triplets (the sixths for a run written in sextuplets)?
    - The beat is measured from where its on-beat notes were played (bars 67 and 69 came 0.05
      early).
    - When the straight grid is nearer and every note is within 0.1 beat of a 16th, the whole beat
      is written straight.
  - Bars 64-71 went from 20% triplets to none, and the whole song from 4.5% to 2.2%. Played
    triplets and the sextuplet run in bar 136 stay.
  - A swung long-short (0.67-0.73 of the beat) becomes a dotted eighth and a sixteenth, as in bar 14.
- **Readable bars** (`src/readable.py`, on by default; `--no-fix-jumps` and `--no-readable` turn
  each off):
  - **Short bars:** a bar the meter model stretched is shortened instead.
    - The meter model's decoder sometimes gives a bar's beats less than a beat's time, so a bar
      that was played short (two beats) became a whole 4/4 bar of long notes and rests.
    - The user's bars 47 and 51 were like this: 0.46 s and 0.66 s against 0.47 s beats.
    - The rule: a beat step that packs its quarters into less than half the time its neighbours
      take gets the beats its time holds. Its bar loses the rest and gets its own time signature,
      keeping two beats at least, so both bars are now 2/4 and every note in them is written.
    - A long gap at the usual tempo (beats the tracker missed) is not touched.
  - **Lengths:** a rest of a 16th or less before the next note of the same staff, in the same bar,
    is closed. A note running up to a 16th past the next one stops there.
    - This happens only when the new length is one plain, dotted or triplet note, and the note
      neither starts on a triplet and ends on a straight 16th nor the other way round.
    - On the user's song: 188 rests closed and 24 overlaps trimmed.
  - **Voices:** where a staff has two voices in a bar, the higher voice's stems go up and the
    other's down. The rests that fill out the lesser voice are hidden.
    - Before, both voices' stems pointed the same way and the viewer drew them over each other: in
      bar 29, half-note chords under a melody looked like 32nd beams.
    - 114 bars of the song had this.
  - **Two voices to a staff at most, ordered by pitch** (`src/preprocessing/detokenize.py
    _two_voices`). The writer used to put each chord into the first voice free when it started,
    whatever its pitch, so a held note under new chords opened a third voice, voice 1 could be
    lower than voice 3, and two voices could get stems down (the user's "unlasting" song, bar 4:
    three treble voices, the up-stem one partly below the two down-stem ones).
    - Now a chord goes into the free voice nearer in pitch unless that crosses the other voice; a
      chord that starts among held notes joins them, and when no voice is free it shares the
      nearer one, the held notes split and tied. The upper voice's stems go up, the lower's down.
    - A pitch struck again while it is held ends there (one key cannot sound twice).
    - On that song: bars with three or four voices 17 -> 0, two voices stemmed the same way
      17 -> 0, stems crossing 47 -> 6, a pitch struck while held 19 -> 0; it sounds the same
      (`tests/test_readable.py`). The score MIDI comes from the old writer, which plays a tied
      note of a chord once (music21's MIDI export strikes it again).
  - Closing rests had cost duration accuracy against ASAP's scores before (`tidy_lengths`), so the
    benchmark measures both rules (`pace/v2_decode.sh`) before they count as more than a demo
    default.
- **Click to play from there:** a click anywhere on the score sends the player playing, else the
  one last used, else the score's, to that moment, and plays on. The bar jumps there at once.
  - The position comes from the line clicked and the cursor steps around the click.
  - The recording and the transcription go there through the beat grid.
  - A MIDI player takes a new time only when stopped, so it is stopped, moved and started again.
- **The score, played back** (`score.mid`, `src/playback.py`): grace notes sound 70 ms each just
  before their main note, and a rolled chord spreads upwards, 35 ms a note, across both staves
  when the roll runs through both. music21's MIDI export alone had left some grace notes out, and
  struck others with their main note and held them for seconds; it played rolled chords as blocks.
- **Grace notes** before one note are written as a run, each its own note in the order played:
  unslashed sixteenths for two or more, a slashed eighth for one (`src/preprocessing/detokenize.py`).
  They had been written as one grace chord. The tokens to MusicXML to tokens round trip now matches
  exactly on 0.880 of 300 validation windows (0.860 before) and 0.747 of 300 test windows (0.733).
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

`python scripts/check_env.py` loads each package the demo needs in its own process and says what is
missing, failing, or blocked (exit code 0 / 1 / 2).

### Windows: files blocked by Application Control

The error is `ImportError: DLL load failed while importing ...: An Application Control policy has
blocked this file` (or `[WinError 4551]`). Windows 11's **Smart App Control** (on by default on
some new installs) refuses to load programs and libraries it cannot verify, and Python packages
ship many unsigned ones (`.dll`, `.pyd`). An organisation's own policy (a school or work computer)
gives the same message. The project itself has no programs of its own (it is Python source), so
what is blocked is a package's library, and which one depends on the machine. `scripts/check_env.py`
lists them, with the exact paths from Windows' Code Integrity log; Windows Security > App &
browser control > Protection history shows them too.

The ways around it, the first changing nothing in Windows' security:

1. **Run it in WSL** (Linux inside Windows). The policy checks Windows programs, not Linux ones; an
   NVIDIA GPU works there through the Windows driver, and the page opens in the Windows browser.
   In PowerShell as administrator, once (then restart, and Ubuntu asks for a Linux user name):
   ```powershell
   wsl --install -d Ubuntu
   ```
   Then in the Ubuntu terminal (keep the repo in the Linux home folder, not under `/mnt/c`, which
   is much slower):
   ```bash
   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && bash Miniforge3-Linux-x86_64.sh -b && ~/miniforge3/bin/conda init bash && exec bash
   git clone https://github.com/alliu7005/dl-proj-2026.git && cd dl-proj-2026 && conda env create -f environment.yml && conda activate dl-proj && pip install --no-deps transkun==2.0.1 && python scripts/check_env.py && python scripts/download_models.py
   python demo/app.py      # then open http://localhost:7860 in the Windows browser
   ```
2. **Turn Smart App Control off** (Windows Security > App & browser control > Smart App Control
   settings). It is a protection against unknown programs, so this is the computer owner's call;
   read the warning Windows shows (on some versions of Windows it cannot be turned back on without
   resetting Windows).
3. **A managed computer:** the policy belongs to its IT department, which can allow the files or
   say whether WSL is allowed; otherwise use another computer.

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
