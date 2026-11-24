# Handoff: where the project stands (2026-11-24)

The course deadline is **2026-12-04** (10 days). This page is for deciding what to do with them:
- what exists;
- how good it is, and how sure we are of that;
- what is still wrong;
- the options, with the evidence and cost of each.

The details behind every number are in [experiments.md](experiments.md).

## In short

- **Status:** the whole system works end to end. A recording goes in, an engraved score comes out,
  in a web demo with playback. Both systems' models are packaged for release on Hugging Face
  (["Models"](#models)).
- **Two systems.** Both read Transkun V2's transcription of the audio:
  - **seq2seq** (the original plan): a transformer writes score tokens, 128 notes at a time.
  - **v2** (since 2026-11-20): a meter model decides the bars of the whole piece from Beat This!'s
    beats, then a note labeller places every note in them.
- **seq2seq is still the better system on the headline metric,** by more than chance. On MUSTER
  mean error, on real recordings / clean MIDI:

  | system | real / clean |
  |---|---|
  | seq2seq | 10.65 / 11.74 |
  | v2 | 12.62 / 14.60 |
  | Beyer & Dai (published baseline) | 11.14 / 11.05 |

  **On real audio, seq2seq already beats Beyer & Dai** (10.65 vs 11.14).
- **v2 is better on several things:** pitch, extra notes, note spelling, and never looping or
  losing its place in the piece. Its gap is rhythm. The meter model often writes half or double
  note values (3/8 for 3/4); the oracle grid shows how much fixing that is worth.
- **Outside the benchmark the order flips.** The benchmark is classical piano (ASAP). On an
  anime piano arrangement (Cruel Angel's Thesis, Animenz: 3,389 transcribed notes, 4/4
  throughout), seq2seq (`sf_base_reg_cab`) went wrong where its design is weakest:
  - it switched time signature 6 times among 4/4, 2/4 and 6/8;
  - it wrote 186 bars where v2 wrote 149, and 3,650 notes for 3,389 played;
  - its guard redid 11 of the 28 windows without context.

  v2 kept 4/4 for the whole piece. The window-by-window model guesses the meter for every window
  (the reason v2 was built), and a long, dense, out-of-domain piece shows that most.
- **Hands-on testing on downloaded songs:**
  - the playback sounds right;
  - the score is still hard to read: rhythm, extra voices;
  - glissandi come out as runs of fast notes;
  - at the trained threshold, real notes were left out. The demo now leaves a note out only when
    the labeller is very sure.

## What exists

```
recording --Transkun V2--> performance MIDI --+--> seq2seq: Scoreformer, window by window --> score tokens --> MusicXML
                                              |
                                              +--> v2: Beat This! beats --> meter model (bars, time signatures,
                                                   whole piece) --> note labeller (every note: position, length,
                                                   staff, flags) --> score tokens --> MusicXML
```

| part | where | state |
|---|---|---|
| data: ASAP + PERiScoPe windows, synthetic noise, real transcriptions | `scripts/build_dataset.py`, [data.md](data.md) | done; rebuilt 2026-11-22 with fixes (ties in chords, pickups, a train/val leak) |
| seq2seq model (Scoreformer, 45M) | `src/scoreformer.py`, `scripts/train_model.py` | done; trained before the rebuild (see "Known caveats") |
| v2 meter model (4.8M) + decoder | `src/meter.py`, `scripts/train_meter.py`, `scripts/eval_meter.py` | done: `meter_v3`, decoder with a tempo prior |
| v2 note labeller (25.6M) | `src/labeller.py`, `scripts/train_labeller.py` | done: `nl_base_gn` |
| whole pipeline | `src/pipeline.py` (`ScorePipeline`, `V2Pipeline`) | done |
| web demo, CLI | `demo/app.py`, `scripts/audio_to_score.py` | done ([demo.md](demo.md)) |
| baselines | beat-snap, XGBoost ([baselines.md](baselines.md)); Beyer & Dai's model (`scripts/beyer_dai.py`) | done |
| whole-piece evaluation | `scripts/system_scores.py`, `scripts/beyer_dai.py score`, `scripts/compare_systems.py` | done |
| release | `scripts/export_models.py`, `upload_models.py`, `download_models.py`, [model_card.md](model_card.md) | built; upload pending (below) |
| tests | `tests/`, one module per part; tiny untrained models, so they run on a laptop | pass (PACE's Python 3.12 and locally) |

## How good it is

Whole pieces, Beyer & Dai's test set, error rates in % (lower is better):
- **real:** 24 performances, transcribed from their recordings by Transkun.
- **clean:** 59 performances as clean MIDI.

| | MUSTER mean real / clean | onset (real) | missing (real) | extra (real) | pitch (real) | spelling (real) |
|---|---|---|---|---|---|---|
| Beyer & Dai | 11.14 / 11.05 | 13.60 | 7.10 | 7.75 | 2.38 | 2.41 |
| seq2seq | **10.65** / 11.74 | **9.90** | 7.57 | 13.48 | 3.06 | 10.92 |
| v2 | 12.62 / 14.60 | 13.45 | 11.89 | 10.66 | **2.22** | 6.37 |

**How sure we are.** Every comparison is paired by performance, with a 95% interval from a
bootstrap over pieces (`scripts/compare_systems.py`):
- **seq2seq vs v2:** seq2seq is better by 1.97 [0.17, 3.34] on real audio and 2.85 [0.70, 4.57]
  on clean. That difference is real.
- **Beyer & Dai vs v2:** Beyer & Dai is better by 1.48 on real audio, but the interval includes 0
  (24 performances of 11 pieces are few). On clean it is better by 3.55 [1.13, 5.46].
- **Run-to-run noise is about ±0.2.** Retraining the labeller with another seed moves the result
  by 0.2. A difference smaller than ~0.3 on the real set needs a second seed before it means
  anything.

**Where v2 loses.** The labeller's onset F1 on PERiScoPe test windows:

| grid given to the labeller | onset F1 |
|---|---|
| the score's true grid | 0.857 |
| the tracked beats with every meter decision right (oracle grid) | 0.697 |
| the meter model's grid (the real system) | 0.488 |

- **The meter model's step costs 0.21.** About 0.15 of it is note values at the other metrical
  level: 3/4 written as 3/8, 2/2 as 2/4.
- **That is a notation choice, not wrong rhythm,** but it makes the score harder to read and
  counts as wrong onsets.
- **The 0.16 from the beat tracker's timing** is mostly beyond our control (Beat This! itself).

## What was tried, and what it found

Round by round, in [experiments.md](experiments.md):

- **Round 1 (v2 design):** with the true grid the labeller places notes far better than seq2seq
  does. The meter model is the bottleneck.
- **Round 2:**
  - adding PERiScoPe helps the meter model;
  - its note encoder overfits, so it was dropped;
  - about half the wrong bars are half/double note values.
- **Data rebuild (2026-11-22):** fixed ties in chords, pickups, and one piece that was in both
  train and validation. A meter-decoder bug that lost bars was fixed too.
- **Round 3:**
  - **Training the labeller on beat-tracker timing** (`--grid-noise`) was the big win. On whole
    pieces it cut missing notes from 34% to 12-15%.
  - **Longer meter training or a larger meter model gained nothing** that held across seeds.
- **Round 4:**
  - **The tempo prior in the meter decoder helps whole pieces:** -0.47 on clean, 48 of 59
    performances better; -0.21 on real, within chance. It does not fix the half/double note
    values.
  - **Keeping more notes:** -0.20 on real; nothing on clean.
  - **Trimming overlapping lengths:** small and consistent.
  - **New labellers** (checkpoint chosen on onset F1) delete more notes, so they are worse. The
    round-3 labeller stays.
  - **Beat ablation:** the recording's beats instead of the rendered MIDI's lower v2's error on
    clean MIDI by **2.8 points**.
- **The seq2seq retrain on the rebuilt data was skipped** (2026-11-24) to put GPU time into v2.
  Its test targets moved by at most 0.0004 onset F1, so its numbers stand, with that caveat.

## Options for the remaining time

Each option has the evidence for it, what it could gain, and a rough cost:
- GPU hours are on PACE, where an H100/H200 or A100 job starts within hours, or in up to 2 days at
  busy times.
- A whole-piece evaluation of one system takes about 30-50 minutes on a GPU, plus about 50
  minutes of CPU for the scoring.

1. **Pick the note-value level per piece** (in progress: see below the list).
   - *What:* decode the grid at the meter model's choice and at half and double note values, then
     keep the version the best evidence supports. Tested so far: tempo and density priors, and the
     labeller's confidence, both negative. A learned written-tempo head trains overnight.
   - *Why:* about 0.15 of the 0.21 onset F1 the meter model costs is exactly this choice. The
     oracle grid shows the labeller is good once the level is right.
   - *Gain:* the largest available for v2's rhythm and readability, possibly most of the gap to
     seq2seq on onsets.
   - *Cost:* 2-4 days of code, no retraining; a few GPU hours to evaluate.
   - *Risk:* realised for the cheap signals (above). The learned head is the last one in reach
     before more data.
2. **Better beats for MIDI input.**
   - *What:* render a MIDI upload with a real piano sound (e.g. FluidSynth and a piano soundfont)
     instead of `src/synth.py`'s additive tone before Beat This!, or track beats from the notes
     directly.
   - *Why:* the ablation shows 2.8 points on clean MIDI, from the beats alone.
   - *Gain:* up to 2.8 on the clean benchmark and for MIDI uploads to the demo. Audio uploads
     already use the recording's beats.
   - *Cost:* 1-2 days, mostly CPU (Beat This! on the 59 renderings), plus one system evaluation.
3. **Decide which system to present as "the model."**
   - *For seq2seq:* it is better on the classical benchmark's headline metric, and on real audio
     it beats Beyer & Dai.
   - *For v2:* it holds up better on music outside the benchmark (the anime arrangement above: one
     time signature, the right bar count), with better spelling and fewer extra notes. The demo
     defaults to it.
   - *A fair framing:* both, with that split. seq2seq wins the classical benchmark; v2 is the
     robust system for what people actually upload.
   - *To make it measurable:* a handful of non-classical pieces with published scores scored the
     same way (1-2 days, if scores with a usable license can be found).
   - *Cost:* none beyond that. A team decision.
4. **Give the seq2seq model v2's grid.**
   - *What:* the seq2seq model is told the true time signature in one evaluation mode (`ts_true`).
     Give it the meter model's bars and time signatures instead.
   - *Why:* forcing the true time signature lifted its onset F1 from 0.350 to 0.424 on real
     audio. A better grid helps it too.
   - *Cost:* 2-3 days (a decoding mode in `src/decoding.py`, plus evaluation).
   - *Risk:* the meter model's time signatures are right on only 44% of real bars, so it may not
     help until option 1 does.
5. **Retrain seq2seq on the rebuilt data.**
   - *Why:* closes the caveat on its numbers. Its training targets moved more than its test ones.
   - *Cost:* about 8-16 GPU hours per run (`pace/v2_jobs.sh seq2seq`; the recipe is
     `sf_base_reg_cab_c128`), then evaluation.
   - *Gain:* probably small.
6. **Readability fixes in the export.**
   - *What:* detect glissandi (a fast run of adjacent notes in one hand) and write them as a
     glissando; group voices and beams more cleanly.
   - *Why:* the hands-on testing; MUSTER barely sees these.
   - *Cost:* 1-3 days each, CPU only, in `src/preprocessing/detokenize.py` / `src/pipeline.py`.
7. **Report the noise.**
   - *What:* a second seed of the meter model through whole pieces, so the report can give a
     spread for v2.
   - *Cost:* about 8 GPU hours plus one evaluation.

8. **An out-of-domain test set.**
   - *What:* a handful of non-classical piano pieces with scores, run through the same whole-piece
     evaluation.
   - *Why:* without it, "more general" cannot be measured. Today the only evidence outside
     classical piano is one anime arrangement, judged by ear.
   - *Sources:* a POP909 split (pop piano arrangements as MIDI with beat, downbeat, key and chord
     annotations: approximate scores from the beat grid), or public-domain scores from PDMX.
     Confirm what each contains, and its license, before building on it.
   - *Cost:* 1-2 days.
9. **More general training data: synthetic performances from public scores.**
   - *What:* take a large set of public piano scores, e.g. PDMX, a public-domain MusicXML set of
     about 250k MuseScore scores with license metadata that should include pop and anime
     arrangements. Render expressive performances from them:
     - tempo curves, timing jitter and velocities, fitted on ASAP's real performances;
     - our fitted Transkun noise;
     - the beat-tracker noise model, or Beat This! on a rendering.
   - *Why:* the labeller needs performances aligned note by note to scores, which barely exist
     outside classical music. Rendering gives exact labels, in any genre, at any amount.
   - *Risk:* rendered timing is easier than a human's. Mix it with the real data, never replace
     it.
   - *Cost:* about a week of pipeline work, then retraining (about a day of GPU for both models).
     The meter model is the cheapest to start with: it needs only beats, downbeats and time
     signatures, so even sources without full scores (beat-annotated pop collections) can feed
     it.
10. **More data for the meter model specifically.** It looks data-limited:
    - the larger model overfit;
    - seeds differ by 0.07 in time-signature accuracy;
    - the note encoder overfit.

    Options 8 and 9 feed it first. Then revisit the note encoder, since the notes say a lot about
    where bars are.
11. **Self-training on unlabelled piano covers.** The system's most confident outputs on
    recordings become training data. Cheap data, but it reinforces the system's own mistakes,
    and the audio is copyrighted. A last resort.

**Option 1, the note-value level, is in progress** (2026-11-24; [experiments.md](experiments.md),
"Fifth round"):
- **The size of the problem:** at the right level on 87% of the ASAP validation performances but
  51% of the test ones. A perfect choice between the three levels would give 96% and 83%.
- **The per-piece choice works mechanically** (`src/meter.py choose_level`, tuned by
  `eval_meter.py --level`).
- **The two cheap signals fail:**
  - priors on the written tempo or note density: 0.80-0.85 on validation, 0.44-0.48 on test;
  - the labeller's confidence: no better than the decoded level.

  The right level depends on the piece, not on what is typical.
- **Overnight:** `pace/v2_level.sh` trains the remaining idea, a learned written-tempo head on the
  meter model (frozen and fine-tuned), and scores it on whole pieces.
  - If it helps, the meter model in the release changes (`src/release.py`, `export_models.py
    --meter`).
  - If not, the level is left to more and more varied training data (options 8-10).

A reasonable split for 10 days: option 3 (a decision), then 1 and 2 in parallel (one person
each), then 6 as time allows. Leave the last 3-4 days for the report and the demo.

## Decisions for the team

- **The main system:** which one the report and demo present as "the model" (option 3).
- **Next steps:** which of the options above fit the time left.
- **The model repo:** public or private on Hugging Face, and whose account or organization hosts
  it.
- **A license** for the code and the weights. The repo has none. MAESTRO, used in the training
  data, is CC BY-NC-SA 4.0.

## Models

What is in the release, and why (`src/release.py`, [model_card.md](model_card.md)):

| folder | trained run | why this one |
|---|---|---|
| `v2_labeller/` | `nl_base_gn` | the best labeller on whole pieces; the round-4 labellers were worse |
| `v2_meter/` | `meter_v3`, with `meter_v3_tp`'s decoder (tempo prior) | the best meter seed; the prior helped whole pieces |
| `seq2seq/` | `sf_base_reg_cab_c128` | the best seq2seq system on whole pieces (10.65 real) |

It runs at p(delete) > 0.95 with lengths tidied. `pace/v2_release.sh` scores exactly that
combination, builds `models/`, then uploads it (pace/README.md, "4m").

**To use the models:**
```bash
python scripts/download_models.py     # needs HF_REPO in src/release.py, or --repo <user>/<name>
```
The demo and the CLI pick up `models/` on their own. On a training machine they fall back to
`outputs/runs/`.

## Known caveats and gotchas

- **The seq2seq model was trained before the data rebuild.** The rebuild kept the old token ids
  (`build_dataset.py --vocab`). The release ships the vocabulary the model was trained with: its
  run folder's, else the old data's (`data/processed_pre1122/vocab.json` on PACE).
  `export_models.py` checks that it fits.
- **A model needs its own `vocab.json`.** A vocabulary from another data build gives other token
  ids. The labeller refuses one that does not fit; the seq2seq model refuses one of another size.
  Runs trained since 2026-11-23 keep theirs in their folder.
- **PACE GPU requests** must name one node feature. PACE adds `&cuda13`, and an OR list with it
  waits forever as `BadConstraints`:
  - `HX00` for H100/H200;
  - `cuda13` for any NVIDIA GPU with bf16 (A100, A40, L40S, H100, H200).

  pace/README.md, "Which GPU".
- **Job limits on PACE:**
  - more than about 10 jobs submitted at once per user are rejected, so every batch script stays
    at 8 or fewer;
  - an 8-hour limit per job, so long trainings resume from `last.pt`.
- **Windows:** `tar` cannot create the symlinks some run folders have on PACE (`meter_v3_tp` links
  `meter_v3`'s checkpoint). The error is harmless.
- **Where things live on PACE:**
  - the repo: `~/scratch/dl-proj-2026`;
  - training runs: `outputs/runs/<name>/` (`best.pt`, `last.pt`, `config.json`, `status.json`,
    logs, evaluations);
  - whole-piece results: `outputs/system_eval/`;
  - Beyer & Dai's code and environment: `~/scratch/beyer_dai` (conda env `bd311`).
- **Results come back as small packs.** Each round's pack command is in pace/README.md; nothing
  large is copied by default.

## Where to start reading

1. **[README](../README.md):** what it is, the results, how to run it.
2. **[experiments.md](experiments.md), "Third round: results" and "Fourth round: results":** the
   current numbers and what they mean.
3. **[pace/README.md](../pace/README.md):** how every batch runs. Each round is one script.
4. **`src/pipeline.py`:** the whole system in one file's worth of calls.
