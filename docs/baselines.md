# Baselines

The window-level baselines (beat-snap, XGBoost) and the note-level error detector. Beyer & Dai's whole-piece system (Baseline 3) is in [experiments](experiments.md) ("Third round"). Moved from the main README on 2026-11-24 (unchanged).

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
  file (windows cut from the performance alone). A seq2seq beat model gets the tracked beats on the
  real transcriptions (what the demo gets from audio) and none on clean MIDI (like any MIDI input),
  or with `--midi-beats synth` Beat This!'s beats of the MIDI rendered to audio (system
  `<run>_<mode>_sb`). v2 (`--v2 <labeller>+<meter>`, system `v2_<labeller>_<meter>`) always gets
  beats: the tracked ones on real, the rendered MIDI's on clean (`scripts/synth_beats.py`). A piece
  whose score cannot be written keeps its tokens and error (`<slug>.tokens.txt`, `.error.txt`).
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
