# Experiments: every round, what it tried and what it found

The seq2seq models (Score2Seq, then the Scoreformer), then v2 (a metrical grid for the whole piece, then a label for every note), round by round, in the order they ran. Each round's plan is followed by its results. Moved from the main README on 2026-11-24 (unchanged up to "Fourth round: results", which is new); [README](../README.md) has the summary and [the handoff](handoff.md) what is open.

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

**Phase 2: the meter model** (`src/meter.py`, `pace/meter.sh`): the grid from the tracked beats and
the notes of a whole piece, so the labeller needs nothing from the score. Beat This!'s beats are
close to the notated ones but not the same (double tempo on 18% of pieces, triplet level on some,
12% missed, 13% extra, downbeats marked almost independently per beat), so the model works on the
TRACKED beats. A transformer over the piece's beat sequence (4.75M parameters) predicts for each:
whether a bar starts on it, how many quarter notes it lasts until the next tracked beat (1/4 to 4:
this absorbs a doubled or tripled tempo and missed or extra beats), and its bar's time signature,
from per-beat note features (where notes fall inside the beat, how loud / low / long the notes on
it are, the harmony and its change across it, Beat This!'s own downbeat flag, the beat's length)
and, since the second round, a note encoder pooling every note of the beat (below). A
Viterbi search then picks the most probable CONSISTENT grid over (time signature, position in the
bar): bars add up to their time signature, which changes only at a bar line. Given the true answers
it reproduces the annotated grid (validation: beat / downbeat F 0.996, time signature 1.000).

- **Training data** (`scripts/build_meter_data.py`): every aligned ASAP performance (756 train / 102
  val / 59 test) with its annotated grid (beats with quarter positions, bar starts, time and key
  signatures), the tracker's errors simulated anew each time (the validated BeatNoise) and synthetic
  transcription noise on the notes; plus the real-transcription performances with Beat This!'s own
  beats; since the second round also PERiScoPe (below).
- **First signal** (1,200 steps on a laptop CPU, batch 8): validation downbeat F 0.615 against the
  tracker's own 0.578 on the same beats, beat F equal (0.808 vs 0.807), time signature 0.45.
- **Evaluation** (`scripts/eval_meter.py`): every test performance's grid scored against the
  annotations (beat F, downbeat F, time signature) next to the tracker's own numbers; the grids are
  saved with a key signature estimated from the notes (Krumhansl-Kessler profiles: 74% of ASAP
  pieces' signatures exact, 76% counting 7 sharps = 5 flats, against the seq2seq model's 0.41).

**Phase 3: the whole v2 system** (`scripts/eval_labeller.py --grid meter:<run>`): the labeller on the
meter model's grid. Each window takes its bars, time signatures and key from the decoded grid
(`window_frame`: its first bar is the bar the grid puts its first note in), so nothing comes from
the score: mode `grid_meter_<run>`, next to `grid_true` in the comparison tables.

### v2 results, first round: the labeller works, the grid is the bottleneck

Onset F1 (the labeller trained 26-60 minutes):

| test set | nl_base, true grid | nl_large, true grid | XGBoost, true grid | nl_base, meter_base grid | nl_large, meter_base grid |
|---|---|---|---|---|---|
| clean | 0.771 | 0.767 | 0.718 | 0.293 | 0.316 |
| synthetic | 0.738 | 0.738 | | 0.291 | 0.312 |
| periscope | 0.851 | 0.853 | | | |
| real | **0.797** | 0.789 | 0.738 | 0.263 | 0.287 |

Given the right grid the labeller is the best system on every test set (bar F1 0.98, duration
0.85-0.87). On the first meter model's grid it falls to the seq2seq models' level: bar F1 0.54,
time signature 0.26-0.35, and with notes landing where the grid makes them implausible the keep
head deletes more (note F1 0.90 -> 0.77-0.86). `meter_base` itself (`eval_meter.json`):

| test set | grid beat F | tracker beat F | grid downbeat F | tracker downbeat F | time signature |
|---|---|---|---|---|---|
| clean (simulated tracker) | 0.844 | 0.814 | 0.680 | 0.618 | 0.394 |
| real (Beat This!) | 0.663 | 0.711 | 0.617 | 0.596 | 0.322 |

Where that comes from:
- **Not the decoder.** Given perfect head outputs on the simulated tracker's beats, the Viterbi
  decoding gets beat F 0.882, downbeat F 0.895 and time signature 0.992 on the clean test set: the
  ceiling the tracked beats allow is far above 0.68 / 0.39. The heads are the limit.
- **Too few pieces.** ASAP's 756 training performances are 156 pieces. A meter model trained on them
  alone overfits within ~2k steps on a laptop: validation time-signature accuracy peaks at
  0.56-0.58 and falls to 0.40-0.47 while training continues, step accuracy with it (0.69 -> 0.59).
  It learns the pieces, not meter.
- **A small, lopsided test set.** 59 performances of 14 pieces: Chopin's Ballade No. 1 (6/4 with
  4/4 passages) is 18 of them, Ravel's Ondine (48 time-signature changes in 94 bars) 7. "Always
  4/4" gets 0.28 of the test bars right (0.38 of train, 0.54 of val). Both the per-performance and
  the per-piece means are reported from now on.
- **Real transcriptions got synthetic noise on top** in training (the note noise was applied to
  every performance, not only clean MIDI). Fixed.

### Meter model, second round

- **6x the pieces.** PERiScoPe has no beat annotations, but every window is aligned to its score,
  so the score's beats timed through the alignments are a ground truth (`scripts/build_meter_data.py`:
  window grids joined by global bar number, one grid per run of consecutive bars; the windows'
  notes stitched back into the performance). Checked against ASAP's human annotations on the ASAP
  test performances, built the same way: the same time signature for 99.97% of beats and the same
  bar structure, beat times within 70 ms for 81% (beat F 0.81, downbeat F 0.80: a beat no note
  falls on is interpolated). That adds 971 training works (5,260 performances of real Transkun
  output, the tracker simulated) to ASAP's 156, 78 validation works and a test set of 104 works no
  model has seen (`periscope`, scored against the alignment beats).
- **A note encoder.** Besides the hand-made beat features, every note of a beat (its phase in the
  beat and sin / cos of it at 1-12 cycles per beat, length in beats, velocity and velocity against
  the beat's mean, lowest / highest of its chord, chord size, height above the beat's bass, a pitch
  embedding) goes through a small MLP and is pooled per beat (mean and max). Plus the beat's
  pitch-class profile.
- **Augmentation**: each training sample stretched in time (x0.85-1.18; notes, ground-truth and
  tracked beats together), transposed (+-6) and velocity-scaled, so a piece is not recognised by
  its tempo and pitches.
- **Longer chunks** (256-1,024 beats, close to whole pieces at test time), **piece-balanced
  sampling** (a piece played 20 times is not drawn 20 times as often), **model selection** on what
  the labeller needs (validation downbeat F + time-signature accuracy, per source: ASAP, real,
  PERiScoPe), and a **tuned decoder** (`eval_meter.py --tune`: bar / time-signature head weights and
  the two penalties chosen on validation; the untuned decoder's scores are kept next to it). The
  decoder is now vectorised (8.5x faster, identical paths, tested against a plain Viterbi).
- **Ablations** (`pace/meter.sh`): `meter_v2` (everything), `meter_v2_s1` (seed 1), `meter_v2_nonote`
  (no note encoder), `meter_v2_asap` (no PERiScoPe), and `meter_base` re-evaluated with a tuned
  decoder and on the PERiScoPe test. `nl_base` labels every meter model's grids, `nl_large` the best
  candidates', on clean, synthetic, real and periscope.

**Local check** (ASAP only, 0.8M-parameter models, 3-4k steps on a laptop CPU; best validation
checkpoint of each, untuned decoder):

| run | val downbeat F | val time sig | selection score | later |
|---|---|---|---|---|
| first design (beat features) | 0.642 | 0.564 | 0.603 | falls (time sig 0.47 at 3k) |
| + note encoder | 0.622 | 0.579 | 0.600 | collapses (time sig 0.40 at 2k) |
| first design + augmentation | 0.647 | 0.534 | 0.591 | stable |
| **note encoder + augmentation** | **0.663** | **0.580** | **0.621** | stable (0.606 at 4k) |

Augmentation is what stops the memorising, and the note encoder only pays with it. The tuned
decoder adds to every model (validation 0.622 -> 0.663 for the last; it weighs the bar head up, the
time-signature head down and makes changes of time signature rarer). On the ASAP test set these
small ASAP-only models reach meter_base's downbeat F (0.679-0.683 tuned) but not a better time
signature (0.33-0.47), and the confusions say why: 3,071 of the wrong bars are the 6/4 Ballade
written as 6/8, then 4/4 as 6/8 or 2/2, 2/4 as 4/4. 6/4 and 6/8 group the beats the same way
at half the note values: the same sound, so which one the score uses is a convention only many
pieces can teach. That is what the PERiScoPe works are for; the PACE runs measure it.

### Meter model, second round: results

PACE, up to 30,000 steps (batch 16, chunks of up to 1,024 beats), each run's best validation
checkpoint with its tuned decoder (`outputs/compare/summary.md`, "Meter models (v2)" and "Note
labeller (v2)"). The PERiScoPe test is the one to read: 576 performances of 91 works no model
trained on, against ASAP's 59 performances of 14 pieces. The last two columns are `nl_base` on each
grid:

| run | data | note encoder | downbeat F | time sig | time sig per piece | labeller onset F1 | labeller bar F1 |
|---|---|---|---|---|---|---|---|
| tracked beats alone | | | 0.606 | | | | |
| meter_base | A+R | no | 0.639 | 0.365 | 0.425 | 0.273 | 0.621 |
| meter_v2_asap | A+R | yes | 0.629 | 0.393 | 0.471 | 0.243 | 0.632 |
| meter_v2 | A+R+P | yes | 0.679 | 0.513 | 0.589 | 0.285 | 0.706 |
| meter_v2_s1 | A+R+P | yes (seed 1) | 0.680 | 0.511 | 0.628 | 0.316 | 0.712 |
| **meter_v2_nonote** | A+R+P | no | **0.747** | **0.604** | **0.708** | **0.353** | **0.754** |
| true grid | | | | 1.000 | | 0.851 | 0.980 |

On the ASAP test, `meter_v2_nonote` against `meter_base`: clean (simulated tracker) downbeat F 0.775
vs 0.718, time signature 0.550 vs 0.416; real (Beat This!) 0.624 vs 0.638 and 0.384 vs 0.315.

- **More works help.** With the note encoder, PERiScoPe's works add 0.05 downbeat F and 0.12
  time-signature accuracy on the PERiScoPe test (`meter_v2` vs `meter_v2_asap`), and 0.05
  time-signature accuracy on clean ASAP.
- **The note encoder overfits at this scale.** Both seeds peaked at 4,000 steps and stopped early
  at 14,000. Without it the model was still improving at 26,000 of 30,000 steps, and it wins on
  every test set: PERiScoPe downbeat F 0.747 vs 0.679, time signature 0.604 vs 0.513. The local
  ASAP-only check (4,000 steps) was too short to see the overfitting. From the third round
  `--no-note-encoder` is the default.
- **Seeds:** the PERiScoPe test barely moves between seeds (time signature 0.513 vs 0.511), but
  the 14-piece ASAP test does (clean 0.446 vs 0.559, real 0.266 vs 0.340). On ASAP, differences
  under about 0.1 are seed noise.
- **The tuned decoder** adds 0.004-0.08 to the validation selection score. It helps the weakest
  model most (`meter_v2_asap` 0.551 -> 0.630; its real time signature 0.147 -> 0.250). For the
  best model it is within 0.01 of the default.
- **What is left: half and double note values.** `meter_v2_nonote` has about 34,600 of
  PERiScoPe's ~57,000 test bars right. About 12,000 of the wrong ones are the same grouping at half
  or double the note values: 3/4 written 3/8 (4,723), 2/2 as 2/4 (3,111), 3/8 as 3/4 (2,360), 2/4
  as 2/2 (846), 6/4 as 6/8 (644). Beat and downbeat F cannot see these (the beats and bar lines fall
  at the same times), and the score reads the same at the other note value. But the window
  metrics count every note in such a bar as wrong. The model writes 3/8 almost only where it is
  wrong: 140 bars right, 2,360 of the 3/8 bars written 3/4, and 4,723 bars of 3/4 written 3/8. The
  note value of a tracked beat is its weakest decision.

**v2 against the seq2seq system** (onset F1 / bar F1 / time signature; the seq2seq model in meter
mode, what its demo runs):

| test set | v2: nl_base on meter_v2_nonote | seq2seq: sf_base_reg_cab_c128 | v2 with the true grid |
|---|---|---|---|
| clean | 0.333 / 0.671 / 0.477 | 0.348 / 0.689 / 0.446 | 0.771 / 0.981 / 1.000 |
| periscope | 0.353 / 0.754 / 0.637 | 0.492 / 0.754 / 0.569 | 0.851 / 0.980 / 1.000 |
| real | 0.295 / 0.606 / 0.300 | 0.357 / 0.681 / 0.391 | 0.797 / 0.981 / 1.000 |

On PERiScoPe, v2 now matches the seq2seq system's bar lines and gets more time signatures right,
but places fewer notes exactly. Note F1 shows a second loss (0.75 vs 0.93 on PERiScoPe): the
labeller, trained only on true grids, deletes notes that a tracked-beat grid puts slightly off
where it expects them.

### Third round: the whole system, end to end

`pace/v2.sh` (pace/README.md, 4k) runs every part of this round, in two waves of at most 8 jobs (PACE's per-user limit; several models share one GPU job, `pace/v2_jobs.sh`):

- **Meter models without the note encoder, trained longer:** up to 60,000 steps, early stopping
  after 8 evaluations without a gain. Each run is two chained 8-hour jobs: `--max-hours` stops one
  cleanly, and `--resume` continues it. There are three seeds (`meter_v3`, `_s1`, `_s2`) and one
  larger model (`meter_v3_large`: d 384, 8 layers, 14.2M parameters against 4.8M).
- **The oracle grid** (`eval_meter.py --oracle`, run `meter_oracle`): no model. Every head is
  certain of its training target on the same tracked beats, decoded like a model's output: the
  best grid those beats allow. The labeller on it separates what the tracker's beats cost from what
  the meter model's errors cost. On the local ASAP test with the simulated tracker it gets beat F
  0.884, downbeat F 0.901 and time signature 0.992 (the first round's check, now a saved grid).
- **A labeller trained with the tracker's timing** (`train_labeller.py --grid-noise 0.8`,
  `src/notelabels.py GridNoise`). The grid the system gives the labeller is anchored on tracked
  beats, not on the beats as played. So 80% of training windows get the fitted Beat This! errors on
  their true grid, one performance profile per window:
  - tempo octaves (anchors added at exact in-between positions, or every k-th kept);
  - missed beats (the grid interpolates across them);
  - spurious beats at the fitted phases, their position snapped to a meter step, so the grid bends
    locally as a meter model's does;
  - offset and jitter.

  The bars and positions stay the same, so the labels still hold. Validation gets one fixed draw,
  so model selection measures the labeller as it is used. `nl_base_gn` and `nl_base_gn_s1` (seed 1)
  are each evaluated with `nl_base` on every grid: the oracle, `meter_v2_nonote` and the four new
  meter models.
- **Onset-scale F1** (`src/eval/metrics.py`): onset-shift F1 with the predicted note values as
  written, all halved, or all doubled, whichever is best. It is the window version of beat
  tracking's "allowed metrical levels", and close to what MUSTER does (its onset error forgives a
  global scaling). It separates a notation convention from rhythm that is really wrong.
  `scripts/rescore.py` adds it to every run's stored results from the saved predictions, so the
  seq2seq models get it without decoding again. It updates a mode only when the re-scored onset F1
  equals the stored one. A result scored against the targets before the rebuild may differ by up to
  0.005; it then gets every window metric re-scored against the current targets.
- **v2 as a system** (`src/pipeline.py V2Pipeline`, used by the demo and the whole-piece
  comparison). It takes the piece's beats, the meter model's grid of the whole piece with its tuned
  decoder, and a key signature from the notes. The labeller runs on each window cut as the test
  sets are, in its stretch of the grid. Every kept note is then rendered once, in the piece's own
  bars (a window's frame records its first bar), as one score: no window guesses where it starts,
  and nothing is stitched.
- **Beats for a MIDI input:** Beat This! hears audio only, so a MIDI file is rendered first by a
  small additive piano synthesizer (`src/synth.py`: about a second for a 10-minute piece, no extra
  packages). For the clean ASAP test MIDI, `scripts/synth_beats.py` tracks every piece with the
  Beat This! fold model that never trained on it. It records the beat F on the rendered MIDI next
  to the same performances' real audio, so the cost of the rendered audio is on record.
- **The whole-piece comparison with Beyer & Dai** (their MUSTER and score-similarity metrics,
  `scripts/beyer_dai.py score --systems all`): six v2 systems (labeller x meter model), and the two
  best seq2seq models with the same rendered-MIDI beats on clean MIDI (`sf_base_reg_cab_meter_sb`,
  `sf_base_reg_cab_c128_meter_sb`), so the clean comparison is like for like. These sit next to
  Beyer & Dai and the seq2seq systems already scored.
- **Score export, four fixes found on the way:**
  - Playback tempo (`pipeline.playback_qpm`): the score MIDI the demo plays had no tempo, so music21
    played every score at 120 quarter notes per minute. A 4-minute slow piece played in about 2
    minutes. It now gets the tempo at which the written score sounds as long as the performance
    (first note to last); a v2 and a seq2seq score of a 233 s recording now play for 230 s and
    232 s. The MusicXML stays without a tempo mark, like the scores the comparison rates.
  - Speed: without its own time signature, a measure made music21 guess one from its contents for
    every rest it filled. Dense or tuplet-grid output took seconds per measure (an 80-note piece:
    32 s), and now takes 0.5 s. Every measure carries its time signature while it is built, and
    repeated ones are removed before writing.
  - Spelling (`detokenize.spell`): the tokens carry MIDI pitches only, and every black key used to
    be written as a sharp, even in flat keys (a piece in five flats came out full of C#, F# and
    naturals). Notes are now named for the bar's key signature: the spelling the signature implies
    if there is one (Db and Gb in five flats, Cb in seven, E# in six sharps), else its direction.
    This changes Beyer & Dai's spelling error for every system of ours.
  - A last resort: the bars that cannot be exported even alone (or that break the export in
    context) are emptied, the rest is written, and the result says which. A piece whose score still
    cannot be written keeps its tokens and its error next to where the score would be.

  The roundtrip check (tokens -> MusicXML -> tokens) gives the same numbers before and after. The
  batch writes the seq2seq systems' scores again with this export (`sysre_*`), so every system in
  the comparison is spelled the same way and the four real pieces they had no score for get one.

Results: the next section.

### Third round: results

Every model on the rebuilt data (`pace/rebuild.sh`), meter results with the fixed decoder, labellers
with the current evaluation (`outputs/compare/summary.md`, `outputs/system_eval/results.md`). The
seq2seq window results below are from before the rebuild (see the last point).

**Meter models.** Downbeat F / time signature per bar, on the three test sets:

| run | clean ASAP (simulated tracker) | real ASAP (Beat This!) | PERiScoPe |
|---|---|---|---|
| tracked beats alone | 0.618 / - | 0.596 / - | 0.608 / - |
| meter_v2_nonote (round 2) | 0.775 / 0.550 | 0.624 / 0.384 | 0.752 / 0.611 |
| meter_v3 | 0.790 / 0.552 | 0.651 / **0.440** | **0.773 / 0.643** |
| meter_v3_s1 | 0.739 / 0.570 | 0.621 / 0.347 | 0.719 / 0.598 |
| meter_v3_s2 | **0.803** / 0.570 | **0.663** / 0.385 | 0.765 / 0.573 |
| meter_v3_large (14.2M) | 0.714 / 0.565 | 0.624 / 0.404 | 0.693 / 0.548 |
| oracle grid (no model) | 0.907 / 0.992 | 0.880 / 0.998 | 0.912 / 0.991 |

- **Longer training alone gains nothing that holds across seeds.** The three seeds average 0.752
  downbeat F and 0.605 time signature on PERiScoPe, the same as `meter_v2_nonote` (0.752 / 0.611).
  Between seeds the time signature moves by 0.07 (0.573-0.643), and by up to 0.09 on the 24 real
  performances. `meter_v3` is the best seed and the one the systems use. Read it as a lucky draw of
  the same model, not a better recipe.
- **The larger model overfits.** At 14.2M parameters it is worse on every test set (PERiScoPe
  0.693 / 0.548).
- **The beats are not the limit.** With every head right on the same tracked beats (the oracle),
  0.99 of the bars have the right time signature. The remaining errors are the meter model's own,
  mostly half and double note values (3/4 written 3/8, 2/2 as 2/4, 6/4 as 6/8).

**Where the labeller loses notes** (`nl_base_gn` on PERiScoPe; note F1 / onset F1 / onset-scale F1):

| grid | nl_base (round 1) | nl_base_gn (tracker timing in training) |
|---|---|---|
| true grid (the score's beats) | 0.924 / 0.854 / 0.855 | 0.932 / 0.857 / 0.858 |
| oracle grid (tracked beats, every meter head right) | 0.739 / 0.548 / 0.609 | **0.920 / 0.697 / 0.788** |
| meter_v3's grid (the real system) | 0.756 / 0.391 / 0.583 | **0.922 / 0.488 / 0.730** |

- **Training with the tracker's timing (`--grid-noise 0.8`) works.** On grids from tracked beats
  the labeller stops deleting notes: PERiScoPe note F1 0.756 -> 0.922, on ASAP clean 0.844 -> 0.922
  and real 0.775 -> 0.906. Onset F1 rises 0.391 -> 0.488 (clean 0.345 -> 0.407, real 0.315 ->
  0.382). On the true grid it loses nothing (0.857 vs 0.854). The second seed (`nl_base_gn_s1`)
  gives the same numbers to 0.002. `nl_large` without grid noise reaches 0.427 on `meter_v3`: size
  does not replace training on the right grids.
- **The decomposition** (onset F1 on PERiScoPe):
  - true grid -> oracle grid, the tracker's timing: 0.857 -> 0.697;
  - oracle -> meter_v3, the meter model's errors: 0.697 -> 0.488.

  In onset-scale F1, which forgives a uniform halving or doubling of the note values, the second
  step is 0.788 -> 0.730. Of the 0.21 the meter model costs, 0.15 goes away once a uniform halving or
  doubling is forgiven: note values at the other metrical level, a notation convention rather than
  wrong rhythm.
- **On the windows v2 now matches or beats the seq2seq system.** Each cell is onset F1 / bar F1 /
  time signature; the seq2seq model runs in meter mode:

  | test set | v2: nl_base_gn on meter_v3 | seq2seq: sf_base_reg_cab_c128 |
  |---|---|---|
  | clean | **0.407 / 0.713 / 0.509** | 0.348 / 0.689 / 0.446 |
  | periscope | 0.488 / **0.780 / 0.670** | **0.492** / 0.754 / 0.569 |
  | real | **0.382** / 0.644 / 0.349 | 0.357 / **0.681 / 0.391** |

  v2 is ahead on clean ASAP and PERiScoPe (time signature +0.06 and +0.10). On the 24 real
  performances the seq2seq model has the better bar lines.

**Whole pieces, Beyer & Dai's metrics** (error rates in %, lower is better; clean = the 59 clean
ASAP test performances, real = the 24 transcribed from audio):

| system | MUSTER mean clean / real | pitch | missing | extra | onset | spelling (sim) |
|---|---|---|---|---|---|---|
| Beyer & Dai | 11.05 / 11.14 | 3.21 / 2.38 | **7.54 / 7.10** | **6.48 / 7.75** | 14.43 / 13.60 | **8.58 / 2.41** |
| seq2seq sf_base_reg_cab_c128 (meter, rendered-MIDI beats on clean) | 11.74 / **10.65** | 3.87 / 3.06 | 8.57 / 7.57 | 13.46 / 13.48 | **12.80 / 9.90** | 12.21 / 10.92 |
| v2 nl_base_gn_s1 + meter_v3 | 14.92 / 12.63 | **2.82 / 2.27** | 15.72 / 11.89 | 9.71 / 10.18 | 18.27 / 13.79 | 11.52 / 6.31 |
| v2 nl_base_gn + meter_v3 | 15.07 / 12.83 | 2.85 / 2.34 | 15.29 / 11.96 | 10.49 / 10.73 | 18.22 / 13.81 | 11.69 / 6.30 |
| v2 nl_base + meter_v3 | 21.07 / 19.75 | 3.08 / 2.49 | 34.49 / 34.00 | 11.52 / 11.82 | 22.27 / 18.40 | 9.32 / 4.42 |

- **Training with the tracker's timing matters even more for whole pieces.** It cuts MUSTER
  missing from 34% to 12-15%, and the mean from 21.1 to 15.1 (clean) and 19.8 to 12.8 (real).
- **v2 is not yet the better whole-piece system.** On real audio it trails the seq2seq system by
  2.0 (12.63 vs 10.65) and Beyer & Dai by 1.5. It has the lowest pitch error of all systems, fewer
  extra notes than seq2seq, and better spelling than seq2seq. Its losses are missing notes and
  onsets.
- **The missing notes are v2's own deletions plus a floor every system has.** Beyer & Dai keeps
  every note and still scores 7.1-7.5% missing: notes MUSTER's alignment does not match. v2's
  summaries record what the labeller deleted: 4.6% of the notes on real, 7.9% on clean. Floor plus
  deletions gives 11.7 and 15.4, against 12.0 and 15.3 measured. Per piece they go together (r =
  0.76 on real). The worst piece deletes 26% (Rachmaninoff op. 23/4 on clean); otherwise the high
  missing rates are pieces every system finds hard (Ondine, Gondoliera).
- **Beats from rendered MIDI are poor.** Clean MIDI has no recording, so its beats come from Beat
  This! on the MIDI rendered by `src/synth.py`, for v2 and for the `_sb` seq2seq systems alike. On
  the 24 performances that also have a real recording, the rendered MIDI gets beat F 0.576 /
  downbeat F 0.422; the recording gets 0.709 / 0.592 (`outputs/system_eval/beats_synth.json`). That
  is likely why v2 deletes more on clean (7.9%) than on real (4.6%). It would also explain why its
  clean MUSTER is 2.3 worse than its real one: the seq2seq system's is 1.1 worse, and Beyer & Dai's,
  with no beats at all, 0.1 better. The ablation below measures it.
- **The seq2seq window numbers predate the rebuild.** Their models were trained on the targets
  before the 2026-11-22 fixes (chord ties, pickups), and the window results were scored against
  them. Scored against the rebuilt targets, their onset F1 changes by at most 0.0004
  (`scripts/rescore.py`, every mode of every run). The test targets barely moved, so the tables
  hold. The training targets moved more; the seq2seq model is not retrained on them (fourth round:
  the GPU time goes to v2), which is a caveat on its numbers.
  `compare_runs.py` marks results scored against the old targets "(old targets)". `rescore.py` now
  re-scores those against the current ones (and adds their onset-scale F1, which it had refused
  because of that 0.0004).

### Fourth round: rhythm and the written score

Not retraining the seq2seq model on the rebuilt data: its test targets moved by at most 0.0004
onset F1 (above), so the comparison stands with that caveat, and the GPU time goes to v2.
`pace/v2_round4.sh` (pace/README.md, 4l), one batch of 8 jobs:

- **The meter model's note value** (~0.15 of the 0.21 onset F1 the meter model costs on
  PERiScoPe). A prior on the written tempo each step implies, in the decoder (`src/meter.py
  tempo_logprior`). Between two tracked beats the decoder chooses a step in quarters, and a step
  of 1/2 and one of 1 differ by a factor of 2 in quarters per minute. The prior is log-normal,
  centred on the training scores (median 122, log-mean 116 quarters per minute over 756 ASAP
  performances, log-std 0.51: a factor of 2 is ~1.4 standard deviations). Its weight and centre are
  tuned on validation with the decoder's other settings (`eval_meter.py --tune --tempo-prior`); at
  weight 0 it decodes as before. `meter_v3_tp` is `meter_v3`'s checkpoint with its decoder tuned
  this way, so `meter_v3`'s results stay as they are.
- **The labeller's checkpoint.** `nl_base_gn` kept its lowest validation loss, at step 5,000, then
  stopped early: the loss rose while the decoded onset F1 kept improving (0.840 to 0.854).
  `train_labeller.py --select onset_f1` keeps, and stops on, the mean decoded onset F1 of the clean
  and real validation windows (on the tracker-timed grid). `nl_base_gn_f1` trains this way;
  `nl_base_gn1_f1` also puts every training window on the tracker-timed grid (`--grid-noise 1`).
- **Written lengths** (`src/pipeline.py tidy_lengths`, `system_scores.py --tidy`). A length running
  at most a 32nd past the next onset of its staff stops there. That removes extra voices, and agrees
  with the scores: on the clean test windows (true grid), duration accuracy goes from 0.849 to
  0.853. Closing gaps (a note ending a 16th early meets the next one) was tried too. It removed
  42-47% of the rests in two demo pieces, but the scores write those rests: duration accuracy fell
  to 0.827. So only overlaps are trimmed. On the demo pieces that removes 5-9% of the rests and
  5-7% of the extra voices.
- **Notes left out**: the round-3 system at `--delete-above 0.95` and keeping every note
  ([demo](demo.md), "Leaving notes out"), each scored on whole pieces.
- **The beat ablation**: v2 on the clean test MIDI with the real recordings' beats against the
  rendered-MIDI beats (24 performances), what clean MIDI's rendered audio costs v2.

Every variant is scored as round 3 was: each labeller on every grid (windows), and v2 on whole
pieces with Beyer & Dai's metrics, so each change's effect is measured on its own and combined.

### Fourth round: results

All 9 jobs finished (`pace/v2_round4.sh` and the 0.8 threshold job). Each change was measured
from the round-3 system (`nl_base_gn` + `meter_v3`), one at a time, on whole pieces with Beyer &
Dai's metrics. Because one system's numbers vary piece to piece, every difference is paired by
performance, with a 95% interval from a bootstrap over pieces (`scripts/compare_systems.py` ->
`outputs/system_eval/paired.md`; the performances of one piece are not independent). MUSTER mean
error, the change minus the round-3 system, in points (negative is better); **bold**: the
interval excludes 0:

| change | real (24 performances, 11 pieces) | clean (59, 14 pieces) |
|---|---|---|
| tempo-prior decoder (`meter_v3_tp`) | -0.21 [-0.69, +0.21], better on 13/24 | **-0.47** [-0.68, -0.31], 48/59 |
| length tidying (`:tidy`) | -0.06 [-0.14, +0.01], 16/24 | **-0.09** [-0.15, -0.02], 41/59 |
| delete only above p 0.8 (`_d80`) | -0.21 [-0.49, +0.08], 18/24 | -0.06 [-0.34, +0.10], 28/59 |
| delete only above p 0.95 (`_d95`) | -0.20 [-0.46, +0.10], 18/24 | +0.04 [-0.31, +0.26], 23/59 |
| keep every note (`_keepall`) | -0.19 [-0.44, +0.10], 18/24 | +0.06 [-0.29, +0.28], 22/59 |
| *the labeller's second seed (chance)* | *-0.20 [-0.47, +0.19], 15/24* | *-0.15 [-0.49, +0.10], 37/59* |
| labeller kept on onset F1 (`nl_base_gn_f1`, on `meter_v3_tp`) | **+0.45** [+0.14, +0.78], 7/24 | **+0.94** [+0.34, +1.39], 15/59 |
| the same, every window tracker-timed (`nl_base_gn1_f1`) | +0.20 [-0.08, +0.43], 9/24 | **+0.75** [+0.11, +1.20], 18/59 |
| *the recording's beats on clean MIDI (ablation, 24 performances)* | - | ***-2.81** [-4.75, -1.53], 21/24* |

- **The tempo prior helps whole pieces, but does not fix the note values.** Tuning chose weight
  1.0 centred on 116 quarters per minute (`outputs/runs/meter_v3_tp/decode.json`; validation
  selection 0.748 -> 0.752). On clean pieces it lowers the onset error from 18.22 to 17.71 and the
  offset error from 28.51 to 27.74, and 48 of 59 performances improve. On the 24 real ones the
  gain (-0.21) is within chance. The grids themselves barely change: downbeat F on PERiScoPe
  0.773 -> 0.757, time signature 0.643 -> 0.645, beat F 0.880 -> 0.863. The most common wrong bars
  are still half and double note values (PERiScoPe: 3/4 written 3/8 on 3,976 bars, 2/2 as 2/4 on
  3,329). The labeller on its grid scores the same onset F1 (0.481 vs 0.488). The prior is kept
  for the release: whole pieces are what the system delivers.
- **Keeping more notes helps a little on real audio, nothing on clean.** At 0.95 MUSTER missing
  falls from 11.96 to 9.00 on real, and extra rises from 10.73 to 11.98: -0.20 net, 18 of 24
  better. 0.8, 0.95 and keeping every note are within 0.02 of each other, so the labeller's
  confident deletions are few, and leaving them out neither helps nor hurts. The release keeps
  0.95, the demo's default.
- **Tidying is small and consistent.** It lowers the score-similarity duration error (37.48 ->
  36.26 real, 47.46 -> 46.49 clean) and stem error, and the MUSTER offset error.
- **The new labellers are worse.** On the true grid they are as good or slightly better (onset F1
  on PERiScoPe 0.864 / 0.862 vs 0.857; staff 0.823-0.888 vs 0.804-0.863 across the test sets), but
  the checkpoint chosen on onset F1 is a later one whose keep head deletes more (note F1 0.910 vs 0.919 on clean
  windows). On whole pieces MUSTER missing rises from 15.29 to 20.63 on clean and from 11.96 to
  15.85 on real. With deletion mostly off (`_d95`), `nl_base_gn1_f1` still does not beat
  `nl_base_gn` (+0.05 real, +0.49 clean). `nl_base_gn` stays.
- **One training run varies as much as most of these changes.** The labeller's second seed moves
  the real result by -0.20 and clean by -0.15. A difference below ~0.3 on the 24 real performances
  needs a second seed (or the paired interval above) before it means anything.
- **The rendered-MIDI beats are v2's largest loss on clean MIDI.** On the 24 clean performances
  that have a recording, the recording's beats instead of the rendered MIDI's lower v2's error
  from 15.04 to 12.22 (21 of 24 better). With those beats, v2 on clean MIDI (12.22) does better
  than v2 on the Transkun transcription of the same recordings (12.83), as clean input should. This
  is the clean benchmark condition and MIDI uploads to the demo; an audio upload already uses the
  recording's beats.
- **v2 still trails the seq2seq system** on MUSTER mean, by more than chance:

  | | real | clean |
  |---|---|---|
  | seq2seq `sf_base_reg_cab_c128` minus v2 (`meter_v3_tp`) | **-1.97** [-3.34, -0.17], better on 16/24 | **-2.85** [-4.57, -0.70], 49/59 |
  | Beyer & Dai minus v2 (`meter_v3_tp`) | -1.48 [-3.68, +1.40], 15/24 | **-3.55** [-5.46, -1.13], 50/59 |

  The gap is timing: on real audio v2 has MUSTER onset 13.45 vs 9.90 and offset 24.89 vs 19.23,
  and missing 11.89 (9.00 at 0.95) vs 7.57. v2 is better on pitch (2.22 vs 3.06), extra notes
  (10.66 vs 13.48), score-similarity insertion (10.42 vs 15.08) and spelling (6.37 vs 10.92). The
  window decomposition is unchanged from round 3. Onset F1 on PERiScoPe goes true grid 0.857 ->
  oracle grid 0.697 -> meter_v3's grid 0.488, so the meter model is still where most of it goes.

**The release.** `nl_base_gn` + `meter_v3` with `meter_v3_tp`'s decoder, at 0.95, tidied
(`src/release.py`), and `sf_base_reg_cab_c128` for seq2seq. The changes were measured one at a
time; `pace/v2_release.sh` scored the combination, the same at 0.8, and with the labeller's
second seed, then built `models/` (`scripts/export_models.py`), uploaded as
[dat-dev/scoreformer](https://huggingface.co/dat-dev/scoreformer), tag `v2-round4`. MUSTER mean
(real / clean) and the paired difference against the round-3 system:

| system | real | clean | vs round 3, real | vs round 3, clean |
|---|---|---|---|---|
| round 3 (`nl_base_gn` + `meter_v3`) | 12.83 | 15.07 | - | - |
| **the release** (tempo prior, 0.95, tidied) | **12.31** | **14.53** | -0.52 [-0.98, +0.06], 16/24 | **-0.54** [-1.01, -0.21], 43/59 |
| the same at 0.8 | 12.30 | 14.41 | (vs the release) -0.01 | (vs the release) **-0.12**, 46/59 |
| the same, the labeller's second seed | 12.13 | 14.38 | (vs the release) -0.18, chance | (vs the release) -0.15, chance |
| seq2seq `sf_base_reg_cab_c128` | 10.65 | 11.74 | (vs the release) **-1.66** [-2.79, -0.12] | **-2.79** [-4.40, -0.79] |
| Beyer & Dai | 11.14 | 11.05 | (vs the release) -1.17 [-3.15, +1.48] | **-3.48** [-5.37, -1.07] |

- **The changes add up.** Each was within about 0.2 on real audio on its own; together they gain
  0.52.
- **0.8 and 0.95 are the same on real audio.** On clean MIDI, 0.8 is 0.12 better. The release
  and the demo keep 0.95: by ear, leaving real notes out was the problem.

### Fifth round: the note-value level of a whole piece

The meter model's largest error is writing a whole piece at the other metrical level: the same bar
lines, every note value halved or doubled (3/4 as 3/8, 6/4 as 6/8, 2/2 as 2/4). Measured directly
(`src/meter.py level_ratio`: decoded quarters per true quarter over each performance; `level_ok`
in every meter evaluation since), `meter_v3` with the tempo-prior decoder:

| | ASAP validation (102) | ASAP test (59 performances, 14 pieces) |
|---|---|---|
| at the right level | 0.873 | 0.508 |
| the best of the three levels (a perfect choice) | 0.961 | 0.831 |

The test errors are whole pieces, and every decoder (default, tuned, tempo prior) makes the same
ones:
- **at double values:** Bach's fugue BWV 846, Debussy's Reflets dans l'eau and Haydn's sonata 31;
- **at half values:** Chopin's first Ballade (6/4 written 6/8);
- **in between:** Ravel's Ondine (1.4-1.5 times the true quarters).

The Ballade alone is many of the 59 performances.

**The decoder cannot make this choice with its own heads.** The grid it returns is already the
most probable under them, the grid at the other level included. So the choice needs outside
evidence (`src/meter.py choose_level`: the decoded grid or the same bar lines at halved / doubled
values, `scale_grid`). Two kinds were tried first, on ASAP (`scripts/level_experiments.py`). For
each, the bonus for the decoded level was chosen on validation:

| signal | validation | test |
|---|---|---|
| the decoded level | 0.873 | 0.508 |
| a prior on the written tempo (log-normal, training scores: 116 qpm, log-std 0.51) | 0.804 | 0.441 |
| a prior on note density (chords per quarter: 3.5, log-std 0.39) | 0.853 | 0.475 |
| both | 0.814 | 0.441 |
| the labeller's confidence on each grid (any head, any bonus) | at most 0.88 | at most 0.51 |

- **Neither beats the decoder.** A prior pulls every piece towards the typical tempo, and the
  pieces it gets wrong are exactly the atypical ones: the Ballade is written at about 210 quarters
  per minute, and decoded at 120.
- **The labeller is no more confident on the right level.** The best it does on validation equals
  the decoded level. One test setting looks better (0.56), but it is worse on validation, so it is
  chance.

The right level depends on the piece's own music, not on what is typical. What is left is a
learned estimate of this piece's written tempo. `pace/v2_level.sh` (pace/README.md, 4n) trains it
overnight:
- **The tempo head** (`MeterConfig.tempo_head`, `train_meter.py --tempo-head`). It reads the mean
  of the meter model's beat states and gives a Gaussian over the piece's log written tempo,
  trained on the annotated grids (`written_tempo`). The decoder then picks the level whose tempo
  it finds likeliest (`DecodeConfig.level_model_weight`; `eval_meter.py --level` tunes it with
  the prior's weight and the stay bonus on validation).
- **`meter_v3_th`:** the head alone, on `meter_v3`'s frozen trunk (`--init --freeze-trunk`, best
  checkpoint on validation right-level share, `--select tempo`). Every other weight is exactly
  `meter_v3`'s, so on the windows it can only change the level.
- **`meter_v3_ft`:** `meter_v3` fine-tuned with the head (`--select both`: the grid score and the
  right-level share).
- **Evaluation:** both are scored like every meter model (windows, the labeller on their grids,
  and whole pieces in the release's configuration), and `scripts/level_experiments.py` repeats the
  priors and the labeller on every validation and test set.

The expectation is modest. The decoder already adds up the step head's evidence over the whole
piece, so a piece-level head may add little. It is, however, the one remaining signal that sees
this piece's music, and the frozen version cannot change anything but the level.

### Fifth round: results

All 4 jobs finished (`pace/v2_level.sh`).

**The priors and the labeller, on every validation and test set** (`scripts/level_experiments.py`
-> `outputs/level_experiments/meter_v3_tp.md`; validation: ASAP 104, real 22, PERiScoPe 150; test:
clean 59, real 24, PERiScoPe 150):

| signal | validation | test |
|---|---|---|
| the decoded level | 0.837 | 0.644 |
| the best of the three levels (a perfect choice) | 0.938 | 0.897 |
| a prior on the written tempo | 0.721 | 0.601 |
| a prior on note density | 0.775 | 0.532 |
| both | 0.717 | 0.571 |
| the labeller's confidence (any head) | 0.837 | 0.644 |

The labeller's best setting on validation is the largest bonus for the decoded level, i.e. never
switching. Per set, the decoded level is right for ASAP val 0.885, real val 0.773, PERiScoPe val
0.813, clean test 0.508, real test 0.708, PERiScoPe test 0.687.

**The tempo head:**
- **`meter_v3_th`** (frozen trunk, 6,000 steps, best at 4,000):
  - in training's validation, the head alone picks the right level for 0.868 of performances
    (mean over the three sources), against 0.859 as decoded;
  - its log written-tempo error is 0.13 on ASAP, 0.29 on real and 0.17 on PERiScoPe;
  - tuned on the full validation sets together with `meter_v3_tp`'s decoder (`eval_meter.py
    --level`), the level weights all came out 0. Its grids are `meter_v3_tp`'s exactly, and so
    are its whole pieces (+0.00 on every performance).
- **`meter_v3_ft`** (`meter_v3` fine-tuned with the head, 16,000 steps): its tuned decoder
  dropped the tempo prior (weight 0) and the level choice (weights 0).

  | | `meter_v3_tp` | `meter_v3_ft` |
  |---|---|---|
  | clean: downbeat F / time signature / right level | 0.786 / 0.552 / 0.508 | 0.786 / 0.582 / 0.610 |
  | real | 0.650 / 0.440 / 0.708 | 0.666 / 0.470 / 0.667 |
  | PERiScoPe | 0.757 / 0.645 / 0.727 | 0.770 / 0.631 / 0.740 |
  | the labeller's onset F1 on its grid (clean / real / PERiScoPe) | 0.402 / 0.377 / 0.481 | 0.416 / 0.383 / 0.480 |
  | whole pieces in the release's configuration, MUSTER mean real / clean | 12.31 / 14.53 | 12.74 / 15.28 |

  Its windows are a little better, but its whole pieces are worse: +0.43 on real (chance) and
  **+0.75** on clean ([+0.22, +1.13], 13 of 59 better). Its PERiScoPe grids write 2/2 as 2/4 on
  5,780 bars, against 3,329.

**Conclusion.** Neither what is typical (the priors), nor what the labeller finds likely, nor a
piece-level readout of the meter model picks the note-value level better than the decoder does.
The right level depends on the piece's own music, which the model has not learned from 1,100
works. The release keeps `meter_v3` with the tempo-prior decoder, without the level choice.
`choose_level`, the tempo head and `--level` stay in the code (off by default) for a meter model
trained on more, and more varied, data (docs/handoff.md, options 7 and 10: Gangqinpu first).

### Sixth round: more varied meter data, beat times from the notes, beats for MIDI input

Three threads, the first two in `pace/v2_gangqinpu.sh` (pace/README.md, 4o).

**Gangqinpu as meter-model data.** Gangqinpu (Zeng, Mathur & Wang, Oct 2026; arXiv 2610.34694;
Hugging Face `Wayne42/gangqinpu`, gated, CC BY-NC-SA 4.0) has 18,786 web piano recordings (pop
8,353, classical 3,064, folk 748, jazz 321, ...; labels automatic), transcribed by Transkun V2 (ours)
and aligned note by note to their scores, with the score's beats timed through the alignment. It
is the "more, and more varied, data" the fifth round points to. `src/preprocessing/gangqinpu.py`
turns each row into meter-model performances as PERiScoPe's are: the transcribed notes (the notes
the release's refinement interpolated are dropped) and the grid of every run of trustworthy bars
(`aligned_grids`). A bar is kept when all its beats are timed from matched notes or between them,
with confidence >= 0.5, its time signature was written in the score (not inferred), the beats fit
the time signature and run forward at a plausible tempo; pickups are dropped and a dropped bar
splits the run. The release's own split keeps Beyer & Dai's ASAP test works in its test set.
- `meter_v4_gq`: `meter_v3`'s recipe with Gangqinpu as half of every batch (`--gangqinpu-share
  0.5`; the other sources keep their proportions in the rest); `meter_v3_gqft`: `meter_v3`
  fine-tuned on the same mix. Both decoded like the release (tempo prior).
- A fourth meter test set, `gangqinpu` (its test split: music outside the classical benchmark),
  with `meter_v3_tp` scored there as the baseline.
- **Checked locally on the real validation split** (499 rows, 2026-11-24): 487 recordings give
  grids, 81% of the beats are kept, in 1,647 runs (median 39 beats; 65% of the beats in runs of 64
  or more). 92.6% of the kept beats have a played note within 50 ms; the median written tempo is
  96 quarters per minute (ASAP 116). Most drops are bars touching an alignment hole or an
  extrapolated beat (18k of 138k beats), bars with an inferred time signature (2.8k), and bars with
  more beats than their meter (1.0k: a 4/4 bar with a 5th beat). The release counts 6/4 (and 12/4,
  15/16) in other beats than ASAP: its beats are mapped onto ours by their place in the bar.
- **Risk:** 81% of its bars are 4/4 (ASAP is more varied); a tiny model trained on it alone guessed
  4/4 nearly everywhere. Half of every batch keeps the classical data in; if the new models' time
  signatures on ASAP get worse, lower `--gangqinpu-share`. The first job writes
  `data/processed/meter/gangqinpu_report.json`: what each filter dropped on the full release.

**Beat times from the notes** (`src/meter.py refine_grid`; `DecodeConfig.snap_s` / `fill_tol`, 0 =
off, so every earlier decoder is unchanged). The oracle grid loses 0.16 onset F1 to the tracker's
timing (0.857 -> 0.697). Beat This!'s beats that it finds are close (18 ms jitter, Transkun's
onsets ~5 ms), but it misses 12% of them, and the grid then interpolates over two beats under
rubato. After decoding, each anchor moves onto a chord onset within `snap_s` seconds, and each
notated beat between two anchors gets one at a chord onset within `fill_tol` beats of where the
anchors put it (none where nothing is played). Bars, time signatures and quarter positions are
untouched. `meter_v3_tp_rf` (snap 0.05 s + fill 0.25), `meter_v3_tp_fill` (fill only) and
`meter_oracle_rf` share `meter_v3`'s checkpoint (`pace/v2_jobs.sh refine`); the labeller on them
and v2 on whole pieces say how much comes back. The settings are a first guess, not tuned.

**Beats for MIDI input: PM2S (screen, local, 2026-11-24).** v2 on clean MIDI gets Beat This!'s
beats of the MIDI rendered by `src/synth.py`, which cost it 2.8 MUSTER points against the real
recordings' beats (fourth round). PM2S (Liu et al., ISMIR 2022; MIT; a CRNN beat tracker on the
notes) on the ASAP test MIDI, scored like Beat This! (`src/beats.compare`):

| beats, ASAP test MIDI (24 performances with recordings) | beat F | downbeat F | wrong tempo octave |
|---|---|---|---|
| Beat This! held-out folds on the rendered MIDI (fourth round) | 0.576 | 0.422 | |
| Beat This! `final0` (trained on all of ASAP) on the rendered MIDI | 0.580 | 0.456 | 4 / 24 |
| PM2S released weights (trained on all of ASAP) on the MIDI | **0.786** | **0.613** | 0 / 24 |
| Beat This! held-out folds on the real recordings (fourth round) | 0.709 | 0.592 | |

On all 59: PM2S 0.745 / 0.559, `final0` 0.545 / 0.396; PM2S is better on 58 of 59 performances
(beats) and 57 (downbeats). Ondine stays hard for both (0.46 / 0.42). **Caveat:** PM2S's released
weights were trained (`--full_train`) on every ASAP performance, ours among them, as Beat This!'s
`final0` was; the leak barely helped Beat This! (0.580 against 0.576 held out), but PM2S saw these
very MIDI files. Before quoting it: PM2S retrained without our test (and validation) pieces.

PM2S is now a beat source of the pipeline (`src/pm2s_beats.py`: its model and post-processing,
vendored under its MIT licence; checked identical to the original code on five test pieces, except
that the last beat, which PM2S's gap filling drops, is kept). `V2Pipeline.run(beat_source="pm2s")`
takes the beats from the notes, for MIDI and for a recording's transcription
(`audio_to_score.py --beats pm2s`). `pace/v2_gangqinpu.sh` scores it against Beat This! on the
whole-piece inputs (`scripts/compare_beats.py`: clean MIDI against the rendered audio and the
recordings, the transcriptions against the recordings) and runs the release with it on whole
pieces (`system_scores.py --clean-beats pm2s --real-beats pm2s`, system `..._pm_rpm`).

**PM2S on music it never saw** (`scripts/fit_pm2s_noise.py`, locally on 200 performances of
Gangqinpu's validation split, 2026-11-24; ground truth: the score's beats timed through the
alignment): beat F 0.705, downbeat F 0.483, the wrong tempo octave on 39% of performances (17%
double, 23% half), at the right tempo 23% of the beats missed and 21% spurious, 6 ms timing jitter.
Beat This! on the ASAP recordings (held out): beat F 0.71-0.81, downbeat F 0.60-0.65, the wrong
octave on 23-33%, 12% / 13%, 18 ms. The ASAP screen (0.786 / 0.613, no octave errors) owed much to
the leak. Not like for like (Beat This! cannot be run on Gangqinpu without its audio, and part of
the "half" errors may be the note-value ambiguity itself), but PM2S is no clear win.

**Trained for PM2S or for Beat This!** A meter model learns the errors of the tracker it is trained
with (`configs/beat_noise.json`: Beat This!'s, fitted on ASAP). `meter_v4_gq_pm` is `meter_v4_gq`
trained on PM2S's fitted errors instead (`configs/beat_noise_pm2s.json`, from Gangqinpu's training
split on PACE; `--real-beats simulated`: the real transcriptions get the simulated PM2S too, not
Beat This!'s recorded beats), its decoder tuned on them (`eval_meter.py --tune-noise`). Both are
tested with both trackers' beats, at the meter level (`eval_meter.py`, `--tracker pm2s`) and on
whole pieces: the 2 x 2 of tracker in training and at test.

**Beat This!, PM2S, or both** (what the sixth round's three meter runs compare). `meter_v4_gq`
reads Beat This!'s beats, `meter_v4_gq_pm` PM2S's, `meter_v4_gq_both` Beat This!'s with PM2S's as
evidence on them (`train_meter.py --aux pm2s`; `src/meter.py aux_features`, 5 features per beat
after the 39 others, which a model without them does not read: does PM2S have a beat there, how
many of its beats fall in the interval and where (2 at 0.5 = it beats twice as fast; its beats at
0.5 and none on ours = ours are on the off-beats), does it mark a downbeat). Their failures differ
(PM2S doubles the tempo, Beat This! now and then tracks the off-beats), so together they can show
which one is wrong. Each is tested with its own tracker(s); the pipeline gives a model that reads
PM2S its beats of the notes by itself.

**How much tracker noise, measured.** Every simulated error rate is fitted on a tracker's real
output, not set by hand. Beat This!'s profile (`configs/beat_noise.json`) was fitted on the ASAP
recordings, where it never tracked the off-beats (`p_offbeat` 0), so a model trained on it has never
seen that. On Gangqinpu its errors are fitted again from its real beats of the downloaded recordings
(`fit_pm2s_noise.py --from-stored beat_this` -> `configs/beat_noise_bt_gangqinpu.json`; at least 30
tracked performances, else none is written and Gangqinpu keeps the ASAP profile), and Gangqinpu's
performances are simulated with it (`--beat-noise-gangqinpu`), the others with the ASAP one.

**Real tracker output in training, mixed with the simulated one.** A fitted profile makes errors at
fixed rates, wherever they fall; a real tracker goes wrong where the music makes it (syncopation,
rubato, sparse passages), PM2S because of the very notes the meter model reads. So the trackers
are also run on the data once (`scripts/track_meter_data.py`, deterministic: running them at every
step would give the same beats): Beat This! on Gangqinpu's recordings (downloaded by
`scripts/fetch_gangqinpu_audio.py` from the URLs the release lists: every validation and test one,
3,000 training ones), PM2S on Gangqinpu's and PERiScoPe's notes; never ASAP, which both trained on.
Each time a performance that has them is drawn in training, it gets its tracker's real beats with
probability 0.5 (`--tracked`, `--tracked-share`), else a fresh simulated draw: the music-dependent
errors and the variety. At test, `eval_meter.py --real-tracked` uses Beat This!'s real beats of the
Gangqinpu test recordings (`gangqinpu_real`), `--tracker pm2s` PM2S's: the 2 x 2 with both real
trackers on the same music, and `tracked_report.json` compares the two trackers' beats there like
for like. The recordings belong to the source site and its uploaders (the dataset's licence covers
its annotations): fetched at one request a second, for this research, not redistributed.

The fit found a bug in the tracker simulation (`src/beats.py BeatNoise`): spurious beats were drawn
as binomial(n, extra / (1 - extra)), which fails above extra = 0.5; 6 of PM2S's 120 profiles are
there (up to 0.73), so its training would have stopped within a few steps. Above 0.5 the count is
now Poisson with the same mean; up to 0.5 the draws are unchanged (Beat This!'s profiles reach 0.37).

### Sixth round: results (2026-11-25)

`pace/v2_gangqinpu.sh` ran end to end (job ids 7295929-36).
- **Data:** Gangqinpu gave 17,347 / 487 / 501 performances (train / val / test) with grids: 3.83M
  training beats, against about 1,100 ASAP + PERiScoPe works before.
- **Recordings:** 4,007 of 4,008 downloaded (all validation and test, 3,000 training). A few MP3s
  were cut short and Beat This! skipped them.
- **Training:**
  - `meter_v4_gq` used all 60,000 steps, best at step 56,000, so it may not have finished
    improving;
  - `meter_v4_gq_both` stopped at 44,000 (best 28,000);
  - `meter_v4_gq_pm` stopped at 30,000 (best 14,000).
- **`meter_v3_gqft`** (above) was not in the batch's runs and was not trained.

**The two trackers on the same unseen music** (Gangqinpu validation, 1,641 performances,
`tracked_report.json`):

| tracker | beat F | downbeat F | wrong tempo octave |
|---|---|---|---|
| Beat This! on the recordings | 0.713 | 0.661 | 34% |
| PM2S on the transcriptions | 0.694 | 0.509 | 42% |

**Meter models** (grid time-signature accuracy per bar / downbeat F; each with the tracker it was
trained for; `clean`, `periscope` and `gangqinpu` with the simulated tracker, `real` and
`gangqinpu_real` with Beat This!'s real beats of the recordings, `_pm2s` with PM2S's real beats):

| meter model | clean | real (24) | PERiScoPe | Gangqinpu | Gangqinpu, real recordings |
|---|---|---|---|---|---|
| `meter_v3_tp` (the release) | 0.552 / 0.786 | 0.440 / 0.650 | 0.645 / 0.757 | 0.633 / 0.781 | 0.542 / 0.672 |
| **`meter_v4_gq`** (Beat This!) | 0.572 / 0.797 | 0.450 / 0.657 | 0.644 / 0.756 | **0.834 / 0.852** | **0.794 / 0.714** |
| `meter_v4_gq_both` (+ PM2S as evidence) | 0.554 / 0.780 | 0.375 / 0.694 | 0.609 / 0.753 | 0.810 / 0.826 | 0.761 / 0.705 |
| `meter_v4_gq_pm`, PM2S's real beats | 0.502 / 0.592 | 0.500 / 0.658 | 0.478 / 0.558 | 0.804 / 0.719 | |
| `meter_v3_tp_rf` (beat times from the notes) | 0.552 / 0.792 | 0.440 / 0.637 | 0.645 / 0.761 | 0.633 / 0.785 | |

**Whole pieces** (MUSTER mean error, %, lower is better; paired differences against the release
with a 95% bootstrap interval over pieces, bold where it excludes 0; `outputs/system_eval/paired.md`):

| system | real (24) | clean (59) | vs the release, real | vs the release, clean |
|---|---|---|---|---|
| the release (`meter_v3_tp`, Beat This!) | 12.31 | 14.53 | | |
| + beat times from the notes (`_rf`: snap + fill) | 12.31 | 14.27 | -0.00 [-0.21, +0.13] | **-0.26** [-0.37, -0.11] |
| + missed beats filled only (`_fill`) | 12.28 | 14.35 | -0.03 [-0.14, +0.04] | **-0.18** [-0.26, -0.08] |
| `meter_v4_gq` (Gangqinpu, Beat This!) | 12.70 | 15.18 | +0.39 [-0.09, +1.09] | **+0.65** [+0.16, +1.15] |
| `meter_v4_gq_both` | 12.10 | 15.84 | -0.21 [-1.11, +0.61] | **+1.31** [+0.83, +1.89] |
| the release with PM2S's beats (`_pm_rpm`) | 11.56 | 12.43 | **-0.74** [-1.49, -0.05] | **-2.10** [-3.08, -1.17] |
| **`meter_v4_gq_pm` with PM2S's beats** | **11.12** | **12.08** | **-1.19** [-2.07, -0.40] | **-2.45** [-3.74, -1.18] |
| seq2seq (`sf_base_reg_cab_c128`) | 10.65 | 11.74 | | |
| Beyer & Dai | 11.14 | 11.05 | | |

- **`meter_v4_gq_pm` + PM2S against the others:**
  - against `meter_v3_tp` + PM2S: -0.45 [-0.83, -0.02] real, -0.35 [-0.91, +0.35] clean;
  - seq2seq minus it: -0.47 [-1.33, +0.63] real, -0.33 [-1.30, +1.07] clean;
  - Beyer & Dai minus it: +0.02 [-1.69, +2.24] real, -1.02 [-2.57, +1.69] clean.

  For the first time, v2 is within chance of both on both conditions.
- **What PM2S changes on whole pieces:** onsets and offsets, missing and extra notes all drop
  (real: onset 13.82 -> 12.93, missing 8.90 -> 7.29, extra 11.80 -> 10.17).

**What it means:**
- **Gangqinpu teaches the meter model music outside classical piano, without unlearning the
  classical.**
  - On Gangqinpu's real recordings, time-signature accuracy goes from 0.54 to 0.79, and the
    right note-value level from 0.65 to 0.75.
  - On the classical sets, the meter scores are equal.
  - On the classical benchmark's whole pieces it is not better: +0.39 real (chance), +0.65 clean
    (sure).

  How much it helps whole scores of pop is not measured: that needs Gangqinpu's scores as
  references (option 7's labeller targets).
- **Beat times from the notes:** a small, sure gain on clean MIDI (-0.26), none on real
  recordings. The settings were a first guess.
- **PM2S gives the largest whole-piece gain of any change so far, but not one we can quote yet.**
  Its released weights were trained on all of ASAP, our 14 test pieces included. That flatters
  it most on clean MIDI, which it saw as is, and also on real recordings: their transcriptions
  are of performances whose MIDI it trained on.
  - On music it never saw (Gangqinpu), its beats are worse than Beat This!'s on the recordings
    (downbeat F 0.51 vs 0.66).
  - Yet a meter model trained for it ends level there (0.804 vs 0.794 time signature).
  - For MIDI input the alternative is Beat This! on a rendering, which costs 2.8 points (fourth
    round), so part of the gain is likely real.
  - Before it is quoted or released, retrain PM2S without our test and validation pieces (its
    training code is public, MIT) and score again.
- **The release stays `meter_v3_tp` + Beat This!** until that retraining. The next candidate is
  `meter_v4_gq_pm` with PM2S's beats, and `meter_v4_gq` for music outside classical piano.

### Seventh round: PM2S retrained without our test pieces

`pace/v2_pm2s.sh` (pace/README.md, 4p). The sixth round's largest gain came with PM2S's released
weights, which were trained on all of ASAP, our 14 test pieces included. That gain can only be
quoted from a PM2S that never saw them.

- **The model:** `scripts/train_pm2s.py` trains PM2S's `RNNJointBeatModel` (27.5M parameters)
  from scratch: `pm2s_ho`.
  - Recipe: PM2S's own, read from its `dev/` code. Every note within 70 ms of a beat is a beat
    target, and downbeats alike. The tempo head learns the inter-beat interval in 10 ms classes.
    Training uses 500-note segments, 32 per batch, and AdamW (1e-3, betas 0.8 / 0.8) with the rate
    x0.1 after 40k steps.
  - Augmentation: tempo x U(0.8, 1.2), up to 12 semitones of transposition, and either octave
    copies or dropped chord notes.
- **The data, ours:** ASAP's training pieces (clean MIDI and Transkun transcriptions), PERiScoPe
  and Gangqinpu, Gangqinpu half of every batch. That is more varied than PM2S's original A-MAPS,
  CPM and ASAP.
- **The selection:** on the post-processed beats (mean of beat F and downbeat F over the
  validation splits). PM2S selects on per-note F.
- **What is measured, all tagged `ho`:**
  - its beats on the ASAP test inputs and on Gangqinpu, against the released PM2S and Beat This!;
  - the release and `meter_v4_gq_pm` on whole pieces with its beats: the sixth round's gain
    without the leak;
  - `meter_v4_gq_pmho`, the meter model trained again for its beats.

### Seventh round: results (2026-11-26)

`pace/v2_pm2s.sh` ran end to end: 8 jobs, the retraining 2 h 51 min.

**The retrained PM2S is under-trained.**
- Its training loss rose from 0.48 at step 8,000 to 0.61 at step 38,000, and validation fell with
  it (mean beat / downbeat F 0.62 at 10k, 0.50 at 38k, 0.54 at 60k).
- A training loss that rises is unstable optimisation, not overfitting. PM2S's recipe (AdamW 1e-3,
  betas 0.8 / 0.8, no clipping) does not train stably on our data, so best.pt is from step 10,000.
- What follows understates a properly trained PM2S without our test pieces.

**Its beats (`beats_compare_ho.md`, `tracked_report_ho.json`):**

| beats | ASAP test clean (beat / downbeat F, wrong tempo octave) | Gangqinpu test (beat / downbeat F) |
|---|---|---|
| PM2S, released (trained on ASAP incl. our test pieces) | 0.786 / 0.618, 0% | 0.710 / 0.513 |
| PM2S, held out (`pm2s_ho`) | 0.655 / 0.514, 50% | **0.780** / 0.636 |
| Beat This!, on the recording | 0.711 / 0.596, 33% | 0.709 / **0.668** |
| Beat This!, on the MIDI rendered to audio | 0.576 / 0.422, 29% | - |

- On ASAP the released weights were far ahead: partly the leak, partly this under-trained retrain.
  Half of its pieces come out at double or half the tempo.
- On Gangqinpu, which it trained on and the released one did not, it beats both.

**Whole pieces (MUSTER mean, real recordings / clean MIDI):**

| system | real (24) | clean (59) |
|---|---|---|
| the release (`meter_v3_tp`, Beat This!) | 12.31 | 14.53 |
| the release + released PM2S | 11.56 | 12.43 |
| the release + `pm2s_ho` | 12.96 | 13.13 |
| `meter_v4_gq_pm` + `pm2s_ho` | 11.80 | 12.35 |
| **`meter_v4_gq_pmho` + `pm2s_ho`** | **11.59** | **11.85** |
| seq2seq / Beyer & Dai | 10.65 / 11.14 | 11.74 / 11.05 |

Paired (`outputs/system_eval/paired.md`):
- **The leak was real.** With the release, `pm2s_ho` against the released PM2S: +1.40 on real
  ([+0.54, +2.78], worse), +0.70 on clean (chance).
- **The PM2S path still helps MIDI input without it.** `meter_v4_gq_pmho` + `pm2s_ho` against the
  release:
  - **-2.68 on clean** ([-3.70, -1.52], 51 of 59);
  - -0.71 on real (chance).
- **That is level with the baselines.**
  - Against seq2seq: -0.10 on clean, -0.94 on real, both within chance.
  - Against Beyer & Dai: -0.79 on clean and -0.45 on real, both within chance.
- **Training the meter model for this PM2S helped:** `meter_v4_gq_pmho` against `meter_v4_gq_pm`,
  both on `pm2s_ho`: -0.50 on clean (sure), -0.21 on real.

**Conclusion.**
- **For MIDI input, with no recording to track,** `meter_v4_gq_pmho` + PM2S is the best honest v2
  and level with seq2seq and Beyer & Dai. Beat This! on a rendered MIDI is far worse.
- **For recordings, Beat This! on the audio stays.** On Gangqinpu's real recordings, PM2S was worse
  (eighth round below).
- **Next:** retrain `pm2s_ho` with stable optimisation: a lower rate, the usual betas, gradient
  clipping. It is about 3 GPU hours.

### Eighth round: the labeller on Gangqinpu's scores, whole pieces outside classical piano

`pace/v2_gqlabel.sh` (pace/README.md, 4q). The sixth round made the meter model general:
Gangqinpu's time signatures right on 0.79 of its real recordings, against 0.54. The labeller has
only seen classical scores, and "more general" had no whole-piece measure.

- **The data:** Gangqinpu's written scores as labeller windows
  (`src/preprocessing/gangqinpu.py score_measures` / `labeller_performance`,
  `scripts/build_gangqinpu_windows.py` -> `{train,val,test}_gangqinpu`, Gangqinpu's own splits).
  - The release's `notation` gives every written note's measure, onset, length, staff and grace
    flag, and the performance note aligned to it. It becomes the measures every MusicXML score
    becomes, and is cut into windows like PERiScoPe.
  - How the release writes a score:
    - a note held across a bar line is one long note: cut at the bar line and tied;
    - a written note the score MIDI does not strike is a tie continuation;
    - duplicates and zero-length notes are dropped, with their performance notes.
  - The inputs are the recording's Transkun notes the alignment kept. They have real timing,
    lengths and missed notes, but not the transcriber's extra notes. So they are stored as
    `source: transkun_aligned`, and the Dataset adds only the noise model's extra notes
    (`NoiseModel.hallucinations_only`).
  - Checked on 40-60 validation pieces:
    - 98% of the performed notes pair with a written note;
    - every window has a grid;
    - no token is outside the vocabulary;
    - rendering the true labels gives onset F1 0.966, duration 1.000, staff 1.000. The 3% missing
      are the notes the alignment interpolated.
    - The references' MusicXML reads back with 25 of 31,841 notes different.
- **The labeller `nl_gq`:** `nl_base_gn`'s recipe (`--grid-noise 0.8`) with
  `--gangqinpu-share 0.5`.
  - The usual data keeps its proportions in the other half. An epoch stays a pass over the usual
    data, so patience covers the same.
  - Validation adds `val_gangqinpu` (`val_gq`), and the selection averages over it too.
- **Window tests** (`scripts/eval_labeller.py`):
  - `gangqinpu`: Gangqinpu's test windows, with the true grid or a meter model's grid from the
    simulated tracker;
  - `gangqinpu_real`: the same windows, with the grid from Beat This!'s beats of the recordings.
    The meter model's Gangqinpu grids cover only the reliably timed bars: windows outside them
    are left out and counted.
- **Whole pieces** (`scripts/gangqinpu_inputs.py` -> `outputs/system_eval_gangqinpu`): 60 test
  performances, one per score, with genres in proportion.
  - The reference is the written score as MusicXML.
  - `gangqinpu` gives the aligned notes as MIDI (like ASAP's clean MIDI). `gangqinpu_real` gives
    Transkun run on the recording here, cut to the aligned span: what the demo gets from an upload.
  - Every system gets Beat This!'s beats of the recording (PM2S's with `rpm`).
  - The systems: the release, `nl_gq` and `meter_v4_gq` each alone and together,
    `meter_v4_gq_pm` with PM2S, seq2seq and Beyer & Dai. They are scored with Beyer & Dai's
    metrics and paired by performance (`compare_systems.py --results
    outputs/system_eval_gangqinpu/results.json`).
- **Also:**
  - `meter_v4_gql`: `meter_v4_gq` with 100k steps. Its best checkpoint was at 56k of 60k.
  - `meter_v3_s1_tp`: the release's meter model's second seed with the tempo-prior decoder, on
    whole pieces. This is v2's run-to-run spread for the report (the labeller's is ±0.2).

### Eighth round: results (2026-11-26)

`pace/v2_gqlabel.sh` ran end to end: 8 jobs, the longest 2 h 07 min.

**The data.**
- 17,392 of Gangqinpu's 17,589 training performances became 322,541 training windows (42.6M input
  notes, 9 minutes on 16 CPUs). 196 had too few aligned notes; 0.5% of windows were dropped.
- Validation: 492 performances, 4,628 windows. Test: 506 performances, 4,762 windows.
- 505 training tokens of 1.3M+ are outside the vocabulary (`<unk>`).
- 324,711 notes were held across a bar line (written tied). Duplicates (72k) and zero-length notes
  (48k) were dropped.

**`nl_gq`** (`nl_base_gn`'s recipe, half of the windows from Gangqinpu) stopped early at step 43,110
(best 40,000) in 1 h 48 min, with the evaluation. Onset F1 with the true grid:

| | ASAP clean | ASAP real | PERiScoPe | Gangqinpu |
|---|---|---|---|---|
| `nl_base_gn` | 0.782 | 0.800 | 0.857 | 0.882 |
| `nl_gq` | **0.785** | **0.806** | **0.866** | **0.921** |

- It is better everywhere, classical included.
- On the meter models' grids, window onset F1 barely moves: on Gangqinpu with `meter_v4_gq`'s grid
  from Beat This!'s beats of the recordings, 0.498 for both. Where the grid is wrong, it is wrong
  for either labeller.

**Whole pieces, ASAP (paired, against the same meter model with `nl_base_gn`):**

| | real | clean |
|---|---|---|
| `nl_gq` + `meter_v3_tp` (12.23 / 14.38) | -0.07 (chance) | **-0.15** [-0.31, -0.04] |
| `nl_gq` + `meter_v4_gq` (12.71 / 14.86) | +0.01 (chance) | **-0.32** [-0.56, -0.14] |

**Whole pieces, Gangqinpu.**
- 59 test pieces, one per score: 25 pop, 15 unlabelled, 10 classical, 5 other, 3 folk, 1 jazz.
- 58 were scored by every system. MUSTER mean:

| system | from the recording (`gangqinpu_real`) | aligned notes (`gangqinpu`) |
|---|---|---|
| Beyer & Dai | 8.07 | 7.65 |
| seq2seq (`sf_base_reg_cab_c128`) | 7.69 | 7.64 |
| v2, the release (`nl_base_gn` + `meter_v3_tp`) | 6.18 | 6.46 |
| v2, `nl_base_gn` + `meter_v4_gq` | 5.84 | 6.37 |
| v2, `nl_base_gn` + `meter_v4_gq_pm` + PM2S (released) | 6.55 | 6.35 |
| v2, `nl_gq` + `meter_v3_tp` | 4.25 | 4.49 |
| **v2, `nl_gq` + `meter_v4_gq`** | **4.00** | **4.40** |

Paired, from the recordings (`outputs/system_eval_gangqinpu/paired.md`; each piece its own, see
the note below):
- **On pop, v2 is far ahead of both baselines.** Even the release beats seq2seq by 1.51
  ([+0.88, +2.10]) and Beyer & Dai by 1.88. With both Gangqinpu models, seq2seq is +3.69 behind
  ([+3.06, +4.37], 1 of 58 better) and Beyer & Dai +4.07.
- **`nl_gq` makes the biggest difference:** -1.93 against the release ([-2.34, -1.54], 57 of 58).
  - MUSTER offsets (written lengths) go from 17.4 to 10.7, onsets from 5.5 to 3.6.
- **`meter_v4_gq` helps too:** -0.35 against `meter_v3_tp` ([-0.72, +0.01]).
  - Score similarity's duration error halves: 48% to 24-28%.
- **PM2S is worse on recordings:** +0.71 ([+0.38, +1.05]). On the aligned notes it is equal.

**Also:**
- **`meter_v4_gql` (100k steps).** It stopped at 66,000 (best 50,000). Its meter scores are a little
  lower than `meter_v4_gq`'s. On whole pieces it is within chance everywhere: -0.58 real, -0.29
  clean, 0.00 on Gangqinpu. Not promoted; the 60k model stays.
- **The meter model's second seed (`meter_v3_s1_tp`)** against the release's `meter_v3_tp`: +0.59
  on real (chance), **+0.93 on clean** ([+0.44, +1.33]). v2's spread from the meter model's
  training seed is up to about 0.9 on whole classical pieces, against 0.2 for the labeller's. So
  differences between meter models smaller than about a point are within training noise,
  `meter_v4_gq`'s +0.65 on clean included.
- **Publishing.**
  - `nl_gq` was made the default on Hugging Face by the batch itself. The job order of that batch
    compared it on the three classical sets only; it won there too.
  - `meter_v4_gql` was archived.
  - The release on the repo is now `nl_gq` + `meter_v4_gq`.
- **The paired comparison's first output grouped all of Gangqinpu's performances as one piece**
  (zero-width intervals). Fixed in `scripts/compare_systems.py` (`piece_of`), and the intervals
  above are from the fixed version.

**Conclusion.** v2 with both Gangqinpu models is the general system:
- best by far on music outside classical piano (4.00 against 7.69 and 8.07);
- on the classical benchmark within chance of the release on real audio, and within the meter
  model's seed spread on clean MIDI.

seq2seq stays best on classical real audio (10.65).

### Ninth round: PM2S retrained with stable optimisation

`pace/v2_pm2s2.sh` (pace/README.md, 4r).
- **Why.** `pm2s_ho` trained unstably with PM2S's own optimiser settings: its training loss rose
  from step 8,000 to 38,000. Even so, v2 for MIDI input with it was level with seq2seq and Beyer &
  Dai on clean MIDI. The demo renders a MIDI upload for Beat This!, about 3 points worse there.
- **Two trainings, same data and recipe otherwise:** AdamW betas 0.9 / 0.999, eps 1e-8, gradient
  clipping 1.0, 1,000 warmup steps, cosine decay to 5% over 60k steps. Learning rate 3e-4
  (`pm2s_ho2`) and 1e-4 (`pm2s_ho3`). The gradient norm is logged.
- **The pick.** The better by its own validation score (mean post-processed beat / downbeat F over
  the validation splits) becomes `pm2s_hos` (`scripts/pick_run.py`). Everything made with it is
  tagged `hos`.
- **What is measured:**
  - its beats against the released PM2S, `pm2s_ho` and Beat This!;
  - the release with it (how much of the leak's gain a properly trained PM2S keeps);
  - `meter_v4_gq_pmhs`, the meter model trained for it;
  - v2 with it and `nl_gq` on whole pieces, on ASAP and on Gangqinpu.
- **The comparisons that decide the demo's MIDI path:**
  - against today's default (Beat This!, which on clean MIDI means the rendered MIDI);
  - on Gangqinpu, against the default with Beat This! on the rendered MIDI (`_rsb`, new in
    `scripts/system_scores.py --real-beats synth`), which is what a MIDI upload gets today.
- **The timing check** (`src/timing.py`, `_tc`), in the same batch: `nl_gq` + `meter_v4_gq` with
  and without it, on ASAP and Gangqinpu.
  - It writes a triplet-written note that was played on a straight eighth or sixteenth straight.
  - It came from the user's song, where one window of the new labeller wrote 40% of bars 64-71 as
    triplets against straight playing (the old labeller wrote 17% there).
  - On that song it moved 89 notes, all on clear evidence: played at 0.44-0.55 of the beat and
    written at 1/3 or 2/3, or played on a sixteenth at least 0.1 of a beat from the triplet spot.
- **The beat-level timing check** (`src/timing.py` check_beats, `_tb`, in `pace/v2_decode.sh`):
  - The per-note check left the song's bars 64-71 at 20% triplets: a syncopation played straight
    (1.5, 1.71, 2.24, 2.73) passed no single note's test.
  - Judging each beat as a whole fixed it: the straight 16ths against the eighth triplets (sixths
    where the labeller wrote sixths), measured from where the beat was played. Bars 64-71 now have
    no triplets, and the whole song went from 4.5% to 2.2%.
  - Expected: positive on Gangqinpu (pop: syncopation), about neutral on ASAP, where played
    triplets are nearer their own grid.
- **The readability rules** (`src/readable.py`, `pace/v2_decode.sh`), each on the timing check
  (`_tc`), on ASAP and Gangqinpu:
  - `_jf` is the short-bar fix. The meter decoder's ts-change penalty makes it stretch a short bar
    rather than change the time signature. In the user's song, bars 47 and 51 (0.46 s and 0.66 s)
    were each written over a whole 4/4 bar. A step below half its neighbours' seconds per quarter
    gets the beats its time holds, and its bar becomes 2/4 (two beats at least).
  - `_rd` closes 16th rests, trims overlaps up to a 16th (only into a length one note can write,
    never across the triplet and straight grids), and sets stems by voice and hides the lesser
    voice's rests.
  - `_tb_rd_jf` is the demo's default, against `_tc` and against the plain system.
  - Expected: `_jf` small but positive where the decoder stretched bars. `_rd` may cost on duration
    against ASAP's scores, which do write 16th rests (closing every rest under an eighth cost about
    2 points before), and is kept for the demo whatever it scores, if it costs under a point.

### Tenth round design: the deep-learning side of the rules (2026-11-26)

The rules above (`src/timing.py`, `src/readable.py`) patch model errors after the fact. This round
asks whether the models can be made not to make them, and keeps every rule as its own system so the
models' scores stay separate. `pace/v2_decode.sh`, 8 jobs.

**1. Beat-level decoding of the labeller** (`src/beat_decode.py`, `_bd`). The labeller picks each
note's split of the quarter on its own (argmax of its `sub` head). The decoding scores each split as
log p_model + a timing term (the played position against the split, weight tuned) and can keep one
family of splits (straight or triplet; 1/2 in both) per beat, per staff or for both hands, measured
from where the beat's on-the-quarter notes were played. `scripts/tune_beat_decode.py` picks the
setting on validation windows with the tracker's timing errors on the grid (as in training).

- ASAP validation, 600 windows (local, CPU; the released `nl_gq`):

  | setting | position accuracy | length accuracy |
  |---|---|---|
  | the labeller's own (argmax) | 91.58% | 84.30% |
  | timing term, weight 2, per note | **91.69%** | 84.30% |
  | weight 4, one family per beat and staff, anchored | 91.60% | 84.00% |
  | weight 4, one family per beat (both hands) | 91.33% | 83.82% |
  | weight 64, per note | 90.39% | 84.30% |

  Classical scores do mix straight and triplet splits within a beat, so the one-family constraint
  costs a little there, and the timing term helps only at a small weight (the grid is noisy).
- On the user's song it changed almost nothing (bars 64-71: 39.8% triplets -> 38.9% at the
  strongest setting). The reason: the labeller writes 526 of the song's notes as triplets that were
  played on a straight sixteenth (median 0.008 of a quarter from it, 0.109 from the written spot),
  with a median confidence of **67,000 to 1**. Overturning that needs a timing weight of about 900;
  above about 4, validation accuracy falls. A confident model error is not a decoding problem.
- The tenth round tunes it on ASAP + Gangqinpu validation and measures it on test windows
  (`eval_labeller.py --beat-decode`, modes `*_bd`) and whole pieces.

**2. The labels the labeller learns from** (`scripts/audit_labels.py`, `nl_gq_mc`). The old
labeller (no Gangqinpu) wrote 17% of bars 64-71 as triplets, `nl_gq` 40%: the Gangqinpu training
pairs are the suspect. Their scores are user-made arrangements; where one writes a beat as triplets
the pianist played straight, the pair teaches exactly the song's error.

- `src/notelabels.py contradicted` judges a beat at a time on the window's true grid (the score's
  beats timed through the alignment): written with a triplet split, every note within 0.06 of a
  quarter of a straight sixteenth and nearer the sixteenths than the written grid by 0.03 on
  average (and the reverse). One note's timing is no evidence: in ASAP's validation performances
  22% of the notes written on the half beat are played more than 0.08 from it, and a per-note test
  flagged 22-25% of ASAP's written triplets (and 10-14% of its sixteenths, the other way).
- ASAP, 1,500 windows each, beat-level: **5.6% (train) / 7.9% (val)** of written triplets played
  straight; 0.7% / 1.2% of written sixteenths played as triplets. That is the baseline; the audit
  measures Gangqinpu against it.
- `nl_gq_mc` retrains `nl_gq` (same recipe) with those notes' position and length left out of the
  loss (`--mask-contradicted`; keep, bar and staff still learned): if the pop labels are the cause,
  it should write fewer triplets where the timing is straight, without losing accuracy elsewhere.

**3. The meter decoder's short bars** (`src/meter.py local_tempo_logprior`, `meter_v4_gq_lt`). The
decoder pays 11.5 for a time-signature change and only about 1.8 (its written-tempo prior) for a
step of three quarters in one beat's time, so it stretches a short bar. A second decoding pass
weighs each step against the median written tempo of the steps around it.

- On the song it removed both jumps but did not write 2/4 bars: it moved the meter by two beats
  instead, and four bar lines (89.8-95.5 s) left the tracked downbeats (the short-bar rule keeps
  all 18 there). Writing a single inserted short bar needs the decoder to price "one bar of another
  meter, then back" below two time-signature changes, which is a larger change.
- It is tuned on validation (ASAP, real, PERiScoPe and Gangqinpu: `eval_meter.py --local-tempo`)
  and measured as its own system; `_jf` stays until something beats it.

**4. Which rules stay: decided by the benchmark** (`pace/v2_decode2.sh`, `scripts/rule_verdict.py`).
The project keeps as much of the decision with the models as it can, so a correction rule has to
earn its place. The deep-learning configuration is the labeller + its tuned beat decoding on
`meter_v4_gq`; each correction rule is measured added to it, for both labellers, on whole pieces
from real recordings (ASAP `real`, Gangqinpu `gangqinpu_real`), MUSTER mean error paired by
performance with a 95% bootstrap over pieces.

| what | decision |
|---|---|
| beat decoding (`_bd`, model side) | on unless significantly worse on a condition |
| timing check (`_tc`), whole-beat check (`_tb`), short-bar fix (`_jf`) | on only if significantly better on a condition and worse on none |
| engraving (tidy, `_rd`, glissandos, grace notes and rolls) | not decided: how a rhythm is written, not what it is |

The verdict is for the labeller the release serves (`nl_gq_mc` if its training publish promoted it,
else `nl_gq`), written to `outputs/rule_verdict.md`, and published in the release manifest, which
the demo follows. A rule whose systems are missing keeps its setting. The report's numbers are
the models alone (`_bd`) and, separately, the demo's configuration with the rules the verdict kept.

### Pilot design: every idea at small scale (2026-11-26)

`pace/v2_pilot.sh` (pace/README.md, 4w) runs before any full-scale run.
- **A: ensembles, with no training.** The release's meter model on 5 views (`meter_v4_gq_tta`).
  Round 7's three PM2S meter models (michael branch: `meter_v4_gq_pm`, `_pm_s1`, `_pm_g30`)
  averaged (`meter_v4_gq_pm_ens3`).
- **B: synthetic performances of written scores** (`src/synthetic.py`). PDMX's public-domain
  MusicXML scores, rendered with a random tempo, rubato, timing noise, articulation and a mild
  accent of random strength; the grid exact. More data was the meter model's largest gain so far
  (sixth round), and synthetic performances can supply the meters the real data lacks. Pilot:
  3,000 scores, meter_v4_gq's recipe for 12k steps without and with them (a quarter of every
  batch).
- **C: PM2S with the ninth round's stable optimisation.** 16k steps at two rates, to check that the
  loss stays down before the 2 x 7.5 h full run.
- **D: the deletion threshold at 0.8** on the release.
- **E: choosing seq2seq or v2 per piece.** Its ceiling only. On round 7's scores (michael branch),
  taking the better system on every performance gains 0.11 over seq2seq alone with v2's Beat This!
  path, and 0.63 with its PM2S path (real recordings). On Gangqinpu, v2 leads by over 3 points. A
  selector without the reference gets part of that at best.
- **Considered and not run:**
  - choosing between candidate grids by the labeller's confidence: the fifth round measured that
    confidence on the level choice, and its best setting on validation was never to switch;
  - a larger meter model: it overfit in the third round;
  - training the labeller on the meter model's grids: its targets are the score, which a wrong
    grid cannot express.

### Eleventh round design: meter ensembles (2026-11-26)

`pace/v2_ensemble.sh` (pace/README.md, 4v). The meter model is the bottleneck, and it is noisy:

- **The grid decides.** The labeller's window onset F1 is 0.78-0.86 on the true grid and 0.26-0.49
  on the meter models' grids. On the oracle grid (every meter head right on the same tracked beats)
  it is 0.66-0.72. A different labeller barely moves it (eighth round: 0.498 for both on the same
  grid), so the gap is the meter model's.
- **One recipe, very different models.** Third round: three seeds of `meter_v3` at 0.57-0.64
  time-signature accuracy on PERiScoPe, up to 0.09 apart on the real test set. Eighth round:
  `meter_v3_s1_tp` +0.93 MUSTER against `meter_v3_tp` on clean whole pieces. The michael branch's
  seventh round: `meter_v4_gq_pm`'s second seed -0.25 / -0.14 (real / clean). Most changes measured
  are smaller than that.
- **Bigger does not help.** `meter_v3_large` (14.2M parameters against 4.8M) overfit: worse on
  every test set.
- **What the errors are.** Most wrong bars are at the wrong note-value level or bar length (3/4 as
  3/8, 2/2 as 2/4, 6/4 as 6/8, 2/4 as 4/4): choices that seeds disagree on.

Variance across seeds with no gain from size is what averaging models removes. So the round:
- **`meter_v4_gq_s1`, `meter_v4_gq_s2`:** the release's recipe, seeds 1 and 2. They show the
  spread on today's system, and they are the ensemble's members.
- **`src/meter.py` MeterEnsemble:** the members' probabilities averaged per beat and head (a
  mixture: the log of the mean probability) before the consistent decoder.
  - A view is the performance transposed or stretched in time, within the training augmentation's
    range (`TTA_VIEWS`: as is, -3 / +3 semitones, x0.9 / x1.1). The beats stay the same beats, so
    the outputs line up.
  - Averaged probabilities are less peaked than one model's, so each ensemble gets its own
    decoder, tuned on validation exactly as the release's (`--tune --tempo-prior`).
- **Systems:** `meter_v4_gq_tta` (the release's model on 5 views: no training), `meter_v4_gq_ens3`
  (3 seeds), `meter_v4_gq_ens3t` (3 seeds x 5 views), and `meter_v4_gq_ens4` when this account's
  `meter_v4_gq` differs from the published one (a fourth member for free).
- **Measured:** whole pieces with `nl_gq` at the release's settings on ASAP and Gangqinpu, paired
  against `nl_gq` + `meter_v4_gq`. Also the deletion threshold at 0.8, on the release and on
  `ens3t`: on the michael branch's seventh round it was -0.03 / -0.13, small but sure on both.
- **The verdict** (`scripts/ensemble_report.py`), the project's rule for a change: better beyond
  chance on a condition and worse on none.
- **Cost:** an ensemble of n members x v views runs the meter model n x v times. It is 4.8M
  parameters, run once per piece: well under a second per piece on a GPU, a few seconds on a CPU.
- **The release.** An ensemble is not a single checkpoint, so `publish_model.py` does not publish
  it. If one wins, the release format needs a meter slot with several members.

### Repo sweep, 2026-11-22: what was wrong, and what it touched

The whole repository was reviewed for errors (four reviews in parallel, each finding reproduced
before fixing, plus a check that the written scores say exactly what the models wrote: tokens ->
MusicXML -> tokens, compared note by note in sounding time). Fixed, with tests:

**Data built from the scores** (`src/preprocessing`; they take effect when the data is rebuilt:
`pace/rebuild.sh`, which keeps the data in use, the token ids (`build_dataset.py --vocab`) and the
feature scaling, so existing models read the new data):
- **Ties in chords.** A chord note without a tie took the chord's first tie (music21's
  `Chord.tie`): about 11% of the tied chord notes in the targets are really struck (17 of 25
  sampled ASAP scores; 69 in Scriabin's Sonata 5). The targets wrote them as continuations, the
  note labels made their performed notes "delete", and the tokens tied them to an earlier note.
- **Pickups.** A score's first, incomplete measure was written from beat 1 (a one-beat pickup in
  4/4 at `pos_0`, then a bar line a beat later). It is now written where it falls, starting
  mid-bar (Beethoven 10-1, Brahms 118/2, Ravel's Ondine in the test set).
- **One piece in train and val.** Only `_no_repeat(s)` was stripped from piece names, so Liszt's
  Paganini Etude 6 and its `_repeat` version were in train and val. Every version suffix
  (`_repeat`, `_no_2_repeat`, `_no_first_repeat`) now maps to one composition.
- Smaller: performances of two folders of one composition now add up in the split weighting; a
  beat time map stays increasing through two backward beats in a row (no ASAP file has them).

**Model inputs and training:**
- **Context notes.** A note struck just before a window's cut but released before it carries
  over was in neither the window nor its context in training (53% of val windows lost such
  notes, ~7 of the most recent ones), while the inference pipeline gives them as context. Now
  they are context in both. Every model with context notes was trained without them.
- Duplicate context notes (a note seen from two windows rounded to neighbouring milliseconds):
  27 of 3,000 training windows; matched within 1 ms now.
- **Resume.** A run resumed after the 8-hour limit replayed its current epoch from the start, and
  every later epoch repeated an earlier epoch's sample (the global RNG was re-seeded). An epoch's
  batches now depend on (seed, epoch) only, and a resumed run skips the batches it trained on.
- **Epoch length.** It was estimated from the token budget only; with context notes the note
  budget binds, so an "epoch" was 13% (64/32 context) to 43% (128/64) shorter than one pass, and
  the schedule, patience and cooldown with it. Both budgets count now (future runs).

**v2:**
- **Lost bars in the meter grid.** The decoder allows a step longer than a bar (a missed beat in
  2/4 tracked at half notes: 4 quarters), but the grid rebuilt each step from the positions,
  modulo the bar: such a step lost a bar, and every later bar number shifted. Even the oracle grid
  lost bars in 15 of the 59 ASAP test performances. The decoder now returns its steps and the
  grid writes every bar a step passes (oracle on the ASAP test: downbeat F 0.901 -> 0.907). All
  meter results and grids are decoded again (`pace/v2.sh`; older ones are marked "(old decoder)").
- **Pickup notes before the grid.** A grid starts at the bar line of the first tracked beat's
  bar, so notes played before it sat at negative positions of bar 0, on top of its downbeat (9 of
  59 ASAP test performances). Bars are now put in front (`src/meter.py cover_notes`), so they are
  a partial first bar.
- **Carried-over notes in the labeller's evaluation.** A window's notes carried over from the
  window before have no label to learn, yet their untrained outputs were rendered and scored. They
  are left out now, as in the whole-piece pipeline and the label ceiling; every labeller is
  evaluated again (older results are marked "(old eval)").

**Written scores** (what the demo shows and the comparison with Beyer & Dai rates):
- **Grace notes** went into the first voice even while it held a longer note; the MusicXML then
  shifted everything after them in that voice (12 notes in one bar of Schumann's Arabeske). They
  go into a free voice now.
- **Bars fuller than their time signature** (model output, and a few cadenza bars) made music21's
  writer move the overflow into the next bar and shift its notes. A note past its bar's end is now
  written at the start of the next bar, a note held across the bar line is tied over: the time it
  sounds is what the tokens (and the metrics) say.
- **Fallbacks per bar.** One bar the notation cannot express exactly (a 2048th gap rest, a
  zero-length note) snapped the whole piece to a 1/48 grid (156 quintuplet / septuplet notes of
  Chopin's Ballade moved). Only the bars that fail are snapped now.
- Plus the three fixes above: spelling for the key signature, playback at the performance's tempo,
  the speed fix.

Measured on 300 ASAP test windows (ground truth): notes whose sounding time differs between the
tokens and the written score went from 572 to 446 of 38,785, what is left being bars the notation
cannot express exactly; on the demo's real outputs, none.

**Jobs:** `pace/reevaluate.sh` no longer submits seq2seq evaluations for labeller and meter runs
(they crashed after queueing); a job waiting on another's success is cancelled when that fails
instead of pending forever; Beyer & Dai's prediction step deletes a piece's old score before
predicting it again; the synthetic-noise baselines get the very noise draws the models are scored
on (`scripts/run_baselines.py`; the stored baseline numbers used an unpaired draw of the same
noise until it is run again); Beat This!'s weights go to scratch, not the 30 GB home.
