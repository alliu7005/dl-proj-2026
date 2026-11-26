# Repo layout: every file

What each file does. [README](../README.md) has the short map.


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
src/pipeline.py                  audio / MIDI -> MusicXML: ScorePipeline (seq2seq, window by window), V2Pipeline (grid + labeller)
src/timing.py                    the timing check: a triplet-written note played on a straight eighth / sixteenth written there (quarter-beat meters, clear timing only)
src/playback.py                  the score MIDI plays grace notes (just before their beat) and rolled chords (spread upwards) as written: music21's MIDI export does not
src/glissando.py                 glissandos: found in the notes (fast one-way runs on one key colour), their two ends in the score, the gliss. line, the swept notes in the score MIDI
src/synth.py                     MIDI -> audio (additive piano), so Beat This! can track a MIDI input
src/release.py                   the released models (models/), their settings, where the demo and the CLI find them
scripts/export_models.py         the release from the training runs: no optimizer state, the right vocabulary, manifest + sha256, load check
scripts/download_models.py       the release from the Hugging Face model repo -> models/, checked against its manifest; update(): the demo's pull of a newer default
scripts/publish_model.py         a finished run -> the Hugging Face repo (runs/<run>/; its folder's default when it scores at least as well); pace/v2_jobs.sh calls it
demo/app.py, demo/static/        web demo (FastAPI + one HTML page); scripts/audio_to_score.py: CLI
scripts/train_model.py           train either model; scripts/evaluate_pieces.py: oracle/chained/none
src/eval/plan.py                 evaluation options and mode names; what a run's results still lack
src/beats.py                     beats: ground-truth grid, per-note features, tracker errors (BeatNoise)
scripts/build_beats.py           ground-truth beats per performance -> <split>/beats_gt.npz
scripts/track_beats.py           Beat This! on the MAESTRO audio (held-out folds) -> beats_tracked.npz
scripts/fit_beat_noise.py        tracker vs ground truth -> configs/beat_noise.json
scripts/compare_runs.py          tables of every evaluated run -> outputs/compare/{summary,margins}.md, results.csv
scripts/compare_systems.py       whole-piece systems paired by performance, bootstrap over pieces -> outputs/system_eval/paired.md
scripts/level_experiments.py     which evidence outside the decoder picks a piece's note-value level (priors, labeller confidence)
scripts/system_inputs.py         whole-piece test inputs (clean MIDI, cut Transkun transcriptions) + references
scripts/system_scores.py         our pipelines on them (seq2seq or v2) -> one MusicXML per performance
scripts/synth_beats.py           Beat This! on the clean test MIDI rendered to audio (held-out folds) -> beats_synth.npz
scripts/rescore.py               metrics added since an evaluation, from its saved predictions (onset-scale F1; results on the old targets re-scored)
scripts/beyer_dai.py             Baseline 3: Beyer & Dai's model (predict) and their metrics for every system (score)
src/notelabels.py                v2: per-note score labels, tie-aware rendering back to tokens, grid features
src/labeller.py                  v2: the note labeller (Scoreformer encoder + per-note heads), prediction -> tokens
scripts/build_notelabels.py      v2: labels + true grids per split -> <split>/notelabels.npz
scripts/label_ceiling.py         v2: every window rendered from its true labels (the labeller's ceiling)
scripts/train_labeller.py        v2: train the note labeller
scripts/eval_labeller.py         v2: evaluate it on the test sets (true grid: grid_true; a meter model's: grid_meter_<run>)
src/meter.py                     v2: the meter model (beat + note-encoder inputs, targets, alignment ground truth, augmentation, transformer, consistent Viterbi decoding, beat times refined from the notes)
src/preprocessing/gangqinpu.py   v2: Gangqinpu rows -> meter-model performances (transcribed notes, trusted bars' grids); its written scores -> score.py measures, labeller windows (aligned notes), references
src/pm2s_beats.py                PM2S's beat tracker (Liu et al. 2022, MIT, vendored): beats from the notes, an alternative to Beat This! for v2
scripts/build_meter_data.py      v2: whole performances with ground-truth grids: ASAP annotations (+ transcriptions, tracked beats), PERiScoPe score alignments, Gangqinpu (--gangqinpu)
scripts/fetch_gangqinpu.py       v2: download the Gangqinpu release's Parquet (gated: accept its licence on Hugging Face first)
scripts/compare_beats.py         PM2S against Beat This! on the whole-piece test inputs, scored against the ASAP beat annotations
scripts/fit_pm2s_noise.py        PM2S's beat-tracking errors fitted on Gangqinpu -> configs/beat_noise_pm2s.json (training a meter model for PM2S)
scripts/train_pm2s.py            PM2S's beat model retrained from scratch on our training splits only (pm2s_ho; PM2S_WEIGHTS / PM2S_TAG use it); --betas/--eps/--clip/--schedule cosine/--warmup: the stable settings
scripts/pick_run.py              the best of several finished runs by their own validation score, linked under one name (chosen.json)
scripts/fetch_gangqinpu_audio.py Gangqinpu's recordings from the URLs the release lists (politely, resumable; for Beat This! on real audio)
scripts/build_gangqinpu_windows.py v2: Gangqinpu's written scores as labeller / seq2seq windows -> data/processed/{train,val,test}_gangqinpu (+ report.json)
scripts/gangqinpu_inputs.py      whole-piece inputs + references from Gangqinpu's test split (aligned notes, Transkun on the recording, Beat This!'s beats) -> outputs/system_eval_gangqinpu
scripts/track_meter_data.py      Beat This! on Gangqinpu's recordings, PM2S on the notes: real beats stored for training (--tracked) and testing
scripts/train_meter.py           v2: train the meter model (ASAP + real + PERiScoPe, per-source validation)
scripts/eval_meter.py            v2: tune its decoder on validation, score its grids (clean / real / periscope), save them for the labeller; --oracle: the best grid the tracked beats allow
pace/train.sbatch                PACE GPU job: train + evaluate
pace/reevaluate.sh               evaluate finished runs: what is missing, one job per run or per test set
pace/beats.sh                    the beat batch (ground-truth beats, Beat This!, 3 beat runs)
pace/tsearch.sh                  the time-signature batch (search modes on the best models, 3 runs)
pace/bd_setup.sh                 Beyer & Dai's environment, code, forks and checkpoint (once)
pace/system_eval.sh              system-level comparison: ours vs Beyer & Dai on whole pieces
pace/labeller.sh                 v2 Phase 1: labels for every split, then two note labellers (true grid)
pace/train_labeller.sbatch       one note labeller: train + evaluate (resumes from last.pt)
pace/meter.sh                    v2 Phases 2 + 3: the meter model, then the labellers on its grids (second round)
pace/v2.sh                       v2 third round, two waves: meter models, oracle, grid-noise labellers, every labeller on every grid, whole-piece comparison
pace/v2_jobs.sh                  the work of pace/v2.sh's jobs (several models per GPU job)
pace/v2_round4.sh                v2 fourth round: tempo prior in the meter decoder, labellers kept on onset F1, length tidying, deletion settings, beat ablation
pace/rebuild.sh                  data/processed rebuilt with the 2026-11-22 data fixes (old data kept, same vocabulary)
pace/v2_release.sh               the release: its configuration scored on whole pieces, models/ built, paired comparisons
pace/v2_level.sh                 v2 fifth round: the written-tempo head for the note-value level, trained, tuned, scored
pace/v2_gangqinpu.sh             v2 sixth round: meter models trained with Gangqinpu; the grid's beat times taken from the notes
pace/v2_pm2s.sh                  v2 seventh round: PM2S retrained without our test pieces, its beats on whole pieces, a meter model for it
pace/v2_timing.sh                the timing check on whole pieces, ASAP and Gangqinpu (3 jobs; pace/v2_pm2s2.sh measures it too)
pace/v2_pm2s2.sh                 v2 ninth round: PM2S retrained with stable optimisation (two learning rates, the better picked), its meter model, MIDI input on ASAP and Gangqinpu
pace/v2_gqlabel.sh               v2 eighth round: the labeller trained with Gangqinpu's scores, whole pieces on Gangqinpu's test split, meter_v4_gq longer, the meter model's second seed
tests/test_scoreformer.py        model checks + chained prompts == training prompts
tests/test_decoding.py           chained / guarded / meter / time-signature search with a stand-in model
tests/test_pipeline.py           demo pipeline end to end: windowing, every mode, output budget, banned tokens
tests/test_context_notes.py      context notes: exact neighbours, unchanged window notes, both models
tests/test_metrical.py           metrical labels: synthetic precision / recall, coverage, the aux head
tests/test_training_stop.py      LR schedule, early stopping, cooldown, resume, status.json, atomic saves
tests/test_eval_plan.py          src/eval/plan.py rules + evaluate_pieces.py --only-missing end to end
tests/test_reevaluate.py         pace/reevaluate.sh (per run / per test set), pace/beats.sh, pace/tsearch.sh: what they submit
tests/test_system_eval.py        system-level comparison: inputs, our pipeline on them, aggregation, job graph
tests/test_notelabels.py         v2: labels, ties, rendering, grid features, GridNoise, dataset, labeller training / evaluation
tests/test_meter.py              v2: meter targets, decoding (vs a plain Viterbi), oracle, note encoder, alignment ground truth, augmentation, time limit + resume, refined beat times, end to end, job graph
tests/test_gangqinpu.py          v2: Gangqinpu notes, beat filters, grids, release reading, build / train / evaluate, batch shares, pace/v2_gangqinpu.sh
tests/test_gangqinpu_scores.py   v2: Gangqinpu's written scores (ties, graces, pickups, parts), windows and labels, extra-notes-only noise, build + train + evaluate, whole-piece inputs, pace/v2_gqlabel.sh
tests/test_pm2s.py               PM2S: post-processing, the tracker on random weights, the weights' hash, beat_source, system names
tests/test_publish.py            publishing: archive, the gate, --force, refusals, the demo's update + swap, the batch jobs' hook
tests/test_timing.py             the timing check: moved / kept notes, meters, lengths, the pipeline's switch, pace/v2_timing.sh
tests/test_playback.py           grace notes and rolled chords in the score MIDI: timing, rolls across staves, tied chords, nothing held on
tests/test_glissando.py          glissandos: the detector (and what it must not take), the ends, the line and the MIDI, V2Pipeline on / off, the switches
tests/test_pm2s_train.py         PM2S retrained: targets, augmentation, training and resuming, the stable settings, the weights' tag, pick_run.py, pace/v2_pm2s.sh and v2_pm2s2.sh, --real-beats synth
tests/test_tracked.py            real tracker output: the recordings' download, tracking the data, the real / simulated mix, real beats at test
tests/test_v2_system.py          v2 system: onset-scale F1, rescoring, the export (per-bar fallbacks, overflow, spelling, playback tempo, grace notes, rests), the synthesizer, V2Pipeline (staves, grace notes, deletion threshold, length tidying), the demo, pace/v2.sh, v2_round4.sh and v2_jobs.sh
tests/test_preprocessing.py      scores -> training data: chord ties, pickups, composition ids, time maps; plan.py leaves v2 runs alone
tests/test_compare_runs.py       the comparison tables: bold best, dashes, alignment, margin order, CSV; the paired system comparison
tests/test_release.py            the release: export (no optimizer, vocabulary, decoder, refusals), download + verify, pace/v2_release.sh
tests/test_beats.py              beats vs ASAP annotations, features, tracker errors (bar classes, octave stretches, profiles, fit, vs Beat This!), tracker folds, train / eval / demo
tests/test_context_noise.py      context corruption: off = unchanged, on = only the context changes
scripts/analyze_chained.py       why chained decoding loses notes (from saved predictions)
scripts/simulate_guard.py        guarded decoding estimated offline, per guard margin
scripts/export_musicxml.py       MusicXML from saved predictions (no GPU)
scripts/           build_dataset, make_bd_split, fetch_maestro_audio, transcribe, fit_noise,
                   build_transcribed, train_xgb_detector, audit_data
tests/test_integrity.py   leakage, pairing, array and loader checks
tests/test_roundtrip.py   tokens -> MusicXML -> tokens
docs/handoff.md           where things stand, open problems, options for next steps
docs/experiments.md       every round of experiments and its results
docs/data.md, demo.md, baselines.md, model_card.md, repo_layout.md   the rest of the documentation
```
