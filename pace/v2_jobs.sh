#!/bin/bash
# The work of pace/v2.sh's jobs (each job: source pace/common.sh; bash pace/v2_jobs.sh <task> ...).
# Several models share one GPU job, so the batch stays under PACE's per-user job limit.
# Every run a task finishes training (or, for a meter model, evaluates) in that job is published to
# the Hugging Face model repo (a run finished before: python scripts/publish_model.py --run <it>)
# (scripts/publish_model.py: archived under runs/<run>/, made the demo's default when it is at least
# as good as the one there): meters, labellers, evals (a labeller it trained) and seq2seq. A failed
# upload (not logged in: `hf auth login` once on PACE) never fails the job. PUBLISH=0: none.
#
#   meters <run>...        train these meter models side by side (each its own log), up to
#                          --max-hours, resuming from last.pt; then tune + evaluate every finished
#                          one not evaluated yet (meter_eval_flags: --tune, or the run's own).
#                          Fails unless all are trained and evaluated.
#   labellers <run>...     train these labellers side by side (resuming), then evaluate each on
#                          the true grid. Fails unless all are trained.
#   grids                  the earlier meter models' grids decoded again (GRIDS_REEVAL: the decoder
#                          now keeps the bars of a step longer than a bar), the oracle grids
#                          (eval_meter.py --oracle), then Beat This! on the clean test MIDI rendered
#                          to audio (synth_beats.py)
#   evals <labeller>...    for each: finish its training if it is not done (a labeller trained
#                          here), else evaluate it on the true grid (an earlier labeller: its
#                          results predate the current evaluation); then on every grid: the oracle,
#                          GRIDS_OLD, METERS (a grid not there is skipped with the reason), on
#                          EVAL_CONDITIONS
#   systems <spec>...      whole pieces (scripts/system_scores.py): <labeller>+<meter> = v2 (only
#                          with both models finished and the meter model evaluated), with options
#                          after a colon, comma-separated: tidy (--tidy), gl (--glissando), tc
#                          (--timing-check), tb (--timing-beats), bd (--beat-decode), rd (--readable),
#                          jf (--fix-jumps), tv (--two-voices),
#                          d<percent> (--delete-above,
#                          e.g. d95), keepall (--delete-above 1), missing (--only-missing: what an
#                          earlier job wrote is kept), pm (PM2S's beats on clean MIDI: --clean-beats
#                          pm2s), rpm (PM2S's beats on the real transcriptions: --real-beats pm2s), rsb (Beat This! on
#                          the input MIDI rendered: --real-beats synth); <run>:<mode> = seq2seq, <run>:<mode>:sb =
#                          seq2seq with the rendered-MIDI beats on clean MIDI (a finished seq2seq run).
#                          SYS_ARGS: more system_scores.py options for every spec (another manifest:
#                          Gangqinpu's test pieces, --manifest/--out/--conditions)
#   tempo <meter> <new>    <new>: <meter>'s checkpoint with its decoder tuned again, the written-tempo
#                          prior included (eval_meter.py --tune --tempo-prior), then evaluated and its
#                          grids saved; <meter>'s own results stay as they are
#   seq2seq <run> <args>   train a seq2seq model (scripts/train_model.py <args>, resuming from
#                          last.pt), then evaluate it (SEQ_MODES on SEQ_CONDITIONS, only what it
#                          lacks). Fails unless trained and evaluated: the next job goes on.
#   levelexp <meter>       which evidence outside the decoder picks a piece's note-value level
#                          (scripts/level_experiments.py: the written-tempo / density priors, the
#                          labeller's confidence), on validation and test -> outputs/level_experiments
#   ablation <spec>        v2 <labeller>+<meter> on clean MIDI with the beats Beat This! tracked in the
#                          performances' real recordings, next to the same system's rendered-MIDI
#                          beats, in outputs/system_eval_ablation (scored on the shared pieces)
#   refine <meter> <new> <snap> <fill>
#                          <new>: <meter>'s checkpoint and decoder (decode.json copied) with the grid's
#                          beat times taken from the notes (eval_meter.py --snap / --fill, src/meter.py
#                          refine_grid), evaluated and its grids saved; <meter> = oracle: the oracle
#                          grids refined alike (eval_meter.py --oracle)
#   trackers <meter>...    these finished meter models evaluated with the real tracker(s) they were
#                          trained for: PM2S_RUNS with PM2S's beats on every condition (eval_meter.py
#                          --tracker pm2s: <condition>_pm2s), the others with Beat This!'s real beats
#                          of Gangqinpu's test recordings (--real-tracked: gangqinpu_real; a model
#                          that reads PM2S as well gets its beats there too)
#   beatdecode <nl>...     the labeller's beat decoding (src/beat_decode.py) tuned on validation
#                          (scripts/tune_beat_decode.py: <nl>/beat_decode.json), the labeller
#                          evaluated with it (true grid, meter_v4_gq's grids: modes *_bd), published
#   localtempo <meter>     <meter>_lt: the same checkpoint with the decoder's local tempo term
#                          (src/meter.py local_tempo_logprior) tuned on validation and evaluated
#   piecetempo <meter>     <meter>_pt: the same checkpoint with the decoder's piece-tempo prior
#                          (src/meter.py piece_tempo_logprior) tuned on validation and evaluated
#   gqbase <meter>...      these finished meter models evaluated on the Gangqinpu test set too
#                          (eval_meter.py --conditions gangqinpu, their own decoders), so the new
#                          models trained with it have something to beat
#
# Run options come from pace/v2.sh (METERS, GRIDS_OLD, NEW_NLS and the flags below).
set -uo pipefail
METERS=${METERS-"meter_v3 meter_v3_s1 meter_v3_s2 meter_v3_large"}
NEW_NLS=${NEW_NLS-"nl_base_gn nl_base_gn_s1"}
GRIDS_OLD=${GRIDS_OLD-"meter_v2_nonote"}
GRIDS_REEVAL=${GRIDS_REEVAL-"meter_v2_nonote meter_v2 meter_v2_s1 meter_v2_asap meter_base"}
MAX_HOURS=${MAX_HOURS:-7.5}
PM2S_RUNS=${PM2S_RUNS-"meter_v4_gq_pm meter_v4_gq_pmho meter_v4_gq_pmhs"}   # meter runs trained for PM2S's beats (the trackers task tests them with PM2S)
SEQ_MODES=${SEQ_MODES:-"oracle,none,guarded,meter"}
SEQ_CONDITIONS=${SEQ_CONDITIONS:-"clean,synthetic,real,periscope"}
EVAL_CONDITIONS=${EVAL_CONDITIONS:-"clean,synthetic,real,periscope,gangqinpu,gangqinpu_real"}   # labellers on meter grids
SYS_ARGS=${SYS_ARGS:-}
JOB=${SLURM_JOB_ID:-local}

meter_flags() {
  case "$1" in
    meter_v3) echo "--steps 60000 --patience 8" ;;
    meter_v3_s1) echo "--steps 60000 --patience 8 --seed 1" ;;
    meter_v3_s2) echo "--steps 60000 --patience 8 --seed 2" ;;
    meter_v3_large) echo "--steps 60000 --patience 8 --d-model 384 --layers 8" ;;
    # the piece's written tempo, for its note-value level (src/meter.py choose_level): a head on
    # meter_v3's frozen trunk, and meter_v3 fine-tuned with it
    meter_v3_th) echo "--init outputs/runs/meter_v3/best.pt --tempo-head --freeze-trunk --select tempo --steps 6000 --lr 1e-3 --warmup 200 --eval-every 1000 --patience 3" ;;
    meter_v3_ft) echo "--init outputs/runs/meter_v3/best.pt --tempo-head --select both --steps 16000 --lr 1e-4 --warmup 500 --eval-every 2000 --patience 4" ;;
    # Gangqinpu (build_meter_data.py --gangqinpu), half of every batch, with the trackers' real beats
    # in half of the draws that have them and their simulated errors in the rest. Which tracker:
    #   meter_v4_gq       Beat This! (on Gangqinpu its errors fitted there: off-beats included)
    #   meter_v4_gq_pm    PM2S (its fitted errors; the real transcriptions get the simulated PM2S
    #                     instead of Beat This!'s recorded beats)
    #   meter_v4_gq_both  Beat This!, with PM2S's beats as evidence on them (--aux pm2s)
    meter_v4_gq) echo "--gangqinpu --gangqinpu-share 0.5 --steps 60000 --patience 8 --tracked beat_this --tracked-share 0.5 --beat-noise-gangqinpu configs/beat_noise_bt_gangqinpu.json" ;;
    meter_v4_gq_both) echo "--gangqinpu --gangqinpu-share 0.5 --steps 60000 --patience 8 --tracked beat_this --tracked-share 0.5 --beat-noise-gangqinpu configs/beat_noise_bt_gangqinpu.json --aux pm2s --aux-noise configs/beat_noise_pm2s.json" ;;
    meter_v4_gq_pm) echo "--gangqinpu --gangqinpu-share 0.5 --steps 60000 --patience 8 --beat-noise configs/beat_noise_pm2s.json --real-beats simulated --tracked pm2s --tracked-share 0.5" ;;
    # meter_v4_gq_pm for PM2S retrained without our test pieces (pace/v2_pm2s.sh: run with
    # PM2S_WEIGHTS / PM2S_TAG=ho set, so --tracked pm2s reads that PM2S's stored beats)
    meter_v4_gq_pmho) echo "--gangqinpu --gangqinpu-share 0.5 --steps 60000 --patience 8 --beat-noise configs/beat_noise_pm2s_ho.json --real-beats simulated --tracked pm2s --tracked-share 0.5" ;;
    # and for PM2S retrained with stable optimisation (pace/v2_pm2s2.sh: PM2S_WEIGHTS / PM2S_TAG=hos)
    meter_v4_gq_pmhs) echo "--gangqinpu --gangqinpu-share 0.5 --steps 60000 --patience 8 --beat-noise configs/beat_noise_pm2s_hos.json --real-beats simulated --tracked pm2s --tracked-share 0.5" ;;
    # meter_v4_gq trained longer (its best checkpoint was at 56,000 of its 60,000 steps)
    meter_v4_gql) echo "--gangqinpu --gangqinpu-share 0.5 --steps 100000 --patience 8 --tracked beat_this --tracked-share 0.5 --beat-noise-gangqinpu configs/beat_noise_bt_gangqinpu.json" ;;
    # (earlier: meter_v3 fine-tuned on the Gangqinpu mix)
    meter_v3_gqft) echo "--init outputs/runs/meter_v3/best.pt --gangqinpu --gangqinpu-share 0.5 --steps 16000 --lr 1e-4 --warmup 500 --eval-every 2000 --patience 4 --tracked beat_this --tracked-share 0.5" ;;
    *) echo "unknown meter run $1 (add it to meter_flags() in pace/v2_jobs.sh)" >&2; return 1 ;;
  esac
}
meter_eval_flags() {  # how a finished meter run is tuned before its evaluation
  case "$1" in
    # its bar / step / time-signature heads are meter_v3's: meter_v3_tp's decoder, the level tuned on top
    meter_v3_th) echo "--level --decode-from outputs/runs/meter_v3_tp/decode.json" ;;
    meter_v3_ft) echo "--tune --tempo-prior --level" ;;
    meter_v4_gq|meter_v4_gq_both|meter_v3_gqft|meter_v4_gql) echo "--tune --tempo-prior" ;;   # as the release's meter_v3_tp
    # its decoder tuned for the tracker it was trained for (tested with it by the trackers task)
    meter_v4_gq_pm) echo "--tune --tempo-prior --tune-noise configs/beat_noise_pm2s.json" ;;
    meter_v4_gq_pmho) echo "--tune --tempo-prior --tune-noise configs/beat_noise_pm2s_ho.json" ;;
    meter_v4_gq_pmhs) echo "--tune --tempo-prior --tune-noise configs/beat_noise_pm2s_hos.json" ;;
    *) echo "--tune" ;;
  esac
}
nl_flags() {
  case "$1" in
    nl_base_gn) echo "--size base --grid-noise 0.8" ;;
    nl_base_gn_s1) echo "--size base --grid-noise 0.8 --seed 1" ;;
    nl_base_gn_f1) echo "--size base --grid-noise 0.8 --select onset_f1" ;;
    nl_base_gn1_f1) echo "--size base --grid-noise 1.0 --select onset_f1" ;;
    # nl_base_gn's recipe with Gangqinpu's scores as half of the training windows (scripts/build_gangqinpu_windows.py)
    nl_gq) echo "--size base --grid-noise 0.8 --gangqinpu-share 0.5" ;;
    # nl_gq's recipe, the training notes whose written split the performance contradicts left out of
    # the loss (src/notelabels.py contradicted; scripts/audit_labels.py)
    nl_gq_mc) echo "--size base --grid-noise 0.8 --gangqinpu-share 0.5 --mask-contradicted" ;;
    *) echo "unknown labeller run $1 (add it to nl_flags() in pace/v2_jobs.sh)" >&2; return 1 ;;
  esac
}
done_run() { grep -q '"phase": "done"' "outputs/runs/$1/status.json" 2>/dev/null; }
PUBLISH=${PUBLISH:-1}
publish() {  # a finished, evaluated run -> the Hugging Face repo (never fails the job)
  [ "$PUBLISH" = 0 ] && return 0
  python scripts/publish_model.py --run "outputs/runs/$1" || echo "$1: not published (see above; the job goes on)"
}
resume() { [ -f "outputs/runs/$1/last.pt" ] && echo --resume; }
in_list() { case " $2 " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

train_labeller() {  # name -> trains (or resumes) it; logs to slurm-labeller_<name>-<job>.log
  local nl=$1 f
  f=$(nl_flags "$nl") || return 1
  # shellcheck disable=SC2086
  python scripts/train_labeller.py --name "$nl" $f --normal-priority --workers 7 $(resume "$nl") \
    > "slurm-labeller_$nl-$JOB.log" 2>&1
}

seq_ready() {  # a seq2seq run to export: its best.pt, and finished (runs older than status.json are)
  if [ ! -f "outputs/runs/$1/best.pt" ] || { [ -f "outputs/runs/$1/status.json" ] && ! done_run "$1"; }; then
    echo "$1: skipped, not finished (outputs/runs/$1)"
    return 1
  fi
}

v2_flags() {  # the options after a v2 spec's colon -> system_scores.py flags
  local o out=""
  for o in ${1//,/ }; do
    case "$o" in
      tidy) out="$out --tidy" ;;
      gl) out="$out --glissando" ;;
      tc) out="$out --timing-check" ;;
      tb) out="$out --timing-beats" ;;
      bd) out="$out --beat-decode" ;;
      rd) out="$out --readable" ;;
      jf) out="$out --fix-jumps" ;;
      tv) out="$out --two-voices" ;;
      keepall) out="$out --delete-above 1" ;;
      missing) out="$out --only-missing" ;;
      pm) out="$out --clean-beats pm2s" ;;
      rpm) out="$out --real-beats pm2s" ;;
      rsb) out="$out --real-beats synth" ;;
      d[0-9]*) out="$out --delete-above 0.${o#d}" ;;
      *) echo "unknown v2 option $o (tidy, gl, tc, tb, bd, rd, jf, tv, d<percent>, keepall, missing, pm, rpm, rsb)" >&2; return 1 ;;
    esac
  done
  echo "$out"
}

task=${1:?task: meters, labellers, grids, evals, systems, seq2seq, ablation, tempo, levelexp, refine, gqbase, trackers, beatdecode, localtempo, piecetempo}
shift
case "$task" in
  meters)
    pids=()
    for m in "$@"; do
      f=$(meter_flags "$m") || exit 1
      if done_run "$m"; then echo "$m: trained already"; continue; fi
      echo "$m: training ($f), log slurm-met_$m-$JOB.log"
      # shellcheck disable=SC2086
      python scripts/train_meter.py --name "$m" $f --workers 6 --normal-priority --max-hours "$MAX_HOURS" \
        $(resume "$m") > "slurm-met_$m-$JOB.log" 2>&1 &
      pids+=("$!")
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "$p"; done
    ok=0
    for m in "$@"; do
      if ! done_run "$m"; then echo "$m: not done yet (the next job resumes it)"; ok=1; continue; fi
      if [ ! "outputs/runs/$m/eval_meter.json" -nt "outputs/runs/$m/best.pt" ]; then
        # shellcheck disable=SC2046
        python scripts/eval_meter.py --run "$m" $(meter_eval_flags "$m") && publish "$m" || ok=1
      fi
      [ -f "outputs/runs/$m/eval_meter.json" ] || ok=1
    done
    exit $ok ;;
  labellers)
    pids=()
    fresh=""
    for nl in "$@"; do
      if done_run "$nl"; then echo "$nl: trained already"; continue; fi
      echo "$nl: training ($(nl_flags "$nl")), log slurm-labeller_$nl-$JOB.log"
      train_labeller "$nl" &
      pids+=("$!")
      fresh="$fresh $nl"
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "$p"; done
    ok=0
    for nl in "$@"; do
      if ! done_run "$nl"; then echo "$nl: not done (its evals job resumes it)"; ok=1; continue; fi
      python scripts/eval_labeller.py --run "$nl" --normal-priority || { ok=1; continue; }
      in_list "$nl" "$fresh" && publish "$nl"   # finished in this job: published
    done
    exit $ok ;;
  beatdecode)   # <labeller>...: its beat decoding tuned on validation, the labeller evaluated with it, published
    ok=0
    for nl in "$@"; do
      if ! done_run "$nl"; then echo "$nl: skipped, not finished"; ok=1; continue; fi
      python scripts/tune_beat_decode.py --run "$nl" --normal-priority || { ok=1; continue; }
      python scripts/eval_labeller.py --run "$nl" --beat-decode --normal-priority || ok=1
      python scripts/eval_labeller.py --run "$nl" --beat-decode --grid meter:meter_v4_gq --conditions "$EVAL_CONDITIONS" \
        --normal-priority || ok=1
      publish "$nl"   # the default's files changed (beat_decode.json): the demo pulls it
    done
    exit $ok ;;
  localtempo)   # <meter>: a copy <meter>_lt (the same checkpoint) with the local tempo term tuned, evaluated
    m=${1:?localtempo <meter>}
    lt="${m}_lt"
    if ! done_run "$m" || [ ! -f "outputs/runs/$m/decode.json" ]; then echo "$m: not finished or not tuned"; exit 1; fi
    mkdir -p "outputs/runs/$lt"
    ln -sf "../$m/best.pt" "outputs/runs/$lt/best.pt"
    for f in config.json status.json decode.json; do cp "outputs/runs/$m/$f" "outputs/runs/$lt/$f"; done
    python scripts/eval_meter.py --run "$lt" --local-tempo || exit 1
    exit 0 ;;
  piecetempo)   # <meter>: a copy <meter>_pt (the same checkpoint) with the piece-tempo prior tuned, evaluated
    m=${1:?piecetempo <meter>}
    pt="${m}_pt"
    if ! done_run "$m" || [ ! -f "outputs/runs/$m/decode.json" ]; then echo "$m: not finished or not tuned"; exit 1; fi
    mkdir -p "outputs/runs/$pt"
    ln -sf "../$m/best.pt" "outputs/runs/$pt/best.pt"
    for f in config.json status.json decode.json; do cp "outputs/runs/$m/$f" "outputs/runs/$pt/$f"; done
    python scripts/eval_meter.py --run "$pt" --piece-tempo || exit 1
    exit 0 ;;
  grids)
    ok=0
    for m in $GRIDS_REEVAL; do  # their tuned decoder (decode.json) kept
      [ -f "outputs/runs/$m/best.pt" ] || { echo "$m: no best.pt, skipped"; continue; }
      python scripts/eval_meter.py --run "$m" --device cpu || ok=1
    done
    python scripts/eval_meter.py --run meter_oracle --oracle --device cpu || ok=1
    python scripts/synth_beats.py || ok=1
    exit $ok ;;
  evals)
    ok=0
    for nl in "$@"; do
      if in_list "$nl" "$NEW_NLS" && ! done_run "$nl"; then
        echo "$nl: not trained yet, resuming"
        train_labeller "$nl" || { echo "$nl: training failed"; ok=1; continue; }
        python scripts/eval_labeller.py --run "$nl" --normal-priority && publish "$nl" || ok=1
      elif ! in_list "$nl" "$NEW_NLS"; then
        echo "== $nl on the true grid"
        python scripts/eval_labeller.py --run "$nl" --normal-priority || ok=1
      fi
      for m in meter_oracle $GRIDS_OLD $METERS; do
        echo "== $nl on $m's grids"
        python scripts/eval_labeller.py --run "$nl" --grid "meter:$m" --conditions "$EVAL_CONDITIONS" \
          --normal-priority || ok=1
      done
    done
    exit $ok ;;
  systems)
    ok=0
    for spec in "$@"; do
      echo "== $spec"
      case "$spec" in
        *+*) base=${spec%%:*}; opts=""
             [ "$base" != "$spec" ] && opts=${spec#*:}
             nl=${base%%+*}; m=${base##*+}
             if ! done_run "$nl" || ! done_run "$m" || [ ! -f "outputs/runs/$m/eval_meter.json" ]; then
               echo "$spec: skipped, $nl or $m not finished (or $m not evaluated)"; ok=1; continue
             fi
             f=$(v2_flags "$opts") || { ok=1; continue; }
             # shellcheck disable=SC2086
             python scripts/system_scores.py --v2 "$base" $f $SYS_ARGS || ok=1 ;;
        *:*:sb) r=${spec%%:*}; rest=${spec#*:}
                seq_ready "$r" || { ok=1; continue; }
                python scripts/system_scores.py --run "$r" --mode "${rest%%:*}" --midi-beats synth $SYS_ARGS || ok=1 ;;
        *:*) seq_ready "${spec%%:*}" || { ok=1; continue; }
             python scripts/system_scores.py --run "${spec%%:*}" --mode "${spec##*:}" $SYS_ARGS || ok=1 ;;
        *) echo "unknown system $spec"; ok=1 ;;
      esac
    done
    exit $ok ;;
  seq2seq)
    r=${1:?seq2seq <run> <train_model.py args>}
    shift
    trained=0
    if ! done_run "$r"; then
      # shellcheck disable=SC2046
      python scripts/train_model.py --name "$r" --normal-priority --workers 7 $(resume "$r") "$@"
      trained=1
    fi
    done_run "$r" || { echo "$r: not done yet (the next job resumes it)"; exit 1; }
    python scripts/evaluate_pieces.py --run "$r" --normal-priority --musicxml 5 --only-missing \
      --modes "$SEQ_MODES" --conditions "$SEQ_CONDITIONS" || exit 1
    [ "$trained" = 1 ] && publish "$r"   # finished in this job: published
    exit 0 ;;
  ablation)
    spec=${1:?ablation <labeller>+<meter>}
    sys="v2_${spec%%+*}_${spec##*+}"
    ok=0
    python scripts/system_scores.py --v2 "$spec" --conditions clean --clean-beats audio \
      --out outputs/system_eval_ablation || ok=1
    if [ -f "outputs/system_eval/$sys/summary.json" ]; then  # the rendered-MIDI beats' version, made before
      ln -sfn "$PWD/outputs/system_eval/$sys" "outputs/system_eval_ablation/$sys"
    else
      echo "outputs/system_eval/$sys missing: no rendered-MIDI version to compare with"; ok=1
    fi
    exit $ok ;;
  tempo)
    src=${1:?tempo <meter run> <new run>}; dst=${2:?tempo <meter run> <new run>}
    if ! done_run "$src" || [ ! -f "outputs/runs/$src/best.pt" ]; then echo "$src: not finished"; exit 1; fi
    mkdir -p "outputs/runs/$dst"
    for f in best.pt config.json status.json; do
      [ -e "outputs/runs/$src/$f" ] && ln -sfn "$PWD/outputs/runs/$src/$f" "outputs/runs/$dst/$f"
    done
    python scripts/eval_meter.py --run "$dst" --tune --tempo-prior --device cpu
    exit $? ;;
  levelexp)
    m=${1:?levelexp <meter run>}
    if [ ! -f "outputs/runs/$m/best.pt" ] || [ ! -f "outputs/runs/$m/decode.json" ]; then
      echo "$m: no best.pt / decode.json"; exit 1
    fi
    python scripts/level_experiments.py --meter "$m"
    exit $? ;;
  refine)
    src=${1:?refine <meter run|oracle> <new run> <snap s> <fill beats>}; dst=${2:?refine: <new run>}
    snap=${3:?refine: <snap s>}; fill=${4:?refine: <fill beats>}
    if [ "$src" = oracle ]; then
      python scripts/eval_meter.py --run "$dst" --oracle --snap "$snap" --fill "$fill" --device cpu
      exit $?
    fi
    if ! done_run "$src" || [ ! -f "outputs/runs/$src/best.pt" ] || [ ! -f "outputs/runs/$src/decode.json" ]; then
      echo "$src: not finished or not tuned"; exit 1
    fi
    mkdir -p "outputs/runs/$dst"
    for f in best.pt config.json status.json; do
      [ -e "outputs/runs/$src/$f" ] && ln -sfn "$PWD/outputs/runs/$src/$f" "outputs/runs/$dst/$f"
    done
    cp "outputs/runs/$src/decode.json" "outputs/runs/$dst/decode.json"   # its own: --snap / --fill are written in
    python scripts/eval_meter.py --run "$dst" --snap "$snap" --fill "$fill" --device cpu
    exit $? ;;
  trackers)
    ok=0
    for m in "$@"; do
      if [ ! -f "outputs/runs/$m/best.pt" ] || [ ! -f "outputs/runs/$m/eval_meter.json" ]; then
        echo "$m: not finished or not evaluated, skipped"; ok=1; continue
      fi
      if in_list "$m" "$PM2S_RUNS"; then   # trained for PM2S: its beats, on every condition
        python scripts/eval_meter.py --run "$m" --tracker pm2s || ok=1
      else   # Beat This! (and PM2S's as evidence, for a model that reads it): its real beats of the recordings
        python scripts/eval_meter.py --run "$m" --real-tracked --conditions gangqinpu || ok=1
      fi
    done
    exit $ok ;;
  gqbase)
    ok=0
    for m in "$@"; do
      if [ ! -f "outputs/runs/$m/best.pt" ]; then echo "$m: no best.pt, skipped"; ok=1; continue; fi
      python scripts/eval_meter.py --run "$m" --conditions gangqinpu --device cpu || ok=1
    done
    exit $ok ;;
  *) echo "unknown task $task (meters, labellers, grids, evals, systems, seq2seq, ablation, tempo, levelexp, refine, gqbase, trackers)"; exit 2 ;;
esac
