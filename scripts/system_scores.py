"""Our systems on the whole-piece inputs of scripts/system_inputs.py: one MusicXML score per performance.

    python scripts/system_scores.py --run sf_base_reg_cab                 # meter mode (the demo default)
    python scripts/system_scores.py --run sf_base_reg_cab --mode tsearch_beat
    python scripts/system_scores.py --run sf_base_reg_cab --midi-beats synth
    python scripts/system_scores.py --v2 nl_base_gn+meter_v3              # v2: labeller + meter model
    python scripts/system_scores.py --v2 nl_base_gn+meter_v3 --conditions clean --clean-beats audio                                     --out outputs/system_eval_ablation    # clean MIDI, real-audio beats
    -> outputs/system_eval/<system>/<condition>/<slug>.musicxml   (+ summary.json)
       system = <run>_<mode> (+ "_sb" with --midi-beats synth), or v2_<labeller>_<meter> (+ "_ab" with
       --clean-beats audio)

Runs the demo pipeline (src/pipeline.py) exactly as the demo does on a file: windows cut from the
performance itself (no score information). The seq2seq models decode window by window
(ScorePipeline); v2 (V2Pipeline) decodes the meter model's grid of the whole piece from its beats
and labels every note against it. Beats: on the real transcriptions, the beats Beat This! tracked
in the audio (data/processed/test_transcribed/beats_tracked.npz, same ASAP timeline as the cropped
MIDI): what the demo gets from audio. A clean MIDI file has no audio: a seq2seq beat model gets no
beats there (as the demo's MIDI input) unless --midi-beats synth; v2 always gets Beat This!'s beats
of the MIDI rendered to audio (scripts/synth_beats.py made them with the cross-validation model
that never heard the piece: outputs/system_eval/beats_synth.npz; tracked here if that is missing).
--clean-beats audio gives v2 on clean MIDI the beats Beat This! tracked in the performance's real
recording instead (the ASAP test performances with MAESTRO audio; the others are left out): what the
rendered audio costs, measured on the same pieces. --clean-beats pm2s gives it PM2S's beats of the
MIDI's notes (src/pm2s_beats.py, no audio; system _pm), and --real-beats pm2s PM2S's beats of the
real transcriptions instead of Beat This!'s of the recordings (_rpm): PM2S against Beat This!, on
the same pieces and metrics. PM2S's released weights were trained on all of ASAP, these test
performances included: optimistic until it is retrained without them.

Gangqinpu's test pieces (scripts/gangqinpu_inputs.py: --manifest
outputs/system_eval_gangqinpu/manifest.json --out outputs/system_eval_gangqinpu --conditions
gangqinpu,gangqinpu_real): every system gets Beat This!'s beats of the recording on both
(beats_gangqinpu.npz next to that manifest), as on the real ASAP transcriptions; --real-beats pm2s
gives v2 PM2S's beats of the input's notes there too.

A piece whose score cannot be written keeps its tokens and the error (<slug>.tokens.txt,
<slug>.error.txt next to where its score would be), so it can be reproduced; --only-missing runs
only the pieces a system has no score for yet (kept in its summary.json with the others).
scripts/beyer_dai.py score then rates the scores against the ASAP references.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.track_beats import checkpoint_for, load_folds  # noqa: E402
from src.beats import beats_path, load_beats  # noqa: E402
from src.pipeline import DELETE_ABOVE, MODES, ScorePipeline, V2Pipeline  # noqa: E402


def parse_v2(spec: str) -> tuple[str, str]:
    lab, sep, met = spec.partition("+")
    if not sep or not lab or not met:
        raise SystemExit(f"--v2 takes <labeller run>+<meter run>, not {spec!r}")
    return lab, met


BEAT_SUFFIX = {"synth": "", "audio": "_ab", "pm2s": "_pm"}       # v2 on clean MIDI
GANGQINPU = ("gangqinpu", "gangqinpu_real")   # scripts/gangqinpu_inputs.py: the recording's beats, as real
REAL_SUFFIX = {"audio": "", "pm2s": "_rpm", "synth": "_rsb"}     # v2 on the real transcriptions (and Gangqinpu's)


def system_name(run: str | None, mode: str, midi_beats: str, v2: str | None, clean_beats: str = "synth",
                delete_above: float = DELETE_ABOVE, tidy: bool = False, real_beats: str = "audio",
                glissando: bool = False, timing_check: bool = False, readable: bool = False,
                fix_jumps: bool = False, timing_beats: bool = False, beat_decode: bool = False) -> str:
    """The folder (and table row) of a system under outputs/system_eval (v2 with another deletion
    threshold: _d<percent>, _keepall for none; with tidied written lengths: _tidy; glissandos written
    as glissandos: _gl; triplets the timing contradicts made straight: _tc; beats the timing says are straight: _tb;
    the readability rules: _rd; the labeller's splits decoded a beat at a time: _bd;
    stretched bars shortened: _jf; with other beats:
    _ab / _pm on clean MIDI, _rpm on the real transcriptions; with PM2S's beats from other weights
    than the released ones, their tag last: src/pm2s_beats.py weights_tag)."""
    if v2:
        lab, met = parse_v2(v2)
        d = "" if delete_above == DELETE_ABOVE else "_keepall" if delete_above >= 1 else f"_d{round(delete_above * 100)}"
        name = (f"v2_{lab}_{met}" + d + ("_tidy" if tidy else "") + ("_gl" if glissando else "")
                + ("_tc" if timing_check else "") + ("_tb" if timing_beats else "") + ("_rd" if readable else "") + ("_jf" if fix_jumps else "")
                + ("_bd" if beat_decode else "")
                + BEAT_SUFFIX[clean_beats]
                + REAL_SUFFIX[real_beats])
        if "pm2s" in (clean_beats, real_beats):   # other PM2S weights ($PM2S_WEIGHTS): their tag, _pm_rpm_<tag>
            from src.pm2s_beats import tagged
            name = tagged(name)
        return name
    return f"{run}_{mode}" + ("_sb" if midi_beats == "synth" else "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sf_base_reg_cab")
    ap.add_argument("--v2", default=None, help="<labeller run>+<meter run>: the v2 system instead of --run")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--vocab", default="data/processed/vocab.json")
    ap.add_argument("--mode", default=MODES[0], choices=list(MODES))
    ap.add_argument("--midi-beats", default="none", choices=["none", "synth"],
                    help="seq2seq beat models on clean MIDI: no beats (as the demo's MIDI input), or Beat This! "
                         "on the MIDI rendered to audio (v2 always gets those)")
    ap.add_argument("--clean-beats", default="synth", choices=list(BEAT_SUFFIX),
                    help="v2 on clean MIDI: Beat This! on the MIDI rendered to audio, on the performance's real "
                         "recording (only the performances that have one), or PM2S on the MIDI's notes")
    ap.add_argument("--real-beats", default="audio", choices=list(REAL_SUFFIX),
                    help="v2 on the real transcriptions (and Gangqinpu's conditions): Beat This! on the recording, "
                         "PM2S on the notes, or Beat This! on the input MIDI rendered to audio (synth: what a MIDI "
                         "upload to the demo gets; system _rsb)")
    ap.add_argument("--delete-above", type=float, default=DELETE_ABOVE,
                    help="v2: leave a note out only above this p(delete) (1 = keep every note); "
                         "a setting other than the default is its own system (_d<percent>, _keepall)")
    ap.add_argument("--glissando", action="store_true",
                    help="v2: glissandos written as their first and last note joined by a gliss. line "
                         "(src/glissando.py, the demo's default); its own system (_gl)")
    ap.add_argument("--timing-check", action="store_true",
                    help="v2: a note written on a triplet position but played on a straight eighth / sixteenth "
                         "moved there (src/timing.py, the demo's default); its own system (_tc)")
    ap.add_argument("--beat-decode", action="store_true",
                    help="v2: the labeller's splits of the quarter decoded a beat at a time, fused with the timing, "
                         "with its tuned beat_decode.json (src/beat_decode.py, the demo's default); its own system (_bd)")
    ap.add_argument("--timing-beats", action="store_true",
                    help="v2: a beat whose timing fits straight sixteenths better than the labeller's triplets "
                         "written straight (src/timing.py check_beats, the demo's default); its own system (_tb)")
    ap.add_argument("--readable", action="store_true",
                    help="v2: rests and overlaps of a 16th or less closed / trimmed, stems by voice (src/readable.py, "
                         "the demo's default); its own system (_rd)")
    ap.add_argument("--fix-jumps", action="store_true",
                    help="v2: a bar the meter decoder stretched with a jump shortened instead (src/readable.py, the "
                         "demo's default); its own system (_jf)")
    ap.add_argument("--tidy", action="store_true",
                    help="v2: written lengths stop at the next onset of their staff when they run just past it "
                         "(src/pipeline.py tidy_lengths); its own system (_tidy)")
    ap.add_argument("--manifest", default="outputs/system_eval/manifest.json")
    ap.add_argument("--conditions", default="clean,real")
    ap.add_argument("--out", default="outputs/system_eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--tracked", default="data/processed/test_transcribed",
                    help="split folder whose beats_tracked.npz holds the real transcriptions' beats")
    ap.add_argument("--synth-beats", default="outputs/system_eval/beats_synth.npz",
                    help="Beat This! on the clean MIDI rendered to audio (scripts/synth_beats.py)")
    ap.add_argument("--folds", default="data/raw/beat_this/asap_8-folds.split")
    ap.add_argument("--only-missing", action="store_true",
                    help="only the inputs this system has no score for yet (e.g. after an export fix)")
    args = ap.parse_args(argv)
    if (args.clean_beats != "synth" or args.real_beats != "audio") and not args.v2:
        raise SystemExit(f"--clean-beats {args.clean_beats} / --real-beats {args.real_beats} are for v2 (--v2)")
    entries = [e for e in json.loads(Path(args.manifest).read_text()) if e["condition"] in args.conditions.split(",")]
    runs = Path(args.runs_dir)
    if args.v2:
        lab, met = parse_v2(args.v2)
        pipe = V2Pipeline(runs / lab / args.ckpt, runs / met / args.ckpt, vocab=args.vocab, device=args.device,
                          delete_above=args.delete_above, tidy=args.tidy, glissando=args.glissando,
                          timing_check=args.timing_check, readable=args.readable, fix_jumps=args.fix_jumps,
                          timing_beats=args.timing_beats, beat_decode=args.beat_decode)
        if args.beat_decode and pipe.beat_params is None:
            raise SystemExit(f"--beat-decode: {runs / lab} has no beat_decode.json (not tuned: pace/v2_jobs.sh "
                             f"beatdecode {lab}); not scored, so no _bd system is written without the decoding")
        wants_beats, midi_beats = True, "synth"
    else:
        pipe = ScorePipeline(runs / args.run / args.ckpt, vocab=args.vocab, device=args.device)
        wants_beats, midi_beats = bool(pipe.info["beats"]), args.midi_beats
    tracked, synth, folds = {}, {}, {}
    bp = beats_path(Path(args.tracked), "tracked")
    if wants_beats and bp.exists():
        tracked = load_beats(bp)
    if wants_beats and any(e["condition"] in GANGQINPU for e in entries):   # the recordings' beats
        tracked.update(load_beats(Path(args.manifest).parent / "beats_gangqinpu.npz"))
    if wants_beats and midi_beats == "synth":
        synth = load_beats(args.synth_beats)
        if not synth:
            folds = load_folds(Path(args.folds))
            print(f"{args.synth_beats} missing: tracking the rendered MIDI here", flush=True)
    if args.clean_beats == "audio":
        n = len(entries)
        entries = [e for e in entries if e["condition"] != "clean" or e["performance"] in tracked]
        print(f"--clean-beats audio: {len(entries)} of {n} inputs (clean ones need a recording)", flush=True)
    if not args.v2 and (args.delete_above != DELETE_ABOVE or args.tidy or args.glissando or args.timing_check
                        or args.readable or args.fix_jumps or args.timing_beats or args.beat_decode):
        ap.error("--delete-above, --tidy, --glissando, --timing-check, --timing-beats, --readable, --fix-jumps and "
                 "--beat-decode are for v2 (--v2)")
    system = system_name(args.run, args.mode, args.midi_beats, args.v2, args.clean_beats, args.delete_above,
                         args.tidy, args.real_beats, args.glissando, args.timing_check, args.readable, args.fix_jumps,
                         args.timing_beats, args.beat_decode)
    out = Path(args.out) / system
    # the summary covers every piece in the folder: pieces not run now (another --conditions, or
    # --only-missing) keep their entries from before
    old = json.loads((out / "summary.json").read_text()).get("pieces", []) if (out / "summary.json").exists() else []
    kept = {(d["condition"], d["performance"]): d for d in old}
    if args.only_missing:
        entries = [e for e in entries if not (out / e["condition"] / f"{e['slug']}.musicxml").exists()]
        print(f"{system}: {len(entries)} inputs without a score", flush=True)
    done = []
    for k, e in enumerate(entries):
        dst = out / e["condition"] / f"{e['slug']}.musicxml"
        beats, kw = None, {}
        real_like = e["condition"] == "real" or e["condition"] in GANGQINPU
        pm2s = (args.real_beats if real_like else args.clean_beats) == "pm2s"
        if pm2s:  # tracked from the notes by the pipeline
            kw = {"beat_source": "pm2s"}
        elif real_like and args.real_beats == "synth":   # the MIDI rendered and tracked by the pipeline itself
            kw = {"beat_checkpoint": "final0"}
        elif real_like or args.clean_beats == "audio":
            beats = tracked.get(e["performance"])
        elif wants_beats and midi_beats == "synth":
            beats = synth.get(e["performance"])
            if beats is None:  # tracked here, with the model that never heard the piece
                kw = ({"beat_checkpoint": checkpoint_for(e["performance"], folds)} if args.v2
                      else {"midi_beats": True})
        t = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                if args.v2:
                    res = pipe.run(e["midi"], tmp, beats=beats, **kw)
                else:
                    res = pipe.run(e["midi"], tmp, mode=args.mode, beats=beats, **kw)
                errors, ok = list(res.errors), "musicxml" in res.files
            except Exception as ex:  # noqa: BLE001
                res, errors, ok = None, [f"{type(ex).__name__}: {ex}", traceback.format_exc()], False
            dst.parent.mkdir(parents=True, exist_ok=True)
            for stale in (dst, dst.with_suffix(".tokens.txt"), dst.with_suffix(".error.txt")):
                if stale.exists():
                    stale.unlink()
            if ok:
                shutil.copy(res.files["musicxml"], dst)
            else:  # keep what is needed to reproduce the failure
                if res is not None and res.tokens:
                    dst.with_suffix(".tokens.txt").write_text(" ".join(res.tokens))
                dst.with_suffix(".error.txt").write_text("\n".join(errors))
        stats = res.stats if res is not None else {}
        done.append({"condition": e["condition"], "performance": e["performance"], "musicxml": str(dst) if ok else None,
                     "errors": errors, "beats": stats.get("beats"), "seconds": round(time.perf_counter() - t, 1),
                     "emptied_bars": len((stats.get("export") or {}).get("emptied_bars", [])),
                     **{k_: stats.get(k_) for k_ in ("windows", "guard_fallbacks", "meter_fixes", "time_signature",
                                                     "notes_kept", "notes_deleted", "bars")}})
        print(f"{k + 1}/{len(entries)} {e['condition']} {e['performance']}: {'ok' if ok else errors[:1]} "
              f"({done[-1]['seconds']} s)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for d in done:
        kept[(d["condition"], d["performance"])] = d
    done = list(kept.values())
    (out / "summary.json").write_text(json.dumps({"system": system, "run": None if args.v2 else args.run,
                                                  "v2": args.v2, "mode": None if args.v2 else args.mode,
                                                  "midi_beats": midi_beats if wants_beats else None,
                                                  "clean_beats": args.clean_beats if args.v2 else None,
                                                  "real_beats": args.real_beats if args.v2 else None,
                                                  "beat_decode": (pipe.beat_params.to_dict() if args.v2 and args.beat_decode
                                                                  else None),
                                                  "pieces": done}, indent=1))
    print(f"wrote {out}: {sum(d['musicxml'] is not None for d in done)}/{len(done)} scores")


if __name__ == "__main__":
    main()
