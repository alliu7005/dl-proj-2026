"""Real beat trackers on the meter model's data: their beats stored next to each performance, so
training can mix them with the simulated tracker (scripts/train_meter.py --tracked) and testing
can use them (scripts/eval_meter.py --real-tracked).

    python scripts/track_meter_data.py        # -> data/processed/meter/tracked_<split>_<tracker>.npz
                                              #    + tracked_report.json

Trackers and where they run (never on music they were trained on):
  beat_this   Beat This! (final0) on Gangqinpu's recordings (scripts/fetch_gangqinpu_audio.py:
              the ones downloaded); it was trained on ASAP among others, not on Gangqinpu
  pm2s        PM2S on the notes (src/pm2s_beats.py): Gangqinpu's transcriptions (the whole
              performance, read again from the release) and PERiScoPe's (each segment's notes);
              its released weights were trained on all of ASAP, so ASAP is left out
  beat_this_render
              Beat This! on the performances rendered to audio with a sampled piano (src/render.py:
              FluidSynth + SoundFonts), for the ones with no recording: ASAP (its performance MIDI,
              pedal included, tracked by the Beat This! fold model that never trained on it),
              PERiScoPe and Gangqinpu (each segment's notes, final0). On recordings and renders of the
              same performances Beat This! tracks alike (src/render.py), so these stand in for the
              simulated tracker: train_meter.py --tracked beat_this,beat_this_render takes a
              recording's beats where there is one, else the render's.
              Needs FluidSynth (--fluidsynth or FLUIDSYNTH) and SoundFonts (--soundfonts or SOUNDFONTS).
              Large: --workers renders at once; --shard K/N takes every N-th performance from the K-th
              (several jobs), saving as it goes (tracked_<split>_beat_this_render.partKofN.npz; a
              job run again goes on where it stopped), and --merge N joins the parts.
With --models <set> (src/bt_models.py: our Beat This!, fine-tuned and cross-fitted) beat_this and
beat_this_render track each performance with the model that never heard its piece, stored as
beat_this_<tag> / beat_this_render_<tag> (--tag, default the set's folder name). Several sets,
comma-separated (--models a,b --tag ft,ftm): each performance is rendered once and tracked by each.
Both track a whole Gangqinpu performance once; each of its segments keeps the beats inside its span.
Stored as src/beats.py save_beats files keyed like the performances (src/meter.py perf_key).
tracked_report.json: each tracker's beat / downbeat F-measure against the alignment-timed beats per
split, and on the segments both trackers have, paired: the like-for-like comparison of Beat This!
on the audio and PM2S on its transcription.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_gangqinpu_audio import audio_path  # noqa: E402
from src.beats import compare, load_beats, save_beats  # noqa: E402
from src.meter import perf_key  # noqa: E402
from src.preprocessing.gangqinpu import iter_rows, performance_notes, split_files  # noqa: E402

RENDER = "beat_this_render"
TRACKERS = ("beat_this", "pm2s", RENDER)
PAD = 1.0   # seconds of the render before a segment's first note


def tracked_path(meter: Path, split: str, tracker: str) -> Path:
    """data/processed/meter/tracked_<split>_<tracker>.npz (split: e.g. train_gangqinpu); PM2S with
    other weights than the released ones: tracked_<split>_pm2s_<tag>.npz (src/pm2s_beats.py
    weights_tag), so training (--tracked pm2s) and testing read the beats of the weights in use."""
    if tracker == "pm2s":
        from src.pm2s_beats import tagged
        tracker = tagged(tracker)
    return Path(meter) / f"tracked_{split}_{tracker}.npz"


def crop(beats, span) -> tuple | None:
    """The beats inside a segment's span; None with fewer than 2."""
    t, d = np.asarray(beats[0], np.float64), np.asarray(beats[1], bool)
    keep = (t >= span[0]) & (t <= span[1])
    return (t[keep], d[keep]) if keep.sum() >= 2 else None


def _source(p: dict, asap_root: Path | None):
    """What to render for a performance and where its render starts on its clock: (source, t0),
    None with too few notes."""
    midi = Path(asap_root) / p["performance"] if asap_root is not None and p.get("dataset") == "asap" else None
    if midi is not None and midi.exists() and p.get("segments", 1) <= 1:
        return midi, 0.0
    notes = np.asarray(p["notes"], np.float64)
    if len(notes) < 2:
        return None
    t0 = max(0.0, float(notes[:, 1].min()) - PAD)
    source = notes.copy()
    source[:, 1:3] -= t0
    return source, t0


def render_track(perfs: list, render, track, checkpoint_of, asap_root: Path | None = None, log=print,
                 workers: int = 1, more: list | None = None):
    """Each performance rendered (``render(source, wav, key)``, src/render.py Renderer) and tracked
    (``track(wav, checkpoint)``) -> {perf_key: (times, down)} on the performance's clock, cropped to
    its span. An ASAP performance renders its own MIDI file (``asap_root`` / performance: velocities
    and pedal as played); the others their notes, from just before the first (``PAD``). A
    performance that fails to render or track is left out (and logged). ``workers`` renders run at
    once (FluidSynth processes, one core each) while the tracker takes the finished ones in order.
    ``more``: further (track, checkpoint_of) pairs run on the same render (rendered once, tracked by
    each); then a list of result dicts is returned, the first for ``track``."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    pairs = [(track, checkpoint_of)] + list(more or [])
    outs = [{} for _ in pairs]
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(max(1, workers)) as pool:
        todo = iter([(i, p, *s) for i, p in enumerate(perfs) for s in [_source(p, asap_root)] if s is not None])
        pending = deque()

        def submit():
            nxt = next(todo, None)
            if nxt is not None:
                i, p, source, t0 = nxt
                wav = Path(tmp) / f"render{i}.wav"
                pending.append((p, t0, wav, pool.submit(render, source, wav, perf_key(p))))
        for _ in range(2 * max(1, workers)):
            submit()
        while pending:
            p, t0, wav, fut = pending.popleft()
            submit()
            key = perf_key(p)
            try:
                fut.result()
                for (tr, ck), out in zip(pairs, outs):
                    t, d = tr(wav, ck(p))
                    beats = (np.asarray(t, np.float64) + t0, np.asarray(d, bool))
                    c = crop(beats, p["span"]) if p.get("span") else (beats if len(beats[0]) >= 2 else None)
                    if c is not None:
                        out[key] = c
            except Exception as e:  # noqa: BLE001
                log(f"{key}: render / track failed: {e}")
                continue
            finally:
                wav.unlink(missing_ok=True)
    return outs if more else outs[0]


def part_path(path: Path, k: int, n: int) -> Path:
    """One shard's file next to the whole one: tracked_<split>_<tracker>.part<k>of<n>.npz."""
    return path.with_name(f"{path.stem}.part{k}of{n}{path.suffix}")


def parse_shard(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    try:
        k, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise SystemExit(f"--shard K/N, not {spec!r}") from None
    if not 0 <= k < n:
        raise SystemExit(f"--shard K/N needs 0 <= K < N, not {spec!r}")
    return k, n


def summarize(perfs: list, beats: dict) -> dict:
    rows = [compare(p["gt"]["t"], p["gt"]["down"], *beats[perf_key(p)]) for p in perfs if perf_key(p) in beats]
    if not rows:
        return {"segments": 0}
    return {"segments": len(rows), "beat_f": float(np.mean([r["beat_f"] for r in rows])),
            "downbeat_f": float(np.mean([r["downbeat_f"] for r in rows])),
            "wrong_octave": float(np.mean([r.get("octave") in ("double", "half") for r in rows]))}


def main(argv=None, beat_this=None, pm2s=None, render=None, track=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--release", default="data/raw/gangqinpu")
    ap.add_argument("--audio", default="data/raw/gangqinpu/audio")
    ap.add_argument("--trackers", default="beat_this,pm2s")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--asap", default="data/raw/asap_bd", help="beat_this_render: ASAP's performance MIDI files")
    ap.add_argument("--folds", default="data/raw/beat_this/asap_8-folds.split",
                    help="beat_this_render: Beat This!'s ASAP folds (each ASAP performance tracked by the model that never heard it)")
    ap.add_argument("--soundfonts", default=None, help="beat_this_render: SoundFonts, joined by os.pathsep (default: $SOUNDFONTS)")
    ap.add_argument("--fluidsynth", default=None, help="beat_this_render: the FluidSynth program (default: $FLUIDSYNTH, else the PATH)")
    ap.add_argument("--workers", type=int, default=1, help="beat_this_render: renders at once (one core each)")
    ap.add_argument("--shard", default=None, help="beat_this_render: K/N, every N-th performance from the K-th (0-based)")
    ap.add_argument("--merge", type=int, default=0, help="beat_this_render: join the N shards' parts and report (no tracking)")
    ap.add_argument("--save-every", type=int, default=200, help="beat_this_render: performances between saves")
    ap.add_argument("--models", default=None, help="our Beat This! model set (src/bt_models.py) instead of the released one")
    ap.add_argument("--tag", default=None, help="with --models: the stored trackers' suffix (default: the set's folder name)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    trackers = [t for t in args.trackers.split(",") if t]
    if any(t not in TRACKERS for t in trackers):
        raise SystemExit(f"--trackers: from {TRACKERS}")
    meter = Path(args.data) / "meter"
    sets = [(None, "")]                   # (model set, suffix of the stored trackers): the released Beat This!
    if args.models:
        from src.bt_models import ModelSet
        paths = [x for x in args.models.split(",") if x]
        tags = [x for x in (args.tag or "").split(",") if x] or [Path(x).name for x in paths]
        if len(tags) != len(paths):
            raise SystemExit(f"--tag: one per --models set ({len(paths)}), not {len(tags)}")
        sets = [(ModelSet(m), f"_{t}") for m, t in zip(paths, tags)]
        if "pm2s" in trackers:
            raise SystemExit("--models is our Beat This!: not with pm2s")
    elif args.tag:
        raise SystemExit("--tag goes with --models")
    ms, suffix = sets[0]

    def tname(t: str, sfx: str | None = None) -> str:   # the stored tracker's name
        return t + (suffix if sfx is None else sfx) if t in ("beat_this", RENDER) else t
    shard = parse_shard(args.shard)
    if (shard or args.merge) and trackers != [RENDER]:
        raise SystemExit(f"--shard / --merge are for --trackers {RENDER} alone")
    if args.merge:
        return {sfx: merge(meter, args.splits.split(","), args.merge, tname(RENDER, sfx)) for _, sfx in sets}
    if "beat_this" in trackers and beat_this is None:
        from scripts.track_beats import Tracker
        bt = Tracker(device)
        beat_this = (lambda path: bt(path, "final0")) if ms is None else (lambda path, ck: bt(path, ck))
    if "pm2s" in trackers and pm2s is None:
        from src.pm2s_beats import PM2SBeats
        pm2s = PM2SBeats(device=device)
    checkpoint_of = None
    if RENDER in trackers:
        from scripts.track_beats import checkpoint_for, load_folds
        if render is None:
            from src.render import Renderer
            render = Renderer(args.soundfonts, args.fluidsynth)
            print(f"rendering with {[str(f) for f in render.fonts]} ({render.exe})", flush=True)
        if track is None:
            from scripts.track_beats import Tracker
            tr = Tracker(device)
            track = lambda path, ck: tr(path, ck)  # noqa: E731
        folds = load_folds(Path(args.folds))
        checkpoint_of = lambda p: checkpoint_for(p["performance"], folds) if p.get("dataset") == "asap" else "final0"  # noqa: E731
        if ms is not None:   # the fine-tuned model that never heard the performance's piece
            checkpoint_of = ms.checkpoint_for
        more = [(track, m.checkpoint_for) for m, _ in sets[1:]]   # further sets: the same render
    report, t0 = {}, time.perf_counter()
    for split in args.splits.split(","):
        # Gangqinpu: whole performances tracked once, each segment keeps its span
        name = f"{split}_gangqinpu"
        f = meter / f"{name}.pkl"
        if f.exists():
            perfs = pickle.load(open(f, "rb"))
            by_perf = defaultdict(list)
            for p in perfs:
                by_perf[p["performance"].split("/", 1)[1]].append(p)
            got = {t: {} for t in trackers}
            got_more = {sfx: {} for _, sfx in sets[1:]}   # beat_this by the further model sets
            if "pm2s" in trackers and split_files(Path(args.release), split):
                for row in iter_rows(Path(args.release), split, columns=["sample_id", "performance_notes", "alignment",
                                                                         "source", "audio_url"]):
                    segs = by_perf.get(str(row["sample_id"]))
                    if not segs:
                        continue
                    whole = pm2s(performance_notes(row))
                    for p in segs:
                        c = crop(whole, p["span"])
                        if c is not None:
                            got["pm2s"][perf_key(p)] = c
            if "beat_this" in trackers:
                for sid, segs in by_perf.items():
                    wav = audio_path(Path(args.audio), split, sid)
                    if not wav.exists():
                        continue
                    try:
                        whole = beat_this(wav) if ms is None else beat_this(wav, ms.checkpoint_for(segs[0]))
                        wholes = {sfx: beat_this(wav, m.checkpoint_for(segs[0])) for m, sfx in sets[1:]}
                    except Exception as e:  # noqa: BLE001  (an unreadable download: skipped, counted)
                        print(f"{name} {sid}: Beat This! failed: {e}", flush=True)
                        continue
                    for p in segs:
                        c = crop(whole, p["span"])
                        if c is not None:
                            got["beat_this"][perf_key(p)] = c
                        for sfx, wh in wholes.items():
                            c = crop(wh, p["span"])
                            if c is not None:
                                got_more[sfx][perf_key(p)] = c
            rep = {"segments": len(perfs)}
            for t, beats in got.items():
                save_beats(tracked_path(meter, name, tname(t)), beats)
                rep[tname(t)] = summarize(perfs, beats)
            for sfx, beats in got_more.items():   # the further model sets' beats of the recordings
                if "beat_this" in trackers:
                    save_beats(tracked_path(meter, name, tname("beat_this", sfx)), beats)
                    rep[tname("beat_this", sfx)] = summarize(perfs, beats)
            if len(trackers) == 2:   # like for like: the segments both trackers have
                both = [p for p in perfs if all(perf_key(p) in got[t] for t in trackers)]
                rep["both"] = {t: summarize(both, got[t]) for t in trackers}
            report[name] = rep
            print(f"{name}: " + json.dumps(rep), flush=True)
        # PERiScoPe: no audio; PM2S on each segment's notes
        name = f"{split}_periscope"
        f = meter / f"{name}.pkl"
        if "pm2s" in trackers and f.exists():
            perfs = pickle.load(open(f, "rb"))
            beats = {}
            for p in perfs:
                c = crop(pm2s(np.asarray(p["notes"], np.float64)), p["span"]) if len(p["notes"]) >= 2 else None
                if c is not None:
                    beats[perf_key(p)] = c
            save_beats(tracked_path(meter, name, "pm2s"), beats)
            report[name] = {"segments": len(perfs), "pm2s": summarize(perfs, beats)}
            print(f"{name}: " + json.dumps(report[name]), flush=True)
        # rendered: ASAP (meter/<split>.pkl), PERiScoPe and Gangqinpu, every performance
        if RENDER in trackers:
            for name in (split, f"{split}_periscope", f"{split}_gangqinpu"):
                f = meter / f"{name}.pkl"
                if not f.exists():
                    continue
                perfs = pickle.load(open(f, "rb"))
                if shard:   # every N-th from the K-th; what an earlier run of this shard did is kept
                    perfs = perfs[shard[0]::shard[1]]
                outs = []
                for _, sfx in sets:
                    o = tracked_path(meter, name, tname(RENDER, sfx))
                    outs.append(part_path(o, *shard) if shard else o)
                beats = [load_beats(o) if shard else {} for o in outs]
                todo = [p for p in perfs if any(perf_key(p) not in b for b in beats)]
                print(f"{name} ({RENDER}): {len(todo)} of {len(perfs)} performances to render and track"
                      + (f" ({len(sets)} trackers)" if len(sets) > 1 else ""), flush=True)
                for a in range(0, len(todo), max(1, args.save_every)):
                    got_r = render_track(todo[a:a + args.save_every], render, track, checkpoint_of,
                                         Path(args.asap), workers=args.workers, more=more)
                    for b, g, o in zip(beats, got_r if more else [got_r], outs):
                        b.update(g)
                        save_beats(o, b)
                    print(f"  {name}: {min(a + args.save_every, len(todo))} of {len(todo)} done "
                          f"({time.perf_counter() - t0:.0f} s)", flush=True)
                for (_, sfx), b, o in zip(sets, beats, outs):
                    save_beats(o, b)
                    report.setdefault(name, {"segments": len(perfs)})[tname(RENDER, sfx)] = summarize(perfs, b)
                    print(f"{name} ({tname(RENDER, sfx)}): " + json.dumps(report[name][tname(RENDER, sfx)]), flush=True)
    meter.mkdir(parents=True, exist_ok=True)
    from src.pm2s_beats import tagged, weights_tag   # other PM2S weights: tracked_report_<tag>.json
    report_name = (tagged("tracked_report") if "pm2s" in trackers else "tracked_report") + ".json"
    if trackers == [RENDER]:   # alone: its own report, the others' kept
        every = "".join(sfx for _, sfx in sets)   # all the model sets' tags
        report_name = (f"tracked_report_render{every}.json" if not shard
                       else f"tracked_report_render{every}.part{shard[0]}of{shard[1]}.json")
    elif suffix:
        report_name = f"tracked_report{''.join(sfx for _, sfx in sets)}.json"
    (meter / report_name).write_text(json.dumps({"trackers": trackers, "device": device,
                                                           "pm2s_tag": weights_tag() if "pm2s" in trackers else None,
                                                           "seconds": round(time.perf_counter() - t0, 1),
                                                           "splits": report}, indent=1))


def merge(meter: Path, splits: list, n: int, tracker: str = RENDER) -> dict:
    """Join the N shards' parts of every split into tracked_<split>_beat_this_render.npz, with the
    report (tracked_report_render.json) over all of them; a missing part is an error (its shard
    has not finished: run it again)."""
    report = {}
    for split in splits:
        for name in (split, f"{split}_periscope", f"{split}_gangqinpu"):
            f = meter / f"{name}.pkl"
            whole = tracked_path(meter, name, tracker)
            parts = [part_path(whole, k, n) for k in range(n)]
            if not f.exists() or not any(x.exists() for x in parts):
                continue
            missing = [str(x) for x in parts if not x.exists()]
            if missing:
                raise SystemExit(f"--merge {n}: missing {missing} (run those shards again)")
            beats = {}
            for x in parts:
                beats.update(load_beats(x))
            save_beats(whole, beats)
            perfs = pickle.load(open(f, "rb"))
            report[name] = {"segments": len(perfs), "tracked": len(beats), tracker: summarize(perfs, beats)}
            print(f"{name}: {len(beats)} of {len(perfs)} tracked; " + json.dumps(report[name][tracker]), flush=True)
    rname = "tracked_report_render.json" if tracker == RENDER else f"tracked_report_{tracker}.json"
    (meter / rname).write_text(json.dumps({"trackers": [tracker], "shards": n,
                                                                  "splits": report}, indent=1))
    return report


if __name__ == "__main__":
    main()
