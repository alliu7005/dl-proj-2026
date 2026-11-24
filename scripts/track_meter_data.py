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
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_gangqinpu_audio import audio_path  # noqa: E402
from src.beats import compare, save_beats  # noqa: E402
from src.meter import perf_key  # noqa: E402
from src.preprocessing.gangqinpu import iter_rows, performance_notes, split_files  # noqa: E402

TRACKERS = ("beat_this", "pm2s")


def tracked_path(meter: Path, split: str, tracker: str) -> Path:
    """data/processed/meter/tracked_<split>_<tracker>.npz (split: e.g. train_gangqinpu)."""
    return Path(meter) / f"tracked_{split}_{tracker}.npz"


def crop(beats, span) -> tuple | None:
    """The beats inside a segment's span; None with fewer than 2."""
    t, d = np.asarray(beats[0], np.float64), np.asarray(beats[1], bool)
    keep = (t >= span[0]) & (t <= span[1])
    return (t[keep], d[keep]) if keep.sum() >= 2 else None


def summarize(perfs: list, beats: dict) -> dict:
    rows = [compare(p["gt"]["t"], p["gt"]["down"], *beats[perf_key(p)]) for p in perfs if perf_key(p) in beats]
    if not rows:
        return {"segments": 0}
    return {"segments": len(rows), "beat_f": float(np.mean([r["beat_f"] for r in rows])),
            "downbeat_f": float(np.mean([r["downbeat_f"] for r in rows])),
            "wrong_octave": float(np.mean([r.get("octave") in ("double", "half") for r in rows]))}


def main(argv=None, beat_this=None, pm2s=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--release", default="data/raw/gangqinpu")
    ap.add_argument("--audio", default="data/raw/gangqinpu/audio")
    ap.add_argument("--trackers", default="beat_this,pm2s")
    ap.add_argument("--splits", default="train,val,test")
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
    if "beat_this" in trackers and beat_this is None:
        from scripts.track_beats import Tracker
        bt = Tracker(device)
        beat_this = lambda path: bt(path, "final0")  # noqa: E731
    if "pm2s" in trackers and pm2s is None:
        from src.pm2s_beats import PM2SBeats
        pm2s = PM2SBeats(device=device)
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
                        whole = beat_this(wav)
                    except Exception as e:  # noqa: BLE001  (an unreadable download: skipped, counted)
                        print(f"{name} {sid}: Beat This! failed: {e}", flush=True)
                        continue
                    for p in segs:
                        c = crop(whole, p["span"])
                        if c is not None:
                            got["beat_this"][perf_key(p)] = c
            rep = {"segments": len(perfs)}
            for t, beats in got.items():
                save_beats(tracked_path(meter, name, t), beats)
                rep[t] = summarize(perfs, beats)
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
    meter.mkdir(parents=True, exist_ok=True)
    (meter / "tracked_report.json").write_text(json.dumps({"trackers": trackers, "device": device,
                                                           "seconds": round(time.perf_counter() - t0, 1),
                                                           "splits": report}, indent=1))


if __name__ == "__main__":
    main()
