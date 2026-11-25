"""Fit PM2S's beat-tracking errors (src.beats.BeatNoise) -> configs/beat_noise_pm2s.json, so a meter
model can be trained for PM2S's beats as configs/beat_noise.json trains it for Beat This!'s.

    python scripts/fit_pm2s_noise.py          # after scripts/build_meter_data.py --gangqinpu

Training never runs a tracker: it corrupts the ground-truth beats with a tracker's fitted errors
(tempo octave errors, missed / extra beats, timing, downbeat marks; scripts/fit_beat_noise.py fits
Beat This!'s on the ASAP real transcriptions). PM2S's released weights were trained on all of ASAP,
so its errors there are too few to fit on. Gangqinpu is music it never saw: PM2S runs on the notes
of up to --n training performances (each one's longest run of trusted bars with at least
--min-beats beats; data/processed/meter/train_gangqinpu.pkl), is compared with the score's beats
timed through the alignment (src/beats.compare), and fit_noise turns that into the profile. The
validation performances are reported, not fitted on. The ground truth here is alignment-timed, not
annotated by hand as ASAP's is (92.6% of its beats have a played note within 50 ms, on the
validation split).

--from-stored beat_this: the same fit from a tracker's beats stored by scripts/track_meter_data.py
instead of running PM2S (Beat This!'s real beats of Gangqinpu's recordings ->
configs/beat_noise_bt_gangqinpu.json): the errors it makes on this music, its off-beat rate
included (0 on the ASAP recordings configs/beat_noise.json was fitted on), measured, not guessed.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fit_beat_noise import summary  # noqa: E402
from src.beats import compare, fit_noise  # noqa: E402
from src.pm2s_beats import PM2SBeats  # noqa: E402


def longest_segments(perfs: list, min_beats: int) -> list:
    """Each performance's longest segment with at least ``min_beats`` ground-truth beats."""
    best = {}
    for p in perfs:
        n = len(p["gt"]["t"])
        if n >= min_beats and n > len(best.get(p["performance"], {"gt": {"t": []}})["gt"]["t"]):
            best[p["performance"]] = p
    return [best[k] for k in sorted(best)]


def comparisons(tracker, perfs: list) -> list[dict]:
    out = []
    for p in perfs:
        t, d = tracker(p["notes"])
        out.append(compare(p["gt"]["t"], p["gt"]["down"], t, d))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--n", type=int, default=400, help="training performances to fit on (a fixed draw)")
    ap.add_argument("--n-report", type=int, default=150, help="validation performances reported (a fixed draw)")
    ap.add_argument("--min-beats", type=int, default=64)
    ap.add_argument("--weights", default=None, help="PM2S checkpoint (default: the released one, cached)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="default: configs/beat_noise_pm2s.json, with --from-stored "
                                                "configs/beat_noise_<tracker>_gangqinpu.json")
    ap.add_argument("--from-stored", default=None, choices=("beat_this", "pm2s"),
                    help="fit on this tracker's stored beats (track_meter_data.py) instead of running PM2S")
    ap.add_argument("--min-performances", type=int, default=30,
                    help="--from-stored: fewer tracked performances than this -> no profile written")
    args = ap.parse_args(argv)
    from src.pm2s_beats import tagged
    if args.out is None:   # other PM2S weights ($PM2S_WEIGHTS): configs/beat_noise_pm2s_<tag>.json
        args.out = (f"configs/beat_noise_{'bt' if args.from_stored == 'beat_this' else args.from_stored}_gangqinpu.json"
                    if args.from_stored else f"configs/{tagged('beat_noise_pm2s')}.json")
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    meter = Path(args.data) / "meter"

    def draw(split, n):
        f = meter / f"{split}_gangqinpu.pkl"
        if not f.exists():
            return []
        segs = longest_segments(pickle.load(open(f, "rb")), args.min_beats)
        if len(segs) > n:
            segs = [segs[i] for i in sorted(np.random.default_rng(0).choice(len(segs), n, replace=False))]
        return segs
    fit_on = draw("train", args.n)
    if not fit_on:
        raise SystemExit(f"no {meter / 'train_gangqinpu.pkl'} performances with {args.min_beats}+ beats: "
                         "run scripts/build_meter_data.py --gangqinpu <release> --gangqinpu-only first")
    if args.from_stored:
        from scripts.track_meter_data import tracked_path
        from src.beats import compare, load_beats
        from src.meter import perf_key

        def stored(split, segs):
            beats = load_beats(tracked_path(meter, f"{split}_gangqinpu", args.from_stored))
            return [compare(p["gt"]["t"], p["gt"]["down"], *beats[perf_key(p)]) for p in segs if perf_key(p) in beats]
        cs = stored("train", draw("train", 10 ** 9))   # every one it has beats for (only the downloaded recordings)
        if len(cs) < args.min_performances:
            print(f"--from-stored {args.from_stored}: {len(cs)} tracked performances (< {args.min_performances}): "
                  f"no profile written to {args.out}", flush=True)
            return
        noise = fit_noise(cs, note=f"fitted by scripts/fit_pm2s_noise.py --from-stored {args.from_stored} on {len(cs)} "
                                   "Gangqinpu training performances: the tracker's stored real beats vs the score's "
                                   "beats timed through the alignment")
        extra = {"tracker": args.from_stored, "fit_on": summary(cs)}
        rep = stored("val", draw("val", args.n_report))
        if rep:
            extra["tracker_on_val_gangqinpu"] = summary(rep)
    else:
        tracker = PM2SBeats(args.weights, device=device)
        cs = comparisons(tracker, fit_on)
        noise = fit_noise(cs, note=f"fitted by scripts/fit_pm2s_noise.py on {len(cs)} Gangqinpu training performances: "
                                   f"PM2S ({tracker.weights}) on their notes vs the score's beats timed through the alignment")
        extra = {"tracker": "pm2s", "weights": tracker.weights, "fit_on": summary(cs)}
        report = draw("val", args.n_report)
        if report:
            extra["tracker_on_val_gangqinpu"] = summary(comparisons(tracker, report))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    noise.to_json(args.out, extra)
    shown = {k: v for k, v in json.loads(Path(args.out).read_text()).items() if k not in ("profiles", "octave_profiles")}
    print(json.dumps(shown, indent=1))
    print(f"(+ {len(noise.profiles or [])} right-tempo and {len(noise.octave_profiles or [])} wrong-tempo "
          f"per-performance profiles in {args.out})")


if __name__ == "__main__":
    main()
