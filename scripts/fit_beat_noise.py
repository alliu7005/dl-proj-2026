"""Fit the beat-tracker error model (src.beats.BeatNoise) -> configs/beat_noise.json.

    python scripts/fit_beat_noise.py          # after scripts/build_beats.py and scripts/track_beats.py

Compares Beat This!'s beats (beats_tracked.npz) with the ground truth (beats_gt.npz: the ASAP
annotations) on the performances of the TRAINING-side real splits only (train_transcribed,
val_transcribed; never test), one performance at a time (src.beats.compare): tempo octave errors,
off-beat tracking, missed / extra beats (and where the extra ones fall), timing, and how often
each kind of beat (downbeat, half-bar beat, other, spurious) is marked as a downbeat, kept per
performance as profiles. Training then corrupts the ground-truth beats with these errors, and the
simulated-tracker test condition uses the same model with a fixed seed. Also prints how well the tracker did on test_transcribed (not used for
the fit), the number to quote for the real-audio condition.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.beats import beats_path, compare, downbeat_rates, fit_noise, load_beats  # noqa: E402

D = Path("data/processed")


def comparisons(split: str) -> list[dict]:
    gt, tr = load_beats(beats_path(D / split, "gt")), load_beats(beats_path(D / split, "tracked"))
    return [compare(*gt[p], *tr[p]) for p in sorted(set(gt) & set(tr))]


def summary(cs: list[dict]) -> dict:
    """F-measures, tempo octaves, and at the right tempo the beat-level rates (means over
    performances) and the pooled downbeat rates."""
    normal = [c for c in cs if c.get("octave") == "normal"]
    mean = lambda key: round(float(np.mean([c[key] for c in normal])), 4) if normal else None  # noqa: E731
    return {"performances": len(cs),
            "beat_f_mean": float(np.mean([c["beat_f"] for c in cs])) if cs else None,
            "downbeat_f_mean": float(np.mean([c["downbeat_f"] for c in cs])) if cs else None,
            "octave": dict(Counter(c.get("octave") for c in cs)),
            "right_tempo": {"miss": mean("miss"), "extra": mean("extra"),
                            **{k: None if v is None else round(v, 4) for k, v in downbeat_rates(normal).items()}},
            "wrong_tempo": {k: None if v is None else round(v, 4) for k, v in
                            downbeat_rates([c for c in cs if c.get("octave") in ("double", "half", "offbeat")]).items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train_transcribed,val_transcribed")
    ap.add_argument("--report", default="test_transcribed", help="evaluated, not fitted on")
    ap.add_argument("--out", default="configs/beat_noise.json")
    args = ap.parse_args()
    cs = []
    for split in args.splits.split(","):
        if not beats_path(D / split, "tracked").exists():
            print(f"skip {split}: no beats_tracked.npz (scripts/track_beats.py)")
            continue
        cs += comparisons(split)
    if not cs:
        raise SystemExit("no performances with both tracked and ground-truth beats: run scripts/build_beats.py "
                         "and scripts/track_beats.py first")
    noise = fit_noise(cs, note=f"fitted by scripts/fit_beat_noise.py on {len(cs)} performances ({args.splits}): "
                               f"Beat This! (held-out cross-validation models) vs the ASAP annotations")
    extra = {"fit_on": summary(cs)}
    if beats_path(D / args.report, "tracked").exists():
        extra["tracker_on_" + args.report] = summary(comparisons(args.report))
    noise.to_json(args.out, extra)
    shown = {k: v for k, v in json.loads(Path(args.out).read_text()).items()
             if k not in ("profiles", "octave_profiles")}
    print(json.dumps(shown, indent=1))
    print(f"(+ {len(noise.profiles or [])} right-tempo and {len(noise.octave_profiles or [])} wrong-tempo "
          f"per-performance profiles in {args.out})")


if __name__ == "__main__":
    main()
