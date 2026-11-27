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
from src.beats import beats_path, compare, downbeat_rates, fit_noise, load_beats, piece_compound  # noqa: E402

D = Path("data/processed")


def compound_of() -> dict:
    """performance -> whether its meter is compound (src/beats.py piece_compound), from the meter
    data's ground truth (data/processed/meter/{train,val,test}.pkl; empty if not built)."""
    import pickle
    out = {}
    for s in ("train", "val", "test"):
        f = D / "meter" / f"{s}.pkl"
        if f.exists():
            for p in pickle.load(open(f, "rb")):
                out[p["performance"]] = piece_compound(p["gt"])
    return out


def comparisons(split: str, kinds: dict | None = None) -> list[dict]:
    """``kinds``: performance -> compound meter? (compound_of), given to compare (None: unknown)."""
    gt, tr = load_beats(beats_path(D / split, "gt")), load_beats(beats_path(D / split, "tracked"))
    return [compare(*gt[p], *tr[p], compound=(kinds or {}).get(p)) for p in sorted(set(gt) & set(tr))]


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
                            downbeat_rates([c for c in cs if c.get("octave") in ("double", "half", "offbeat")]).items()},
            "levels": level_summary(cs)}


def level_summary(cs: list[dict]) -> dict:
    """How the tracker's level per stretch went (compare's "levels"), by meter kind: performances,
    the share with a stretch at another level, with two or more switches, and the share of the
    beats at each level."""
    out = {}
    for kind, name in ((True, "compound"), (False, "simple"), (None, "unknown")):
        sel = [c for c in cs if c.get("compound") is kind and c.get("levels")]
        if not sel:
            continue
        by = Counter()
        for c in sel:
            for lv, f in c["levels"]:
                by[round(lv, 3)] += f
        tot = sum(by.values())
        out[name] = {"performances": len(sel),
                     "another_level": round(sum(any(abs(lv - 1) > 1e-9 for lv, _ in c["levels"]) for c in sel) / len(sel), 4),
                     "two_or_more_switches": round(sum(len(c["levels"]) >= 3 for c in sel) / len(sel), 4),
                     "beats_at_level": {str(k): round(v / tot, 4) for k, v in sorted(by.items())}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train_transcribed,val_transcribed")
    ap.add_argument("--report", default="test_transcribed", help="evaluated, not fitted on")
    ap.add_argument("--out", default=None, help="default: configs/beat_noise.json, with --levels "
                                                "configs/beat_noise_levels.json")
    ap.add_argument("--levels", action="store_true",
                    help="also fit the tracker's level per stretch (src/beats.py level_runs: several switches a "
                         "piece, the 2/3 level), per meter kind, replayed instead of the octave errors")
    args = ap.parse_args()
    args.out = args.out or ("configs/beat_noise_levels.json" if args.levels else "configs/beat_noise.json")
    kinds = compound_of() if args.levels else None
    cs = []
    for split in args.splits.split(","):
        if not beats_path(D / split, "tracked").exists():
            print(f"skip {split}: no beats_tracked.npz (scripts/track_beats.py)")
            continue
        cs += comparisons(split, kinds)
    if not cs:
        raise SystemExit("no performances with both tracked and ground-truth beats: run scripts/build_beats.py "
                         "and scripts/track_beats.py first")
    noise = fit_noise(cs, note=f"fitted by scripts/fit_beat_noise.py on {len(cs)} performances ({args.splits}): "
                               f"Beat This! (held-out cross-validation models) vs the ASAP annotations"
                               + (", with level profiles" if args.levels else ""), levels=args.levels)
    extra = {"fit_on": summary(cs)}
    if beats_path(D / args.report, "tracked").exists():
        extra["tracker_on_" + args.report] = summary(comparisons(args.report, kinds))
    noise.to_json(args.out, extra)
    shown = {k: v for k, v in json.loads(Path(args.out).read_text()).items()
             if k not in ("profiles", "octave_profiles", "level_profiles")}
    print(json.dumps(shown, indent=1))
    print(f"(+ {len(noise.profiles or [])} right-tempo, {len(noise.octave_profiles or [])} wrong-tempo and "
          f"{len(noise.level_profiles or [])} level per-performance profiles in {args.out})")


if __name__ == "__main__":
    main()
