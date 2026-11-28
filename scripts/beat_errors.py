"""Which beat-tracking errors cost v2 the most -> outputs/beat_errors/report.md (+ report.json).

    python scripts/beat_errors.py

Round 13 found that with the true beats and a perfect meter model v2 would gain 3.8 MUSTER on real
ASAP recordings and 6.0 on clean MIDI, against 1.4 / 1.0 for a perfect meter model on the tracked
beats: most of the loss is in the beats. Per test performance this compares the beats v2 was given
(scripts/system_scores.py's: Beat This! on the recording, real; on the MIDI rendered to audio, clean;
on Gangqinpu's recordings) with the ground truth (data/processed/meter: ASAP's annotations,
Gangqinpu's reliable aligned stretches), inside the annotated span:
  beat F            F-measure of the beats (70 ms)
  missed / extra    annotated beats with no tracked beat near / tracked beats with no annotated near
  off-beat          tracked beats near the midpoint between two annotated ones (half a beat off)
  level             median tracked beat length / median annotated one: 1 (right level), 2 (every
                    other beat: half tempo), 0.5 (double), 1.5 / 0.67 (a dotted / triplet level)
  downbeat F        the tracker's own downbeats
and what the beats cost on whole pieces: MUSTER with a perfect meter model on the tracked beats
(_og) minus with the annotated beats too (_ogt), from results.json. Reported per condition: the
cost by level class, its correlation with each error, and the costliest performances.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REL = "v2_nl_gq_meter_v4_gq_d95_tidy"
CONDS = {  # condition: (results.json, the beats v2 was given, the ground-truth split)
    "real": ("outputs/system_eval/results.json", "data/processed/test_transcribed/beats_tracked.npz", "test"),
    "clean": ("outputs/system_eval/results.json", "outputs/system_eval/beats_synth.npz", "test"),
    "gangqinpu_real": ("outputs/system_eval_gangqinpu/results.json",
                       "outputs/system_eval_gangqinpu/beats_gangqinpu.npz", "test_gangqinpu"),
}
LEVELS = ((0.8, 1.25, "right level"), (1.6, 2.5, "half tempo (x2)"), (0.4, 0.625, "double tempo (x0.5)"),
          (1.25, 1.6, "dotted (x1.5)"), (0.625, 0.8, "triplet (x0.67)"))


def level_class(r: float) -> str:
    for lo, hi, name in LEVELS:
        if lo <= r < hi:
            return name
    return "other"


def errors(bt: np.ndarray, bd: np.ndarray, gts: list) -> dict | None:
    """The tracked beats against the ground truth's spans (module docstring)."""
    from src.beats import MATCH_TOL, f_measure
    ref_t, ref_d, mids, lens, est_in, est_d = [], [], [], [], [], []
    for gt in gts:
        t, d = np.asarray(gt["t"], np.float64), np.asarray(gt["down"], bool)
        if len(t) < 4:
            continue
        lo, hi = t[0] - MATCH_TOL, t[-1] + MATCH_TOL
        inside = (bt >= lo) & (bt <= hi)
        ref_t.append(t)
        ref_d.append(t[d])
        mids.append((t[:-1] + t[1:]) / 2)
        lens.append(np.diff(t))
        est_in.append(bt[inside])
        est_d.append(bt[inside & bd])
    if not ref_t:
        return None
    rt, et = np.concatenate(ref_t), np.concatenate(est_in)
    if len(et) < 2:
        return {"beat_f": 0.0, "missed": 1.0, "extra": 0.0, "offbeat": 0.0, "level": float("nan"),
                "level_class": "other", "downbeat_f": 0.0, "beats": int(len(rt))}

    def near(a, b):
        if not len(b):
            return np.zeros(len(a), bool)
        j = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
        dist = np.minimum(np.abs(b[j] - a), np.abs(b[np.maximum(j - 1, 0)] - a))
        return dist <= MATCH_TOL
    rs, es = np.sort(rt), np.sort(et)
    est_diffs = np.concatenate([np.diff(e) for e in est_in if len(e) > 1]) if any(len(e) > 1 for e in est_in) else np.array([np.nan])
    ratio = float(np.median(est_diffs) / np.median(np.concatenate(lens)))
    return {"beat_f": float(f_measure(rt, et)), "missed": float(1 - near(rs, es).mean()),
            "extra": float(1 - near(es, rs).mean()), "offbeat": float(near(es, np.sort(np.concatenate(mids))).mean()),
            "level": ratio, "level_class": level_class(ratio),
            "downbeat_f": float(f_measure(np.concatenate(ref_d), np.concatenate(est_d))), "beats": int(len(rt))}


def corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meter-data", default="data/processed/meter")
    ap.add_argument("--out", default="outputs/beat_errors")
    args = ap.parse_args(argv)
    from scripts.compare_systems import load
    from src.beats import load_beats
    from src.grid_sources import ground_truth
    L = ["# Which beat-tracking errors cost v2 the most", "",
         "Cost = MUSTER with a perfect meter model on the tracked beats (`_og`) minus with the annotated beats too "
         "(`_ogt`): what the beats alone cost on that performance, given a perfect meter model. Errors inside the "
         "annotated span (70 ms).", ""]
    rep = {}
    gts_cache = {}
    for cond, (res_f, beats_f, split) in CONDS.items():
        if not Path(res_f).exists() or not Path(beats_f).exists():
            L += [f"## {cond}", "", f"skipped: {res_f if not Path(res_f).exists() else beats_f} missing", ""]
            continue
        if split not in gts_cache:
            f = Path(args.meter_data) / f"{split}.pkl"
            gts_cache[split] = ground_truth(pickle.load(open(f, "rb"))) if f.exists() else {}
        gts = gts_cache[split]
        mu = load(res_f)
        og, ogt, rel = mu.get((REL + "_og", cond), {}), mu.get((REL + "_ogt", cond), {}), mu.get((REL, cond), {})
        beats = load_beats(beats_f)
        rows = []
        for perf, (bt, bd) in beats.items():
            g = gts.get(perf)
            if not g or perf not in og or perf not in ogt:
                continue
            e = errors(np.asarray(bt, np.float64), np.asarray(bd, bool), g)
            if e is None:
                continue
            e.update(performance=perf, cost=og[perf] - ogt[perf], total=(rel.get(perf, np.nan) - ogt[perf]))
            rows.append(e)
        rep[cond] = rows
        L += [f"## {cond} ({len(rows)} performances)", ""]
        if not rows:
            L += ["no performance with beats, ground truth and both oracle systems", ""]
            continue
        mean = lambda k: float(np.nanmean([r[k] for r in rows]))  # noqa: E731
        L += [f"Mean: beat F {mean('beat_f'):.3f}, missed {mean('missed'):.1%}, extra {mean('extra'):.1%}, off-beat "
              f"{mean('offbeat'):.1%}, downbeat F {mean('downbeat_f'):.3f}; the beats' cost {mean('cost'):+.2f} MUSTER "
              f"(the release's whole gap to the oracle {mean('total'):+.2f}).", "",
              "| tracker's level | performances | beat F | the beats' cost (MUSTER) | share of the total cost |",
              "|---|---|---|---|---|"]
        total_cost = sum(max(r["cost"], 0) for r in rows) or 1.0
        for name in [n for _, _, n in LEVELS] + ["other"]:
            grp = [r for r in rows if r["level_class"] == name]
            if grp:
                c = float(np.mean([r["cost"] for r in grp]))
                share = sum(max(r["cost"], 0) for r in grp) / total_cost
                L.append(f"| {name} | {len(grp)} | {np.mean([r['beat_f'] for r in grp]):.3f} | {c:+.2f} | {share:.0%} |")
        L += ["", "Correlation of the beats' cost with each error (Pearson, over performances):", ""]
        cs = {k: corr([r[k] for r in rows], [r["cost"] for r in rows]) for k in ("missed", "extra", "offbeat", "downbeat_f")}
        cs["1 - beat F"] = corr([1 - r["beat_f"] for r in rows], [r["cost"] for r in rows])
        L.append(", ".join(f"{k} {v:+.2f}" for k, v in cs.items()))
        L += ["", "The costliest performances:", "", "| performance | cost | beat F | missed | extra | off-beat | level |",
              "|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: -r["cost"])[:8]:
            L.append(f"| {r['performance']} | {r['cost']:+.2f} | {r['beat_f']:.2f} | {r['missed']:.0%} | {r['extra']:.0%} | "
                     f"{r['offbeat']:.0%} | {r['level']:.2f} ({r['level_class']}) |")
        L.append("")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
