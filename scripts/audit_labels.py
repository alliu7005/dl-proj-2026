"""How often the performance contradicts the written split of the quarter, per dataset.

    python scripts/audit_labels.py                                   # train, train_gangqinpu, val, val_gangqinpu
    python scripts/audit_labels.py --splits val --windows 500

The labeller learns where to write a note from score-performance pairs. Where a score writes a note
on the triplet grid but the pianist played it on a straight sixteenth (or the reverse), the pair
teaches it to write that timing the other way. This counts such notes (src/notelabels.py
contradicted, on each window's true grid: the score's beats timed through the alignment) among the
notes written off the quarter and the half, per split and per genre (Gangqinpu's), and how much of
it a whole window holds. Written to outputs/label_audit.json; scripts/train_labeller.py
--mask-contradicted leaves those notes' positions out of the loss.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import ScoreSegmentDataset  # noqa: E402
from src.notelabels import LABEL_FILE, STRAIGHT_PLAYED_TRIPLET, TRIPLET_PLAYED_STRAIGHT  # noqa: E402


def written_family(rows: np.ndarray) -> np.ndarray:
    """Per label row: 1 written on the triplet grid (a third / sixth), 2 on a sixteenth (1/4, 3/4), 0 else."""
    out = np.zeros(len(rows), np.int8)
    for i in np.nonzero(rows[:, 0] == 1)[0]:
        if int(rows[i, 3]) <= 0:
            continue
        w = Fraction(int(rows[i, 2]), int(rows[i, 3]))
        d = (w - w.numerator // w.denominator).denominator
        out[i] = 1 if d in (3, 6) else (2 if d == 4 else 0)
    return out


def audit(ds: ScoreSegmentDataset, idx) -> dict:
    ds.mask_contradicted = True
    ds.grid_noise = None
    tot = defaultdict(lambda: defaultdict(int))
    windows = defaultdict(int)
    for i in idx:
        i = int(i)
        it = ds[i]
        c = it["contradicted"].numpy()
        rows = np.asarray(ds._nl["labels"][ds.store.index[i, 0]:ds.store.index[i, 1]])   # no context, no noise: aligned
        fam = written_family(rows) if len(rows) == len(c) else None
        genre = ds.store.meta[i].get("genre") or "all"
        for g in {genre, "all"}:
            t = tot[g]
            t["windows"] += 1
            t["triplet_played_straight"] += int((c == TRIPLET_PLAYED_STRAIGHT).sum())
            t["straight_played_triplet"] += int((c == STRAIGHT_PLAYED_TRIPLET).sum())
            if fam is not None:
                t["written_triplet"] += int((fam == 1).sum())
                t["written_sixteenth"] += int((fam == 2).sum())
            t["notes"] += int((it["labels"][:, 0] == 1).sum())
        n1 = int((c == TRIPLET_PLAYED_STRAIGHT).sum())
        if n1 >= 3:
            windows["with_3_or_more_triplets_played_straight"] += 1
    out = {}
    for g, t in tot.items():
        r = dict(t)
        if t.get("written_triplet"):
            r["share_of_written_triplets_played_straight"] = t["triplet_played_straight"] / t["written_triplet"]
        if t.get("written_sixteenth"):
            r["share_of_written_sixteenths_played_triplet"] = t["straight_played_triplet"] / t["written_sixteenth"]
        r["per_1000_notes_triplet_played_straight"] = 1000 * t["triplet_played_straight"] / max(t["notes"], 1)
        out[g] = r
    out["all"].update(windows)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--splits", default="train,train_gangqinpu,val,val_gangqinpu")
    ap.add_argument("--windows", type=int, default=20000, help="windows per split (a fixed random draw; 0 = all)")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="outputs/label_audit.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    data = Path(args.data)
    res = {}
    for split in args.splits.split(","):
        if not (data / split / LABEL_FILE).exists():
            print(f"skip {split}: no {data / split / LABEL_FILE}")
            continue
        ds = ScoreSegmentDataset(str(data / split), str(data / "vocab.json"), notelabels=True)
        n = len(ds)
        idx = np.arange(n) if not args.windows or args.windows >= n else \
            np.sort(np.random.default_rng(args.seed).choice(n, args.windows, replace=False))
        res[split] = audit(ds, idx)
        a = res[split]["all"]
        print(f"{split} ({len(idx)} windows): written triplets played straight "
              f"{a.get('share_of_written_triplets_played_straight', float('nan')):.1%} "
              f"({a['triplet_played_straight']} of {a.get('written_triplet', '?')}), written sixteenths played as "
              f"triplets {a.get('share_of_written_sixteenths_played_triplet', float('nan')):.1%}; "
              f"{a['per_1000_notes_triplet_played_straight']:.1f} per 1000 notes", flush=True)
        for g, r in sorted(res[split].items()):
            if g != "all" and r["windows"] >= 50:
                print(f"   {g:18s} {r['windows']:6d} windows: written triplets played straight "
                      f"{r.get('share_of_written_triplets_played_straight', float('nan')):.1%}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
