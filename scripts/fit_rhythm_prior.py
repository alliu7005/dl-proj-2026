"""The rhythm prior of the beat decoding (src/beat_decode.py): how often a written beat is straight,
triplet or other, and which family follows which, counted on the training scores' note labels.

    python scripts/fit_rhythm_prior.py                       # train + train_gangqinpu -> outputs/rhythm_prior/prior.json
    python scripts/fit_rhythm_prior.py --splits train --windows 2000 --out /tmp/p.json

A beat (its notes' bar and quarter, the true grid) is triplet if a note in it is on a triplet-only
split (1/3, 2/3, 1/6, ...), other if on another non-straight one (quintuplets, ...), else straight
(on the quarter, 1/2, 1/4, ...). Consecutive beats of a window give the transitions. The windows are
a fixed random draw (they overlap; every beat counts about once per window it is in). Written
scores only: no audio, no model. Add-one smoothing; the families' shares per split go in the file
as well (how much rarer triplets are in pop than in classical).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.beat_decode import OTHER, STRAIGHT, TRIPLET, Families  # noqa: E402

NAMES = ("straight", "triplet", "other")


def beat_families(labels: np.ndarray, bar_est: np.ndarray, own: np.ndarray, fam_of_class: np.ndarray) -> list[int]:
    """A window's beats in time order, each one's family. ``labels`` [n, >=4]: delta, pos, sub in
    columns 1-3 (-1: none)."""
    beats: dict = {}
    for i in np.flatnonzero(own & (labels[:, 1] >= 0) & (labels[:, 2] >= 0) & (labels[:, 3] >= 0)):
        key = (int(bar_est[i]) + int(labels[i, 1]) - 1, int(labels[i, 2]))
        beats.setdefault(key, set()).add(int(fam_of_class[int(labels[i, 3])]))
    out = []
    for key in sorted(beats):
        f = beats[key]
        out.append(TRIPLET if TRIPLET in f else OTHER if OTHER in f else STRAIGHT)
    return out


def count(seqs, init=None, trans=None):
    init = np.zeros(3) if init is None else init
    trans = np.zeros((3, 3)) if trans is None else trans
    for seq in seqs:
        if seq:
            init[seq[0]] += 1
        for a, b in zip(seq, seq[1:]):
            trans[a, b] += 1
    return init, trans


def table(init: np.ndarray, trans: np.ndarray) -> dict:
    i = (init + 1) / (init + 1).sum()
    t = (trans + 1) / (trans + 1).sum(1, keepdims=True)
    return {"log_init": np.log(i).tolist(), "log_trans": np.log(t).tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="train,train_gangqinpu")
    ap.add_argument("--windows", type=int, default=20000, help="per split (a fixed random draw)")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/rhythm_prior/prior.json")
    args = ap.parse_args(argv)
    from src.dataset import ROLE_WINDOW, ScoreSegmentDataset
    from src.notelabels import LABEL_FILE
    data = Path(args.data)
    init, trans = np.zeros(3), np.zeros((3, 3))
    by_split = {}
    for split in args.splits.split(","):
        if not (data / split / LABEL_FILE).exists():
            print(f"skip {split}: no {data / split / LABEL_FILE}")
            continue
        ds = ScoreSegmentDataset(str(data / split), str(data / "vocab.json"), notelabels=True)
        fams = Families(ds.label_space)
        fam_of_class = np.where(fams.fam < 0, STRAIGHT, fams.fam)
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(ds), min(args.windows, len(ds)), replace=False)
        seqs = []
        for i in idx:
            it = ds[int(i)]
            own = (it["roles"] <= ROLE_WINDOW).numpy()
            seqs.append(beat_families(it["labels"].numpy(), it["bar_est"].numpy(), own, fam_of_class))
        si, st = count(seqs)
        init += si
        trans += st
        n = st.sum(1) + np.bincount([s[-1] for s in seqs if s], minlength=3)   # every beat once
        share = n / max(n.sum(), 1)
        by_split[split] = {"windows": len(idx), "beats": int(n.sum()),
                           **{f"{NAMES[k]}_share": float(share[k]) for k in range(3)},
                           "triplet_after_triplet": float((st[1, 1] + 1) / (st[1].sum() + 3)),
                           "triplet_after_straight": float((st[0, 1] + 1) / (st[0].sum() + 3))}
        print(f"{split}: {len(idx)} windows, {int(n.sum())} beats: " + ", ".join(f"{NAMES[k]} {share[k]:.1%}" for k in range(3))
              + f"; P(triplet | triplet) {by_split[split]['triplet_after_triplet']:.2f}, "
                f"P(triplet | straight) {by_split[split]['triplet_after_straight']:.3f}", flush=True)
    if not by_split:
        raise SystemExit("no split with note labels")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**table(init, trans), "families": list(NAMES), "counts_init": init.tolist(),
                               "counts_trans": trans.tolist(), "by_split": by_split}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
