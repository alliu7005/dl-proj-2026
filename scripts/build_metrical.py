"""Precompute the metrical labels of every split (for --aux-metrical training), once per data build.

    python scripts/build_metrical.py                       # every split in data/processed
    python scripts/build_metrical.py --splits train,val --workers 8

Aligns each window's performed notes to its score (src/metrical.py) and writes
data/processed/<split>/metrical.npy: int8 [notes, 2] = (downbeat, phase), -1 where unknown, row-
aligned with notes.npy. ScoreSegmentDataset(metrical=True) reads it (memory-mapped); without it the
labels are aligned on the fly, which costs ~5 ms per window in every data-loader worker each epoch.
Rebuild after rebuilding the data (the file is ignored if its size no longer matches the split).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrical import IGNORE, build_split_labels  # noqa: E402

D = Path("data/processed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="", help="comma-separated (default: every split with an index.npy)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    splits = [s for s in args.splits.split(",") if s] or sorted(p.parent.name for p in D.glob("*/index.npy"))
    for split in splits:
        if not (D / split / "index.npy").exists():
            print(f"skip {split}: no index.npy")
            continue
        t = time.time()
        lab = build_split_labels(D / split, D / "vocab.json", workers=args.workers)
        n = len(lab)
        if n == 0:
            print(f"{split}: empty")
            continue
        labelled = lab[:, 0] != IGNORE
        print(f"{split}: {labelled.mean():.1%} of {n} notes labelled (carried-over and context notes "
              f"never are); downbeats {np.mean(lab[labelled, 0] == 1):.1%} of labelled; "
              f"{time.time() - t:.0f} s -> {D / split / 'metrical.npy'}", flush=True)


if __name__ == "__main__":
    main()
