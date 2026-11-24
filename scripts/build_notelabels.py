"""Precompute the note labeller's labels and true grids for every split, once per data build.

    python scripts/build_notelabels.py                        # every split in data/processed
    python scripts/build_notelabels.py --splits train,val --workers 8

For each window, its performed notes are aligned to its target (src/notelabels.py align_notes):
every paired note gets its score labels (bar, position, written length with ties, staff, grace /
arpeggio / ornament flags), every other performed note "delete"; and the score's beats are timed
through the same alignment (the true grid the labeller is given in Phase 1). Writes
data/processed/<split>/notelabels.npz (row-aligned with notes.npy, the grid per window).
ScoreSegmentDataset(notelabels=True) reads it. Rebuild after rebuilding the data.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.notelabels import build_split  # noqa: E402

D = Path("data/processed")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="", help="comma-separated (default: every split with an index.npy)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--data", default=str(D))
    args = ap.parse_args(argv)
    d = Path(args.data)
    splits = [s for s in args.splits.split(",") if s] or sorted(p.parent.name for p in d.glob("*/index.npy"))
    for split in splits:
        if not (d / split / "index.npy").exists():
            print(f"skip {split}: no index.npy")
            continue
        t = time.time()
        out = build_split(d / split, d / "vocab.json", workers=args.workers)
        lab, gidx = out["labels"], out["grid_index"]
        live = lab[:, 0] >= 0
        if not live.any():
            print(f"{split}: empty")
            continue
        has_grid = (gidx[:, 1] - gidx[:, 0]) >= 2
        print(f"{split}: {live.sum()} performed notes, {np.mean(lab[live, 0] == 1):.1%} paired with a score note "
              f"({np.mean(lab[live, 0] == 0):.1%} to delete); grid for {has_grid.mean():.1%} of {len(gidx)} windows; "
              f"{time.time() - t:.0f} s -> {d / split / 'notelabels.npz'}", flush=True)


if __name__ == "__main__":
    main()
