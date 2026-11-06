"""On-disk segment storage: flat, memory-mappable numpy arrays + an index (+ jsonl metadata).

Layout of one split directory, e.g. ``data/processed/train/``::

    notes.npy    float32 [total_notes, 4]   pitch, onset (s, rel. to segment), duration (s), velocity
    pedal.npy    float32 [total_pedal, 2]   sustain (on, off), seconds rel. to segment
    tokens.npy   int16   [total_tokens]     target token ids (no <bos>/<eos>), see vocab.json
    index.npy    int64   [n_segments, 6]    note_lo, note_hi, pedal_lo, pedal_hi, tok_lo, tok_hi
    meta.jsonl   one JSON object per segment (piece, performance, composition, bar_start, ...)

Segment i's notes are ``notes[index[i,0]:index[i,1]]`` etc. Arrays are opened with
``mmap_mode="r"`` so nothing is loaded into RAM until it is used, and DataLoader workers share pages.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .vocab import Vocab

META_KEYS = ("dataset", "piece", "composition", "performance", "split", "bar_start", "n_bars",
             "window_sec", "perf_t0", "source", "maestro_split")


def write_split(out_dir: Path, segment_iter, sizes: tuple[int, int, int, int], vocab: Vocab):
    """``sizes`` = (n_segments, n_notes, n_pedal, n_tokens), known up front so we can stream."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seg, n_notes, n_ped, n_tok = sizes
    notes = open_memmap(out_dir / "notes.npy", "w+", np.float32, (n_notes, 4))
    pedal = open_memmap(out_dir / "pedal.npy", "w+", np.float32, (n_ped, 2))
    tokens = open_memmap(out_dir / "tokens.npy", "w+", np.int16, (n_tok,))
    index = np.zeros((n_seg, 6), dtype=np.int64)
    a = b = c = i = 0
    with open(out_dir / "meta.jsonl", "w") as meta:
        for s in segment_iter:
            ids = vocab.encode(s["tokens"], add_special=False)
            na, nb, nc = len(s["notes"]), len(s["pedal"]), len(ids)
            notes[a:a + na] = s["notes"]
            pedal[b:b + nb] = s["pedal"]
            tokens[c:c + nc] = ids
            index[i] = (a, a + na, b, b + nb, c, c + nc)
            m = {k: (float(s[k]) if isinstance(s[k], np.floating) else s[k]) for k in META_KEYS if k in s}
            meta.write(json.dumps(m) + "\n")
            a, b, c, i = a + na, b + nb, c + nc, i + 1
    assert (i, a, b, c) == sizes, f"size mismatch {(i, a, b, c)} vs {sizes}"
    np.save(out_dir / "index.npy", index)
    notes.flush(), pedal.flush(), tokens.flush()


class SegmentStore:
    """Read-only, memory-mapped view of one split."""

    def __init__(self, split_dir: str | Path, load_meta: bool = True):
        d = Path(split_dir)
        self.notes = np.load(d / "notes.npy", mmap_mode="r")
        self.pedal = np.load(d / "pedal.npy", mmap_mode="r")
        self.tokens = np.load(d / "tokens.npy", mmap_mode="r")
        self.index = np.load(d / "index.npy")
        self.meta = [json.loads(line) for line in open(d / "meta.jsonl")] if load_meta else None

    def __len__(self):
        return len(self.index)

    def get(self, i: int):
        a0, a1, b0, b1, c0, c1 = self.index[i]
        return (np.array(self.notes[a0:a1]), np.array(self.pedal[b0:b1]),
                np.array(self.tokens[c0:c1], dtype=np.int64))
