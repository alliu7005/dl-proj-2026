"""Composition-level train/val/test split (done BEFORE segmentation to avoid leakage)."""
from __future__ import annotations

import hashlib
import json
import re


def composition_id(dataset: str, piece: str) -> str:
    """Normalized ID shared by every performance/version of the same composition.

    Strips variants such as ``_no_repeat`` so they can't land in different splits.
    When adding PianoCoRe, map its pieces to the SAME ids as ASAP (composer/work/movement)
    so a composition never appears in two splits across datasets.
    """
    pid = piece.lower().replace("\\", "/")
    pid = re.sub(r"_no_repeats?$", "", pid)
    return pid


def assign_splits(comp_ids, ratios=(0.8, 0.1, 0.1), seed: int = 0, fixed: dict | None = None) -> dict:
    """Deterministic split by hashing composition ids. ``fixed`` (id -> split) overrides,
    e.g. to reuse Beyer & Dai's test pieces for a comparable baseline."""
    fixed = fixed or {}
    if fixed:  # test set is pinned: unlisted compositions only go to train/val
        ratios = (ratios[0] / (ratios[0] + ratios[1]), ratios[1] / (ratios[0] + ratios[1]), 0.0)
    out = {}
    for cid in sorted(set(comp_ids)):
        if cid in fixed:
            out[cid] = fixed[cid]
            continue
        h = int(hashlib.sha1(f"{seed}:{cid}".encode()).hexdigest(), 16) / 16 ** 40
        out[cid] = "train" if h < ratios[0] else "val" if h < ratios[0] + ratios[1] else "test"
    return out


def load_fixed_split(path: str | None) -> tuple[dict, set]:
    """Read a split file ``{"pieces": {piece: split}, "exclude_performances": [...]}``
    (see scripts/make_bd_split.py). Returns (composition_id -> split, excluded performance paths)."""
    if not path:
        return {}, set()
    with open(path) as f:
        d = json.load(f)
    pieces = d.get("pieces", d)
    fixed = {}
    for k, v in pieces.items():
        cid = composition_id("", k)
        if fixed.setdefault(cid, v) != v:
            raise ValueError(f"split file puts variants of {cid} in different splits")
    return fixed, set(d.get("exclude_performances", []))
