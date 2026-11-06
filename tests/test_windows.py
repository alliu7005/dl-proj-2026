"""Checks for ``mode: windows`` segmentation on real ASAP performances (needs data/raw/asap_bd).

    python -m tests.test_windows --n 40

For each performance, cut into non-overlapping eval windows with all quality filters off, and check:
  1. stitching: the windows' targets, concatenated, parse to exactly the same bars / time and key
     signatures / notes as tokenizing the whole piece (nothing lost or duplicated at a cut);
  2. every performed note is an in-window input (onset >= 0) of exactly one window, so carried-over
     notes (onset < 0) are only ever extra copies;
  3. each window's prompt equals what inference would feed: ``prompt_from_output`` of all earlier
     windows' targets;
  4. some windows and whole stitched pieces render to MusicXML.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.preprocessing.detokenize import parse_tokens, tokens_to_musicxml
from src.preprocessing.performance import load_performance
from src.preprocessing.score import load_score, tokenize_bars
from src.preprocessing.segment import (SegmentConfig, bar_timeline, parse_downbeat_map,
                                       prompt_from_output, segment_windows)


def semantic(tokens):
    """Bars as (ts, key, sorted notes) - ignores redundant pos/staff/ts/key tokens and pedal."""
    return [(b.ts, b.key, sorted((n.pos, n.staff, n.pitch, n.dur, n.grace, n.tie, n.arp, tuple(n.orns))
                                 for n in b.notes)) for b in parse_tokens(tokens)]


def no_pedal(tokens):
    return [t for t in tokens if not t.startswith("ped_")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asap", default="data/raw/asap_bd")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    root = Path(args.asap)
    ann = json.load(open(root / "asap_annotations.json"))
    rows = [r for r in csv.DictReader(open(root / "metadata.csv"))
            if ann.get(r["midi_performance"], {}).get("score_and_performance_aligned")
            and isinstance(ann[r["midi_performance"]].get("downbeats_score_map"), list)]
    rows = [rows[i] for i in np.random.default_rng(args.seed).choice(len(rows), args.n, replace=False)]
    cfg = replace(SegmentConfig(), min_pitch_f1=0.0, max_input_notes=10**6, max_target_tokens=10**7,
                  max_bar_quarters=10**6)

    counts = {"performances": 0, "skipped": 0, "windows": 0, "stitch_ok": 0, "notes_once": 0,
              "prompts_checked": 0, "prompts_ok": 0}
    failures, render = [], []
    for r in rows:
        a = ann[r["midi_performance"]]
        try:
            ms = load_score(root / r["xml_score"])
        except Exception:  # noqa: BLE001  (unparseable score: build_dataset skips these too)
            counts["skipped"] += 1
            continue
        notes, pedal = load_performance(root / r["midi_performance"])
        segs, st = segment_windows(notes, pedal, a, ms, cfg, {"performance": r["midi_performance"]}, train=False)
        if not segs:
            counts["skipped"] += 1
            continue
        counts["performances"] += 1
        counts["windows"] += len(segs)
        bars, *_ = bar_timeline(parse_downbeat_map(a["downbeats_score_map"]), ms, a["midi_score_downbeats"])

        stitched = [t for s in segs for t in s["tokens"]]
        ok = semantic(stitched) == semantic(tokenize_bars(ms, bars))
        counts["stitch_ok"] += ok
        if not ok:
            failures.append(("stitch", r["midi_performance"]))

        n_in = sum(int((s["notes"][:, 1] >= 0).sum()) for s in segs)
        counts["notes_once"] += n_in == len(notes)
        if n_in != len(notes):
            failures.append(("notes", r["midi_performance"], n_in, len(notes)))

        history = []
        for prev, cur in zip(segs, segs[1:]):
            history += prev["tokens"]
            counts["prompts_checked"] += 1
            same = semantic(prompt_from_output(history)) == semantic(no_pedal(cur["prompt"]))
            counts["prompts_ok"] += same
            if not same:
                failures.append(("prompt", r["midi_performance"], cur["win"]))
        if len(render) < 3:
            render.append(stitched)
        render.append(segs[len(segs) // 2]["tokens"])

    rendered = 0
    with tempfile.TemporaryDirectory() as tmp:
        for k, toks in enumerate(render[:12]):
            tokens_to_musicxml(toks, Path(tmp) / f"{k}.musicxml")
            rendered += 1
    print(counts, "| rendered:", rendered)
    for f in failures[:10]:
        print("FAIL", f)
    p = counts["performances"]
    assert p and counts["stitch_ok"] == p and counts["notes_once"] == p
    assert counts["prompts_ok"] == counts["prompts_checked"]


if __name__ == "__main__":
    main()
