"""Checks for ``mode: windows`` segmentation on real performances.

    python -m tests.test_windows --n 40                      # ASAP (needs data/raw/asap_bd)
    python -m tests.test_windows --source pianocore --n 40   # PERiScoPe (needs fetch_pianocore.py)

For each performance, cut into non-overlapping eval windows with all quality filters off, and check:
  1. stitching: the windows' targets, concatenated, parse to exactly the same bars / time and key
     signatures / notes as tokenizing the whole piece (nothing lost or duplicated at a cut);
  2. every performed note is an in-window input (onset >= 0) of exactly one window, so carried-over
     notes (onset < 0) are only ever extra copies;
  3. each window's prompt equals what inference would feed: ``prompt_from_output`` of all earlier
     windows' targets;
  4. some windows and whole stitched pieces render to MusicXML;
  5. each window's input notes match its target's pitches (median F1 >= 0.85, at most 5% of
     windows below 0.6), i.e. the time map puts performed notes with the right score notes.
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
from src.preprocessing.segment import (SegmentConfig, pitch_f1, prompt_from_output, segment_windows,
                                       timeline_from_asap)


def semantic(tokens):
    """Bars as (ts, key, sorted notes) - ignores redundant pos/staff/ts/key tokens and pedal."""
    return [(b.ts, b.key, sorted((n.pos, n.staff, n.pitch, n.dur, n.grace, n.tie, n.arp, tuple(n.orns))
                                 for n in b.notes)) for b in parse_tokens(tokens)]


def no_pedal(tokens):
    return [t for t in tokens if not t.startswith("ped_")]


def asap_cases(args, rng):
    """Yield (name, notes, pedal, timeline, measures, bars) for random ASAP performances."""
    root = Path(args.asap)
    ann = json.load(open(root / "asap_annotations.json"))
    rows = [r for r in csv.DictReader(open(root / "metadata.csv"))
            if ann.get(r["midi_performance"], {}).get("score_and_performance_aligned")
            and isinstance(ann[r["midi_performance"]].get("downbeats_score_map"), list)]
    for k in rng.choice(len(rows), min(args.n, len(rows)), replace=False):
        r, a = rows[k], ann[rows[k]["midi_performance"]]
        try:
            ms = load_score(root / r["xml_score"])
        except Exception:  # noqa: BLE001  (unparseable score: build_dataset skips these too)
            yield r["midi_performance"], None
            continue
        notes, pedal = load_performance(root / r["midi_performance"])
        tl = timeline_from_asap(a, ms)
        yield r["midi_performance"], (None if isinstance(tl, str) else (notes, pedal, tl, ms))


def pianocore_cases(args, rng):
    """Same for downloaded PERiScoPe rows: RAW Transkun input + alignment timeline (the build's path)."""
    import pyarrow.parquet as pq

    from src.preprocessing.pianocore import load_unfolded_score, pianocore_performance, score_midi_shift
    d = Path(args.pianocore)
    rows = [(f, i) for f in sorted(d.glob("rows-*.parquet")) for i in range(pq.ParquetFile(f).metadata.num_rows)]
    xml = {}
    for f in d.glob("scores-*.parquet"):
        for r in pq.read_table(f).to_pylist():
            xml[r["score_xml_path"]] = r  # score_id is not unique in PianoCoRe
    for k in rng.choice(len(rows), min(args.n, len(rows)), replace=False):
        f, i = rows[k]
        row = pq.read_table(f).slice(i, 1).to_pylist()[0]
        try:
            sc = xml[row["score_xml_path"]]
            ms = load_unfolded_score(sc["score_xml_bytes"], Path(sc["score_xml_path"]).suffix)
            shift, match = score_midi_shift(row["refined_score_midi_bytes"], ms)
        except Exception:  # noqa: BLE001
            yield row["id"], None
            continue
        res = pianocore_performance(row, ms, shift) if match >= 0.9 else "score mismatch"
        yield row["id"], (None if isinstance(res, str) else res[:3] + (ms,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["asap", "pianocore"], default="asap")
    ap.add_argument("--asap", default="data/raw/asap_bd")
    ap.add_argument("--pianocore", default="data/raw/pianocore/periscope")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(args.seed)
    cases = asap_cases(args, rng) if args.source == "asap" else pianocore_cases(args, rng)
    cfg = replace(SegmentConfig(), min_pitch_f1=0.0, max_input_notes=10**6, max_target_tokens=10**7,
                  max_bar_quarters=10**6)

    counts = {"performances": 0, "skipped": 0, "windows": 0, "stitch_ok": 0, "notes_once": 0,
              "prompts_checked": 0, "prompts_ok": 0}
    failures, render, f1s = [], [], []
    for name, case in cases:
        if case is None:
            counts["skipped"] += 1
            continue
        notes, pedal, tl, ms = case
        segs, st = segment_windows(notes, pedal, tl, ms, cfg, {"performance": name}, train=False)
        if not segs:
            counts["skipped"] += 1
            continue
        counts["performances"] += 1
        counts["windows"] += len(segs)
        f1s += [pitch_f1(s["notes"][s["notes"][:, 1] >= 0, 0], s["tokens"]) for s in segs]

        stitched = [t for s in segs for t in s["tokens"]]
        ok = semantic(stitched) == semantic(tokenize_bars(ms, tl.bars))
        counts["stitch_ok"] += ok
        if not ok:
            failures.append(("stitch", name))

        n_in = sum(int((s["notes"][:, 1] >= 0).sum()) for s in segs)
        counts["notes_once"] += n_in == len(notes)
        if n_in != len(notes):
            failures.append(("notes", name, n_in, len(notes)))

        history = []
        for prev, cur in zip(segs, segs[1:]):
            history += prev["tokens"]
            counts["prompts_checked"] += 1
            same = semantic(prompt_from_output(history)) == semantic(no_pedal(cur["prompt"]))
            counts["prompts_ok"] += same
            if not same:
                failures.append(("prompt", name, cur["win"]))
        if len(render) < 3:
            render.append(stitched)
        render.append(segs[len(segs) // 2]["tokens"])

    rendered = 0
    with tempfile.TemporaryDirectory() as tmp:
        for k, toks in enumerate(render[:12]):
            tokens_to_musicxml(toks, Path(tmp) / f"{k}.musicxml")
            rendered += 1
    print(counts, "| rendered:", rendered)
    print(f"input pitches vs target pitches per window (F1): median {np.median(f1s):.3f}, "
          f"10th percentile {np.percentile(f1s, 10):.3f}")
    for f in failures[:10]:
        print("FAIL", f)
    p = counts["performances"]
    assert p and counts["stitch_ok"] == p and counts["notes_once"] == p
    assert counts["prompts_ok"] == counts["prompts_checked"]
    # inputs must belong with their targets (catches time-map / alignment errors that the
    # structural checks above cannot see)
    assert np.median(f1s) >= 0.85, "window inputs don't match their targets"
    assert np.mean(np.array(f1s) < 0.6) <= 0.05, "too many windows whose inputs don't match the target"


if __name__ == "__main__":
    main()
