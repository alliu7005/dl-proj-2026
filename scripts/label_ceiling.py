"""How good can the note labeller (v2) be? Render every window from its own TRUE labels and score
the result like any system (src/eval/metrics.py).

    python scripts/label_ceiling.py --split test            # clean MIDI
    python scripts/label_ceiling.py --split test_transcribed --limit 200

A perfect labeller writes exactly these scores, so the numbers are its ceiling. What it cannot
reach is what the label scheme loses (src/notelabels.py): score notes no performed note was paired
with (missed by the alignment, or not played), and tie continuations the bar-line rule writes
differently from the score (a tie inside a bar, a held note split another way). Also reported:
how many score notes each loss touches, and the extra performed notes (labelled "delete").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import Accumulator  # noqa: E402
from src.notelabels import align_notes, oracle_placed, render  # noqa: E402
from src.preprocessing.store import SegmentStore  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--limit", type=int, default=0, help="windows (0 = all)")
    args = ap.parse_args(argv)
    d = Path(args.data)
    store, vocab = SegmentStore(d / args.split), Vocab.load(str(d / "vocab.json"))
    n = len(store) if not args.limit else min(args.limit, len(store))
    acc = Accumulator()
    c = dict(score=0, sounding=0, sounding_matched=0, grace=0, grace_matched=0, ties=0, perf=0, extra=0)
    for i in range(n):
        notes, _, tokens = store.get(i)
        gt = vocab.decode(tokens.tolist())
        snotes, win, match = align_notes(notes[:, 1], notes[:, 0], gt)
        pred = render(win, oracle_placed(snotes, match))
        acc.add(gt, pred)
        matched = set(match.values())
        c["score"] += len(snotes)
        c["sounding"] += sum(not s.tie and not s.grace for s in snotes)
        c["sounding_matched"] += sum(k in matched for k, s in enumerate(snotes) if not s.tie and not s.grace)
        c["grace"] += sum(s.grace for s in snotes)
        c["grace_matched"] += sum(k in matched for k, s in enumerate(snotes) if s.grace)
        c["ties"] += sum(s.tie for s in snotes)
        live = int((notes[:, 1] >= 0).sum())
        c["perf"] += live
        c["extra"] += live - len(match)
    r = acc.result()
    print(f"{args.split}: {n} windows, {c['score']} score notes")
    for k in ("note_f1", "onset_f1", "onset_shift_f1", "duration_acc", "staff_acc", "bar_f1", "ts_acc", "key_acc",
              "grace_f1", "tie_f1", "arp_f1", "ornament_f1"):
        print(f"  {k:15s} {r[k]:.3f}")
    print(f"  sounding notes paired {c['sounding_matched'] / max(c['sounding'], 1):.3f} "
          f"({c['sounding'] - c['sounding_matched']} not), grace notes paired "
          f"{c['grace_matched'] / max(c['grace'], 1):.3f} of {c['grace']}, tie continuations "
          f"{c['ties'] / max(c['score'], 1):.3f} of score notes, performed notes left over (delete) "
          f"{c['extra'] / max(c['perf'], 1):.3f}")
    return r, c


if __name__ == "__main__":
    main()
