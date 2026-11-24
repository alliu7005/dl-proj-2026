"""Rebuild the MusicXML of an evaluation from its saved predictions (no GPU, no decoding).

    python scripts/export_musicxml.py --run s2s                     # first 5 pieces per condition
    python scripts/export_musicxml.py --run s2s --n 20 --conditions periscope --mode chained

Reads outputs/runs/<run>/pred_pieces_<ckpt>_<condition>.jsonl (written by evaluate_pieces.py),
stitches each piece's windows in order exactly as evaluate_pieces.py does, and writes
musicxml/<condition>/<piece>_true.musicxml plus the prediction:
  --mode auto (default)  meter, else guarded, else chained (the best mode evaluated) ->
                         <piece>_pred.musicxml, the
                         same file evaluate_pieces.py --musicxml N writes (regenerated, e.g. after a
                         fix to src/preprocessing/detokenize.py)
  --mode meter / piece / tsearch / tsearch_beat / ts_true / guarded / chained / oracle / none
                         -> <piece>_pred_<mode>.musicxml, so modes can be
                         compared side by side
Pieces are taken in the same sorted order as evaluate_pieces.py --musicxml N.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_pieces import SPLITS, write_xml  # noqa: E402
from src.decoding import stitch  # noqa: E402
from src.preprocessing.store import SegmentStore  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402

D = Path("data/processed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best", help="checkpoint stem in the file names (best, last)")
    ap.add_argument("--conditions", default="clean,synthetic,real,real_kong,periscope")
    ap.add_argument("--mode", default="auto", choices=["auto", "meter", "piece", "tsearch", "tsearch_beat", "ts_true",
                                                       "guarded", "chained", "oracle", "none"])
    ap.add_argument("--n", type=int, default=5, help="pieces per condition (0 = all)")
    args = ap.parse_args()
    run = Path("outputs/runs") / args.run
    vocab = Vocab.load(str(D / "vocab.json"))

    for cond in args.conditions.split(","):
        pred_file = run / f"pred_pieces_{args.ckpt}_{cond}.jsonl"
        if not pred_file.exists():
            print(f"skip {cond}: {pred_file} not found")
            continue
        by_mode = defaultdict(lambda: defaultdict(dict))  # mode -> performance -> win -> tokens
        for line in open(pred_file):
            r = json.loads(line)
            by_mode[r["mode"]][r["performance"]][r["win"]] = r["pred"]
        mode = args.mode
        if mode == "auto":
            mode = next(m for m in ("meter", "guarded", "chained") if m in by_mode or m == "chained")
        preds = by_mode.get(mode, {})
        if not preds:
            print(f"skip {cond}: no {mode} predictions in {pred_file.name} (has: {', '.join(sorted(by_mode))})")
            continue
        suffix = "pred" if args.mode == "auto" else f"pred_{mode}"
        store = SegmentStore(D / SPLITS[cond])
        truth = defaultdict(dict)
        for i, m in enumerate(store.meta):
            if m["performance"] in preds:
                truth[m["performance"]][m["win"]] = vocab.decode(store.get(i)[2].tolist())
        names = sorted(preds)[: args.n or None]
        ok = 0
        for p in names:
            pred, true = [], []
            for w in sorted(preds[p]):
                pred = stitch(pred, preds[p][w])
                true = stitch(true, truth[p].get(w, []))
            slug = re.sub(r"[^A-Za-z0-9_-]+", "_", p.rsplit(".", 1)[0])
            d = run / "musicxml" / cond
            a = write_xml(pred, d / f"{slug}_{suffix}.musicxml")
            b = write_xml(true, d / f"{slug}_true.musicxml")
            ok += a == "ok"
            print(f"  {cond}/{slug}: {mode} {a}, true {b}", flush=True)
        print(f"{cond}: {ok}/{len(names)} predicted scores written ({mode} mode -> *_{suffix}.musicxml)")


if __name__ == "__main__":
    main()
