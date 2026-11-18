"""Rebuild the MusicXML of an evaluation from its saved predictions (no GPU, no decoding).

    python scripts/export_musicxml.py --run sf_base                 # first 5 pieces per condition
    python scripts/export_musicxml.py --run sf_base --n 20 --conditions periscope --mode none

Reads outputs/runs/<run>/pred_pieces_<ckpt>_<condition>.jsonl (written by evaluate_pieces.py),
stitches each piece's windows in order exactly as evaluate_pieces.py does, and writes
musicxml/<condition>/<piece>_{pred,true}.musicxml. Pieces are taken in the same sorted order as
evaluate_pieces.py --musicxml N, so this regenerates (and overwrites) the same files, e.g. after
a fix to src/preprocessing/detokenize.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_pieces import SPLITS, stitch, write_xml  # noqa: E402
from src.preprocessing.store import SegmentStore  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402

D = Path("data/processed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best", help="checkpoint stem in the file names (best, last)")
    ap.add_argument("--conditions", default="clean,synthetic,real,real_kong,periscope")
    ap.add_argument("--mode", default="chained", choices=["chained", "oracle", "none"])
    ap.add_argument("--n", type=int, default=5, help="pieces per condition (0 = all)")
    args = ap.parse_args()
    run = Path("outputs/runs") / args.run
    vocab = Vocab.load(str(D / "vocab.json"))

    for cond in args.conditions.split(","):
        pred_file = run / f"pred_pieces_{args.ckpt}_{cond}.jsonl"
        if not pred_file.exists():
            print(f"skip {cond}: {pred_file} not found")
            continue
        preds = defaultdict(dict)  # performance -> win -> tokens
        for line in open(pred_file):
            r = json.loads(line)
            if r["mode"] == args.mode:
                preds[r["performance"]][r["win"]] = r["pred"]
        if not preds:
            print(f"skip {cond}: no {args.mode} predictions in {pred_file.name}")
            continue
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
            a = write_xml(pred, d / f"{slug}_pred.musicxml")
            b = write_xml(true, d / f"{slug}_true.musicxml")
            ok += a == "ok"
            print(f"  {cond}/{slug}: pred {a}, true {b}", flush=True)
        print(f"{cond}: {ok}/{len(names)} predicted scores written")


if __name__ == "__main__":
    main()
