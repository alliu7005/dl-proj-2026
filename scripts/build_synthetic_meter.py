"""Synthetic meter-model performances from written scores (src/synthetic.py render) ->
data/processed/meter/{train,val}_synthetic.pkl + synthetic_report.json, for
scripts/train_meter.py --synthetic.

    python scripts/fetch_pdmx.py --max 20000                       # once, on the login node
    python scripts/build_synthetic_meter.py --scores data/raw/pdmx/mxl --max 3000 --workers 16

Which scores: scripts/fetch_pdmx.py unpacks PDMX's solo-piano scores only (its metadata: every
track a piano, no licence conflict); here a score with more than two parts none of which is named
piano / keyboard is left out too, and with the metadata CSV (--csv, default data/raw/pdmx/PDMX.csv
when there) the licence conflicts again. Each is parsed by src/preprocessing/score.py as every
other source's MusicXML. A score is train or val by a hash of its name (5% val), so a rebuild with
more scores keeps the earlier split. Each score is rendered --renders times (other tempi, timing,
articulation). The report counts scores kept / dropped (and why) and the time signatures of the
rendered beats.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PIANO_WORDS = ("piano", "keyboard", "pno", "klavier", "pianoforte")


def split_of(name: str, val_share: float = 0.05) -> str:
    h = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_share else "train"


def conflicts(csv_path: Path) -> set:
    """Base names of the scores PDMX marks as a licence conflict (empty without the column)."""
    out = set()
    with open(csv_path, newline="") as f:
        rows = csv.DictReader(f)
        cols = rows.fieldnames or []
        flag = next((c for c in cols if "license_conflict" in c.lower()), None)
        paths = [c for c in cols if "mxl" in c.lower() or c.lower() in ("path", "file")]
        if not flag or not paths:
            return out
        for r in rows:
            if str(r.get(flag, "")).strip().lower() in ("true", "1", "yes"):
                for c in paths:
                    if r.get(c):
                        out.add(Path(r[c]).stem)
    return out


def one(args):
    path, renders, seed = args
    import music21 as m21

    from src.preprocessing.score import load_score
    from src.synthetic import render
    name = Path(path).stem
    try:
        score = m21.converter.parse(str(path))
    except Exception as e:  # noqa: BLE001
        return name, [], f"unparsable ({type(e).__name__})"
    parts = list(score.parts)
    names = " ".join(str(getattr(p.getInstrument(returnDefault=True), "instrumentName", "") or p.partName or "")
                     for p in parts).lower()
    if not parts or (len(parts) > 2 and not any(w in names for w in PIANO_WORDS)):
        return name, [], "not piano"
    try:
        measures = load_score(score)
    except Exception as e:  # noqa: BLE001
        return name, [], f"unreadable measures ({type(e).__name__})"
    rng = np.random.default_rng(int(hashlib.sha1(f"{seed}:{name}".encode()).hexdigest()[:8], 16))
    perfs = []
    for r in range(renders):
        for p in render(measures, rng, name):
            p["performance"] = f"{p['performance']}/r{r}"
            perfs.append(p)
    return name, perfs, None if perfs else "no run of 8 regular bars"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="data/raw/pdmx/mxl")
    ap.add_argument("--csv", default="data/raw/pdmx/PDMX.csv")
    ap.add_argument("--max", type=int, default=3000, help="at most this many score files (0 = all)")
    ap.add_argument("--renders", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/processed")
    args = ap.parse_args(argv)
    files = sorted(p for p in Path(args.scores).glob("*") if p.suffix.lower() in (".mxl", ".musicxml", ".xml"))
    if not files:
        raise SystemExit(f"no scores in {args.scores}: python scripts/fetch_pdmx.py first")
    skip = conflicts(Path(args.csv)) if Path(args.csv).exists() else set()
    why = Counter()
    if skip:
        kept = [f for f in files if f.stem.split("__")[-1] not in skip and f.stem not in skip]
        why["licence conflict"] = len(files) - len(kept)
        files = kept
    if args.max and len(files) > args.max:
        files = [files[i] for i in sorted(np.random.default_rng(args.seed).choice(len(files), args.max, replace=False))]
    out = {"train": [], "val": []}
    with ProcessPoolExecutor(args.workers) as ex:
        for name, perfs, err in ex.map(one, [(str(f), args.renders, args.seed) for f in files], chunksize=8):
            if err:
                why[err] += 1
                continue
            why["kept"] += 1
            out[split_of(name)].extend(perfs)
    meter = Path(args.data) / "meter"
    meter.mkdir(parents=True, exist_ok=True)
    ts = Counter()
    for split, perfs in out.items():
        pickle.dump(perfs, open(meter / f"{split}_synthetic.pkl", "wb"))
        for p in perfs:
            ts.update(p["gt"]["ts_names"])
    report = {"scores_seen": len(files), "why": dict(why),
              "performances": {k: len(v) for k, v in out.items()},
              "beats": {k: int(sum(len(p["gt"]["t"]) for p in v)) for k, v in out.items()},
              "time_signatures_beats": dict(ts.most_common())}
    (meter / "synthetic_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if not out["train"]:
        raise SystemExit("no synthetic training performances")


if __name__ == "__main__":
    main()
