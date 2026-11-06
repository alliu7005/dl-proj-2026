"""Build measure-aligned (performance MIDI -> score tokens) segments from ASAP.

    python scripts/build_dataset.py --config configs/preprocess.yaml

Writes to ``out_dir``: {train,val,test}/ (memory-mapped arrays, see src/preprocessing/store.py),
vocab.json, splits.json and report.json.

Pieces are processed in parallel and written to per-piece shards first; the vocabulary is then
built from the train shards and every split is streamed into flat arrays. Nothing has to fit in RAM
at once, so the same flow scales to PianoCoRe.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.performance import load_performance  # noqa: E402
from src.preprocessing.score import load_score  # noqa: E402
from src.preprocessing.segment import SegmentConfig, segment_pair  # noqa: E402
from src.preprocessing.split import assign_splits, composition_id, load_fixed_split  # noqa: E402
from src.preprocessing.store import write_split  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402

SPLITS = ("train", "val", "test")


def load_asap(root: Path, exclude: set[str]):
    with open(root / "asap_annotations.json") as f:
        ann = json.load(f)
    by_piece = defaultdict(list)
    skipped = Counter()
    with open(root / "metadata.csv", newline="") as f:
        for row in csv.DictReader(f):
            a = ann.get(row["midi_performance"])
            if row["midi_performance"] in exclude:
                skipped["excluded_by_split_file"] += 1
            elif a is None:
                skipped["no_annotation"] += 1
            elif not a.get("score_and_performance_aligned", False):
                skipped["not_aligned"] += 1
            elif not isinstance(a.get("downbeats_score_map"), list):
                skipped["no_downbeat_map"] += 1
            else:
                by_piece[row["folder"]].append((row, a))
    return by_piece, skipped


def process_piece(root: str, folder: str, items, cfg: SegmentConfig, split: str, cid: str,
                  shard_dir: str, fallback_root: str | None = None):
    import warnings
    warnings.filterwarnings("ignore")  # music21 MusicXML warnings (overfull measures, wedges, ...)
    root = Path(root)
    measures, err = None, None
    for r in [root] + ([Path(fallback_root)] if fallback_root else []):
        try:
            measures = load_score(r / items[0][0]["xml_score"])
            break
        except Exception as e:  # noqa: BLE001
            err = err or e
    if measures is None:
        return None, [{"piece": folder, "error": f"score: {err!r}"}], (0, 0, 0, 0), Counter()
    segs, reports = [], []
    for row, ann in items:
        perf = row["midi_performance"]
        try:
            notes, pedal = load_performance(root / perf)
            meta = {"dataset": "asap", "piece": folder, "composition": cid, "performance": perf,
                    "split": split, "source": "clean"}
            s, st = segment_pair(notes, pedal, ann, measures, cfg, meta)
        except Exception as e:  # noqa: BLE001
            s, st = [], {"error": repr(e)}
        segs += s
        reports.append({"performance": perf, **st})
    shard = Path(shard_dir) / f"{split}_{hashlib.sha1(folder.encode()).hexdigest()[:12]}.pkl"
    with open(shard, "wb") as f:
        pickle.dump(segs, f, protocol=pickle.HIGHEST_PROTOCOL)
    sizes = (len(segs), sum(len(s["notes"]) for s in segs), sum(len(s["pedal"]) for s in segs),
             sum(len(s["tokens"]) for s in segs))
    tok_counts = Counter(t for s in segs for t in s["tokens"]) if split == "train" else Counter()
    return str(shard), reports, sizes, tok_counts


def iter_shards(paths):
    for p in sorted(paths):
        with open(p, "rb") as f:
            segs = pickle.load(f)
        yield from sorted(segs, key=lambda s: (s["performance"], s["bar_start"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--limit", type=int, default=0, help="only process N pieces (debugging)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out = Path(cfg["asap_root"]), Path(cfg["out_dir"])
    shard_dir = out / "_shards"
    if out.exists():
        for sp in SPLITS:
            shutil.rmtree(out / sp, ignore_errors=True)
        shutil.rmtree(shard_dir, ignore_errors=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    fixed, exclude = load_fixed_split(cfg.get("fixed_split"))
    by_piece, skipped = load_asap(root, exclude)
    cids = {f: composition_id("asap", f) for f in by_piece}
    splits = assign_splits(cids.values(), tuple(cfg["split_ratios"]), cfg["seed"], fixed)
    pieces = sorted(by_piece)[: args.limit] if args.limit else sorted(by_piece)

    base = SegmentConfig(**cfg["segment"])
    eval_cfg = replace(base, stride=base.bars_per_segment)
    shards, sizes = defaultdict(list), defaultdict(lambda: [0, 0, 0, 0])
    reports, train_counts = [], Counter()
    with ProcessPoolExecutor(max_workers=cfg.get("workers", 4)) as ex:
        futs = [ex.submit(process_piece, str(root), f, by_piece[f],
                          base if splits[cids[f]] == "train" else eval_cfg,
                          splits[cids[f]], cids[f], str(shard_dir),
                          cfg.get("score_fallback_root")) for f in pieces]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="pieces"):
            shard, rep, sz, tc = fu.result()
            reports += rep
            train_counts += tc
            if shard:
                sp = Path(shard).name.split("_")[0]
                shards[sp].append(shard)
                sizes[sp] = [x + y for x, y in zip(sizes[sp], sz)]

    vocab = Vocab.from_counts(train_counts, cfg.get("vocab_min_count", 2))
    vocab.save(out / "vocab.json")
    for sp in SPLITS:
        write_split(out / sp, iter_shards(shards[sp]), tuple(sizes[sp]), vocab)
    if not cfg.get("keep_shards", False):
        shutil.rmtree(shard_dir, ignore_errors=True)

    json.dump({f: splits[cids[f]] for f in sorted(by_piece)}, open(out / "splits.json", "w"), indent=1)
    perf_by_split = defaultdict(set)
    for f in pieces:
        for row, _ in by_piece[f]:
            perf_by_split[splits[cids[f]]].add(row["midi_performance"])
    summary = {sp: {"segments": sizes[sp][0], "notes": sizes[sp][1], "tokens": sizes[sp][3],
                    "performances": len(perf_by_split[sp]),
                    "compositions": len({cids[f] for f in pieces if splits[cids[f]] == sp})}
               for sp in SPLITS}
    dropped = Counter()
    for r in reports:
        for k in ("too_long", "empty", "long_bar", "low_pitch_f1"):
            dropped[k] += r.get(k, 0)
    errs = [r for r in reports if "error" in r or "skipped" in r]
    json.dump({"summary": summary, "skipped_at_load": skipped, "dropped_segments": dropped,
               "vocab_size": len(vocab), "problems": errs, "performances": reports},
              open(out / "report.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("skipped at load:", dict(skipped), "| dropped segments:", dict(dropped),
          "| problem performances:", len(errs), "| vocab:", len(vocab))


if __name__ == "__main__":
    main()
