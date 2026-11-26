"""Gangqinpu's written scores as note-labeller (and seq2seq) training windows, like PERiScoPe's.

    python scripts/build_gangqinpu_windows.py                       # train, val, test
    python scripts/build_gangqinpu_windows.py --splits val --limit 50
    python scripts/build_notelabels.py --splits train_gangqinpu,val_gangqinpu,test_gangqinpu

Reads the release's Parquet (scripts/fetch_gangqinpu.py; ``--release``, default
data/raw/gangqinpu) and writes data/processed/{train,val,test}_gangqinpu/ in the store format of
every other split (src/preprocessing/store.py), cut into windows by the configuration's
``segment`` settings (training windows overlap, validation and test ones do not), with the
vocabulary already in data/processed/vocab.json (a token it does not have becomes <unk> and is
counted in the report). Each split's report.json says what the converter changed
(src/preprocessing/gangqinpu.py score_measures), which performances were skipped and why, and how
many windows the segmenter dropped.

The splits are the release's own: its README says they were made jointly with ASAP by score
similarity, and every group holding one of Beyer & Dai's ASAP test pieces is in its test split, so
train_gangqinpu is safe for our benchmark. Windows carry ``source: transkun_aligned``
(src/dataset.py: the noise model adds extra notes only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_dataset import _write_shard, iter_shards  # noqa: E402
from src.preprocessing import gangqinpu as G  # noqa: E402
from src.preprocessing.segment import SegmentConfig, segment_windows  # noqa: E402
from src.preprocessing.store import write_split  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402

SPLITS = ("train", "val", "test")


def process_rows(rows: list, split: str, cfg: SegmentConfig, shard_dir: str, name: str):
    """One chunk of release rows -> a shard of windows, a report per performance, the shard's
    sizes and its token counts."""
    import warnings
    warnings.filterwarnings("ignore")
    segs, reports, toks = [], [], Counter()
    for r in rows:
        perf = f"gangqinpu/{r['sample_id']}"
        try:
            res = G.labeller_performance(r)
            if isinstance(res, str):
                reports.append({"performance": perf, "dataset": "gangqinpu", "skipped": res})
                continue
            notes, pedal, tl, measures, info = res
            seed = int(hashlib.sha1(perf.encode()).hexdigest()[:8], 16)
            s, st = segment_windows(notes, pedal, tl, measures, cfg, G.window_meta(r, split),
                                    train=split == "train", seed=seed)
            st.update(info, genre=r.get("genre") or "unknown")
        except Exception as e:  # noqa: BLE001
            s, st = [], {"error": repr(e)[:300]}
        for x in s:
            toks.update(x["tokens"] + x.get("prompt", []))
        segs += s
        reports.append({"performance": perf, "dataset": "gangqinpu", **st})
    shard, reports, sizes, _ = _write_shard(segs, reports, f"{split}_gangqinpu", name, shard_dir)
    return shard, reports, sizes, toks


def build(release: Path, out: Path, split: str, cfg: SegmentConfig, vocab: Vocab, workers: int,
          limit: int = 0, chunk: int = 16) -> dict:
    """One split: every row windowed in parallel (a bounded number of chunks in flight, so the
    release never has to fit in memory), then the split written. Returns its report."""
    dst, shard_dir = out / f"{split}_gangqinpu", out / f"_shards_gangqinpu_{split}"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.rmtree(shard_dir, ignore_errors=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shards, reports, toks, sizes = [], [], Counter(), [0, 0, 0, 0]

    def collect(done):
        nonlocal sizes
        for fu in done:
            shard, rep, sz, tc = fu.result()
            shards.append(shard)
            reports.extend(rep)
            toks.update(tc)
            sizes = [a + b for a, b in zip(sizes, sz)]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        pending, buf, n_rows, k = set(), [], 0, 0
        for row in G.iter_rows(release, split, columns=G.SCORE_COLUMNS, batch_size=chunk):
            buf.append(row)
            n_rows += 1
            if len(buf) >= chunk or (limit and n_rows >= limit):
                pending.add(ex.submit(process_rows, buf, split, cfg, str(shard_dir), f"{split}:{k}"))
                buf, k = [], k + 1
                if len(pending) >= 2 * workers:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    collect(done)
                    print(f"{split}: {len(reports)} performances, {sizes[0]} windows, {time.time() - t0:.0f} s",
                          flush=True)
            if limit and n_rows >= limit:
                break
        if buf:
            pending.add(ex.submit(process_rows, buf, split, cfg, str(shard_dir), f"{split}:{k}"))
        done, _ = wait(pending)
        collect(done)
    write_split(dst, iter_shards(shards), tuple(sizes), vocab)
    shutil.rmtree(shard_dir, ignore_errors=True)

    oov = Counter({t: n for t, n in toks.items() if t not in vocab.stoi})
    conv, drops, skips, genres = Counter(), Counter(), Counter(), Counter()
    for r in reports:
        if "skipped" in r or "error" in r:
            skips[(r.get("skipped") or r.get("error", ""))[:60]] += 1
            continue
        genres[r.get("genre", "unknown")] += 1
        for key in ("written_note", "grace_note", "tie_in_bar", "held_across_bar_line", "duplicate_note",
                    "zero_length_note", "input_notes", "performance_notes", "interpolated_performance_notes"):
            conv[key] += r.get(key, 0)
        for key in ("too_long", "empty", "long_bar", "low_pitch_f1"):
            drops[key] += r.get(key, 0)
    report = {"split": f"{split}_gangqinpu", "rows": n_rows, "windows": sizes[0], "notes": sizes[1],
              "tokens": sizes[3], "performances": sum(1 for r in reports if r.get("segments")),
              "skipped_performances": dict(skips.most_common()), "converter": dict(conv),
              "dropped_windows": dict(drops), "genres": dict(genres.most_common()),
              "tokens_not_in_vocab": {"count": sum(oov.values()), "kinds": dict(oov.most_common(30))},
              "minutes": round((time.time() - t0) / 60, 1), "performances_detail": reports}
    (dst / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default="data/raw/gangqinpu")
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--out", default=None, help="default: the configuration's out_dir")
    ap.add_argument("--workers", type=int, default=0, help="default: the configuration's workers")
    ap.add_argument("--limit", type=int, default=0, help="the first N rows of each split only (debugging)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out or cfg["out_dir"])
    if not (out / "vocab.json").exists():
        sys.exit(f"{out / 'vocab.json'} not found: build the main data first (scripts/build_dataset.py)")
    vocab = Vocab.load(out / "vocab.json")
    seg = dict(cfg["segment"])
    seg["window_notes"] = tuple(seg.get("window_notes", SegmentConfig.window_notes))
    base = SegmentConfig(**seg)
    eval_cfg = replace(base, stride=base.bars_per_segment)
    workers = args.workers or cfg.get("workers", 4)
    for split in [s for s in args.splits.split(",") if s]:
        if split not in SPLITS:
            sys.exit(f"--splits: {split!r} is not one of {', '.join(SPLITS)}")
        if not G.split_files(Path(args.release), split):
            sys.exit(f"no {Path(args.release) / 'data' / (G.SPLITS[split] + '-*.parquet')}: "
                     "scripts/fetch_gangqinpu.py first")
        rep = build(Path(args.release), out, split, base if split == "train" else eval_cfg, vocab, workers,
                    args.limit)
        print(f"{rep['split']}: {rep['performances']} of {rep['rows']} performances -> {rep['windows']} windows, "
              f"{rep['notes']} notes ({rep['minutes']} min) | skipped {rep['skipped_performances']} | windows "
              f"dropped {rep['dropped_windows']} | tokens not in the vocabulary: {rep['tokens_not_in_vocab']['count']}"
              f" | converter {rep['converter']}", flush=True)


if __name__ == "__main__":
    main()
