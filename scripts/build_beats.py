"""Ground-truth beats for every performance of the processed splits -> <split>/beats_gt.npz.

    python scripts/build_beats.py                                  # every split that exists
    python scripts/build_beats.py --splits test,test_transcribed --workers 8

Per performance (src/beats.py):
  ASAP        the human beat / downbeat annotations (asap_annotations.json: performance_beats,
              performance_downbeats), in the performance's own time, the frame of the stored notes
              (transcribed splits: the transcription was shifted onto that timeline when the split
              was built, so the same annotations apply)
  PERiScoPe   the score's beats timed by the note alignment of every window (window_grid), joined
              by global bar number (merge_performance). Measured against the ASAP annotations on
              the ASAP test split (printed at the end with --check): ~84% of beats within 70 ms.

Also writes <split>/beats_gt.json: how many performances came from each source, and with --check
the agreement of the alignment-derived beats with the annotations (beat / downbeat F-measure).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.beats import beats_path, f_measure, merge_performance, save_beats, window_grid  # noqa: E402
from src.preprocessing.store import SegmentStore  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402

D = Path("data/processed")
SPLITS = ("train", "val", "test", "test_periscope", "train_transcribed", "val_transcribed", "test_transcribed",
          "test_transcribed_kong")


def annotated(a: dict):
    """(times, down) from one ASAP annotation entry."""
    t = np.asarray(a["performance_beats"], np.float64)
    downs = np.asarray(a.get("performance_downbeats", []), np.float64)
    down = np.isin(np.round(t, 6), np.round(downs, 6))
    order = np.argsort(t, kind="stable")
    return t[order], down[order]


def _grids(args):
    split_dir, vocab_path, idx = args
    store, vocab = SegmentStore(split_dir), Vocab.load(vocab_path)
    out = []
    for i in idx:
        notes, _, toks = store.get(i)
        m = store.meta[i]
        out.append((m["performance"], m["bar_start"], m["perf_t0"],
                    window_grid(notes[:, 1], notes[:, 0], vocab.decode(toks.tolist()))))
    return out


def aligned_beats(split_dir: Path, vocab_path: Path, windows: list[int], workers: int) -> dict:
    """performance -> (times, down) from the alignment of the given windows."""
    chunks = [windows[k:k + 400] for k in range(0, len(windows), 400)]
    jobs = [(str(split_dir), str(vocab_path), c) for c in chunks]
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_grids, jobs))
    else:
        results = [_grids(j) for j in jobs]
    by = defaultdict(list)
    for res in results:
        for perf, bar_start, t0, g in res:
            by[perf].append((bar_start, t0, g))
    return {p: merge_performance(ws) for p, ws in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--annotations", default="data/raw/asap_bd/asap_annotations.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", action="store_true",
                    help="also time the ASAP performances by alignment and compare with their annotations")
    args = ap.parse_args()
    ann = json.loads(Path(args.annotations).read_text()) if Path(args.annotations).exists() else {}
    if not ann:
        print(f"note: {args.annotations} not found: every performance is timed by alignment")
    vocab_path = D / "vocab.json"
    for split in args.splits.split(","):
        split_dir = D / split
        if not (split_dir / "index.npy").exists():
            print(f"skip {split}: {split_dir} missing")
            continue
        store = SegmentStore(split_dir)
        if len(store) == 0:
            print(f"skip {split}: empty")
            continue
        by_perf = defaultdict(list)
        for i, m in enumerate(store.meta):
            by_perf[m["performance"]].append(i)
        beats, source = {}, Counter()
        todo = []
        for p, idx in by_perf.items():
            if p in ann and len(ann[p].get("performance_beats", [])) >= 2:
                beats[p] = annotated(ann[p])
                source["asap annotations"] += 1
            else:
                todo += idx
        if todo:
            al = aligned_beats(split_dir, vocab_path, todo, args.workers)
            for p, b in al.items():
                beats[p] = b
                source["score alignment"] += 1
        report = {"split": split, "performances": len(beats), "source": dict(source),
                  "beats": int(sum(len(b[0]) for b in beats.values())),
                  "empty": sum(len(b[0]) < 2 for b in beats.values())}
        if args.check:
            asap = [i for p, idx in by_perf.items() if p in ann for i in idx]
            al = aligned_beats(split_dir, vocab_path, asap, args.workers) if asap else {}
            fb, fd = [], []
            for p, (t, d) in al.items():
                at, ad = annotated(ann[p])
                if len(t) < 2:
                    continue
                lo, hi = max(at[0], t[0]), min(at[-1], t[-1])
                a_in, t_in = (at >= lo) & (at <= hi), (t >= lo) & (t <= hi)
                fb.append(f_measure(at[a_in], t[t_in]))
                fd.append(f_measure(at[a_in & ad], t[t_in & d]))
            if fb:
                report["alignment_vs_annotations"] = {
                    "performances": len(fb), "beat_f_median": float(np.median(fb)), "beat_f_mean": float(np.mean(fb)),
                    "downbeat_f_median": float(np.median(fd)), "downbeat_f_mean": float(np.mean(fd))}
        save_beats(beats_path(split_dir, "gt"), beats)
        (split_dir / "beats_gt.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report))


if __name__ == "__main__":
    main()
