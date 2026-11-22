"""Add metrics introduced after a run was evaluated, from its saved predictions (CPU, no model).

    python scripts/rescore.py                         # every run in outputs/runs
    python scripts/rescore.py --runs nl_base,sf_base_reg_cab --conditions real
    python scripts/rescore.py --workers 8             # one prediction file per process

Every evaluation saves its window outputs (outputs/runs/<run>/pred_pieces_<ckpt>_<condition>.jsonl:
performance, window, mode, tokens; scripts/evaluate_pieces.py, scripts/eval_labeller.py). This
re-scores them against the windows' targets with the current src/eval/metrics.py and adds the
metrics a run's results lack (``--keys``, default the ones added since: onset_scaled_*), so every
system gets them without decoding again. A mode is only updated when the re-scored onset F1
equals the stored one (the same windows, the same targets): otherwise it is reported and left
alone. real_unseen_audio (the real windows whose audio Transkun did not train on) is re-scored
with real.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_pieces import merge_results  # noqa: E402
from src.eval.metrics import Accumulator  # noqa: E402
from src.eval.plan import SPLITS  # noqa: E402
from src.preprocessing.store import SegmentStore  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402

NEW_KEYS = ("onset_scaled_p", "onset_scaled_r", "onset_scaled_f1")
TOL = 1e-9


def rescore_file(pred_file: Path, store: SegmentStore, vocab: Vocab, unseen: bool) -> dict:
    """{mode: (result, unseen result or None, windows not in the split)} of one condition's
    prediction file."""
    index = {(m["performance"], m["win"]): i for i, m in enumerate(store.meta)}
    acc, acc_u = defaultdict(Accumulator), defaultdict(Accumulator)
    missing = defaultdict(int)
    with open(pred_file) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            i = index.get((r["performance"], r["win"]))
            if i is None:
                missing[r["mode"]] += 1
                continue
            gt = vocab.decode(store.get(i)[2].tolist())
            acc[r["mode"]].add(gt, r["pred"])
            if unseen and store.meta[i].get("maestro_split") != "train":
                acc_u[r["mode"]].add(gt, r["pred"])
    out = {}
    for mode, a in acc.items():
        out[mode] = (a.result(), acc_u[mode].result() if unseen and mode in acc_u else None, missing[mode])
    return out


_STORES: dict = {}  # per process: each split's store and the vocabulary, loaded once


def _task(job):
    """One prediction file re-scored (in a worker process with --workers)."""
    pred_file, split, vocab_path, unseen = job
    if split not in _STORES:
        _STORES[split] = SegmentStore(split)
    if "vocab" not in _STORES:
        _STORES["vocab"] = Vocab.load(vocab_path)
    return rescore_file(Path(pred_file), _STORES[split], _STORES["vocab"], unseen)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="comma-separated run names (default: every run with predictions)")
    ap.add_argument("--conditions", default="")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--keys", default=",".join(NEW_KEYS))
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--workers", type=int, default=1, help="prediction files re-scored in parallel")
    args = ap.parse_args(argv)
    data, runs_dir = Path(args.data), Path(args.runs_dir)
    keys = [k for k in args.keys.split(",") if k]
    runs = [r for r in args.runs.split(",") if r] or sorted(
        {p.parent.name for p in runs_dir.glob(f"*/pred_pieces_{args.ckpt}_*.jsonl")})
    conds = [c for c in args.conditions.split(",") if c]
    jobs, stored_of = [], {}
    for run in runs:
        res_file = runs_dir / run / f"eval_pieces_{args.ckpt}.json"
        if not res_file.exists():
            print(f"{run}: no {res_file.name}, skipped")
            continue
        stored_of[run] = json.loads(res_file.read_text())
        for pred_file in sorted((runs_dir / run).glob(f"pred_pieces_{args.ckpt}_*.jsonl")):
            cond = pred_file.stem[len(f"pred_pieces_{args.ckpt}_"):]
            if (conds and cond not in conds) or cond not in SPLITS or cond not in stored_of[run]:
                continue
            split = data / SPLITS[cond]
            if not (split / "index.npy").exists():
                print(f"{run}/{cond}: {split} missing, skipped")
                continue
            jobs.append((run, cond, (str(pred_file), split, str(data / "vocab.json"), cond == "real")))
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(_task, [j[2] for j in jobs]))
    else:
        results = [_task(j[2]) for j in jobs]
    for (run, cond, (_, split, _, _)), scored in zip(jobs, results):
        stored = stored_of[run]
        new, report = {}, []
        for mode, (res, unseen, missing) in scored.items():
            old = stored[cond].get(mode)
            if not isinstance(old, dict):
                continue
            if missing or abs(res["onset_f1"] - old.get("onset_f1", float("nan"))) > TOL:
                report.append(f"{mode}: NOT updated (onset F1 {res['onset_f1']:.4f} re-scored vs "
                              f"{old.get('onset_f1', float('nan')):.4f} stored, {missing} windows not in {split.name})")
                continue
            new.setdefault(cond, {})[mode] = {**old, **{k: res[k] for k in keys}}
            if unseen is not None and isinstance(stored.get("real_unseen_audio", {}).get(mode), dict):
                new.setdefault("real_unseen_audio", {})[mode] = {**stored["real_unseen_audio"][mode],
                                                                 **{k: unseen[k] for k in keys}}
            report.append(f"{mode} {res.get('onset_scaled_f1', float('nan')):.3f}")
        if new:
            merge_results(runs_dir / run / f"eval_pieces_{args.ckpt}.json", new)
        print(f"{run}/{cond}: " + ", ".join(report), flush=True)


if __name__ == "__main__":
    main()
