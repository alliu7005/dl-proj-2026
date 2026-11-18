"""Why does chained decoding lose notes? Diagnose from saved predictions (CPU only, no model).

    python scripts/analyze_chained.py --runs s2s,s2s_ctx,sf_small_ctx --conditions synthetic,clean

Rebuilds, for every window, the context chained decoding gave it (segment.prompt_from_output of
the model's own output so far, as in evaluate_pieces.py), and relates each window's pitch F1 to:
  * the context's length, versus the lengths of the true contexts (what training saw);
  * whether the previous window's output was degenerate (far longer than its target, or no notes);
  * the same window decoded with no context ("none" mode), when it was evaluated.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_pieces import SPLITS, stitch  # noqa: E402
from src.preprocessing.segment import prompt_from_output  # noqa: E402
from src.preprocessing.store import SegmentStore  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402

D = Path("data/processed")
warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean of empty buckets


def pitch_f1(gt: list[str], pred: list[str]) -> float:
    a = Counter(t for t in gt if t.startswith("p_"))
    b = Counter(t for t in pred if t.startswith("p_"))
    tp = sum((a & b).values())
    n = sum(a.values()) + sum(b.values())
    return 2 * tp / n if n else 1.0


def analyze(run: str, cond: str, ckpt: str, vocab: Vocab):
    f = Path("outputs/runs") / run / f"pred_pieces_{ckpt}_{cond}.jsonl"
    if not f.exists():
        print(f"{run}/{cond}: {f} missing")
        return
    preds = defaultdict(dict)
    for line in open(f):
        r = json.loads(line)
        preds[r["mode"]][(r["performance"], r["win"])] = r["pred"]
    store = SegmentStore(D / SPLITS[cond])
    idx = {(m["performance"], m["win"]): i for i, m in enumerate(store.meta)}
    true_ctx = np.array([len(store.get_prompt(i)) for i in range(len(store))])
    p99, pmax = np.percentile(true_ctx[true_ctx > 0], 99), true_ctx.max()

    by_perf = defaultdict(list)
    for (p, w) in preds["chained"]:
        by_perf[p].append(w)
    rows = []  # ctx_len, f1_chained, f1_none, prev_degenerate, degenerate
    for p, wins in by_perf.items():
        wins.sort()
        hist, prev_i, prev_bad = [], None, False
        for w in wins:
            i = idx[(p, w)]
            m = store.meta[i]
            contiguous = (prev_i is not None and store.meta[prev_i].get("s_hi") is not None
                          and store.meta[prev_i]["s_hi"] == m.get("s_lo"))
            ctx = prompt_from_output(hist) if contiguous else []
            gt = vocab.decode(store.get(i)[2].tolist())
            out = preds["chained"][(p, w)]
            n_gt = max(1, len(gt))
            bad = len(out) > 3 * n_gt or not any(t.startswith("p_") for t in out)
            f_none = pitch_f1(gt, preds["none"][(p, w)]) if (p, w) in preds.get("none", {}) else np.nan
            rows.append((len(ctx), pitch_f1(gt, out), f_none, prev_bad and contiguous, bad))
            hist, prev_i, prev_bad = stitch(hist, out), i, bad
    a = np.array(rows, dtype=float)
    L, fc, fn, pb, bad = a.T
    print(f"\n{run} / {cond}: {len(a)} windows | pitch F1 chained {fc.mean():.3f}, none {np.nanmean(fn):.3f}")
    print(f"  true context length: 99th pct {p99:.0f}, max {pmax} tokens")
    buckets = [("no context", L == 0), (f"<= {p99:.0f} (normal)", (L > 0) & (L <= p99)),
               (f"{p99:.0f}-{pmax} (rare)", (L > p99) & (L <= pmax)), (f"> {pmax} (never in training)", L > pmax)]
    for name, sel in buckets:
        if sel.any():
            print(f"  context {name:28s} {sel.mean():6.1%} of windows  pitch F1 chained {fc[sel].mean():.3f}"
                  f"  none {np.nanmean(fn[sel]):.3f}")
    print(f"  degenerate outputs (>3x target length or no notes): {bad.mean():.1%} of windows")
    if pb.any():
        print(f"  after a degenerate window: pitch F1 {fc[pb == 1].mean():.3f} ({pb.mean():.1%} of windows)"
              f"  vs otherwise {fc[pb == 0].mean():.3f}")
    worse = (fn - fc) > 0.2
    print(f"  windows where 'none' beats chained by > 0.2 F1: {np.nanmean(worse):.1%}"
          f" (their mean context length {L[worse].mean() if worse.any() else 0:.0f} tokens vs {L.mean():.0f} overall)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="s2s,s2s_ctx,sf_small,sf_small_ctx,sf_base,sf_base_reg")
    ap.add_argument("--conditions", default="synthetic,clean")
    ap.add_argument("--ckpt", default="best")
    args = ap.parse_args()
    vocab = Vocab.load(str(D / "vocab.json"))
    for run in args.runs.split(","):
        for cond in args.conditions.split(","):
            analyze(run, cond, args.ckpt, vocab)


if __name__ == "__main__":
    main()
