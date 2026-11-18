"""Estimate guarded decoding offline from saved chained / none predictions (CPU only, no model).

    python scripts/simulate_guard.py --runs s2s,s2s_ctx,sf_base_reg --conditions synthetic,clean

For each window, keep the chained output unless the no-prompt output agrees better with the
window's input pitches by more than a margin (evaluate_pieces.input_agreement), then score with the
standard metrics. This picks between the outputs that were actually decoded, so it cannot show the
second effect of the real guarded mode (a replaced window also gives the next window a better
context): it is a conservative estimate, used to choose --guard-margin before spending GPU time.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_pieces import SPLITS, input_agreement  # noqa: E402
from src.dataset import ScoreSegmentDataset  # noqa: E402
from src.eval.metrics import Accumulator  # noqa: E402
from src.preprocessing.noise import NoiseModel  # noqa: E402

D = Path("data/processed")
KEYS = ["note_f1", "onset_f1", "onset_shift_f1", "duration_acc", "staff_acc", "bar_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="s2s,s2s_ctx,sf_small,sf_small_ctx,sf_base,sf_base_reg")
    ap.add_argument("--conditions", default="synthetic,clean")
    ap.add_argument("--margins", default="0,0.02,0.05,0.1,0.2")
    ap.add_argument("--ckpt", default="best")
    args = ap.parse_args()
    margins = [float(m) for m in args.margins.split(",")]
    vocab_path = str(D / "vocab.json")
    nm = NoiseModel.from_json("configs/noise_stats.json")
    nm.cfg.clean_prob = 0.0
    datasets = {}
    print(f"{'run':13s}{'condition':10s}{'mode':16s}{'fallback':>9s}" + "".join(k[:9].rjust(10) for k in KEYS))
    for cond in args.conditions.split(","):
        split = SPLITS[cond]
        if cond == "synthetic":  # the same fixed noise draw as evaluate_pieces.py
            ds = ScoreSegmentDataset(str(D / split), vocab_path, noise=nm, seed=12345, prompt_dropout=0.0)
        else:
            ds = ScoreSegmentDataset(str(D / split), vocab_path)
        idx = {(m["performance"], m["win"]): i for i, m in enumerate(ds.store.meta)}
        datasets[cond] = (ds, idx)
    for run in args.runs.split(","):
        for cond in args.conditions.split(","):
            f = Path("outputs/runs") / run / f"pred_pieces_{args.ckpt}_{cond}.jsonl"
            if not f.exists():
                print(f"{run}/{cond}: {f} missing")
                continue
            ds, idx = datasets[cond]
            preds = defaultdict(dict)
            for line in open(f):
                r = json.loads(line)
                preds[r["mode"]][(r["performance"], r["win"])] = r["pred"]
            keys = [k for k in preds["chained"] if k in preds.get("none", {})]
            if not keys:
                print(f"{run}/{cond}: needs both chained and none predictions")
                continue
            rows = []  # (gt, chained, none, agree_chained, agree_none)
            for k in keys:
                item = ds[idx[k]]
                gt = ds.vocab.decode(item["target"][item["prompt_len"]:-1].tolist())
                c, n = preds["chained"][k], preds["none"][k]
                rows.append((gt, c, n, input_agreement(item["notes"], c), input_agreement(item["notes"], n)))

            def score(pick):
                acc, fb = Accumulator(), 0
                for gt, c, n, ac, an in rows:
                    use_none = pick(ac, an)
                    fb += use_none
                    acc.add(gt, n if use_none else c)
                return acc.result(), fb / len(rows)

            variants = [("chained", lambda ac, an: False), ("none", lambda ac, an: True)]
            variants += [(f"guarded m={m:g}", (lambda m: lambda ac, an: an > ac + m)(m)) for m in margins]
            for name, pick in variants:
                res, fb = score(pick)
                fbs = f"{fb:.1%}" if name.startswith("guarded") else ""
                print(f"{run:13s}{cond:10s}{name:16s}{fbs:>9s}" + "".join(f"{res[k]:10.3f}" for k in KEYS))
            print()


if __name__ == "__main__":
    main()
