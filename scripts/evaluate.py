"""Evaluate a trained checkpoint on the test sets, with the same metrics and windows as the baselines.

    python scripts/evaluate.py --run base                       # best.pt of outputs/runs/base
    python scripts/evaluate.py --run base --limit 100           # quick check on 100 windows per set

Conditions (same windows / targets as scripts/run_baselines.py):
  clean          test (Beyer & Dai's ASAP test set), clean MIDI
  synthetic      test + our fitted noise (fixed seed)
  real           test_transcribed: Transkun transcriptions (+ real_unseen_audio subset)
  real_kong      test_transcribed_kong: Kong transcriptions (robustness to another transcriber)
  periscope      test_periscope: real Transkun transcriptions of unseen PERiScoPe works

Prompts are the ground-truth score context before each window ("oracle prompt"), the
window-level protocol the baselines use too. Writes outputs/runs/<run>/eval_<ckpt>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import SHOW, TruncatingCollate, lower_priority, pick_device  # noqa: E402
from src.dataset import ScoreSegmentDataset  # noqa: E402
from src.eval.metrics import Accumulator  # noqa: E402
from src.model import ModelConfig, Score2Seq  # noqa: E402
from src.preprocessing.noise import NoiseModel  # noqa: E402
from src.preprocessing.vocab import EOS  # noqa: E402

D = Path("data/processed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--limit", type=int, default=0, help="windows per condition (0 = all)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--conditions", default="clean,synthetic,real,real_kong,periscope")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--normal-priority", action="store_true")
    args = ap.parse_args()
    if not args.normal_priority:
        lower_priority()
    device = pick_device(args.device)
    amp = device.type == "cuda"
    run = Path("outputs/runs") / args.run
    ck = torch.load(run / args.ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = Score2Seq(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    vocab_path = str(D / "vocab.json")
    coll = TruncatingCollate(cfg.max_notes, cfg.max_tokens)
    nm = NoiseModel.from_json("configs/noise_stats.json")
    nm.cfg.clean_prob = 0.0

    sets = {}
    for c in args.conditions.split(","):
        split = {"clean": "test", "synthetic": "test", "real": "test_transcribed",
                 "real_kong": "test_transcribed_kong", "periscope": "test_periscope"}[c]
        if not (D / split).exists():
            print(f"skip {c}: {D / split} missing")
            continue
        if c == "synthetic":  # fixed, reproducible noise draw; no augmentation, no prompt dropout
            sets[c] = ScoreSegmentDataset(str(D / split), vocab_path, noise=nm, seed=12345, prompt_dropout=0.0)
        else:
            sets[c] = ScoreSegmentDataset(str(D / split), vocab_path)

    results = {}
    rng = np.random.default_rng(0)
    for c, ds in sets.items():
        vocab = ds.vocab
        eos = vocab.stoi[EOS]
        idx = list(range(len(ds)))
        if args.limit and len(idx) > args.limit:
            idx = sorted(rng.choice(idx, size=args.limit, replace=False).tolist())
        idx.sort(key=lambda i: int(ds.store.index[i, 1] - ds.store.index[i, 0]))  # similar lengths per batch
        acc, acc_unseen = Accumulator(), Accumulator()
        preds_out = []
        for k in range(0, len(idx), args.batch_size):
            items = [ds[i] for i in idx[k:k + args.batch_size]]
            b = coll(items)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                preds = model.generate(b["notes"].to(device), b["note_mask"].to(device),
                                       [it["target"][:it["prompt_len"]] for it in items], eos)
            for i, it, p in zip(idx[k:k + args.batch_size], items, preds):
                gt = vocab.decode(it["target"][it["prompt_len"]:-1].tolist())
                pr = vocab.decode(p)
                acc.add(gt, pr)
                meta = ds.store.meta[i]
                if c == "real" and meta.get("maestro_split") != "train":
                    acc_unseen.add(gt, pr)
                preds_out.append({"performance": meta["performance"], "win": meta.get("win"), "pred": pr})
            print(f"{c}: {min(k + args.batch_size, len(idx))}/{len(idx)}", end="\r", flush=True)
        results[c] = acc.result()
        if c == "real":
            results["real_unseen_audio"] = acc_unseen.result()
        with open(run / f"pred_{Path(args.ckpt).stem}_{c}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in preds_out)
        print(f"{c}: done ({len(idx)} windows)            ", flush=True)

    out = run / f"eval_{Path(args.ckpt).stem}.json"
    json.dump(results, open(out, "w"), indent=1)
    conds = list(results)
    print(f"\n{'metric':16s}" + "".join(f"{c:>18s}" for c in conds))
    for key in SHOW + ("ped_f1",):
        print(f"{key:16s}" + "".join(f"{results[c][key]:18.3f}" for c in conds))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
