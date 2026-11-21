"""Evaluate a note labeller (v2) on the test sets with the same metrics as every other system.

    python scripts/eval_labeller.py --run nl_base                   # every test set
    python scripts/eval_labeller.py --run nl_base --conditions real --limit-perfs 3

Each test window is labelled in one pass (src/labeller.py) and rendered back to tokens
(src/notelabels.py render), then scored against its target with src/eval/metrics.py: the numbers
compare directly with scripts/evaluate_pieces.py (the seq2seq models) and scripts/run_baselines.py.

Grid (where each note sits in the bars, the labeller's input): ``true`` = the score's beats timed
through the note alignment (Phase 1: the labeller given the right grid, as the beat-snap baselines
are given the annotated beats). The result is stored as mode ``grid_true`` in the run's
eval_pieces_<ckpt>.json, next to the other systems' modes, so scripts/compare_runs.py tables it;
per-head accuracies are kept with it. Conditions: clean (ASAP test, clean MIDI), synthetic (the
same with the fixed synthetic noise draw), real (Transkun transcriptions), periscope; a split
without notelabels.npz (scripts/build_notelabels.py) is skipped with the reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_pieces import merge_results  # noqa: E402
from scripts.train import TruncatingCollate, lower_priority, pick_device  # noqa: E402
from src.dataset import ScoreSegmentDataset  # noqa: E402
from src.eval.metrics import Accumulator  # noqa: E402
from src.eval.plan import stamp  # noqa: E402
from src.labeller import head_counts, load_labeller, predict_windows  # noqa: E402
from src.notelabels import LABEL_FILE, score_notes  # noqa: E402
from src.preprocessing.noise import NoiseModel  # noqa: E402

SPLITS = {"clean": "test", "synthetic": "test", "real": "test_transcribed", "periscope": "test_periscope"}
GRIDS = ("true",)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--conditions", default="clean,synthetic,real,periscope")
    ap.add_argument("--grid", default="true", choices=list(GRIDS))
    ap.add_argument("--limit-perfs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--normal-priority", action="store_true")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--runs-dir", default="outputs/runs")
    args = ap.parse_args(argv)
    if not args.normal_priority:
        lower_priority()
    device = pick_device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    run, data = Path(args.runs_dir) / args.run, Path(args.data)
    model = load_labeller(run / args.ckpt, device)
    cfg = model.cfg
    ctx = {"context_before": cfg.context_before, "context_after": cfg.context_after}
    coll = TruncatingCollate(cfg.max_notes, 10 ** 9)
    mode = f"grid_{args.grid}"
    out_file = run / f"eval_pieces_{Path(args.ckpt).stem}.json"
    nm = NoiseModel.from_json("configs/noise_stats.json")
    nm.cfg.clean_prob = 0.0
    for cond in args.conditions.split(","):
        split = SPLITS[cond]
        if not (data / split / "index.npy").exists():
            print(f"skip {cond}: {data / split} missing")
            continue
        if not (data / split / LABEL_FILE).exists():
            print(f"skip {cond}: {data / split / LABEL_FILE} missing (scripts/build_notelabels.py)")
            continue
        kw = dict(noise=nm, seed=12345, prompt_dropout=0.0) if cond == "synthetic" else {}
        ds = ScoreSegmentDataset(str(data / split), str(data / "vocab.json"), notelabels=True, **ctx, **kw)
        perfs = defaultdict(list)
        for i, m in enumerate(ds.store.meta):
            perfs[m["performance"]].append(i)
        names = sorted(perfs)
        if args.limit_perfs and len(names) > args.limit_perfs:
            names = sorted(np.random.default_rng(0).choice(names, args.limit_perfs, replace=False).tolist())
        windows = [i for p in names for i in sorted(perfs[p], key=lambda i: ds.store.meta[i]["win"])]
        items = [ds[i] for i in windows]
        gts = [ds.vocab.decode(it["target"][it["prompt_len"]:-1].tolist()) for it in items]
        preds = predict_windows(model, items, [score_notes(g)[1] for g in gts], coll, device, amp, ds.label_space,
                                args.batch_size)
        acc, acc_unseen = Accumulator(), Accumulator()
        for i, g, p in zip(windows, gts, preds):
            acc.add(g, p)
            if cond == "real" and ds.store.meta[i].get("maestro_split") != "train":
                acc_unseen.add(g, p)
        counts = {}
        with torch.no_grad():
            for lo in range(0, len(items), args.batch_size):
                b = coll([dict(it) for it in items[lo:lo + args.batch_size]])
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    o = model(b["notes"].to(device), b["note_mask"].to(device))
                for k, v in head_counts({k: v.float().cpu() for k, v in o.items()}, b["labels"]).items():
                    counts[k] = counts.get(k, 0) + v
        res = stamp(acc.result(), None, args.limit_perfs)
        res.update({f"head_{h}_acc": counts[f"{h}_ok"] / max(counts[f"{h}_n"], 1)
                    for h in ("keep", "delta", "pos", "sub", "dur", "staff")})
        new = {cond: {mode: res}}
        if cond == "real":
            new["real_unseen_audio"] = {mode: stamp(acc_unseen.result(), None, args.limit_perfs)}
        merge_results(out_file, new)
        with open(run / f"pred_pieces_{Path(args.ckpt).stem}_{cond}.jsonl", "w") as f:
            for i, p in zip(windows, preds):
                m = ds.store.meta[i]
                f.write(json.dumps({"performance": m["performance"], "win": m["win"], "mode": mode, "pred": p}) + "\n")
        print(f"{cond} ({len(windows)} windows, grid {args.grid}): onset F1 {res['onset_f1']:.3f}, onset-shift "
              f"{res['onset_shift_f1']:.3f}, duration {res['duration_acc']:.3f}, staff {res['staff_acc']:.3f}, bar F1 "
              f"{res['bar_f1']:.3f}, note F1 {res['note_f1']:.3f} | heads: keep {res['head_keep_acc']:.3f} pos "
              f"{res['head_pos_acc']:.3f} sub {res['head_sub_acc']:.3f} dur {res['head_dur_acc']:.3f} staff "
              f"{res['head_staff_acc']:.3f}",
              flush=True)
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
