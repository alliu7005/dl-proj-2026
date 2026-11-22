"""Evaluate a note labeller (v2) on the test sets with the same metrics as every other system.

    python scripts/eval_labeller.py --run nl_base                   # every test set
    python scripts/eval_labeller.py --run nl_base --conditions real --limit-perfs 3

Each test window is labelled in one pass (src/labeller.py) and rendered back to tokens
(src/notelabels.py render), then scored against its target with src/eval/metrics.py: the numbers
compare directly with scripts/evaluate_pieces.py (the seq2seq models) and scripts/run_baselines.py.

Grid (where each note sits in the bars, the labeller's input): ``true`` = the score's beats timed
through the note alignment (Phase 1: the labeller given the right grid, as the beat-snap baselines
are given the annotated beats); ``meter:<run>`` = the grid a meter model decoded from the tracked
beats (Phase 3, the real system: scripts/eval_meter.py --run <run> saves it per performance; clean
and synthetic use its grids of the clean test performances with the simulated tracker, real its
grids of the transcriptions with Beat This!'s beats, periscope its grids of the PERiScoPe test
transcriptions with the simulated tracker, one per segment: a window takes the one whose span
holds it), each window's bars, time signatures and key signature taken from it (src/notelabels.py window_frame), so nothing comes from the score. The
result is stored as mode ``grid_true`` / ``grid_meter_<run>`` in the run's
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
from src.notelabels import LABEL_FILE, score_notes, window_frame  # noqa: E402
from src.preprocessing.noise import NoiseModel  # noqa: E402

SPLITS = {"clean": "test", "synthetic": "test", "real": "test_transcribed", "periscope": "test_periscope"}
METER_GRIDS = {"clean": "clean", "synthetic": "clean", "real": "real", "periscope": "periscope"}  # -> eval_meter.py's grids


def grid_for(cands: list, t: float) -> dict:
    """The grid of a performance (one per segment) whose span holds time ``t``, else the nearest."""
    def gap(g):
        lo, hi = g.get("span") or (-np.inf, np.inf)
        return 0.0 if lo <= t <= hi else min(abs(t - lo), abs(t - hi))
    return min(cands, key=gap)


def merge_predictions(path: Path, mode: str, rows: list) -> None:
    """Replace one mode's lines of a condition's prediction file (the other modes stay: the
    labeller on other grids), under a lock as the results (scripts/evaluate_pieces.py
    merge_results): evaluations on other grids may be writing the same file."""
    import os
    with open(path.with_suffix(".lock"), "w") as lock:
        try:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        except ImportError:  # Windows: one evaluation at a time
            pass
        old = [ln for ln in open(path) if ln.strip()] if path.exists() else []
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            f.writelines(ln for ln in old if json.loads(ln).get("mode") != mode)
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)


def grid_mode(grid: str) -> str:
    """--grid -> the result's mode name: true -> grid_true, meter:<run> -> grid_meter_<run>."""
    if grid == "true":
        return "grid_true"
    if grid.startswith("meter:") and grid[6:]:
        return "grid_meter_" + grid[6:]
    raise SystemExit(f"--grid must be 'true' or 'meter:<run>', not {grid!r}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--conditions", default="clean,synthetic,real,periscope")
    ap.add_argument("--grid", default="true", help="true, or meter:<run> (a meter model's decoded grids)")
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
    mode = grid_mode(args.grid)
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
        grids = None
        if args.grid != "true":
            if cond not in METER_GRIDS:
                print(f"skip {cond}: the meter model's grids cover {', '.join(METER_GRIDS)} only")
                continue
            gfile = Path(args.runs_dir) / args.grid[6:] / f"grids_{METER_GRIDS[cond]}.pkl"
            if not gfile.exists():
                print(f"skip {cond}: {gfile} missing (scripts/eval_meter.py --run {args.grid[6:]})")
                continue
            import pickle
            grids = defaultdict(list)
            for key, g in pickle.load(open(gfile, "rb")).items():
                grids[g.get("performance", key)].append(g)
        perfs = defaultdict(list)
        for i, m in enumerate(ds.store.meta):
            perfs[m["performance"]].append(i)
        names = sorted(perfs)
        if args.limit_perfs and len(names) > args.limit_perfs:
            names = sorted(np.random.default_rng(0).choice(names, args.limit_perfs, replace=False).tolist())
        if grids is not None:
            names = [p for p in names if p in grids]
        windows = [i for p in names for i in sorted(perfs[p], key=lambda i: ds.store.meta[i]["win"])]
        frames = {}
        if grids is not None:  # each window's grid, bars and key from the decoded grid, nothing from the score
            for i in windows:
                m = ds.store.meta[i]
                on = ds.store.get(i)[0][:, 1]
                t_mid = float(m["perf_t0"]) + (float(np.median(on[on >= 0])) if (on >= 0).any() else 0.0)
                g = grid_for(grids[m["performance"]], t_mid)
                frames[i] = window_frame(g, float(m["perf_t0"]), on, int(g.get("key", 0)))
            ds.grid_override = frames
        items = [ds[i] for i in windows]
        gts = [ds.vocab.decode(it["target"][it["prompt_len"]:-1].tolist()) for it in items]
        wins = [frames[i][2] for i in windows] if frames else [score_notes(g)[1] for g in gts]
        preds = predict_windows(model, items, wins, coll, device, amp, ds.label_space, args.batch_size)
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
        merge_predictions(run / f"pred_pieces_{Path(args.ckpt).stem}_{cond}.jsonl", mode,
                          [{"performance": ds.store.meta[i]["performance"], "win": ds.store.meta[i]["win"],
                            "mode": mode, "pred": p} for i, p in zip(windows, preds)])
        print(f"{cond} ({len(windows)} windows, grid {args.grid}): onset F1 {res['onset_f1']:.3f}, onset-shift "
              f"{res['onset_shift_f1']:.3f}, duration {res['duration_acc']:.3f}, staff {res['staff_acc']:.3f}, bar F1 "
              f"{res['bar_f1']:.3f}, note F1 {res['note_f1']:.3f} | heads: keep {res['head_keep_acc']:.3f} pos "
              f"{res['head_pos_acc']:.3f} sub {res['head_sub_acc']:.3f} dur {res['head_dur_acc']:.3f} staff "
              f"{res['head_staff_acc']:.3f}",
              flush=True)
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
