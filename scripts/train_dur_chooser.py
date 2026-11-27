"""Train the duration chooser (src/dur_chooser.py) for a labeller, and measure it on the test windows.

    python scripts/train_dur_chooser.py --run nl_gq
    python scripts/train_dur_chooser.py --run nl_gq --max-windows 2000 --eval-windows 500   # quick
    -> outputs/runs/<run>/dur_chooser.pt (what V2Pipeline(dur_chooser=True) loads) and dur_chooser.json

Data: the labeller on the validation windows (val: ASAP + PERiScoPe, val_transcribed: real ASAP
transcriptions, val_gangqinpu: pop), none of which it trained on, on the true grid with the beat
tracker's timing errors on ``--grid-noise`` of the windows (as the labeller was trained and
selected). Each note whose true length is among the dur head's top 4 is one example; the
performances are split 85 / 15 into the chooser's training and selection sets, so a performance
never sits in both.

Measured, as scripts/headroom.py labeller (true grid, no noise): on each test set, the windows
rendered with the labeller's likeliest length against the chooser's pick, with the rendered
duration accuracy, note F1 and onset F1 (src/eval/metrics.py). The whole-piece benchmark is the
system's _dc suffix (scripts/system_scores.py --dur-chooser; pace/v2_jobs.sh option dc).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TRAIN_SPLITS = ("val", "val_transcribed", "val_gangqinpu")
DUR_COL = 4


def _held_out(perf: str) -> bool:
    return int(hashlib.md5(str(perf).encode()).hexdigest()[:8], 16) % 100 < 15


def collect(model, ds, idx, coll, device, amp, k, log=print) -> dict:
    """Candidates and features of every note of the items ``idx`` whose true length is among the
    top k: {"X" [n, k, F], "y" [n], "held" [n] (selection set), "notes", "in_topk"}."""
    import torch

    from src.dataset import ROLE_WINDOW
    from src.dur_chooser import features
    from src.labeller import performed, placed_notes
    from src.notelabels import score_notes
    Xs, ys, held = [], [], []
    n_notes = n_in = 0
    order = list(idx)
    t0 = time.time()
    for lo in range(0, len(order), 32):
        sel = order[lo:lo + 32]
        items = [ds[j] for j in sel]
        b = coll([dict(it) for it in items])
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(b["notes"].to(device), b["note_mask"].to(device))
        out = {h: v.float() for h, v in out.items()}
        for r, (j, it) in enumerate(zip(sel, items)):
            gt = ds.vocab.decode(it["target"][it["prompt_len"]:-1].tolist())
            win = score_notes(gt)[1]
            own = (it["roles"] <= ROLE_WINDOW).numpy()
            pl = placed_notes(out, r, it["notes"][:, 0].numpy(), it["bar_est"].numpy(), own, ds.label_space)
            n = min(len(own), out["dur"].shape[1])
            onset, dur = performed(model.cfg, it["notes"])
            use, cls, X = features(pl, win, out["dur"][r, :n].cpu().numpy(), onset[:n], dur[:n], ds.label_space, k)
            if not len(use):
                continue
            y_true = it["labels"][:, DUR_COL].numpy()
            src = np.array([pl[u].src for u in use])
            yt = y_true[src]
            ok = yt >= 0
            hit = (cls == yt[:, None]) & ok[:, None]
            n_notes += int(ok.sum())
            n_in += int(hit.any(1).sum())
            m = hit.any(1)
            if m.any():
                Xs.append(X[m])
                ys.append(hit[m].argmax(1))
                perf = ds.store.meta[j].get("performance", j)
                held.append(np.full(int(m.sum()), _held_out(perf)))
        if lo // 32 % 50 == 0:
            log(f"  {lo + len(sel)}/{len(order)} windows, {n_in} examples ({time.time() - t0:.0f} s)")
    F = Xs[0].shape[-1] if Xs else 0
    return {"X": np.concatenate(Xs) if Xs else np.zeros((0, k, F), np.float32),
            "y": np.concatenate(ys) if ys else np.zeros(0, int),
            "held": np.concatenate(held) if held else np.zeros(0, bool), "notes": n_notes, "in_topk": n_in}


def evaluate(model, chooser, ds, idx, coll, device, amp) -> dict:
    """The windows rendered with the likeliest length and with the chooser's (true grid)."""
    import torch

    from src.dataset import ROLE_WINDOW
    from src.eval.metrics import Accumulator
    from src.labeller import performed, placed_notes
    from src.notelabels import render, score_notes
    acc = {"argmax": Accumulator(), "chooser": Accumulator()}
    changed = total = 0
    for lo in range(0, len(idx), 32):
        sel = idx[lo:lo + 32]
        items = [ds[j] for j in sel]
        b = coll([dict(it) for it in items])
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(b["notes"].to(device), b["note_mask"].to(device))
        out = {h: v.float() for h, v in out.items()}
        for r, it in enumerate(items):
            gt = ds.vocab.decode(it["target"][it["prompt_len"]:-1].tolist())
            win = score_notes(gt)[1]
            own = (it["roles"] <= ROLE_WINDOW).numpy()
            args = (out, r, it["notes"][:, 0].numpy(), it["bar_est"].numpy(), own, ds.label_space)
            base = placed_notes(*args)
            mine = placed_notes(*args, dur_chooser=chooser, frame=win, perf=performed(model.cfg, it["notes"]))
            changed += sum(a.total != c.total for a, c in zip(base, mine))
            total += len(base)
            acc["argmax"].add(gt, render(win, base))
            acc["chooser"].add(gt, render(win, mine))
    keys = ("note_f1", "onset_f1", "duration_acc", "staff_acc")
    res = {m: {k: a.result()[k] for k in keys} for m, a in acc.items()}
    res["changed"] = changed / max(total, 1)
    res["windows"] = len(idx)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="nl_gq")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--splits", default=",".join(TRAIN_SPLITS))
    ap.add_argument("--conditions", default="clean,real,periscope,gangqinpu", help="test sets to measure on")
    ap.add_argument("--max-windows", type=int, default=0, help="per training split (0 = all)")
    ap.add_argument("--eval-windows", type=int, default=3000, help="per test set (0 = all)")
    ap.add_argument("--grid-noise", type=float, default=0.8)
    ap.add_argument("--beat-noise", default="configs/beat_noise.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    import torch

    from scripts.eval_labeller import SPLITS
    from scripts.train import TruncatingCollate, pick_device
    from src.dataset import ScoreSegmentDataset
    from src.dur_chooser import FEATURES, FILE, K, train_chooser
    from src.labeller import load_labeller
    from src.notelabels import LABEL_FILE, GridNoise
    device = pick_device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    run = Path(args.runs_dir) / args.run
    model = load_labeller(run / "best.pt", device)
    cfg = model.cfg
    ctx = {"context_before": cfg.context_before, "context_after": cfg.context_after}
    coll = TruncatingCollate(cfg.max_notes, 10 ** 9)
    data = Path(args.data)
    rng = np.random.default_rng(args.seed)
    gnoise = (GridNoise.from_json(args.beat_noise, args.grid_noise)
              if args.grid_noise > 0 and Path(args.beat_noise).exists() else None)
    parts = []
    for split in args.splits.split(","):
        if not (data / split / LABEL_FILE).exists():
            print(f"skip {split}: {data / split / LABEL_FILE} missing")
            continue
        ds = ScoreSegmentDataset(str(data / split), str(data / "vocab.json"), notelabels=True, **ctx)
        ds.grid_noise = gnoise
        idx = list(range(len(ds)))
        if args.max_windows and len(idx) > args.max_windows:
            idx = sorted(rng.choice(len(idx), args.max_windows, replace=False).tolist())
        print(f"{split}: {len(idx)} windows", flush=True)
        c = collect(model, ds, idx, coll, device, amp, K)
        print(f"{split}: {c['in_topk']} of {c['notes']} notes with the true length in the top {K}", flush=True)
        parts.append((split, c))
    if not parts:
        raise SystemExit("no training split found (scripts/build_notelabels.py)")
    X = np.concatenate([c["X"] for _, c in parts])
    y = np.concatenate([c["y"] for _, c in parts])
    held = np.concatenate([c["held"] for _, c in parts])
    print(f"examples: {len(y)} ({int((~held).sum())} train, {int(held.sum())} selection)", flush=True)
    chooser, fit = train_chooser(X[~held], y[~held], X[held], y[held], epochs=args.epochs, seed=args.seed,
                                 device=device)
    report = {"run": args.run, "k": K, "features": list(FEATURES), "fit": fit,
              "training": {s: {"notes": c["notes"], "in_topk": c["in_topk"]} for s, c in parts},
              "grid_noise": args.grid_noise if gnoise else 0.0, "test": {}}
    chooser.save(run / FILE, meta={"fit": fit, "splits": [s for s, _ in parts]})
    chooser = chooser.to(device).eval()
    for cond in args.conditions.split(","):
        split = SPLITS[cond]
        if not (data / split / LABEL_FILE).exists():
            print(f"skip {cond}: {data / split / LABEL_FILE} missing")
            continue
        ds = ScoreSegmentDataset(str(data / split), str(data / "vocab.json"), notelabels=True, **ctx)
        idx = list(range(len(ds)))
        if args.eval_windows and len(idx) > args.eval_windows:
            idx = sorted(np.random.default_rng(0).choice(len(idx), args.eval_windows, replace=False).tolist())
        r = evaluate(model, chooser, ds, idx, coll, device, amp)
        report["test"][cond] = r
        print(f"{cond} ({r['windows']} windows, {r['changed']:.1%} of lengths changed): duration "
              f"{r['argmax']['duration_acc']:.3f} -> {r['chooser']['duration_acc']:.3f}, note F1 "
              f"{r['argmax']['note_f1']:.3f} -> {r['chooser']['note_f1']:.3f}", flush=True)
    (run / "dur_chooser.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {run / FILE} and {run / 'dur_chooser.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
