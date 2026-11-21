"""Train the meter model (v2, Phase 2: src/meter.py) on whole performances.

    python scripts/build_meter_data.py                  # once: data/processed/meter/<split>.pkl
    python scripts/train_meter.py --name meter_base
    python scripts/train_meter.py --name meter_base --resume

Training: every step draws --batch performances (the clean ASAP training performances and, when
built, the real-transcription ones in their natural proportion), cuts a random --chunk of beats
from each, and for a clean one simulates the tracker on its annotated beats (configs/beat_noise.json:
octave errors, missed / extra beats, jitter, downbeat errors; a new draw every time) and, with
--note-noise, transcription noise on its notes. Validation (every --eval-every steps): the heads'
accuracies on the validation performances (fixed tracker draw) and their DECODED grids scored
against the annotations (src/meter.py grid_scores: beat F, downbeat F, time-signature accuracy),
next to the raw tracker's beat / downbeat F on the same beats. The best validation downbeat F
keeps best.pt; early stopping, resume and atomic checkpoints as in the other training scripts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import lower_priority, pick_device  # noqa: E402
from scripts.train_model import learning_rate  # noqa: E402
from src.beats import BeatNoise, f_measure  # noqa: E402
from src.meter import (DecodeConfig, MeterConfig, build_meter_model, decode, grid_from_states, grid_scores,  # noqa: E402
                       meter_loss, sample)
from src.preprocessing.noise import NoiseModel  # noqa: E402


def load(data: Path, split: str) -> list:
    p = data / "meter" / f"{split}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else []


def batch(samples: list, device) -> tuple:
    k = max(len(s[0]) for s in samples)
    x = torch.zeros(len(samples), k, samples[0][0].shape[1])
    mask = torch.zeros(len(samples), k, dtype=torch.bool)
    tg = {h: torch.full((len(samples), k), -1, dtype=torch.long) for h in ("bar", "step", "ts")}
    for i, (f, t, _) in enumerate(samples):
        x[i, :len(f)] = torch.from_numpy(f)
        mask[i, :len(f)] = True
        for h in tg:
            tg[h][i, :len(f)] = torch.from_numpy(t[h])
    return x.to(device), mask.to(device), {h: v.to(device) for h, v in tg.items()}


@torch.no_grad()
def validate(model, val: list, device, noise, seed: int = 0) -> dict:
    """Head accuracies and decoded-grid scores over every validation performance (the tracker
    simulated with a fixed seed per performance; the real one where it exists)."""
    model.eval()
    ok = {h: 0 for h in ("bar", "step", "ts")}
    n = {h: 0 for h in ok}
    scores, raw = [], []
    for k, perf in enumerate(val):
        rng = np.random.default_rng((seed, k))
        f, tg, bt = sample(perf, rng, noise)
        if len(f) < 2:
            continue
        x = torch.from_numpy(f)[None].to(device)
        out = model(x, torch.ones(1, len(f), dtype=torch.bool, device=device))
        lp = {h: torch.log_softmax(v[0].float(), -1).cpu().numpy() for h, v in out.items()}
        for h in ok:
            m = tg[h] >= 0
            ok[h] += int((lp[h].argmax(-1) == tg[h])[m].sum())
            n[h] += int(m.sum())
        grid = grid_from_states(bt, decode(lp["bar"], lp["step"], lp["ts"], DecodeConfig()))
        scores.append(grid_scores(grid, perf["gt"]))
        gt = perf["gt"]
        rng2 = np.random.default_rng((seed, k))  # the same tracked beats again, for the tracker's own scores
        tb, tdn = perf["beats"] if perf.get("beats") is not None else noise(gt["t"], gt["down"], rng2)
        raw.append({"beat_f": f_measure(gt["t"], tb), "downbeat_f": f_measure(gt["t"][gt["down"]], tb[tdn])})
    model.train()
    res = {f"{h}_acc": ok[h] / max(n[h], 1) for h in ok}
    for key in ("beat_f", "downbeat_f", "ts_acc"):
        res[f"grid_{key}"] = float(np.mean([s[key] for s in scores])) if scores else 0.0
    for key in ("beat_f", "downbeat_f"):
        res[f"tracker_{key}"] = float(np.mean([s[key] for s in raw])) if raw else 0.0
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=384, help="beats per training sample")
    ap.add_argument("--beat-noise", default="configs/beat_noise.json")
    ap.add_argument("--note-noise", default="configs/noise_stats.json", help="'none' = clean notes")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=6, help="evaluations without a better downbeat F (0 = off)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--normal-priority", action="store_true")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--runs-dir", default="outputs/runs")
    args = ap.parse_args(argv)
    if not args.normal_priority:
        lower_priority()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    data, run = Path(args.data), Path(args.runs_dir) / args.name
    run.mkdir(parents=True, exist_ok=True)
    train = load(data, "train") + load(data, "train_transcribed")
    val = load(data, "val") + load(data, "val_transcribed")
    if not train or not val:
        raise SystemExit(f"{data / 'meter'}: no train / val performances: run scripts/build_meter_data.py first")
    noise = BeatNoise.from_json(args.beat_noise)
    note_noise = None if args.note_noise == "none" else NoiseModel.from_json(args.note_noise)
    cfg = MeterConfig(d_model=args.d_model, layers=args.layers, dropout=args.dropout,
                      n_heads=max(4, args.d_model // 32), ff=4 * args.d_model).to_dict()
    ck = None
    if args.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        cfg = ck["config"]
    elif (run / "last.pt").exists():
        raise SystemExit(f"{run} already has a checkpoint: use --resume, or a different --name")
    model = build_meter_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    step, best, best_step, done = 0, -1.0, 0, False
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best, best_step, done = ck["step"], ck["best"], ck["best_step"], ck.get("done", False)
    n_params = sum(p.numel() for p in model.parameters())
    json.dump({"args": vars(args), "arch": "meter", "model": cfg, "params": n_params,
               "data": {"train": len(train), "val": len(val),
                        "real_transcriptions": sum(p.get("beats") is not None for p in train)}},
              open(run / "config.json", "w"), indent=1)
    print(f"meter model: {n_params / 1e6:.2f}M parameters on {device} | {len(train)} training performances "
          f"({sum(p.get('beats') is not None for p in train)} with real tracked beats), {len(val)} validation", flush=True)

    def save(path):
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "best_step": best_step, "done": done, "arch": "meter", "config": cfg}, tmp)
        os.replace(tmp, path)
        st = run / "status.json.tmp"
        st.write_text(json.dumps({"phase": "done" if done else "train", "step": step, "total_steps": args.steps,
                                  "best": best, "best_step": best_step}))
        os.replace(st, run / "status.json")

    rng = np.random.default_rng(args.seed + step)
    w = np.array([len(p["gt"]["t"]) for p in train], float)
    w /= w.sum()  # longer performances hold more beats: sample in proportion
    log_path, t0, run_loss = run / "log.csv", time.time(), 0.0
    model.train()
    try:
        while not done:
            idx = rng.choice(len(train), args.batch, p=w)
            x, mask, tg = batch([sample(train[i], rng, noise, args.chunk, note_noise) for i in idx], device)
            for g in opt.param_groups:
                g["lr"] = learning_rate(step, lr=args.lr, warmup=args.warmup, total_steps=args.steps,
                                        min_lr_frac=args.min_lr_frac)
            loss, parts = meter_loss(model(x, mask), tg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            run_loss += float(loss.detach())
            if step % 100 == 0:
                print(f"step {step}/{args.steps} loss {run_loss / 100:.4f} ({time.time() - t0:.0f} s)", flush=True)
                run_loss = 0.0
            if step % args.eval_every == 0 or step >= args.steps:
                res = validate(model, val, device, noise, args.seed)
                res.update(step=step, minutes=round((time.time() - t0) / 60, 1))
                new = not log_path.exists()
                with open(log_path, "a", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(res))
                    if new:
                        wr.writeheader()
                    wr.writerow(res)
                print("  eval " + " | ".join(f"{k}={v:.3f}" for k, v in res.items() if isinstance(v, float)), flush=True)
                if res["grid_downbeat_f"] > best:
                    best, best_step = res["grid_downbeat_f"], step
                    save(run / "best.pt")
                if step >= args.steps or (args.patience and (step - best_step) >= args.patience * args.eval_every):
                    done = True
                save(run / "last.pt")
    except KeyboardInterrupt:
        save(run / "last.pt")
        sys.exit(130)
    print(f"done: {step} steps; best validation downbeat F {best:.3f} at step {best_step} -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
