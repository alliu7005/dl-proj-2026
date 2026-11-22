"""Train the meter model (v2, Phase 2: src/meter.py) on whole performances.

    python scripts/build_meter_data.py                  # once: data/processed/meter/<split>.pkl
    python scripts/train_meter.py --name meter_v2
    python scripts/train_meter.py --name meter_v2 --resume

Training data (every source that data/processed/meter has):
  ASAP           the clean training performances: the tracker simulated on the annotated beats
                 (configs/beat_noise.json: octave errors, missed / extra beats, jitter, downbeat
                 errors; a new draw every time) and synthetic transcription noise on the notes
                 (--note-noise)
  real           the Transkun transcriptions of ASAP training audio with Beat This!'s own beats
  PERiScoPe      real Transkun transcriptions of 971 more works, the grid from their score
                 alignment (scripts/build_meter_data.py); the tracker simulated, the notes as they
                 are (--no-periscope leaves them out)
Every step draws --batch performances (--balance piece: each piece in proportion to its length,
however many performances it has; beats: each performance in proportion to its length), augments
each (src/meter.py Augment: tempo stretch, transposition, velocity; --no-augment turns it off: on
ASAP alone a model then memorises its 156 pieces within ~2k steps, validation time-signature
accuracy falling while training goes on) and cuts a random stretch of --chunk-min .. --chunk beats
from each. --workers processes make the samples.

Validation (every --eval-every steps), per source (ASAP val, real val, a fixed --val-periscope
PERiScoPe val performances): the heads' accuracies and the DECODED grids scored against the
ground truth (src/meter.py grid_scores: beat F, downbeat F, time-signature accuracy) next to the
raw tracker's beat / downbeat F on the same beats. best.pt keeps the best mean over the sources of
(downbeat F + time-signature accuracy) / 2: the bar lines and meters the note labeller is given.
Early stopping, resume and atomic checkpoints as in the other training scripts; --max-hours stops
cleanly (last.pt) before a job's time limit, and --resume goes on from there.

Inputs: per-beat features (src/meter.py beat_features). --note-encoder adds an embedding of every
note pooled into its beat; off by default since the second round: with PERiScoPe it overfits
(best validation at 4k steps, then early stopping) where the beat features alone kept improving
to 26k and won on every test set (README, "Meter model, second round: results").
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import lower_priority, pick_device  # noqa: E402
from scripts.train_model import learning_rate  # noqa: E402
from src.beats import BeatNoise, f_measure  # noqa: E402
from src.meter import (N_BEAT_FEATURES, N_NOTE_FEATURES, Augment, DecodeConfig, MeterConfig,  # noqa: E402
                       build_meter_model, collate, grid_from_logp, grid_scores, meter_loss, sample)
from src.preprocessing.noise import NoiseModel  # noqa: E402

FIRST_FEATURES = 27  # the first meter model's beat features (--beat-features first)
HEADS = ("bar", "step", "ts")


def load(data: Path, split: str) -> list:
    """data/processed/meter/<split>.pkl ([] when it was not built); every performance gets a
    ``piece`` (its folder for the ASAP ones built before pieces were stored)."""
    p = data / "meter" / f"{split}.pkl"
    perfs = pickle.load(open(p, "rb")) if p.exists() else []
    for x in perfs:
        x.setdefault("piece", str(Path(x["performance"]).parent))
    return perfs


def weights(perfs: list, balance: str) -> np.ndarray:
    """Sampling weights: ``beats`` = each performance by its length; ``piece`` = each piece by its
    (mean performance) length, shared by its performances, so a piece played 20 times is not
    drawn 20 times as often as one played once."""
    n = np.array([len(p["gt"]["t"]) for p in perfs], float)
    if balance == "piece":
        count = Counter(p["piece"] for p in perfs)
        n = n / np.array([count[p["piece"]] for p in perfs], float)
    return n / n.sum()


class Stream(torch.utils.data.IterableDataset):
    """Endless training batches (collated on the CPU), each worker with its own random stream."""

    def __init__(self, perfs, w, noise, note_noise, batch, chunk, chunk_min, seed, augment=None):
        self.perfs, self.w, self.noise, self.note_noise, self.augment = perfs, w, noise, note_noise, augment
        self.batch, self.chunk, self.chunk_min, self.seed = batch, chunk, chunk_min, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng((self.seed, info.id if info else 0))
        while True:
            length = int(rng.integers(self.chunk_min, self.chunk + 1))
            idx = rng.choice(len(self.perfs), self.batch, p=self.w)
            yield collate([sample(self.perfs[i], rng, self.noise, length, self.note_noise, self.augment)
                           for i in idx])


@torch.no_grad()
def validate(model, groups: dict, device, noise, seed: int = 0) -> dict:
    """Per validation source: head accuracies and decoded-grid scores (the tracker simulated with a
    fixed seed per performance; the real one where it exists); ``select`` = the mean over the
    sources of (downbeat F + time-signature accuracy) / 2."""
    model.eval()
    res, sel = {}, []
    ok = {h: 0 for h in HEADS}
    n = {h: 0 for h in HEADS}
    for gi, (name, perfs) in enumerate(groups.items()):
        scores, raw = [], []
        for k, perf in enumerate(perfs):
            f, tg, bt, nin = sample(perf, np.random.default_rng((seed, gi, k)), noise)
            if len(f) < 2:
                continue
            x, mask, notes, _ = collate([(f, None, bt, nin)], device)
            out = model(x, mask, notes)
            lp = {h: torch.log_softmax(v[0].float(), -1).cpu().numpy() for h, v in out.items()}
            for h in HEADS:
                m = tg[h] >= 0
                ok[h] += int((lp[h].argmax(-1) == tg[h])[m].sum())
                n[h] += int(m.sum())
            grid = grid_from_logp(bt, lp, DecodeConfig())
            scores.append(grid_scores(grid, perf["gt"]))
            gt = perf["gt"]
            rng2 = np.random.default_rng((seed, gi, k))  # the same tracked beats again, for the tracker's own scores
            tb, tdn = perf["beats"] if perf.get("beats") is not None else noise(gt["t"], gt["down"], rng2)
            raw.append({"beat_f": f_measure(gt["t"], tb), "downbeat_f": f_measure(gt["t"][gt["down"]], tb[tdn])})
        if not scores:
            continue
        for key in ("beat_f", "downbeat_f", "ts_acc"):
            res[f"{name}_grid_{key}"] = float(np.mean([s[key] for s in scores]))
        for key in ("beat_f", "downbeat_f"):
            res[f"{name}_tracker_{key}"] = float(np.mean([s[key] for s in raw]))
        sel.append((res[f"{name}_grid_downbeat_f"] + res[f"{name}_grid_ts_acc"]) / 2)
    model.train()
    res.update({f"{h}_acc": ok[h] / max(n[h], 1) for h in HEADS})
    res["select"] = float(np.mean(sel)) if sel else 0.0
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--note-encoder", action=argparse.BooleanOptionalAction, default=False,
                    help="pool a learned embedding of every note into its beat (--no-note-encoder: beat features only)")
    ap.add_argument("--d-note", type=int, default=64)
    ap.add_argument("--beat-features", choices=("all", "first"), default="all",
                    help="first = the first meter model's 27 (no pitch-class profile)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=1024, help="most beats per training sample")
    ap.add_argument("--chunk-min", type=int, default=0, help="fewest beats per training sample (0 = chunk / 4)")
    ap.add_argument("--balance", choices=("piece", "beats"), default="piece")
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                    help="tempo stretch, transposition and velocity scaling of every training sample")
    ap.add_argument("--periscope", action=argparse.BooleanOptionalAction, default=True,
                    help="train (and validate) on the PERiScoPe performances too, when built")
    ap.add_argument("--val-periscope", type=int, default=150, help="PERiScoPe validation performances (fixed draw)")
    ap.add_argument("--beat-noise", default="configs/beat_noise.json")
    ap.add_argument("--note-noise", default="configs/noise_stats.json", help="'none' = clean notes")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=5, help="evaluations without a better selection score (0 = off)")
    ap.add_argument("--workers", type=int, default=6, help="processes making training samples (0 = in the main one)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop cleanly (last.pt saved, not done) after this long; --resume continues (0 = off)")
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
    sources = {"asap": load(data, "train"), "real": load(data, "train_transcribed"),
               "periscope": load(data, "train_periscope") if args.periscope else []}
    train = [p for v in sources.values() for p in v]
    groups = {"asap": load(data, "val"), "real": load(data, "val_transcribed")}
    if args.periscope:
        pv = load(data, "val_periscope")
        if len(pv) > args.val_periscope:
            pv = [pv[i] for i in sorted(np.random.default_rng(0).choice(len(pv), args.val_periscope, replace=False))]
        groups["periscope"] = pv
    groups = {k: v for k, v in groups.items() if v}
    if not train or not groups:
        raise SystemExit(f"{data / 'meter'}: no train / val performances: run scripts/build_meter_data.py first")
    noise = BeatNoise.from_json(args.beat_noise)
    note_noise = None if args.note_noise == "none" else NoiseModel.from_json(args.note_noise)
    cfg = MeterConfig(n_features=N_BEAT_FEATURES if args.beat_features == "all" else FIRST_FEATURES,
                      d_model=args.d_model, layers=args.layers, dropout=args.dropout,
                      n_heads=max(4, args.d_model // 32), ff=4 * args.d_model,
                      n_note_features=N_NOTE_FEATURES if args.note_encoder else 0, d_note=args.d_note).to_dict()
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
    counts = {k: len(v) for k, v in sources.items()}
    pieces = {k: len({p["piece"] for p in v}) for k, v in sources.items()}
    json.dump({"args": vars(args), "arch": "meter", "model": cfg, "params": n_params,
               "data": {"train": len(train), "train_by_source": counts, "pieces_by_source": pieces,
                        "val_by_source": {k: len(v) for k, v in groups.items()},
                        "real_transcriptions": counts["real"]}},
              open(run / "config.json", "w"), indent=1)
    print(f"meter model: {n_params / 1e6:.2f}M parameters on {device} (note encoder: "
          f"{'yes' if cfg.get('n_note_features') else 'no'}, {cfg['n_features']} beat features) | training: "
          + ", ".join(f"{k} {counts[k]} performances / {pieces[k]} pieces" for k in counts)
          + " | validation: " + ", ".join(f"{k} {len(v)}" for k, v in groups.items()), flush=True)

    def save(path):
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "best_step": best_step, "done": done, "arch": "meter", "config": cfg}, tmp)
        os.replace(tmp, path)
        st = run / "status.json.tmp"
        st.write_text(json.dumps({"phase": "done" if done else "train", "step": step, "total_steps": args.steps,
                                  "best": best, "best_step": best_step}))
        os.replace(st, run / "status.json")

    stream = Stream(train, weights(train, args.balance), noise, note_noise, args.batch, args.chunk,
                    args.chunk_min or max(args.chunk // 4, 2), args.seed * 100003 + step,
                    Augment() if args.augment else None)
    loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers,
                                         prefetch_factor=4 if args.workers else None,
                                         persistent_workers=bool(args.workers))
    batches = iter(loader)
    log_path, t0, run_loss = run / "log.csv", time.time(), 0.0
    deadline = t0 + args.max_hours * 3600 if args.max_hours > 0 else float("inf")
    model.train()
    try:
        while not done:
            x, mask, notes, tg = next(batches)
            x, mask, tg = x.to(device), mask.to(device), {h: v.to(device) for h, v in tg.items()}
            notes = tuple(v.to(device) for v in notes)
            for g in opt.param_groups:
                g["lr"] = learning_rate(step, lr=args.lr, warmup=args.warmup, total_steps=args.steps,
                                        min_lr_frac=args.min_lr_frac)
            loss, parts = meter_loss(model(x, mask, notes), tg)
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
                res = validate(model, groups, device, noise, args.seed)
                res.update(step=step, minutes=round((time.time() - t0) / 60, 1))
                new = not log_path.exists()
                with open(log_path, "a", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(res))
                    if new:
                        wr.writeheader()
                    wr.writerow(res)
                print("  eval " + " | ".join(f"{k}={v:.3f}" for k, v in res.items() if isinstance(v, float)), flush=True)
                if res["select"] > best:
                    best, best_step = res["select"], step
                    save(run / "best.pt")
                if step >= args.steps or (args.patience and (step - best_step) >= args.patience * args.eval_every):
                    done = True
                save(run / "last.pt")
            if not done and time.time() > deadline:
                save(run / "last.pt")
                print(f"time limit ({args.max_hours:g} h) at step {step}: python scripts/train_meter.py --name "
                      f"{args.name} --resume goes on", flush=True)
                return
    except KeyboardInterrupt:
        save(run / "last.pt")
        sys.exit(130)
    print(f"done: {step} steps; best validation score {best:.3f} (mean of downbeat F and time-signature "
          f"accuracy) at step {best_step} -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
