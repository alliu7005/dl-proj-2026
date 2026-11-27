"""Self-supervised pretraining of the meter model on unlabelled piano MIDI (Aria-MIDI:
scripts/fetch_aria_midi.py), then fine-tuning with the usual recipe (train_meter.py --init).

    python scripts/pretrain_meter.py prep --midi data/raw/aria_midi/unique     # PM2S beats + beat features
    python scripts/pretrain_meter.py train --name meter_pt                     # masked-beat pretraining
    -> data/processed/meter/pretrain.pkl; outputs/runs/meter_pt/best.pt (a meter checkpoint --init takes)

Why: the meter model is v2's bottleneck and trains on a few thousand labelled performances; every
meter idea so far (decoder tuning, repetition features, tolerant training, synthetic scores, the
note-value level of round 5) moved it little, which points at data. Aria-MIDI has ~1.2M solo-piano
transcriptions (like our real inputs) but no scores.

prep: each file's notes (src/preprocessing/performance.py load_performance), PM2S's beats of them
  (src/pm2s_beats.py: from the notes, no audio), and the beat features the meter model reads
  (src/meter.py beat_features), stored as float16. Performances with fewer than ``--min-beats``
  beats are left out.
train: the meter model of meter_v4_gq's size (train_meter.py's defaults: 256 wide, 6 layers) reads
  each chunk of beats with ``--mask`` of them hidden (spans of 1-4 beats, their features zeroed) and
  rebuilds the hidden beats' standardised features from the beats around them (mean squared error,
  a linear head on the trunk): it has to learn how beats, accents, bass notes and harmony recur,
  the regularity a bar is. The checkpoint holds the meter model's weights and config (the recon
  head kept apart), so ``train_meter.py --init outputs/runs/meter_pt/best.pt`` starts from the
  pretrained trunk; its bar / step / ts heads were never trained and are learnt in fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def meter_cfg(d_model: int = 256, layers: int = 6, dropout: float = 0.1) -> dict:
    """train_meter.py's config for its defaults (meter_v4_gq: Beat This! features, no second tracker,
    no note encoder, no repetition features)."""
    from src.meter import N_BEAT_FEATURES, MeterConfig
    return MeterConfig(n_features=N_BEAT_FEATURES, d_model=d_model, layers=layers, dropout=dropout,
                       n_heads=max(4, d_model // 32), ff=4 * d_model).to_dict()


def prep(args) -> int:
    from src.meter import N_BEAT_FEATURES, beat_features
    from src.pm2s_beats import PM2SBeats
    from src.preprocessing.performance import load_performance
    files = sorted(Path(args.midi).glob("*.mid")) + sorted(Path(args.midi).glob("*.midi"))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"no MIDI files in {args.midi} (scripts/fetch_aria_midi.py)")
    import torch
    tracker = PM2SBeats(device="cuda" if torch.cuda.is_available() else "cpu")
    out, skipped, t0 = [], {}, time.time()
    for k, f in enumerate(files):
        try:
            notes, _ = load_performance(f)
            if len(notes) < 64:
                skipped["few notes"] = skipped.get("few notes", 0) + 1
                continue
            bt, bd = tracker(notes)
            if len(bt) < args.min_beats:
                skipped["few beats"] = skipped.get("few beats", 0) + 1
                continue
            feats = beat_features(notes, np.asarray(bt, np.float64), np.asarray(bd, bool))[:, :N_BEAT_FEATURES]
            if not np.isfinite(feats).all():
                skipped["bad features"] = skipped.get("bad features", 0) + 1
                continue
            out.append({"file": f.name, "x": feats.astype(np.float16)})
        except Exception as e:  # noqa: BLE001  (an unreadable file is left out)
            key = type(e).__name__
            skipped[key] = skipped.get(key, 0) + 1
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(files)}: {len(out)} kept ({time.time() - t0:.0f} s)", flush=True)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    rep = {"files": len(files), "kept": len(out), "beats": int(sum(len(p["x"]) for p in out)), "skipped": skipped}
    dst.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {dst}: {rep}")
    return 0


def span_mask(k: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """[k] bool: about ``ratio`` of the beats hidden, in spans of 1-4 beats."""
    m = np.zeros(k, bool)
    target = max(1, int(round(ratio * k)))
    while m.sum() < target:
        s, n = int(rng.integers(0, k)), int(rng.integers(1, 5))
        m[s:s + n] = True
    return m


def trunk(model, x, mask):
    """The meter model's beat states [B, K, d] (its forward without the heads)."""
    import torch

    from src.scoreformer import rope_angles
    c = model.cfg
    pos = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
    cos, sin = rope_angles(pos, model.freq)
    h = model.inp(x[..., :c.n_features])
    m = mask[:, None, None, :]
    for layer in model.layers:
        h = layer(h, cos, sin, m)
    return model.norm(h)


def batch(perfs, idx, chunk, ratio, rng, mean, std, device):
    import torch
    xs, hide = [], []
    for i in idx:
        x = perfs[i]["x"].astype(np.float32)
        if len(x) > chunk:
            s = int(rng.integers(0, len(x) - chunk + 1))
            x = x[s:s + chunk]
        xs.append(x)
        hide.append(span_mask(len(x), ratio, rng))
    k, f = max(len(x) for x in xs), xs[0].shape[1]
    X = np.zeros((len(xs), k, f), np.float32)
    M = np.zeros((len(xs), k), bool)
    H = np.zeros((len(xs), k), bool)
    for j, (x, h) in enumerate(zip(xs, hide)):
        X[j, :len(x)], M[j, :len(x)], H[j, :len(x)] = x, True, h
    Z = (X - mean) / std
    Xin = X.copy()
    Xin[H] = 0.0
    return tuple(torch.from_numpy(a).to(device) for a in (Xin, M, Z, H & M))


def train(args) -> int:
    import torch
    import torch.nn as nn

    from scripts.train import pick_device
    from src.meter import build_meter_model
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with open(args.data, "rb") as fh:
        perfs = pickle.load(fh)
    if len(perfs) < 20:
        raise SystemExit(f"{args.data}: {len(perfs)} performances (prep first)")
    order = rng.permutation(len(perfs))
    n_val = max(10, len(perfs) // 50)
    val, tr = order[:n_val].tolist(), order[n_val:].tolist()
    sample = np.concatenate([perfs[i]["x"].astype(np.float32) for i in tr[:2000]])
    mean, std = sample.mean(0), np.maximum(sample.std(0), 1e-3)
    cfg = meter_cfg(args.d_model, args.layers, args.dropout)
    model = build_meter_model(cfg).to(device)
    recon = nn.Linear(cfg["d_model"], cfg["n_features"]).to(device)
    params = list(model.parameters()) + list(recon.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.98))
    run = Path(args.runs_dir) / args.name
    run.mkdir(parents=True, exist_ok=True)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    mean_t, std_t = mean.astype(np.float32), std.astype(np.float32)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = min(1.0, (s - args.warmup) / max(1, args.steps - args.warmup))
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    def loss_of(idx, r):
        Xin, M, Z, H = batch(perfs, idx, args.chunk, args.mask, r, mean_t, std_t, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            h = trunk(model, Xin, M)
        pred = recon(h.float())
        return ((pred - Z) ** 2)[H].mean()

    best, log, t0 = float("inf"), [], time.time()
    for step in range(args.steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = rng.choice(tr, size=min(args.batch, len(tr)), replace=False)
        loss = loss_of(idx, rng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            model.eval()
            vr = np.random.default_rng(1)   # the same hidden beats at every evaluation
            with torch.no_grad():
                vl = float(np.mean([float(loss_of(val[i:i + args.batch], vr)) for i in range(0, len(val), args.batch)]))
            log.append({"step": step + 1, "train": float(loss.detach()), "val": vl, "seconds": round(time.time() - t0)})
            print(f"step {step + 1}: train {float(loss.detach()):.4f} val {vl:.4f} ({time.time() - t0:.0f} s)", flush=True)
            ck = {"model": model.state_dict(), "recon": recon.state_dict(), "config": cfg, "arch": "meter",
                  "pretrain": {"data": str(args.data), "performances": len(perfs), "step": step + 1, "val": vl,
                               "mask": args.mask, "mean": mean.tolist(), "std": std.tolist()}}
            if vl < best:
                best = vl
                tmp = run / "best.pt.tmp"
                torch.save(ck, tmp)
                os.replace(tmp, run / "best.pt")
            (run / "pretrain_log.json").write_text(json.dumps(log, indent=1))
    (run / "status.json").write_text(json.dumps({"phase": "done", "step": args.steps, "best": best}))
    print(f"wrote {run / 'best.pt'} (best val {best:.4f}; a model that predicts the mean scores 1.0)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--midi", default="data/raw/aria_midi/unique")
    p.add_argument("--out", default="data/processed/meter/pretrain.pkl")
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--min-beats", type=int, default=32)
    t = sub.add_parser("train")
    t.add_argument("--name", default="meter_pt")
    t.add_argument("--data", default="data/processed/meter/pretrain.pkl")
    t.add_argument("--runs-dir", default="outputs/runs")
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--chunk", type=int, default=512)
    t.add_argument("--mask", type=float, default=0.15)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--d-model", type=int, default=256)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    return prep(args) if args.cmd == "prep" else train(args)


if __name__ == "__main__":
    sys.exit(main())
