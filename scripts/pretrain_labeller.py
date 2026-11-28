"""Masked-note pretraining of the note labeller's encoder on unlabelled piano MIDI (Aria-MIDI:
scripts/fetch_aria_midi.py), then the labeller's recipe from it (train_labeller.py --init).

    python scripts/pretrain_labeller.py prep --midi data/raw/aria_midi/unique
    python scripts/pretrain_labeller.py train --name nl_pt
    -> data/processed/labeller_pretrain.npz; outputs/runs/nl_pt/best.pt (train_labeller.py --init takes it)

The meter model's masked-beat pretraining on the same files lifted its 12k-step recipe on every real
test set (round 13: real ASAP +0.041 on (downbeat F + time signature) / 2). This is the same idea
for the labeller, which reads the notes themselves.

prep: each file cut into the windows the labeller reads (src/pipeline.py performance_windows: 128
  notes, 64 context notes before, 32 after), their note features as V2Pipeline.items makes them
  (src/dataset.py note_features with roles, standardised with data/processed/feature_stats.json),
  at most ``--per-file`` windows a file. No grid: the grid features stay 0, the labeller's own
  "no grid" input, so pretraining teaches the notes and fine-tuning the grid.
train: the labeller of nl_gq's size and inputs (base: 512 wide, 8 layers; 27 features) reads each
  window with ``--mask`` of its own notes hidden (pitch set to A0, the rarest key, standing in for
  a mask; length, velocity and pedal-at-release set to their means; onset kept) and predicts each
  hidden note's pitch (128 classes) and played length (16 bins of its standardised log
  duration) from the notes around it. The checkpoint holds the labeller's weights (its label heads
  untrained) and its config; train_labeller.py --init loads the encoder and starts the heads fresh.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DUR_EDGES = np.linspace(-2.5, 2.5, 15)     # standardised log duration -> 16 bins
MASK_PITCH = 21.0                          # A0: the embedding the hidden notes get


def labeller_cfg(data: Path, size: str = "base") -> dict:
    """train_labeller.py's model config for nl_gq's recipe (context 64 / 32, grid features, base)."""
    from src.dataset import N_FEATURES_CONTEXT
    from src.labeller import PRESETS
    from src.notelabels import N_GRID_FEATURES, LabelSpace
    from src.preprocessing.vocab import Vocab
    space = LabelSpace(Vocab.load(str(data / "vocab.json")).itos)
    cfg = dict(n_sub=space.n_sub, n_dur=space.n_dur, n_features=N_FEATURES_CONTEXT + N_GRID_FEATURES, dropout=0.1,
               max_notes=1024, context_before=64, context_after=32, **PRESETS[size])
    st = data / "feature_stats.json"
    if st.exists():
        s = json.loads(st.read_text())
        cfg.update(feat_mean=s["mean"], feat_std=s["std"])
    return cfg


def prep(args) -> int:
    from src.dataset import assembly, note_features
    from src.pipeline import performance_windows
    from src.preprocessing.performance import load_performance
    data = Path(args.data)
    cfg = labeller_cfg(data)
    mean, std = np.asarray(cfg.get("feat_mean", [0] * 4), np.float32), np.asarray(cfg.get("feat_std", [1] * 4), np.float32)
    files = sorted(Path(args.midi).glob("*.mid"))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(files)
    feats, roles, lens = [], [], []
    t0, n_win = time.time(), 0
    for k, f in enumerate(files):
        if n_win >= args.max_windows:
            break
        try:
            notes, pedal = load_performance(f)
            if len(notes) < 64:
                continue
            wins = performance_windows(notes, pedal, 128, context_before=64, context_after=32)
        except Exception:  # noqa: BLE001  (an unreadable file is left out)
            continue
        if len(wins) > args.per_file:
            wins = [wins[i] for i in sorted(rng.choice(len(wins), args.per_file, replace=False))]
        for w in wins:
            nts, rl, _ = assembly(w["notes"], w["before"], w["after"])
            x = note_features(nts, w["pedal"], rl)
            x[:, 1:5] = (x[:, 1:5] - mean) / std
            x = x[: cfg["max_notes"]]
            feats.append(x.astype(np.float16))
            roles.append(rl[: len(x)].astype(np.int8))
            lens.append(len(x))
            n_win += 1
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1} files, {n_win} windows ({time.time() - t0:.0f} s)", flush=True)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, feats=np.concatenate(feats), roles=np.concatenate(roles), offsets=off)
    rep = {"files_read": k + 1, "windows": n_win, "notes": int(off[-1]), "config": {kk: v for kk, v in cfg.items()
                                                                                    if not kk.startswith("feat_")}}
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {out}: {n_win} windows, {off[-1]} notes")
    return 0


def batch(store, idx, n_feat, ratio, rng, device):
    """(features [B, N, F] with the hidden notes changed, mask [B, N], hidden [B, N], pitch targets,
    duration-bin targets) for windows ``idx``."""
    import torch

    from src.dataset import ROLE_WINDOW
    feats, roles, off = store
    rows = [(feats[off[i]:off[i + 1]].astype(np.float32), roles[off[i]:off[i + 1]]) for i in idx]
    n = max(len(r[0]) for r in rows)
    X = np.zeros((len(rows), n, n_feat), np.float32)
    M = np.zeros((len(rows), n), bool)
    H = np.zeros((len(rows), n), bool)
    P = np.zeros((len(rows), n), np.int64)
    D = np.zeros((len(rows), n), np.int64)
    for j, (x, rl) in enumerate(rows):
        k = len(x)
        X[j, :k, :x.shape[1]] = x
        M[j, :k] = True
        own = np.nonzero(rl == ROLE_WINDOW)[0]
        if len(own):
            hide = own[rng.random(len(own)) < ratio]
            if not len(hide):
                hide = own[rng.integers(0, len(own), 1)]
            H[j, hide] = True
        P[j, :k] = np.clip(np.round(x[:, 0]), 0, 127).astype(np.int64)
        D[j, :k] = np.searchsorted(DUR_EDGES, x[:, 3])
    X[..., 0] = np.where(H, MASK_PITCH, X[..., 0])
    for c in (3, 4, 6):                                 # length, velocity, pedal at release
        X[..., c] = np.where(H, 0.0, X[..., c])
    return tuple(torch.from_numpy(a).to(device) for a in (X, M, H, P, D))


def trunk(model, x, mask):
    """The labeller's note states [B, N, d] (its forward without the label heads)."""
    from src.scoreformer import rope_angles
    h, onset = model.embed(x, mask)
    cos, sin = rope_angles(onset, model.time_freq)
    m = mask[:, None, None, :]
    for layer in model.enc:
        h = layer(h, cos, sin, m)
    return model.norm(h)


def train(args) -> int:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from scripts.train import pick_device
    from src.labeller import build_labeller
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    z = np.load(args.store)
    store = (z["feats"], z["roles"], z["offsets"])
    n_win = len(store[2]) - 1
    if n_win < 5:
        raise SystemExit(f"{args.store}: {n_win} windows (prep first)")
    cfg = labeller_cfg(Path(args.data), args.size)
    model = build_labeller(cfg).to(device)
    d = cfg["d_model"]
    heads = nn.ModuleDict({"pitch": nn.Linear(d, 128), "dur": nn.Linear(d, len(DUR_EDGES) + 1)}).to(device)
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.98))
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    order = rng.permutation(n_win)
    n_val = max(2, min(n_win // 5, max(20, n_win // 100)))   # 1% (at least 20) of a real store
    val, tr = order[:n_val].tolist(), order[n_val:]
    run = Path(args.runs_dir) / args.name
    run.mkdir(parents=True, exist_ok=True)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = min(1.0, (s - args.warmup) / max(1, args.steps - args.warmup))
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    def losses(idx, r):
        X, M, H, P, D = batch(store, idx, cfg["n_features"], args.mask, r, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            h = trunk(model, X, M)
        h = h.float()[H]
        lp, ld = heads["pitch"](h), heads["dur"](h)
        return F.cross_entropy(lp, P[H]), F.cross_entropy(ld, D[H]), (lp.argmax(-1) == P[H]).float().mean()

    best, log, t0 = float("inf"), [], time.time()
    for step in range(args.steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = rng.choice(tr, size=min(args.batch, len(tr)), replace=False)
        lp, ld, _ = losses(idx, rng)
        loss = lp + ld
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            model.eval()
            vr = np.random.default_rng(1)
            with torch.no_grad():
                vals = [losses(val[i:i + args.batch], vr) for i in range(0, len(val), args.batch)]
            vp = float(np.mean([float(v[0]) for v in vals]))
            vd = float(np.mean([float(v[1]) for v in vals]))
            va = float(np.mean([float(v[2]) for v in vals]))
            log.append({"step": step + 1, "train": float(loss.detach()), "val_pitch": vp, "val_dur": vd,
                        "val_pitch_acc": va, "seconds": round(time.time() - t0)})
            print(f"step {step + 1}: train {float(loss.detach()):.3f} | val pitch {vp:.3f} (acc {va:.3f}) dur {vd:.3f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            if vp + vd < best:
                best = vp + vd
                tmp = run / "best.pt.tmp"
                torch.save({"model": model.state_dict(), "heads": heads.state_dict(), "config": cfg, "arch": "labeller",
                            "pretrain": {"store": str(args.store), "windows": n_win, "step": step + 1,
                                         "val_pitch": vp, "val_dur": vd, "val_pitch_acc": va, "mask": args.mask}}, tmp)
                os.replace(tmp, run / "best.pt")
            (run / "pretrain_log.json").write_text(json.dumps(log, indent=1))
    (run / "status.json").write_text(json.dumps({"phase": "done", "step": args.steps, "best": best}))
    print(f"wrote {run / 'best.pt'} (best val pitch + length loss {best:.3f}; chance: {math.log(88):.2f} + {math.log(16):.2f})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--midi", default="data/raw/aria_midi/unique")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="data/processed/labeller_pretrain.npz")
    p.add_argument("--per-file", type=int, default=6)
    p.add_argument("--max-windows", type=int, default=150000)
    p.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--name", default="nl_pt")
    t.add_argument("--store", default="data/processed/labeller_pretrain.npz")
    t.add_argument("--data", default="data/processed")
    t.add_argument("--runs-dir", default="outputs/runs")
    t.add_argument("--size", default="base")
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--mask", type=float, default=0.15)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    return prep(args) if args.cmd == "prep" else train(args)


if __name__ == "__main__":
    sys.exit(main())
