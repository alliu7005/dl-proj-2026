"""Train the Scoreformer (src/scoreformer.py), or Score2Seq (src/model.py) under the same loop.

    python scripts/train_model.py --name sf_base                       # Scoreformer, base size (~45M)
    python scripts/train_model.py --name sf_base --resume              # continue after Ctrl+C / crash
    python scripts/train_model.py --name sf_clean --no-noise           # clean-input ablation
    python scripts/train_model.py --name s2s --arch score2seq          # Michael's model, same loop

Same data, batching and noise as scripts/train.py (it reuses that script's helpers), so the only
difference between two runs is what you change on the command line. Differences from train.py:
  * ``--arch`` / ``--size``: architecture and size preset;
  * splits that don't exist yet (train_transcribed, val_transcribed, PERiScoPe) are skipped
    instead of crashing, so it runs on an ASAP-only build too;
  * best.pt = lowest mean validation loss over all available validation sets (ASAP clean,
    PERiScoPe real, ASAP real transcriptions), instead of the small val_real set alone;
  * ``--compile`` (torch.compile) and ``--grad-accum`` for big GPUs / big models.

Outputs in outputs/runs/<name>/: config.json, log.csv (one row per eval), last.pt, best.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import (SHOW, TokenBudgetBatches, TruncatingCollate, lower_priority,  # noqa: E402
                           pick_device, store_lengths)
from src.dataset import AugmentConfig, MixedTrainDataset, ScoreSegmentDataset  # noqa: E402
from src.eval.metrics import Accumulator  # noqa: E402
from src.model import token_loss  # noqa: E402
from src.preprocessing.noise import NoiseModel  # noqa: E402
from src.preprocessing.vocab import EOS  # noqa: E402
from src.scoreformer import PRESETS, build_model  # noqa: E402

D = Path("data/processed")


def has_split(name: str) -> bool:
    return (D / name / "index.npy").exists() and len(np.load(D / name / "index.npy", mmap_mode="r")) > 0


class Evaluator:
    """Fixed validation subsets: teacher-forced loss / token accuracy + greedy-decoding metrics
    (ground-truth prompt, as in scripts/train.py). Sets with no data are left out."""

    def __init__(self, vocab_path, n_loss: int, n_decode: int, coll, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.coll = coll
        self.sets = {}

        def sample(idx, n):
            idx = list(idx)
            return sorted(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())

        val = ScoreSegmentDataset(str(D / "val"), vocab_path)
        self.vocab = val.vocab
        for name, ds_name in (("val_asap", "asap"), ("val_periscope", "pianocore")):
            idx = [i for i, m in enumerate(val.store.meta) if m.get("dataset", "asap") == ds_name]
            if idx:
                self.sets[name] = (val, sample(idx, n_loss), sample(idx, n_decode))
        if has_split("val_transcribed"):
            real = ScoreSegmentDataset(str(D / "val_transcribed"), vocab_path)
            self.sets["val_real"] = (real, sample(range(len(real)), n_loss), sample(range(len(real)), n_decode))
        print("validation sets:", {k: len(v[1]) for k, v in self.sets.items()}, flush=True)

    @torch.no_grad()
    def run(self, model, device, amp, batch_size: int, weights) -> dict:
        model.eval()
        res, eos = {}, self.vocab.stoi[EOS]
        for name, (ds, loss_idx, dec_idx) in self.sets.items():
            tot, n, correct, count = 0.0, 0, 0, 0
            for k in range(0, len(loss_idx), batch_size):
                chunk = loss_idx[k:k + batch_size]
                b = {kk: v.to(device) for kk, v in self.coll([ds[i] for i in chunk]).items()}
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = model(b["notes"], b["note_mask"], b["target"][:, :-1])
                # unweighted CE for model selection (comparable across --ornament-weight settings)
                tot += token_loss(logits, b["target"], b["loss_mask"]).item() * len(chunk)
                n += len(chunk)
                m = b["loss_mask"][:, 1:]
                correct += ((logits.argmax(-1) == b["target"][:, 1:]) & m).sum().item()
                count += m.sum().item()
            res[f"{name}_loss"] = tot / max(n, 1)
            res[f"{name}_tok_acc"] = correct / max(count, 1)
            acc = Accumulator()
            for k in range(0, len(dec_idx), batch_size):
                items = [ds[i] for i in dec_idx[k:k + batch_size]]
                b = self.coll(items)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    preds = model.generate(b["notes"].to(device), b["note_mask"].to(device),
                                           [it["target"][:it["prompt_len"]] for it in items], eos)
                for it, p in zip(items, preds):
                    acc.add(self.vocab.decode(it["target"][it["prompt_len"]:-1].tolist()), self.vocab.decode(p))
            r = acc.result()
            for key in SHOW:
                res[f"{name}_{key}"] = r[key]
        res["val_loss"] = float(np.mean([res[f"{k}_loss"] for k in self.sets]))
        model.train()
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="scoreformer", choices=["scoreformer", "score2seq"])
    ap.add_argument("--size", default="base", choices=list(PRESETS), help="Scoreformer size preset")
    ap.add_argument("--d-model", type=int, help="override the preset (both archs)")
    ap.add_argument("--enc-layers", type=int)
    ap.add_argument("--dec-layers", type=int)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=float, default=20)
    ap.add_argument("--batch-tokens", type=int, default=16384, help="token budget per batch (incl. padding)")
    ap.add_argument("--batch-notes", type=int, default=8192, help="input-note budget per batch")
    ap.add_argument("--grad-accum", type=int, default=1, help="batches per optimizer step")
    ap.add_argument("--batch-size", type=int, default=32, help="batch size for evaluation")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--ornament-weight", type=float, default=5.0)
    ap.add_argument("--max-notes", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--no-noise", action="store_true", help="clean inputs only (no synthetic noise, no real transcriptions)")
    ap.add_argument("--noise", default="configs/noise_stats.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-loss-windows", type=int, default=400)
    ap.add_argument("--eval-decode-windows", type=int, default=48)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (debugging)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--normal-priority", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.normal_priority:
        lower_priority()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda" and not amp:
        print("note: this GPU has no bf16 support (e.g. V100), running in fp32: slower, ask for A100/H100")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run = Path("outputs/runs") / args.name
    run.mkdir(parents=True, exist_ok=True)
    vocab_path = str(D / "vocab.json")

    # ---- data (same recipe as scripts/train.py)
    real_ok = has_split("train_transcribed")
    if args.no_noise:
        train = ScoreSegmentDataset(str(D / "train"), vocab_path, augment=AugmentConfig())
        clean_idx = [i for i, m in enumerate(train.store.meta) if m.get("source", "clean") == "clean"]
        w = np.zeros(len(train))
        w[clean_idx] = train.weights[clean_idx]
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                                         num_samples=len(clean_idx), replacement=True)
        tok_len, note_len = store_lengths(train.store, args.max_notes, args.max_tokens)
    elif real_ok:
        nm = NoiseModel.from_json(args.noise)
        train = MixedTrainDataset(str(D / "train"), str(D / "train_transcribed"), vocab_path, noise=nm,
                                  augment=AugmentConfig(), real_fraction=0.0)
        train.real_fraction = len(train.real) / len(train)  # natural proportion
        sampler = train.sampler()
        a, b = store_lengths(train.synthetic.store, args.max_notes, args.max_tokens)
        c, d = store_lengths(train.real.store, args.max_notes, args.max_tokens)
        tok_len, note_len = np.concatenate([a, c]), np.concatenate([b, d])
    else:
        print("note: no data/processed/train_transcribed, training on `train` only", flush=True)
        train = ScoreSegmentDataset(str(D / "train"), vocab_path, noise=NoiseModel.from_json(args.noise),
                                    augment=AugmentConfig())
        sampler = train.sampler()
        tok_len, note_len = store_lengths(train.store, args.max_notes, args.max_tokens)
    vocab = train.vocab
    coll = TruncatingCollate(args.max_notes, args.max_tokens)
    batches = TokenBudgetBatches(sampler, tok_len, note_len, args.batch_tokens, args.batch_notes, seed=args.seed)
    dl = torch.utils.data.DataLoader(train, batch_sampler=batches, collate_fn=coll, num_workers=args.workers,
                                     pin_memory=device.type == "cuda",
                                     prefetch_factor=4 if args.workers else None,
                                     persistent_workers=False)
    steps_per_epoch = max(1, len(dl) // args.grad_accum)
    total_steps = int(args.epochs * steps_per_epoch)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    # ---- model
    if args.arch == "scoreformer":
        cfg = dict(vocab_size=len(vocab), dropout=args.dropout, max_tokens=args.max_tokens,
                   max_notes=args.max_notes, **PRESETS[args.size])
        stats = D / "feature_stats.json"
        if stats.exists():
            st = json.loads(stats.read_text())
            assert st["index"] == [1, 2, 3, 4], st["index"]
            cfg.update(feat_mean=st["mean"], feat_std=st["std"])
    else:
        cfg = dict(vocab_size=len(vocab), dropout=args.dropout, max_tokens=args.max_tokens,
                   max_notes=args.max_notes, d_model=256, enc_layers=4, dec_layers=6)
    for k in ("d_model", "enc_layers", "dec_layers"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.arch == "score2seq":
        cfg.update(ff=4 * cfg["d_model"], n_heads=max(4, cfg["d_model"] // 32))
    elif args.d_model is not None:
        cfg.update(ff=4 * cfg["d_model"], n_heads=max(4, cfg["d_model"] // 64))
    ck = None
    if args.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        cfg = ck["config"]  # the checkpoint's architecture wins over the command line
        if ck.get("arch", args.arch) != args.arch:
            raise SystemExit(f"{run} is a {ck['arch']} run, not {args.arch}")
    model = build_model(args.arch, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), fused=device.type == "cuda")
    fwd = torch.compile(model) if args.compile else model

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = min(1.0, (step - args.warmup) / max(1, total_steps - args.warmup))
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))

    weights = torch.tensor(vocab.loss_weights(args.ornament_weight), device=device)
    step, epoch, best = 0, 0, float("inf")
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, epoch, best = ck["step"], ck["epoch"], ck.get("best", best)
        print(f"resumed {run} at step {step} (epoch {epoch})", flush=True)
    elif (run / "last.pt").exists():
        raise SystemExit(f"{run} already has a checkpoint: use --resume, or a different --name")
    json.dump({"args": vars(args), "arch": args.arch, "model": cfg, "params": n_params,
               "steps_per_epoch": steps_per_epoch, "total_steps": total_steps},
              open(run / "config.json", "w"), indent=1)
    print(f"{args.arch}: {n_params / 1e6:.1f}M parameters on {device} | {len(train)} training windows, "
          f"{steps_per_epoch} steps/epoch, {total_steps} steps total | Ctrl+C saves and stops", flush=True)

    evaluator = Evaluator(vocab_path, args.eval_loss_windows, args.eval_decode_windows, coll)

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": epoch,
                    "best": best, "arch": args.arch, "config": cfg}, path)

    log_path = run / "log.csv"
    model.train()
    t0 = t_int = time.time()
    tok_int, wait_int, run_loss, n_loss, micro = 0, 0.0, 0.0, 0, 0
    try:
        while step < total_steps:
            if hasattr(train, "set_epoch"):
                train.set_epoch(epoch)
            batches.set_epoch(epoch)
            t_wait = time.time()
            for b in dl:
                wait_int += time.time() - t_wait
                b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = fwd(b["notes"], b["note_mask"], b["target"][:, :-1])
                loss = token_loss(logits, b["target"], b["loss_mask"], weights, args.label_smoothing)
                (loss / args.grad_accum).backward()
                run_loss += loss.item()
                n_loss += 1
                tok_int += int(b["loss_mask"].sum())
                micro += 1
                if micro % args.grad_accum:
                    t_wait = time.time()
                    continue
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 100 == 0:
                    el = time.time() - t_int
                    print(f"step {step}/{total_steps} ep {epoch} loss {run_loss / n_loss:.4f} lr {lr_at(step):.2e} "
                          f"| {tok_int / el:.0f} tok/s, waiting for data {100 * wait_int / el:.0f}%", flush=True)
                    t_int, tok_int, wait_int, run_loss, n_loss = time.time(), 0, 0.0, 0.0, 0
                if step % args.eval_every == 0 or step == total_steps:
                    res = evaluator.run(model, device, amp, args.batch_size, weights)
                    res.update(step=step, epoch=epoch, lr=lr_at(step), minutes=round((time.time() - t0) / 60, 1))
                    new = not log_path.exists()
                    with open(log_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(res))
                        if new:
                            w.writeheader()
                        w.writerow(res)
                    print("  eval " + " | ".join(f"{k}={v:.3f}" for k, v in res.items()
                                                if isinstance(v, float) and ("loss" in k or "f1" in k or "acc" in k)),
                          flush=True)
                    if res["val_loss"] < best:
                        best = res["val_loss"]
                        save(run / "best.pt")
                    save(run / "last.pt")
                    t_int, tok_int, wait_int = time.time(), 0, 0.0
                if step >= total_steps:
                    break
                t_wait = time.time()
            else:  # the epoch's batches ran out (not stopped by total_steps)
                epoch += 1
    except KeyboardInterrupt:
        save(run / "last.pt")
        print(f"\nstopped at step {step}: saved {run / 'last.pt'}. Continue with: "
              f"python scripts/train_model.py --name {args.name} --resume", flush=True)
        sys.exit(130)
    save(run / "last.pt")
    print(f"done: {step} steps. best mean val loss {best:.4f} -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
