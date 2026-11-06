"""Train the score transformer (src/model.py).

    python scripts/train.py --name base                 # start (or --resume to continue a run)
    python scripts/train.py --name base --resume        # continue after Ctrl+C / crash
    python scripts/train.py --name clean --no-noise     # Beyer & Dai-style clean-input baseline

Ctrl+C saves the current state and stops; ``--resume`` continues from it. Runs at below-normal CPU
priority by default. Everything goes to outputs/runs/<name>/:
  config.json, log.csv (one row per eval), last.pt (latest state), best.pt (best val_real loss)

Training data: data/processed/train (ASAP clean windows get synthetic Transkun noise +
augmentation; PERiScoPe windows are real Transkun transcriptions, augmented but never noised) plus
the ASAP real-transcription windows (train_transcribed), in their natural proportion.

Every ``--eval-every`` steps: teacher-forced loss / token accuracy on fixed validation subsets, and
greedy decoding of a few windows each from val (ASAP clean), val (PERiScoPe real) and
val_transcribed (ASAP, real Transkun), scored with src/eval/metrics.py like the baselines.
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
from src.dataset import AugmentConfig, MixedTrainDataset, ScoreSegmentDataset, collate  # noqa: E402
from src.eval.metrics import Accumulator  # noqa: E402
from src.model import ModelConfig, Score2Seq, token_loss  # noqa: E402
from src.preprocessing.noise import NoiseModel  # noqa: E402
from src.preprocessing.vocab import EOS  # noqa: E402

D = Path("data/processed")
SHOW = ("note_f1", "onset_f1", "duration_acc", "staff_acc", "bar_f1", "tie_f1", "grace_f1")


def lower_priority():
    try:
        if sys.platform == "win32":
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            import os
            os.nice(10)
    except Exception:  # noqa: BLE001
        pass


class TruncatingCollate:
    """collate() with inputs truncated to max_notes and token sequences to max_tokens (rare: dense
    real transcriptions). A class, not a closure, so DataLoader workers can pickle it."""

    def __init__(self, max_notes: int, max_tokens: int):
        self.max_notes, self.max_tokens = max_notes, max_tokens

    def __call__(self, batch):
        for b in batch:
            if len(b["notes"]) > self.max_notes:
                b["notes"] = b["notes"][:self.max_notes]
            if len(b["target"]) > self.max_tokens:
                b["target"] = b["target"][:self.max_tokens]
        return collate(batch)


class TokenBudgetBatches(torch.utils.data.Sampler):
    """Length-bucketed batches with a token budget instead of a fixed batch size.

    Draws a pool of indices from ``base`` (e.g. the weighted sampler, so oversampling is kept),
    sorts the pool by length, and cuts it into batches whose padded size stays within budget:
    batch_size * longest target <= ``max_tokens`` and batch_size * longest input <= ``max_notes``.
    Batches are then shuffled. Similar lengths per batch -> little padding -> more real tokens
    per GPU step."""

    def __init__(self, base, tok_len, note_len, max_tokens: int, max_notes: int, pool_batches: int = 50,
                 max_batch: int = 256, seed: int = 0):
        self.base, self.tok, self.notes = base, np.asarray(tok_len), np.asarray(note_len)
        self.max_tokens, self.max_notes, self.max_batch = max_tokens, max_notes, max_batch
        self.pool = pool_batches * max(1, max_tokens // max(int(np.median(self.tok)), 1))
        self.seed, self.epoch = seed, 0
        # estimated number of batches per pass (used for schedules / progress)
        mean_tok = float(np.mean(np.minimum(self.tok, max_tokens)))
        self._len = max(1, int(len(base) * mean_tok * 1.08 / max_tokens))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self._len

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        it = iter(self.base)
        while True:
            pool = [i for _, i in zip(range(self.pool), it)]
            if not pool:
                return
            pool.sort(key=lambda i: (self.tok[i], self.notes[i]))
            batches, cur, mt, mn = [], [], 0, 0
            for i in pool:
                t, n = int(self.tok[i]), int(self.notes[i])
                if cur and (len(cur) >= self.max_batch or (len(cur) + 1) * max(mt, t) > self.max_tokens
                            or (len(cur) + 1) * max(mn, n) > self.max_notes):
                    batches.append(cur)
                    cur, mt, mn = [], 0, 0
                cur.append(i)
                mt, mn = max(mt, t), max(mn, n)
            if cur:
                batches.append(cur)
            for k in rng.permutation(len(batches)):
                yield batches[k]


def store_lengths(store, max_notes: int, max_tokens: int):
    """(token length incl. bos/sep/eos, input notes) per window, capped like the collate."""
    idx = store.index
    tok = np.minimum(idx[:, 5] - idx[:, 4] + 3, max_tokens)
    notes = np.minimum(idx[:, 1] - idx[:, 0], max_notes)
    return tok, notes


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Evaluator:
    """Fixed validation subsets: teacher-forced loss + greedy-decoding metrics."""

    def __init__(self, vocab_path, n_loss: int, n_decode: int, coll, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.coll = coll
        val = ScoreSegmentDataset(str(D / "val"), vocab_path)
        real = ScoreSegmentDataset(str(D / "val_transcribed"), vocab_path)
        self.vocab = val.vocab
        asap = [i for i, m in enumerate(val.store.meta) if m.get("dataset", "asap") == "asap"]
        pc = [i for i, m in enumerate(val.store.meta) if m.get("dataset") == "pianocore"]

        def sample(idx, n):
            idx = list(idx)
            return sorted(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist()) if idx else []
        self.sets = {"val_asap": (val, sample(asap, n_loss), sample(asap, n_decode)),
                     "val_periscope": (val, sample(pc, n_loss), sample(pc, n_decode)),
                     "val_real": (real, sample(range(len(real)), n_loss), sample(range(len(real)), n_decode))}

    @torch.no_grad()
    def run(self, model, device, amp, batch_size: int, weights) -> dict:
        model.eval()
        res = {}
        for name, (ds, loss_idx, dec_idx) in self.sets.items():
            tot, n, correct, count = 0.0, 0, 0, 0
            for k in range(0, len(loss_idx), batch_size):
                b = self.coll([ds[i] for i in loss_idx[k:k + batch_size]])
                b = {kk: v.to(device) for kk, v in b.items()}
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = model(b["notes"], b["note_mask"], b["target"][:, :-1])
                tot += token_loss(logits, b["target"], b["loss_mask"], weights).item() * len(loss_idx[k:k + batch_size])
                n += len(loss_idx[k:k + batch_size])
                m = b["loss_mask"][:, 1:]
                correct += ((logits.argmax(-1) == b["target"][:, 1:]) & m).sum().item()
                count += m.sum().item()
            res[f"{name}_loss"] = tot / max(n, 1)
            res[f"{name}_tok_acc"] = correct / max(count, 1)
            acc = Accumulator()
            eos = self.vocab.stoi[EOS]
            for k in range(0, len(dec_idx), batch_size):
                items = [ds[i] for i in dec_idx[k:k + batch_size]]
                b = self.coll(items)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    preds = model.generate(b["notes"].to(device), b["note_mask"].to(device),
                                           [it["target"][:it["prompt_len"]] for it in items], eos)
                for it, p in zip(items, preds):
                    gt = self.vocab.decode(it["target"][it["prompt_len"]:-1].tolist())
                    acc.add(gt, self.vocab.decode(p))
            r = acc.result()
            for key in SHOW:
                res[f"{name}_{key}"] = r[key]
        model.train()
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch-tokens", type=int, default=16384,
                    help="token budget per batch (batch size x longest target, incl. padding)")
    ap.add_argument("--batch-notes", type=int, default=8192,
                    help="input-note budget per batch (batch size x longest input)")
    ap.add_argument("--batch-size", type=int, default=32, help="batch size for evaluation")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--ornament-weight", type=float, default=5.0)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--dec-layers", type=int, default=6)
    ap.add_argument("--max-notes", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--no-noise", action="store_true", help="clean inputs only (no synthetic noise, no real transcriptions)")
    ap.add_argument("--noise", default="configs/noise_stats.json")
    ap.add_argument("--workers", type=int, default=6,
                    help="CPU processes preparing batches (noise, augmentation, features); lower if the PC lags")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-loss-windows", type=int, default=400)
    ap.add_argument("--eval-decode-windows", type=int, default=48)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (debugging)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--normal-priority", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the training step (needs Triton; Windows: pip install triton-windows). "
                         "Measured SLOWER here (~167k vs ~350k tok/s on an RTX 5070 Ti): variable batch "
                         "shapes force dynamic compilation, which loses to PyTorch's fused attention")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.normal_priority:
        lower_priority()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    amp = device.type == "cuda"
    run = Path("outputs/runs") / args.name
    run.mkdir(parents=True, exist_ok=True)
    vocab_path = str(D / "vocab.json")

    # ---- data
    if args.no_noise:
        train = ScoreSegmentDataset(str(D / "train"), vocab_path, augment=AugmentConfig())
        clean_idx = [i for i, m in enumerate(train.store.meta) if m.get("source", "clean") == "clean"]
        weights = np.zeros(len(train))
        weights[clean_idx] = train.weights[clean_idx]
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                                         num_samples=len(clean_idx), replacement=True)
    else:
        nm = NoiseModel.from_json(args.noise)
        train = MixedTrainDataset(str(D / "train"), str(D / "train_transcribed"), vocab_path, noise=nm,
                                  augment=AugmentConfig(), real_fraction=0.0)
        train.real_fraction = len(train.real) / len(train)  # natural proportion
        sampler = train.sampler()
    vocab = train.vocab
    coll = TruncatingCollate(args.max_notes, args.max_tokens)
    if args.no_noise:
        tok_len, note_len = store_lengths(train.store, args.max_notes, args.max_tokens)
    else:
        a, b = store_lengths(train.synthetic.store, args.max_notes, args.max_tokens)
        c, d = store_lengths(train.real.store, args.max_notes, args.max_tokens)
        tok_len, note_len = np.concatenate([a, c]), np.concatenate([b, d])
    batches = TokenBudgetBatches(sampler, tok_len, note_len, args.batch_tokens, args.batch_notes, seed=args.seed)
    dl = torch.utils.data.DataLoader(train, batch_sampler=batches, collate_fn=coll,
                                     num_workers=args.workers, pin_memory=device.type == "cuda",
                                     prefetch_factor=4 if args.workers else None)
    steps_per_epoch = len(dl)
    total_steps = int(args.epochs * steps_per_epoch)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    # ---- model
    cfg = ModelConfig(vocab_size=len(vocab), d_model=args.d_model, enc_layers=args.enc_layers,
                      dec_layers=args.dec_layers, ff=4 * args.d_model, n_heads=max(4, args.d_model // 32),
                      max_tokens=args.max_tokens, max_notes=args.max_notes)
    model = Score2Seq(cfg).to(device)
    torch.set_float32_matmul_precision("high")
    # compiled wrapper only for the training forward; checkpoints / eval / generation use `model`
    fwd = torch.compile(model, dynamic=True) if args.compile else model
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay,
                            fused=device.type == "cuda")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

    weights = torch.tensor(vocab.loss_weights(args.ornament_weight), device=device)
    step, epoch, best = 0, 0, float("inf")
    if args.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, epoch, best = ck["step"], ck["epoch"], ck.get("best", best)
        print(f"resumed {run} at step {step} (epoch {epoch})", flush=True)
    elif (run / "last.pt").exists():
        raise SystemExit(f"{run} already has a checkpoint: use --resume, or a different --name")
    json.dump({"args": vars(args), "model": cfg.to_dict(), "params": n_params,
               "steps_per_epoch": steps_per_epoch, "total_steps": total_steps},
              open(run / "config.json", "w"), indent=1)
    print(f"{n_params / 1e6:.1f}M parameters on {device} | {len(train)} training windows, "
          f"{steps_per_epoch} steps/epoch, {total_steps} steps total | Ctrl+C saves and stops", flush=True)

    evaluator = Evaluator(vocab_path, args.eval_loss_windows, args.eval_decode_windows, coll)

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": epoch,
                    "best": best, "config": cfg.to_dict()}, path)

    log_path = run / "log.csv"
    model.train()
    t0 = time.time()
    t_int, tok_int, wait_int, run_loss, n_loss = time.time(), 0, 0.0, 0.0, 0
    try:
        while step < total_steps:
            if hasattr(train, "set_epoch"):
                train.set_epoch(epoch)
            batches.set_epoch(epoch)
            t_wait = time.time()
            for b in dl:
                wait_int += time.time() - t_wait
                if step >= total_steps:
                    break
                b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = fwd(b["notes"], b["note_mask"], b["target"][:, :-1])
                loss = token_loss(logits, b["target"], b["loss_mask"], weights, args.label_smoothing)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                run_loss += loss.item()
                n_loss += 1
                tok_int += int(b["loss_mask"].sum())
                if step % 100 == 0:
                    el = time.time() - t_int
                    print(f"step {step}/{total_steps} ep {epoch} loss {run_loss / n_loss:.4f} "
                          f"lr {lr_at(step):.2e} | {tok_int / el:.0f} tok/s, waiting for data "
                          f"{100 * wait_int / el:.0f}%", flush=True)
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
                                                if isinstance(v, float) and ("loss" in k or "f1" in k or "acc" in k)), flush=True)
                    if res["val_real_loss"] < best:
                        best = res["val_real_loss"]
                        save(run / "best.pt")
                    save(run / "last.pt")
                    t_int, tok_int, wait_int = time.time(), 0, 0.0  # don't count eval time
                t_wait = time.time()
            else:
                epoch += 1
                continue
            break
    except KeyboardInterrupt:
        save(run / "last.pt")
        print(f"\nstopped at step {step}: saved {run / 'last.pt'}. Continue with: "
              f"python scripts/train.py --name {args.name} --resume", flush=True)
        sys.exit(130)
    save(run / "last.pt")
    print(f"done: {step} steps. best val_real loss {best:.4f} -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
