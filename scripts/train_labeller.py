"""Train the note labeller (v2, src/labeller.py) on the precomputed labels and grids.

    python scripts/build_notelabels.py                          # once per data build
    python scripts/train_labeller.py --name nl_base             # 8-layer encoder, ~26M parameters
    python scripts/train_labeller.py --name nl_base --resume    # continue from last.pt

Data: the clean training windows with synthetic transcription noise, mixed in their natural
proportion with the real Transkun transcriptions (train_transcribed) when that split exists, with
transposition / tempo / velocity augmentation; 64 / 32 context notes around each window (as the
best seq2seq model). Every note's input includes where the TRUE grid (the score's beats timed
through the note alignment) puts it: Phase 1 measures the labeller given the right grid, like
the beat-snap baselines given the annotated beats; Phase 2 swaps in an estimated grid.

Validation (every --eval-every steps): the heads' losses and accuracies on --eval-loss-windows
windows of val (and val_transcribed), and the windows rendered back to tokens and scored with
src/eval/metrics.py on --eval-decode-windows of them (onset F1 etc., the numbers every system is
compared on). The best mean validation loss keeps best.pt; early stopping, cooldown, resume,
atomic checkpoints and status.json work as in scripts/train_model.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import TokenBudgetBatches, TruncatingCollate, lower_priority, pick_device, store_lengths  # noqa: E402
from scripts.train_model import learning_rate  # noqa: E402
from src.dataset import AugmentConfig, MixedTrainDataset, ScoreSegmentDataset  # noqa: E402
from src.eval.metrics import Accumulator  # noqa: E402
from src.labeller import PRESETS, build_labeller, head_counts, labeller_loss, predict_windows  # noqa: E402
from src.notelabels import score_notes  # noqa: E402
from src.preprocessing.noise import NoiseModel  # noqa: E402

D = Path("data/processed")
SHOW = ("onset_f1", "onset_shift_f1", "duration_acc", "staff_acc", "bar_f1", "note_f1")


def has_split(d: Path, name: str) -> bool:
    return (d / name / "index.npy").exists() and len(np.load(d / name / "index.npy", mmap_mode="r")) > 0


class Validator:
    """Fixed validation windows: losses / head accuracies, and rendered-token metrics."""

    def __init__(self, data: Path, vocab_path: str, ctx: dict, n_loss: int, n_decode: int, coll, seed: int = 0):
        self.sets, self.coll = {}, coll
        rng = np.random.default_rng(seed)
        for name, split in (("val", "val"), ("val_real", "val_transcribed")):
            if not has_split(data, split):
                continue
            ds = ScoreSegmentDataset(str(data / split), vocab_path, notelabels=True, **ctx)
            idx = np.arange(len(ds))
            loss_idx = np.sort(rng.choice(idx, min(n_loss, len(idx)), replace=False))
            dec_idx = np.sort(rng.choice(idx, min(n_decode, len(idx)), replace=False))
            self.sets[name] = (ds, loss_idx, dec_idx)
        self.space = next(iter(self.sets.values()))[0].label_space
        print("validation windows:", {k: (len(v[1]), len(v[2])) for k, v in self.sets.items()}, flush=True)

    @torch.no_grad()
    def run(self, model, device, amp, batch_size) -> dict:
        model.eval()
        res, losses = {}, []
        for name, (ds, loss_idx, dec_idx) in self.sets.items():
            tot, n, counts = 0.0, 0, {}
            for lo in range(0, len(loss_idx), batch_size):
                b = self.coll([ds[int(i)] for i in loss_idx[lo:lo + batch_size]])
                b = {k: v.to(device) for k, v in b.items()}
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    out = model(b["notes"], b["note_mask"])
                loss, _ = labeller_loss({k: v.float() for k, v in out.items()}, b["labels"], b["flags"])
                tot, n = tot + float(loss) * len(b["notes"]), n + len(b["notes"])
                for k, v in head_counts(out, b["labels"]).items():
                    counts[k] = counts.get(k, 0) + v
            res[f"{name}_loss"] = tot / max(n, 1)
            losses.append(res[f"{name}_loss"])
            for h in ("keep", "delta", "pos", "sub", "dur", "staff"):
                res[f"{name}_head_{h}_acc"] = counts[f"{h}_ok"] / max(counts[f"{h}_n"], 1)
            items = [ds[int(i)] for i in dec_idx]
            gts = [ds.vocab.decode(it["target"][it["prompt_len"]:-1].tolist()) for it in items]
            preds = predict_windows(model, items, [score_notes(g)[1] for g in gts], self.coll, device, amp,
                                    self.space, batch_size)
            acc = Accumulator()
            for g, p in zip(gts, preds):
                acc.add(g, p)
            r = acc.result()
            for k in SHOW:
                res[f"{name}_{k}"] = r[k]
        res["val_loss"] = float(np.mean(losses))
        model.train()
        return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", default="base", choices=list(PRESETS))
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=30)
    ap.add_argument("--patience", type=float, default=4, help="epochs without a better val loss (0 = off)")
    ap.add_argument("--cooldown", type=float, default=1)
    ap.add_argument("--batch-notes", type=int, default=16384, help="input notes per batch (context included)")
    ap.add_argument("--batch-size", type=int, default=32, help="evaluation batch size")
    ap.add_argument("--max-notes", type=int, default=1024)
    ap.add_argument("--context-before", type=int, default=64)
    ap.add_argument("--context-after", type=int, default=32)
    ap.add_argument("--noise", default="configs/noise_stats.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-loss-windows", type=int, default=400)
    ap.add_argument("--eval-decode-windows", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--normal-priority", action="store_true")
    ap.add_argument("--data", default=str(D))
    ap.add_argument("--runs-dir", default="outputs/runs")
    args = ap.parse_args(argv)
    if not args.normal_priority:
        lower_priority()
    torch.manual_seed(args.seed)
    data = Path(args.data)
    device = pick_device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    run = Path(args.runs_dir) / args.name
    run.mkdir(parents=True, exist_ok=True)
    vocab_path = str(data / "vocab.json")
    ctx = {"context_before": args.context_before, "context_after": args.context_after}

    # ---- data
    nm = NoiseModel.from_json(args.noise)
    if has_split(data, "train_transcribed"):
        train = MixedTrainDataset(str(data / "train"), str(data / "train_transcribed"), vocab_path, noise=nm,
                                  augment=AugmentConfig(), real_fraction=0.0, notelabels=True, **ctx)
        train.real_fraction = len(train.real) / len(train)
        sampler = train.sampler()
        a, b_ = store_lengths(train.synthetic.store, args.max_notes, 10 ** 9)
        c, d = store_lengths(train.real.store, args.max_notes, 10 ** 9)
        note_len = np.concatenate([b_, d])
        space = train.synthetic.label_space
    else:
        print(f"note: no {data / 'train_transcribed'}, training on `train` only", flush=True)
        train = ScoreSegmentDataset(str(data / "train"), vocab_path, noise=nm, augment=AugmentConfig(),
                                    notelabels=True, **ctx)
        sampler = train.sampler()
        _, note_len = store_lengths(train.store, args.max_notes, 10 ** 9)
        space = train.label_space
    note_len = np.minimum(note_len + args.context_before + args.context_after, args.max_notes)
    coll = TruncatingCollate(args.max_notes, 10 ** 9)
    batches = TokenBudgetBatches(sampler, note_len, note_len, args.batch_notes, args.batch_notes, seed=args.seed)
    dl = torch.utils.data.DataLoader(train, batch_sampler=batches, collate_fn=coll, num_workers=args.workers,
                                     pin_memory=device.type == "cuda", prefetch_factor=4 if args.workers else None)
    steps_per_epoch = max(1, len(dl))
    total_steps = int(args.epochs * steps_per_epoch)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    # ---- model
    first = train[0] if not hasattr(train, "synthetic") else train.synthetic[0]
    cfg = dict(n_sub=space.n_sub, n_dur=space.n_dur, n_features=int(first["notes"].shape[1]), dropout=args.dropout,
               max_notes=args.max_notes, **ctx, **PRESETS[args.size])
    stats = data / "feature_stats.json"
    if stats.exists():
        st = json.loads(stats.read_text())
        assert st["index"] == [1, 2, 3, 4], st["index"]
        cfg.update(feat_mean=st["mean"], feat_std=st["std"])
    ck = None
    if args.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        cfg = ck["config"]
    elif (run / "last.pt").exists():
        raise SystemExit(f"{run} already has a checkpoint: use --resume, or a different --name")
    model = build_labeller(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), fused=device.type == "cuda")
    phase, cool = "train", {"start": 0, "steps": 0, "lr0": 0.0}

    def lr_at(step):
        return learning_rate(step, lr=args.lr, warmup=args.warmup, total_steps=total_steps,
                             min_lr_frac=args.min_lr_frac, phase=phase, cool=cool)

    step, epoch, best, best_step, stopped_early = 0, 0, float("inf"), 0, False
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, epoch, best, best_step = ck["step"], ck["epoch"], ck["best"], ck["best_step"]
        phase, cool, stopped_early = ck["phase"], ck["cool"], ck.get("stopped_early", False)
        print(f"resumed {run} at step {step} (epoch {epoch}, phase {phase})", flush=True)
    json.dump({"args": vars(args), "arch": "labeller", "model": cfg, "params": n_params,
               "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
               "data": {"train_windows": len(train), "train_transcribed": has_split(data, "train_transcribed"),
                        "grid": "true (score beats timed through the note alignment)"}},
              open(run / "config.json", "w"), indent=1)
    print(f"labeller: {n_params / 1e6:.1f}M parameters on {device} | {len(train)} training windows, "
          f"{steps_per_epoch} steps/epoch, {total_steps} steps total", flush=True)
    validator = Validator(data, vocab_path, ctx, args.eval_loss_windows, args.eval_decode_windows, coll, args.seed)

    def write_status():
        tmp = run / "status.json.tmp"
        tmp.write_text(json.dumps({"phase": phase, "step": step, "total_steps": total_steps, "epoch": epoch,
                                   "best": best, "best_step": best_step, "stopped_early": stopped_early}))
        os.replace(tmp, run / "status.json")

    def save(path):
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": epoch,
                    "best": best, "best_step": best_step, "phase": phase, "cool": cool,
                    "stopped_early": stopped_early, "arch": "labeller", "config": cfg}, tmp)
        os.replace(tmp, path)
        if path.name == "last.pt":
            write_status()

    patience_steps = max(1, int(args.patience * steps_per_epoch)) if args.patience > 0 else 0

    def end_of_training():
        nonlocal phase, stopped_early
        if phase == "train" and step >= total_steps:
            phase = "done"
        elif phase == "train" and patience_steps and step - best_step >= patience_steps:
            stopped_early = True
            print(f"early stop at step {step}: no better validation loss for {args.patience:g} epochs", flush=True)
            n = max(1, int(args.cooldown * steps_per_epoch)) if args.cooldown > 0 else 0
            if not n:
                phase = "done"
                return
            ckb = torch.load(run / "best.pt", map_location=device, weights_only=False)
            model.load_state_dict(ckb["model"])
            opt.load_state_dict(ckb["opt"])
            cool.update(start=step, steps=n, lr0=lr_at(best_step))
            phase = "cooldown"
        elif phase == "cooldown" and step >= cool["start"] + cool["steps"]:
            phase = "done"

    write_status()
    if phase == "done":
        print(f"{run} already finished (best {best:.4f} at step {best_step}); nothing to train", flush=True)
    log_path = run / "log.csv"
    model.train()
    t_int, run_loss, n_loss, parts_sum = time.time(), 0.0, 0, {}
    try:
        while phase != "done":
            if hasattr(train, "set_epoch"):
                train.set_epoch(epoch)
            batches.set_epoch(epoch)
            for b in dl:
                b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    out = model(b["notes"], b["note_mask"])
                loss, parts = labeller_loss(out, b["labels"], b["flags"])
                loss.backward()
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                run_loss, n_loss = run_loss + float(loss.detach()), n_loss + 1
                for k, v in parts.items():
                    parts_sum[k] = parts_sum.get(k, 0.0) + v
                if step % 100 == 0:
                    print(f"step {step}/{total_steps} ep {epoch} loss {run_loss / n_loss:.4f} ("
                          + " ".join(f"{k} {v / n_loss:.3f}" for k, v in parts_sum.items())
                          + f") lr {lr_at(step):.2e} | {time.time() - t_int:.0f} s", flush=True)
                    t_int, run_loss, n_loss, parts_sum = time.time(), 0.0, 0, {}
                cool_end = phase == "cooldown" and step >= cool["start"] + cool["steps"]
                if step % args.eval_every == 0 or step == total_steps or cool_end:
                    res = validator.run(model, device, amp, args.batch_size)
                    res.update(step=step, epoch=epoch, lr=lr_at(step))
                    new = not log_path.exists()
                    with open(log_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(res))
                        if new:
                            w.writeheader()
                        w.writerow(res)
                    print("  eval " + " | ".join(f"{k}={v:.3f}" for k, v in res.items()
                                                if isinstance(v, float) and k not in ("lr",)), flush=True)
                    if res["val_loss"] < best:
                        best, best_step = res["val_loss"], step
                        save(run / "best.pt")
                    end_of_training()
                    save(run / "last.pt")
                if phase == "done":
                    break
            else:
                epoch += 1
    except KeyboardInterrupt:
        save(run / "last.pt")
        print(f"\nstopped at step {step}: python scripts/train_labeller.py --name {args.name} --resume", flush=True)
        sys.exit(130)
    save(run / "last.pt")
    print(f"done: {step} steps. best mean val loss {best:.4f} at step {best_step} -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
