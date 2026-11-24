"""Train the Scoreformer (src/scoreformer.py), or Score2Seq (src/model.py) under the same loop.

    python scripts/train_model.py --name sf_base                       # Scoreformer, base size (~45M)
    python scripts/train_model.py --name sf_base --resume              # continue after Ctrl+C / crash
    python scripts/train_model.py --name sf_clean --no-noise           # clean-input ablation
    python scripts/train_model.py --name s2s --arch score2seq          # Michael's model, same loop
    python scripts/train_model.py --name s2s_ctx --arch score2seq --context-noise 0.5   # imperfect context

Same data, batching and noise as scripts/train.py (it reuses that script's helpers), so the only
difference between two runs is what you change on the command line. Differences from train.py:
  * ``--arch`` / ``--size``: architecture and size preset;
  * splits that don't exist yet (train_transcribed, val_transcribed, PERiScoPe) are skipped
    instead of crashing, so it runs on an ASAP-only build too;
  * best.pt = lowest mean validation loss over all available validation sets (ASAP clean,
    PERiScoPe real, ASAP real transcriptions), instead of the small val_real set alone;
  * ``--compile`` (torch.compile) and ``--grad-accum`` for big GPUs / big models;
  * ``--aux-metrical``: an auxiliary head on the encoder predicts each note's downbeat and beat
    phase (src/metrical.py), weighted by ``--aux-weight``, so the encoder learns where the bar
    is; validation logs its downbeat F1 and phase accuracy. Build the labels once per data build
    with scripts/build_metrical.py (otherwise they are aligned on the fly, more slowly);
  * ``--context-before`` / ``--context-after``: the encoder also sees that many performed notes
    before / after each window (src/dataset.py), so it can read the meter around the window;
    recorded in the checkpoint, which evaluation and the demo follow;
  * ``--beats``: every note also gets 7 beat features (src/beats.py) from the performance's beats:
    the ground truth (scripts/build_beats.py) corrupted like a real beat tracker (``--beat-noise``,
    fitted by scripts/fit_beat_noise.py), and on the real transcriptions the tracker's own beats
    (scripts/track_beats.py) when they exist. ``--beat-dropout`` of the windows get no beats, so the
    model also works without a tracker (MIDI input). Validation uses the same fitted noise with a
    fixed seed. Recorded in the checkpoint (``beat_features``);
  * early stopping with a cooldown. ``--epochs`` (default 20) is the maximum and the length of the
    cosine learning-rate schedule. When the mean validation loss has not improved for
    ``--patience`` epochs (default 5; 0 = off), training goes back to the best checkpoint (weights
    and optimizer) and runs a ``--cooldown`` (default 1 epoch): the learning rate decays linearly
    from its value at the best step to the schedule's minimum, the low-learning-rate phase an
    early-stopped run would otherwise never get. best.pt keeps improving if the cooldown helps.
    The phase (train / cooldown / done) is saved in every checkpoint, so a resubmitted job picks up
    a cooldown where it was and a finished run exits at once.

Outputs in outputs/runs/<name>/: config.json, log.csv (one row per eval), last.pt, best.pt, and
status.json (phase, step, best: the same state as last.pt, readable without torch; written from
the start, so tools can tell a finished run from one still training).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import (SHOW, TokenBudgetBatches, TruncatingCollate, lower_priority,  # noqa: E402
                           pick_device, store_lengths)
from src.beats import BeatNoise, beats_path  # noqa: E402
from src.dataset import AugmentConfig, ContextNoise, MixedTrainDataset, ScoreSegmentDataset, n_features  # noqa: E402
from src.eval.metrics import Accumulator  # noqa: E402
from src.metrical import aux_counts, aux_loss, aux_scores  # noqa: E402
from src.model import token_loss  # noqa: E402
from src.preprocessing.noise import NoiseModel  # noqa: E402
from src.preprocessing.vocab import EOS, pin_vocab  # noqa: E402
from src.scoreformer import PRESETS, build_model  # noqa: E402

D = Path("data/processed")


def learning_rate(step: int, *, lr: float, warmup: int, total_steps: int, min_lr_frac: float,
                  phase: str = "train", cool: dict | None = None) -> float:
    """Linear warm-up, then cosine decay to ``lr * min_lr_frac`` at ``total_steps``. In the cooldown
    after an early stop (``phase == "cooldown"``), linear decay from ``cool["lr0"]`` (the rate at the
    best step) to the same minimum over ``cool["steps"]`` steps from ``cool["start"]``."""
    lr_min = lr * min_lr_frac
    if phase == "cooldown":
        prog = min(1.0, max(0.0, (step - cool["start"]) / max(1, cool["steps"])))
        return cool["lr0"] + (lr_min - cool["lr0"]) * prog
    if step < warmup:
        return lr * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return lr_min + (lr - lr_min) * 0.5 * (1 + math.cos(math.pi * prog))


def has_split(name: str) -> bool:
    return (D / name / "index.npy").exists() and len(np.load(D / name / "index.npy", mmap_mode="r")) > 0


class Evaluator:
    """Fixed validation subsets: teacher-forced loss / token accuracy + greedy-decoding metrics
    (ground-truth prompt, as in scripts/train.py). Sets with no data are left out."""

    def __init__(self, vocab_path, n_loss: int, n_decode: int, coll, seed: int = 0, beats=None,
                 real_beats=None, **ctx):
        """``beats`` / ``real_beats``: beat arguments of ScoreSegmentDataset for val / val_transcribed."""
        rng = np.random.default_rng(seed)
        self.coll = coll
        self.sets = {}

        def sample(idx, n):
            idx = list(idx)
            return sorted(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())

        val = ScoreSegmentDataset(str(D / "val"), vocab_path, **ctx, **(beats or {}))
        self.vocab = val.vocab
        for name, ds_name in (("val_asap", "asap"), ("val_periscope", "pianocore")):
            idx = [i for i, m in enumerate(val.store.meta) if m.get("dataset", "asap") == ds_name]
            if idx:
                self.sets[name] = (val, sample(idx, n_loss), sample(idx, n_decode))
        if has_split("val_transcribed"):
            real = ScoreSegmentDataset(str(D / "val_transcribed"), vocab_path, **ctx,
                                       **(real_beats if real_beats is not None else beats or {}))
            self.sets["val_real"] = (real, sample(range(len(real)), n_loss), sample(range(len(real)), n_decode))
        print("validation sets:", {k: len(v[1]) for k, v in self.sets.items()}, flush=True)

    @torch.no_grad()
    def run(self, model, device, amp, batch_size: int, weights) -> dict:
        model.eval()
        res, eos = {}, self.vocab.stoi[EOS]
        for name, (ds, loss_idx, dec_idx) in self.sets.items():
            tot, n, correct, count = 0.0, 0, 0, 0
            aux_c = {}
            for k in range(0, len(loss_idx), batch_size):
                chunk = loss_idx[k:k + batch_size]
                b = {kk: v.to(device) for kk, v in self.coll([ds[i] for i in chunk]).items()}
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits, aux = model(b["notes"], b["note_mask"], b["target"][:, :-1], return_aux=True)
                if aux is not None and "downbeat" in b:
                    for kk, v in aux_counts(aux, b["downbeat"], b["phase"]).items():
                        aux_c[kk] = aux_c.get(kk, 0) + v
                # unweighted CE for model selection (comparable across --ornament-weight settings)
                tot += token_loss(logits, b["target"], b["loss_mask"]).item() * len(chunk)
                n += len(chunk)
                m = b["loss_mask"][:, 1:]
                correct += ((logits.argmax(-1) == b["target"][:, 1:]) & m).sum().item()
                count += m.sum().item()
            res[f"{name}_loss"] = tot / max(n, 1)
            res[f"{name}_tok_acc"] = correct / max(count, 1)
            if aux_c:
                res.update({f"{name}_{kk}": v for kk, v in aux_scores(aux_c).items()})
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
    ap.add_argument("--epochs", type=float, default=20, help="maximum; also the length of the LR schedule")
    ap.add_argument("--patience", type=float, default=5,
                    help="stop when the mean validation loss has not improved for this many epochs (0 = off)")
    ap.add_argument("--cooldown", type=float, default=1,
                    help="after an early stop: epochs of linear LR decay from the best checkpoint (0 = stop at once)")
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
    ap.add_argument("--aux-metrical", action="store_true",
                    help="auxiliary head predicting each note's downbeat / beat phase (src/metrical.py)")
    ap.add_argument("--aux-weight", type=float, default=0.5, help="weight of the auxiliary loss")
    ap.add_argument("--context-before", type=int, default=0,
                    help="performed notes before each window the encoder also sees (0 = none)")
    ap.add_argument("--context-after", type=int, default=0,
                    help="performed notes after each window the encoder also sees (0 = none)")
    ap.add_argument("--context-noise", type=float, default=0.0,
                    help="fraction of training windows whose context (prompt) is corrupted with the kind of "
                         "errors a model's own previous output contains (src.dataset.ContextNoise); 0 = off")
    ap.add_argument("--beats", action="store_true",
                    help="beat features per note (src/beats.py): ground truth + tracker-like noise in training")
    ap.add_argument("--beat-noise", default="configs/beat_noise.json",
                    help="tracker errors to simulate (scripts/fit_beat_noise.py); 'none' = exact beats")
    ap.add_argument("--beat-dropout", type=float, default=0.15,
                    help="share of training windows without beats (input without a tracker)")
    ap.add_argument("--transcribed-beats", default="tracked", choices=["tracked", "gt"],
                    help="beats of the real-transcription windows: the tracker's own (beats_tracked.npz), or the "
                         "ground truth with --beat-noise like the clean data (with --beat-noise none: exact "
                         "beats everywhere, the ceiling of what beats can give)")
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
    pin_vocab(vocab_path, run)  # the run keeps the vocabulary it trains with (demo, pipelines)

    # ---- data (same recipe as scripts/train.py)
    real_ok = has_split("train_transcribed")
    ctx = ContextNoise(prob=args.context_noise) if args.context_noise > 0 else None
    # context notes: a resumed run keeps the inputs it was started with (config.json)
    notes_ctx = {"context_before": args.context_before, "context_after": args.context_after}
    aux_on, beats_on = args.aux_metrical, args.beats
    recorded_noise = None  # a resumed beat run: the tracker errors it started with, as values
    if args.resume and (run / "config.json").exists():
        rec = json.loads((run / "config.json").read_text())
        started = rec.get("model", {})
        notes_ctx = {k: int(started.get(k, 0)) for k in notes_ctx}
        aux_on = bool(started.get("aux_metrical", False))
        beats_on = bool(started.get("beat_features", False))
        beat_rec = (rec.get("data") or {}).get("beats") or {}
        if beats_on and beat_rec:  # the same tracker noise and dropout as when it started
            args.beat_noise, args.beat_dropout = beat_rec.get("noise", args.beat_noise), beat_rec.get("dropout", args.beat_dropout)
            recorded_noise = beat_rec.get("noise_values")
            if beat_rec.get("train_transcribed") in ("tracked", "gt"):  # kept even if the file is gone now
                args.transcribed_beats = beat_rec["train_transcribed"]
    n_ctx = notes_ctx["context_before"] + notes_ctx["context_after"]
    notes_ctx_m = {**notes_ctx, "metrical": aux_on}  # dataset arguments
    # beats: ground truth + fitted tracker noise; the real transcriptions use the tracker's own beats
    beat_args = real_beat_args = val_beats = val_real_beats = None
    beat_record = None
    if beats_on:
        if args.beat_noise == "none":
            bn = None
        elif recorded_noise is not None:  # not the file, which may have been refitted since
            bn = BeatNoise.from_dict(recorded_noise)
        elif Path(args.beat_noise).exists():
            bn = BeatNoise.from_json(args.beat_noise)
        else:
            raise SystemExit(f"{args.beat_noise} not found: fit it with scripts/fit_beat_noise.py "
                             f"(pace/beats.sh does), or pass --beat-noise none for exact beats")
        src = ("exact beats" if bn is None else "the tracker errors recorded when this run started"
               if recorded_noise is not None else f"tracker errors from {args.beat_noise}")
        print(f"beats: {src}" + ("" if bn is None else f" (fitted: {bn.fitted}; p_double {bn.p_double:.3f}, "
                                 f"miss {bn.miss:.3f}, down_hit {bn.down_hit:.3f})"), flush=True)
        gt_splits = ["train", "val"] + ([s for s in ("train_transcribed", "val_transcribed") if has_split(s)]
                                        if args.transcribed_beats == "gt" else [])
        for split in gt_splits:
            if not beats_path(D / split, "gt").exists():
                raise SystemExit(f"{beats_path(D / split, 'gt')} missing: run scripts/build_beats.py first")

        def split_beats(split, dropout):
            """Tracked beats where they exist (real errors already), else ground truth + noise;
            the ground truth + noise always with --transcribed-beats gt."""
            if args.transcribed_beats == "tracked" and beats_path(D / split, "tracked").exists():
                return {"beats": "tracked", "beat_noise": None, "beat_dropout": dropout}
            return {"beats": "gt", "beat_noise": bn, "beat_dropout": dropout}
        beat_args = {"beats": "gt", "beat_noise": bn, "beat_dropout": args.beat_dropout}
        real_beat_args = split_beats("train_transcribed", args.beat_dropout)
        val_beats = {"beats": "gt", "beat_noise": bn}
        val_real_beats = split_beats("val_transcribed", 0.0)
        beat_record = {"noise": args.beat_noise, "fitted": bool(bn is not None and bn.fitted),
                       "noise_values": bn.to_dict() if bn is not None else None,
                       "dropout": args.beat_dropout, "train_transcribed": real_beat_args["beats"],
                       "val_transcribed": val_real_beats["beats"]}
    if args.no_noise:
        train = ScoreSegmentDataset(str(D / "train"), vocab_path, augment=AugmentConfig(), context_noise=ctx,
                                    **notes_ctx_m, **(beat_args or {}))
        clean_idx = [i for i, m in enumerate(train.store.meta) if m.get("source", "clean") == "clean"]
        w = np.zeros(len(train))
        w[clean_idx] = train.weights[clean_idx]
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                                         num_samples=len(clean_idx), replacement=True)
        tok_len, note_len = store_lengths(train.store, args.max_notes, args.max_tokens)
    elif real_ok:
        nm = NoiseModel.from_json(args.noise)
        train = MixedTrainDataset(str(D / "train"), str(D / "train_transcribed"), vocab_path, noise=nm,
                                  augment=AugmentConfig(), real_fraction=0.0, context_noise=ctx, **notes_ctx_m,
                                  beats=beat_args, real_beats=real_beat_args)
        train.real_fraction = len(train.real) / len(train)  # natural proportion
        sampler = train.sampler()
        a, b = store_lengths(train.synthetic.store, args.max_notes, args.max_tokens)
        c, d = store_lengths(train.real.store, args.max_notes, args.max_tokens)
        tok_len, note_len = np.concatenate([a, c]), np.concatenate([b, d])
    else:
        print("note: no data/processed/train_transcribed, training on `train` only", flush=True)
        train = ScoreSegmentDataset(str(D / "train"), vocab_path, noise=NoiseModel.from_json(args.noise),
                                    augment=AugmentConfig(), context_noise=ctx, **notes_ctx_m, **(beat_args or {}))
        sampler = train.sampler()
        tok_len, note_len = store_lengths(train.store, args.max_notes, args.max_tokens)
    vocab = train.vocab
    note_len = np.minimum(note_len + n_ctx, args.max_notes)  # batch budget includes the context notes
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
    if n_ctx:
        cfg.update(**notes_ctx)
    if n_ctx or beats_on:
        cfg["n_features"] = n_features(bool(n_ctx), beats_on)
    if beats_on:
        cfg["beat_features"] = True
    if aux_on:
        cfg["aux_metrical"] = True
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

    # phase: "train" (cosine schedule) -> "cooldown" (after an early stop) -> "done"
    phase, cool = "train", {"start": 0, "steps": 0, "lr0": 0.0}
    lr_min = args.lr * args.min_lr_frac

    def lr_at(step):
        return learning_rate(step, lr=args.lr, warmup=args.warmup, total_steps=total_steps,
                             min_lr_frac=args.min_lr_frac, phase=phase, cool=cool)

    weights = torch.tensor(vocab.loss_weights(args.ornament_weight), device=device)
    step, epoch, best = 0, 0, float("inf")
    in_epoch = 0
    best_step = 0  # step of the best validation loss (early stopping)
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, epoch, best = ck["step"], ck["epoch"], ck.get("best", best)
        in_epoch = ck.get("batch_in_epoch", 0)  # batches of this epoch trained on already
        batches.skip = in_epoch
        best_step = ck.get("best_step", step)
        phase = ck.get("phase", "train" if step < total_steps else "done")
        cool = ck.get("cool", cool)
        print(f"resumed {run} at step {step} (epoch {epoch}, phase {phase})", flush=True)
    elif (run / "last.pt").exists():
        raise SystemExit(f"{run} already has a checkpoint: use --resume, or a different --name")
    # which data this run trained on: runs started before / after the real-transcription splits
    # were built (pace/real_asap.sbatch) are not trained on the same data
    data = {"train_windows": len(train), "train_transcribed": bool(real_ok and not args.no_noise),
            "val_transcribed": has_split("val_transcribed"), "beats": beat_record,
            "train_periscope": any(m.get("dataset") == "pianocore" for m in
                                   (train.synthetic if hasattr(train, "synthetic") else train).store.meta)}
    if ck is not None and (run / "config.json").exists():  # a resumed run keeps its original record
        data = json.loads((run / "config.json").read_text()).get("data", data)
    json.dump({"args": vars(args), "arch": args.arch, "model": cfg, "params": n_params,
               "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "data": data},
              open(run / "config.json", "w"), indent=1)
    print(f"{args.arch}: {n_params / 1e6:.1f}M parameters on {device} | {len(train)} training windows, "
          f"{steps_per_epoch} steps/epoch, {total_steps} steps total | Ctrl+C saves and stops", flush=True)

    evaluator = Evaluator(vocab_path, args.eval_loss_windows, args.eval_decode_windows, coll, beats=val_beats,
                          real_beats=val_real_beats, **notes_ctx_m)

    def save(path):
        # atomic: a job killed mid-save (PACE's time limit) keeps the previous complete checkpoint
        # instead of a truncated file that the resubmitted job cannot load
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": epoch,
                    "batch_in_epoch": in_epoch,
                    "best": best, "best_step": best_step, "phase": phase, "cool": cool,
                    "stopped_early": stopped_early,
                    "arch": args.arch, "config": cfg}, tmp)
        os.replace(tmp, path)
        if path.name == "last.pt":
            write_status()

    def write_status():
        """status.json: the training phase, readable without torch (pace/reevaluate.sh and
        src/eval/plan.py leave a run alone until it says "done"). Written atomically."""
        tmp = run / "status.json.tmp"
        tmp.write_text(json.dumps({"phase": phase, "step": step, "total_steps": total_steps, "epoch": epoch,
                                   "best": best, "best_step": best_step, "stopped_early": stopped_early}))
        os.replace(tmp, run / "status.json")

    log_path = run / "log.csv"
    model.train()
    t0 = t_int = time.time()
    tok_int, wait_int, run_loss, n_loss, micro = 0, 0.0, 0.0, 0, 0
    patience_steps = max(1, int(args.patience * steps_per_epoch)) if args.patience > 0 else 0
    stopped_early = bool(ck.get("stopped_early", phase == "cooldown")) if ck is not None else False
    write_status()  # from the start: a run in training has a best.pt long before it is finished

    def end_of_training():
        """After an eval: move train -> cooldown / done, or cooldown -> done."""
        nonlocal phase, stopped_early
        if phase == "train" and step >= total_steps:
            phase = "done"
        elif phase == "train" and patience_steps and step - best_step >= patience_steps:
            stopped_early = True
            print(f"early stop at step {step} (epoch {epoch}): mean validation loss has not improved for "
                  f"{args.patience:g} epochs (best {best:.4f} at step {best_step})", flush=True)
            n = max(1, int(args.cooldown * steps_per_epoch)) if args.cooldown > 0 else 0
            if n <= 0:
                phase = "done"
                return
            ck_best = torch.load(run / "best.pt", map_location=device, weights_only=False)
            model.load_state_dict(ck_best["model"])
            opt.load_state_dict(ck_best["opt"])
            cool.update(start=step, steps=n, lr0=lr_at(best_step))
            phase = "cooldown"
            print(f"cooldown: back to the best checkpoint (step {best_step}); learning rate "
                  f"{cool['lr0']:.2e} -> {lr_min:.2e} over {n} steps ({args.cooldown:g} epochs)", flush=True)
        elif phase == "cooldown" and step >= cool["start"] + cool["steps"]:
            phase = "done"
            print(f"cooldown finished at step {step}", flush=True)

    if phase == "done":
        print(f"{run} already finished (step {step}, best {best:.4f} at step {best_step}); nothing to train",
              flush=True)
    try:
        while phase != "done":
            if hasattr(train, "set_epoch"):
                train.set_epoch(epoch)
            batches.set_epoch(epoch)
            t_wait = time.time()
            for b in dl:
                in_epoch += 1
                wait_int += time.time() - t_wait
                b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits, aux = fwd(b["notes"], b["note_mask"], b["target"][:, :-1], return_aux=True)
                loss = token_loss(logits, b["target"], b["loss_mask"], weights, args.label_smoothing)
                if aux is not None:
                    loss = loss + args.aux_weight * aux_loss(aux, b["downbeat"], b["phase"])
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
                cool_end = phase == "cooldown" and step >= cool["start"] + cool["steps"]
                if step % args.eval_every == 0 or step == total_steps or cool_end:
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
                        best, best_step = res["val_loss"], step
                        save(run / "best.pt")
                    end_of_training()
                    save(run / "last.pt")
                    t_int, tok_int, wait_int = time.time(), 0, 0.0
                if phase == "done":
                    break
                t_wait = time.time()
            else:  # the epoch's batches ran out (not stopped by total_steps or early stopping)
                epoch += 1
                in_epoch = 0
    except KeyboardInterrupt:
        save(run / "last.pt")
        print(f"\nstopped at step {step}: saved {run / 'last.pt'}. Continue with: "
              f"python scripts/train_model.py --name {args.name} --resume", flush=True)
        sys.exit(130)
    save(run / "last.pt")
    how = " (early stop + cooldown)" if stopped_early and cool["steps"] else " (early stop)" if stopped_early else ""
    print(f"done: {step} steps{how}. best mean val loss {best:.4f} at step {best_step} -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
