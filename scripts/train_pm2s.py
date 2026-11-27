"""Train PM2S's beat tracker (src/pm2s_beats.py, RNNJointBeatModel) from scratch on our training
splits only, so it can be tested on Beyer & Dai's ASAP test pieces without having seen them.

    python scripts/train_pm2s.py --name pm2s_ho          # -> outputs/runs/pm2s_ho/best.pt
    PM2S_WEIGHTS=outputs/runs/pm2s_ho/best.pt PM2S_TAG=ho python scripts/system_scores.py ...

Why: PM2S's released weights were trained on all of ASAP (its ``--full_train``), our 14 test pieces
included, and with them v2 gained the most of any change measured (sixth round: -2.10 MUSTER on
clean MIDI, -0.74 on real recordings), too much to quote while the tracker has seen the test set.
``best.pt`` is a plain state dict, so everything that runs PM2S takes it through $PM2S_WEIGHTS, and
$PM2S_TAG names what is made with it (src/pm2s_beats.py weights_tag).

Data: the meter model's performances (data/processed/meter, scripts/build_meter_data.py), each with
its notes and its ground-truth beats and downbeats, by source:
  asap        train.pkl: ASAP's clean performance MIDI of our training pieces, annotated beats
  real        train_transcribed.pkl: Transkun's transcriptions of those performances
  periscope   train_periscope.pkl: PERiScoPe's Transkun transcriptions, beats from the score alignment
  gangqinpu   train_gangqinpu.pkl: Gangqinpu's, likewise (pop, folk, jazz, classical)
Validation: the same sources' val splits (a fixed draw of --val-n for the large two). Never the
test splits. Each performance (or trusted segment of one) keeps the notes between its first and
last beat, so every note has a label.

Recipe: PM2S's own (its dev/ code, read 2026-11-25), except where said.
- Targets per note: beat = 1 when a beat is within TOLERANCE (0.07 s) of its onset, every such note
  (not only the nearest); downbeat alike; tempo = the inter-beat interval around the note in 10 ms
  classes (0 = none, beyond 4 s, or after the last beat: not trained on).
- Segments of 500 notes from a random start (here never shorter than the performance allows),
  32 per batch.
- Augmentation, every sample: tempo x U(0.8, 1.2); a transposition of up to 12 semitones (kept on
  the keyboard); then either extra notes (octave copies of a random up to 30% of the notes) or
  missing ones (up to 30% of the notes that sound with the next one).
- Loss: binary cross-entropy for beats and downbeats, NLL for the tempo classes, weights 1. Padding
  is masked (PM2S counts its zeros in the mean).
- AdamW (lr 1e-3, betas 0.8 / 0.8, eps 1e-4, weight decay 1e-2), the rate x0.1 after
  --lr-step steps (PM2S: after 50 epochs of 800 steps). On our data this is unstable: pm2s_ho's
  training loss rose from step 8,000 to 38,000 and its validation fell with it (best at 10,000;
  docs/experiments.md, "Seventh round: results"). The stable settings (pm2s_ho2 / pm2s_ho3,
  pace/v2_pm2s2.sh):
      --lr 3e-4 --betas 0.9,0.999 --eps 1e-8 --clip 1.0 --schedule cosine --warmup 1000
  (the gradient norm before clipping is logged: "grad_norm").
- Ours, not PM2S's:
  - the sources and their mix: Gangqinpu --gangqinpu-share of the samples, the others by their size;
  - the selection: best.pt keeps the step with the best mean of beat F and downbeat F of the
    post-processed beats (src/pm2s_beats.py postprocess, as at test), averaged over the validation
    sources. PM2S selects on the per-note beat F.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import lower_priority, pick_device  # noqa: E402
from src.beats import compare  # noqa: E402
from src.pm2s_beats import IBI_VOCAB, RESOLUTION, TOLERANCE, PM2SBeats, build_model  # noqa: E402

SOURCES = {"asap": "train", "real": "train_transcribed", "periscope": "train_periscope",
           "gangqinpu": "train_gangqinpu"}
VAL = {"asap": "val", "real": "val_transcribed", "periscope": "val_periscope", "gangqinpu": "val_gangqinpu"}
MAX_IBI = 4.0   # seconds: longer inter-beat intervals get no tempo target (PM2S)


def load(data: Path, name: str) -> list:
    f = Path(data) / "meter" / f"{name}.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else []


def example(perf: dict) -> dict | None:
    """A performance -> its notes between its first and last beat (sorted by onset; pitch, onset,
    offset, velocity) and its beat / downbeat times; None if too short to train on."""
    t = np.asarray(perf["gt"]["t"], np.float64)
    down = np.asarray(perf["gt"]["down"], bool)
    notes = np.asarray(perf["notes"], np.float64)
    if len(t) < 4 or len(notes) < 8:
        return None
    keep = (notes[:, 1] >= t[0] - TOLERANCE) & (notes[:, 1] <= t[-1] + TOLERANCE)
    notes = notes[keep]
    if len(notes) < 8:
        return None
    notes = notes[np.argsort(notes[:, 1], kind="stable")]
    return {"notes": notes.astype(np.float32), "t": t, "down": t[down], "down_flags": down,
            "key": perf.get("performance", "")}


def near(onsets: np.ndarray, times: np.ndarray) -> np.ndarray:
    """1.0 for each onset within TOLERANCE of one of ``times``."""
    out = np.zeros(len(onsets), np.float32)
    if not len(times) or not len(onsets):
        return out
    j = np.clip(np.searchsorted(times, onsets), 1, len(times) - 1) if len(times) > 1 else np.zeros(len(onsets), int)
    d = np.abs(onsets - times[j])
    if len(times) > 1:
        d = np.minimum(d, np.abs(onsets - times[j - 1]))
    out[d <= TOLERANCE + 1e-9] = 1.0
    return out


def targets(onsets: np.ndarray, beats: np.ndarray, downbeats: np.ndarray):
    """PM2S's per-note targets: beat, downbeat (0 / 1) and the tempo class (the inter-beat interval
    the note sits in, in RESOLUTION steps; 0 = none)."""
    yb, yd = near(onsets, beats), near(onsets, downbeats)
    yt = np.zeros(len(onsets), np.int64)
    if len(beats) >= 2:
        i = np.searchsorted(beats, onsets, side="right") - 1
        ok = (i >= 0) & (i < len(beats) - 1)
        gap = np.zeros(len(onsets))
        gap[ok] = beats[i[ok] + 1] - beats[i[ok]]
        q = np.round(gap / RESOLUTION).astype(np.int64)
        ok &= (gap <= MAX_IBI) & (q >= 1) & (q < IBI_VOCAB)
        yt[ok] = q[ok]
    return yb, yd, yt


def augment(ex: dict, rng: np.random.Generator) -> dict:
    """PM2S's augmentation: tempo, transposition, then extra or missing notes."""
    notes = ex["notes"].astype(np.float64).copy()
    r = rng.uniform(0.8, 1.2)
    notes[:, 1:3] /= r
    t, down = ex["t"] / r, ex["down"] / r
    lo, hi = max(-12, int(-notes[:, 0].min())), min(12, int(127 - notes[:, 0].max()))
    k = int(np.clip(round(rng.uniform(-12, 12)), lo, hi)) if lo <= hi else 0
    notes[:, 0] += k
    if rng.random() < 0.5:   # extra notes: octave copies of a random share of them
        copies = notes[rng.random(len(notes)) < rng.random() * 0.3].copy()
        if len(copies):
            copies[:, 0] += rng.choice([-12, 12], len(copies))
            copies = copies[(copies[:, 0] >= 0) & (copies[:, 0] <= 127)]
            notes = np.concatenate([notes, copies])
            notes = notes[np.argsort(notes[:, 1], kind="stable")]
    else:   # missing notes: some of those sounding with the next one (a chord's)
        chord = np.zeros(len(notes), bool)
        chord[:-1] = np.diff(notes[:, 1]) <= TOLERANCE
        drop = chord & (rng.random(len(notes)) < rng.random() * 0.3)
        if (~drop).sum() >= 8:
            notes = notes[~drop]
    return {**ex, "notes": notes.astype(np.float32), "t": t, "down": down}


def segment(ex: dict, length: int, rng: np.random.Generator):
    """A random window of ``length`` notes and its targets."""
    notes = ex["notes"]
    start = int(rng.integers(0, len(notes) - length + 1)) if len(notes) > length else 0
    seg = notes[start:start + length]
    yb, yd, yt = targets(seg[:, 1].astype(np.float64), ex["t"], ex["down"])
    return seg, yb, yd, yt


def model_input(notes: np.ndarray) -> np.ndarray:
    """Our [N, 4] notes (pitch, onset, offset, velocity) -> PM2S's (pitch, onset, duration, velocity)."""
    return np.stack([notes[:, 0], notes[:, 1], np.maximum(notes[:, 2] - notes[:, 1], 0.0), notes[:, 3]],
                    axis=1).astype(np.float32)


def batch(pool: dict, probs: np.ndarray, names: list, size: int, length: int, rng: np.random.Generator):
    """``size`` training samples: a source drawn by ``probs``, then one of its performances,
    augmented, cut to a window. Returns tensors x [B, L, 4], beat / downbeat / tempo targets, mask."""
    x = np.zeros((size, length, 4), np.float32)
    yb = np.zeros((size, length), np.float32)
    yd = np.zeros((size, length), np.float32)
    yt = np.zeros((size, length), np.int64)
    m = np.zeros((size, length), np.float32)
    for i in range(size):
        exs = pool[names[rng.choice(len(names), p=probs)]]
        seg, b, d, t = segment(augment(exs[int(rng.integers(len(exs)))], rng), length, rng)
        n = len(seg)
        x[i, :n], yb[i, :n], yd[i, :n], yt[i, :n], m[i, :n] = model_input(seg), b, d, t, 1.0
    return tuple(torch.from_numpy(a) for a in (x, yb, yd, yt, m))


def loss_of(model, x, yb, yd, yt, m):
    pb, pd, pt = model.forward_all(x)
    eps = 1e-6
    n = m.sum().clamp(min=1.0)
    lb = (F.binary_cross_entropy(pb.clamp(eps, 1 - eps), yb, reduction="none") * m).sum() / n
    ld = (F.binary_cross_entropy(pd.clamp(eps, 1 - eps), yd, reduction="none") * m).sum() / n
    lt = F.nll_loss(pt.transpose(1, 2), yt, ignore_index=0) if (yt > 0).any() else pt.sum() * 0.0
    return lb + ld + lt, {"beat": lb.item(), "downbeat": ld.item(), "tempo": lt.item()}


def validate(model, groups: dict, device) -> dict:
    """Post-processed beats (as at test) against the ground truth, per validation source."""
    model.eval()
    tracker = PM2SBeats(model=model, device=str(device))
    out = {}
    with torch.no_grad():
        for name, exs in groups.items():
            rows = []
            for ex in exs:
                bt, bd = tracker(ex["notes"])
                c = compare(ex["t"], ex["down_flags"], bt, bd)
                rows.append((c["beat_f"], c["downbeat_f"]))
            if rows:
                out[f"{name}_beat_f"] = float(np.mean([r[0] for r in rows]))
                out[f"{name}_downbeat_f"] = float(np.mean([r[1] for r in rows]))
    names = [g for g in groups if f"{g}_beat_f" in out]
    out["select"] = float(np.mean([(out[f"{g}_beat_f"] + out[f"{g}_downbeat_f"]) / 2 for g in names])) if names else 0.0
    model.train()
    return out


def write_status(run: Path, **kw):
    (run / "status.json").write_text(json.dumps(kw, indent=1))


def make_schedule(opt, args):
    """PM2S's step schedule (x0.1 after --lr-step), or linear warmup then cosine to --min-lr-frac."""
    import math
    if args.schedule == "step" and not args.warmup:
        return torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=0.1)

    def factor(step):
        warm = min(1.0, (step + 1) / args.warmup) if args.warmup else 1.0
        if args.schedule == "step":
            return warm * 0.1 ** (step // args.lr_step)
        progress = min(1.0, max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup)))
        return warm * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="pm2s_ho")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-step", type=int, default=40000, help="--schedule step: steps before the learning rate x0.1")
    ap.add_argument("--schedule", default="step", choices=["step", "cosine"],
                    help="step: PM2S's (x0.1 after --lr-step); cosine: down to --min-lr-frac by --steps, after --warmup")
    ap.add_argument("--warmup", type=int, default=0, help="steps of linear warmup (cosine)")
    ap.add_argument("--min-lr-frac", type=float, default=0.05, help="cosine: the last rate, as a share of --lr")
    ap.add_argument("--betas", default="0.8,0.8", help="AdamW betas (PM2S: 0.8,0.8; usual: 0.9,0.999)")
    ap.add_argument("--eps", type=float, default=1e-4, help="AdamW eps (PM2S: 1e-4)")
    ap.add_argument("--clip", type=float, default=0.0, help="gradient norm clipping (0 = off, as PM2S)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--length", type=int, default=500, help="notes per training segment")
    ap.add_argument("--gangqinpu-share", type=float, default=0.5,
                    help="share of the samples from Gangqinpu (the other sources by their size); 0 = by size too")
    ap.add_argument("--sources", default="asap,real,periscope,gangqinpu")
    ap.add_argument("--val-n", type=int, default=150, help="validation performances per source, at most (a fixed draw)")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=0, help="evaluations without a better score before stopping (0 = off)")
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop cleanly (last.pt) after this long; --resume goes on")
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
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    data, run = Path(args.data), Path(args.runs_dir) / args.name
    run.mkdir(parents=True, exist_ok=True)

    names = [s for s in args.sources.split(",") if s]
    if any(s not in SOURCES for s in names):
        raise SystemExit(f"--sources: from {', '.join(SOURCES)}")
    pool = {}
    for s in names:
        exs = [e for e in (example(p) for p in load(data, SOURCES[s])) if e is not None]
        if exs:
            pool[s] = exs
    if not pool:
        raise SystemExit(f"no training performances in {data / 'meter'} (scripts/build_meter_data.py)")
    names = list(pool)
    sizes = np.array([len(pool[s]) for s in names], np.float64)
    if "gangqinpu" in pool and args.gangqinpu_share > 0 and len(names) > 1:
        rest = sizes * np.array([s != "gangqinpu" for s in names])
        probs = rest / rest.sum() * (1 - args.gangqinpu_share)
        probs[names.index("gangqinpu")] = args.gangqinpu_share
    else:
        probs = sizes / sizes.sum()
    groups = {}
    for s in names:
        exs = [e for e in (example(p) for p in load(data, VAL[s])) if e is not None]
        if len(exs) > args.val_n:
            exs = [exs[i] for i in sorted(np.random.default_rng(0).choice(len(exs), args.val_n, replace=False))]
        if exs:
            groups[s] = exs
    if not groups:
        raise SystemExit(f"no validation performances in {data / 'meter'}")

    model = build_model().to(device)
    try:
        betas = tuple(float(x) for x in args.betas.split(","))
        assert len(betas) == 2
    except (ValueError, AssertionError):
        raise SystemExit(f"--betas: two numbers, e.g. 0.9,0.999 (not {args.betas!r})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=betas, eps=args.eps, weight_decay=1e-2)
    sched = make_schedule(opt, args)
    step, best, best_step, bad = 0, -1.0, 0, 0
    if args.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best, best_step, bad = ck["step"], ck["best"], ck["best_step"], ck.get("bad", 0)
        rng = np.random.default_rng([args.seed, step])
        print(f"resumed at step {step} (best {best:.4f} at {best_step})", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    json.dump({"args": vars(args), "arch": "pm2s_beat", "params": n_params,
               "data": {"train_by_source": {s: len(pool[s]) for s in names},
                        "sample_share": {s: float(p) for s, p in zip(names, probs)},
                        "val_by_source": {k: len(v) for k, v in groups.items()}},
               "recipe": "PM2S RNNJointBeatModel from scratch; see scripts/train_pm2s.py"},
              open(run / "config.json", "w"), indent=1)
    print(f"PM2S beat model: {n_params / 1e6:.2f}M parameters on {device} | training: "
          + ", ".join(f"{s} {len(pool[s])} ({p:.0%})" for s, p in zip(names, probs))
          + " | validation: " + ", ".join(f"{k} {len(v)}" for k, v in groups.items()), flush=True)

    log_path = run / "log.csv"
    new_log = not (args.resume and log_path.exists())
    log = open(log_path, "w" if new_log else "a", newline="")
    writer = None
    t0, acc, n_acc = time.time(), {"beat": 0.0, "downbeat": 0.0, "tempo": 0.0, "grad_norm": 0.0}, 0
    write_status(run, phase="training", step=step, total_steps=args.steps, best=best, best_step=best_step)
    model.train()
    stopped = False
    while step < args.steps:
        x, yb, yd, yt, m = (a.to(device) for a in batch(pool, probs, names, args.batch_size, args.length, rng))
        loss, parts = loss_of(model, x, yb, yd, yt, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip if args.clip > 0 else float("inf")))
        opt.step()
        sched.step()
        step += 1
        for k in ("beat", "downbeat", "tempo"):
            acc[k] += parts[k]
        acc["grad_norm"] += grad if np.isfinite(grad) else 0.0
        n_acc += 1
        if step % args.eval_every == 0 or step == args.steps:
            val = validate(model, groups, device)
            row = {"step": step, "lr": opt.param_groups[0]["lr"], "minutes": round((time.time() - t0) / 60, 1),
                   **{(k if k == "grad_norm" else f"loss_{k}"): v / max(n_acc, 1) for k, v in acc.items()}, **val}
            acc, n_acc = {k: 0.0 for k in acc}, 0
            if writer is None:
                writer = csv.DictWriter(log, fieldnames=list(row))
                if new_log:
                    writer.writeheader()
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
            log.flush()
            print("  step " + " | ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
            if val["select"] > best:
                best, best_step, bad = val["select"], step, 0
                torch.save(model.state_dict(), run / "best.pt")
            else:
                bad += 1
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "best": best, "best_step": best_step, "bad": bad}, run / "last.pt")
            write_status(run, phase="training", step=step, total_steps=args.steps, best=best, best_step=best_step)
            if args.patience and bad >= args.patience:
                print(f"no better validation score in {bad} evaluations: stopping at step {step}", flush=True)
                break
            if args.max_hours and time.time() - t0 > args.max_hours * 3600 and step < args.steps:
                stopped = True
                break
    log.close()
    if stopped:
        write_status(run, phase="stopped", step=step, total_steps=args.steps, best=best, best_step=best_step)
        print(f"stopped after {args.max_hours} h at step {step}: --resume goes on", flush=True)
        return
    write_status(run, phase="done", step=step, total_steps=args.steps, best=best, best_step=best_step)
    print(f"done: {step} steps; best validation score {best:.4f} (mean of beat F and downbeat F) at step "
          f"{best_step} -> {run / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
