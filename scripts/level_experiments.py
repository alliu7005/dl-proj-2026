"""Can evidence outside the meter decoder choose a piece's note-value level? (docs/experiments.md,
"Fifth round"). For every validation and test performance: the meter run's decoded grid (its
decode.json), the same bar lines at halved and doubled note values (src/meter.py scale_grid), and
which of them each signal would pick, against the level the annotations give (level_ratio):

  prior      the written tempo the grid implies (quarters per minute) and its note density (chords
             per quarter), each against a log-normal fitted on the training performances' true
             grids, plus a bonus for the decoded level
  labeller   the note labeller's confidence on each grid: the mean log-probability of the labels it
             picks (bar correction, position, sub-position, length, keep), plus the same bonus

The decoder's own heads cannot choose (the decoded grid is the best under them, the other levels
included), which is why only outside evidence is tried.

    python scripts/level_experiments.py --meter meter_v3_tp --labeller outputs/runs/nl_base_gn/best.pt
    python scripts/level_experiments.py --meter meter_v3_tp --no-labeller          # the priors only (fast)

Writes outputs/level_experiments/<meter>.json (per performance) and .md (the tables).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_meter import CONDITIONS, SEED, VAL, tracked  # noqa: E402
from scripts.train import pick_device  # noqa: E402
from scripts.train_meter import load  # noqa: E402
from src.beats import BeatNoise  # noqa: E402
from src.meter import (LEVEL_TOL, LEVELS, DecodeConfig, build_meter_model, grid_from_logp, level_ratio,  # noqa: E402
                       piece_qpm, predict_logp, scale_grid)

STAYS = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
PRIORS = {"tempo": (1.0, 0.0), "density": (0.0, 1.0), "tempo + density": (1.0, 1.0)}
HEADS = {"position": ("pos", "sub"), "length": ("dur",), "bar correction": ("delta",), "keep": ("keep",),
         "all": ("delta", "pos", "sub", "dur", "keep")}


def density(notes: np.ndarray, grid: dict) -> float | None:
    """Chords (onsets 30 ms apart or more) per quarter of the grid, over its beats."""
    t, q = np.asarray(grid["t"], np.float64), np.asarray(grid["q"], np.float64)
    if len(t) < 2 or not len(notes):
        return None
    on = np.sort(np.asarray(notes, np.float64)[:, 1])
    on = on[(on >= t[0]) & (on <= t[-1])]
    if len(on) < 2 or q[-1] <= q[0]:
        return None
    chords = int(np.concatenate([[True], np.diff(on) > 0.03]).sum())
    return chords / (q[-1] - q[0])


def fit_priors(perfs: list) -> dict:
    """log-normal (mean, std) of the true grids' written tempo and density over the training set."""
    lq, ld = [], []
    for p in perfs:
        g = {"t": p["gt"]["t"], "q": p["gt"]["q"]}
        a, b = piece_qpm(g), density(p["notes"], g)
        if a and b:
            lq.append(math.log(a))
            ld.append(math.log(b))
    return {"tempo": (float(np.mean(lq)), float(np.std(lq))), "density": (float(np.mean(ld)), float(np.std(ld))),
            "n": len(lq), "corr": float(np.corrcoef(lq, ld)[0, 1]) if len(lq) > 2 else float("nan")}


class LabellerConfidence:
    """The labeller (V2Pipeline's) on a grid: mean log-probability of its picks per head."""

    def __init__(self, labeller: str, meter_ckpt: str, device: str):
        from src.pipeline import V2Pipeline
        self.pipe = V2Pipeline(labeller, meter_ckpt, device=device)

    @torch.no_grad()
    def __call__(self, notes: np.ndarray, grid: dict) -> dict:
        from scripts.train import TruncatingCollate
        from src.dataset import ROLE_WINDOW
        from src.pipeline import performance_windows
        pipe = self.pipe
        windows = performance_windows(notes, np.zeros((0, 2), np.float32), pipe.window_notes,
                                      context_before=int(pipe.cfg.context_before),
                                      context_after=int(pipe.cfg.context_after))
        items, _ = pipe.items(windows, grid)
        coll = TruncatingCollate(pipe.cfg.max_notes, 10 ** 9)
        tot = {k: 0.0 for k in ("delta", "pos", "sub", "dur", "keep")}
        n = 0
        for lo in range(0, len(items), 16):
            chunk = items[lo:lo + 16]
            b = coll([dict(x) for x in chunk])
            out = pipe.labeller(b["notes"].to(pipe.device), b["note_mask"].to(pipe.device))
            for r, it in enumerate(chunk):
                m = len(it["notes"])
                own = (it["roles"][:m] <= ROLE_WINDOW).numpy()
                for k in ("delta", "pos", "sub", "dur"):
                    lp = torch.log_softmax(out[k][r, :m].float(), -1).max(-1).values.cpu().numpy()
                    tot[k] += float(lp[own].sum())
                pk = torch.sigmoid(out["keep"][r, :m].float()).cpu().numpy()
                tot["keep"] += float(np.log(np.maximum(pk, 1 - pk))[own].sum())
                n += int(own.sum())
        return {k: v / max(n, 1) for k, v in tot.items()}


def right(ratio: float, f) -> bool:
    return abs(math.log2(ratio * float(f))) <= LEVEL_TOL


def pick(scores: dict, stay: float):
    s = {f: v + (stay if f == "1" else 0.0) for f, v in scores.items()}
    return max(s, key=lambda f: (s[f], f == "1"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meter", default="meter_v3_tp", help="the meter run (its best.pt and decode.json)")
    ap.add_argument("--labeller", default="outputs/runs/nl_base_gn/best.pt")
    ap.add_argument("--no-labeller", action="store_true", help="the priors only")
    ap.add_argument("--max-perfs", type=int, default=150, help="at most this many performances per set (0 = all)")
    ap.add_argument("--beat-noise", default="configs/beat_noise.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--out", default="outputs/level_experiments")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    data, run = Path(args.data), Path(args.runs_dir) / args.meter
    ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model = build_meter_model(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    dfile = run / "decode.json"
    cfg = DecodeConfig.from_dict(json.loads(dfile.read_text())["decode"]) if dfile.exists() else DecodeConfig()
    cfg = DecodeConfig(**{**cfg.to_dict(), "level_weight": 0.0, "level_model_weight": 0.0})   # as decoded
    noise = BeatNoise.from_json(args.beat_noise)
    priors = fit_priors(load(data, "train"))
    lab = None if args.no_labeller else LabellerConfidence(args.labeller, str(run / "best.pt"), str(device))
    sets = [("val", name, split, SEED + 1) for name, split in VAL.items()] + \
           [("test", name, split, SEED) for name, split in CONDITIONS.items()]
    rows = []
    for part, name, split, seed in sets:
        perfs = load(data, split)
        if args.max_perfs and len(perfs) > args.max_perfs:
            perfs = [perfs[i] for i in sorted(np.random.default_rng(1).choice(len(perfs), args.max_perfs, replace=False))]
        for k, p in enumerate(perfs):
            bt, bd = tracked(p, k, noise, seed=seed)
            g = grid_from_logp(bt, predict_logp(model, p["notes"], bt, bd, device), cfg)
            r = level_ratio(g, p["gt"])
            if r is None or r <= 0:
                continue
            notes = np.asarray(p["notes"], np.float32)
            row = {"part": part, "set": name, "performance": p["performance"], "ratio": r, "prior": {}, "labeller": {}}
            for f in LEVELS:
                sg = scale_grid(g, f)
                if sg is None:
                    continue
                a, b = piece_qpm(sg), density(notes, sg)
                if a and b:
                    row["prior"][str(f)] = {"tempo": -0.5 * ((math.log(a) - priors["tempo"][0]) / priors["tempo"][1]) ** 2,
                                            "density": -0.5 * ((math.log(b) - priors["density"][0]) /
                                                               priors["density"][1]) ** 2}
                if lab is not None:
                    row["labeller"][str(f)] = lab(notes, sg)
            rows.append(row)
        print(f"{part} {name}: {sum(r['set'] == name and r['part'] == part for r in rows)} performances", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.meter}.json").write_text(json.dumps({"priors": priors, "rows": rows}, indent=0))
    md = [f"# The note-value level: which signal picks it ({args.meter})", "",
          "Share of performances at the right level (decoded quarters / true quarters within "
          f"2^{LEVEL_TOL} of 1). `decoded`: the grid as decoded; `best possible`: the best of the levels on "
          "offer; each signal row: its best bonus for the decoded level on validation, and the test share at "
          "that bonus. From `scripts/level_experiments.py`.", ""]
    parts = {"val": [r for r in rows if r["part"] == "val"], "test": [r for r in rows if r["part"] == "test"]}
    signals = {f"prior: {n}": ("prior", lambda c, w=w: w[0] * c["tempo"] + w[1] * c["density"]) for n, w in PRIORS.items()}
    if lab is not None:
        signals.update({f"labeller: {n}": ("labeller", lambda c, hs=hs: sum(c[h] for h in hs)) for n, hs in HEADS.items()})

    def share(rs, kind=None, fn=None, stay=0.0):
        if kind is None:
            return float(np.mean([right(r["ratio"], 1) for r in rs])) if rs else float("nan")
        ok = []
        for r in rs:
            cands = r[kind]
            ok.append(right(r["ratio"], Fraction(pick({f: fn(c) for f, c in cands.items()}, stay))) if cands
                      else right(r["ratio"], 1))
        return float(np.mean(ok)) if ok else float("nan")
    md += ["| signal | bonus (chosen on val) | val | test |", "|---|---|---|---|",
           f"| decoded | - | {share(parts['val']):.3f} | {share(parts['test']):.3f} |",
           "| best possible | - | {:.3f} | {:.3f} |".format(
               *[float(np.mean([any(right(r['ratio'], Fraction(f)) for f in (r['prior'] or {'1': 0})) for r in rs]))
                 if rs else float("nan") for rs in (parts["val"], parts["test"])])]
    for label, (kind, fn) in signals.items():
        stay = max(STAYS, key=lambda s: (share(parts["val"], kind, fn, s), s))
        md.append(f"| {label} | {stay} | {share(parts['val'], kind, fn, stay):.3f} | {share(parts['test'], kind, fn, stay):.3f} |")
    md += ["", "Per set (decoded):", "", "| part | set | performances | right level |", "|---|---|---|---|"]
    for part, rs in parts.items():
        for name in dict.fromkeys(r["set"] for r in rs):
            sub = [r for r in rs if r["set"] == name]
            md.append(f"| {part} | {name} | {len(sub)} | {share(sub):.3f} |")
    md += ["", f"Priors fitted on {priors['n']} training performances: written tempo log-mean "
               f"{priors['tempo'][0]:.3f} ({math.exp(priors['tempo'][0]):.0f} qpm), std {priors['tempo'][1]:.3f}; "
               f"density log-mean {priors['density'][0]:.3f} ({math.exp(priors['density'][0]):.2f} chords per quarter), "
               f"std {priors['density'][1]:.3f}; correlation {priors['corr']:.2f}."]
    (out / f"{args.meter}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"wrote {out / (args.meter + '.md')}")


if __name__ == "__main__":
    main()
