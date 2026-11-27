"""Headroom tests: how much an idea could gain at most, with the trained models as they are (no
training), before anyone builds it. outputs/headroom/<test>.json and report.md.

    python scripts/headroom.py labeller --run nl_gq             # the duration / position choosers
    python scripts/headroom.py meter --run meter_v4_gq          # local meter repair

labeller: a learned chooser among the labeller's own top-k candidates (a written length chosen from
  the surrounding notes, the idea after Nakamura, Yoshii & Dixon's note-value MRF) can at most pick
  the right one whenever it is among them. Per test set (the true grid: the labeller's targets are
  defined on it), each head's accuracy with its likeliest class and with the best of its top 2 / 4,
  and the windows rendered and scored (src/eval/metrics.py, as scripts/eval_labeller.py) with that
  oracle for the length (dur), the position in the beat (sub), and both. If even the top-4 oracle
  barely moves the rendered duration accuracy and note F1, a learned chooser is not worth training.
meter: repairing the meter where the grid is uncertain can at most pick, for each piece or each
  stretch of bars, the best of a few alternative grids: the decoded one, the bar lines shifted by
  1-3 beats, the note values halved / doubled, the piece decoded in each of its 3 likeliest time
  signatures, one meter per piece (consistent_meter). Per test set, the decoded grid's (downbeat F
  + time-signature accuracy) / 2 against the best alternative per piece and per stretch of
  ``--segment`` bars (each stretch's bar lines scored on their own: an upper bound, as a real repair
  must also rejoin the grid around it). If a correct alternative is rarely among them, a ranker has
  nothing to find.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HEADS = ("pos", "sub", "dur", "staff", "delta")
LABEL_COL = {"delta": 1, "pos": 2, "sub": 3, "dur": 4, "staff": 5}


# ------------------------------------------------------------------------------------------ labeller
def oracle_logits(out: dict, labels, heads: tuple, k: int) -> dict:
    """The heads' logits with the true class made the likeliest wherever it is among the top k."""
    import torch
    o = {h: v.clone() for h, v in out.items()}
    for h in heads:
        y = labels[..., LABEL_COL[h]].to(o[h].device)
        ok = y >= 0
        top = o[h].topk(min(k, o[h].shape[-1]), -1).indices
        hit = (top == y.clamp(min=0)[..., None]).any(-1) & ok
        boost = torch.zeros_like(o[h]).scatter_(-1, y.clamp(min=0)[..., None], 1e4)
        o[h] = torch.where(hit[..., None], o[h] + boost, o[h])
    return o


def labeller(args) -> dict:
    import torch

    from scripts.eval_labeller import SPLITS
    from scripts.train import TruncatingCollate, pick_device
    from src.dataset import ROLE_WINDOW, ScoreSegmentDataset
    from src.eval.metrics import Accumulator
    from src.labeller import load_labeller, placed_notes
    from src.notelabels import LABEL_FILE, render, score_notes
    device = pick_device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = load_labeller(Path(args.runs_dir) / args.run / "best.pt", device)
    cfg = model.cfg
    ctx = {"context_before": cfg.context_before, "context_after": cfg.context_after}
    coll = TruncatingCollate(cfg.max_notes, 10 ** 9)
    data = Path(args.data)
    modes = {"argmax": (), "dur@2": (("dur",), 2), "dur@4": (("dur",), 4), "sub@2": (("sub",), 2),
             "sub@4": (("sub",), 4), "dur+sub@4": (("dur", "sub"), 4)}
    results = {}
    for cond in args.conditions.split(","):
        split = SPLITS[cond]
        if not (data / split / LABEL_FILE).exists():
            print(f"skip {cond}: {data / split / LABEL_FILE} missing")
            continue
        ds = ScoreSegmentDataset(str(data / split), str(data / "vocab.json"), notelabels=True, **ctx)
        idx = list(range(len(ds)))
        if args.max_windows and len(idx) > args.max_windows:
            idx = sorted(np.random.default_rng(0).choice(len(idx), args.max_windows, replace=False).tolist())
        items = [ds[i] for i in idx]
        gts = [ds.vocab.decode(it["target"][it["prompt_len"]:-1].tolist()) for it in items]
        wins = [score_notes(g)[1] for g in gts]
        accs = {m: Accumulator() for m in modes}
        topk = {h: np.zeros(5) for h in HEADS}           # n, top1, top2, top4 hits
        order = sorted(range(len(items)), key=lambda j: len(items[j]["notes"]))
        with torch.no_grad():
            for lo in range(0, len(order), args.batch_size):
                sel = order[lo:lo + args.batch_size]
                b = coll([dict(items[j]) for j in sel])
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    out = model(b["notes"].to(device), b["note_mask"].to(device))
                out = {h: v.float() for h, v in out.items()}
                labels = b["labels"]
                for h in HEADS:
                    y = labels[..., LABEL_COL[h]].to(device)
                    ok = y >= 0
                    top = out[h].topk(min(4, out[h].shape[-1]), -1).indices
                    hit = top == y.clamp(min=0)[..., None]
                    topk[h] += [int(ok.sum()), int((hit[..., :1].any(-1) & ok).sum()),
                                int((hit[..., :2].any(-1) & ok).sum()), int((hit.any(-1) & ok).sum()), 0]
                for m, spec in modes.items():
                    o = out if not spec else oracle_logits(out, labels, *spec)
                    for r, j in enumerate(sel):
                        it = items[j]
                        own = (it["roles"] <= ROLE_WINDOW).numpy()
                        pl = placed_notes(o, r, it["notes"][:, 0].numpy(), it["bar_est"].numpy(), own, ds.label_space)
                        accs[m].add(gts[j], render(wins[j], pl))
        res = {"windows": len(items),
               "heads": {h: {"top1": v[1] / max(v[0], 1), "top2": v[2] / max(v[0], 1), "top4": v[3] / max(v[0], 1)}
                         for h, v in topk.items()},
               "rendered": {m: {k: a.result()[k] for k in ("note_f1", "onset_f1", "duration_acc", "staff_acc")}
                            for m, a in accs.items()}}
        results[cond] = res
        r = res["rendered"]
        print(f"{cond} ({len(items)} windows): dur head top1 {res['heads']['dur']['top1']:.3f} top4 "
              f"{res['heads']['dur']['top4']:.3f}; rendered duration {r['argmax']['duration_acc']:.3f} -> "
              f"{r['dur@4']['duration_acc']:.3f} (dur@4), note F1 {r['argmax']['note_f1']:.3f} -> {r['dur+sub@4']['note_f1']:.3f}",
              flush=True)
    return results


# --------------------------------------------------------------------------------------------- meter
def alternatives(bt, lp, cfg) -> dict:
    """Alternative grids for one performance (module docstring), by name."""
    from src.meter import (TS_CLASSES, consistent_meter, decode, grid_from_logp, grid_from_states, scale_grid,
                           tempo_logprior)
    out = {"decoded": grid_from_logp(bt, lp, cfg)}
    for k in (1, 2, 3):
        out[f"shift{k}"] = grid_from_logp(bt, dict(lp, bar=np.roll(lp["bar"], k, axis=0)), cfg)
    for f, name in ((0.5, "half"), (2, "double")):
        g = scale_grid(out["decoded"], f)
        if g is not None:
            out[name] = g
    step = lp["step"] + tempo_logprior(bt, cfg) if cfg.tempo_weight > 0 else lp["step"]
    for c in np.argsort(-lp["ts"].sum(0))[:3]:
        states, steps = decode(lp["bar"], step, lp["ts"], cfg, ts_idx=(int(c),), return_steps=True,
                               logp_pos=lp.get("pos"))
        out[f"ts_{TS_CLASSES[int(c)]}"] = grid_from_states(bt, states, steps)
    out["one_meter"] = consistent_meter(out["decoded"], lp, 0.5)
    return out


def segment_scores(grid: dict, gt: dict, seg: int) -> list[tuple[float, float]]:
    """Per stretch of ``seg`` annotated bars: (downbeat F, time-signature accuracy) of the grid's bar
    lines inside its time span."""
    from src.beats import MATCH_TOL, f_measure
    downs = np.nonzero(gt["down"])[0]
    bar_t = np.asarray(grid["bar_t"], np.float64)
    out = []
    for s in range(0, len(downs), seg):
        group = downs[s:s + seg]
        lo = gt["t"][group[0]] - MATCH_TOL
        hi = gt["t"][downs[s + seg]] - MATCH_TOL if s + seg < len(downs) else gt["t"][-1] + MATCH_TOL
        est = bar_t[(bar_t >= lo) & (bar_t < hi)]
        ok = 0
        for j in group:
            b = int(np.searchsorted(bar_t, gt["t"][j] + MATCH_TOL)) - 1
            ok += 0 <= b < len(grid["bars"]) and grid["bars"][b] == gt["ts_names"][j]
        out.append((f_measure(gt["t"][group], est), ok / len(group)))
    return out


def meter(args) -> dict:
    from scripts.eval_meter import CONDITIONS, tracked
    from scripts.train import pick_device
    from scripts.train_meter import load
    from src.beats import BeatNoise
    from src.meter import DecodeConfig, grid_scores, load_meter, predict_logp, uses_aux
    device = pick_device(args.device)
    run = Path(args.runs_dir) / args.run
    model = load_meter(run / "best.pt", device)
    if uses_aux(model.cfg):
        raise SystemExit(f"{args.run} reads a second tracker: not supported here")
    dfile = run / "decode.json"
    cfg = DecodeConfig.from_dict(json.loads(dfile.read_text())["decode"]) if dfile.exists() else DecodeConfig()
    noise = BeatNoise.from_json(args.beat_noise)
    data = Path(args.data)
    results = {}
    for cond in args.conditions.split(","):
        perfs = load(data, CONDITIONS[cond])
        if args.max_perfs and len(perfs) > args.max_perfs:
            perfs = [perfs[i] for i in sorted(np.random.default_rng(0).choice(len(perfs), args.max_perfs, replace=False))]
        if not perfs:
            print(f"skip {cond}: no performances")
            continue
        piece, seg_dec, seg_best, wins = [], [], [], {}
        for k, p in enumerate(perfs):
            bt, bd = tracked(p, k, noise)
            lp = predict_logp(model, p["notes"], bt, bd, device)
            if lp is None or len(bt) < 4:
                continue
            alts = alternatives(bt, lp, cfg)
            sel = {n: grid_scores(g, p["gt"]) for n, g in alts.items()}
            val = {n: (s["downbeat_f"] + s["ts_acc"]) / 2 for n, s in sel.items()}
            best = max(val, key=val.get)
            wins[best] = wins.get(best, 0) + 1
            piece.append((val["decoded"], val[best], sel["decoded"]["downbeat_f"], max(s["downbeat_f"] for s in sel.values()),
                          sel["decoded"]["ts_acc"], max(s["ts_acc"] for s in sel.values())))
            segs = {n: segment_scores(g, p["gt"], args.segment) for n, g in alts.items()}
            for j in range(len(segs["decoded"])):
                seg_dec.append(segs["decoded"][j])
                seg_best.append(max((segs[n][j] for n in segs), key=lambda x: x[0] + x[1]))
        a = np.array(piece)
        sd, sb = np.array(seg_dec), np.array(seg_best)
        res = {"performances": len(a),
               "piece": {"decoded": float(a[:, 0].mean()), "oracle": float(a[:, 1].mean()),
                         "downbeat_f": [float(a[:, 2].mean()), float(a[:, 3].mean())],
                         "ts_acc": [float(a[:, 4].mean()), float(a[:, 5].mean())]},
               "segment": {"stretches": len(sd), "decoded": float(sd.sum(1).mean() / 2), "oracle": float(sb.sum(1).mean() / 2),
                           "downbeat_f": [float(sd[:, 0].mean()), float(sb[:, 0].mean())],
                           "ts_acc": [float(sd[:, 1].mean()), float(sb[:, 1].mean())]},
               "best_alternative": dict(sorted(wins.items(), key=lambda x: -x[1]))}
        results[cond] = res
        print(f"{cond} ({len(a)} performances): (downbeat F + ts) / 2 decoded {res['piece']['decoded']:.3f}, best per piece "
              f"{res['piece']['oracle']:.3f}, best per {args.segment} bars {res['segment']['oracle']:.3f} "
              f"(decoded there {res['segment']['decoded']:.3f}); winners {res['best_alternative']}", flush=True)
    return results


def report(out: Path) -> str:
    L = ["# Headroom: what each idea could gain at most", ""]
    lab = out / "labeller.json"
    if lab.exists():
        d = json.loads(lab.read_text())
        L += ["## Labeller: the best of its top-k candidates (true grid)", "",
              "| test set | dur head top1 / top2 / top4 | sub head top1 / top4 | rendered duration acc: argmax / dur@2 / dur@4 "
              "| note F1: argmax / dur+sub@4 |", "|---|---|---|---|---|"]
        for c, r in d.items():
            h, x = r["heads"], r["rendered"]
            L.append(f"| {c} | {h['dur']['top1']:.3f} / {h['dur']['top2']:.3f} / {h['dur']['top4']:.3f} | "
                     f"{h['sub']['top1']:.3f} / {h['sub']['top4']:.3f} | {x['argmax']['duration_acc']:.3f} / "
                     f"{x['dur@2']['duration_acc']:.3f} / {x['dur@4']['duration_acc']:.3f} | {x['argmax']['note_f1']:.3f} / "
                     f"{x['dur+sub@4']['note_f1']:.3f} |")
        L += ["", "A learned duration chooser is worth training if dur@4 lifts the rendered duration accuracy by "
              "several points (one seed of the meter model moves whole pieces by ~0.9 MUSTER).", ""]
    met = out / "meter.json"
    if met.exists():
        d = json.loads(met.read_text())
        L += ["## Meter: the best of a few alternative grids", "",
              "(downbeat F + time-signature accuracy) / 2; per stretch: each stretch of bars scored on its own (upper bound).", "",
              "| test set | decoded | best per piece | decoded (stretches) | best per stretch | downbeat F per stretch: decoded / best | most often best |",
              "|---|---|---|---|---|---|---|"]
        for c, r in d.items():
            p, s = r["piece"], r["segment"]
            top = ", ".join(f"{k} {v}" for k, v in list(r["best_alternative"].items())[:3])
            L.append(f"| {c} | {p['decoded']:.3f} | {p['oracle']:.3f} | {s['decoded']:.3f} | {s['oracle']:.3f} | "
                     f"{s['downbeat_f'][0]:.3f} / {s['downbeat_f'][1]:.3f} | {top} |")
        L += ["", "A local-repair ranker is worth building if the best per stretch is well above the decoded grid "
              "(and a correct alternative exists for most stretches).", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=("labeller", "meter", "report"))
    ap.add_argument("--run", default=None)
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--max-windows", type=int, default=3000, help="labeller: windows per test set (0 = all)")
    ap.add_argument("--max-perfs", type=int, default=0, help="meter: performances per test set (0 = all)")
    ap.add_argument("--segment", type=int, default=8, help="meter: bars per stretch")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--beat-noise", default="configs/beat_noise.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--out", default="outputs/headroom")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.test == "labeller":
        args.run = args.run or "nl_gq"
        args.conditions = args.conditions or "clean,real,periscope,gangqinpu"
        (out / "labeller.json").write_text(json.dumps(labeller(args), indent=1))
    elif args.test == "meter":
        args.run = args.run or "meter_v4_gq"
        args.conditions = args.conditions or "clean,real,periscope,gangqinpu"
        (out / "meter.json").write_text(json.dumps(meter(args), indent=1))
    text = report(out)
    (out / "report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
