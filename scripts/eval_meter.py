"""Evaluate a meter model (v2, Phase 2) and save its grids for the note labeller (Phase 3).

    python scripts/eval_meter.py --run meter_v2 --tune      # tune the decoder on val, then every test set
    python scripts/eval_meter.py --run meter_v2             # every test set (decode.json if tuned before)

For every test performance: the tracked beats, the model's decoded grid (src/meter.py
predict_logp + decode) and its scores against the ground truth (grid_scores: beat F, downbeat F,
time-signature accuracy) next to the tracker's own beat / downbeat F on the same beats.
Conditions:
  clean      ASAP test MIDI, the tracker's errors simulated (a fixed draw per performance, as the
             seq2seq evaluation's simulated-tracker condition), scored against the annotations
  real       the Transkun transcriptions of the ASAP test audio with Beat This!'s own beats
  periscope  PERiScoPe test (104 works no model trained on): real transcriptions, the tracker
             simulated on, and scored against, the score's beats timed through the note alignment
             (scripts/build_meter_data.py; within 70 ms of ASAP's human annotations for 81% of
             beats, the same time signature for >99.9%), one grid per segment of the performance

--tune: the decoder's weights and penalties (src/meter.py DecodeConfig: bar / time-signature head
weights, the time-signature change and mid-step bar line penalties) chosen on the validation
performances (ASAP val, real val, --tune-periscope PERiScoPe val) by coordinate search on the
training script's selection score (mean of downbeat F and time-signature accuracy), saved to
outputs/runs/<run>/decode.json and used from then on. Every condition is also scored with the
default decoder (``default`` in the results), so the tuning's effect is on record.

Writes outputs/runs/<run>/eval_meter.json (means per condition and per piece, the time-signature
confusions), meter_scores_<condition>.json (per performance) and grids_<condition>.pkl ({key:
grid, with its performance, its span and the key signature estimated from its notes:
src/notelabels.py estimate_key}), which scripts/eval_labeller.py --grid meter:<run> reads.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import pick_device  # noqa: E402
from scripts.train_meter import load  # noqa: E402
from src.beats import MATCH_TOL, BeatNoise, f_measure  # noqa: E402
from src.meter import DecodeConfig, build_meter_model, grid_from_logp, grid_scores, predict_logp  # noqa: E402
from src.notelabels import estimate_key  # noqa: E402

CONDITIONS = {"clean": "test", "real": "test_transcribed", "periscope": "test_periscope"}
VAL = {"asap": "val", "real": "val_transcribed", "periscope": "val_periscope"}
SEED = 12345  # the simulated tracker's draw on the clean performances (fixed: comparable runs)
SEARCH = {  # coordinate search over the decoder's settings (each list holds the default)
    "bar_weight": [0.5, 1.0, 2.0, 4.0],
    "ts_weight": [0.25, 0.5, 1.0, 2.0],
    "ts_change": [math.log(x) for x in (1e-1, 1e-2, 1e-3, 1e-5, 1e-8, 1e-12)],
    "mid_step_bar": [math.log(x) for x in (0.3, 1e-1, 1e-2, 1e-3)],
}


def key_of(perf: dict) -> str:
    """A grid's key in grids_<condition>.pkl: the performance, #segment when it has several."""
    return perf["performance"] if perf.get("segments", 1) <= 1 else f"{perf['performance']}#{perf['segment']}"


def tracked(perf: dict, k: int, noise, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """The beats the system gets for a performance: the real tracker's, else the simulated one."""
    if perf.get("beats") is not None:
        return np.asarray(perf["beats"][0], np.float64), np.asarray(perf["beats"][1], bool)
    gt = perf["gt"]
    bt, bd = noise(gt["t"], gt["down"], np.random.default_rng((seed, k)))
    return np.asarray(bt, np.float64), np.asarray(bd, bool)


def score(perfs, lps, beats, cfg: DecodeConfig) -> list:
    return [grid_scores(grid_from_logp(bt, lp, cfg), p["gt"]) for p, lp, (bt, _) in zip(perfs, lps, beats)]


def select(groups_scores: list) -> float:
    """The training script's selection score: mean over the sources of (downbeat F + ts acc) / 2."""
    vals = [(np.mean([s["downbeat_f"] for s in g]) + np.mean([s["ts_acc"] for s in g])) / 2 for g in groups_scores if g]
    return float(np.mean(vals)) if vals else 0.0


def tune(model, data: Path, noise, device, n_periscope: int, rounds: int = 2,
         max_perfs: int = 0) -> tuple[DecodeConfig, dict]:
    """Coordinate search of DecodeConfig on the validation performances (their log-probabilities
    computed once; the tracker drawn with another seed than the test draw)."""
    groups = []
    for name, split in VAL.items():
        perfs = load(data, split)
        cap = min(n_periscope, max_perfs or n_periscope) if name == "periscope" else max_perfs
        if cap and len(perfs) > cap:
            perfs = [perfs[i] for i in sorted(np.random.default_rng(1).choice(len(perfs), cap, replace=False))]
        if not perfs:
            continue
        beats = [tracked(p, k, noise, seed=SEED + 1) for k, p in enumerate(perfs)]
        lps = [predict_logp(model, p["notes"], bt, bd, device) for p, (bt, bd) in zip(perfs, beats)]
        groups.append((name, perfs, lps, beats))
    cfg, cache = DecodeConfig(), {}

    def value(c: DecodeConfig) -> float:
        key = tuple(sorted(c.to_dict().items()))
        if key not in cache:
            cache[key] = select([score(p, lp, b, c) for _, p, lp, b in groups])
        return cache[key]
    start = value(cfg)
    for _ in range(rounds):
        for name, options in SEARCH.items():  # ties go to the value nearest the default
            best = max(options, key=lambda v: (value(DecodeConfig(**{**cfg.to_dict(), name: v})),
                                               -abs(v - getattr(DecodeConfig(), name))))
            cfg = DecodeConfig(**{**cfg.to_dict(), name: best})
    info = {"validation": {name: len(p) for name, p, _, _ in groups}, "select_default": start,
            "select_tuned": value(cfg), "configs_tried": len(cache)}
    return cfg, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--conditions", default="clean,real,periscope")
    ap.add_argument("--tune", action="store_true", help="tune the decoder on the validation performances first")
    ap.add_argument("--tune-periscope", type=int, default=150, help="PERiScoPe val performances used by --tune")
    ap.add_argument("--tune-max", type=int, default=0, help="at most this many per validation source (0 = all)")
    ap.add_argument("--beat-noise", default="configs/beat_noise.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--runs-dir", default="outputs/runs")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    run, data = Path(args.runs_dir) / args.run, Path(args.data)
    ck = torch.load(run / args.ckpt, map_location=device, weights_only=False)
    model = build_meter_model(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    noise = BeatNoise.from_json(args.beat_noise)
    dfile = run / "decode.json"
    if args.tune:
        cfg, info = tune(model, data, noise, device, args.tune_periscope, max_perfs=args.tune_max)
        dfile.write_text(json.dumps({"decode": cfg.to_dict(), **info}, indent=1))
        print(f"decoder tuned on validation ({info['validation']}): selection score {info['select_default']:.3f} "
              f"-> {info['select_tuned']:.3f} with {cfg.to_dict()}", flush=True)
    cfg = DecodeConfig.from_dict(json.loads(dfile.read_text())["decode"]) if dfile.exists() else DecodeConfig()
    out_file = run / "eval_meter.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    for cond in args.conditions.split(","):
        perfs = load(data, CONDITIONS[cond])
        if not perfs:
            print(f"skip {cond}: no {CONDITIONS[cond]} performances (scripts/build_meter_data.py)")
            continue
        grids, scores, default, raw, per = {}, [], [], [], []
        confusion = Counter()
        for k, perf in enumerate(perfs):
            bt, bd = tracked(perf, k, noise)
            lp = predict_logp(model, perf["notes"], bt, bd, device)
            grid = grid_from_logp(bt, lp, cfg)
            grid["key"] = estimate_key(perf["notes"])  # the key signature the rendered score gets
            grid["performance"] = perf["performance"]
            grid["span"] = tuple(perf.get("span") or (float(bt[0]) - 1.0, float(bt[-1]) + 1.0))
            grids[key_of(perf)] = grid
            s = grid_scores(grid, perf["gt"])
            scores.append(s)
            default.append(grid_scores(grid_from_logp(bt, lp, DecodeConfig()), perf["gt"]))
            gt = perf["gt"]
            raw.append({"beat_f": f_measure(gt["t"], bt), "downbeat_f": f_measure(gt["t"][gt["down"]], bt[bd])})
            bar_t = np.asarray(grid["bar_t"])
            for j in np.nonzero(gt["down"])[0]:  # per ground-truth bar: true -> decoded time signature
                b = int(np.searchsorted(bar_t, gt["t"][j] + MATCH_TOL)) - 1
                confusion[(gt["ts_names"][j], grid["bars"][b] if 0 <= b < len(grid["bars"]) else "-")] += 1
            per.append({"performance": key_of(perf), "piece": perf.get("piece"),
                        "true_ts": Counter(gt["ts_names"]).most_common(1)[0][0],
                        "decoded_ts": Counter(grid["bars"]).most_common(1)[0][0] if grid["bars"] else "-",
                        **{f"grid_{a}": v for a, v in s.items()}, **{f"tracker_{a}": v for a, v in raw[-1].items()}})
        res = {f"grid_{key}": float(np.mean([s[key] for s in scores])) for key in scores[0]}
        res.update({f"tracker_{key}": float(np.mean([s[key] for s in raw])) for key in raw[0]})
        res["default"] = {f"grid_{key}": float(np.mean([s[key] for s in default])) for key in default[0]}
        by_piece = {}
        for r in per:
            by_piece.setdefault(r["piece"], []).append(r)
        res["piece_mean"] = {f"grid_{key}": float(np.mean([np.mean([r[f'grid_{key}'] for r in v])
                                                          for v in by_piece.values()])) for key in scores[0]}
        res["performances"], res["pieces"] = len(perfs), len(by_piece)
        res["decode"] = cfg.to_dict()
        res["ts_confusion"] = [[a, b, n] for (a, b), n in confusion.most_common(25)]
        results[cond] = res
        with open(run / f"grids_{cond}.pkl", "wb") as f:
            pickle.dump(grids, f, protocol=pickle.HIGHEST_PROTOCOL)
        (run / f"meter_scores_{cond}.json").write_text(json.dumps(per, indent=1))
        wrong = [(a, b, n) for (a, b), n in confusion.most_common() if a != b][:4]
        print(f"{cond} ({len(perfs)} performances, {len(by_piece)} pieces): grid beat F {res['grid_beat_f']:.3f} "
              f"(tracker {res['tracker_beat_f']:.3f}), downbeat F {res['grid_downbeat_f']:.3f} (tracker "
              f"{res['tracker_downbeat_f']:.3f}), time signature {res['grid_ts_acc']:.3f} | default decoder: "
              f"downbeat F {res['default']['grid_downbeat_f']:.3f}, time signature {res['default']['grid_ts_acc']:.3f}"
              " | most bars wrong: " + ", ".join(f"{a}->{b} {n}" for a, b, n in wrong), flush=True)
    out_file.write_text(json.dumps(results, indent=1))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
