"""Evaluate a meter model (v2, Phase 2) and save its grids for the note labeller (Phase 3).

    python scripts/eval_meter.py --run meter_base           # clean + real test performances

For every test performance: the tracked beats (clean: the ASAP test MIDI with the tracker's errors
simulated, a fixed draw per performance, as the seq2seq evaluation's simulated-tracker condition;
real: the Transkun transcription with Beat This!'s own beats), the model's decoded grid
(src/meter.py predict_grid), and its scores against the annotations (grid_scores: beat F,
downbeat F, time-signature accuracy) next to the tracker's own beat / downbeat F on the same beats.
Writes outputs/runs/<run>/eval_meter.json (means per condition) and grids_<condition>.pkl
({performance: grid, with the key signature estimated from its notes: src/notelabels.py
estimate_key}), which scripts/eval_labeller.py --grid meter:<run> reads.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import pick_device  # noqa: E402
from scripts.train_meter import load  # noqa: E402
from src.beats import BeatNoise, f_measure  # noqa: E402
from src.meter import build_meter_model, grid_scores, predict_grid  # noqa: E402
from src.notelabels import estimate_key  # noqa: E402

CONDITIONS = {"clean": "test", "real": "test_transcribed"}
SEED = 12345  # the simulated tracker's draw on the clean performances (fixed: comparable runs)


def tracked(perf: dict, k: int, noise) -> tuple[np.ndarray, np.ndarray]:
    """The beats the system gets for a performance: the real tracker's, else the simulated one."""
    if perf.get("beats") is not None:
        return np.asarray(perf["beats"][0], np.float64), np.asarray(perf["beats"][1], bool)
    gt = perf["gt"]
    bt, bd = noise(gt["t"], gt["down"], np.random.default_rng((SEED, k)))
    return np.asarray(bt, np.float64), np.asarray(bd, bool)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--conditions", default="clean,real")
    ap.add_argument("--beat-noise", default="configs/beat_noise.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--runs-dir", default="outputs/runs")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    run = Path(args.runs_dir) / args.run
    ck = torch.load(run / args.ckpt, map_location=device, weights_only=False)
    model = build_meter_model(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    noise = BeatNoise.from_json(args.beat_noise)
    out_file = run / "eval_meter.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    for cond in args.conditions.split(","):
        perfs = load(Path(args.data), CONDITIONS[cond])
        if not perfs:
            print(f"skip {cond}: no {CONDITIONS[cond]} performances (scripts/build_meter_data.py)")
            continue
        grids, scores, raw = {}, [], []
        for k, perf in enumerate(perfs):
            bt, bd = tracked(perf, k, noise)
            grid = predict_grid(model, perf["notes"], bt, bd, device)
            grid["key"] = estimate_key(perf["notes"])  # the key signature the rendered score gets
            grids[perf["performance"]] = grid
            scores.append(grid_scores(grid, perf["gt"]))
            gt = perf["gt"]
            raw.append({"beat_f": f_measure(gt["t"], bt), "downbeat_f": f_measure(gt["t"][gt["down"]], bt[bd])})
        res = {f"grid_{key}": float(np.mean([s[key] for s in scores])) for key in scores[0]}
        res.update({f"tracker_{key}": float(np.mean([s[key] for s in raw])) for key in raw[0]})
        res["performances"] = len(perfs)
        results[cond] = res
        with open(run / f"grids_{cond}.pkl", "wb") as f:
            pickle.dump(grids, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{cond} ({len(perfs)} performances): grid beat F {res['grid_beat_f']:.3f} (tracker "
              f"{res['tracker_beat_f']:.3f}), downbeat F {res['grid_downbeat_f']:.3f} (tracker "
              f"{res['tracker_downbeat_f']:.3f}), time signature {res['grid_ts_acc']:.3f}", flush=True)
    out_file.write_text(json.dumps(results, indent=1))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
