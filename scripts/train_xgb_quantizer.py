"""Train Baseline 2 (beat-snap + per-note XGBoost corrections) on the training split.

    python scripts/train_xgb_quantizer.py --windows 6000
    python scripts/run_baselines.py --methods beat_snap,xgb_quantizer

Training windows: a random subset of clean train windows with synthetic noise, plus all real
Kong transcriptions of training performances (train_transcribed). Saves outputs/baselines/xgb_quantizer/.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_baselines import Timings  # noqa: E402
from src.baselines.xgb_quantizer import XGBQuantizer, window_features, window_labels  # noqa: E402
from src.dataset import drop_released_carried  # noqa: E402
from src.preprocessing.noise import NoiseModel  # noqa: E402
from src.preprocessing.store import SegmentStore  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402

D = Path("data/processed")


def collect(store, idxs, vocab, timings, nm=None, rng=None):
    X, L = [], {k: [] for k in ("keep", "staff", "dur", "shift")}
    for i in idxs:
        meta = store.meta[i]
        notes, pedal, tok = store.get(i)
        if nm is not None:
            notes, pedal = nm(notes, pedal, rng)
        notes = drop_released_carried(notes, meta.get("carry_min_sec"))
        try:
            a, x = window_features(notes, pedal, meta, timings.get(meta))
        except Exception:  # noqa: BLE001  (rare windows the timing map can't place)
            continue
        if not len(x):
            continue
        for k, v in zip(L, window_labels(notes, a, vocab.decode(tok.tolist()))):
            L[k].append(v)
        X.append(x)
    return np.concatenate(X), {k: np.concatenate(v) for k, v in L.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=6000, help="clean train windows (+ synthetic noise)")
    ap.add_argument("--out", default="outputs/baselines/xgb_quantizer")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    vocab = Vocab.load(D / "vocab.json")
    nm = NoiseModel.from_json("configs/noise_stats.json")
    timings = Timings()
    rng = np.random.default_rng(0)

    t = time.time()
    tr = SegmentStore(D / "train")
    syn_idx = rng.choice(len(tr), size=min(args.windows, len(tr)), replace=False)
    X1, L1 = collect(tr, syn_idx, vocab, timings, nm, rng)
    real = SegmentStore(D / "train_transcribed")
    X2, L2 = collect(real, range(len(real)), vocab, timings)
    X = np.concatenate([X1, X2])
    L = {k: np.concatenate([L1[k], L2[k]]) for k in L1}
    print(f"features {X.shape} in {time.time() - t:.0f}s | keep rate {L['keep'].mean():.3f} | "
          f"labelled staff {np.mean(L['staff'] >= 0):.3f}", flush=True)

    t = time.time()
    q = XGBQuantizer().fit(X, L)
    q.save(Path(args.out))
    print(f"trained 4 classifiers in {time.time() - t:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
