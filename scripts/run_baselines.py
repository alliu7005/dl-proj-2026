"""Evaluate baselines on the test windows with the shared metrics (src/eval/metrics.py).

Conditions (same windows, same ground-truth targets):
  clean      : clean performance MIDI (what Beyer & Dai were trained/evaluated on)
  synthetic  : clean MIDI + our fitted noise model (fixed seed)
  real       : real Kong transcriptions (test_transcribed), also split into audio Kong never saw

    python scripts/run_baselines.py                    # beat-snap baseline
    python scripts/run_baselines.py --methods beat_snap,xgb_quantizer

Writes outputs/baselines/results.json and per-method predictions (jsonl) for later inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.quantize import PieceTiming, quantize_window  # noqa: E402
from src.dataset import drop_released_carried  # noqa: E402
from src.eval.metrics import Accumulator  # noqa: E402
from src.preprocessing.noise import NoiseModel  # noqa: E402
from src.preprocessing.score import load_score  # noqa: E402
from src.preprocessing.store import SegmentStore  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402

D = Path("data/processed")
ASAP = Path("data/raw/asap_bd")

SHOW = [("note_f1", "note F1"), ("onset_f1", "onset F1"), ("onset_shift_f1", "onset F1 (shift)"),
        ("duration_acc", "duration acc"), ("staff_acc", "staff acc"), ("bar_f1", "bar F1"),
        ("tie_f1", "tie F1"), ("grace_f1", "grace F1"), ("ped_f1", "pedal F1")]


class Timings:
    """Caches parsed scores and per-performance timing."""

    def __init__(self):
        self.ann = json.load(open(ASAP / "asap_annotations.json"))
        self.scores, self.timing = {}, {}

    def get(self, meta) -> PieceTiming:
        perf = meta["performance"]
        if perf not in self.timing:
            if meta["piece"] not in self.scores:
                self.scores[meta["piece"]] = load_score(ASAP / meta["piece"] / "xml_score.musicxml")
            self.timing[perf] = PieceTiming(self.ann[perf], self.scores[meta["piece"]])
        return self.timing[perf]


def conditions(nm: NoiseModel):
    """Yield (condition name, store, notes-getter)."""
    test = SegmentStore(D / "test")
    rng = np.random.default_rng(12345)

    def clean(i):
        return test.get(i)[:2]

    noisy_cache = {}

    def synthetic(i):
        if i not in noisy_cache:
            n, p, _ = test.get(i)
            noisy_cache[i] = nm(n, p, rng)
        return noisy_cache[i]

    yield "clean", test, clean
    yield "synthetic", test, synthetic
    real = SegmentStore(D / "test_transcribed")
    yield "real", real, lambda i: real.get(i)[:2]
    if (D / "test_transcribed_kong").exists():  # second transcriber, for robustness across transcribers
        kong = SegmentStore(D / "test_transcribed_kong")
        yield "real_kong", kong, lambda i: kong.get(i)[:2]
    # clean inputs on exactly the windows that have a real transcription (paired comparison)
    key = {(m["performance"], m.get("win")): i for i, m in enumerate(test.meta)}
    paired = [key[(m["performance"], m.get("win"))] for m in real.meta]
    yield "clean_paired", _Subset(test, paired), lambda i: test.get(paired[i])[:2]


class _Subset:
    """A view of some windows of a SegmentStore (meta + targets), in a given order."""

    def __init__(self, store, idx):
        self.store, self.idx = store, idx
        self.meta = [store.meta[i] for i in idx]

    def get(self, i):
        return self.store.get(self.idx[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="beat_snap")
    ap.add_argument("--out", default="outputs/baselines")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = Vocab.load(D / "vocab.json")
    nm = NoiseModel.from_json("configs/noise_stats.json")
    nm.cfg.clean_prob = 0.0
    timings = Timings()

    methods = {}
    for name in args.methods.split(","):
        if name == "beat_snap":
            methods[name] = lambda notes, pedal, meta, timing: quantize_window(notes, pedal, meta, timing)
        elif name == "xgb_quantizer":
            from src.baselines.xgb_quantizer import XGBQuantizer
            q = XGBQuantizer.load(out / "xgb_quantizer")
            methods[name] = q.predict_window
        else:
            raise ValueError(name)

    results = json.load(open(out / "results.json")) if (out / "results.json").exists() else {}
    for cond, store, get in conditions(nm):
        for mname, fn in methods.items():
            acc, acc_unseen = Accumulator(), Accumulator()
            with open(out / f"pred_{mname}_{cond}.jsonl", "w") as f:
                for i, meta in enumerate(store.meta):
                    notes, pedal = get(i)
                    notes = drop_released_carried(notes, meta.get("carry_min_sec"))  # as in training
                    pred = fn(notes, pedal, meta, timings.get(meta))
                    gt = vocab.decode(store.get(i)[2].tolist())
                    acc.add(gt, pred)
                    if cond == "real" and meta.get("maestro_split") != "train":
                        acc_unseen.add(gt, pred)
                    f.write(json.dumps({"performance": meta["performance"], "win": meta.get("win"),
                                        "pred": pred}) + "\n")
            results.setdefault(mname, {})[cond] = acc.result()
            if cond == "real":
                results[mname]["real_unseen_audio"] = acc_unseen.result()
            print(f"done {mname} / {cond}", flush=True)
    json.dump(results, open(out / "results.json", "w"), indent=1)

    conds = ["clean", "synthetic", "clean_paired", "real", "real_unseen_audio", "real_kong"]
    for mname in results:
        print(f"\n{mname}")
        print(f"  {'metric':18s}" + "".join(f"{c:>18s}" for c in conds))
        for key, label in SHOW:
            row = [results[mname].get(c, {}).get(key, float("nan")) for c in conds]
            print(f"  {label:18s}" + "".join(f"{x:18.3f}" for x in row))


if __name__ == "__main__":
    main()
