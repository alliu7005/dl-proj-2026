"""End-to-end tests for the demo pipeline (src/pipeline.py) with a tiny untrained model.

    python -m tests.test_pipeline        # ~1 min on a CPU; needs data/processed/vocab.json

An untrained model writes junk (loops, few notes), which is exactly what exercises the output
budget and the guard. Checks:
  * windowing: every performed note lands in exactly one window, carried-over notes come first
    with negative onsets, pedal intervals are clipped to their window
  * every mode (guarded / chained / none) writes tokens, MusicXML or a logged export error, and
    result.json with consistent stats
  * none == each window decoded on its own and stitched; chained == guarded with an infinite margin
  * every window's output respects decoding.max_new_tokens
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import torch

from src.decoding import max_new_tokens, stitch
from src.pipeline import MODES, ScorePipeline, performance_windows
from src.preprocessing.performance import load_performance
from src.preprocessing.vocab import Vocab
from src.scoreformer import PRESETS, build_model

V = Path("data/processed/vocab.json")
WINDOW = 24  # small windows keep the untrained model's decoding fast


def make_performance(path: Path, n: int = 90, seed: int = 0):
    """A 2-voice performance with rubato, overlapping held notes and sustain pedal."""
    rng = np.random.default_rng(seed)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    t = 0.5
    for k in range(n):
        t += float(rng.uniform(0.08, 0.35))
        pitch = int(rng.choice([48, 52, 55, 60, 62, 64, 65, 67, 69, 71, 72]))
        dur = float(rng.uniform(0.1, 1.5))
        inst.notes.append(pretty_midi.Note(int(rng.integers(40, 100)), pitch, t, t + dur))
    for a in np.arange(0.6, t, 2.0):
        inst.control_changes += [pretty_midi.ControlChange(64, 100, float(a)),
                                 pretty_midi.ControlChange(64, 0, float(a + 1.2))]
    pm.instruments.append(inst)
    pm.write(str(path))


def make_checkpoint(d: Path) -> Path:
    vocab = Vocab.load(str(V))
    torch.manual_seed(0)
    cfg = dict(vocab_size=len(vocab), dropout=0.0, max_tokens=2048, max_notes=1024, **PRESETS["small"])
    stats = V.with_name("feature_stats.json")
    if stats.exists():
        st = json.loads(stats.read_text())
        cfg.update(feat_mean=st["mean"], feat_std=st["std"])
    model = build_model("scoreformer", cfg)
    path = d / "best.pt"
    torch.save({"model": model.state_dict(), "arch": "scoreformer", "config": cfg}, path)
    shutil.copy(V, d / "vocab.json")
    return path


def test_windows():
    with tempfile.TemporaryDirectory() as tmp:
        midi = Path(tmp) / "perf.mid"
        make_performance(midi)
        notes, pedal = load_performance(midi)
        wins = performance_windows(notes, pedal, size=WINDOW)
        assert sum(w["n"] for w in wins) == len(notes)                       # each note in one window
        for j, w in enumerate(wins):
            x = w["notes"]
            c = w["carried"]
            assert (x[:c, 1] < 0).all() and (x[c:, 1] >= 0).all()           # carried first, negative onset
            assert j > 0 or c == 0                                          # nothing carried into window 0
            assert (x[:, 2] > 0).all()                                      # durations positive
            if len(w["pedal"]):
                assert (w["pedal"][:, 1] > 0).all() and (w["pedal"][:, 0] < w["t1"] - w["t0"]).all()
        assert all(a["t0"] < b["t0"] for a, b in zip(wins, wins[1:]))


def test_modes_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        midi = tmp / "perf.mid"
        make_performance(midi)
        pipe = ScorePipeline(make_checkpoint(tmp), device="cpu", window_notes=WINDOW)
        notes, pedal = load_performance(midi)
        wins = performance_windows(notes, pedal, size=WINDOW)

        budgets, real_decode = [], pipe.decode_window

        def spy(feats, prefix):  # record each window's output against its budget
            out = real_decode(feats, prefix)
            budgets.append((len(out), max_new_tokens(len(feats), len(prefix), pipe.cfg.max_tokens)))
            return out
        pipe.decode_window = spy

        results = {}
        for mode in MODES:
            res = pipe.run(midi, tmp / mode, mode=mode)
            results[mode] = res
            out = tmp / mode
            assert (out / "tokens.txt").read_text().split() == res.tokens
            assert ("musicxml" in res.files) or any(e.startswith("MusicXML") for e in res.errors), res.errors
            saved = json.loads((out / "result.json").read_text())
            st = saved["stats"]
            assert st["mode"] == mode and st["windows"] == len(wins) and st["notes"] == len(notes)
            assert st["bars"] == res.tokens.count("bar") and st["tokens"] == len(res.tokens)
            assert 0 <= st["guard_fallbacks"] <= len(wins) and (mode == "guarded" or st["guard_fallbacks"] == 0)
        assert budgets and all(n <= cap for n, cap in budgets), max(budgets)

        # none: each window decoded on its own, stitched
        alone = []
        for w in wins:
            alone = stitch(alone, real_decode(pipe.features(w), pipe.prefix([])))
        assert results["none"].tokens == alone
        # chained == guarded with an infinite margin (the guard never fires)
        pipe.decode_window = real_decode
        chained, _ = pipe.decode(wins, "chained")
        never, n_fb = pipe.decode(wins, "guarded", guard_margin=math.inf)
        assert chained == never and n_fb == 0
        assert results["chained"].tokens == chained
        try:
            pipe.decode(wins, "beam")
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown mode should raise")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
