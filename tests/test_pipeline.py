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


def make_checkpoint(d: Path, **extra) -> Path:
    vocab = Vocab.load(str(V))
    torch.manual_seed(0)
    cfg = dict(vocab_size=len(vocab), dropout=0.0, max_tokens=2048, max_notes=1024, **PRESETS["small"], **extra)
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

        def spy(feats, prefix, n_notes=None, ban_ts=False):  # record each window's output vs its budget
            out = real_decode(feats, prefix, n_notes, ban_ts)
            n = len(feats) if n_notes is None else n_notes
            budgets.append((len(out), max_new_tokens(n, len(prefix), pipe.cfg.max_tokens)))
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
            assert 0 <= st["guard_fallbacks"] <= len(wins) and (mode in ("guarded", "meter") or st["guard_fallbacks"] == 0)
            assert 0 <= st["meter_fixes"] <= len(wins) and (mode == "meter" or st["meter_fixes"] == 0)
        assert budgets and all(n <= cap for n, cap in budgets), max(budgets)

        # none: each window decoded on its own, stitched
        alone = []
        for w in wins:
            alone = stitch(alone, real_decode(pipe.features(w), pipe.prefix([])))
        assert results["none"].tokens == alone
        # chained == guarded with an infinite margin (the guard never fires)
        pipe.decode_window = real_decode
        chained, _ = pipe.decode(wins, "chained")
        never, counts = pipe.decode(wins, "guarded", guard_margin=math.inf)
        assert chained == never and counts["guard_fallbacks"] == 0
        assert results["chained"].tokens == chained
        try:
            pipe.decode(wins, "beam")
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown mode should raise")


def test_context_model_end_to_end():
    """A model trained with context notes: the pipeline feeds it 10 features, budgets and guards on
    the window's own notes only, and decodes every mode."""
    from src.dataset import N_FEATURES_CONTEXT
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        midi = tmp / "perf.mid"
        make_performance(midi)
        ck = make_checkpoint(tmp, n_features=N_FEATURES_CONTEXT, context_before=16, context_after=8)
        pipe = ScorePipeline(ck, device="cpu", window_notes=WINDOW)
        notes, pedal = load_performance(midi)
        wins = performance_windows(notes, pedal, size=WINDOW, context_before=16, context_after=8)
        f = pipe.features(wins[1])
        assert f.shape == (len(wins[1]["notes"]) + len(wins[1]["before"]) + len(wins[1]["after"]), N_FEATURES_CONTEXT)
        seen, real = [], pipe.decode_window

        def spy(feats, prefix, n_notes=None, ban_ts=False):
            seen.append((len(feats), n_notes))
            return real(feats, prefix, n_notes, ban_ts)
        pipe.decode_window = spy
        for mode in MODES:
            res = pipe.run(midi, tmp / mode, mode=mode)
            assert res.stats["windows"] == len(wins) and (tmp / mode / "tokens.txt").exists()
        assert all(n is not None and n < total for total, n in seen[1:]), "budget counts window notes only"


def test_banned_tokens_in_generate():
    """generate(banned=...): a banned row never produces a banned token, the other rows of the batch
    are unaffected, and banned=None changes nothing (both architectures)."""
    from src.dataset import ScoreSegmentDataset, collate
    vocab = Vocab.load(str(V))
    ds = ScoreSegmentDataset("data/processed/test", str(V))
    b = collate([ds[0], ds[1]])
    ts_ids = [i for i, t in enumerate(vocab.itos) if t.startswith("ts_")]
    bos = torch.tensor([vocab.stoi["<bos>"]])
    for arch, cfg in (("scoreformer", dict(PRESETS["small"])),
                      ("score2seq", dict(d_model=128, enc_layers=2, dec_layers=2, ff=256, n_heads=4))):
        torch.manual_seed(1)
        m = build_model(arch, dict(vocab_size=len(vocab), dropout=0.0, **cfg)).eval()
        head = torch.nn.Linear(m.head.in_features, m.head.out_features, bias=True)
        head.weight = m.head.weight
        with torch.no_grad():
            head.bias.zero_()
            head.bias[ts_ids] = 10.0  # make time signatures the likeliest tokens, so the ban has work to do
        m.head = head
        with torch.no_grad():
            free = m.generate(b["notes"], b["note_mask"], [bos, bos], vocab.stoi["<eos>"], max_new=40)
            assert m.generate(b["notes"], b["note_mask"], [bos, bos], vocab.stoi["<eos>"], max_new=40,
                              banned=None) == free
            ban = torch.zeros(2, len(vocab), dtype=torch.bool)
            ban[0, ts_ids] = True
            out = m.generate(b["notes"], b["note_mask"], [bos, bos], vocab.stoi["<eos>"], max_new=40, banned=ban)
        assert any(t in ts_ids for t in free[0]), "test setup: the free row should produce a time signature"
        assert not any(t in ts_ids for t in out[0]), arch
        assert out[1] == free[1], arch


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
