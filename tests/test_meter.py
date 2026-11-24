"""Tests for the meter model (v2, Phase 2) and its grids in the note labeller (Phase 3).

    python -m tests.test_meter        # or: pytest tests/test_meter.py

Covers the annotated grid, the per-beat targets under tracker errors, the consistent decoding (a
perfect prediction gives back the annotated grid; bars always add up to their time signature; the
vectorised decoder equals a plain Viterbi, weights and penalties included), the grid's scores, the
beat features, the note encoder's inputs and pooling, the ground truth from score alignments
(PERiScoPe: merged window grids -> segments, stitched notes), noise only on clean notes, the key
signature estimate, an estimated grid in a window's frame and the segment lookup, the scripts end
to end on the local data (skipped without data/processed/meter), and the PACE job graph.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.batch_env import batch_env  # noqa: E402
from src.beats import merge_performance, merge_performance_bars  # noqa: E402
from src.meter import (GRID, N_BEAT_FEATURES, N_NOTE_FEATURES, Q_STEPS, TS_CLASSES, DecodeConfig,  # noqa: E402
                       aligned_grids, annotation_grid, bar_quarters, beat_features, build_meter_model, collate,
                       decode, grid_beats, grid_from_logp, grid_from_states, grid_scores, is_clean, note_inputs,
                       predict_logp,
                       sample, stitch_notes, targets)
from src.notelabels import Placed, estimate_key, render, window_frame  # noqa: E402

D = ROOT / "data" / "processed"


def ann(beats_per_bar=(4,), ts_changes=None, pickup=1, bars=6, ibi=0.5):
    """An annotation dict: a ``pickup``-beat upbeat, then ``bars`` bars (beats per bar cycling
    through ``beats_per_bar``), time signatures at the given beat indices."""
    t, downs, k = [], [], 0
    for _ in range(pickup):
        t.append(k * ibi)
        k += 1
    for b in range(bars):
        n = beats_per_bar[b % len(beats_per_bar)]
        for j in range(n):
            t.append(k * ibi)
            if j == 0:
                downs.append(k * ibi)
            k += 1
    ts = ts_changes or {downs[0]: ["4/4", 4]}
    return {"performance_beats": t, "performance_downbeats": downs,
            "perf_time_signatures": {str(x): v for x, v in ts.items()}, "perf_key_signatures": {"0": [0, 0]}}


def onehot(y, n, eps=1e-4):
    lp = np.full((len(y), n), math.log(eps))
    for i, v in enumerate(y):
        if v >= 0:
            lp[i, v] = 0.0
        else:
            lp[i] = math.log(1.0 / n)
    return lp


def reference_decode(lb, ls, lt, cfg, ts_idx):
    """A plain Viterbi over (time signature, position in the bar), written for clarity: the
    decoder's specification."""
    lens = {c: int(bar_quarters(TS_CLASSES[c]) * GRID) for c in ts_idx}
    states = [(c, p) for c in ts_idx for p in range(lens[c])]
    steps = [int(s * GRID) for s in Q_STEPS]

    def emit(j, c, p):
        return cfg.bar_weight * (lb[j, 1] if p == 0 else lb[j, 0]) + cfg.ts_weight * lt[j, c]
    score = {s: emit(0, *s) for s in states}
    back = []
    for j in range(1, len(lb)):
        new, arg = {s: -np.inf for s in states}, {}
        for (c, p), v in score.items():
            for k, st in enumerate(steps):
                q, pen = p + st, 0.0
                if q >= lens[c]:
                    pen = cfg.mid_step_bar if q > lens[c] else 0.0
                    q = (q - lens[c]) % lens[c]
                val = v + ls[j - 1, k] + pen
                if val > new[(c, q)]:
                    new[(c, q)], arg[(c, q)] = val, (c, p)
        starts = [s for s in states if s[1] == 0]
        best = max(starts, key=lambda s: new[s])       # a new time signature at a bar line
        alt = new[best] + cfg.ts_change
        for s in starts:
            if alt > new[s]:
                new[s], arg[s] = alt, arg[best]
        back.append(arg)
        score = {s: new[s] + emit(j, *s) for s in states}
    path = [max(score, key=score.get)]
    for arg in reversed(back):
        path.append(arg[path[-1]])
    return path[::-1]


def test_annotation_grid_and_targets():
    a = ann(beats_per_bar=(4, 4, 2, 2), ts_changes={0.5: ["4/4", 4], 4.5: ["6/8", 2]}, bars=4)
    gt = annotation_grid(a)
    # 4/4 beats are quarters, 6/8 beats dotted quarters; the pickup takes the first meter
    assert gt["ts_names"][:1] == ["4/4"] and gt["ts_names"][-1] == "6/8"
    assert np.allclose(np.diff(gt["q"])[:8], 1.0) and np.allclose(np.diff(gt["q"])[-2:], 1.5)
    assert gt["down"].sum() == 4 and gt["ts"][-1] == TS_CLASSES.index("6/8")
    # exact beats: every step a whole beat, bar targets on the downbeats
    tg = targets(gt["t"], gt)
    assert tg["bar"].tolist() == gt["down"].astype(int).tolist()
    assert Q_STEPS[tg["step"][1]] == 1 and Q_STEPS[tg["step"][-1]] == Fraction(3, 2)
    # a tracker at double tempo (a beat between every two) and one that misses every other beat
    dbl = np.sort(np.concatenate([gt["t"], (gt["t"][:-1] + gt["t"][1:]) / 2]))
    assert Q_STEPS[targets(dbl, gt)["step"][2]] == Fraction(1, 2)
    half = gt["t"][::2]
    assert Q_STEPS[targets(half, gt)["step"][1]] == 2
    # beats outside the annotated span: no targets
    out = targets(np.array([-5.0, gt["t"][0], 99.0]), gt)
    assert out["bar"][0] == out["bar"][2] == -1 and out["ts"][2] == -1


def test_decode_and_grid():
    """A perfect prediction decodes to the annotated grid, time-signature change included; with a
    wrong bar cue on one beat the decoding stays consistent (bars add up to their meter)."""
    a = ann(beats_per_bar=(3, 3, 3, 2, 2, 2), ts_changes={0.5: ["3/4", 3], 5.0: ["6/8", 2]}, bars=6)
    gt = annotation_grid(a)
    tg = targets(gt["t"], gt)
    lb, ls, lt = onehot(tg["bar"], 2), onehot(tg["step"], len(Q_STEPS)), onehot(tg["ts"], len(TS_CLASSES))
    g = grid_from_states(gt["t"], decode(lb, ls, lt))
    assert g["bars"] == ["3/4"] * 4 + ["6/8"] * 3, g["bars"]  # the pickup's bar + 3, then 3 of 6/8
    assert np.allclose(g["bar_t"][1:], gt["t"][gt["down"]]), (g["bar_t"], gt["t"][gt["down"]])
    s = grid_scores(g, gt)
    assert s["beat_f"] == 1.0 and s["downbeat_f"] == 1.0 and s["ts_acc"] == 1.0, s
    lb2 = lb.copy()
    lb2[5] = [math.log(1e-4), 0.0]          # a false downbeat cue in the middle of a 3/4 bar
    g2 = grid_from_states(gt["t"], decode(lb2, ls, lt))
    lens = np.diff(np.concatenate([g2["bar_q"], [g2["q"][-1] + 1]]))[:-1]
    want = [float(Fraction(int(ts.split("/")[0]) * 4, int(ts.split("/")[1]))) for ts in g2["bars"][:-1]]
    assert np.allclose(lens[1:], want[1:]), (lens, g2["bars"])   # every full bar adds up
    bt, bd = grid_beats(g)
    assert len(bt) == len(gt["t"]) and bd.sum() == gt["down"].sum()  # the pickup bar line is before the first beat


def test_decode_matches_reference():
    """The vectorised decoder gives the plain Viterbi's path, with tuned weights and penalties too;
    the bar weight moves the decision the way it should."""
    rng = np.random.default_rng(3)
    ts_idx = (TS_CLASSES.index("2/4"), TS_CLASSES.index("3/4"), TS_CLASSES.index("3/8"))

    def lsm(a):
        return a - np.log(np.exp(a).sum(1, keepdims=True))
    for trial in range(6):
        k = int(rng.integers(4, 14))
        lb, ls = lsm(rng.normal(size=(k, 2)) * 2), lsm(rng.normal(size=(k, len(Q_STEPS))) * 2)
        lt = lsm(rng.normal(size=(k, len(TS_CLASSES))))
        cfg = DecodeConfig() if trial % 2 == 0 else DecodeConfig(ts_change=math.log(0.2), mid_step_bar=math.log(0.5),
                                                                 ts_weight=0.5, bar_weight=2.0)
        assert decode(lb, ls, lt, cfg, ts_idx) == reference_decode(lb, ls, lt, cfg, ts_idx), trial
    # bar_weight: bar cues every 3 beats against a time-signature head sure of 4/4: a weak bar head
    # keeps 4/4, a strong one makes the bars 3/4
    gt = annotation_grid(ann(bars=6))
    tg = targets(gt["t"], gt)
    ls, lt = onehot(tg["step"], len(Q_STEPS)), onehot(tg["ts"], len(TS_CLASSES), 0.2)  # sure of quarter steps
    lb3 = onehot((np.arange(len(gt["t"])) % 3 == 1).astype(int), 2, 0.2)
    weak = grid_from_states(gt["t"], decode(lb3, ls, lt, DecodeConfig(bar_weight=0.01)))
    strong = grid_from_states(gt["t"], decode(lb3, ls, lt, DecodeConfig(bar_weight=50.0)))
    assert set(weak["bars"]) == {"4/4"} and set(strong["bars"]) == {"3/4"}, (weak["bars"], strong["bars"])
    assert DecodeConfig.from_dict({"bar_weight": 2, "junk": 1}) == DecodeConfig(bar_weight=2.0)


def test_beat_features_and_key():
    t = 0.5 * np.arange(8)
    notes = np.array([[36, 0.0, 0.4, 100], [60, 0.0, 0.4, 70], [64, 0.25, 0.4, 50], [67, 1.0, 1.4, 90]], np.float32)
    f = beat_features(notes, t, np.arange(8) % 4 == 0)
    from src.meter import N_ALL_FEATURES
    assert f.shape == (8, N_ALL_FEATURES) and np.isfinite(f).all()
    assert (f[:, N_BEAT_FEATURES:] == 0).all()                # no second tracker: its features all 0
    assert f[0, 3] == 1 and f[1, 3] == 0                      # the tracker's downbeat flag
    assert f[0, 5 + 6] > 0                                    # a note half way through beat 0
    assert (f[:, 26] == 0).all()                              # unused, as in the first meter model
    chroma = f[0, 27:39]                                      # beat 0: C (36, 60) and E (64)
    assert abs(chroma.sum() - 1) < 1e-6 and chroma[0] > chroma[4] > 0 and chroma[7] == 0
    assert f[2, 27 + 7] == 1 and f[1, 27:39].sum() == 0       # beat 2: G alone; beat 1: nothing starts
    scale = lambda tonic, minor=False: np.array(  # noqa: E731
        [[tonic + s, k, k + 1, 80] for k, s in enumerate(([0, 2, 3, 5, 7, 8, 10, 12, 7, 3, 0] if minor else
                                                          [0, 2, 4, 5, 7, 9, 11, 12, 7, 4, 0]))], np.float32)
    assert estimate_key(scale(60)) == 0 and estimate_key(scale(67)) == 1 and estimate_key(scale(62)) == 2
    assert estimate_key(scale(57, minor=True)) == 0 and estimate_key(scale(65)) == -1


def test_note_inputs_and_model():
    """Every note goes to the beat it starts in, with its phase in that beat; the note encoder pools
    them per beat (changing a note changes only its beat's input, padding changes nothing); a
    model without it (the first meter model's config) runs on the same batches."""
    import torch
    t = 0.5 * np.arange(6)
    notes = np.array([[48, 0.0, 0.9, 100], [60, 0.0, 0.2, 60], [64, 0.25, 0.4, 50], [67, 0.98, 1.4, 90],
                      [70, 9.0, 9.5, 80]], np.float32)  # 0.98 s: 20 ms before beat 2 -> beat 2; 9 s: after the last
    f, p, j = note_inputs(notes, t)
    assert f.shape == (4, N_NOTE_FEATURES) and j.tolist() == [0, 0, 0, 2] and p.tolist() == [48, 60, 64, 67]
    assert abs(f[2, 0] - 0.5) < 1e-6 and f[3, 0] < 0                 # half way through beat 0; just before beat 2
    assert f[0, 16] == 1 and f[1, 17] == 1 and f[0, 17] == 0          # lowest / highest of the first chord
    assert abs(f[0, 13] - math.log2(0.9 / 0.5) / 4) < 1e-6            # 1.8 beats long
    torch.manual_seed(0)
    cfg = dict(n_note_features=N_NOTE_FEATURES, d_model=32, n_heads=4, layers=1, ff=64, dropout=0.0)
    model = build_meter_model(cfg).eval()
    feats = beat_features(notes, t, np.zeros(6, bool))
    x, mask, nin, _ = collate([(feats, None, t, (f, p, j))])
    with torch.no_grad():
        pooled = model.pool_notes(nin, 1, len(t))
        assert pooled.shape == (1, 6, 2 * 64) and pooled[0, 1].abs().sum() == 0 and pooled[0, 0].abs().sum() > 0
        f2 = f.copy()
        f2[3, 14] = 0.1                                              # the note in beat 2 gets quieter
        _, _, nin2, _ = collate([(feats, None, t, (f2, p, j))])
        d = (model.pool_notes(nin2, 1, len(t)) - pooled).abs().sum(-1)[0]
        assert d[2] > 0 and d[[0, 1, 3, 4, 5]].sum() == 0
        out = model(x, mask, nin)
        assert out["bar"].shape == (1, 6, 2) and out["ts"].shape == (1, 6, len(TS_CLASSES))
        _, _, nin3, _ = collate([(feats, None, t, (f, p, j)), (feats[:3], None, t[:3], (f[:2], p[:2], j[:2]))])
        assert torch.allclose(model.pool_notes(nin3, 2, 6)[0], pooled[0], atol=1e-6)
        first = build_meter_model(dict(n_features=27, d_model=32, n_heads=4, layers=1, ff=64)).eval()
        assert first(x, mask)["step"].shape == (1, 6, len(Q_STEPS))
    lp = predict_logp(model, notes, t, np.zeros(6, bool))
    assert lp["bar"].shape == (6, 2) and np.allclose(np.exp(lp["step"]).sum(1), 1, atol=1e-5)


def test_aligned_grids_and_stitched_notes():
    """PERiScoPe's ground truth: window grids merged by global bar, split where bars are missing,
    quarter positions from the bar lengths (a lost beat does not shift the rest); the notes joined
    back from overlapping windows once each, carried-over copies dropped."""
    def win(bars, ts):  # a window's grid: every beat of its bars, 0.5 s apart from its start
        b, k, t, s = [], [], [], []
        x = 0.0
        for j, (n, name) in enumerate(zip(bars, ts)):
            for i in range(n):
                b.append(j)
                k.append(i)
                t.append(x)
                s.append(name)
                x += 0.5
        return {"bar": np.array(b), "beat": np.array(k), "t": np.array(t), "extrap": np.zeros(len(t)),
                "ts": np.array(s, object)}
    w1 = (0, 0.0, win([4, 4, 3], ["4/4", "4/4", "3/4"]))          # bars 0-2 from 0 s
    w2 = (2, 4.0, win([3, 3], ["3/4", "3/4"]))                    # bars 2-3 from 4 s (overlap: bar 2)
    w3 = (6, 20.0, win([4] * 6, ["4/4"] * 6))                     # bars 6-11: bars 4-5 missing
    t, down, bars, beats, ts = merge_performance_bars([w1, w2, w3])
    assert bars.tolist()[:14] == [0] * 4 + [1] * 4 + [2] * 3 + [3] * 3 and ts[9] == "3/4"
    assert np.allclose(merge_performance([w1, w2, w3])[0], t)
    keep = np.ones(len(t), bool)
    keep[5] = False                                               # a beat lost inside bar 1
    segs = aligned_grids(t[keep], bars[keep], beats[keep], [ts[i] for i in np.nonzero(keep)[0]], min_beats=8)
    assert len(segs) == 2                                         # split at the missing bars 4-5
    a, b = segs
    assert a["q"].tolist() == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13], a["q"]
    assert a["down"].sum() == 4 and a["ts_names"][-1] == "3/4" and a["ts"][-1] == TS_CLASSES.index("3/4")
    assert b["q"][0] == 0 and len(b["t"]) == 24 and b["down"].sum() == 6
    assert aligned_grids(t[:5], bars[:5], beats[:5], ts[:5], min_beats=8) == []
    n1 = np.array([[60, 0.0, 0.5, 80], [62, 3.9, 0.5, 70]], np.float32)                     # window 1, from 0 s
    n2 = np.array([[55, -0.2, 1.0, 60], [62, 0.9, 0.5, 70], [64, 1.0, 0.5, 70]], np.float32)  # window 2, from 3 s
    st = stitch_notes([(0.0, n1), (3.0, n2)])
    assert st[:, 0].tolist() == [60, 62, 64] and np.allclose(st[:, 1], [0.0, 3.9, 4.0], atol=1e-5)
    assert np.allclose(st[:, 2] - st[:, 1], 0.5) and st.dtype == np.float32
    assert stitch_notes([]).shape == (0, 4)


def test_sample_noise_only_on_clean_notes():
    """Synthetic transcription noise goes on clean MIDI only, never on real transcriptions."""
    gt = annotation_grid(ann(bars=8))
    notes = np.array([[60 + i % 12, 0.5 * i, 0.5 * i + 0.3, 80] for i in range(30)], np.float32)
    calls = []

    def note_noise(store, prompt, rng):
        calls.append(len(store))
        return store, None
    rng = np.random.default_rng(0)
    for perf, want in (({"gt": gt, "notes": notes, "beats": None}, 1),
                       ({"gt": gt, "notes": notes, "beats": None, "real_notes": True}, 0),
                       ({"gt": gt, "notes": notes, "beats": (gt["t"], gt["down"])}, 0)):
        calls.clear()
        f, tg, bt, nin = sample(perf, rng, None, 0, note_noise)
        assert len(calls) == want and is_clean(perf) == bool(want), perf.keys()
        assert len(f) == len(bt) and len(nin[0]) == len(nin[2]) > 0
    f, tg, bt, nin = sample({"gt": gt, "notes": notes, "beats": None}, rng, None, 10)
    assert len(f) == 10 and nin[2].max() < 10


def test_augment():
    """Augmentation stretches the notes, the ground-truth beats and the tracked beats together (the
    targets stay the same), keeps the notes on the keyboard, and leaves the stored performance alone."""
    from src.meter import Augment
    gt = annotation_grid(ann(bars=8))
    notes = np.array([[100 + i % 6, 0.5 * i, 0.5 * i + 0.3, 80] for i in range(30)], np.float32)  # up to 105
    perf = {"gt": gt, "notes": notes, "beats": (gt["t"].copy(), gt["down"].copy())}
    keep = notes.copy()
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = Augment()(perf, rng)
        s = a["gt"]["t"][1] / gt["t"][1]
        assert 0.85 - 1e-9 <= s <= 1.18 + 1e-9
        assert np.allclose(a["notes"][:, 1], notes[:, 1] * s, atol=1e-4) and np.allclose(a["beats"][0], gt["t"] * s)
        k = a["notes"][0, 0] - notes[0, 0]
        assert a["notes"][:, 0].max() <= 108 and np.allclose(a["notes"][:, 0] - notes[:, 0], k) and k <= 3
        assert np.array_equal(a["gt"]["q"], gt["q"]) and 1 <= a["notes"][:, 3].min()
        assert all(np.array_equal(targets(a["beats"][0], a["gt"])[h], targets(gt["t"], gt)[h]) for h in ("bar", "step", "ts"))
    assert np.array_equal(perf["notes"], keep) and perf["gt"] is gt                       # nothing changed in place
    assert Augment(prob=0.0)(perf, rng) is perf


def test_oracle_logp():
    """A perfect meter model on tracked beats: certain of each training target, uniform where it is
    unknown; decoded, it gives back the annotated grid on clean beats, and on tracked beats (missed
    and extra ones) still the right time signature and every bar line the beats allow."""
    from src.meter import oracle_logp
    gt = annotation_grid(ann(beats_per_bar=(3,), ts_changes={0.5: ["3/4", 3]}, bars=10))
    assert set(gt["ts_names"]) == {"3/4"}
    lp = oracle_logp(gt["t"], gt)
    tg = targets(gt["t"], gt)
    assert set(lp) == {"bar", "step", "ts"} and lp["step"].shape == (len(gt["t"]), len(Q_STEPS))
    assert (lp["ts"].argmax(1) == tg["ts"]).all() and np.allclose(lp["bar"].max(1), 0)
    s = grid_scores(grid_from_logp(gt["t"], lp), gt)
    assert s["beat_f"] == 1.0 and s["downbeat_f"] == 1.0 and s["ts_acc"] == 1.0, s
    noisy = np.sort(np.concatenate([np.delete(gt["t"], [4, 9]), gt["t"][[2, 6]] + 0.21]))  # 2 missed, 2 extra
    s = grid_scores(grid_from_logp(noisy, oracle_logp(noisy, gt)), gt)
    # an extra beat's step is snapped to a Q_STEPS value (0.42 of a beat -> 1/2), which bends the grid
    # locally: the bar line after it lands 83 ms early, outside the 70 ms tolerance (GridNoise's snap)
    assert s["ts_acc"] == 1.0 and s["downbeat_f"] >= 0.9, s
    assert oracle_logp(gt["t"][:1], gt) is None


def test_long_steps_and_pickup_bars():
    """A decoded step longer than the bar (a 2/4 piece tracked at half notes that missed a beat: 4
    quarters) keeps the bar it passes; cover_notes puts bars in front of the grid for notes played
    before its first bar line (the pickup becomes a partial first bar), and leaves a grid that
    already covers them alone."""
    from src.meter import cover_notes, oracle_logp
    for ts, q_beats in (("2/4", [0, 2, 4, 8, 10, 12]), ("3/4", [0, 1, 2, 3, 4, 8, 9, 10, 11, 12])):
        c = TS_CLASSES.index(ts)
        L = int(bar_quarters(ts))
        gq = np.arange(13, dtype=float)
        gt = {"t": gq * 0.5, "q": gq, "down": np.arange(13) % L == 0, "ts": np.full(13, c), "ts_names": [ts] * 13,
              "key": np.zeros(13)}
        t = np.array(q_beats, float) * 0.5
        g = grid_from_logp(t, oracle_logp(t, gt))
        assert np.allclose(g["q"], q_beats), (ts, g["q"])
        assert np.allclose(g["bar_q"], np.arange(0, 13, L)) and g["bars"] == [ts] * len(g["bar_q"]), (ts, g["bar_q"])
        assert grid_scores(g, gt) == {"beat_f": 1.0, "downbeat_f": 1.0, "ts_acc": 1.0, "level_ok": 1.0}
    g = grid_from_states(np.array([1.5, 2.0, 2.5, 3.0, 3.5]), [(0, 0), (0, 12), (0, 24), (0, 36), (0, 0)])  # 4/4 from 1.5 s
    assert cover_notes(g, 1.5) is g and cover_notes(g, 1.49) is g      # within 1/24 quarter: no bar added
    c = cover_notes(g, 1.0)                                             # one quarter (0.5 s) early: a pickup
    assert c["bars"] == ["4/4", "4/4"] + g["bars"][1:] and np.allclose(c["bar_q"][:2], [0, 4])
    assert np.allclose(c["q"], g["q"] + 4) and np.allclose(c["bar_t"][:2], [-0.5, 1.5])
    from src.notelabels import grid_position
    assert abs(grid_position(np.array([1.0]), c["t"], c["q"])[0][0] - 3.0) < 1e-9   # beat 4 of the added bar
    assert len(cover_notes(g, -1.0)["bars"]) == len(g["bars"]) + 2       # 2.5 s early: two bars


def test_tempo_prior():
    """The written-tempo prior: off by default (an old decode.json decodes as before); on, it picks
    between steps the step head cannot tell apart (a half or a whole quarter between beats 0.5 s
    apart: 60 or 120 quarters per minute) the one nearer its centre; --tempo-prior adds it to the
    tuning's search."""
    import numpy as np
    from scripts.eval_meter import SEARCH, TEMPO_SEARCH
    from src.meter import Q_STEPS, TS_CLASSES, DecodeConfig, grid_from_logp, tempo_logprior
    assert DecodeConfig().tempo_weight == 0.0 and DecodeConfig.from_dict({"bar_weight": 2.0}).tempo_weight == 0.0
    assert set(TEMPO_SEARCH) == {"tempo_weight", "tempo_mu"} and not set(TEMPO_SEARCH) & set(SEARCH)
    k = 33
    t = 1.0 + 0.5 * np.arange(k)
    half, one = Q_STEPS.index(Fraction(1, 2)), Q_STEPS.index(Fraction(1))
    step = np.full((k, len(Q_STEPS)), -20.0)
    step[:, [half, one]] = math.log(0.5)                      # the step head cannot tell them apart
    lp = {"bar": np.log(np.full((k, 2), 0.5)), "step": step,
          "ts": np.log(np.full((k, len(TS_CLASSES)), 1 / len(TS_CLASSES)))}
    assert not tempo_logprior(t, DecodeConfig()).any()
    pri = tempo_logprior(t, DecodeConfig(tempo_weight=1.0))
    assert pri.shape == step.shape and not pri[-1].any() and pri[0, one] > pri[0, half]   # 120 nearer 116 than 60
    near_120 = grid_from_logp(t, lp, DecodeConfig(tempo_weight=2.0))
    near_60 = grid_from_logp(t, lp, DecodeConfig(tempo_weight=2.0, tempo_mu=math.log(60.0)))
    assert np.allclose(np.diff(near_120["q"]), 1.0) and np.allclose(np.diff(near_60["q"]), 0.5)


def test_level_choice():
    """The note-value level of a piece: time signatures at halved / doubled values (only ones the
    classes have), the grid scaled with its bar lines kept, the written tempo, the right-level
    score; choose_level off by default, the prior moving a halved grid back to its level, the stay
    bonus and a tie keeping it; the tempo head's estimate choosing alone."""
    from src.meter import LEVELS, choose_level, level_ratio, level_scores, piece_qpm, scale_grid, scaled_ts
    assert LEVELS[0] == 1
    assert scaled_ts("3/8", 2) == "3/4" and scaled_ts("2/4", 2) == "2/2" and scaled_ts("6/4", Fraction(1, 2)) == "6/8"
    assert scaled_ts("2/2", 2) is None and scaled_ts("9/8", 2) is None and scaled_ts("4/4", 1) == "4/4"
    # a 3/4 piece at 120 quarters per minute (a quarter every 0.5 s), decoded at halved note values (3/8)
    t = 0.5 * np.arange(25)
    true = {"t": t, "q": np.arange(25.0), "down": np.arange(25) % 3 == 0, "ts": np.full(25, TS_CLASSES.index("3/4")),
            "ts_names": ["3/4"] * 25, "key": np.zeros(25)}
    half = grid_from_states(t, [(TS_CLASSES.index("3/8"), (6 * k) % 18) for k in range(25)])
    assert half["bars"][0] == "3/8" and np.allclose(np.diff(half["q"]), 0.5)
    assert abs(piece_qpm(half) - 60.0) < 1e-9 and abs(level_ratio(half, true) - 0.5) < 1e-9
    assert grid_scores(half, true)["level_ok"] == 0.0 and grid_scores(half, true)["ts_acc"] == 0.0
    up = scale_grid(half, 2)
    assert up["bars"] == ["3/4"] * len(half["bars"]) and np.allclose(up["bar_t"], half["bar_t"]) and up["level"] == 2.0
    assert grid_scores(up, true) == {"beat_f": 1.0, "downbeat_f": 1.0, "ts_acc": 1.0, "level_ok": 1.0}
    assert scale_grid(half, 1) is half and scale_grid(dict(half, bars=["9/8"]), 2) is None
    assert choose_level(half, DecodeConfig())["level"] == 1.0                      # off by default
    prior = DecodeConfig(level_weight=1.0)                                          # 60 qpm is far from 116, 120 is near
    assert set(level_scores(half, prior)) == {Fraction(1), Fraction(2)}            # 3/16 is no class
    assert choose_level(half, prior)["bars"][0] == "3/4"
    assert choose_level(half, DecodeConfig(level_weight=1.0, level_stay=10.0))["bars"][0] == "3/8"   # the bonus wins
    tie = DecodeConfig(level_weight=1.0, level_mu=math.log(60.0 * math.sqrt(2)))   # exactly between 60 and 120
    assert choose_level(half, tie)["level"] == 1.0                                  # a tie keeps the decoded level
    head = DecodeConfig(level_model_weight=1.0)
    assert choose_level(half, head)["level"] == 1.0                                 # no estimate: nothing to choose by
    assert choose_level(half, head, np.array([math.log(30.0), 0.2]))["level"] == 1.0   # no 3/16: the nearest there is
    down = choose_level(up, head, np.array([math.log(61.0), 0.2]))                     # 3/4 at 120 qpm, estimate 61
    assert down["level"] == 0.5 and down["bars"][0] == "3/8" and np.allclose(down["q"], half["q"])
    assert choose_level(half, head, np.array([math.log(118.0), 0.2]))["bars"][0] == "3/4"


def test_tempo_head():
    """The piece-level written-tempo head: off by default (old checkpoints load as before); on, an
    estimate per piece (mean near the training centre at first, a positive std), its target from
    the annotations over the sample's beats, its loss counted, predict_logp handing it to the
    decoder; a sample without a target adds no loss."""
    import torch
    from src.meter import TEMPO_CENTRE, MeterConfig, meter_loss, written_tempo
    a = annotation_grid(ann(bars=8))
    assert abs(written_tempo(a) - math.log(120.0)) < 1e-9 and math.isnan(written_tempo(a, 100.0, 101.0))
    tg = targets(a["t"][2:10], a)
    assert abs(float(tg["tempo"][0]) - math.log(120.0)) < 1e-6
    off = build_meter_model(MeterConfig(d_model=32, layers=1, n_heads=4, ff=64).to_dict())
    assert not hasattr(off, "tempo") and "tempo" not in off(torch.zeros(1, 5, N_BEAT_FEATURES), torch.ones(1, 5, dtype=torch.bool))
    torch.manual_seed(0)
    m = build_meter_model(MeterConfig(d_model=32, layers=1, n_heads=4, ff=64, tempo_head=True).to_dict())
    assert {k for k in m.state_dict() if k.startswith("tempo.")} and set(off.state_dict()) < set(m.state_dict())
    notes = np.array([[60, 0.1 * k, 0.1 * k + 0.2, 70] for k in range(40)], np.float32)
    s1 = sample({"notes": notes, "gt": a}, np.random.default_rng(0))
    s2 = sample({"notes": notes, "gt": dict(a, t=a["t"] + 100.0)}, np.random.default_rng(0))   # no annotations under it
    x, mask, nin, tgs = collate([s1, (s2[0], {**s2[1], "tempo": np.array([np.nan], np.float32)}, s2[2], s2[3])])
    assert torch.isfinite(tgs["tempo"][0]) and torch.isnan(tgs["tempo"][1])
    out = m(x, mask, nin)
    assert out["tempo"].shape == (2, 2) and abs(float(out["tempo"][0, 0].detach()) - TEMPO_CENTRE) < 1.0
    loss, parts = meter_loss(out, tgs)
    assert "tempo" in parts and math.isfinite(parts["tempo"]) and abs(float(loss) - sum(parts.values())) < 1e-4
    _, no_tempo = meter_loss(out, {k: v for k, v in tgs.items() if k != "tempo"})
    assert "tempo" not in no_tempo
    lp = predict_logp(m, notes, a["t"], a["down"])
    assert lp["tempo"].shape == (2,) and lp["tempo"][1] > 0 and lp["bar"].shape == (len(a["t"]), 2)
    assert "tempo" not in predict_logp(off, notes, a["t"], a["down"])


def test_window_frame():
    """A decoded grid in a window's frame: its first bar is the bar its first note falls in; a
    window takes the grid of the segment whose span holds it."""
    from scripts.eval_labeller import grid_for
    gt = annotation_grid(ann(bars=6))            # pickup at 0 s, bars of 4 quarters from 0.5 s
    tg = targets(gt["t"], gt)
    g = grid_from_states(gt["t"], decode(onehot(tg["bar"], 2), onehot(tg["step"], len(Q_STEPS)),
                                         onehot(tg["ts"], len(TS_CLASSES))))
    perf_t0 = 2.0                                 # the window starts 2 s in: bar 1 starts at 2.5 s
    t, q, win = window_frame(g, perf_t0, np.array([-0.3, 0.75, 1.0]), key=2)
    assert win.bars[0] == ("4/4", 2) and not win.opens_with_bar and win.start_pos is None
    k = int(np.argmin(np.abs(t - 0.5)))           # the bar line at 2.5 s is quarter 0 of the window
    assert abs(q[k]) < 1e-9 and abs(t[k] - 0.5) < 1e-9, (t, q)
    toks = render(win, [Placed(0, Fraction(1, 2), 60, Fraction(1, 2), 0)])
    assert toks[:4] == ["ts_4/4", "key_2", "pos_0", "sub_1/2"], toks   # mid-bar: no leading bar line
    t2, q2, win2 = window_frame(g, perf_t0, np.array([0.5]), key=0)  # first note on the bar line
    assert win2.opens_with_bar
    # only the bars its notes span, not the rest of the piece (each would render as an empty bar)
    assert len(win.bars) == 1 and len(win2.bars) == 1, (win.bars, win2.bars)
    t3, q3, win3 = window_frame(g, perf_t0, np.array([0.5, 2.6, 4.6]), key=0)  # notes at 2.5 / 4.6 / 6.6 s: bars at 2.5 / 4.5 / 6.5 s
    assert len(win3.bars) == 3, win3.bars
    # which of the piece's bars the window starts with (V2Pipeline renders every window into them):
    # bar 0 holds the pickup beat (0 s), bar 1 starts at 0.5 s, bar 2 at 2.5 s
    assert (win.first_bar, win2.first_bar, win3.first_bar) == (2, 2, 2), (win.first_bar, win2.first_bar)
    assert window_frame(g, 0.0, np.array([0.1]), key=0)[2].first_bar == 0    # the pickup bar
    segs = [dict(g, span=(0.0, 10.0), name="a"), dict(g, span=(30.0, 50.0), name="b")]
    assert grid_for(segs, 5.0)["name"] == "a" and grid_for(segs, 41.0)["name"] == "b"
    assert grid_for(segs, 26.0)["name"] == "b" and grid_for([g], 99.0) is g   # the nearest; no span = everywhere


def test_scripts_end_to_end():
    """Build the alignment ground truth of two ASAP test performances and check it against their
    annotations; train a tiny meter model, tune its decoder and evaluate it (grids with keys and
    spans, per-performance scores); one without the note encoder (the default) stopped by its time
    limit and resumed; the oracle grids; a tiny labeller (and one trained with GridNoise) on the
    model's grids, the oracle's and the true grid: the results land as modes grid_meter_<run> /
    grid_true and every grid's predictions are kept."""
    if not (D / "meter" / "train.pkl").exists() or not (D / "test" / "notelabels.npz").exists():
        print("  (skipped: data/processed/meter or test/notelabels.npz missing)")
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        import pickle
        from scripts.build_meter_data import check, from_alignment
        from scripts.eval_labeller import main as eval_labeller
        from scripts.eval_meter import main as eval_meter
        from scripts.train_labeller import main as train_labeller
        from scripts.train_meter import main as train_meter
        test = pickle.load(open(D / "meter" / "test.pkl", "rb"))
        only = {p["performance"] for p in test[:2]}
        al = from_alignment(D / "test", D / "vocab.json", "asap", 1, only=only)
        assert al and {p["performance"] for p in al} == only and all(p["span"][0] < p["gt"]["t"][0] for p in al)
        assert all(len(p["notes"]) and not p["real_notes"] for p in al)
        res = check(al, test)
        assert res["beat_f"] > 0.5 and res["ts_agreement"] > 0.9, res
        tiny = ["--steps", "6", "--eval-every", "3", "--batch", "2", "--chunk", "64", "--d-model", "64", "--layers", "2",
                "--device", "cpu", "--normal-priority", "--runs-dir", str(tmp), "--warmup", "2", "--workers", "0"]
        train_meter(["--name", "m", "--note-encoder"] + tiny)
        assert (tmp / "m" / "best.pt").exists() and json.loads((tmp / "m" / "status.json").read_text())["phase"] == "done"
        cfg = json.loads((tmp / "m" / "config.json").read_text())
        assert cfg["model"]["n_note_features"] == N_NOTE_FEATURES and cfg["data"]["train_by_source"]["asap"] > 0
        train_meter(["--name", "m0", "--max-hours", "1e-9"] + tiny)  # the time limit: one step, then stop
        st = json.loads((tmp / "m0" / "status.json").read_text())
        assert st["phase"] == "train" and st["step"] == 1 and not (tmp / "m0" / "best.pt").exists(), st
        train_meter(["--name", "m0", "--resume"] + tiny)
        st = json.loads((tmp / "m0" / "status.json").read_text())
        assert st["phase"] == "done" and st["step"] == 6 and (tmp / "m0" / "best.pt").exists(), st
        assert json.loads((tmp / "m0" / "config.json").read_text())["model"]["n_note_features"] == 0
        eval_meter(["--run", "m", "--conditions", "clean", "--device", "cpu", "--runs-dir", str(tmp), "--tune",
                    "--tune-max", "2"])
        dec = json.loads((tmp / "m" / "decode.json").read_text())
        assert set(dec["decode"]) == set(DecodeConfig().to_dict()) and dec["select_tuned"] >= dec["select_default"]
        grids = pickle.load(open(tmp / "m" / "grids_clean.pkl", "rb"))
        g = next(iter(grids.values()))
        assert {"t", "q", "bars", "bar_q", "bar_t", "key", "performance", "span"} <= set(g) and len(g["bars"]) > 0
        res = json.loads((tmp / "m" / "eval_meter.json").read_text())["clean"]
        assert 0 <= res["grid_downbeat_f"] <= 1 and 0 < res["tracker_beat_f"] <= 1
        assert {"default", "piece_mean", "ts_confusion", "decode"} <= set(res) and res["pieces"] <= res["performances"]
        per = json.loads((tmp / "m" / "meter_scores_clean.json").read_text())
        assert len(per) == res["performances"] and "true_ts" in per[0]
        (tmp / "mtp").mkdir()   # the same model, its decoder tuned with the written-tempo prior too
        for f in ("best.pt", "config.json"):
            shutil.copy(tmp / "m" / f, tmp / "mtp" / f)
        eval_meter(["--run", "mtp", "--conditions", "clean", "--device", "cpu", "--runs-dir", str(tmp), "--tune",
                    "--tune-max", "2", "--tempo-prior"])
        tdec = json.loads((tmp / "mtp" / "decode.json").read_text())
        assert {"tempo_weight", "tempo_mu", "tempo_sigma"} <= set(tdec["decode"])
        assert tdec["configs_tried"] > dec["configs_tried"] and tdec["select_tuned"] >= dec["select_tuned"] - 1e-12
        assert "grid_level_ok" in json.loads((tmp / "mtp" / "eval_meter.json").read_text())["clean"]
        # the tempo head: trained alone on m0's frozen trunk (only its weights move), then with the trunk
        import torch
        base = torch.load(tmp / "m0" / "best.pt", weights_only=False)["model"]
        train_meter(["--name", "th", "--init", str(tmp / "m0" / "best.pt"), "--tempo-head", "--freeze-trunk",
                     "--select", "tempo"] + tiny)
        th = torch.load(tmp / "th" / "best.pt", weights_only=False)
        assert th["config"]["tempo_head"] and th["config"]["d_model"] == 64
        assert all(torch.equal(base[k], th["model"][k]) for k in base), "a frozen weight moved"
        log = (tmp / "th" / "log.csv").read_text().splitlines()
        assert "asap_head_level_ok" in log[0] and "asap_tempo_mae" in log[0], log[0]
        train_meter(["--name", "ft", "--init", str(tmp / "m0" / "best.pt"), "--tempo-head", "--select", "both"] + tiny)
        ft = torch.load(tmp / "ft" / "best.pt", weights_only=False)["model"]
        assert any(not torch.equal(base[k], ft[k]) for k in base)                     # the trunk trained too
        for bad in (["--name", "x1", "--freeze-trunk", "--tempo-head"], ["--name", "x2", "--select", "tempo"],
                    ["--name", "x3", "--init", str(tmp / "m" / "best.pt"), "--tempo-head", "--no-note-encoder"]):
            try:
                train_meter(bad + tiny)
            except SystemExit as e:
                assert e.code != 0
            else:
                if bad[1] != "x3":
                    raise AssertionError(f"{bad}: not refused")
        assert json.loads((tmp / "x3" / "config.json").read_text())["model"]["n_note_features"] == N_NOTE_FEATURES
        # the level tuned on its own, from another run's decoder (the head's estimate searched too)
        eval_meter(["--run", "th", "--conditions", "clean", "--device", "cpu", "--runs-dir", str(tmp), "--level",
                    "--decode-from", str(tmp / "mtp" / "decode.json"), "--tune-max", "2"])
        ldec = json.loads((tmp / "th" / "decode.json").read_text())
        assert {k: ldec["decode"][k] for k in ("tempo_weight", "bar_weight")} == \
            {k: tdec["decode"][k] for k in ("tempo_weight", "bar_weight")}          # the rest kept
        assert ldec["level_select_tuned"] >= ldec["level_select_decoded"] - 1e-12 and ldec["decode_from"]
        assert ldec["level_configs_tried"] > 1 and "level_ok_tuned" in ldec
        lres = json.loads((tmp / "th" / "eval_meter.json").read_text())["clean"]
        assert 0 <= lres["grid_level_ok"] <= 1 and lres["level_changed"] >= 0
        eval_meter(["--run", "m0", "--conditions", "clean", "--device", "cpu", "--runs-dir", str(tmp), "--tune",
                    "--tune-max", "2", "--level"])                                     # no head: the prior and the bonus
        assert {"level_weight", "level_stay"} <= set(json.loads((tmp / "m0" / "decode.json").read_text())["decode"])
        try:
            eval_meter(["--run", "ft", "--conditions", "clean", "--device", "cpu", "--runs-dir", str(tmp), "--level"])
        except SystemExit as e:
            assert "decode-from" in str(e)
        else:
            raise AssertionError("--level without a tuned decoder: refused")
        eval_meter(["--run", "oracle", "--oracle", "--conditions", "clean", "--device", "cpu", "--runs-dir", str(tmp)])
        orc = json.loads((tmp / "oracle" / "eval_meter.json").read_text())["clean"]
        assert json.loads((tmp / "oracle" / "config.json").read_text())["arch"] == "meter_oracle"
        assert orc["tracker_beat_f"] == res["tracker_beat_f"]  # the same tracked beats as the model's
        assert orc["grid_ts_acc"] > 0.9 and orc["grid_downbeat_f"] > max(res["grid_downbeat_f"], 0.7), orc
        try:
            eval_meter(["--run", "oracle", "--oracle", "--tune", "--runs-dir", str(tmp)])
        except SystemExit:
            pass
        else:
            raise AssertionError("--oracle --tune should refuse: no model to tune for")
        train_labeller(["--name", "nl", "--size", "small", "--max-steps", "2", "--eval-every", "2",
                        "--eval-loss-windows", "4", "--eval-decode-windows", "2", "--workers", "0", "--device", "cpu",
                        "--batch-notes", "2048", "--warmup", "1", "--normal-priority", "--runs-dir", str(tmp)])
        eval_labeller(["--run", "nl", "--grid", "meter:m", "--conditions", "clean,real", "--limit-perfs", "1",
                       "--device", "cpu", "--normal-priority", "--runs-dir", str(tmp)])
        eval_labeller(["--run", "nl", "--grid", "meter:oracle", "--conditions", "clean", "--limit-perfs", "1",
                       "--device", "cpu", "--normal-priority", "--runs-dir", str(tmp)])
        eval_labeller(["--run", "nl", "--grid", "true", "--conditions", "clean", "--limit-perfs", "1",
                       "--device", "cpu", "--normal-priority", "--runs-dir", str(tmp)])
        out = json.loads((tmp / "nl" / "eval_pieces_best.json").read_text())
        r = out["clean"]["grid_meter_m"]
        assert 0 <= r["onset_f1"] <= 1 and 0 <= r["key_acc"] <= 1 and "real" not in out  # no real grids here
        assert 0 <= out["clean"]["grid_meter_oracle"]["onset_scaled_f1"] <= 1
        assert r["onset_scaled_f1"] >= r["onset_shift_f1"] >= r["onset_f1"] - 1e-12  # each forgives more
        modes = {json.loads(ln)["mode"] for ln in open(tmp / "nl" / "pred_pieces_best_clean.jsonl")}
        assert modes == {"grid_meter_m", "grid_meter_oracle", "grid_true"}, modes   # each grid's predictions kept
        train_labeller(["--name", "gn", "--size", "small", "--max-steps", "2", "--eval-every", "2", "--grid-noise", "0.8",
                        "--eval-loss-windows", "4", "--eval-decode-windows", "2", "--workers", "0", "--device", "cpu",
                        "--batch-notes", "2048", "--warmup", "1", "--normal-priority", "--runs-dir", str(tmp),
                        "--select", "onset_f1"])
        gall = json.loads((tmp / "gn" / "config.json").read_text())
        gcfg = gall["data"]
        assert gcfg["grid_noise"] == 0.8 and gcfg["beat_noise"] == "configs/beat_noise.json", gcfg
        assert gall["args"]["select"] == "onset_f1" and (tmp / "gn" / "best.pt").exists()
        assert -1.0 <= json.loads((tmp / "gn" / "status.json").read_text())["best"] <= 0.0   # minus an onset F1
        assert (tmp / "gn" / "vocab.json").exists() and (tmp / "nl" / "vocab.json").exists()   # pinned
        assert json.loads((tmp / "nl" / "config.json").read_text())["data"]["grid_noise"] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_meter_batch():
    """pace/meter.sh: data + check + tests; every meter run after them with its own options, then
    its decoder tuned and evaluated; the first meter model re-evaluated; each labeller on each
    meter model's grids after that meter job and after the labeller jobs still in the queue."""
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "meter.sh", tmp / "pace")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "squeue").write_text('#!/bin/bash\necho 901\necho 902\n')  # two labeller jobs queued
        for f in (tmp / "bin").iterdir():
            f.chmod(0o755)

        def submit(env=None):
            if (tmp / "jobs.txt").exists():
                (tmp / "jobs.txt").unlink()
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/meter.sh'], cwd=tmp,
                               env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", **(env or {})),
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            return [(ln.split("\t")[0], ln.split("\t")[1]) for ln in (tmp / "jobs.txt").read_text().splitlines()]
        jobs = submit()
        by = {cmd.split("-J ")[1].split()[0]: (jid, cmd) for jid, cmd in jobs}
        prep = by["metprep"]
        assert "build_meter_data.py --workers 8 --check" in prep[1] and "tests.test_meter" in prep[1]
        meters = ["meter_v2", "meter_v2_s1", "meter_v2_nonote", "meter_v2_asap"]
        want = {"meter_v2": "--name meter_v2 --note-encoder --workers", "meter_v2_s1": "--note-encoder --seed 1",
                "meter_v2_nonote": "--no-note-encoder", "meter_v2_asap": "--note-encoder --no-periscope"}
        for m in meters:
            jid, cmd = by["met_" + m]
            assert f"--dependency=afterok:{prep[0]}" in cmd and f"train_meter.py --name {m}" in cmd and want[m] in cmd
            assert f"eval_meter.py --run {m} --tune" in cmd
        base = by["metev_meter_base"]
        assert "eval_meter.py --run meter_base --tune" in base[1] and f"afterok:{prep[0]}" in base[1]
        labelled = sorted(v for v in by if v.startswith("v2_"))
        assert labelled == sorted([f"v2_nl_base@{m}" for m in meters + ["meter_base"]]
                                  + ["v2_nl_large@meter_v2", "v2_nl_large@meter_base"]), labelled
        for v in labelled:
            jid, cmd = by[v]
            m = v.split("@")[1]
            dep = by["met_" + m][0] if m != "meter_base" else base[0]
            assert f"--dependency=afterok:{dep},afterany:901:902" in cmd and f"--grid meter:{m}" in cmd
            assert "--conditions clean,synthetic,real,periscope" in cmd
        # SKIP_PREP: no data job, the meter jobs start at once; a subset of the runs
        jobs = submit({"SKIP_PREP": "1", "METERS": "meter_v2", "LARGE_ON": ""})
        names = [cmd.split("-J ")[1].split()[0] for _, cmd in jobs]
        assert names == ["met_meter_v2", "metev_meter_base", "v2_nl_base@meter_v2", "v2_nl_base@meter_base"], names
        assert "--dependency" not in jobs[0][1]


def test_refine_grid():
    """Beat times onto the notes (refine_grid): a rubato 4/4 performance with a chord on every beat
    and a single note between, tracked with 15 ms jitter and every fifth beat missed, decoded by
    the oracle: snapping brings the anchors to the chords, filling adds the missed beats at theirs,
    times stay increasing, the bars stay, and with both off the grid is untouched."""
    from src.meter import DecodeConfig, chord_onsets, grid_beats, grid_from_logp, oracle_logp, refine_grid
    rng = np.random.default_rng(0)
    k = np.arange(64)
    true_t = 1.0 + np.cumsum(np.r_[0, 0.5 + 0.08 * np.sin(k[1:] / 5.0)])         # rubato
    chords = np.concatenate([true_t + d for d in (-0.008, 0.0, 0.012)])          # a spread chord per beat
    offbeats = (true_t[:-1] + true_t[1:]) / 2 + 0.03
    onsets = np.sort(np.concatenate([chords, offbeats]))
    assert np.allclose(chord_onsets(onsets)[::2], true_t + 0.0, atol=0.005)       # one time per chord
    gt = {"t": true_t, "q": k.astype(float), "down": k % 4 == 0, "ts": np.full(64, TS_CLASSES.index("4/4")),
          "key": np.zeros(64, np.int64), "ts_names": ["4/4"] * 64}
    keep = k % 5 != 2                                                             # every fifth beat missed
    bt = true_t[keep] + rng.normal(0, 0.015, keep.sum())
    grid = grid_from_logp(bt, oracle_logp(bt, gt), DecodeConfig())
    assert refine_grid(grid, onsets, DecodeConfig()) is grid                      # off: untouched
    cfg = DecodeConfig(snap_s=0.05, fill_tol=0.25)
    ref = refine_grid(grid, onsets, cfg)
    assert ref["refined"]["filled"] == int((~keep[1:-1]).sum()) and ref["refined"]["snapped"] > 40
    assert np.all(np.diff(ref["t"]) > 0) and ref["bars"] == grid["bars"] and np.allclose(ref["bar_q"], grid["bar_q"])

    def worst(g):  # the grid's notated beats against the true ones
        t, _ = grid_beats(g)
        return float(np.max(np.abs(t - np.interp(np.arange(len(t)), k, true_t))))
    assert worst(ref) < 0.01 < worst(grid), (worst(ref), worst(grid))
    # snapping alone moves anchors but adds none; filling alone adds the missed beats
    assert refine_grid(grid, onsets, DecodeConfig(snap_s=0.05))["refined"]["filled"] == 0
    assert refine_grid(grid, onsets, DecodeConfig(fill_tol=0.25))["refined"]["snapped"] == 0
    # nothing played where a beat is missing: no anchor is made up
    silent = onsets[np.abs(onsets[:, None] - true_t[~keep][None]).min(1) > 0.05]
    assert refine_grid(grid, silent, DecodeConfig(fill_tol=0.25))["refined"]["filled"] == 0
    # settings round-trip through decode.json
    assert DecodeConfig.from_dict(cfg.to_dict()).snap_s == 0.05


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
