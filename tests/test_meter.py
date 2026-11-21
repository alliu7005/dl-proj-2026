"""Tests for the meter model (v2, Phase 2) and its grids in the note labeller (Phase 3).

    python -m tests.test_meter        # or: pytest tests/test_meter.py

Covers the annotated grid, the per-beat targets under tracker errors, the consistent decoding (a
perfect prediction gives back the annotated grid; bars always add up to their time signature),
the grid's scores, the beat features, the key signature estimate, an estimated grid in a window's
frame, the scripts end to end on the local data (skipped without data/processed/meter), and the
PACE job graph.
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
from src.meter import (Q_STEPS, TS_CLASSES, annotation_grid, beat_features, decode, grid_beats,  # noqa: E402
                       grid_from_states, grid_scores, targets)
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


def test_beat_features_and_key():
    t = 0.5 * np.arange(8)
    notes = np.array([[36, 0.0, 0.4, 100], [60, 0.0, 0.4, 70], [64, 0.25, 0.4, 50], [67, 1.0, 1.4, 90]], np.float32)
    f = beat_features(notes, t, np.arange(8) % 4 == 0)
    assert f.shape[0] == 8 and np.isfinite(f).all()
    assert f[0, 3] == 1 and f[1, 3] == 0                      # the tracker's downbeat flag
    assert f[0, 5 + 6] > 0                                    # a note half way through beat 0
    scale = lambda tonic, minor=False: np.array(  # noqa: E731
        [[tonic + s, k, k + 1, 80] for k, s in enumerate(([0, 2, 3, 5, 7, 8, 10, 12, 7, 3, 0] if minor else
                                                          [0, 2, 4, 5, 7, 9, 11, 12, 7, 4, 0]))], np.float32)
    assert estimate_key(scale(60)) == 0 and estimate_key(scale(67)) == 1 and estimate_key(scale(62)) == 2
    assert estimate_key(scale(57, minor=True)) == 0 and estimate_key(scale(65)) == -1


def test_window_frame():
    """A decoded grid in a window's frame: its first bar is the bar its first note falls in."""
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


def test_scripts_end_to_end():
    """Train a tiny meter model, evaluate it (grids with keys), and run a tiny labeller on its
    grids: the result lands as mode grid_meter_<run> next to grid_true."""
    if not (D / "meter" / "train.pkl").exists() or not (D / "test" / "notelabels.npz").exists():
        print("  (skipped: data/processed/meter or test/notelabels.npz missing)")
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        from scripts.eval_labeller import main as eval_labeller
        from scripts.eval_meter import main as eval_meter
        from scripts.train_labeller import main as train_labeller
        from scripts.train_meter import main as train_meter
        train_meter(["--name", "m", "--steps", "6", "--eval-every", "3", "--batch", "2", "--chunk", "64", "--d-model", "64",
                     "--layers", "2", "--device", "cpu", "--normal-priority", "--runs-dir", str(tmp), "--warmup", "2"])
        assert (tmp / "m" / "best.pt").exists() and json.loads((tmp / "m" / "status.json").read_text())["phase"] == "done"
        eval_meter(["--run", "m", "--conditions", "clean", "--device", "cpu", "--runs-dir", str(tmp)])
        import pickle
        grids = pickle.load(open(tmp / "m" / "grids_clean.pkl", "rb"))
        g = next(iter(grids.values()))
        assert {"t", "q", "bars", "bar_q", "bar_t", "key"} <= set(g) and len(g["bars"]) > 0
        res = json.loads((tmp / "m" / "eval_meter.json").read_text())["clean"]
        assert 0 <= res["grid_downbeat_f"] <= 1 and 0 < res["tracker_beat_f"] <= 1
        train_labeller(["--name", "nl", "--size", "small", "--max-steps", "2", "--eval-every", "2",
                        "--eval-loss-windows", "4", "--eval-decode-windows", "2", "--workers", "0", "--device", "cpu",
                        "--batch-notes", "2048", "--warmup", "1", "--normal-priority", "--runs-dir", str(tmp)])
        eval_labeller(["--run", "nl", "--grid", "meter:m", "--conditions", "clean,real", "--limit-perfs", "1",
                       "--device", "cpu", "--normal-priority", "--runs-dir", str(tmp)])
        out = json.loads((tmp / "nl" / "eval_pieces_best.json").read_text())
        r = out["clean"]["grid_meter_m"]
        assert 0 <= r["onset_f1"] <= 1 and 0 <= r["key_acc"] <= 1 and "real" not in out  # no real grids here
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_meter_batch():
    """pace/meter.sh: data + tests, the meter model after them, each labeller on its grids after
    the meter model and after the labeller jobs still in the queue."""
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
        r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/meter.sh'], cwd=tmp,
                           env=dict(os.environ, SUBMITS=str(tmp / "jobs.txt"), USER="me"), capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        jobs = [ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()]
        prep, met, v2 = jobs[0], jobs[1], jobs[2:]
        assert "build_meter_data.py" in prep[1] and "tests.test_meter" in prep[1]
        assert f"afterok:{prep[0]}" in met[1] and "train_meter.py --name meter_base" in met[1] and "eval_meter.py" in met[1]
        assert [j[1].split("-J ")[1].split()[0] for j in v2] == ["v2_nl_base", "v2_nl_large"]
        for j in v2:
            assert f"--dependency=afterok:{met[0]},afterany:901:902" in j[1] and "--grid meter:meter_base" in j[1]


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
