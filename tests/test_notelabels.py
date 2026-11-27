"""Tests for the note labeller (v2): labels, rendering, grid features, dataset, model, training and
evaluation scripts, and the PACE job graph.

    python -m tests.test_notelabels        # or: pytest tests/test_notelabels.py

The data tests run on data/processed/test (skipped without it); pace/labeller.sh runs this file on
the PACE data build before any training starts.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.batch_env import batch_env  # noqa: E402
from src.eval.metrics import Accumulator  # noqa: E402
from src.notelabels import (IGNORE, LABEL_COLS, N_FLAGS, LabelSpace, Placed, Window, align_notes,  # noqa: E402
                            bar_of, flags_of, grid_features, oracle_placed, render, score_notes, targets,
                            transfer_rows, unflag)
from src.preprocessing.vocab import Vocab  # noqa: E402

D = ROOT / "data" / "processed"
V = D / "vocab.json"


def have_data() -> bool:
    return (D / "test" / "index.npy").exists() and V.exists()


def test_ties_and_render():
    """A note held across a bar line: its total length is read through the tie, and rendering
    writes it back as the note + a tied continuation; inside its bar it stays one note."""
    toks = ["bar", "ts_4/4", "key_0", "pos_0", "p_60", "d_2", "pos_2", "p_64", "d_2",
            "bar", "pos_0", "p_64", "d_1", "tie", "pos_1", "p_67", "d_3"]
    notes, win = score_notes(toks)
    sounding = [n for n in notes if not n.tie]
    assert [(n.pitch, n.total) for n in sounding] == [(60, 2), (64, 3), (67, 3)], [(n.pitch, n.total) for n in sounding]
    placed = [Placed(n.bar, n.pos, n.pitch, n.total, n.staff) for n in sounding]
    assert render(win, placed) == toks, render(win, placed)
    # a window starting mid-bar keeps no leading bar line
    # (a span starting mid-bar states its first staff, as tokenize_bars writes it)
    mid = ["ts_3/4", "key_-1", "pos_1", "staff_0", "p_62", "d_1", "pos_2", "p_65", "d_1", "bar", "pos_0", "p_67", "d_3"]
    notes, win = score_notes(mid)
    assert not win.opens_with_bar and win.start_pos == 1
    assert render(win, oracle_placed(notes, {k: k for k in range(len(notes))})) == mid


def test_label_space_and_flags():
    space = LabelSpace(Vocab.load(str(V)).itos)
    for pos in (Fraction(0), Fraction(5, 2), Fraction(7, 3), Fraction(13, 4)):
        k, sub = space.pos_classes(pos)
        assert k >= 0 and space.pos_value(k, sub) == pos, (pos, k, sub)
    assert space.pos_classes(Fraction(17)) == (IGNORE, IGNORE)  # past pos_15
    for d in (Fraction(1, 4), Fraction(3), Fraction(0)):
        assert space.dur_value(space.dur_class(d)) == d
    assert space.dur_value(space.dur_class(Fraction(1001, 7))) == max(space.durs)  # nearest
    for f in [(False, False, ()), (True, False, ("trill",)), (False, True, ("fermata", "turn"))]:
        assert unflag(flags_of(*f)) == f


def test_grid_features_and_bars():
    """A 4/4 grid of quarter beats at 0.5 s: notes on and between beats get their bar and
    position, before and after the grid by extension."""
    win = Window(bars=[("4/4", 0), ("3/4", 0)])
    gt = 0.5 * np.arange(8)                       # beats 0..7 = quarters 0..7 (bar 0: 0-4, bar 1: 4-7)
    gq = np.arange(8, dtype=float)
    on = np.array([0.0, 0.75, 2.0, 2.25, 3.4, -0.5])
    f, bar, pos = grid_features(on, gt, gq, win)
    assert f.shape == (6, 17) and np.all(f[:, -1] == 1)
    assert bar.tolist() == [0, 0, 1, 1, 1, 0], bar
    assert np.allclose(pos, [0, 1.5, 0, 0.5, 2.8, -1.0]), pos
    b, p = bar_of(np.array([7.0, 10.5]), win)     # past the window's bars: its last bar repeated
    assert b.tolist() == [2, 3] and np.allclose(p, [0, 0.5]), (b, p)
    f, bar, pos = grid_features(on, np.zeros(1), np.zeros(1), win)  # no grid
    assert not f.any() and not bar.any()


def test_grid_noise():
    """GridNoise moves a true grid's anchors as the tracker would, never its positions' meaning:
    p = 0 changes nothing; every draw stays increasing in time and position, covers the window and
    keeps the anchors on the true line within the jitter; a doubled tempo adds anchors at exact
    in-between positions; spurious anchors sit at snapped positions of their beat."""
    from src.beats import BeatNoise
    from src.notelabels import EXTRA_SNAP, GridNoise
    t = 0.5 * np.arange(24)
    q = np.arange(24, dtype=float)
    fitted = GridNoise.from_json(ROOT / "configs" / "beat_noise.json", p=1.0)
    t0, q0 = GridNoise(fitted.bn, p=0.0)(t, q, np.random.default_rng(0))
    assert np.array_equal(t0, t) and np.array_equal(q0, q)
    for seed in range(40):
        tt, qq = fitted(t, q, np.random.default_rng(seed))
        assert len(tt) >= 2 and np.all(np.diff(tt) > 0.03) and np.all(np.diff(qq) > 0), seed
        assert qq[0] <= 1e-9 + q[0] + 1 and qq[-1] >= q[-1] - 1, (seed, qq[[0, -1]])  # still covers the window
    # only timing: anchors that are true beats stay within a few jitters of their time
    quiet = BeatNoise(p_double=0.0, p_half=0.0, miss=0.0, extra=0.0, offset=0.0, jitter=0.01)
    tt, qq = GridNoise(quiet, 1.0)(t, q, np.random.default_rng(1))
    assert np.array_equal(qq, q) and np.abs(tt - t).max() < 0.06
    double = BeatNoise(p_double=1.0, p_half=0.0, double_k=[1.0, 0.0, 0.0], miss=0.0, extra=0.0, offset=0.0, jitter=0.0)
    tt, qq = GridNoise(double, 1.0)(t, q, np.random.default_rng(2))
    assert len(tt) == 2 * len(t) - 1 and np.allclose(np.diff(qq), 0.5) and np.allclose(np.diff(tt), 0.25)
    extra = BeatNoise(p_double=0.0, p_half=0.0, miss=0.0, extra=0.3, offset=0.0, jitter=0.0)
    tt, qq = GridNoise(extra, 1.0)(t, q, np.random.default_rng(3))
    frac = np.round(qq % 1.0, 9)
    assert len(tt) > len(t) and set(frac[frac > 0]) <= {round(x, 9) for x in EXTRA_SNAP}, set(frac)
    assert np.all(np.diff(tt) > 0.03) and np.all(np.diff(qq) > 0)
    # more spurious beats than real ones (extra > 0.5, as some of PM2S's profiles): drawn, not a crash
    many = BeatNoise(p_double=0.0, p_half=0.0, miss=0.0, extra=0.73, offset=0.0, jitter=0.0)
    for seed in range(10):
        tt, qq = GridNoise(many, 1.0)(t, q, np.random.default_rng(seed))
        assert len(tt) > len(t) and np.all(np.diff(tt) > 0.03) and np.all(np.diff(qq) > 0), seed
    pm2s = ROOT / "configs" / "beat_noise_pm2s.json"
    if pm2s.exists():  # the fitted PM2S profiles (scripts/fit_pm2s_noise.py): every one draws
        fitted_pm = GridNoise.from_json(pm2s, p=1.0)
        for seed in range(200):
            tt, qq = fitted_pm(t, q, np.random.default_rng(seed))
            assert len(tt) >= 2 and np.all(np.diff(tt) > 0.03) and np.all(np.diff(qq) > 0), seed


def test_targets_and_transfer():
    space = LabelSpace(Vocab.load(str(V)).itos)
    rows = np.full((4, len(LABEL_COLS)), IGNORE, np.int32)
    rows[0] = (1, 2, 5, 2, 1, 2, 1, flags_of(False, True, ("trill",)))   # bar 2, pos 5/2, 1/2, staff 1
    rows[1] = (1, 0, 3, 1, 1, 1, 0, 0)                                   # bar 0 but the grid says bar 2
    rows[2] = (0,) + (IGNORE,) * (len(LABEL_COLS) - 1)                   # delete
    tg, fl = targets(rows, np.array([1, 2, 0, 0]), space)
    assert tg[0].tolist()[:2] == [1, 2] and space.pos_value(tg[0, 2], tg[0, 3]) == Fraction(5, 2)
    assert space.dur_value(tg[0, 4]) == Fraction(1, 2) and tg[0, 5] == 1
    assert fl[0].tolist() == [0, 1, 0, 0, 0, 1, 0]
    assert tg[1, 1] == IGNORE and tg[1, 2] == 3        # more than one bar off: no bar target
    assert tg[2].tolist() == [0] + [IGNORE] * 5 and (fl[2] == IGNORE).all()
    assert tg[3, 0] == IGNORE                          # a context / carried note: nothing at all
    ref = np.array([[60, 0.1, 0.2, 80], [64, 0.5, 0.2, 80]], np.float32)
    noisy = np.array([[60, 0.12, 0.2, 80], [71, 0.3, 0.1, 40]], np.float32)  # moved, dropped, added
    out = transfer_rows(rows[:2], ref, noisy)
    assert out[0].tolist() == rows[0].tolist() and out[1, 0] == 0 and (out[1, 1:] == IGNORE).all()


def test_ceiling_on_data():
    """Rendering every window from its own true labels: close to perfect where the labels reach."""
    if not have_data():
        print("  (skipped: no data/processed/test)")
        return
    from scripts.label_ceiling import main as ceiling
    r, c = ceiling(["--split", "test", "--limit", "60"])
    assert r["onset_f1"] > 0.9 and r["duration_acc"] > 0.97 and r["staff_acc"] > 0.97 and r["ts_acc"] == 1.0, r
    assert c["sounding_matched"] / c["sounding"] > 0.85, c


def ensure_labels(split="test"):
    if not (D / split / "notelabels.npz").exists():
        from src.notelabels import build_split
        build_split(D / split, V, workers=1)


def test_dataset_items():
    """Without noise the item's targets are the stored labels; with context notes those are
    features only; with noise and augmentation the shapes hold and added notes are deletes."""
    if not have_data():
        print("  (skipped: no data/processed/test)")
        return
    ensure_labels()
    from src.dataset import AugmentConfig, ScoreSegmentDataset, collate
    from src.preprocessing.noise import NoiseModel
    ds = ScoreSegmentDataset(str(D / "test"), str(V), notelabels=True, context_before=16, context_after=8)
    stored = np.load(D / "test" / "notelabels.npz")["labels"]
    it = ds[3]
    a0, a1 = ds.store.index[3, 0], ds.store.index[3, 1]
    own = it["roles"].numpy() <= 1
    assert it["notes"].shape[1] == 10 + 17 and it["labels"].shape == (len(it["notes"]), 6)
    keep = it["labels"][:, 0].numpy()
    assert (keep[~own] == IGNORE).all()
    # the window's own notes keep their stored labels (carried-over notes have none, and the dataset
    # drops the ones already released, so compare the labelled ones)
    mine, kept = keep[own], stored[a0:a1, 0]
    assert sorted(mine[mine != IGNORE].tolist()) == sorted(kept[kept != IGNORE].tolist())
    tr = ScoreSegmentDataset(str(D / "test"), str(V), notelabels=True, noise=NoiseModel.from_json("configs/noise_stats.json"),
                             augment=AugmentConfig(), context_before=16, context_after=8)
    b = collate([tr[i] for i in range(6)])
    assert b["labels"].shape[:2] == b["notes"].shape[:2] and b["flags"].shape[-1] == N_FLAGS
    # the tracker's timing on the grid: the grid features change, the notes and every label do not
    from src.notelabels import GridNoise
    plain = ds[3]
    ds.grid_noise = GridNoise.from_json(ROOT / "configs" / "beat_noise.json", p=1.0)
    moved = ds[3]
    ng = 17  # N_GRID_FEATURES, the last columns
    assert torch.equal(moved["notes"][:, :-ng], plain["notes"][:, :-ng]) and torch.equal(moved["roles"], plain["roles"])
    assert not torch.equal(moved["notes"][:, -ng:], plain["notes"][:, -ng:])
    assert torch.equal(moved["labels"][:, [0, 2, 3, 4, 5]], plain["labels"][:, [0, 2, 3, 4, 5]])
    assert torch.equal(ds[3]["notes"], moved["notes"])  # the same draw for the same window and epoch
    assert ((b["labels"][..., 0] == 1) | (b["labels"][..., 0] == 0) | (b["labels"][..., 0] == -1)).all()


def test_model_and_scripts():
    """The labeller trains a few steps (loss and validation run, checkpoints, resume), and the
    evaluation writes a grid_true result the comparison tables read."""
    if not have_data():
        print("  (skipped: no data/processed/test)")
        return
    for split in ("train", "val", "test"):
        if (D / split / "index.npy").exists():
            ensure_labels(split)
    tmp = Path(tempfile.mkdtemp())
    try:
        from scripts.eval_labeller import main as evaluate
        from scripts.train_labeller import main as train
        base = ["--name", "t", "--size", "small", "--max-steps", "4", "--eval-every", "2", "--eval-loss-windows", "8",
                "--eval-decode-windows", "4", "--workers", "0", "--device", "cpu", "--batch-notes", "2048",
                "--warmup", "2", "--normal-priority", "--runs-dir", str(tmp)]
        train(base)
        run = tmp / "t"
        assert (run / "best.pt").exists() and (run / "last.pt").exists()
        assert json.loads((run / "status.json").read_text())["phase"] == "done"
        log = (run / "log.csv").read_text().splitlines()
        assert len(log) == 3 and "val_onset_f1" in log[0] and "val_head_pos_acc" in log[0]
        train(base + ["--resume"])  # finished: nothing to train, no error
        evaluate(["--run", "t", "--runs-dir", str(tmp), "--conditions", "clean", "--limit-perfs", "1",
                  "--device", "cpu", "--normal-priority"])
        res = json.loads((run / "eval_pieces_best.json").read_text())
        r = res["clean"]["grid_true"]
        assert 0 <= r["onset_f1"] <= 1 and "head_staff_acc" in r and "staff_acc" in r and r["limit_perfs"] == 1
        cfg = json.loads((run / "config.json").read_text())
        assert cfg["arch"] == "labeller"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_labeller_batch():
    """pace/labeller.sh: the label job (with this file's tests), then the two labellers after it."""
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "labeller.sh", tmp / "pace")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\t%s\\t%s\\n" "$n" "$NAME" "$ARGS" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "sbatch").chmod(0o755)
        r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/labeller.sh'], cwd=tmp,
                           env=batch_env(SUBMITS=str(tmp / "jobs.txt")), capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        jobs = [ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()]
        prep = jobs[0]
        assert "-J labelprep" in prep[3] and "build_notelabels.py" in prep[3] and "tests.test_notelabels" in prep[3]
        runs = {j[1]: j for j in jobs[1:]}
        assert set(runs) == {"nl_base", "nl_large"}
        assert runs["nl_base"][2] == "--size base" and "--size large" in runs["nl_large"][2]
        assert all(f"--dependency=afterok:{prep[0]}" in j[3] and "train_labeller.sbatch" in j[3] for j in runs.values())


def test_empty_accumulator():
    """No windows (a subset with nothing in it, e.g. real_unseen_audio under --limit-perfs): rates
    are nan, not a ZeroDivisionError (it stopped pace/v2_gangqinpu.sh's tests on PACE)."""
    import math
    r = Accumulator().result()
    assert r["windows"] == 0 and math.isnan(r["ts_acc"]) and math.isnan(r["key_acc"])
    assert math.isnan(r["onset_f1"]) and math.isnan(r["duration_acc"])


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
