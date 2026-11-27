"""Tests for the 2/4-against-4/4 work (src/meter.py): one meter per piece after decoding
(consistent_meter, DecodeConfig.one_meter), the repetition features (rep_features, MeterConfig.n_rep),
bar-length-tolerant training (targets / meter_loss tolerant, TS_EQUIV) and the metrics that see it
(grid_scores ts_acc_equiv, ts_changes_per_100_bars).

    python -m tests.test_bar_length        # or: pytest tests/test_bar_length.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.meter import (IGNORE, N_ALL_FEATURES, N_FEATURE_COLUMNS, N_REP_FEATURES, REP_LAGS, TS_CLASSES,  # noqa: E402
                       DecodeConfig, beat_features, build_meter_model, consistent_meter, grid_from_logp, grid_scores,
                       meter_loss, rep_features, targets, ts_class)


def grid_of(bars, bar_q, n_beats=None):
    """A grid with a beat every quarter from 0 to the end of the last bar, one second apart."""
    from src.meter import bar_quarters
    end = bar_q[-1] + float(bar_quarters(bars[-1]))
    q = np.arange(0, end + 1e-9, 1.0) if n_beats is None else np.arange(n_beats, dtype=float)
    return {"t": q.copy(), "q": q, "bars": list(bars), "bar_q": np.asarray(bar_q, float), "bar_t": np.asarray(bar_q, float)}


def bar_lp(q, downs):
    """The bar head certain of bar lines at ``downs`` (quarters)."""
    lp = np.full((len(q), 2), np.log(0.5))
    for j, x in enumerate(q):
        yes = 0.95 if any(abs(x - d) < 1e-9 for d in downs) else 0.05
        lp[j] = np.log([1 - yes, yes])
    return {"bar": lp}


def test_consistent_meter():
    # 2/2 and 4/4: the same bar, all written as the more common one; nothing else changes
    g = grid_of(["4/4", "4/4", "2/2", "4/4"], [0, 4, 8, 12])
    out = consistent_meter(g, None, 0.0)
    assert out["bars"] == ["4/4"] * 4 and np.allclose(out["bar_q"], g["bar_q"]) and np.allclose(out["q"], g["q"])
    # mostly 4/4 with a run of four 2/4 bars: paired into two 4/4 bars on the lines the bar head likes
    bars = ["4/4"] * 6 + ["2/4"] * 4 + ["4/4"] * 2
    bq = [0, 4, 8, 12, 16, 20, 24, 26, 28, 30, 32, 36]
    g = grid_of(bars, bq)
    out = consistent_meter(g, bar_lp(g["q"], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]), 0.6)
    assert out["bars"] == ["4/4"] * 10 and list(out["bar_q"]) == [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
    assert np.allclose(out["q"], g["q"]) and out["meter_fixed"] > 0
    # the same run when the model puts the bar lines a 2/4 bar later: the first 2/4 bar stays alone
    out = consistent_meter(g, bar_lp(g["q"], [0, 4, 8, 12, 16, 20, 26, 30, 32, 36]), 0.6)
    assert out["bars"][6:9] == ["2/4", "4/4", "2/4"] and list(out["bar_q"][6:9]) == [24, 26, 30]
    # mostly 2/4 with two 4/4 bars: those split; below the share, nothing changes
    bars = ["2/4"] * 10 + ["4/4"] * 2
    bq = list(range(0, 20, 2)) + [20, 24]
    g = grid_of(bars, bq)
    out = consistent_meter(g, None, 0.6)
    assert out["bars"] == ["2/4"] * 14 and list(out["bar_q"]) == list(range(0, 28, 2))
    even = grid_of(["2/4"] * 4 + ["4/4"] * 2, [0, 2, 4, 6, 8, 12])
    assert consistent_meter(even, None, 0.9)["bars"] == even["bars"]
    # other meters are left alone; off by default in the decoder
    g = grid_of(["3/4", "3/4", "6/8"], [0, 3, 6])
    assert consistent_meter(g, None, 0.6)["bars"] == ["3/4", "3/4", "6/8"]
    assert DecodeConfig().one_meter == 0.0 and DecodeConfig.from_dict({"one_meter": 0.6}).one_meter == 0.6


def test_decoder_applies_it():
    """grid_from_logp with one_meter: a decoding that mixes 4/4 with a 2/4 pair comes out all 4/4."""
    k = 32
    t = np.arange(k) * 0.5
    nts, nst = len(TS_CLASSES), 11
    lp = {"bar": np.log(np.full((k, 2), 0.5)), "step": np.full((k, nst), np.log(1e-4)), "ts": np.full((k, nts), np.log(1e-4))}
    lp["step"][:, 5] = 0.0                                                   # a quarter per beat
    for j in range(k):
        down = j % 4 == 0 if not 12 <= j < 16 else j % 2 == 0
        lp["bar"][j] = np.log([0.1, 0.9]) if down else np.log([0.9, 0.1])
        lp["ts"][j, TS_CLASSES.index("2/4" if 12 <= j < 16 else "4/4")] = 0.0
    plain = grid_from_logp(t, lp, DecodeConfig(ts_change=np.log(0.5)))
    fixed = grid_from_logp(t, lp, DecodeConfig(ts_change=np.log(0.5), one_meter=0.6))
    assert "2/4" in plain["bars"] and set(fixed["bars"]) == {"4/4"} and np.allclose(plain["q"], fixed["q"])


def test_rep_features():
    """A two-beat figure repeated: the music repeats at 2 and 4 beats, not at 1 or 3."""
    t = np.arange(32) * 0.5
    rows = []
    for k, b in enumerate(t):
        pitch = (60, 64) if k % 2 == 0 else (55, 59, 62)                     # alternate two chords
        for p in pitch:
            rows.append((p, b, b + 0.4, 70))
        if k % 2 == 0:
            rows.append((72, b + 0.25, b + 0.4, 60))
    f = beat_features(np.array(rows, np.float32), t, np.zeros(32, bool))
    assert f.shape == (32, N_FEATURE_COLUMNS) and N_FEATURE_COLUMNS == N_ALL_FEATURES + N_REP_FEATURES
    r = f[:, N_ALL_FEATURES:]
    assert np.allclose(r, rep_features(f[:, :N_ALL_FEATURES]))
    lag = {L: r[10:, i].mean() for i, L in enumerate(REP_LAGS)}
    assert lag[2] > 0.99 and lag[4] > 0.99 and lag[1] < 0.5 and lag[3] < 0.5, lag
    assert (r[:2, 1] == 0).all()                                             # no beat 2 back yet
    assert np.isfinite(r).all() and (np.abs(r) <= 1 + 1e-6).all()


def test_model_reads_them():
    import torch
    from src.meter import collate
    torch.manual_seed(0)
    old = build_meter_model(dict(d_model=32, n_heads=4, layers=1, ff=64)).eval()
    new = build_meter_model(dict(d_model=32, n_heads=4, layers=1, ff=64, n_rep=N_REP_FEATURES)).eval()
    assert old.inp[0].in_features + N_REP_FEATURES == new.inp[0].in_features
    t = np.arange(8) * 0.5
    notes = np.array([(60, x, x + 0.3, 70) for x in t], np.float32)
    f = beat_features(notes, t, np.zeros(8, bool))
    x, mask, nin, _ = collate([(f, None, t, (np.zeros((0, 20), np.float32), np.zeros(0, int), np.zeros(0, int)))])
    with torch.no_grad():
        assert new(x, mask, nin)["bar"].shape == (1, 8, 2) and old(x, mask, nin)["ts"].shape == (1, 8, len(TS_CLASSES))
        x2 = x.clone()
        x2[..., N_ALL_FEATURES:] = 0
        assert torch.allclose(old(x, mask, nin)["bar"], old(x2, mask, nin)["bar"])   # an old model ignores them
        assert not torch.allclose(new(x, mask, nin)["bar"], new(x2, mask, nin)["bar"])


def test_tolerant():
    import torch
    t = np.arange(16) * 0.5
    gt = {"t": t, "q": np.arange(16.0), "down": np.arange(16) % 4 == 0, "ts": np.full(16, TS_CLASSES.index("4/4")),
          "key": np.zeros(16, int), "ts_names": ["4/4"] * 16}
    strict, tol = targets(t, gt), targets(t, gt, tolerant=True)
    mid = np.arange(16) % 4 == 2
    assert (tol["bar"][mid] == IGNORE).all() and (strict["bar"][mid] == 0).all()
    assert (tol["bar"][~mid] == strict["bar"][~mid]).all() and (tol["ts"] == strict["ts"]).all()
    gt3 = dict(gt, ts=np.full(16, TS_CLASSES.index("3/4")), ts_names=["3/4"] * 16, down=np.arange(16) % 3 == 0)
    assert (targets(t, gt3, tolerant=True)["bar"] == targets(t, gt3)["bar"]).all()   # a triple bar has no middle
    assert ts_class("2/2") == ts_class("2/4") == "2/4" and ts_class("12/8") == "6/8" and ts_class("3/4") == "3/4"
    # the loss: a model sure of 2/4 against a 4/4 target costs nothing tolerant, much strict
    logits = torch.full((1, 4, len(TS_CLASSES)), -10.0)
    logits[..., TS_CLASSES.index("2/4")] = 10.0
    out = {"ts": logits}
    tg = {"ts": torch.full((1, 4), TS_CLASSES.index("4/4"), dtype=torch.long)}
    assert meter_loss(out, tg, tolerant=True)[0] < 1e-3 < meter_loss(out, tg)[0]
    tg3 = {"ts": torch.full((1, 4), TS_CLASSES.index("3/4"), dtype=torch.long)}
    assert meter_loss(out, tg3, tolerant=True)[0] > 10                       # 3/4 is no reading of 2/4


def test_metrics():
    t = np.arange(16) * 0.5
    gt = {"t": t, "q": np.arange(16.0), "down": np.arange(16) % 4 == 0, "ts": np.full(16, TS_CLASSES.index("4/4")),
          "key": np.zeros(16, int), "ts_names": ["4/4"] * 16}
    two = {"t": t, "q": np.arange(16.0), "bars": ["2/4"] * 8, "bar_q": np.arange(0, 16, 2.0), "bar_t": np.arange(0, 8, 1.0)}
    s = grid_scores(two, gt)
    assert s["ts_acc"] == 0 and s["ts_acc_equiv"] == 1 and s["ts_changes_per_100_bars"] == 0
    mixed = dict(two, bars=["2/4", "4/4"] * 4)
    assert grid_scores(mixed, gt)["ts_changes_per_100_bars"] == 100 * 7 / 8


def test_pilot2_graph():
    """pace/v2_pilot2.sh with stand-ins for sbatch / squeue (the first pilot's jobs still queued):
    6 jobs, each waiting for its first-pilot counterpart by name, nothing published, the report last."""
    import shutil
    import subprocess
    import tempfile

    from tests.batch_env import batch_env
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_pilot2.sh", tmp / "pace")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        # squeue -n <name>: the first pilot's job ids by name
        (tmp / "bin" / "squeue").write_text('#!/bin/bash\nfor a in "$@"; do case "$prev" in -n) case "$a" in '
                                            'q1prep) echo 900;; q1short) echo 901;; q1asap) echo 902;; q1gq) echo 903;; '
                                            'q1final) echo 904;; esac;; esac; prev=$a; done\n')
        for f in (tmp / "bin").iterdir():
            f.chmod(0o755)
        (tmp / "bd").mkdir()
        (tmp / "bd" / "MIDI2ScoreTF.ckpt").write_text("x")
        r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_pilot2.sh'], cwd=tmp,
                           env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd")),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        by = {ln.split("\t")[1].split("-J ")[1].split()[0]: ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()}
        assert list(by) == ["r2tests", "r2cons", "r2train", "r2asap", "r2gq", "r2final"], list(by)
        tests, cons, train, asap, gq, final = (by[k] for k in by)
        assert "tests.test_bar_length" in tests[1]
        assert f"afterok:{tests[0]},afterany:900" in cons[1] and "onemeter meter_v4_gq 0.5 meter_v4_gq_om5" in cons[1]
        assert f"afterok:{tests[0]},afterany:901" in train[1] and "meter_v4_gq_rep_p12k meter_v4_gq_tol_p12k" in train[1]
        assert f"afterok:{cons[0]},afterany:902" in asap[1] and "nl_gq+meter_v4_gq_om5:tidy,d95,missing" in asap[1]
        assert f"afterok:{cons[0]},afterany:903" in gq[1] and "--manifest outputs/system_eval_gangqinpu/manifest.json" in gq[1]
        assert f"afterany:{asap[0]}:{gq[0]}:{train[0]}:{cons[0]}:904" in final[1] and "pilot_report.py" in final[1]
        for row in (cons, train, asap, gq):
            assert "PUBLISH=0" in row[1]


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
