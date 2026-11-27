"""The duration chooser (src/dur_chooser.py), its use in placed_notes, the _dc system, the release's
new defaults, and the twelfth round's batch (pace/v2_round12.sh).

    python -m tests.test_dur_chooser
"""
from __future__ import annotations

import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ITOS = ["bar", "sub_1/2", "d_1/2", "d_1", "d_2", "d_4"]   # lengths: 1/2, 1, 2, 4 quarters


def _space():
    from src.notelabels import LabelSpace
    return LabelSpace(ITOS)


def _window():
    from src.notelabels import Window
    return Window(bars=[("4/4", 0), ("4/4", 0)])


def _placed(specs):
    """(bar, pos, staff, total) -> Placed, src = order."""
    from src.notelabels import Placed
    return [Placed(b, Fraction(p), 60 + k, Fraction(d), s, src=k) for k, (b, p, s, d) in enumerate(specs)]


def test_features():
    from src.dur_chooser import FEATURES, features
    space = _space()
    pl = _placed([(0, 0, 0, 2), (0, 1, 0, 1), (0, 1, 1, 1), (0, 3, 0, 1)])
    logits = np.full((4, space.n_dur), -5.0)
    logits[0, space.dur_index[Fraction(2)]] = 3.0     # the labeller's likeliest for note 0: a half note
    logits[0, space.dur_index[Fraction(1)]] = 2.0     # then the quarter, which reaches the staff's next onset
    onset = np.array([0.0, 0.5, 0.5, 1.5])
    dur = np.array([0.45, 0.4, 0.4, 0.4])
    use, cls, X = features(pl, _window(), logits, onset, dur, space)
    f = {name: k for k, name in enumerate(FEATURES)}
    assert list(use) == [0, 1, 2, 3] and X.shape == (4, 4, len(FEATURES))
    assert space.dur_value(cls[0, 0]) == 2 and space.dur_value(cls[0, 1]) == 1
    # note 0: the next onset on its staff is 1 quarter later: the quarter fits it, the half runs past
    assert X[0, 1, f["eq_s"]] == 1 and X[0, 0, f["gt_s"]] == 1 and X[0, 0, f["eq_s"]] == 0
    assert abs(X[0, 1, f["ratio_s"]]) < 1e-6 and abs(X[0, 0, f["ratio_s"]] - 1) < 1e-6
    assert X[0, 0, f["rank0"]] == 1 and X[0, 1, f["rank1"]] == 1
    assert X[0, 1, f["lp_gap"]] < 0 and X[0, 0, f["lp_gap"]] == 0
    # performed: held 0.45 s of a 0.5 s gap
    assert X[0, 0, f["has_perf"]] == 1 and abs(X[0, 0, f["perf_ratio"]] - np.log2(0.9)) < 1e-5
    # notes 1 and 2 share a position but not a staff: no chord; note 3 has no next onset on its staff
    assert X[1, 0, f["has_chord"]] == 0 and X[3, 0, f["has_next_s"]] == 0
    # a whole note at beat 3 of a 4/4 bar runs past the bar line
    k4 = list(cls[3]).index(space.dur_index[Fraction(4)])
    assert X[3, k4, f["past_bar"]] == 1 and X[3, k4, f["end_quarter"]] == 1


def test_chooser_learns_and_applies():
    import torch

    from src.dur_chooser import FEATURES, DurChooser, train_chooser
    rng = np.random.default_rng(0)
    n, k = 4000, 4
    X = rng.normal(size=(n, k, len(FEATURES))).astype(np.float32)
    eq = FEATURES.index("eq_s")
    X[:, :, eq] = 0
    y = rng.integers(0, k, n)
    X[np.arange(n), y, eq] = 1          # the right candidate reaches the next onset
    model, fit = train_chooser(X[:3000], y[:3000], X[3000:], y[3000:], epochs=10, batch=128, log=lambda *_: None)
    assert fit["val_acc"] > 0.95, fit
    space = _space()
    pl = _placed([(0, 0, 0, 2), (0, 1, 0, 1)])
    logits = np.full((2, space.n_dur), -5.0)
    logits[0, space.dur_index[Fraction(2)]] = 3.0
    logits[0, space.dur_index[Fraction(1)]] = 2.0
    logits[1, space.dur_index[Fraction(1)]] = 3.0
    onset, dur = np.array([0.0, 0.5]), np.array([0.45, 0.4])
    changed = model.apply(pl, _window(), logits, onset, dur, space)
    assert pl[0].total == 1 and changed >= 1
    # a note another step set the length of is left alone
    pl = _placed([(0, 0, 0, 2), (0, 1, 0, 1)])
    model.apply(pl, _window(), logits, onset, dur, space, keep=np.array([True, False]))
    assert pl[0].total == 2
    with tempfile.TemporaryDirectory() as t:
        model.save(Path(t) / "dur_chooser.pt", meta={"x": 1})
        back = DurChooser.load(t)
        assert back is not None and back.meta == {"x": 1}
        Xt = torch.from_numpy(X[3000:3100])
        assert torch.allclose(back(Xt), model.cpu()(Xt))
        assert DurChooser.load(Path(t) / "nothing") is None


def test_placed_notes_with_chooser():
    import torch

    from src.dur_chooser import DurChooser
    from src.labeller import performed, placed_notes
    from src.notelabels import N_FLAGS, N_POS
    space = _space()

    class Prefer(DurChooser):   # picks the candidate whose length reaches the next onset on the staff
        def choose(self, X):
            from src.dur_chooser import FEATURES
            return np.argmax(X[:, :, FEATURES.index("eq_s")] + 1e-3 * X[:, :, FEATURES.index("lp")], 1)
    n = 2
    out = {"keep": torch.full((1, n), 5.0), "delta": torch.zeros(1, n, 3), "pos": torch.zeros(1, n, N_POS),
           "sub": torch.zeros(1, n, space.n_sub), "dur": torch.full((1, n, space.n_dur), -5.0),
           "staff": torch.zeros(1, n, 4), "flags": torch.full((1, n, N_FLAGS), -5.0)}
    out["delta"][..., 1] = 5
    out["pos"][0, 0, 0] = out["pos"][0, 1, 1] = 5        # positions 0 and 1
    out["staff"][..., 0] = 5
    out["dur"][0, 0, space.dur_index[Fraction(2)]] = 3.0
    out["dur"][0, 0, space.dur_index[Fraction(1)]] = 2.0
    out["dur"][0, 1, space.dur_index[Fraction(1)]] = 3.0
    pitches, bar_est, own = np.array([60.0, 62.0]), np.array([1, 1]), np.array([True, True])
    base = placed_notes(out, 0, pitches, bar_est, own, space)
    assert [p.total for p in base] == [2, 1]
    mine = placed_notes(out, 0, pitches, bar_est, own, space, dur_chooser=Prefer(), frame=_window(),
                        perf=(np.array([0.0, 0.5]), np.array([0.45, 0.4])))
    assert [p.total for p in mine] == [1, 1] and [p.pos for p in mine] == [0, 1]

    class Cfg:
        feat_mean = [1.0, 0.0, -1.0, 0.5]
        feat_std = [2.0, 1.0, 0.5, 0.25]
    f = torch.zeros(2, 8)
    f[:, 1] = torch.tensor([0.0, 1.0])        # onset (2 * x + 1)
    f[:, 3] = torch.tensor([2.0, 0.0])        # log duration (0.5 * x - 1)
    on, du = performed(Cfg, f)
    assert np.allclose(on, [1.0, 3.0]) and np.allclose(du, np.exp([0.0, -1.0]))


def test_system_name_and_defaults():
    from scripts.system_scores import system_name
    from src import release
    assert system_name(None, "meter", "synth", "nl_gq+meter_v4_gq", delete_above=0.95, tidy=True,
                       dur_chooser=True) == "v2_nl_gq_meter_v4_gq_d95_tidy_dc"
    assert system_name(None, "meter", "synth", "nl_gq+meter_v4_gq_om6", delete_above=0.95, tidy=True, swing=True,
                       dur_chooser=True) == "v2_nl_gq_meter_v4_gq_om6_d95_tidy_sw_dc"
    assert release.DELETE_ABOVE == 0.95 and release.DELETE_ABOVE_MIDI == 0.8
    assert release.SWING is True and release.ONE_METER == 0.6 and release.DUR_CHOOSER is False


def test_pipeline_with_chooser():
    """V2Pipeline(dur_chooser=True) loads the chooser saved next to the labeller and runs end to end;
    without one it runs as before and says so; one_meter fills in a decoder that has none."""
    from tests.test_v2_system import V, make_v2
    if not V.exists():
        print("  (skipped: no data/processed/vocab.json)")
        return
    from src.dur_chooser import DurChooser
    from src.pipeline import V2Pipeline
    from src.preprocessing.performance import load_performance
    from tests.test_pipeline import make_performance
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        lab, met = make_v2(tmp)
        midi = tmp / "perf.mid"
        make_performance(midi, n=120)
        notes, _ = load_performance(midi)
        times = np.arange(0.3, float(notes[:, 1].max()) + 1.0, 0.5)
        beats = (times, np.arange(len(times)) % 4 == 0)
        plain = V2Pipeline(lab, met, vocab=V, device="cpu", window_notes=48, dur_chooser=True)
        assert plain.dur_chooser is None and plain.dur_chooser_missing
        res = plain.run(midi, tmp / "a", beats=beats)
        assert res.stats["dur_chooser"] is False and res.stats["dur_chooser_missing"] is True
        DurChooser().save(Path(lab).parent / "dur_chooser.pt")
        pipe = V2Pipeline(lab, met, vocab=V, device="cpu", window_notes=48, dur_chooser=True, one_meter=0.6,
                          swing=True)
        assert pipe.dur_chooser is not None and pipe.decode_cfg.one_meter == 0.6
        res = pipe.run(midi, tmp / "b", beats=beats)
        assert res.tokens and res.stats["dur_chooser"] is True
        assert res.stats["notes_kept"] + res.stats["notes_deleted"] == len(notes)


def test_round12_batch():
    import shutil
    import subprocess

    from tests.batch_env import batch_env
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_round12.sh", tmp / "pace")
        # a stand-in for round 11's batch: one job, chained the way the real one is (AFTER / JOBLOG)
        (tmp / "pace" / "v2_ensemble.sh").write_text(
            '#!/bin/bash\nID=$(sbatch --parsable -J e1x ${AFTER:+--dependency=afterany:$AFTER} --wrap=x)\n'
            'echo "$ID" >> "$JOBLOG"\n')
        for f in ("data/processed/train_gangqinpu/index.npy", "data/processed/val_gangqinpu/index.npy",
                  "data/processed/train/beats_gt.npz", "bd/MIDI2ScoreTF.ckpt"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "sbatch").chmod(0o755)
        r = subprocess.run(["bash", "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_round12.sh'], cwd=tmp,
                           env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd"),
                                         STAGES="ensemble chooser seq2seq"),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        rows = [ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()]
        by = {cmd.split("-J ")[1].split()[0]: (i, cmd) for i, cmd in rows}
        assert list(by) == ["e1x", "c1tests", "c1train", "c1asap", "c1gq", "c1final", "s1data", "s1train_a",
                            "s1train_b", "s1train_c", "s1asap", "s1gq", "s1final"]
        e = by["e1x"][0]
        assert "--dependency" not in by["e1x"][1]
        assert f"--dependency=afterany:{e} " in by["c1tests"][1]           # the stage before, all of it
        assert f"afterok:{by['c1tests'][0]}" in by["c1train"][1] and "train_dur_chooser.py --run nl_gq" in by["c1train"][1]
        assert "onemeter meter_v4_gq 0.6 meter_v4_gq_om6" in by["c1train"][1]
        assert "nl_gq+meter_v4_gq_om6:tidy,d95,sw,dc,missing" in by["c1asap"][1]
        assert "--conditions gangqinpu,gangqinpu_real" in by["c1gq"][1]
        assert "round12_report.py" in by["c1final"][1]
        chooser = ":".join(by[k][0] for k in ("e1x", "c1tests", "c1train", "c1asap", "c1gq", "c1final"))
        assert f"--dependency=afterany:{chooser} " in by["s1data"][1]
        assert f"afterok:{by['s1data'][0]}" in by["s1train_a"][1]
        assert f"afterany:{by['s1train_a'][0]}" in by["s1train_b"][1]
        assert "seq2seq sf_c128_gq30" in by["s1train_c"][1] and "--gangqinpu-share 0.3" in by["s1train_c"][1]
        assert f"afterany:{by['s1train_c'][0]}" in by["s1asap"][1] and "sf_c128_gq30:meter:sb" in by["s1gq"][1]
        # the real batches chain the same way
        for f in ("v2_ensemble.sh", "v2_pm2s2.sh"):
            s = (ROOT / "pace" / f).read_text()
            assert '--dependency=afterany:$AFTER' in s and '>> "$JOBLOG"' in s, f


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
