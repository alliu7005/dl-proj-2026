"""Other grid sources for v2 (src/grid_sources.py), the refinement loop, their systems, meter
pretraining (scripts/pretrain_meter.py) and the thirteenth round's batch (pace/v2_round13.sh).

    python -m tests.test_grid_sources
"""
from __future__ import annotations

import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F = Fraction
sys.path.insert(0, str(ROOT))


def _gt(t0=1.0, n=17, beat=0.5):
    from src.meter import TS_CLASSES
    t = t0 + beat * np.arange(n)
    return {"t": t, "q": np.arange(n, dtype=float), "down": np.arange(n) % 4 == 0,
            "ts": np.full(n, TS_CLASSES.index("4/4")), "key": np.zeros(n, int), "ts_names": ["4/4"] * n}


def test_spliced_logp_and_annotated_beats():
    from src.grid_sources import annotated_beats, spliced_logp
    from src.meter import Q_STEPS, TS_CLASSES
    bt = 0.5 * np.arange(30)                    # tracked beats from 0 s; the ground truth covers 1-9 s
    k = len(bt)
    lp = {"bar": np.log(np.full((k, 2), 0.5)), "step": np.log(np.full((k, len(Q_STEPS)), 1 / len(Q_STEPS))),
          "ts": np.log(np.full((k, len(TS_CLASSES)), 1 / len(TS_CLASSES)))}
    out = spliced_logp(bt, lp, [_gt()])
    inside = (bt >= 1.0) & (bt <= 9.0)
    assert np.allclose(out["bar"][~inside], lp["bar"][~inside])          # the model's where nothing is known
    assert (out["bar"][inside].max(1) == 0).all()                         # certain where it is
    downs = bt[inside][out["bar"][inside].argmax(1) == 1]
    assert np.allclose(downs, [1.0, 3.0, 5.0, 7.0, 9.0]), downs
    assert (out["ts"][inside].argmax(1) == TS_CLASSES.index("4/4")).all()
    assert np.allclose(lp["bar"], np.log(0.5))                            # the input is left as it was
    tt, dd = annotated_beats(bt + 0.03, np.zeros(k, bool), [_gt()])
    assert np.all(np.diff(tt) > 0)
    assert np.isclose(tt, 1.0).any() and not np.isclose(tt, 1.03).any()   # replaced inside the span
    assert np.isclose(tt, 0.53).any() and np.isclose(tt, 12.03).any()     # kept outside
    assert dd[np.isclose(tt, 3.0)].all() and not dd[np.isclose(tt, 3.5)].any()


def test_score_grid_many_notes():
    """More notes than quarters (eighths in both hands): the map is evaluated at the grid's quarters
    (an earlier version indexed the matches' positions with the grid's and went out of bounds)."""
    from src.grid_sources import score_grid
    toks, pitches, onsets = [], [], []
    for b in range(4):
        toks += ["bar"] + (["ts_4/4", "key_0"] if b == 0 else [])
        for e in range(8):
            q = F(e, 2)
            toks += [f"pos_{q.numerator // q.denominator}"] + (["sub_1/2"] if q.denominator == 2 else [])
            for p in (48 + e % 5, 72 + e % 7):
                toks += [f"p_{p}", "d_1/2"]
                pitches.append(p)
                onsets.append(1.0 + 0.5 * (b * 4 + float(q)))
    g = score_grid(toks, np.array(onsets), np.array(pitches))
    assert g is not None and g["bars"] == ["4/4"] * 4 and len(g["q"]) == 17
    assert np.allclose(np.interp(g["bar_q"], g["q"], g["t"]), 1.0 + 2.0 * np.arange(4), atol=0.02)


def test_score_grid():
    from src.grid_sources import score_grid
    bars = []
    pitches, onsets = [], []
    for b in range(6):                                # 6 bars of 3/4, a quarter at 0.6 s, notes on every beat
        bars += ["bar"] + (["ts_3/4", "key_0"] if b == 0 else [])
        for q in range(3):
            p = 60 + (b * 3 + q) % 12
            bars += [f"pos_{q}", f"p_{p}", "d_1"]
            pitches.append(p)
            onsets.append(2.0 + 0.6 * (b * 3 + q))
    g = score_grid(bars, np.array(onsets), np.array(pitches))
    assert g is not None and g["bars"] == ["3/4"] * 6 and np.allclose(g["bar_q"], [0, 3, 6, 9, 12, 15])
    assert np.allclose(np.interp(g["bar_q"], g["q"], g["t"]), 2.0 + 1.8 * np.arange(6), atol=0.02)
    assert np.all(np.diff(g["t"]) > 0) and g["matched"] >= 15
    assert score_grid(["bar", "ts_4/4", "key_0", "pos_0", "p_60", "d_1"], np.array([1.0]), np.array([60])) is None


def test_votes():
    from src.grid_sources import labeller_bar_votes, with_votes
    bt = 0.5 * np.arange(8)
    onsets = np.array([0.0, 0.02, 0.5, 1.0, 2.0, 2.24])
    placed = [(0, 0, Fraction(0)), (1, 0, Fraction(0)), (2, 0, Fraction(1)), (3, 0, Fraction(2)),
              (4, 1, Fraction(0)), (5, 1, Fraction(1, 2))]
    down, on = labeller_bar_votes(placed, {}, onsets, bt)
    assert list(on[:5]) == [2, 1, 1, 0, 1] and list(down[:5]) == [2, 0, 0, 0, 1]     # 2.24 s is off every beat
    lp = {"bar": np.log(np.full((8, 2), 0.5))}
    out = with_votes(lp, (down, on), 1.0)
    assert out["bar"][0, 1] > out["bar"][0, 0] and out["bar"][1, 0] > out["bar"][1, 1]
    assert np.allclose(out["bar"][3], lp["bar"][3]) and np.allclose(lp["bar"], np.log(0.5))


def test_system_names():
    from scripts.system_scores import system_name
    base = dict(delete_above=0.95, tidy=True)
    assert system_name(None, "meter", "synth", "nl_gq+meter_v4_gq", grid="oracle", **base) == "v2_nl_gq_meter_v4_gq_d95_tidy_og"
    assert system_name(None, "meter", "synth", "nl_gq+meter_v4_gq", grid="oracle_beats", **base).endswith("_tidy_ogt")
    assert system_name(None, "meter", "synth", "nl_gq+meter_v4_gq", grid="seq2seq", **base).endswith("_tidy_s2g")
    assert system_name(None, "meter", "synth", "nl_gq+meter_v4_gq", refine=1.0, **base).endswith("_tidy_rf")


def test_pipeline_grid_sources():
    """V2Pipeline with a given grid (as score_grid makes it), with a log-probability change, and with
    the refinement loop: each runs end to end."""
    from tests.test_v2_system import V, make_v2
    if not V.exists():
        print("  (skipped: no data/processed/vocab.json)")
        return
    from src.pipeline import V2Pipeline
    from src.preprocessing.performance import load_performance
    from tests.test_pipeline import make_performance
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        lab, met = make_v2(tmp)
        midi = tmp / "perf.mid"
        make_performance(midi, n=120)
        notes, _ = load_performance(midi)
        end = float(notes[:, 1].max()) + 1.0
        times = np.arange(0.3, end, 0.5)
        beats = (times, np.arange(len(times)) % 4 == 0)
        pipe = V2Pipeline(lab, met, vocab=V, device="cpu", window_notes=48)
        n_q = int(np.ceil((end - 0.3) / 0.5)) + 4
        q = np.arange(n_q, dtype=float)
        given = {"t": 0.3 + 0.5 * q, "q": q, "bars": ["3/4"] * (n_q // 3 + 1), "bar_q": 3.0 * np.arange(n_q // 3 + 1)}
        res = pipe.run(midi, tmp / "a", beats=beats, grid_given=given)
        assert res.tokens and set(res.stats["time_signatures"]) == {"3/4"}, res.stats["time_signatures"]
        res = pipe.run(midi, tmp / "b", beats=beats, grid_given=lambda n, b: None)
        assert not res.tokens and "no grid" in res.errors[0]
        seen = []

        def fn(bt, lp):
            seen.append(len(bt))
            return lp
        res = pipe.run(midi, tmp / "c", beats=beats, logp_fn=fn)
        assert res.tokens and seen == [len(times)]
        rp = V2Pipeline(lab, met, vocab=V, device="cpu", window_notes=48, refine=1.0)
        res = rp.run(midi, tmp / "d", beats=beats)
        assert res.tokens and res.stats["refine"] == 1.0 and res.stats["refined_bar_lines"] is not None


def test_pretrain_meter():
    """The masked-beat pretraining runs on stored beat features, and its checkpoint is one
    train_meter.py --init loads with nothing missing."""
    import pickle

    import torch

    from scripts.pretrain_meter import main, meter_cfg, span_mask
    from src.meter import N_BEAT_FEATURES, build_meter_model
    rng = np.random.default_rng(0)
    m = span_mask(200, 0.15, rng)
    assert 25 <= m.sum() <= 40
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        perfs = [{"file": f"{k}.mid", "x": rng.normal(size=(int(rng.integers(40, 90)), N_BEAT_FEATURES)).astype(np.float16)}
                 for k in range(40)]
        with open(tmp / "pre.pkl", "wb") as fh:
            pickle.dump(perfs, fh)
        assert main(["train", "--data", str(tmp / "pre.pkl"), "--runs-dir", str(tmp / "runs"), "--name", "pt",
                     "--steps", "4", "--eval-every", "2", "--batch", "8", "--chunk", "64", "--d-model", "64",
                     "--layers", "2", "--device", "cpu"]) == 0
        ck = torch.load(tmp / "runs" / "pt" / "best.pt", weights_only=False)
        assert ck["config"] == meter_cfg(64, 2) and "recon" in ck
        model = build_meter_model(dict(ck["config"]))
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        assert not missing and not unexpected


def test_round13_batch():
    import shutil
    import subprocess

    from tests.batch_env import batch_env
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_round13.sh", tmp / "pace")
        for f in ("data/processed/meter/test.pkl", "data/processed/meter/test_gangqinpu.pkl",
                  "outputs/runs/sf_base_reg_cab_c128/best.pt", "outputs/system_eval_gangqinpu/manifest.json",
                  "outputs/system_eval/manifest.json", "bd/MIDI2ScoreTF.ckpt"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "squeue").write_text("#!/bin/bash\necho 900\necho 901\n")
        for f in (tmp / "bin").iterdir():
            f.chmod(0o755)
        r = subprocess.run(["bash", "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_round13.sh'], cwd=tmp,
                           env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd")),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        by = {ln.split("\t")[1].split("-J ")[1].split()[0]: ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()}
        assert list(by) == ["g1tests", "g1asap", "g1gq", "g1final", "p1fetch", "p1prep", "p1train", "p1ft", "p1final"]
        assert "--dependency" not in by["g1tests"][1]
        assert f"afterok:{by['g1tests'][0]}" in by["g1asap"][1]
        for o in ("og", "ogt", "s2g", "rf"):
            assert f"nl_gq+meter_v4_gq:tidy,d95,{o},missing" in by["g1gq"][1]
        assert "--dependency=afterany:900:901" in by["p1fetch"][1] and "fetch_aria_midi.py" in by["p1fetch"][1]
        assert f"afterok:{by['p1train'][0]}" in by["p1ft"][1] and "meters meter_v4_gq_pt_p12k" in by["p1ft"][1]
        assert f"afterany:{by['p1ft'][0]}:{by['g1final'][0]}" in by["p1final"][1]
        r = subprocess.run(["bash", "-c", 'export PATH="$PWD/bin:$PATH"; rm -f jobs.txt; bash pace/v2_round13.sh'],
                           cwd=tmp, env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd"),
                                                  PRETRAIN_AFTER="now"), capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        fetch = [ln for ln in (tmp / "jobs.txt").read_text().splitlines() if "-J p1fetch" in ln][0]
        assert "--dependency" not in fetch


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
