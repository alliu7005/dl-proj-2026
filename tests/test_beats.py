"""Tests for beats as an input (src/beats.py and everything that uses it).

    python -m tests.test_beats        # or: pytest tests/test_beats.py

Covers the meter convention, the ground-truth grid against ASAP's human annotations (needs
data/processed/test and data/raw/asap_bd; skipped without them), merging windows, the features
(including their invariance under tempo augmentation), the tracker-error model (simulate ->
classify -> fit recovers the rates), storage, the Beat This! wrapper and its held-out fold choice
(with a stand-in beat_this package, so no download), and the integration: dataset, training,
evaluation conditions and the demo pipeline.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.beats import (BEAT_FEATURES, N_BEAT_FEATURES, BeatNoise, beat_features, beat_layout,  # noqa: E402
                       beats_path, compare, f_measure, fit_noise, load_beats, merge_performance, save_beats)

D = ROOT / "data" / "processed"
ANN = ROOT / "data" / "raw" / "asap_bd" / "asap_annotations.json"


def grid(n_beats=40, ibi=0.5, bar=4, t0=1.0):
    t = t0 + ibi * np.arange(n_beats)
    return t, np.arange(n_beats) % bar == 0


def test_beat_layout():
    """ASAP's convention (all 1,067 annotated performances): dotted beats only for numerators that
    are multiples of 3 above 3."""
    cases = {"4/4": (4, 1.0), "3/4": (3, 1.0), "2/4": (2, 1.0), "2/2": (2, 2.0), "4/2": (4, 2.0), "3/2": (3, 2.0),
             "6/8": (2, 1.5), "9/8": (3, 1.5), "12/8": (4, 1.5), "3/8": (3, 0.5), "6/4": (2, 3.0),
             "6/16": (2, 0.75), "12/16": (4, 0.75), "24/16": (8, 0.75), "5/4": (5, 1.0), "junk": (4, 1.0)}
    for ts, want in cases.items():
        assert beat_layout(ts) == want, (ts, beat_layout(ts), want)


def test_ground_truth_vs_annotations():
    """The alignment-derived grid (used for PERiScoPe, which has no beat annotations) against the
    human ASAP annotations on the ASAP test split: what scripts/build_beats.py --check reports."""
    if not ((D / "test" / "index.npy").exists() and ANN.exists()):
        print("  (skipped: data/processed/test or the ASAP annotations missing)")
        return
    from collections import defaultdict
    from scripts.build_beats import annotated
    from src.beats import window_grid
    from src.preprocessing.store import SegmentStore
    from src.preprocessing.vocab import Vocab
    store, vocab = SegmentStore(D / "test"), Vocab.load(str(D / "vocab.json"))
    ann = json.loads(ANN.read_text())
    by = defaultdict(list)
    perfs = sorted({m["performance"] for m in store.meta})[::4]  # every 4th performance: fast
    for i, m in enumerate(store.meta):
        if m["performance"] in perfs:
            n, _, t = store.get(i)
            by[m["performance"]].append((m["bar_start"], m["perf_t0"], window_grid(n[:, 1], n[:, 0], vocab.decode(t.tolist()))))
    fs, errs = [], []
    for p, ws in by.items():
        t, d = merge_performance(ws)
        at, ad = annotated(ann[p])
        assert np.all(np.diff(t) > 0) and d.any(), p
        lo, hi = max(at[0], t[0]), min(at[-1], t[-1])
        a_in, t_in = (at >= lo) & (at <= hi), (t >= lo) & (t <= hi)
        fs.append(f_measure(at[a_in], t[t_in]))
        j = np.clip(np.searchsorted(at, t[t_in]), 1, len(at) - 1)
        errs.append(np.minimum(np.abs(at[j] - t[t_in]), np.abs(at[j - 1] - t[t_in])))
    e = np.concatenate(errs)
    assert np.median(fs) > 0.78, np.median(fs)                 # measured 0.85 on the full test split
    assert np.median(e) < 0.01 and (e <= 0.07).mean() > 0.78, (np.median(e), (e <= 0.07).mean())


def test_merge_performance():
    """Overlapping windows: each beat from the window that timed it with least extrapolation;
    times strictly increase; downbeat = beat 0 of a bar."""
    def g(bars, beats, t, ex):
        return {"bar": np.array(bars), "beat": np.array(beats), "t": np.array(t, float), "extrap": np.array(ex, float)}
    w0 = g([0, 0, 1, 1], [0, 1, 0, 1], [0.0, 0.5, 1.0, 1.5], [0, 0, 0, 0.8])
    w1 = g([0, 0, 1], [0, 1, 0], [0.0, 0.48, 1.02], [0.5, 0.5, 0])   # window 1 starts at bar 1 (bar_start 1)
    t, d = merge_performance([(0, 10.0, w0), (1, 11.0, w1), (5, 0.0, None)])
    # bar 1 beat 0: w0 (extrap 0, t 11.0) vs w1 (bar 1 = its bar 0: extrap 0.5) -> w0; bar 1 beat 1:
    # w0 extrap 0.8 vs w1 extrap 0.5 -> w1 (11.48); bar 2 beat 0 only in w1 (12.02)
    assert np.allclose(t, [10.0, 10.5, 11.0, 11.48, 12.02]) and d.tolist() == [True, False, True, False, True]
    t, d = merge_performance([(0, 0.0, g([0, 0, 1], [0, 1, 0], [0.0, 0.6, 0.5], [0, 0, 0]))])  # non-increasing
    assert np.allclose(t, [0.0, 0.6]) and d.tolist() == [True, False]


def test_features():
    t, d = grid()                                       # beats every 0.5 s from 1.0, bars of 4
    on = np.array([1.0, 1.25, 3.0, 3.5, 0.6, 30.0])     # beat, half beat, downbeat, beat 2, before, far after
    f = beat_features(on, t, d)
    assert f.shape == (6, N_BEAT_FEATURES) and BEAT_FEATURES[-1] == "has_beats"
    assert np.allclose(f[0, :2], [0, 1], atol=1e-6) and np.allclose(f[1, :2], [0, -1], atol=1e-6)
    assert np.allclose(f[2, 2:4], [0, 1], atol=1e-6)            # 3.0 s = beat 4 = a downbeat
    assert np.allclose(f[3, 2:4], [1, 0], atol=1e-6)            # one beat into the bar: a quarter turn
    assert np.allclose(f[:4, 4], 0) and np.allclose(f[:4, 5], 1.0) and np.all(f[:5, 6] == 1)
    assert f[4, 6] == 1 and np.isclose(f[4, 1], np.cos(2 * np.pi * 0.2))  # within one extrapolated beat
    assert np.all(f[5] == 0)                                    # outside the beats: nothing
    gap = np.concatenate([t[:10], t[10:] + 20])                 # a 20 s hole: no beats inside it
    assert beat_features(np.array([t[9] + 5]), gap, d)[0, 6] == 0
    nodown = beat_features(on[:4], t, np.zeros_like(d))         # beats but no downbeats
    assert np.all(nodown[:, 2:4] == 0) and np.all(nodown[:, 5] == 0) and np.all(nodown[:, 6] == 1)
    assert np.all(beat_features(on, np.zeros(0), np.zeros(0, bool)) == 0)


def test_noise_model():
    rng = lambda s: np.random.default_rng(s)  # noqa: E731
    t, d = grid(200)
    exact = BeatNoise(p_double=0, p_half=0, p_offbeat=0, miss=0, extra=0, jitter=0, p_no_downbeats=0,
                      p_downbeat_shift=0, p_meter=0)
    t2, d2 = exact(t, d, rng(0))
    assert np.allclose(t2, t) and np.array_equal(d2, d)
    only = lambda **k: BeatNoise(**{**exact.__dict__, **k})  # noqa: E731
    t2, d2 = only(p_double=1)(t, d, rng(1))
    assert len(t2) == 2 * len(t) - 1 and d2.sum() == d.sum() and compare(t, d, t2, d2)["octave"] == "double"
    t2, d2 = only(p_half=1)(t, d, rng(2))
    assert len(t2) == len(t) // 2 and compare(t, d, t2, d2)["octave"] == "half"
    assert abs(int(d2.sum()) - int(d.sum())) <= 1  # dropped downbeats move to the next kept beat
    t2, d2 = only(p_offbeat=1)(t, d, rng(3))
    assert compare(t, d, t2, d2)["octave"] == "offbeat"
    t2, d2 = only(p_downbeat_shift=1)(t, d, rng(4))
    c = compare(t, d, t2, d2)
    assert c["octave"] == "normal" and c["downbeats"] == "shift" and np.median(np.diff(np.nonzero(d2)[0])) == 4
    t2, d2 = only(p_meter=1)(t, d, rng(5))
    assert compare(t, d, t2, d2)["downbeats"] == "meter"
    t2, d2 = only(p_no_downbeats=1)(t, d, rng(6))
    assert not d2.any() and compare(t, d, t2, d2)["downbeats"] == "none"
    t2, d2 = only(miss=0.2, jitter=0.01)(t, d, rng(7))
    c = compare(t, d, t2, d2)
    assert c["octave"] == "normal" and 0.12 < c["miss"] < 0.28 and c["downbeats"] == "correct", c
    assert 0.005 < np.std(c["errors"]) < 0.015
    a, b = BeatNoise()(t, d, rng(8)), BeatNoise()(t, d, rng(8))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])  # reproducible from the seed


def test_fit_recovers_rates():
    """Simulate a tracker with known errors on 300 pieces, classify, fit: the rates come back."""
    true = BeatNoise(p_double=0.1, p_half=0.2, p_offbeat=0.05, miss=0.1, extra=0.05, jitter=0.02,
                     p_no_downbeats=0.1, p_downbeat_shift=0.2, p_meter=0.1)
    rng = np.random.default_rng(0)
    cs = []
    for k in range(300):
        t, d = grid(int(rng.integers(80, 200)), ibi=float(rng.uniform(0.35, 0.8)), bar=int(rng.choice([3, 4])))
        cs.append(compare(t, d, *true(t, d, rng)))
    fit = fit_noise(cs)
    assert fit.fitted
    for name, tol in (("p_double", 0.05), ("p_half", 0.06), ("p_offbeat", 0.04), ("miss", 0.03), ("extra", 0.03),
                      ("jitter", 0.006), ("p_no_downbeats", 0.06), ("p_downbeat_shift", 0.08), ("p_meter", 0.07)):
        assert abs(getattr(fit, name) - getattr(true, name)) < tol, (name, getattr(fit, name), getattr(true, name))
    with tempfile.TemporaryDirectory() as tmp:
        fit.to_json(Path(tmp) / "n.json", {"fit_on": {"performances": 300}})
        back = BeatNoise.from_json(Path(tmp) / "n.json")
        assert back == fit


def test_storage():
    with tempfile.TemporaryDirectory() as tmp:
        b = {"a/x.mid": grid(10), "b/y.mid": grid(3, t0=5.0), "c/empty.mid": (np.zeros(0), np.zeros(0, bool))}
        path = beats_path(tmp, "gt")
        assert path.name == "beats_gt.npz"
        save_beats(path, b)
        back = load_beats(path)
        assert set(back) == set(b)
        for p in b:
            assert np.allclose(back[p][0], b[p][0]) and np.array_equal(back[p][1], b[p][1])
        assert load_beats(Path(tmp) / "nope.npz") == {}
    try:
        beats_path(".", "estimated")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown beat source was accepted")


def fake_beat_this(beats, downbeats, swap=False):
    """A stand-in beat_this package: File2Beats returns fixed arrays (optionally in swapped order)."""
    calls = []

    class File2Beats:
        def __init__(self, checkpoint_path, device, dbn):
            calls.append(checkpoint_path)

        def __call__(self, path):
            return (downbeats, beats) if swap else (beats, downbeats)
    pkg, inf = types.ModuleType("beat_this"), types.ModuleType("beat_this.inference")
    inf.File2Beats = File2Beats
    pkg.inference = inf
    sys.modules["beat_this"], sys.modules["beat_this.inference"] = pkg, inf
    return calls


def test_tracker_and_folds():
    from scripts import track_beats as tb
    assert tb.asap_stem("Bach/Fugue/bwv_846/Shi05M.mid") == "Bach_Fugue_bwv_846_Shi05M"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        f = tmp / "8-folds.split"
        f.write_text("Bach_Fugue_bwv_846_Shi05M\t4\nChopin_Ballades_1_BuiJL04M\t0\nbad line\n", encoding="utf-8")
        folds = tb.load_folds(f)
        assert folds == {"Bach_Fugue_bwv_846_Shi05M": 4, "Chopin_Ballades_1_BuiJL04M": 0}
        assert tb.checkpoint_for("Bach/Fugue/bwv_846/Shi05M.mid", folds) == "fold4"   # never trained on it
        assert tb.checkpoint_for("Ravel/Ondine/X01.mid", folds) == "final0"           # never in its data
        # ASAP metadata -> MAESTRO audio
        (tmp / "asap").mkdir()
        with open(tmp / "asap" / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["midi_performance", "maestro_midi_performance"])
            w.writerow(["Bach/Fugue/bwv_846/Shi05M.mid", "{maestro}/2004/a.midi"])
            w.writerow(["No/Audio/X.mid", ""])
        (tmp / "maestro").mkdir()
        with open(tmp / "maestro" / "maestro-v3.0.0.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["midi_filename", "audio_filename", "split"])
            w.writerow(["2004/a.midi", "2004/a.wav", "test"])
        audio = tb.audio_files(tmp / "asap", tmp / "maestro")
        assert audio == {"Bach/Fugue/bwv_846/Shi05M.mid": tmp / "maestro" / "2004/a.wav"}
    beats, downs = np.array([0.5, 1.0, 1.5, 2.0, 2.5]), np.array([0.5, 2.5])
    for swap in (False, True):  # either output order gives the same beat list
        calls = fake_beat_this(beats, downs, swap)
        try:
            t, d = tb.Tracker("cpu")("x.wav", "fold3")
            tb.Tracker("cpu")("y.wav", "fold3")
        finally:
            sys.modules.pop("beat_this", None), sys.modules.pop("beat_this.inference", None)
        assert np.allclose(t, beats) and d.tolist() == [True, False, False, False, True] and calls[0] == "fold3"


def ensure_test_beats():
    if not (D / "test" / "index.npy").exists():
        return False
    if not beats_path(D / "test", "gt").exists():
        r = subprocess.run([sys.executable, "scripts/build_beats.py", "--splits", "test", "--workers", "1"],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
    return True


def test_dataset():
    if not ensure_test_beats():
        print("  (skipped: data/processed/test missing)")
        return
    import torch
    from src.dataset import AugmentConfig, ScoreSegmentDataset, n_features
    V = str(D / "vocab.json")
    plain, gt = ScoreSegmentDataset(str(D / "test"), V), ScoreSegmentDataset(str(D / "test"), V, beats="gt")
    assert n_features(False, True) == 15 and n_features(True, True) == 17
    for i in (0, 7, 40):
        a, b = plain[i], gt[i]
        assert torch.equal(a["notes"], b["notes"][:, :8]) and torch.equal(a["target"], b["target"])
        assert b["notes"].shape[1] == 15 and b["notes"][:, -1].mean() > 0.9
    none = ScoreSegmentDataset(str(D / "test"), V, beats="none")[0]
    assert none["notes"].shape[1] == 15 and (none["notes"][:, 8:] == 0).all()
    # tempo augmentation scales notes and beats together: the phases stay, the beat period follows
    aug = ScoreSegmentDataset(str(D / "test"), V, beats="gt", augment=AugmentConfig(), prompt_dropout=0.0)
    for i in (3, 11):
        x, y = gt[i]["notes"].numpy(), aug[i]["notes"].numpy()
        assert np.allclose(x[:, [8, 9, 10, 11, 13, 14]], y[:, [8, 9, 10, 11, 13, 14]], atol=1e-4)
    # noise and dropout only in their own random stream: the other features are untouched
    noisy = ScoreSegmentDataset(str(D / "test"), V, beats="gt", beat_noise=BeatNoise())
    for i in (0, 7):
        assert torch.equal(noisy[i]["notes"][:, :8], plain[i]["notes"])
        assert torch.equal(noisy[i]["notes"], noisy[i]["notes"])  # fixed per window (the sim condition)
    try:
        ScoreSegmentDataset(str(D / "test"), V, beats="tracked")
    except FileNotFoundError as e:
        assert "track_beats" in str(e)
    else:
        if not beats_path(D / "test", "tracked").exists():
            raise AssertionError("missing tracked beats were not reported")


def test_training_with_beats():
    if not ensure_test_beats() or not beats_path(D / "train", "gt").exists() or not beats_path(D / "val", "gt").exists():
        print("  (skipped: needs beats_gt.npz in train / val: python scripts/build_beats.py)")
        return
    import torch
    name = "_test_beats"
    run = ROOT / "outputs" / "runs" / name
    shutil.rmtree(run, ignore_errors=True)
    base = [sys.executable, "scripts/train_model.py", "--name", name, "--arch", "score2seq", "--lr", "1e-3",
            "--warmup", "2", "--eval-every", "3", "--eval-loss-windows", "8", "--eval-decode-windows", "1",
            "--workers", "0", "--batch-tokens", "2048", "--max-steps", "6", "--patience", "0"]
    try:
        r = subprocess.run(base + ["--beats", "--beat-noise", "none"], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        cfg = json.loads((run / "config.json").read_text())
        assert cfg["model"]["beat_features"] is True and cfg["model"]["n_features"] == 15
        assert cfg["data"]["beats"]["noise"] == "none"
        ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        ck["phase"], ck["step"] = "train", 3
        torch.save(ck, run / "last.pt")
        r = subprocess.run(base + ["--resume"], cwd=ROOT, capture_output=True, text=True)  # no --beats: kept
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        assert torch.load(run / "last.pt", map_location="cpu", weights_only=False)["config"]["beat_features"] is True
        r = subprocess.run(base + ["--name", "_test_beats_missing", "--beats", "--beat-noise", "no/such.json"],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode != 0 and "fit_beat_noise" in (r.stdout + r.stderr)
    finally:
        shutil.rmtree(run, ignore_errors=True)
        shutil.rmtree(ROOT / "outputs" / "runs" / "_test_beats_missing", ignore_errors=True)


def test_evaluation_conditions():
    """A beat model evaluated end to end: the simulated tracker on clean MIDI, the annotated beats,
    and the conditions it cannot have here skipped with the reason."""
    if not ensure_test_beats():
        print("  (skipped: data/processed/test missing)")
        return
    import torch
    from src.preprocessing.vocab import EOS, Vocab
    from src.scoreformer import PRESETS, build_model
    vocab = Vocab.load(str(D / "vocab.json"))
    cfg = dict(vocab_size=len(vocab), dropout=0.0, max_tokens=2048, max_notes=1024, n_features=15,
               beat_features=True, **PRESETS["small"])
    model = build_model("scoreformer", cfg)
    with torch.no_grad():  # ends every window at once (fast)
        model.out_norm.weight.zero_()
        model.out_norm.bias.zero_()
        model.out_norm.bias[0] = 1.0
        model.head.weight.zero_()
        model.head.weight[vocab.stoi[EOS], 0] = 10.0
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "b").mkdir()
        torch.save({"model": model.state_dict(), "arch": "scoreformer", "config": cfg}, tmp / "b" / "best.pt")
        (tmp / "b" / "config.json").write_text(json.dumps({"model": cfg}))
        BeatNoise().to_json(tmp / "noise.json")
        r = subprocess.run([sys.executable, "scripts/evaluate_pieces.py", "--run", "b", "--runs-dir", str(tmp),
                            "--device", "cpu", "--normal-priority", "--limit-perfs", "1", "--modes", "none,guarded,piece",
                            "--conditions", "clean,clean_gt_beats,real", "--beat-noise", str(tmp / "noise.json")],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        res = json.loads((tmp / "b" / "eval_pieces_best.json").read_text())
        assert set(res) >= {"clean", "clean_gt_beats"} and set(res["clean"]) == {"none", "guarded", "piece"}
        assert res["clean"]["piece"]["guard_margin"] == 0.0 and "meter_fix_rate" in res["clean"]["piece"]
        if not (D / "test_transcribed" / "beats_tracked.npz").exists():
            assert "skip real:" in r.stdout, r.stdout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pipeline_beats():
    """The demo pipeline with a beat model: MIDI input -> no beats; audio -> the tracker's beats."""
    from src.pipeline import ScorePipeline, performance_windows
    from src.preprocessing.performance import load_performance
    from tests.test_pipeline import WINDOW, make_checkpoint, make_performance
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        midi = tmp / "perf.mid"
        make_performance(midi)
        pipe = ScorePipeline(make_checkpoint(tmp, n_features=15, beat_features=True), device="cpu", window_notes=WINDOW)
        assert pipe.info["beats"] is True
        res = pipe.run(midi, tmp / "m", mode="piece")
        assert res.stats["beats"] == "none (MIDI input)" and res.tokens is not None
        notes, pedal = load_performance(midi)
        t = np.arange(0.0, float(notes[:, 1].max()) + 2, 0.5)
        wins = performance_windows(notes, pedal, size=WINDOW, beats=(t, np.arange(len(t)) % 4 == 0))
        f = pipe.features(wins[1]).numpy()
        assert f.shape[1] == 15 and f[:, -1].mean() > 0.9  # every note between two beats
        assert np.allclose(pipe.features({**wins[1], "beat_t": np.zeros(0), "beat_d": np.zeros(0, bool)}).numpy()[:, 8:], 0)
        # audio input: Beat This! (stand-in) runs after transcription
        calls = fake_beat_this(t, t[::4])
        try:
            pipe.transcribe = lambda audio, out: midi  # no Transkun in the test
            (tmp / "a.wav").write_bytes(b"")
            res = pipe.run(tmp / "a.wav", tmp / "a", mode="meter")
        finally:
            sys.modules.pop("beat_this", None), sys.modules.pop("beat_this.inference", None)
        assert res.stats["beats"].startswith("Beat This!: ") and calls == ["final0"] and "beats_s" in res.stats


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
