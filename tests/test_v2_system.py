"""Tests for the third v2 round's system pieces outside the models' training.

    python -m tests.test_v2_system        # or: pytest tests/test_v2_system.py

Covers the onset-scale metric (src/eval/metrics.py), re-scoring saved predictions
(scripts/rescore.py), the MusicXML export's last resort (src/preprocessing/detokenize.py), the
MIDI synthesizer (src/synth.py), V2Pipeline end to end on tiny untrained models (src/pipeline.py:
given beats, beats of a MIDI input rendered to audio, no beats), the demo serving both systems
(demo/app.py), and the job graph of pace/v2.sh (stand-in sbatch).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.metrics import Accumulator  # noqa: E402
from src.notelabels import Placed, Window, render  # noqa: E402

D = ROOT / "data" / "processed"
V = D / "vocab.json"


def rhythm(scale=Fraction(1)):
    """Eight notes of a 3/4 melody (or the same at ``scale`` times the note values)."""
    pos = [Fraction(x) for x in ("0", "1", "2", "3", "7/2", "4", "5", "11/2")]
    out = []
    for k, p in enumerate(pos):
        q = p * scale
        bar_len = 3 * scale
        out.append(Placed(int(q // bar_len), q % bar_len, 60 + k, Fraction(1, 2) * scale, 0))
    return out


def test_onset_scaled():
    """3/8 written for 3/4 (every note value halved): onset F1 and onset-shift F1 miss most notes,
    the onset-scale F1 gets them all; on a right score all three agree; a wrong rhythm stays wrong."""
    gt = render(Window(bars=[("3/4", 0)] * 3), rhythm())
    half = render(Window(bars=[("3/8", 0)] * 3), rhythm(Fraction(1, 2)))
    double = render(Window(bars=[("3/2", 0)] * 3), rhythm(Fraction(2)))
    for pred in (half, double):
        a = Accumulator()
        a.add(gt, pred)
        r = a.result()
        assert r["onset_f1"] < 0.5 and r["onset_shift_f1"] < 0.7 and r["onset_scaled_f1"] == 1.0, r
        assert r["onset_scaled_p"] == 1.0 and r["onset_scaled_r"] == 1.0
    a = Accumulator()
    a.add(gt, gt)
    r = a.result()
    assert r["onset_f1"] == r["onset_shift_f1"] == r["onset_scaled_f1"] == 1.0
    wrong = render(Window(bars=[("3/4", 0)] * 3), [Placed(0, Fraction(k, 3), 60 + k, Fraction(1, 3), 0) for k in range(8)])
    a = Accumulator()
    a.add(gt, wrong)
    assert a.result()["onset_scaled_f1"] < 0.5


def test_rescore():
    """scripts/rescore.py adds the new metric to a run's stored results from its saved predictions,
    the same value an evaluation computes now; a mode whose re-scored onset F1 differs from the
    stored one (other windows, other targets) is left alone."""
    if not (D / "test" / "index.npy").exists():
        print("  (skipped: no data/processed/test)")
        return
    from scripts.rescore import main as rescore
    from src.preprocessing.store import SegmentStore
    from src.preprocessing.vocab import Vocab
    store, vocab = SegmentStore(D / "test"), Vocab.load(str(V))
    with tempfile.TemporaryDirectory() as t:
        run = Path(t) / "r"
        run.mkdir()
        rows, acc = [], {"a": Accumulator(), "b": Accumulator()}
        for i in range(6):
            m = store.meta[i]
            gt = vocab.decode(store.get(i)[2].tolist())
            pred = gt[: len(gt) // 2] if i % 2 else gt
            for mode in ("a", "b"):
                rows.append({"performance": m["performance"], "win": m["win"], "mode": mode, "pred": pred})
                acc[mode].add(gt, pred)
        (run / "pred_pieces_best_clean.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        full = {k: v.result() for k, v in acc.items()}
        stored = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("onset_scaled")} for k, v in full.items()}
        stored["b"]["onset_f1"] += 0.01  # as if evaluated on other windows
        stored["a"]["guard_margin"] = 0.5  # keys the metric does not make are kept
        (run / "eval_pieces_best.json").write_text(json.dumps({"clean": stored}))
        rescore(["--runs", "r", "--runs-dir", t, "--data", str(D)])
        out = json.loads((run / "eval_pieces_best.json").read_text())["clean"]
        assert out["a"]["onset_scaled_f1"] == full["a"]["onset_scaled_f1"] and out["a"]["guard_margin"] == 0.5
        assert "onset_scaled_f1" not in out["b"]


def test_export_last_resort():
    """A score some bars of which cannot be exported at all: those bars are emptied, the rest is
    written, and the report says which; a score that exports needs none of it."""
    import src.preprocessing.detokenize as dt
    good = render(Window(bars=[("4/4", 0)] * 3), [Placed(b, Fraction(k), 60 + b, Fraction(1), 0)
                                                   for b in range(3) for k in range(4)])
    real_write = dt.write_musicxml

    def picky(score, path):  # an exporter that cannot write pitch 61 (bar 1)
        if any(getattr(n, "pitch", None) is not None and n.pitch.midi == 61 for n in score.recurse().notes):
            raise ValueError("cannot export pitch 61")
        return real_write(score, path)
    with tempfile.TemporaryDirectory() as t:
        rep = {}
        dt.tokens_to_musicxml(good, str(Path(t) / "a.musicxml"), report=rep)
        assert rep == {"emptied_bars": [], "attempt": 0}, rep
        dt.write_musicxml = picky
        try:
            rep = {}
            out = dt.tokens_to_musicxml(good, str(Path(t) / "b.musicxml"), report=rep)
            assert Path(out).exists() and rep["attempt"] == 4 and rep["emptied_bars"] == [1], rep
            import music21 as m21
            back = m21.converter.parse(out)
            pitches = sorted({n.pitch.midi for n in back.recurse().notes})
            assert pitches == [60, 62], pitches
        finally:
            dt.write_musicxml = real_write


def test_synth():
    """The rendered MIDI: silence before the first note, energy right after each onset, a short
    release after each offset, peak 0.9; a 16-bit mono WAV of it."""
    from src.synth import SR, render as synth, write_wav
    notes = np.array([[60, 0.5, 0.9, 100], [64, 1.0, 1.2, 40], [30, 1.5, 3.0, 80], [110, 2.0, 2.1, 90]], np.float32)
    a = synth(notes)
    assert a.dtype == np.float32 and abs(np.abs(a).max() - 0.9) < 1e-6
    assert np.abs(a[: int(0.49 * SR)]).max() == 0.0
    rms = lambda lo, hi: float(np.sqrt(np.mean(a[int(lo * SR):int(hi * SR)] ** 2)))  # noqa: E731
    assert rms(0.5, 0.55) > 10 * rms(0.45, 0.5) + 1e-6
    assert 0 < rms(3.0, 3.08) < rms(2.9, 3.0) and rms(3.1, 3.5) == 0.0  # the last note's release, then silence
    assert len(a) >= int(4.0 * SR)                                       # 1 s of padding after it
    with tempfile.TemporaryDirectory() as t:
        p = write_wav(Path(t) / "x.wav", a)
        with wave.open(str(p)) as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, SR, len(a))
    assert len(synth(np.zeros((0, 4)))) == SR


def make_v2(d: Path):
    """Untrained tiny labeller + meter checkpoints, the layouts the training scripts save."""
    from src.labeller import PRESETS, build_labeller
    from src.meter import MeterConfig, N_BEAT_FEATURES, build_meter_model
    from src.notelabels import N_GRID_FEATURES, LabelSpace
    from src.preprocessing.vocab import Vocab
    space = LabelSpace(Vocab.load(str(V)).itos)
    torch.manual_seed(0)
    cfg = dict(n_sub=space.n_sub, n_dur=space.n_dur, n_features=10 + N_GRID_FEATURES, dropout=0.0, max_notes=1024,
               context_before=16, context_after=8, **PRESETS["small"])
    st = V.with_name("feature_stats.json")
    if st.exists():
        js = json.loads(st.read_text())
        cfg.update(feat_mean=js["mean"], feat_std=js["std"])
    (d / "nl").mkdir(parents=True)
    torch.save({"model": build_labeller(cfg).state_dict(), "arch": "labeller", "config": cfg}, d / "nl" / "best.pt")
    mc = MeterConfig(n_features=N_BEAT_FEATURES, d_model=32, layers=1, n_heads=4, ff=64).to_dict()
    (d / "m").mkdir(parents=True)
    torch.save({"model": build_meter_model(mc).state_dict(), "arch": "meter", "config": mc}, d / "m" / "best.pt")
    (d / "m" / "decode.json").write_text(json.dumps({"decode": {"bar_weight": 2.0, "ts_weight": 0.5}}))
    return d / "nl" / "best.pt", d / "m" / "best.pt"


def test_v2_pipeline():
    """V2Pipeline on a MIDI file: with given beats, the grid of the whole piece, every own note kept
    or deleted once, all rendered in the grid's bars (as many bar lines as the grid has bars), the
    meter run's tuned decoder used; without beats, Beat This! on the MIDI rendered to audio (a
    stand-in tracker here); without Beat This!, an error and no score."""
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
        make_performance(midi, n=150)
        notes, _ = load_performance(midi)
        pipe = V2Pipeline(lab, met, vocab=V, device="cpu", window_notes=48)
        assert pipe.decode_cfg.bar_weight == 2.0 and pipe.decode_cfg.ts_weight == 0.5
        times = np.arange(0.3, float(notes[:, 1].max()) + 1.0, 0.5)
        beats = (times, np.arange(len(times)) % 4 == 0)
        far = (np.concatenate([beats[0], beats[0][-1] + 60 + 0.5 * np.arange(120)]),  # a recording's other piece
               np.concatenate([beats[1], np.arange(120) % 4 == 0]))
        res = pipe.run(midi, tmp / "a", beats=far)
        assert res.tokens == pipe.run(midi, tmp / "a2", beats=beats).tokens  # beats past the music change nothing
        st = res.stats
        assert st["beats"].startswith("given:") and st["notes"] == len(notes) and st["windows"] >= 3, st
        assert st["notes_kept"] + st["notes_deleted"] == len(notes), st
        # the score's bars: the grid's up to the one the music ends in (the far beats add none)
        assert 0 < st["bars"] <= st["grid_bars"], st
        assert res.tokens.count("bar") in (st["bars"], st["bars"] - 1), (res.tokens.count("bar"), st["bars"])
        assert ("musicxml" in res.files) or any(e.startswith("MusicXML") for e in res.errors), res.errors
        assert sum(st["time_signatures"].values()) == st["bars"] and st["time_signature"] in st["time_signatures"]
        saved = json.loads((tmp / "a" / "result.json").read_text())
        assert saved["stats"]["system"] == "v2" and (tmp / "a" / "tokens.txt").read_text().split() == res.tokens
        heard = []

        def tracker(audio, checkpoint):  # stand-in for Beat This!: the audio it gets, the beats above
            with wave.open(str(audio)) as w:
                heard.append((w.getnframes() / w.getframerate(), checkpoint))
            return beats
        pipe._tracker = tracker
        res = pipe.run(midi, tmp / "b", beat_checkpoint="fold3")
        assert heard and heard[0][1] == "fold3" and heard[0][0] > float(notes[:, 2].max()), heard
        assert res.stats["beats"].startswith("Beat This! on the MIDI rendered to audio (fold3)"), res.stats["beats"]
        assert res.tokens == pipe.run(midi, tmp / "c", beats=beats).tokens  # same beats, same score
        pipe._tracker = False  # Beat This! not installed
        res = pipe.run(midi, tmp / "d")
        assert not res.tokens and "musicxml" not in res.files and res.errors[0].startswith("v2 needs the beats")


def test_demo_serves_both():
    """demo/app.py with a v2 and a seq2seq pipeline: /api/info lists both and the default, a job
    runs on the system it names (v2 without the seq2seq context mode), an unknown system is refused."""
    try:
        from fastapi.testclient import TestClient
    except Exception:  # noqa: BLE001 (needs httpx)
        print("  (skipped: fastapi.testclient needs httpx)")
        return
    sys.path.insert(0, str(ROOT / "demo"))
    from demo.app import create_app
    from src.pipeline import Result
    calls = []

    class Fake:
        def __init__(self, name):
            self.info, self.name = {"system": name, "params_M": 1.0, "device": "cpu"}, name

        def run(self, src, out, progress=None, **kw):
            calls.append((self.name, kw))
            return Result(["bar"], {}, {"system": self.name})
    with tempfile.TemporaryDirectory() as t:
        app = create_app({"v2": Fake("v2"), "seq2seq": Fake("seq2seq")}, Path(t))
        c = TestClient(app)
        info = c.get("/api/info").json()
        assert set(info["systems"]) == {"v2", "seq2seq"} and info["default"] == "v2"
        for system in ("v2", "seq2seq"):
            jid = c.post("/api/jobs", files={"file": ("x.mid", b"MThd")}, data={"system": system, "mode": "guarded"}).json()["id"]
            for _ in range(100):
                if c.get(f"/api/jobs/{jid}").json()["status"] in ("done", "error"):
                    break
                import time
                time.sleep(0.05)
        assert calls == [("v2", {}), ("seq2seq", {"mode": "guarded"})], calls
        assert c.post("/api/jobs", files={"file": ("x.mid", b"MThd")}, data={"system": "v9"}).status_code == 400


def test_v2_batch():
    """pace/v2.sh: the tests first; each meter run in two chained jobs (the second resuming, failing
    unless trained and evaluated); the oracle; the new labellers; every labeller on every grid after
    both (pairs evaluated before skipped); the rescoring after all of those; the whole-piece systems
    after the rendered-MIDI beats and their models, then the scoring after all of them."""
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2.sh", tmp / "pace")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\t%s\\t%s\\n" "$n" "${NAME:-}" '
                                            '"${ARGS:-}" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "sbatch").chmod(0o755)
        (tmp / "outputs" / "runs" / "nl_base").mkdir(parents=True)
        (tmp / "outputs" / "runs" / "nl_base" / "eval_pieces_best.json").write_text(
            json.dumps({"clean": {"grid_meter_meter_v2_nonote": {}}}))
        bd = tmp / "bd"
        bd.mkdir()

        def submit(env):
            for f in (tmp / "jobs.txt", tmp / "jobs.txt.n"):
                if f.exists():
                    f.unlink()
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2.sh'], cwd=tmp,
                               env=dict(os.environ, SUBMITS=str(tmp / "jobs.txt"), BD=str(bd), **env),
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            rows = [ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()]
            return r.stdout, rows
        out, rows = submit({})
        assert "pace/bd_setup.sh" in out  # no manifest / checkpoint: no whole-piece comparison
        name = lambda r: r[3].split("-J ")[1].split()[0] if "-J " in r[3] else r[1]  # noqa: E731
        by = {}
        for r in rows:
            by.setdefault(name(r), []).append(r)
        tests = by["v2tests"][0]
        assert "tests.test_meter" in tests[3] and "tests.test_notelabels" in tests[3]
        meters = ["meter_v3", "meter_v3_s1", "meter_v3_s2", "meter_v3_large"]
        for m in meters:
            a, b = by["met_" + m]
            assert f"afterok:{tests[0]}" in a[3] and f"--dependency=afterany:{a[0]}" in b[3]
            assert f"train_meter.py --name {m} --steps 60000 --patience 8" in a[3] and "--max-hours 7.5" in a[3]
            assert f"eval_meter.py --run {m} --tune" in a[3] and "--resume" in a[3]
            assert b[3].rstrip().endswith(f"[ -f outputs/runs/{m}/eval_meter.json ]"), b[3][-120:]
        assert "--seed 2" in by["met_meter_v3_s2"][0][3] and "--d-model 384 --layers 8" in by["met_meter_v3_large"][0][3]
        assert "--oracle" in by["metev_meter_oracle"][0][3]
        new = {r[1]: r for r in rows if r[1] in ("nl_base_gn", "nl_base_gn_s1")}
        assert set(new) == {"nl_base_gn", "nl_base_gn_s1"} and "--grid-noise 0.8" in new["nl_base_gn"][2]
        assert "--seed 1" in new["nl_base_gn_s1"][2] and all("train_labeller.sbatch" in r[3] for r in new.values())
        evals = {k: v[0] for k, v in by.items() if k.startswith("v2_")}
        want = {f"v2_{nl}@{m}" for nl in ("nl_base", "nl_base_gn", "nl_base_gn_s1")
                for m in ["meter_oracle", "meter_v2_nonote"] + meters} - {"v2_nl_base@meter_v2_nonote"}
        assert set(evals) == want, sorted(set(evals) ^ want)
        jid = {"meter_oracle": by["metev_meter_oracle"][0][0], **{m: by["met_" + m][1][0] for m in meters},
               **{k: v[0] for k, v in new.items()}}
        for k, r in evals.items():
            nl, m = k[3:].split("@")
            deps = [jid[x] for x in (m, nl) if x in jid]
            assert f"--dependency=afterok:{':'.join(deps)}" in r[3] if deps else "--dependency" not in r[3], (k, r[3])
            assert f"--grid meter:{m}" in r[3]
        re = by["rescore"][0]
        assert all(r[0] in re[3] for r in evals.values()) and "rescore.py" in re[3] and "compare_runs.py" in re[3]
        # with the system comparison set up
        (tmp / "outputs" / "system_eval" / "sf_base_reg_cab_meter").mkdir(parents=True)
        (tmp / "outputs" / "system_eval" / "sf_base_reg_cab_meter" / "summary.json").write_text("{}")
        (tmp / "outputs" / "system_eval" / "manifest.json").write_text("[]")
        (bd / "MIDI2ScoreTF.ckpt").write_text("x")
        out, rows = submit({"METERS": "meter_v3", "NEW_NLS": "nl_base_gn", "GRIDS_OLD": "",
                            "SYSTEMS": "nl_base_gn+meter_v3 nl_base+meter_v2_nonote"})
        by = {}
        for r in rows:
            by.setdefault(name(r), []).append(r)
        sb = by["synthbeats"][0]
        assert "synth_beats.py" in sb[3]
        s1, s2 = by["sys_v2_nl_base_gn_meter_v3"][0], by["sys_v2_nl_base_meter_v2_nonote"][0]
        nl_job = next(r[0] for r in rows if r[1] == "nl_base_gn")
        assert f"--dependency=afterok:{sb[0]}:{by['met_meter_v3'][1][0]}:{nl_job}" in s1[3]
        assert f"--dependency=afterok:{sb[0]} " in s2[3] and "--v2 nl_base+meter_v2_nonote" in s2[3]
        seq = [by[f"sys_{r}_sb"][0] for r in ("sf_base_reg_cab", "sf_base_reg_cab_c128")]
        assert all("--midi-beats synth" in q[3] and f"afterok:{sb[0]}" in q[3] for q in seq)
        fix = by["sysfix_sf_base_reg_cab"][0]  # the one scored system here: its missing scores again
        assert "--run sf_base_reg_cab --mode meter --only-missing" in fix[3] and "sysfix_s2s_m_cab" not in by
        sc = by["sysscore"][0]
        assert f"afterany:{':'.join(r[0] for r in [s1, s2] + seq + [fix])}" in sc[3] and "--systems all" in sc[3]


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
