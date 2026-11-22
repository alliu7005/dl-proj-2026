"""Tests for the third v2 round's system pieces outside the models' training.

    python -m tests.test_v2_system        # or: pytest tests/test_v2_system.py

Covers the onset-scale metric (src/eval/metrics.py), re-scoring saved predictions
(scripts/rescore.py), the MusicXML export's last resort (src/preprocessing/detokenize.py), the
MIDI synthesizer (src/synth.py), V2Pipeline end to end on tiny untrained models (src/pipeline.py:
given beats, beats of a MIDI input rendered to audio, no beats), the demo serving both systems
(demo/app.py), the job graph of pace/v2.sh (stand-in sbatch) and pace/v2_jobs.sh (stand-in python).
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
            assert Path(out).exists() and rep["attempt"] == 1 and rep["emptied_bars"] == [1], rep
            assert rep["fixed_bars"] == {1: "empty"}, rep
            import music21 as m21
            back = m21.converter.parse(out)
            pitches = sorted({n.pitch.midi for n in back.recurse().notes})
            assert pitches == [60, 62], pitches
        finally:
            dt.write_musicxml = real_write


def test_export_fixes_only_failing_bars():
    """When some bars cannot be written exactly, only those are snapped to the grid: the other
    bars keep their exact positions and lengths (here quintuplets, which a 1/48 grid would move);
    a note past its bar's end is written at the start of the next bar, when it sounds."""
    import music21 as m21
    import src.preprocessing.detokenize as dt
    sept = [Placed(0, Fraction(k, 7), 60 + k % 5, Fraction(1, 7), 0) for k in range(7)]
    quint = [Placed(2, Fraction(k, 5), 70 + k, Fraction(1, 5), 0) for k in range(5)]
    toks = render(Window(bars=[("4/4", 0)] * 3), sept + [Placed(1, Fraction(0), 65, Fraction(4), 0)] + quint)
    real_write = dt.write_musicxml

    def no_septuplets(score, path):  # an exporter that cannot write 1/7-quarter notes
        if any(Fraction(n.quarterLength).limit_denominator(1000).denominator % 7 == 0 for n in score.recurse().notes):
            raise ValueError("septuplet")
        return real_write(score, path)
    with tempfile.TemporaryDirectory() as t:
        dt.write_musicxml = no_septuplets
        try:
            rep = {}
            out = dt.tokens_to_musicxml(toks, str(Path(t) / "q.musicxml"), report=rep)
        finally:
            dt.write_musicxml = real_write
        assert rep["attempt"] == 1 and rep["fixed_bars"] == {0: "grid"} and rep["emptied_bars"] == [], rep
        back = m21.converter.parse(out)
        bars = list(back.parts[0].getElementsByClass("Measure"))
        q = [(float(n.offset), Fraction(n.quarterLength).limit_denominator(1000)) for n in bars[2].recurse().notes]
        assert q == [(k / 5, Fraction(1, 5)) for k in range(5)], q   # untouched: the exact quintuplets
        # overflow: a note at beat 5 of a 4/4 bar is written at beat 1 of the next bar
        over = dt.tokens_to_musicxml(["bar", "ts_4/4", "key_0", "pos_4", "p_60", "d_1", "bar", "pos_1", "p_62", "d_1"],
                                     str(Path(t) / "o.musicxml"))
        bars = list(m21.converter.parse(over).parts[0].getElementsByClass("Measure"))
        got = [(i, float(n.offset), n.pitch.midi) for i, m in enumerate(bars) for n in m.recurse().notes]
        assert got == [(1, 0.0, 60), (1, 1.0, 62)], got


def test_spelling():
    """Notes are named for the key signature: what the signature implies first (Db, Gb in five
    flats; Cb, Fb in seven; E# in six sharps), else its direction (flats in flat keys, sharps in
    sharp keys, C# F# G# Eb Bb with none); every name keeps its MIDI pitch, octaves included (B#,
    Cb across the octave line); the written score has no accidental on a note its key implies."""
    from src.preprocessing.detokenize import spell, tokens_to_musicxml
    names = lambda key: [spell(m, key).name for m in range(60, 72)]  # noqa: E731
    assert names(-5) == ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]
    assert names(-7)[4] == "F-" and names(-7)[11] == "C-"
    assert names(6)[5] == "E#" and names(2) == ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    assert names(0) == ["C", "C#", "D", "E-", "E", "F", "F#", "G", "G#", "A", "B-", "B"]
    for key in range(-7, 8):
        for m in range(21, 109):
            assert spell(m, key).midi == m, (key, m, spell(m, key))
    toks = render(Window(bars=[("4/4", -5)]), [Placed(0, Fraction(k), p, Fraction(1), 0)
                                               for k, p in enumerate((61, 66, 70, 59))])
    with tempfile.TemporaryDirectory() as t:
        import music21 as m21
        back = m21.converter.parse(tokens_to_musicxml(toks, str(Path(t) / "k.musicxml")))
        notes = list(back.parts[0].recurse().notes)
        assert [n.pitch.nameWithOctave for n in notes] == ["D-4", "G-4", "B-4", "B3"], [n.pitch for n in notes]
        assert [n.pitch.accidental.name if n.pitch.accidental else None for n in notes][:3] == ["flat"] * 3


def test_playback_tempo():
    """The score MIDI is played at the tempo that makes the written score sound as long as the
    performance (its first note's start to its last note's end: four quarter notes, one per 4/4 bar,
    span 13 quarters; a pickup counts from its first note), within 20-300; the MusicXML carries no
    tempo mark, as before."""
    import pretty_midi
    from src.pipeline import Result, engrave, playback_qpm
    from src.preprocessing.detokenize import tokens_to_score
    bars4 = render(Window(bars=[("4/4", 0)] * 4), [Placed(b, Fraction(0), 60, Fraction(1), 0) for b in range(4)])
    assert abs(playback_qpm(tokens_to_score(bars4), 13 * 60 / 90) - 90.0) < 1e-6
    pickup = render(Window(bars=[("4/4", 0)] * 3, opens_with_bar=False, start_pos=None),
                    [Placed(0, Fraction(2), 60, Fraction(1), 0)] + [Placed(b, Fraction(0), 62, Fraction(1), 0) for b in (1, 2)])
    assert abs(playback_qpm(tokens_to_score(pickup), 7.0) - 60.0) < 1e-6   # beat 3 of the pickup to bar 3's end of note 1
    assert playback_qpm(tokens_to_score(bars4), 0.01) == 300.0 and playback_qpm(tokens_to_score(bars4), None) is None
    with tempfile.TemporaryDirectory() as t:
        res = Result(bars4, {})
        rep = engrave(bars4, Path(t), res, 13 * 60 / 90)  # not music21's default 120
        pm = pretty_midi.PrettyMIDI(str(res.files["score_midi"]))
        assert rep["playback_qpm"] == 90.0 and abs(pm.get_tempo_changes()[1][0] - 90.0) < 1e-3
        assert abs(pm.get_end_time() - 13 * 60 / 90) < 0.01   # as long as the performance
        assert "metronome" not in Path(res.files["musicxml"]).read_text().lower()


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
        if "score_midi" in res.files:  # played back as long as the performance
            import pretty_midi
            length = pretty_midi.PrettyMIDI(str(res.files["score_midi"])).get_end_time()
            perf = float(notes[:, 2].max() - notes[:, 1].min())
            # (these untrained models write far too many quarters: the tempo then stops at its 300 cap)
            ok = 0.98 * perf < length < 1.02 * perf if st["playback_qpm"] < 300 else length > perf
            assert ok, (length, perf, st["playback_qpm"])
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


def _stub_bin(tmp: Path, sbatch: bool = True):
    """bin/ with a stand-in sbatch (records each submission: id, NAME, ARGS, arguments)."""
    (tmp / "bin").mkdir(exist_ok=True)
    if sbatch:
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\t%s\\t%s\\n" "$n" "${NAME:-}" '
                                            '"${ARGS:-}" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "sbatch").chmod(0o755)


def test_v2_batch():
    """pace/v2.sh in two waves of at most 8 jobs (PACE's per-user submission limit). Wave 1: the
    tests; per meter group two GPU jobs (the second resuming the first); one job for both new
    labellers; one CPU job for the grids and the rendered-MIDI beats; and a last job, after all
    of them, that submits wave 2. A job waiting on another's success is cancelled if that fails.
    Wave 2 (at most 7, plus the job submitting it): one evaluation job per labeller (the first
    also nl_large); v2 on whole pieces after the evaluations; the seq2seq systems in two jobs (the
    rendered-MIDI beats, and those scored before written again); the rescoring and the scoring
    in one last job after all of them."""
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2.sh", tmp / "pace")
        _stub_bin(tmp)
        bd = tmp / "bd"
        bd.mkdir()

        def submit(wave, env=None):
            for f in (tmp / "jobs.txt", tmp / "jobs.txt.n"):
                if f.exists():
                    f.unlink()
            r = subprocess.run([shutil.which("bash"), "-c", f'export PATH="$PWD/bin:$PATH"; bash pace/v2.sh {wave}'],
                               cwd=tmp, env=dict(os.environ, SUBMITS=str(tmp / "jobs.txt"), BD=str(bd), **(env or {})),
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            rows = [ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()]
            by = {}
            for row in rows:
                by.setdefault(row[3].split("-J ")[1].split()[0], []).append(row)
            return r.stdout, rows, by
        out, rows, by = submit("")
        assert len(rows) <= 8 and "wave 2 is submitted by job" in out, (len(rows), out)
        tests = by["v2tests"][0]
        assert "tests.test_meter" in tests[3] and "tests.test_v2_system" in tests[3] and "--gres" not in tests[3]
        for grp, runs in (("meter_v3+meter_v3_s1", "meter_v3 meter_v3_s1"), ("meter_v3_s2+meter_v3_large", "meter_v3_s2 meter_v3_large")):
            a, b = by["met_" + grp]
            assert f"--dependency=afterok:{tests[0]}" in a[3] and f"--dependency=afterany:{a[0]}" in b[3]
            assert all(f"bash pace/v2_jobs.sh meters {runs}" in x[3] and "--gres=gpu:1" in x[3] for x in (a, b))
            assert "--kill-on-invalid-dep=yes" in a[3] and "--kill-on-invalid-dep" not in b[3]
        lab = by["labellers"][0]
        assert "v2_jobs.sh labellers nl_base_gn nl_base_gn_s1" in lab[3] and f"afterok:{tests[0]}" in lab[3]
        grids = by["v2grids"][0]
        assert "v2_jobs.sh grids" in grids[3] and "--gres" not in grids[3]
        assert "--kill-on-invalid-dep=yes" in lab[3] and "--kill-on-invalid-dep=yes" in grids[3]
        nx = by["v2next"][0]
        others = [r[0] for r in rows if r is not nx and r is not tests]
        # wave 2 only if the tests passed (cancelled with the rest otherwise), after everything else ended
        assert f"--dependency=afterok:{tests[0]},afterany:{':'.join(others)}" in nx[3], nx[3]
        assert "--kill-on-invalid-dep=yes" in nx[3] and "bash pace/v2.sh wave2" in nx[3]
        # wave 2 without the system comparison set up: the evaluations and the rescoring
        out, rows, by = submit("wave2")
        assert "pace/bd_setup.sh" in out and set(by) == {"evals_nl_base", "evals_nl_base_gn", "evals_nl_base_gn_s1", "v2final"}
        evals = [by[k][0][0] for k in sorted(by) if k.startswith("evals_")]
        assert all("--dependency" not in by[k][0][3] for k in by if k.startswith("evals_"))
        assert "v2_jobs.sh evals nl_base_gn_s1" in by["evals_nl_base_gn_s1"][0][3]
        assert by["evals_nl_base"][0][3].rstrip().endswith("v2_jobs.sh evals nl_base nl_large")
        fin = by["v2final"][0]
        assert set(fin[3].split("--dependency=afterany:")[1].split()[0].split(":")) == set(evals)
        assert "rescore.py --workers 16" in fin[3] and "compare_runs.py" in fin[3] and "beyer_dai.py" not in fin[3]
        # with it set up, and one seq2seq system scored before
        (tmp / "outputs" / "system_eval" / "sf_base_reg_cab_meter").mkdir(parents=True)
        (tmp / "outputs" / "system_eval" / "sf_base_reg_cab_meter" / "summary.json").write_text("{}")
        (tmp / "outputs" / "system_eval" / "manifest.json").write_text("[]")
        (bd / "MIDI2ScoreTF.ckpt").write_text("x")
        out, rows, by = submit("wave2")
        assert len(rows) <= 7, len(rows)   # + the job submitting them: 8
        evals = [by[k][0][0] for k in sorted(by) if k.startswith("evals_")]
        v2 = by["sys_v2"][0]
        assert "v2_jobs.sh systems nl_base+meter_v3 nl_base_gn+meter_v3" in v2[3]
        assert set(v2[3].split("--dependency=afterany:")[1].split()[0].split(":")) == set(evals)
        a, b = by["sys_seq_a"][0], by["sys_seq_b"][0]
        assert "systems sf_base_reg_cab:meter:sb sf_base_reg_cab_c128:meter:sb" in a[3]
        assert b[3].rstrip().endswith("systems sf_base_reg_cab:meter") and "s2s_m_cab" not in b[3]   # only the one scored before
        fin = by["v2final"][0]
        assert set(fin[3].split("--dependency=afterany:")[1].split()[0].split(":")) == set(evals) | {v2[0], a[0], b[0]}
        assert "rescore.py" in fin[3] and "beyer_dai.py score" in fin[3] and "--systems all" in fin[3]
        assert fin[3].index("rescore.py") < fin[3].index("beyer_dai.py")


def test_rebuild_batch():
    """pace/rebuild.sh: one CPU job that keeps the data in use (moved to data/processed_pre1122),
    rebuilds with its vocabulary and feature scaling, every derived step in order (the note labels
    before the meter data), the tracked beats copied over, the data tests last; refused when the
    old folder exists already. pace/v2.sh with AFTER waits for it."""
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        for f in ("rebuild.sh", "v2.sh"):
            shutil.copy(ROOT / "pace" / f, tmp / "pace")
        _stub_bin(tmp)
        (tmp / "data" / "processed").mkdir(parents=True)
        (tmp / "data" / "processed" / "vocab.json").write_text("{}")

        def run(script, env=None):
            for f in (tmp / "jobs.txt", tmp / "jobs.txt.n"):
                if f.exists():
                    f.unlink()
            r = subprocess.run([shutil.which("bash"), "-c", f'export PATH="$PWD/bin:$PATH"; bash {script}'], cwd=tmp,
                               env=dict(os.environ, SUBMITS=str(tmp / "jobs.txt"), BD=str(tmp / "nobd"), **(env or {})),
                               capture_output=True, text=True)
            rows = [ln.split("	") for ln in (tmp / "jobs.txt").read_text().splitlines()] if (tmp / "jobs.txt").exists() else []
            return r, rows
        r, rows = run("pace/rebuild.sh")
        assert r.returncode == 0 and len(rows) == 1 and "AFTER=101 bash pace/v2.sh" in r.stdout, r.stdout + r.stderr
        job = rows[0][3]
        assert "--gres" not in job and "mv data/processed data/processed_pre1122" in job
        order = ["mv data/processed", "build_dataset.py --vocab data/processed_pre1122/vocab.json",
                 "cp data/processed_pre1122/feature_stats.json", "build_transcribed.py --asap-split test",
                 "build_metrical.py", "build_beats.py", "build_notelabels.py", "build_meter_data.py --workers 8 --check",
                 "tests.test_integrity", "tests.test_preprocessing"]
        pos = [job.index(x) for x in order]
        assert pos == sorted(pos), list(zip(order, pos))
        assert "beats_tracked" in job and "set -euo pipefail" in job
        (tmp / "data" / "processed_pre1122").mkdir()
        r, rows = run("pace/rebuild.sh")
        assert r.returncode == 1 and not rows and "exists already" in r.stdout
        r, rows = run("pace/v2.sh", {"AFTER": "555"})
        tests = next(x for x in rows if "-J v2tests" in x[3])
        assert "--dependency=afterok:555" in tests[3] and "--kill-on-invalid-dep=yes" in tests[3], tests[3]


def test_v2_jobs():
    """pace/v2_jobs.sh with a stand-in python: meter models trained side by side with their own
    options and logs, then each finished one tuned and evaluated (once: not again on the next job),
    an unfinished one failing the job; a labeller not trained yet resumed before its evaluations,
    an earlier one evaluated again on the true grid and every grid; whole-piece systems by kind, a
    v2 system with an unfinished model refused; the earlier meter models' grids decoded again."""
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_jobs.sh", tmp / "pace")
        _stub_bin(tmp, sbatch=False)
        # stand-in python: logs its arguments; training finishes the runs named in $DONE (status + an
        # old best.pt), evaluations write their result file
        (tmp / "bin" / "python").write_text(r'''#!/bin/bash
echo "$*" >> "$CALLS"
name=""; prev=""
for a in "$@"; do [ "$prev" = "--name" ] || [ "$prev" = "--run" ] && name=$a; prev=$a; done
case "$1" in
  scripts/train_meter.py|scripts/train_labeller.py)
    mkdir -p "outputs/runs/$name"; touch "outputs/runs/$name/last.pt"
    case " $DONE " in *" $name "*) echo '{"phase": "done"}' > "outputs/runs/$name/status.json"
      touch -d '2 days ago' "outputs/runs/$name/best.pt" ;; *) echo '{"phase": "train"}' > "outputs/runs/$name/status.json" ;; esac ;;
  scripts/eval_meter.py) echo '{}' > "outputs/runs/$name/eval_meter.json" ;;
esac
exit 0
''')
        (tmp / "bin" / "python").chmod(0o755)

        def run(args, done=""):
            calls = tmp / "calls.txt"
            if calls.exists():
                calls.unlink()
            r = subprocess.run([shutil.which("bash"), "-c", f'export PATH="$PWD/bin:$PATH"; bash pace/v2_jobs.sh {args}'],
                               cwd=tmp, env=dict(os.environ, CALLS=str(calls), DONE=done, SLURM_JOB_ID="77"),
                               capture_output=True, text=True)
            return r.returncode, (calls.read_text().splitlines() if calls.exists() else []), r.stdout + r.stderr
        code, calls, out = run("meters meter_v3 meter_v3_large", done="meter_v3")
        trains = [c for c in calls if c.startswith("scripts/train_meter.py")]
        assert len(trains) == 2 and code == 1, (calls, out)   # meter_v3_large not done: the job fails
        assert any("--name meter_v3 --steps 60000 --patience 8 --workers 6" in c and "--max-hours 7.5" in c for c in trains)
        assert any("--name meter_v3_large" in c and "--d-model 384 --layers 8" in c for c in trains)
        assert "--resume" not in " ".join(trains) and (tmp / "slurm-met_meter_v3-77.log").exists()
        assert [c for c in calls if c.startswith("scripts/eval_meter.py")] == ["scripts/eval_meter.py --run meter_v3 --tune"]
        code, calls, out = run("meters meter_v3 meter_v3_large", done="meter_v3 meter_v3_large")
        assert code == 0, out
        assert calls == ["scripts/train_meter.py --name meter_v3_large --steps 60000 --patience 8 --d-model 384 --layers 8 "
                         "--workers 6 --normal-priority --max-hours 7.5 --resume",
                         "scripts/eval_meter.py --run meter_v3_large --tune"], calls   # meter_v3: done and evaluated
        code, calls, out = run("evals nl_base nl_large")
        for nl in ("nl_base", "nl_large"):
            mine = [c for c in calls if f"--run {nl} " in c + " "]
            assert mine[0] == f"scripts/eval_labeller.py --run {nl} --normal-priority", mine   # the true grid
            assert [c.split("--grid meter:")[1].split()[0] for c in mine[1:]] == [
                "meter_oracle", "meter_v2_nonote", "meter_v3", "meter_v3_s1", "meter_v3_s2", "meter_v3_large"], mine
        code, calls, out = run("evals nl_base_gn", done="nl_base_gn")
        assert calls[0].startswith("scripts/train_labeller.py --name nl_base_gn --size base --grid-noise 0.8")
        assert calls[1] == "scripts/eval_labeller.py --run nl_base_gn --normal-priority"
        assert [c.split("--grid meter:")[1].split()[0] for c in calls[2:]][:2] == ["meter_oracle", "meter_v2_nonote"]
        code, calls, out = run("systems nl_base_gn+meter_v3 sf_base_reg_cab:meter:sb s2s_m_cab:meter")
        assert calls == ["scripts/system_scores.py --v2 nl_base_gn+meter_v3",
                         "scripts/system_scores.py --run sf_base_reg_cab --mode meter --midi-beats synth",
                         "scripts/system_scores.py --run s2s_m_cab --mode meter"], calls
        code, calls, out = run("labellers nl_base_gn nl_base_gn_s1", done="nl_base_gn nl_base_gn_s1")
        # nl_base_gn finished above (its evals job trained it): only nl_base_gn_s1 trains, both are evaluated
        trains = [c for c in calls if c.startswith("scripts/train_labeller.py")]
        assert len(trains) == 1 and "--name nl_base_gn_s1" in trains[0] and "--seed 1" in trains[0] and code == 0
        assert [c for c in calls if c.startswith("scripts/eval_labeller.py")] == [
            "scripts/eval_labeller.py --run nl_base_gn --normal-priority", "scripts/eval_labeller.py --run nl_base_gn_s1 --normal-priority"]
        code, calls, out = run("systems nl_base_gn+meter_v3_s1")   # meter_v3_s1 never trained here: refused
        assert calls == [] and code == 1 and "not finished" in out, (calls, out)
        (tmp / "outputs" / "runs" / "meter_v2_nonote").mkdir(parents=True)
        (tmp / "outputs" / "runs" / "meter_v2_nonote" / "best.pt").write_text("x")
        code, calls, out = run("grids")
        assert calls == ["scripts/eval_meter.py --run meter_v2_nonote --device cpu",   # the others: no best.pt here
                         "scripts/eval_meter.py --run meter_oracle --oracle --device cpu", "scripts/synth_beats.py"], calls


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
