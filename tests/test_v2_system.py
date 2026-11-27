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
from tests.batch_env import batch_env  # noqa: E402
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


def test_piece_staves():
    """A third / fourth staff only when enough of the piece is on it; otherwise each of its notes
    goes on its likeliest of the first two (the labeller's own scores, not a pitch rule)."""
    from src.labeller import N_POS, N_STAFF, placed_notes
    from src.notelabels import N_FLAGS, LabelSpace
    from src.pipeline import MIN_EXTRA_STAFF, piece_staves
    from src.preprocessing.vocab import Vocab
    assert piece_staves([]) == 2 and piece_staves([0, 1, 1]) == 2
    assert piece_staves([0] * 610 + [1] * 610 + [3]) == 2            # the demo piece: one stray note
    assert piece_staves([0] * 70 + [1] * 70 + [2] * 60) == 3          # a real third staff
    assert piece_staves([0] * 50 + [1] * 45 + [3] * 5) == 4           # staff 4 used: 3 kept below it
    n = 1000
    assert piece_staves([0] * (n - 20) + [2] * 20) == 3 and MIN_EXTRA_STAFF == 0.02
    assert piece_staves([0] * (n - 19) + [2] * 19) == 2
    space = LabelSpace(Vocab.load(str(V)).itos)
    staff = torch.tensor([[[0.0, 1.0, 0.0, 5.0], [2.0, 0.5, 9.0, 0.0]]])   # 2 notes: argmax 3 and 2
    out = {"keep": torch.ones(1, 2), "delta": torch.zeros(1, 2, 3), "pos": torch.zeros(1, 2, N_POS),
           "sub": torch.zeros(1, 2, space.n_sub), "dur": torch.zeros(1, 2, space.n_dur), "staff": staff,
           "flags": torch.full((1, 2, N_FLAGS), -1.0)}
    args = (out, 0, np.array([60.0, 50.0]), np.array([1, 1]), np.ones(2, bool), space)
    assert [p.staff for p in placed_notes(*args)] == [3, 2] and N_STAFF == 4
    assert [p.staff for p in placed_notes(*args, staves=2)] == [1, 0]   # next best of staves 1 and 2
    assert [p.staff for p in placed_notes(*args, staves=3)] == [1, 2]


def test_tidy_lengths():
    """A written length running just past the next onset of its staff stops there (at most
    TIDY_TOL); one held over moving notes, a gap before the next note, the other staff and the
    last note stay; across a bar line too."""
    from src.pipeline import TIDY_TOL, tidy_lengths
    F = Fraction
    assert TIDY_TOL == 0.125
    notes = [Placed(0, F(0), 60, F(9, 8), 0),     # 1/8 past the next onset (1): trimmed
             Placed(0, F(1), 62, F(3, 4), 0),     # 1/4 short of the next (2): a gap, kept
             Placed(0, F(2), 64, F(4), 0),        # held over the moving notes after it: kept
             Placed(0, F(3), 65, F(1), 0),        # meets the next bar's first note exactly
             Placed(1, F(0), 67, F(1, 2), 0),
             Placed(1, F(1, 2), 69, F(1, 2), 0),  # the last of its staff: kept
             Placed(0, F(0), 48, F(17, 16), 1),   # the other staff: its own next onset (none): kept
             Placed(0, F(7, 2), 71, F(5, 8), 0)]  # 1/8 past bar 1's first note: trimmed across the bar line
    assert tidy_lengths(notes, ["4/4", "4/4"]) == 2
    assert [n.total for n in notes] == [F(1), F(3, 4), F(4), F(1), F(1, 2), F(1, 2), F(17, 16), F(1, 2)]


def test_select_onset_f1():
    """The labeller's checkpoint selection: the validation loss, or minus the mean decoded onset F1."""
    from scripts.train_labeller import selection
    res = {"val_loss": 1.4, "val_onset_f1": 0.84, "val_real_onset_f1": 0.80}
    assert selection(res, "loss") == 1.4 and abs(selection(res, "onset_f1") + 0.82) < 1e-12
    assert selection({"val_loss": 1.0, "val_onset_f1": 0.5}, "onset_f1") == -0.5
    assert selection({"val_loss": 1.0}, "onset_f1") == float("inf")
    from scripts.system_scores import system_name
    assert system_name(None, "meter", "none", "nl+m", delete_above=0.95, tidy=True) == "v2_nl_m_d95_tidy"
    assert system_name(None, "meter", "none", "nl+m", "audio", tidy=True) == "v2_nl_m_tidy_ab"


def test_rescore():
    """scripts/rescore.py adds the new metric to a run's stored results from its saved predictions,
    the same value an evaluation computes now; a mode whose re-scored onset F1 differs from the
    stored one (other windows) is left alone. A mode scored on the targets before the rebuild (no
    stamp) that differs by what the rebuild explains gets every metric re-scored and the stamp; the
    same gap on a stamped mode is refused."""
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
        modes = ("a", "b", "c", "d")
        rows, acc = [], {k: Accumulator() for k in modes}
        for i in range(6):
            m = store.meta[i]
            gt = vocab.decode(store.get(i)[2].tolist())
            pred = gt[: len(gt) // 2] if i % 2 else gt
            for mode in modes:
                rows.append({"performance": m["performance"], "win": m["win"], "mode": mode, "pred": pred})
                acc[mode].add(gt, pred)
        (run / "pred_pieces_best_clean.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        full = {k: v.result() for k, v in acc.items()}
        stored = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("onset_scaled")} for k, v in full.items()}
        stored["b"]["onset_f1"] += 0.01  # as if evaluated on other windows
        stored["a"]["guard_margin"] = 0.5  # keys the metric does not make are kept
        for k in ("c", "d"):   # as if scored on the targets before the rebuild
            stored[k]["onset_f1"] -= 0.0003
            stored[k]["bar_f1"] -= 0.02
            stored[k]["fallback_rate"] = 0.4
        stored["d"]["targets"] = 2   # ... but stamped as the current targets: a mismatch
        (run / "eval_pieces_best.json").write_text(json.dumps({"clean": stored}))
        rescore(["--runs", "r", "--runs-dir", t, "--data", str(D)])
        out = json.loads((run / "eval_pieces_best.json").read_text())["clean"]
        assert out["a"]["onset_scaled_f1"] == full["a"]["onset_scaled_f1"] and out["a"]["guard_margin"] == 0.5
        assert "onset_scaled_f1" not in out["b"]
        c = out["c"]
        assert c["targets"] == 2 and c["fallback_rate"] == 0.4, c
        assert all(c[k] == v for k, v in full["c"].items() if v == v), [k for k, v in full["c"].items() if c[k] != v]
        assert "onset_scaled_f1" not in out["d"] and out["d"]["bar_f1"] == stored["d"]["bar_f1"]


def test_export_graces_and_rests():
    """The export writes a grace note in its main note's voice: one written just before its note
    (where it was played) joins the main note of its staff up to 1/2 quarter later; an after-grace
    (no later note in its bar) stays. Rests music21 would write with two or more dots are split into
    standard rests. Every voice of every measure adds up to its bar."""
    import music21 as m21
    from src.preprocessing.detokenize import _split_rest, tokens_to_score
    F = Fraction
    assert _split_rest(F(1, 8), F(31, 8)) == [(F(1, 8), F(1, 8)), (F(1, 4), F(1, 4)), (F(1, 2), F(1, 2)),
                                                 (F(1), F(1)), (F(2), F(2))]
    assert _split_rest(F(0), F(31, 8))[0] == (F(0), F(2))
    toks = ("bar ts_4/4 key_0 pos_0 staff_0 p_60 d_1 pos_1 sub_7/8 p_62 grace pos_2 p_64 d_2 "
            "staff_1 pos_0 p_48 d_1/8 "
            "bar pos_0 staff_0 p_65 d_3 pos_3 sub_7/8 p_67 grace staff_1 pos_0 p_41 d_4").split()
    score = tokens_to_score(toks)
    up, low = score.parts[0], score.parts[1]
    m1 = up.getElementsByClass("Measure")[0]
    grace = next(n for n in m1.recurse().notes if n.duration.isGrace)
    main = next(n for n in m1.recurse().notes if not n.duration.isGrace and n.pitch.midi == 64)
    assert grace.pitch.midi == 62 and grace.activeSite is main.activeSite and grace.offset == main.offset == 2.0
    m2 = up.getElementsByClass("Measure")[1]
    after = next(n for n in m2.recurse().notes if n.duration.isGrace)
    assert after.pitch.midi == 67 and after.offset == 3.875              # an after-grace stays
    for part in (up, low):
        for meas in part.getElementsByClass("Measure"):
            for v in (meas.voices or [meas]):
                els = list(v.notesAndRests)
                assert all(r.duration.dots < 2 for r in els if r.isRest), [(r.offset, r.duration.dots) for r in els]
                assert abs(sum(e.quarterLength for e in els if not e.duration.isGrace) - 4.0) < 1e-9
    rests = [r for r in low.getElementsByClass("Measure")[0].notesAndRests if r.isRest]
    assert [r.quarterLength for r in rests] == [0.125, 0.25, 0.5, 1.0, 2.0], [r.quarterLength for r in rests]
    with tempfile.TemporaryDirectory() as t:   # and the file says the same
        from src.preprocessing.detokenize import tokens_to_musicxml
        path = Path(t) / "g.musicxml"
        tokens_to_musicxml(toks, str(path))
        back = m21.converter.parse(str(path))
        for meas in back.parts[0].getElementsByClass("Measure"):
            assert meas.duration.quarterLength == 4.0


def test_attach_graces():
    """v2 writes a grace note where it was played; it joins the main note of its staff starting up to
    a quarter after it, across a bar line too, and one with none becomes a short note."""
    from src.pipeline import GRACE_AS_NOTE, attach_graces
    F = Fraction
    notes = [Placed(0, F(7, 4), 62, F(0), 0, grace=True), Placed(0, F(15, 8), 64, F(1), 0),   # 1/8 before
             Placed(0, F(31, 8), 66, F(0), 0, grace=True), Placed(1, F(0), 67, F(2), 0),        # across the bar
             Placed(1, F(0), 50, F(0), 1, grace=True), Placed(1, F(3), 48, F(1), 1),            # 3 quarters: a note
             Placed(1, F(2), 69, F(0), 0, grace=True), Placed(1, F(2), 70, F(1), 0)]            # in place already
    assert attach_graces(notes, ["4/4", "4/4"]) == 3
    g = [(n.bar, n.pos, n.grace, n.total) for n in notes if n.pitch in (62, 66, 50, 69)]
    assert g == [(0, F(15, 8), True, F(0)), (1, F(0), True, F(0)), (1, F(0), False, F(GRACE_AS_NOTE)),
                 (1, F(2), True, F(0))], g
    lone = [Placed(0, F(3), 72, F(0), 0, grace=True)]   # nothing after it on its staff: a 16th
    assert attach_graces(lone, ["3/4"]) == 1 and not lone[0].grace and lone[0].total == F(1, 4)
    assert attach_graces([Placed(0, F(0), 60, F(1), 0)], ["4/4"]) == 0


def test_export_last_resort():
    """A score some bars of which cannot be exported at all: those bars are emptied, the rest is
    written, and the report says which; a score that exports needs none of it."""
    import src.preprocessing.detokenize as dt
    good = render(Window(bars=[("4/4", 0)] * 3), [Placed(b, Fraction(k), 60 + b, Fraction(1), 0)
                                                   for b in range(3) for k in range(4)])
    real_write = dt.write_musicxml

    def picky(score, path, decorate=None):  # an exporter that cannot write pitch 61 (bar 1)
        if any(getattr(n, "pitch", None) is not None and n.pitch.midi == 61 for n in score.recurse().notes):
            raise ValueError("cannot export pitch 61")
        return real_write(score, path, decorate)
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
            pm = pretty_midi.PrettyMIDI(str(res.files["score_midi"]))   # first note to last, as playback_qpm
            heard_notes = [n for i in pm.instruments for n in i.notes]
            length = max(n.end for n in heard_notes) - min(n.start for n in heard_notes)
            perf = float(notes[:, 2].max() - notes[:, 1].min())
            if st["playback_tempo"] == "grid":   # each bar in the time the performance took for it
                f = st["score_follow"]
                assert np.all(np.diff(f["t"]) > 0) and np.all(np.diff(f["q"]) > 0), f
                assert len(pm.get_tempo_changes()[1]) >= 1
                # up to the last bar line, the score MIDI takes about the performance's time (the
                # grid covers it), give or take the first and the last bars' empty parts
                span = np.interp(max(f["q"][-2], 0.0), f["q"], f["t"])
                assert 0.7 * perf < span < 1.3 * perf + 4.0, (span, perf)
            else:
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
    runs on the system it names (v2 without the seq2seq context mode, with the page's deletion
    setting: only when sure by default), an unknown system or setting is refused."""
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
        for data in ({"system": "v2", "mode": "guarded"}, {"system": "seq2seq", "mode": "guarded"},
                     {"system": "v2", "delete": "off"}, {"system": "v2", "delete": "trained"}):
            jid = c.post("/api/jobs", files={"file": ("x.mid", b"MThd")}, data=data).json()["id"]
            for _ in range(100):
                if c.get(f"/api/jobs/{jid}").json()["status"] in ("done", "error"):
                    break
                import time
                time.sleep(0.05)
        assert calls == [("v2", {"delete_above": 0.95}), ("seq2seq", {"mode": "guarded"}), ("v2", {"delete_above": 1.0}),
                         ("v2", {"delete_above": 0.5})], calls
        assert c.post("/api/jobs", files={"file": ("x.mid", b"MThd")}, data={"system": "v9"}).status_code == 400
        assert c.post("/api/jobs", files={"file": ("x.mid", b"MThd")}, data={"delete": "all"}).status_code == 400


def test_delete_threshold():
    """A note is left out only above the deletion threshold: the head's own decision at 0.5, fewer
    above, none at 1; another threshold is its own system in the whole-piece comparison."""
    from scripts.system_scores import system_name
    from src.labeller import keep_mask
    from src.pipeline import DELETE_ABOVE
    logits = torch.tensor([-3.0, -1.0, 0.5, 2.0])            # p(delete) 0.95, 0.73, 0.38, 0.12
    assert keep_mask(logits).tolist() == (logits > 0).tolist() == [False, False, True, True]
    assert keep_mask(logits, 0.8).tolist() == [False, True, True, True]
    assert keep_mask(logits, 0.96).tolist() == [True] * 4 and keep_mask(logits, 1.0).tolist() == [True] * 4
    assert DELETE_ABOVE == 0.5
    assert system_name(None, "meter", "none", "nl+m") == "v2_nl_m"
    assert system_name(None, "meter", "none", "nl+m", delete_above=0.8) == "v2_nl_m_d80"
    assert system_name(None, "meter", "none", "nl+m", "audio", delete_above=1.0) == "v2_nl_m_keepall_ab"


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
                               cwd=tmp, env=batch_env(SUBMITS=str(tmp / "jobs.txt"), BD=str(bd), **(env or {})),
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
            # the second job resumes the first, and also only runs if the tests passed
            assert f"--dependency=afterok:{tests[0]}" in a[3] and f"--dependency=afterok:{tests[0]},afterany:{a[0]}" in b[3]
            assert all(f"bash pace/v2_jobs.sh meters {runs}" in x[3] and "--gres=gpu:1" in x[3] for x in (a, b))
            assert "--kill-on-invalid-dep=yes" in a[3] and "--kill-on-invalid-dep=yes" in b[3]
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
                               env=batch_env(SUBMITS=str(tmp / "jobs.txt"), BD=str(tmp / "nobd"), **(env or {})),
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


def test_level_batch():
    """pace/v2_level.sh: the tests first; the two tempo-head runs after them (cancelled if they
    fail) on one GPU job; the level experiments, the labeller on their grids and the whole-piece
    systems after the training (finished or not); the scoring after those and after AFTER; 4 jobs;
    refused without meter_v3 and its tempo-prior decoder, or without the system comparison."""
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_level.sh", tmp / "pace")
        _stub_bin(tmp)
        bd = tmp / "bd"
        bd.mkdir()

        def run(after=""):
            if (tmp / "jobs.txt").exists():
                (tmp / "jobs.txt").unlink()
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_level.sh'], cwd=tmp,
                               env=batch_env(SUBMITS=str(tmp / "jobs.txt"), BD=str(bd), AFTER=after),
                               capture_output=True, text=True)
            rows = [ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()] if (tmp / "jobs.txt").exists() else []
            return r, rows
        r, rows = run()
        assert r.returncode == 1 and not rows and "v2_round4.sh" in r.stdout
        for f in ("meter_v3/best.pt", "meter_v3_tp/decode.json"):
            (tmp / "outputs" / "runs" / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / "outputs" / "runs" / f).write_text("x")
        r, rows = run()
        assert r.returncode == 1 and not rows and "bd_setup.sh" in r.stdout
        (tmp / "outputs" / "system_eval").mkdir(parents=True)
        (tmp / "outputs" / "system_eval" / "manifest.json").write_text("[]")
        (bd / "MIDI2ScoreTF.ckpt").write_text("x")
        r, rows = run(after="999")
        assert r.returncode == 0 and len(rows) == 4, (r.stdout + r.stderr, rows)
        by = {row[3].split("-J ")[1].split()[0]: row for row in rows}
        tests, train, sys_, fin = by["lvtests"], by["lvtrain"], by["lvsys"], by["lvfinal"]
        assert "--dependency" not in tests[3] and "tests.test_meter" in tests[3]
        assert f"--dependency=afterok:{tests[0]}" in train[3] and "--kill-on-invalid-dep=yes" in train[3]
        assert "v2_jobs.sh meters meter_v3_th meter_v3_ft" in train[3] and "--constraint=cuda13" in train[3]
        assert f"--dependency=afterany:{train[0]}" in sys_[3]
        assert sys_[3].index("levelexp meter_v3_tp") < sys_[3].index("evals nl_base_gn") < sys_[3].index(
            "systems nl_base_gn+meter_v3_th:tidy,d95 nl_base_gn+meter_v3_ft:tidy,d95")
        assert f"--dependency=afterany:{sys_[0]}:999" in fin[3] and "--gres" not in fin[3]
        assert fin[3].index("compare_runs.py") < fin[3].index("beyer_dai.py score") < fin[3].index("compare_systems.py")
        r, rows = run()
        fin = next(row for row in rows if "-J lvfinal" in row[3])
        assert f"--dependency=afterany:{rows[2][0]} " in fin[3] + " ", fin[3]     # no AFTER: the systems only


def test_round4_batch():
    """pace/v2_round4.sh: the tests first; the new labellers and the tempo-prior decoder after them
    (cancelled if they fail); the evaluations and the whole-piece systems with the new models after
    both; the round-3 system's variants and the ablation after the tests; the scoring after
    everything; 8 jobs; refused without the system comparison set up."""
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_round4.sh", tmp / "pace")
        _stub_bin(tmp)
        bd = tmp / "bd"
        bd.mkdir()

        def run():
            for f in (tmp / "jobs.txt", tmp / "jobs.txt.n"):
                if f.exists():
                    f.unlink()
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_round4.sh'], cwd=tmp,
                               env=batch_env(SUBMITS=str(tmp / "jobs.txt"), BD=str(bd)), capture_output=True, text=True)
            rows = [ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()] if (tmp / "jobs.txt").exists() else []
            return r, rows
        r, rows = run()
        assert r.returncode == 1 and not rows and "bd_setup.sh" in r.stdout
        (tmp / "outputs" / "system_eval").mkdir(parents=True)
        (tmp / "outputs" / "system_eval" / "manifest.json").write_text("[]")
        (bd / "MIDI2ScoreTF.ckpt").write_text("x")
        r, rows = run()
        assert r.returncode == 0 and len(rows) == 8, (r.stdout + r.stderr, len(rows))
        by = {row[3].split("-J ")[1].split()[0]: row for row in rows}
        tests = by["r4tests"][0]
        assert "--dependency" not in by["r4tests"][3] and "tests.test_v2_system" in by["r4tests"][3]
        lab, tempo = by["r4lab"], by["r4tempo"]
        for job in (lab, tempo, by["r4sys_c"]):
            assert f"--dependency=afterok:{tests}" in job[3] and "--kill-on-invalid-dep=yes" in job[3], job[3]
        assert "v2_jobs.sh labellers nl_base_gn_f1 nl_base_gn1_f1" in lab[3]
        assert "v2_jobs.sh tempo meter_v3 meter_v3_tp" in tempo[3] and "--gres" not in tempo[3]
        for name in ("r4evals", "r4sys_a", "r4sys_b"):
            assert f"--dependency=afterany:{lab[0]}:{tempo[0]}" in by[name][3], by[name][3]
        assert "v2_jobs.sh evals nl_base_gn_f1 nl_base_gn1_f1 nl_base_gn" in by["r4evals"][3]
        assert "nl_base_gn_f1+meter_v3_tp:tidy,d95" in by["r4sys_a"][3] and "nl_base_gn1_f1+meter_v3_tp" in by["r4sys_b"][3]
        assert "nl_base_gn+meter_v3:d95,missing" in by["r4sys_c"][3] and "v2_jobs.sh ablation nl_base_gn+meter_v3" in by["r4sys_c"][3]
        fin = by["r4final"]
        deps = set(fin[3].split("--dependency=afterany:")[1].split()[0].split(":"))
        assert deps == {row[0] for row in rows if row is not fin}, deps
        assert "compare_runs.py" in fin[3] and fin[3].count("beyer_dai.py score") == 2


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
  scripts/train_meter.py|scripts/train_labeller.py|scripts/train_model.py)
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
                               cwd=tmp, env=batch_env(CALLS=str(calls), DONE=done, SLURM_JOB_ID="77"),
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
                         "scripts/eval_meter.py --run meter_v3_large --tune",   # meter_v3: done and evaluated
                         "scripts/publish_model.py --run outputs/runs/meter_v3_large"], calls   # evaluated now: published
        code, calls, out = run("evals nl_base nl_large")
        for nl in ("nl_base", "nl_large"):
            mine = [c for c in calls if f"--run {nl} " in c + " "]
            assert mine[0] == f"scripts/eval_labeller.py --run {nl} --normal-priority", mine   # the true grid
            assert [c.split("--grid meter:")[1].split()[0] for c in mine[1:]] == [
                "meter_oracle", "meter_v2_nonote", "meter_v3", "meter_v3_s1", "meter_v3_s2", "meter_v3_large"], mine
        code, calls, out = run("evals nl_base_gn", done="nl_base_gn")
        assert calls[0].startswith("scripts/train_labeller.py --name nl_base_gn --size base --grid-noise 0.8")
        assert calls[1] == "scripts/eval_labeller.py --run nl_base_gn --normal-priority"
        assert calls[2] == "scripts/publish_model.py --run outputs/runs/nl_base_gn"   # trained here: published
        assert [c.split("--grid meter:")[1].split()[0] for c in calls[3:]][:2] == ["meter_oracle", "meter_v2_nonote"]
        for r in ("sf_base_reg_cab", "s2s_m_cab"):   # seq2seq runs from before status.json: finished
            (tmp / "outputs" / "runs" / r).mkdir(parents=True)
            (tmp / "outputs" / "runs" / r / "best.pt").write_text("x")
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
        code, calls, out = run("systems nl_base_gn+meter_v3_s1 never_trained:meter")   # not trained here: refused
        assert calls == [] and code == 1 and out.count("not finished") == 2, (calls, out)
        code, calls, out = run("systems nl_base_gn+meter_v3_s1")
        assert calls == [] and code == 1 and "not finished" in out, (calls, out)
        # a seq2seq model trained (resuming) then evaluated; not finished: the job fails, no evaluation
        code, calls, out = run("seq2seq sq --size base --beats")
        assert calls == ["scripts/train_model.py --name sq --normal-priority --workers 7 --size base --beats"] and code == 1
        code, calls, out = run("seq2seq sq --size base --beats", done="sq")
        assert calls[0] == "scripts/train_model.py --name sq --normal-priority --workers 7 --resume --size base --beats"
        assert calls[1].startswith("scripts/evaluate_pieces.py --run sq --normal-priority --musicxml 5 --only-missing")
        assert "--modes oracle,none,guarded,meter --conditions clean,synthetic,real,periscope" in calls[1] and code == 0
        code, calls, out = run("seq2seq sq --size base --beats")   # trained already: only the evaluation
        assert [c.split()[0] for c in calls] == ["scripts/evaluate_pieces.py"], calls
        # the ablation: real-audio beats on clean MIDI, next to the rendered-MIDI version
        (tmp / "outputs" / "system_eval" / "v2_nl_base_gn_meter_v3").mkdir(parents=True)
        (tmp / "outputs" / "system_eval" / "v2_nl_base_gn_meter_v3" / "summary.json").write_text("{}")
        (tmp / "outputs" / "system_eval_ablation").mkdir(parents=True)
        code, calls, out = run("ablation nl_base_gn+meter_v3")
        assert calls == ["scripts/system_scores.py --v2 nl_base_gn+meter_v3 --conditions clean --clean-beats audio "
                         "--out outputs/system_eval_ablation"] and code == 0, (calls, out)
        assert (tmp / "outputs" / "system_eval_ablation" / "v2_nl_base_gn_meter_v3" / "summary.json").exists()
        # v2 options after a colon: their system_scores.py flags; an unknown one fails that system only
        code, calls, out = run("systems nl_base_gn+meter_v3:tidy,d95,missing nl_base_gn+meter_v3:keepall "
                               "nl_base_gn+meter_v3:bogus")
        assert calls == ["scripts/system_scores.py --v2 nl_base_gn+meter_v3 --tidy --delete-above 0.95 --only-missing",
                         "scripts/system_scores.py --v2 nl_base_gn+meter_v3 --delete-above 1"], calls
        assert code == 1 and "unknown v2 option bogus" in out, out
        # the tempo prior: a new run sharing meter_v3's checkpoint, its decoder tuned again
        code, calls, out = run("tempo meter_v3 meter_v3_tp")
        assert calls == ["scripts/eval_meter.py --run meter_v3_tp --tune --tempo-prior --device cpu"] and code == 0
        tp = tmp / "outputs" / "runs" / "meter_v3_tp"
        assert all((tp / f).exists() for f in ("best.pt", "status.json", "eval_meter.json")), list(tp.iterdir())
        code, calls, out = run("systems nl_base_gn+meter_v3_tp")   # finished and evaluated: a v2 system
        assert calls == ["scripts/system_scores.py --v2 nl_base_gn+meter_v3_tp"] and code == 0, (calls, out)
        code, calls, out = run("tempo meter_v9 meter_v9_tp")      # nothing to tune
        assert calls == [] and code == 1 and "not finished" in out
        # the tempo head: started from meter_v3, trained side by side, each tuned its own way
        code, calls, out = run("meters meter_v3_th meter_v3_ft", done="meter_v3_th meter_v3_ft")
        assert code == 0, out
        th = next(c for c in calls if "--name meter_v3_th" in c)
        ft = next(c for c in calls if "--name meter_v3_ft" in c)
        assert "--init outputs/runs/meter_v3/best.pt --tempo-head --freeze-trunk --select tempo" in th, th
        assert "--tempo-head --select both" in ft and "--freeze-trunk" not in ft, ft
        assert set(c for c in calls if c.startswith("scripts/eval_meter.py")) == {
            "scripts/eval_meter.py --run meter_v3_th --level --decode-from outputs/runs/meter_v3_tp/decode.json",
            "scripts/eval_meter.py --run meter_v3_ft --tune --tempo-prior --level"}, calls
        code, calls, out = run("levelexp meter_v3_tp")              # decode.json: not in the stand-in run yet
        assert code == 1 and calls == [] and "decode.json" in out
        (tp / "decode.json").write_text("{}")
        code, calls, out = run("levelexp meter_v3_tp")
        assert code == 0 and calls == ["scripts/level_experiments.py --meter meter_v3_tp"], calls
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
