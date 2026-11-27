"""Tests for the score's tempo from the meter model's grid (src/tempo.py, src/pipeline.py engrave).

    python -m tests.test_tempo        # or: pytest tests/test_tempo.py

Covers: each bar's tempo is the time the performance took for it; printed marks only where the
tempo changes and stays changed (a one-bar outlier is no new section), in the time signature's beat
unit; the score MIDI playing each bar at its tempo, a pickup included; the marks in the MusicXML;
the follow map agreeing with the MIDI; one tempo for the whole piece without a grid tempo.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def grid_of(beat_s: list[float], bar: int = 4, ts: str = "4/4") -> dict:
    """A grid with one tracked beat per quarter, the k-th beat ``beat_s[k]`` seconds long."""
    t = np.concatenate([[0.0], np.cumsum(beat_s)])
    q = np.arange(len(t), dtype=np.float64)
    n_bars = (len(t) - 1) // bar
    return {"t": t, "q": q, "bars": [ts] * n_bars, "bar_q": np.arange(n_bars, dtype=np.float64) * bar}


def test_bar_tempi_and_marks():
    from src.tempo import bar_tempi, tempo_marks
    g = grid_of([0.5] * 32 + [0.75] * 32)               # 8 bars at 120, then 8 at 80
    tempi = bar_tempi(g, 0.0, g["bars"])
    assert len(tempi) == 16
    assert np.allclose([b["qpm"] for b in tempi[:8]], 120) and np.allclose([b["qpm"] for b in tempi[8:]], 80)
    assert [b["offset"] for b in tempi[:3]] == [0, 4, 8]
    marks = tempo_marks(tempi, g["bars"])
    assert [(m["bar"], m["unit"], m["value"]) for m in marks] == [(0, 1.0, 120), (8, 1.0, 80)], marks

    # one bar rushed (a tracker's glitch, or a fill) is not a new section; rubato within 15% neither
    rng = np.random.default_rng(0)
    beats = list(0.5 * np.exp(rng.normal(0, 0.04, 64)))
    beats[12:16] = [0.25] * 4
    g = grid_of(beats)
    marks = tempo_marks(bar_tempi(g, 0.0, g["bars"]), g["bars"])
    assert len(marks) == 1 and abs(marks[0]["value"] - 120) <= 6, marks


def test_beat_unit():
    from src.tempo import bar_tempi, beat_unit, tempo_marks
    assert [beat_unit(ts) for ts in ("4/4", "3/4", "6/8", "12/8", "2/2", "3/8", "12/16")] == \
        [1.0, 1.0, 1.5, 1.5, 2.0, 0.5, 0.75]
    g = grid_of([60 / 90] * 24, bar=3, ts="6/8")      # quarter = 90: dotted quarter = 60
    marks = tempo_marks(bar_tempi(g, 0.0, g["bars"]), g["bars"])
    assert [(m["unit"], m["value"]) for m in marks] == [(1.5, 60)], marks


def scale_notes(bars: int, pickup: bool = False) -> list[str]:
    toks = ["ts_4/4", "key_0", "pos_3", "p_59", "d_1"] if pickup else []
    for b in range(bars):
        toks += (["bar"] if (b or pickup) else ["bar", "ts_4/4", "key_0"])
        for k in range(4):
            toks += [f"pos_{k}", f"p_{60 + k}", "d_1"]
    return toks


def midi_onsets(path: Path) -> np.ndarray:
    from src.preprocessing.performance import load_performance
    notes, _ = load_performance(path)
    return np.sort(notes[:, 1])


def test_engrave_plays_grid_tempo():
    import music21 as m21

    from src.pipeline import Result, engrave
    from src.tempo import score_tempo
    g = grid_of([0.5] * 16 + [1.0] * 16)               # bars 1-4 at 120, bars 5-8 at 60
    tempo = score_tempo(g, 0.0, g["bars"], 0.0)
    assert [m["value"] for m in tempo["marks"]] == [120, 60]
    with tempfile.TemporaryDirectory() as t:
        res = Result([])
        rep = engrave(scale_notes(8), Path(t), res, 24.0, tempo=tempo)
        assert not res.errors, res.errors
        assert rep["playback_tempo"] == "grid" and rep["tempo_marks_printed"] == 2, rep
        on = midi_onsets(Path(t) / "score.mid")
        want = np.concatenate([np.arange(16) * 0.5, 8.0 + np.arange(16) * 1.0])   # as the performance played it
        assert np.allclose(on, want, atol=1e-2), on
        sc = m21.converter.parse(str(Path(t) / "score.musicxml"))
        mms = [(mm.getContextByClass(m21.stream.Measure).number, mm.number, mm.referent.quarterLength)
               for mm in sc.recurse().getElementsByClass(m21.tempo.MetronomeMark)]
        assert mms == [(1, 120, 1.0), (5, 60, 1.0)], mms
    # the follow map: the score MIDI's seconds <-> quarters from the first note, as played
    f = tempo["follow"]
    assert np.isclose(np.interp(18.0, f["q"], f["t"]), 10.0) and np.isclose(np.interp(3.0, f["t"], f["q"]), 6.0)


def test_engrave_pickup():
    from src.pipeline import Result, engrave
    from src.tempo import score_tempo
    g = grid_of([1.0] * 4 + [0.5] * 8)                 # the pickup's bar at 60, then 120
    tempo = score_tempo(g, 0.0, ["4/4"] * 3, 3.0)      # the first note on beat 4 of the first bar
    assert [b["offset"] for b in tempo["bars"]] == [0, 4, 8]
    f = tempo["follow"]                                 # counted from the first note: 1 s to the bar line
    assert np.isclose(np.interp(1.0, f["q"], f["t"]), 1.0) and np.isclose(np.interp(3.0, f["q"], f["t"]), 2.0)
    with tempfile.TemporaryDirectory() as t:
        res = Result([])
        engrave(scale_notes(2, pickup=True), Path(t), res, 5.0, tempo=tempo)
        assert not res.errors, res.errors
        on = midi_onsets(Path(t) / "score.mid")
        on = on - on[0]                                 # the page counts from the first note too
        assert np.allclose(on, np.concatenate([[0.0], 1.0 + np.arange(8) * 0.5]), atol=1e-2), on


def test_engrave_constant_without_tempo():
    from src.pipeline import Result, engrave
    with tempfile.TemporaryDirectory() as t:
        res = Result([])
        rep = engrave(scale_notes(4), Path(t), res, 8.0)   # 16 quarters in 8 s: 120 throughout
        assert rep["playback_tempo"] == "constant" and "tempo_marks_printed" not in rep, rep
        assert np.allclose(midi_onsets(Path(t) / "score.mid"), np.arange(16) * 0.5, atol=1e-2)
        assert "<metronome" not in (Path(t) / "score.musicxml").read_text()


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
