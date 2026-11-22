"""Tests for how scores become training data (src/preprocessing).

    python -m tests.test_preprocessing        # or: pytest tests/test_preprocessing.py

Covers the fixes of 2026-11-22: a chord's newly struck notes are not marked tied because another
note of the chord is; a pickup is written where it falls in its bar (starting mid-bar, no bar line
before it); every version of a piece (ASAP's _repeat / _no_repeat / _no_2_repeat ...) gets one
composition id, so versions cannot land in different splits; a beat time map stays increasing
through backward steps; src/eval/plan.py leaves v2 runs (labeller, meter) to their own evaluation.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import music21 as m21
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing.performance import TimeMap  # noqa: E402
from src.preprocessing.score import load_score, tokenize_bars  # noqa: E402
from src.preprocessing.split import composition_id  # noqa: E402


def score_of(measures, ts="4/4"):
    """A one-part score from lists of (offset, element) per measure."""
    s, p = m21.stream.Score(), m21.stream.Part()
    for k, content in enumerate(measures):
        m = m21.stream.Measure(number=k + 1)
        if k == 0:
            m.timeSignature = m21.meter.TimeSignature(ts)
        for off, el in content:
            m.insert(off, el)
        p.append(m)
    s.append(p)
    return s


def test_chord_ties():
    """C4+E4 with only C4 tied into C4+G4: the second chord's C4 is a tie continuation, its G4 a
    new note."""
    c1 = m21.chord.Chord(["C4", "E4"], quarterLength=4)
    c1.notes[0].tie = m21.tie.Tie("start")
    c2 = m21.chord.Chord(["C4", "G4"], quarterLength=4)
    c2.notes[0].tie = m21.tie.Tie("stop")
    toks = tokenize_bars(load_score(score_of([[(0, c1)], [(0, c2)]])), [[0], [1]])
    second = toks[toks.index("bar", 1):]
    assert second == ["bar", "pos_0", "p_60", "d_4", "tie", "p_67", "d_4"], second
    n = m21.note.Note("D4", quarterLength=4)   # a single note's tie still read
    n2 = m21.note.Note("D4", quarterLength=4)
    n.tie, n2.tie = m21.tie.Tie("start"), m21.tie.Tie("stop")
    toks = tokenize_bars(load_score(score_of([[(0, n)], [(0, n2)]])), [[0], [1]])
    assert toks[-1] == "tie" and toks.count("tie") == 1, toks


def test_pickup_positions():
    """A one-beat pickup in 4/4 is written at pos_3 of a bar it starts in the middle of (no bar
    line before it); a full first measure is unchanged; a cut window inside the pickup keeps the
    measure's own offsets for cutting."""
    pick = score_of([[(0, m21.note.Note("G4", quarterLength=1))], [(0, m21.note.Note("C5", quarterLength=4))]])
    ms = load_score(pick)
    assert ms[0].length == 1
    toks = tokenize_bars(ms, [[0], [1]])
    assert toks[:6] == ["ts_4/4", "key_0", "pos_3", "staff_0", "p_67", "d_1"], toks
    assert toks[6:9] == ["bar", "pos_0", "p_72"], toks
    full = score_of([[(0, m21.note.Note("G4", quarterLength=4))]])
    assert tokenize_bars(load_score(full), [[0]])[:4] == ["bar", "ts_4/4", "key_0", "pos_0"]
    two = score_of([[(0, m21.note.Note("E4", quarterLength=0.5)), (0.5, m21.note.Note("G4", quarterLength=0.5))],
                    [(0, m21.note.Note("C5", quarterLength=4))]])
    cut = tokenize_bars(load_score(two), [[0]], start_pos=0.5)   # from the second pickup note on
    assert "p_64" not in cut and "pos_3" in cut and "sub_1/2" in cut, cut


def test_composition_ids():
    """Every version of a piece is one composition; other names are left alone."""
    base = composition_id("asap", "Liszt/Gran_Etudes_de_Paganini/6_Theme_and_Variations")
    for v in ("_repeat", "_no_repeat", "_no_repeats", "_no_2_repeat", "_no_first_repeat"):
        assert composition_id("asap", "Liszt/Gran_Etudes_de_Paganini/6_Theme_and_Variations" + v) == base, v
    assert composition_id("asap", "Chopin/Etudes_op_25/repeated_notes") == "chopin/etudes_op_25/repeated_notes"
    assert composition_id("asap", "Schumann/Toccata_repeat") == "schumann/toccata"


def test_time_map_monotone():
    """Two backward beats in a row are dropped; the map is increasing and interpolates the rest."""
    m = TimeMap([0, 1, 0.8, 0.9, 2, 3], [0, 10, 8, 9, 20, 30])
    assert np.all(np.diff(m.src) > 0) and list(m.src) == [0, 1, 2, 3]
    assert np.allclose(m(np.array([0.5, 1.5, 2.5])), [5, 15, 25])


def test_plan_skips_v2_runs():
    """src/eval/plan.py: a labeller or meter run is not a seq2seq model (exit 5), so
    pace/reevaluate.sh does not submit seq2seq evaluations for it."""
    from src.eval import plan
    with tempfile.TemporaryDirectory() as t:
        for arch in ("labeller", "meter", "meter_oracle"):
            d = Path(t) / arch
            d.mkdir()
            (d / "config.json").write_text(json.dumps({"arch": arch}))
            assert plan.main(["--run", arch, "--runs-dir", t]) == plan.NOT_SEQ2SEQ
        d = Path(t) / "s2s"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"arch": "scoreformer"}))
        assert plan.main(["--run", "s2s", "--runs-dir", t, "--conditions", "clean"]) != plan.NOT_SEQ2SEQ


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
