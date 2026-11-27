"""Tests for playing grace notes and rolled chords back as written (src/playback.py, src/pipeline.py
engrave).

    python -m tests.test_playback        # or: pytest tests/test_playback.py

music21's MIDI export alone drops a grace note, or (in a full score) strikes it with its main note
and holds it for seconds, and strikes a rolled chord as a block. Covers: grace notes just before
their main note (on the beat at the very start), short; a rolled chord spread upwards and ending
together, across both staves as one roll; a chord with a tied note left alone; no note held past
the written music; the MusicXML unchanged; engrave's counts.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOKENS = ("bar ts_4/4 key_0 pos_0 staff_0 p_62 grace p_60 d_1 "            # a grace note on the first beat
          "pos_1 p_76 grace p_74 grace p_72 d_1 "                           # a run of two before beat 2
          "pos_2 p_64 d_2 arp p_67 d_2 arp staff_1 p_48 d_2 arp "          # rolled on both staves
          "bar pos_0 staff_0 p_60 d_2 arp p_64 d_2 arp pos_2 p_65 d_2 "     # rolled, E tied into the next bar
          "bar pos_0 p_64 d_1 tie p_67 d_3").split()


def midi_of(tokens, qpm, realize_it=True):
    import music21 as m21

    from src.playback import realize
    from src.preprocessing.detokenize import tokens_to_score
    from src.preprocessing.performance import load_performance
    score = tokens_to_score(tokens)
    counts = realize(score, qpm) if realize_it else {}
    score.parts[0].getElementsByClass(m21.stream.Measure).first().insert(0, m21.tempo.MetronomeMark(number=qpm))
    with tempfile.TemporaryDirectory() as t:
        score.write("midi", fp=str(Path(t) / "s.mid"))
        notes, _ = load_performance(Path(t) / "s.mid")
    return notes, counts


def at(notes, pitch, near):
    m = notes[(notes[:, 0] == pitch) & (np.abs(notes[:, 1] - near) < 0.3)]
    return m[np.argsort(m[:, 1])]


def test_graces_and_rolls():
    notes, counts = midi_of(TOKENS, 60)             # 1 quarter = 1 s
    assert counts == {"graces_played": 3, "rolled_chords_played": 1}, counts
    g = at(notes, 62, 0.0)                          # the piece's first beat: no room before, on the beat
    assert len(g) == 1 and abs(g[0, 1]) < 1e-3 and abs(g[0, 2] - g[0, 1] - 0.07) < 1e-3
    a, b = at(notes, 76, 1.0), at(notes, 74, 1.0)   # the run, in its written order, ending on beat 2
    assert abs(a[0, 1] - 0.86) < 1e-3 and abs(b[0, 1] - 0.93) < 1e-3, (a, b)
    assert abs(at(notes, 72, 1.0)[0, 1] - 1.0) < 1e-3                    # the main note stays on its beat
    roll = [at(notes, p, 2.0)[0] for p in (48, 64, 67)]                  # across both staves, from the bass up
    assert np.allclose([r[1] for r in roll], [2.0, 2.035, 2.07], atol=1e-3), roll
    assert np.allclose([r[2] for r in roll], 4.0, atol=1e-3)             # all held to the chord's end
    tied = [at(notes, p, 4.0)[0] for p in (60, 64)]                      # a tied note: struck as written
    assert all(abs(r[1] - 4.0) < 1e-3 for r in tied), tied
    assert notes[:, 2].max() <= 12.0 + 1e-3 and (notes[:, 2] - notes[:, 1]).max() <= 4.0 + 1e-3   # nothing stuck
    plain, _ = midi_of(TOKENS, 60, realize_it=False)
    assert len(at(plain, 76, 1.0)) == 0 or abs(at(plain, 76, 1.0)[0, 1] - 1.0) < 1e-3   # without it: lost or late
    assert len({round(float(r[1]), 3) for p in (48, 64, 67) for r in at(plain, p, 2.0)}) == 1   # a block


def test_engrave():
    """engrave: the counts, the score MIDI with them, the MusicXML still writing them as written."""
    import music21 as m21

    from src.pipeline import Result, engrave
    with tempfile.TemporaryDirectory() as t:
        res = Result([])
        rep = engrave(TOKENS, Path(t), res, 12.0)
        assert not res.errors and rep["graces_played"] == 3 and rep["rolled_chords_played"] == 1, (res.errors, rep)
        sc = m21.converter.parse(str(Path(t) / "score.musicxml"))
        graces = [n for n in sc.parts[0].recurse().notes if n.duration.isGrace]
        # each grace note its own note: the single one a slashed eighth, the run unslashed 16ths in order
        assert [(n.pitch.midi, n.duration.type, n.duration.slash) for n in graces] == [
            (62, "eighth", True), (76, "16th", False), (74, "16th", False)], graces
        assert sum(any(isinstance(e, m21.expressions.ArpeggioMark) for e in n.expressions)
                   for n in sc.recurse().notes) >= 3


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
