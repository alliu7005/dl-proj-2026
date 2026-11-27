"""Tests for glissandos (src/glissando.py and its use in src/pipeline.py, the export, the scripts).

    python -m tests.test_glissando        # or: pytest tests/test_glissando.py

Covers: the detector (a white-key sweep among chords, a black-key one, one split by a missed note,
one with a note Transkun timed out of order, a rolled chord, a fast written scale, a short run: the
last three not glissandos); the two ends found among the placed notes (by the performed note each
came from, else by pitch and place), put where the grid says to the sixteenth, the first written to
reach the last, a left-out end written back on a staff it reads on, the end on the start's staff;
the "gliss." line in the MusicXML, an octave sign for an end far off its staff (the sounding pitch
kept), the swept notes in the score MIDI; V2Pipeline end to end with tiny models: every bar but
the glissando's written exactly as with glissandos off; the scripts' switches.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.glissando as G  # noqa: E402

V = ROOT / "data" / "processed" / "vocab.json"


def sweep(start: int, n: int, t0: float, rate: float, direction: int = -1, keys: str = "white", skip=()):
    """n notes from ``start`` over the keys of one colour, ``rate`` a second (``skip``: missed ones)."""
    ks = G.KEYS[keys]
    k0 = ks.index(start)
    out = []
    for i in range(n):
        if i in skip:
            continue
        out.append([ks[k0 + direction * i], t0 + i / rate, t0 + i / rate + 0.05, 70])
    return out


def chords(t0: float, t1: float, step: float = 0.5):
    """C major / F major chords on the beats, both hands."""
    out = []
    for k, t in enumerate(np.arange(t0, t1, step)):
        for p in ((48, 60, 64, 67) if k % 2 == 0 else (41, 57, 60, 65)):
            out.append([p, t, t + 0.4, 80])
    return out


def test_detector():
    g1 = sweep(89, 18, 2.03, 68)                                   # F6 down to A3, white keys (the user's)
    g2 = sweep(73, 12, 6.01, 50, direction=+1, keys="black")      # a black-key sweep up
    g3 = sweep(105, 20, 9.02, 83, skip=(7,))                      # down, one note missed in the middle
    late = [[G.KEYS["white"][G.KEYS["white"].index(105) - 7], 9.02 + 8.5 / 83, 9.2, 60]]   # the missed one, late
    rolled = [[p, 12.0 + i * 0.004, 12.5, 70] for i, p in enumerate((48, 52, 55, 58, 60, 64, 67))]
    scale = sweep(60, 16, 13.0, 20, direction=+1)                  # a written scale: 32nds at 150 qpm
    short = sweep(84, 6, 15.0, 60)                                 # too few notes
    notes = np.array(chords(0, 17) + g1 + g2 + g3 + late + rolled + scale + short, np.float32)
    notes = notes[np.random.default_rng(0).permutation(len(notes))]   # any order
    found = G.find_glissandi(notes)
    got =[(g.pitches[0], g.pitches[-1], g.keys, g.direction, len(g.idx)) for g in found]
    blk = G.KEYS["black"]
    wht = G.KEYS["white"]
    assert got == [(89, wht[wht.index(89) - 17], "white", -1, 18), (73, blk[blk.index(73) + 11], "black", 1, 12),
                   (105, G.KEYS["white"][G.KEYS["white"].index(105) - 19], "white", -1, 20)], got
    assert all(sorted(notes[g.idx, 1].tolist()) == notes[g.idx, 1].tolist() for g in found)   # in time order
    assert abs(found[0].rate - 68) < 1 and found[0].summary()["direction"] == "down"
    keep = G.inner_mask(len(notes), found)
    assert (~keep).sum() == 16 + 10 + 18 and keep[found[0].idx[[0, -1]]].all()
    assert G.find_glissandi(np.array(chords(0, 5), np.float32)) == []
    assert G.find_glissandi(np.zeros((0, 4), np.float32)) == []


def test_attach_and_line():
    """The ends among the placed notes; a left-out end written; the line and the swept notes."""
    from src.notelabels import Placed
    from src.preprocessing.detokenize import tokens_to_musicxml, tokens_to_score
    from src.notelabels import Window, render
    bars = ["4/4"] * 4
    grid = {"t": np.arange(0, 8.5, 0.5), "q": np.arange(0, 17.0), "bar_q": np.arange(0, 16, 4.0), "bars": bars}
    g = G.Glissando(np.arange(18), -1, "white", 5.5, 5.75, [int(p[0]) for p in sweep(89, 17, 5.5, 68)] + [57])
    a = Placed(2, Fraction(3), 89, Fraction(1, 2), 0)          # beat 4 of bar 3 (q = 11 = 5.5 s)
    b = Placed(2, Fraction(7, 2), 57, Fraction(1, 4), 1)       # q = 11.5 (5.75 s), the other staff
    other = Placed(2, Fraction(0), 89, Fraction(1), 0)         # the same pitch elsewhere: not it
    placed = [other, a, b]
    out = G.attach([g], placed, grid, bars)
    assert len(out) == 1 and out[0]["start"] == (2, Fraction(3), 89, 0) and out[0]["end"][:3] == (2, Fraction(7, 2), 57)
    assert a.total == Fraction(1, 2) and out[0]["written"] == 0 and len(out[0]["inner"]) == 16
    # one hand's glissando: its end (A3, two ledger lines under the treble) written on the start's staff
    assert out[0]["end_moved"] and b.staff == 0 and out[0]["end"][3] == 0
    assert G.fits(57, 0) and not G.fits(55, 0) and G.fits(64, 1) and not G.fits(65, 1)
    wide = G.Glissando(np.arange(30), -1, "white", 5.5, 5.75, [89] + [80] * 28 + [40])
    c, d = Placed(2, Fraction(3), 89, Fraction(1, 2), 0), Placed(2, Fraction(7, 2), 40, Fraction(1, 4), 1)
    w = G.attach([wide], [c, d], grid, bars)
    assert not w[0]["end_moved"] and d.staff == 1                 # a sweep to E2: the line crosses the staves
    # the first note left out by the labeller: written back at the grid's sixteenth
    placed2 = [Placed(2, Fraction(7, 2), 57, Fraction(1, 4), 1)]
    out2 = G.attach([g], placed2, grid, bars)
    # (on the treble: F6 does not read on the bass, where the other end is; that end then joins it)
    assert out2[0]["written"] == 1 and len(placed2) == 2 and (placed2[1].bar, placed2[1].pos, placed2[1].staff) == (2, 3, 0)
    assert placed2[0].staff == 0 and out2[0]["end_moved"]
    # by the performed note each came from (V2Pipeline's placed notes carry it): exact, even when the
    # labeller put it elsewhere (at 4/7 of a beat, reading the run as a tuplet), then snapped to the grid
    s_a = Placed(2, Fraction(3), 89, Fraction(1, 2), 0, src=0)
    s_b = Placed(2, Fraction(26, 7), 57, Fraction(1, 16), 0, src=17)
    decoy = Placed(2, Fraction(7, 2), 57, Fraction(1, 4), 0, src=40)        # the same pitch where the grid says
    o3 = G.attach([g], [s_a, decoy, s_b], grid, bars)
    assert o3[0]["end"][:3] == (2, Fraction(7, 2), 57) and s_b.pos == Fraction(7, 2) and decoy.pos == Fraction(7, 2)
    assert s_b.total == Fraction(1, 4)                                        # a sixteenth at least
    gone = G.attach([g], [Placed(2, Fraction(7, 2), 57, Fraction(1, 4), 0, src=17),
                          Placed(2, Fraction(3), 89, Fraction(1, 2), 0, src=99)], grid, bars)
    assert gone[0]["written"] == 1                                            # src 0 not placed: written back
    # both ends on one sixteenth: no line
    assert G.attach([G.Glissando(np.arange(9), -1, "white", 5.5, 5.52, list(range(89, 80, -1)))],
                    [Placed(2, Fraction(3), 89, Fraction(1, 4), 0), Placed(2, Fraction(3), 81, Fraction(1, 4), 0)],
                    grid, bars) == []
    # the export: the tokens of these notes, the line, the swept notes in the MIDI only
    tokens = render(Window(bars=[(ts, 0) for ts in bars]), placed)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "s.musicxml"
        report = {}

        def deco(score):
            report["drawn"] = G.decorate(score, out)
        tokens_to_musicxml(tokens, str(path), decorate=deco)
        x = path.read_text(encoding="utf-8")
        assert report["drawn"] == 1
        assert re.findall(r'<glissando line-type="wavy" number="1" type="start">gliss\.</glissando>', x)
        assert re.findall(r'<glissando line-type="wavy" number="1" type="stop" ?/>', x)
        import music21 as m21
        back = m21.converter.parse(str(path))
        line = [sp for sp in back.spannerBundle if isinstance(sp, m21.spanner.Glissando)]
        assert len(line) == 1 and [n.pitch.midi for n in line[0].getSpannedElements()] == [89, 57]
        # an end far above the treble (A7: six ledger lines) under 15ma, an end at G6 under 8va
        assert (G.octave_sign(105, 0), G.octave_sign(91, 0), G.octave_sign(89, 0)) == ("15ma", "8va", None)
        assert (G.octave_sign(21, 1), G.octave_sign(33, 1), G.octave_sign(40, 1)) == ("15mb", "8vb", None)
        hi = G.Glissando(np.arange(20), -1, "white", 5.5, 5.75, [105] + [90] * 18 + [69])
        e1, e2 = Placed(2, Fraction(3), 105, Fraction(1, 2), 0), Placed(2, Fraction(7, 2), 69, Fraction(1, 2), 0)
        ohi = G.attach([hi], [e1, e2], grid, bars)
        path2 = Path(t) / "hi.musicxml"
        tokens_to_musicxml(render(Window(bars=[(ts, 0) for ts in bars]), [e1, e2]), str(path2),
                           decorate=lambda sc_: G.decorate(sc_, ohi))
        x2 = path2.read_text(encoding="utf-8")
        assert '<octave-shift number="1" size="15" type="down"' in x2 and "<glissando" in x2
        back2 = m21.converter.parse(str(path2))
        assert sorted(n.pitch.midi for n in back2.recurse().notes) == [69, 105]     # it still sounds A7
        sc = tokens_to_score(tokens)
        n0 = len(list(sc.flatten().notes))
        assert G.decorate(sc, out, playback=True) == 1
        sc.write("midi", fp=str(Path(t) / "s.mid"))
        from src.preprocessing.performance import load_performance
        notes, _ = load_performance(Path(t) / "s.mid")
        swept = notes[(notes[:, 0] < 89) & (notes[:, 0] > 57)]
        assert len(swept) == 16 and np.all(np.diff(swept[np.argsort(swept[:, 1]), 0]) < 0)   # going down
        assert len(list(sc.flatten().notes)) == n0 + 16


def test_pipeline():
    """V2Pipeline with tiny untrained models on a performance with a glissando: found, the middle
    not labelled, the line in the MusicXML, the swept notes in the score MIDI; off: none of it."""
    if not V.exists():
        print("  (skipped: no data/processed/vocab.json)")
        return
    from scripts.system_inputs import write_midi
    from src.pipeline import V2Pipeline
    from src.preprocessing.performance import load_performance
    from tests.test_v2_system import make_v2
    import torch
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        lab, met = make_v2(tmp)
        notes = np.array(chords(0.5, 12) + sweep(89, 18, 5.51, 68), np.float32)
        notes = notes[np.lexsort((notes[:, 0], notes[:, 1]))]
        midi = tmp / "perf.mid"
        write_midi(notes[:, [0, 1, 2, 3]], np.zeros((0, 2), np.float32), midi)
        n_in = len(load_performance(midi)[0])
        times = np.arange(0.5, 12.5, 0.5)
        beats = (times, np.arange(len(times)) % 4 == 0)
        pipe = V2Pipeline(lab, met, vocab=V, device="cpu", window_notes=48, delete_above=1.0, glissando=True)
        res = pipe.run(midi, tmp / "on", beats=beats)
        st = res.stats
        assert st["glissando"] and st["glissandi_found"] == 1, st
        assert st["notes_kept"] + st["notes_deleted"] == n_in - 16, st          # the middle is not labelled
        assert len(st["glissandi"]) == 1 and st["glissandi"][0]["from"] == 89 and st["glissandi"][0]["bar"] >= 1
        x = (tmp / "on" / "score.musicxml").read_text(encoding="utf-8")
        assert x.count("<glissando") == 2 and st["export"]["glissandi_drawn"] == 1
        json.loads((tmp / "on" / "result.json").read_text())                    # its stats save as JSON
        heard, _ = load_performance(tmp / "on" / "score.mid")
        off = pipe.run(midi, tmp / "off", beats=beats, glissando=False)
        assert off.stats["glissandi_found"] == 0 and "<glissando" not in (tmp / "off" / "score.musicxml").read_text()
        assert off.stats["notes_kept"] + off.stats["notes_deleted"] == n_in
        assert len(heard) >= st["notes_kept"] + 16 - 1                          # the sweep plays

        # the labeller gets every note either way (an earlier version left the run out of its input,
        # which moved the windows and rewrote 46 of the 152 bars of the user's song): the same input,
        # the same labels; afterwards only the run's notes are taken out (and a grace note that had
        # one as its next note joins another), so nothing away from the glissando changes
        import src.labeller as LB
        seen, real = [], LB.predict_placed

        def spy(model, items, *a, **k):
            out = real(model, items, *a, **k)
            seen.append(([it["notes"] for it in items],
                         [[(p.bar, p.pos, p.pitch, p.total, p.staff, p.grace, p.src) for p in pl] for pl in out]))
            return out
        LB.predict_placed = spy
        try:
            pipe.run(midi, tmp / "on2", beats=beats)
            n_on = len(seen)    # a run labels twice when notes land on a stray third staff (folded): compare like with like
            pipe.run(midi, tmp / "off2", beats=beats, glissando=False)
        finally:
            LB.predict_placed = real
        assert len(seen) == 2 * n_on
        (in_on, lab_on), (in_off, lab_off) = seen[0], seen[n_on]
        assert len(in_on) == len(in_off) and all(torch.equal(x, y) for x, y in zip(in_on, in_off))
        assert lab_on == lab_off
        assert not V2Pipeline(lab, met, vocab=V, device="cpu").glissando        # off unless asked


def test_switches():
    """The demo and the command-line tool have it on (--no-glissando), the scoring script off
    (--glissando: system _gl), the batch jobs pass gl."""
    from scripts.system_scores import system_name
    from src.release import GLISSANDO
    assert GLISSANDO is True
    assert system_name(None, "meter", "none", "nl+m", tidy=True, delete_above=0.95, glissando=True) == "v2_nl_m_d95_tidy_gl"
    assert system_name(None, "meter", "none", "nl+m", tidy=True, delete_above=0.95) == "v2_nl_m_d95_tidy"
    for f in ("demo/app.py", "scripts/audio_to_score.py"):
        assert "--no-glissando" in (ROOT / f).read_text(encoding="utf-8"), f
    jobs = (ROOT / "pace" / "v2_jobs.sh").read_text()
    assert 'gl) out="$out --glissando" ;;' in jobs
    page = (ROOT / "demo" / "static" / "index.html").read_text(encoding="utf-8")
    assert "function forViewer" in page and "forViewer(await (await fetch(url)).text())" in page


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
