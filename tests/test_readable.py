"""Tests for the readability rules (src/readable.py, V2Pipeline readable / fix_jumps, the switches).

    python -m tests.test_readable        # or: pytest tests/test_readable.py

fix_jumps: a step that packs three beats into one beat's time shortens its bar to 2/4 and moves
everything after it back; a long gap at the usual tempo (the tracker missed beats) is no jump; a
bar keeps two beats; a step that leaves its bar is left. readable_lengths: short rests closed,
small overlaps trimmed, not across the bar line, not into an unwritable length or across grids,
grace notes left. stem_voices: stems by voice (three: by height), the other voices' rests hidden.
two_voices (src/preprocessing/detokenize.py): the user's measure 4 in two voices, and random bars that
sound the same in two voices at most. The pipeline's
switches, the system names, the demo's, the CLI's and the jobs' switches.
"""
from __future__ import annotations

import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.notelabels import Placed  # noqa: E402
from src.readable import fix_jumps, readable_lengths, same_grid, stem_voices, writable  # noqa: E402

V = ROOT / "data" / "processed" / "vocab.json"


def steady(n_bars=6, jump_at=None, jump_s=0.5, jump_q=3.0):
    """A 4/4 grid, one beat 0.5 s; with ``jump_at`` (a beat's q) that beat's step is ``jump_q``
    quarters in ``jump_s`` seconds."""
    q, t = [0.0], [0.0]
    while q[-1] < 4 * n_bars:
        s = jump_q if jump_at is not None and abs(q[-1] - jump_at) < 1e-9 else 1.0
        q.append(q[-1] + s)
        t.append(t[-1] + (jump_s if s != 1.0 else 0.5))
    return {"t": np.array(t), "q": np.array(q), "bar_q": np.arange(0, 4 * n_bars + 1, 4.0)[:n_bars],
            "bars": ["4/4"] * n_bars}


def test_fix_jumps():
    g = steady(jump_at=8.0)                         # bar 3: beat 1 -> beat 4 in 0.5 s
    assert fix_jumps(g) == 1
    assert g["bars"] == ["4/4", "4/4", "2/4", "4/4", "4/4", "4/4"]
    assert list(g["bar_q"]) == [0, 4, 8, 10, 14, 18]
    i = int(np.flatnonzero(np.isclose(g["q"], 8.0))[0])
    assert g["q"][i + 1] == 9.0 and g["q"][i + 2] == 10.0          # 3 quarters -> 1; the next bar line follows
    # the tracker missed beats: three quarters in 1.5 s (the usual tempo) is no jump
    g = steady(jump_at=8.0, jump_s=1.5)
    assert fix_jumps(g) == 0 and g["bars"] == ["4/4"] * 6
    # a whole bar in 0.5 s: one beat's time, but the bar keeps two beats -> 2/4, not 1/4
    g = steady(jump_at=8.0, jump_q=4.0)
    assert fix_jumps(g) == 1 and g["bars"][2] == "2/4" and list(g["bar_q"])[3] == 10
    # a step that leaves its bar (beat 3 -> the next bar's beat 2) is not this case
    g = steady(jump_at=10.0, jump_q=3.0)
    assert fix_jumps(g) == 0
    # too short a grid; a jump above a lower threshold is left
    assert fix_jumps({"t": [0, 1], "q": [0, 1], "bar_q": [0], "bars": ["4/4"]}) == 0
    assert fix_jumps(steady(jump_at=8.0), ratio=0.1) == 0
    # a 3/4 bar: beat 1 -> the next bar line in one beat's time keeps two beats -> 2/4
    g = steady(jump_at=8.0, jump_q=3.0)
    g["bars"][2] = "3/4"
    g["bar_q"] = np.array([0, 4, 8, 11, 15, 19], np.float64)
    assert fix_jumps(g) == 1 and g["bars"][2] == "2/4" and list(g["bar_q"])[3] == 10


def test_readable_lengths():
    bars = ["4/4", "4/4"]
    rest = Placed(0, Fraction(0), 60, Fraction(3, 4), 0)            # next at 1: a 16th rest -> closed to 1
    nxt = Placed(0, Fraction(1), 62, Fraction(1, 2), 0)
    over = Placed(0, Fraction(2), 64, Fraction(3, 4), 0)            # next at 2 1/2: 1/4 over -> trimmed to 1/2
    nxt2 = Placed(0, Fraction(5, 2), 65, Fraction(1, 2), 0)
    last = Placed(0, Fraction(7, 2), 67, Fraction(1, 4), 0)         # the next is the bar line + 0: a rest to the
    first = Placed(1, Fraction(0), 69, Fraction(1), 0)              # bar line is closed (nxt <= bar end)
    trip = Placed(1, Fraction(1, 3), 71, Fraction(1, 3), 0)         # triplet -> next at 3/4: across grids, left
    straight = Placed(1, Fraction(3, 4), 72, Fraction(1, 4), 0)
    long_rest = Placed(1, Fraction(2), 74, Fraction(1, 4), 0)       # a rest of 3/4: longer than a 16th, left
    after = Placed(1, Fraction(3), 76, Fraction(1), 0)
    grace = Placed(1, Fraction(3), 75, Fraction(0), 0, grace=True)
    other = Placed(0, Fraction(0), 40, Fraction(1, 2), 1)           # staff 2 alone: nothing after it
    placed = [rest, nxt, over, nxt2, last, first, trip, straight, long_rest, after, grace, other]
    closed, trimmed = readable_lengths(placed, bars)
    assert rest.total == 1 and over.total == Fraction(1, 2)
    assert last.total == Fraction(1, 2)
    assert trip.total == Fraction(1, 3) and long_rest.total == Fraction(1, 4)
    assert grace.total == 0 and other.total == Fraction(1, 2) and after.total == 1
    assert (closed, trimmed) == (2, 1)
    assert writable(Fraction(3, 4)) and writable(Fraction(1, 3)) and writable(Fraction(1, 6)) and writable(4)
    assert not writable(Fraction(5, 12)) and not writable(Fraction(5, 4)) and not writable(0)
    assert same_grid(Fraction(1), Fraction(4, 3)) and same_grid(Fraction(1, 4), Fraction(3, 4))
    assert not same_grid(Fraction(1, 3), Fraction(1, 2))


def test_stem_voices():
    import music21 as m21
    s = m21.stream.Score()
    part = m21.stream.Part()
    m = m21.stream.Measure(number=1)
    hi, lo = m21.stream.Voice(id="1"), m21.stream.Voice(id="2")
    for p in ("E5", "F5", "G5", "A5"):
        hi.append(m21.note.Note(p, quarterLength=1))
    lo.append(m21.chord.Chord(["C4", "E4"], quarterLength=2))
    lo.append(m21.note.Rest(quarterLength=2))
    m.insert(0, hi)
    m.insert(0, lo)
    one = m21.stream.Measure(number=2)
    one.append(m21.note.Note("C4", quarterLength=4))
    part.append([m, one])
    s.append(part)
    assert stem_voices(s) == 1
    assert all(n.stemDirection == "up" for n in hi.notes) and all(n.stemDirection == "down" for n in lo.notes)
    assert lo.getElementsByClass(m21.note.Rest)[0].style.hideObjectOnPrint is True
    assert one.notes[0].stemDirection == "unspecified"
    with tempfile.TemporaryDirectory() as t:
        x = s.write("musicxml", Path(t) / "s.musicxml").read_text(encoding="utf-8")
    assert 'print-object="no"' in x and "<stem>up</stem>" in x and "<stem>down</stem>" in x


M4 = ("bar pos_0 p_55 d_1 p_58 d_1/2 p_60 d_1/2 p_65 d_1/2 staff_1 p_41 d_4 p_48 d_4 pos_0 sub_1/2 staff_0 p_62 "
      "d_7/2 p_67 d_1/2 pos_1 p_58 d_1 p_60 d_1 p_65 d_1 staff_1 p_55 d_1 pos_2 staff_0 p_58 d_2 p_60 d_2 p_65 d_2 "
      "p_67 d_2 staff_1 p_55 d_2 pos_2 sub_1/2 staff_0 p_58 d_3/2").split()   # the user's song, measure 4


def sounding(score):
    """{(staff, onset, pitch, length)} the score plays, ties joined per pitch (quarters from the
    start). By hand: music21's stripTies does not join a tied note of a chord into the next chord
    when the chords' other notes differ, as the two-voice writer writes held notes."""
    import music21 as m21
    out = set()
    for si, part in enumerate(score.parts):
        notes = []
        for el in part.recurse().notes:
            if el.duration.isGrace:
                continue
            on = Fraction(el.getOffsetInHierarchy(part)).limit_denominator(1000)
            ln = Fraction(el.quarterLength).limit_denominator(1000)
            for n in (el.notes if isinstance(el, m21.chord.Chord) else [el]):
                notes.append((on, n.pitch.midi, ln, n.tie.type if n.tie is not None else None))
        held = {}   # pitch -> [onset, end] of the note a tie can continue
        for on, p, ln, tie in sorted(notes):
            h = held.get(p)
            if tie in ("continue", "stop") and h is not None and h[1] == on:
                h[1] = on + ln
            else:
                if h is not None:
                    out.add((si, h[0], p, h[1] - h[0]))
                held[p] = [on, on + ln]
        out |= {(si, h[0], p, h[1] - h[0]) for p, h in held.items()}
    return out


def voices_of(score):
    import music21 as m21
    return [[list(m.voices) for m in part.getElementsByClass(m21.stream.Measure)] for part in score.parts]


def test_two_voices_measure_4():
    """The user's measure 4: three voices in the treble, two of them with stems down, the up-stem
    one partly below them, B-flat 3 struck while held. Written with two_voices: two voices, the
    upper one higher, stems up / down; the held D4 tied on in the upper voice; B-flat 3 ends where
    it is struck again; everything else sounds as before."""
    from src.preprocessing.detokenize import tokens_to_score
    old, new = tokens_to_score(M4), tokens_to_score(M4, two_voices=True)
    assert len(voices_of(old)[0][0]) == 3 and len(voices_of(new)[0][0]) == 2
    stem_voices(new)
    up, down = voices_of(new)[0][0]

    def pitches(v):
        return [p.midi for el in v.notes for p in el.pitches]
    assert sum(pitches(up)) / len(pitches(up)) > sum(pitches(down)) / len(pitches(down))
    assert all(el.stemDirection == "up" for el in up.notes) and all(el.stemDirection == "down" for el in down.notes)
    assert any(n.tie is not None for el in up.notes for n in getattr(el, "notes", [el]) if n.pitch.midi == 62)
    a, b = sounding(old), sounding(new)
    assert (0, Fraction(2), 58, Fraction(2)) in a and (0, Fraction(2), 58, Fraction(1, 2)) in b   # struck again at 2.5
    assert a - {(0, Fraction(2), 58, Fraction(2))} == b - {(0, Fraction(2), 58, Fraction(1, 2))}


def test_two_voices_sound_the_same():
    """Random dense bars (held notes, chords of different lengths, notes between held ones,
    tuplets, a note across the bar line): at most two voices per staff with two_voices, and the
    score sounds the same as the one-voice-per-free-slot writer's once a pitch struck while it is
    held is ended there (detokenize.py _one_key_one_note). Both write MusicXML."""
    import random
    from src.preprocessing.detokenize import (_carry_overflow, _one_key_one_note, bars_to_score, parse_tokens,
                                              write_musicxml)
    rng = random.Random(3)
    lengths = ("1/2", "1", "3/2", "2", "1/4", "1/3", "3", "7/2")
    for trial in range(25):
        toks = []
        for bar in range(3):
            toks.append("bar")
            if bar == 0:
                toks.append("ts_3/4" if trial % 3 == 0 else "ts_4/4")
            for pos in range(4):
                toks.append(f"pos_{pos}")
                for staff in (0, 1):
                    toks.append(f"staff_{staff}")
                    for _ in range(rng.randint(0, 3)):
                        toks += [f"p_{rng.randint(36, 60) + 24 * (1 - staff)}", f"d_{rng.choice(lengths)}"]
        bars = parse_tokens(toks)
        cut = _carry_overflow(parse_tokens(toks))   # notes held over a bar line are in the next bar, as the writer has them
        for b in cut:   # the reference: the old writer on the notes with one key, one note
            b.notes = [n for s in (0, 1) for n in _one_key_one_note([n for n in b.notes if n.staff == s])] +                       [n for n in b.notes if n.staff > 1]
        new, ref = bars_to_score(bars, two_voices=True), bars_to_score(cut)
        assert sounding(new) == sounding(ref), trial
        assert all(len(vs) <= 2 for staff in voices_of(new) for vs in staff), trial
        with tempfile.TemporaryDirectory() as t:
            write_musicxml(new, str(Path(t) / "s.musicxml"))


def test_stems_three_voices():
    """Three voices (the writer without two_voices): the highest up, the lowest down, the middle
    one by the side it is nearer."""
    import music21 as m21
    m = m21.stream.Measure(number=1)
    vs = []
    for k, ps in enumerate((["C6"], ["G5"], ["C3"])):
        v = m21.stream.Voice(id=str(k + 1))
        v.append(m21.note.Note(ps[0], quarterLength=4))
        m.insert(0, v)
        vs.append(v)
    part = m21.stream.Part()
    part.append(m)
    s = m21.stream.Score()
    s.append(part)
    stem_voices(s)
    assert [v.notes[0].stemDirection for v in vs] == ["up", "up", "down"]


def test_pipeline_and_switches():
    """V2Pipeline: off unless asked; on, its counts in the stats; the scripts' switches."""
    from scripts.system_scores import system_name
    from src.release import FIX_JUMPS, READABLE
    assert READABLE is True and FIX_JUMPS is True
    assert system_name(None, "", "", "nl+m", "synth", 0.95, True, "audio", readable=True,
                       fix_jumps=True) == "v2_nl_m_d95_tidy_rd_jf"
    for f in ("demo/app.py", "scripts/audio_to_score.py"):
        x = (ROOT / f).read_text(encoding="utf-8")
        assert "--no-readable" in x and "--no-fix-jumps" in x, f
    jobs = (ROOT / "pace" / "v2_jobs.sh").read_text()
    assert 'rd) out="$out --readable" ;;' in jobs and 'jf) out="$out --fix-jumps" ;;' in jobs
    if not V.exists():
        print("  (pipeline part skipped: no data/processed/vocab.json)")
        return
    from scripts.system_inputs import write_midi
    from src.pipeline import V2Pipeline
    from tests.test_glissando import chords
    from tests.test_v2_system import make_v2
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        lab, met = make_v2(tmp)
        notes = np.array(chords(0.5, 10, step=0.25), np.float32)
        midi = tmp / "p.mid"
        write_midi(notes, np.zeros((0, 2), np.float32), midi)
        times = np.arange(0.5, 10.5, 0.5)
        beats = (times, np.arange(len(times)) % 4 == 0)
        pipe = V2Pipeline(lab, met, vocab=V, device="cpu", window_notes=48)
        assert not pipe.readable and not pipe.fix_jumps
        off = pipe.run(midi, tmp / "off", beats=beats)
        on = pipe.run(midi, tmp / "on", beats=beats, readable=True, fix_jumps=True)
        assert off.stats["readable"] is False and off.stats["rests_closed"] == 0 and off.stats["jumps_fixed"] == 0
        assert on.stats["readable"] is True and "stems_set" in on.stats["export"]
        assert on.stats["notes_kept"] == off.stats["notes_kept"]           # lengths and stems, never notes
        assert (tmp / "on" / "score.musicxml").exists()


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
