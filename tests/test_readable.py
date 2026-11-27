"""Tests for the readability rules (src/readable.py, V2Pipeline readable / fix_jumps, the switches).

    python -m tests.test_readable        # or: pytest tests/test_readable.py

fix_jumps: a step that packs three beats into one beat's time shortens its bar to 2/4 and moves
everything after it back; a long gap at the usual tempo (the tracker missed beats) is no jump; a
bar keeps two beats; a step that leaves its bar is left. readable_lengths: short rests closed,
small overlaps trimmed, not across the bar line, not into an unwritable length or across grids,
grace notes left. stem_voices: stems by voice, the other voices' rests hidden. The pipeline's
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
