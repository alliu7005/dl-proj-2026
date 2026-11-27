"""Rules that make the written score easier to read, without changing what it plays (much).

From the user's hands-on reading of the demo's scores (2026-11-26):

``fix_jumps`` (the grid, before the labeller): the meter model's decoder may give one beat interval
several beats: in the user's song, bar 47 went from beat 1 to beat 4 in 0.46 s (the beats around it
0.47 s apart), and bar 51 took a whole bar in 0.66 s. The decoder wanted a downbeat where it heard
one, but the bar had only two beats (one), and a short bar costs it a time-signature change it is
tuned to avoid, so it stretched the bar: under a second of music written over three or four beats,
as long notes and rests. Here a step is a jump when it packs its beats into far less time than the
steps around it take for theirs (seconds per quarter below ``ratio`` of their median); it gets the
beats its time holds (rounded up from 0.4: a slowing-down keeps its beat), and its bar is shortened
by the rest and given its own time signature (bar 47 -> 2/4). A bar keeps two beats at least
(``MIN_BEATS``): a one-beat bar is all but never written, and the song's bar 51 (0.66 s, 1.4 beats,
after a slowing-down) reads better as 2/4 than as 1/4. The bars after keep their lengths and
downbeats. A jump where the tracker missed beats (a long gap at the usual tempo) is no jump here.

``readable_lengths`` (the labelled notes): a rest of a 16th or less between a note and the next
onset of its staff, in its bar, is closed (the note held to it), and a note running up to a 16th
past the next onset of its staff stops there (tidy_lengths trims up to a 32nd), when the new length
is one note to write (``writable``: plain, dotted or triplet; closing a triplet eighth's rest to the
next sixteenth would make it 5/12 of a beat, worse than the rest) and the note does not go from a
triplet position to a straight one or back (``same_grid``: a triplet eighth trimmed to a straight
sixteenth's onset would be a 1/6 written across two grids). Scores do write
such rests: closing them lowered duration accuracy against ASAP's scores by about 2 points
(src/pipeline.py tidy_lengths), so the scoring script measures this as its own system (_rd).

``stem_voices`` (the MusicXML): in a measure where a staff has two voices or more, the voice with
the higher notes gets its stems up and the lower one down (of three, the middle one by the side it
is nearer), as engravers write them. With ``readable`` the writer gives a staff two voices at most,
ordered by pitch (src/preprocessing/detokenize.py ``_two_voices``: the user's "unlasting" song had
three treble voices in bar 4, the up-stem one partly below two down-stem ones); without, both
voices' stems point the same way and the demo's viewer draws them over each other (the user's
bar 29: half-note chords under a melody looked like 32nd-beamed chords). The rests that fill the
other voices out (all but the one with the most notes) are hidden (MusicXML print-object="no"), as
engravers leave them out where a held note makes a second voice: the user's bar 50 had a half, a
quarter and an eighth rest under its chords.
"""
from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction

import numpy as np

RATIO = 0.5        # a step is a jump below this share of its neighbours' seconds per quarter
WINDOW = 8         # neighbours on each side
MIN_BEATS = 2      # a shortened bar keeps this many beats at least
GAP = Fraction(1, 4)       # the longest rest closed (a 16th), the longest overlap trimmed


def fix_jumps(grid: dict, ratio: float = RATIO, window: int = WINDOW, min_beats: int = MIN_BEATS) -> int:
    """Shorten the bars the decoder stretched with a jump (module docstring). Changes ``grid``
    ("q", "bars", "bar_q") in place; returns how many bars were shortened."""
    from .beats import beat_layout
    from .meter import bar_quarters
    t = np.asarray(grid.get("t", []), np.float64)
    q = np.asarray(grid.get("q", []), np.float64).copy()
    bar_q = np.asarray(grid.get("bar_q", []), np.float64).copy()
    bars = list(grid.get("bars", []))
    if len(t) < 4 or len(bar_q) < 2 or len(bars) != len(bar_q):
        return 0
    fixed = 0
    for i in range(len(t) - 1):
        s, dt = q[i + 1] - q[i], t[i + 1] - t[i]
        if s <= 0 or dt <= 0:
            continue
        lo, hi = max(0, i - window), min(len(t) - 1, i + window + 1)
        spq = [(t[j + 1] - t[j]) / (q[j + 1] - q[j]) for j in range(lo, hi - 1)
               if j != i and q[j + 1] > q[j] and t[j + 1] > t[j]]
        if len(spq) < 3:
            continue
        med = float(np.median(spq))
        if dt / s >= ratio * med:
            continue
        b = int(np.searchsorted(bar_q, q[i] + 1e-9, side="right")) - 1
        if b < 0 or b + 1 >= len(bar_q) or q[i + 1] > bar_q[b + 1] + 1e-9:
            continue                                   # the step leaves its bar: not this case
        unit = beat_layout(bars[b])[1]
        new = max(unit, np.floor(dt / med / unit + 0.6) * unit)
        length = bar_q[b + 1] - bar_q[b]
        delta = min(s - new, length - min_beats * unit)     # the bar keeps min_beats beats
        if delta <= 1e-9:
            continue
        try:
            den = int(bars[b].split("/")[1])
        except (IndexError, ValueError):
            continue
        num = Fraction(length - delta).limit_denominator(48) * den / 4
        if num.denominator != 1 or num < 1:
            continue
        q[i + 1:] -= delta
        bar_q[b + 1:] -= delta
        bars[b] = f"{num.numerator}/{den}"
        assert abs(float(bar_quarters(bars[b])) - (length - delta)) < 1e-6
        fixed += 1
    if fixed:
        grid["q"], grid["bar_q"], grid["bars"] = q, bar_q, bars
    return fixed


def writable(d: Fraction) -> bool:
    """A length (quarters) one note can be written with: a plain value (a 64th to a breve), dotted,
    or a triplet's (2/3 of a plain one)."""
    d = Fraction(d)
    for k in range(-4, 4):
        v = Fraction(2) ** k
        if d in (v, v * Fraction(3, 2), v * Fraction(2, 3)):
            return True
    return False


def same_grid(a: Fraction, b: Fraction) -> bool:
    """Not one position on the triplet grid and the other off the beat on the straight one (bar
    starts are whole quarters or halves in the meters the grid writes, so positions from the first
    bar line tell)."""
    fam = lambda x: "beat" if x.denominator == 1 else ("triplet" if x.denominator % 3 == 0 else "straight")  # noqa: E731
    return {fam(Fraction(a)), fam(Fraction(b))} != {"triplet", "straight"}


def readable_lengths(placed: list, bar_ts: list[str], gap: Fraction = GAP) -> tuple[int, int]:
    """Close rests of at most ``gap`` before the next onset of a note's staff in its bar, and trim
    overlaps of at most ``gap`` past it (module docstring). Changes ``placed`` (src/notelabels.py
    Placed) in place; returns (rests closed, overlaps trimmed)."""
    from .pipeline import _bar_starts
    starts = _bar_starts(placed, bar_ts)
    at = lambda p: starts[p.bar] + Fraction(p.pos)  # noqa: E731
    onsets = {}
    for p in placed:
        if not p.grace:
            onsets.setdefault(p.staff, set()).add(at(p))
    onsets = {s: sorted(v) for s, v in onsets.items()}
    closed = trimmed = 0
    for p in placed:
        if p.grace or not p.total:
            continue
        q = at(p)
        on = onsets[p.staff]
        k = bisect_right(on, q)
        if k >= len(on):
            continue
        nxt, end = on[k], q + Fraction(p.total)
        if not writable(nxt - q) or not same_grid(q, nxt):
            continue
        if 0 < end - nxt <= gap:
            p.total = nxt - q
            trimmed += 1
        elif 0 < nxt - end <= gap and nxt <= starts[p.bar + 1]:   # a short rest, inside the bar
            p.total = nxt - q
            closed += 1
    return closed, trimmed


def stem_voices(score) -> int:
    """Stems up for the higher voice, down for the others, and the rests of all but the fullest voice
    hidden, in every measure where a staff has two voices or more (module docstring). Changes the
    music21 score in place; returns how many measures."""
    import music21 as m21
    n = 0
    for part in score.parts:
        for meas in part.getElementsByClass(m21.stream.Measure):
            vs = list(meas.voices)
            if len(vs) < 2:
                continue

            def height(v):
                ps = [p.midi for el in v.notes for p in el.pitches]
                return sum(ps) / len(ps) if ps else -1
            hs = [height(v) for v in vs if v.notes]
            top, bottom = max(hs, default=0), min(hs, default=0)
            full = max(vs, key=lambda v: sum(float(el.quarterLength) for el in v.notes))
            for v in vs:
                h = height(v)   # the highest voice up, the lowest down, one between by the side it is nearer
                up = h == top if len(hs) < 3 else (h == top or (h != bottom and h - bottom > top - h))
                for el in v.notes:
                    el.stemDirection = "up" if up else "down"
                if v is not full:
                    for r in v.getElementsByClass(m21.note.Rest):
                        r.style.hideObjectOnPrint = True
            n += 1
    return n
