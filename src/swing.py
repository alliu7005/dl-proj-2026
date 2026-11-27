"""Swing written as straight notes with a swing marking (the labelled notes, before rendering).

A swung piece plays its pairs of eighths (or sixteenths) long-short, about 2:1, and its score
writes them as equal notes under one marking at the top ("swing", or ♪♪ = ♪³♪). The labeller
writes what it hears: in the Lemon arrangement (swung sixteenths) the second sixteenth of a pair
came out at 5/6 of the beat (a sextuplet), 1/3, or as a late 32nd (3/8, 7/8), where the score writes
3/4 and 1/4; the timing checks (src/timing.py) leave notes played between 2/3 and 3/4 alone by design.

``detect_swing(placed, bar_ts)`` looks at the positions in the quarter-beat bars:
  * swung sixteenths: the second note of each half-beat pair at 1/3 or 5/6, at least ``MIN_NOTES``
    of them, ``DOMINANCE`` times the straight 1/4 and 3/4, and few at the other triplet positions
    (1/6, 2/3) that real sextuplets and triplets also fill;
  * the swing's leftovers: most sixteenths written straight already (the labeller and the timing
    checks), the rest at 1/3 / 5/6 or as late 32nds at 3/8 / 7/8, which outnumber the 32nds at 1/8 /
    5/8 that real 32nd runs fill as well (Lemon: 366 straight, 75 swung, 58 late, almost no early 32nds);
  * swung eighths: 2/3 holding most of the off-beat notes, the triplet's middle note 1/3 rare.
``write_swing(placed, bar_ts)``: in a swung piece every note start and end at those positions moves
to the straight one, lengths following; the kind is returned for the marking (src/pipeline.py
engrave). Other positions are left as they are.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction

MIN_NOTES = 16          # swung second notes needed before a piece is called swung
DOMINANCE = 3.0
STRAY = 0.25            # the other triplet positions: at most this share of the swung ones
MARKINGS = {"16th": "Swing 16ths", "8th": "Swing"}   # plain text: viewers lack the note glyphs
F = Fraction
SWUNG_16 = {F(1, 3): F(1, 4), F(5, 6): F(3, 4)}
LATE_32 = {F(3, 8): F(1, 4), F(7, 8): F(3, 4)}
SWUNG_8 = {F(2, 3): F(1, 2)}


def _frac(x: Fraction) -> Fraction:
    return x - (x.numerator // x.denominator)


def detect_swing(placed: list, bar_ts: list[str]) -> tuple[str | None, dict]:
    """(kind, positions to straighten): '16th', '8th' or None (module docstring)."""
    from .timing import quarter_beat
    c = Counter()
    for p in placed:
        if not p.grace and p.bar < len(bar_ts) and quarter_beat(bar_ts[p.bar]):
            c[_frac(Fraction(p.pos))] += 1
    swung = c[F(1, 3)] + c[F(5, 6)]
    straight = c[F(1, 4)] + c[F(3, 4)]
    stray = c[F(1, 6)] + c[F(2, 3)]
    late, early = c[F(3, 8)] + c[F(7, 8)], c[F(1, 8)] + c[F(5, 8)]
    if swung >= MIN_NOTES and swung >= DOMINANCE * max(straight, 1) and stray <= STRAY * swung:
        return "16th", dict(SWUNG_16)
    if (straight >= DOMINANCE * max(swung, 1) and swung + late >= MIN_NOTES
            and stray <= STRAY * (swung + late) and late >= DOMINANCE * early):
        return "16th", {**SWUNG_16, **(LATE_32 if late else {})}
    two3, half = c[F(2, 3)], c[F(1, 2)]
    if (two3 >= MIN_NOTES and two3 >= DOMINANCE * max(half, 1) and c[F(1, 3)] <= STRAY * two3
            and c[F(5, 6)] + c[F(1, 6)] <= STRAY * two3):
        return "8th", dict(SWUNG_8)
    return None, {}


def write_swing(placed: list, bar_ts: list[str]) -> tuple[int, str | None]:
    """Straighten a swung piece's notes in place (module docstring); (notes changed, kind or None)."""
    from .timing import quarter_beat
    kind, m = detect_swing(placed, bar_ts)
    if kind is None:
        return 0, None

    def move(x: Fraction) -> Fraction:
        f = _frac(x)
        return x - f + m[f] if f in m else x
    changed = 0
    for p in placed:
        if p.grace or p.bar >= len(bar_ts) or not quarter_beat(bar_ts[p.bar]):
            continue
        pos, end = Fraction(p.pos), Fraction(p.pos) + Fraction(p.total)
        npos, nend = move(pos), move(end)
        if nend <= npos:
            nend = npos + Fraction(p.total)
        if (npos, nend) != (pos, end):
            p.pos, p.total = npos, nend - npos
            changed += 1
    return changed, kind
