"""The written rhythm checked against when each note was played: a triplet the timing contradicts.

The note labeller (src/labeller.py) places each note from its window of about 128 notes. Trained on
pop scores as well (nl_gq), it learned to write a swung long-short as a dotted rhythm, but now and
then a whole window tips the other way: in the user's song (2026-11-26) it wrote 40% of the notes of
bars 64-71 on triplet positions where the pianist played straight eighths, tighter on the beat grid
than anywhere else in the piece (70% of the notes within 30 ms of an eighth); the earlier labeller
had written 17% there.

``check_timing`` looks at every note the labeller wrote on a triplet position (a third or a sixth of
a beat) in a meter whose beat is a quarter (2/4, 3/4, 4/4, 5/4, ...; never 6/8, 9/8, 12/8, where
triplet groupings are the meter itself) and moves it to the straight position (an eighth or a
sixteenth) it was played on when the timing is clear about it:
  * within ``tol`` of a beat (0.06: 28 ms at the song's tempo) of that straight position, and
  * at least ``margin`` of a beat (0.10) from the triplet position it was written on.
Swung notes (played between 2/3 and 3/4 of the beat) sit closer than ``margin`` to both, so the
labeller's choice stays: the dotted rhythms it learned are untouched. A moved note's written length,
when it was a triplet length, becomes the nearest sixteenth multiple. V2Pipeline (src/pipeline.py,
``timing_check=True``) runs it after labelling, before grace notes are attached and lengths tidied;
the scoring script measures it as its own system (scripts/system_scores.py --timing-check, _tc).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

TOL = 0.06       # of a beat: how close to a straight position the note was played
MARGIN = 0.10    # of a beat: how far from the triplet position it was written on


def quarter_beat(ts: str) -> bool:
    """Whether a time signature's beat is a quarter (so a third of it is a triplet, not the meter)."""
    from .beats import beat_layout
    return beat_layout(ts)[1] == 1.0


def is_triplet(frac: Fraction) -> bool:
    return frac != 0 and frac.denominator % 3 == 0


def check_timing(placed: list, onsets: np.ndarray, grid: dict, bar_ts: list[str],
                 tol: float = TOL, margin: float = MARGIN) -> int:
    """Move the triplet-written notes the timing contradicts to the straight position they were
    played on (module docstring). ``placed``: src/notelabels.py Placed with ``src`` = the
    performed note's index in ``onsets`` (seconds); ``grid``: the meter model's (t, q, bar_q);
    ``bar_ts``: each written bar's time signature. Changes ``placed`` in place; returns how many
    notes moved."""
    from .notelabels import bar_quarters, grid_position
    t, q = np.asarray(grid.get("t", []), np.float64), np.asarray(grid.get("q", []), np.float64)
    bar_q = np.asarray(grid.get("bar_q", []), np.float64)
    if len(t) < 2 or not len(bar_q):
        return 0
    todo = [p for p in placed if not p.grace and getattr(p, "src", -1) >= 0 and p.bar < min(len(bar_q), len(bar_ts))
            and quarter_beat(bar_ts[p.bar]) and is_triplet(Fraction(p.pos) - int(p.pos))]
    if not todo:
        return 0
    played = grid_position(np.asarray([onsets[p.src] for p in todo], np.float64), t, q)[0]
    moved = 0
    for p, qp in zip(todo, played):
        w = Fraction(p.pos)
        beat = int(w)
        pp = float(qp - bar_q[p.bar])                       # where it was played, in the written bar
        straight = [Fraction(beat) + Fraction(j, 4) for j in range(5)]
        s = min(straight, key=lambda x: abs(pp - float(x)))
        if abs(pp - float(s)) > tol or abs(pp - float(w)) < margin:
            continue
        length = bar_quarters(bar_ts[p.bar])
        if s >= length:                                     # the next bar's downbeat
            if p.bar + 1 >= len(bar_ts):
                continue
            p.bar, p.pos = p.bar + 1, Fraction(0)
        else:
            p.pos = s
        tot = Fraction(p.total)
        if tot.denominator % 3 == 0:                        # a triplet length: the nearest sixteenth multiple
            p.total = max(Fraction(round(tot * 4), 4), Fraction(1, 4))
        moved += 1
    return moved
