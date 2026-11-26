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

``check_beats`` (``timing_beats=True``, _tb) judges a whole beat instead of one note: the user's
bars 64-71 repeat a syncopated figure played at 0, 0.5, 1, 1.5, 1.71, 2.24, 2.73, 3, 3.5 (the
straight 16th syncopation 16th-8th-8th-16th over beats 2-3), which the labeller wrote at 1 2/3,
2 1/3, 2 2/3. No one note there passes check_timing: 2.24 is 0.09 from 2 1/3 (under ``margin``),
and 1.71 sits halfway between 1 2/3 and 1 3/4. The beat as a whole does: in each beat (of a
quarter-beat bar) where the labeller wrote a triplet position or length, every note's played
position is measured against the straight sixteenths (0, 1/4, 1/2, 3/4 of the beat) and against
the eighth triplets (0, 1/3, 2/3; the sixths where the labeller wrote a sixth: a fast run in
sextuplets, as in the song's bar 136, is judged against the grid it was written on), measured from where the beat itself was played (its notes
written on the beat, their median, up to ``anchor`` early or late: in bars 67 and 69 the beat came
0.05 early, and its notes at 1.42 and 1.69 on the grid were 0.47 and 0.74 of the beat as played).
When the straight grid is the nearer on average (by ``edge``)
and no note is further than ``beat_tol`` from a straight sixteenth, the beat is written straight:
each triplet-written note at the sixteenth it was played nearest, each triplet length the nearest
sixteenth multiple. Played triplets (0, 0.33, 0.67) are nearer the triplet grid and stay as written.
A swung long-short (0, 0.67-0.73 of the beat as played) mostly goes to the dotted eighth and
sixteenth: the bar the user said was written right (14) writes that figure straight, and bar 15
had one chord played at 0.70 written half at 2 3/4 and half at 2 2/3.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

TOL = 0.06       # of a beat: how close to a straight position the note was played
MARGIN = 0.10    # of a beat: how far from the triplet position it was written on
BEAT_TOL = 0.10  # check_beats: the furthest any note of a straightened beat was played from a sixteenth
EDGE = 0.01      # check_beats: by how much the straight grid must be the nearer on average
ANCHOR = 0.10    # check_beats: the most a beat's on-beat notes may shift its positions (played early / late)


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


def _straighten(p, s: Fraction, bar_ts: list[str]) -> bool:
    """Write a note at straight position ``s`` of its bar (the next bar's downbeat when ``s`` is the
    bar's length), its triplet length the nearest sixteenth multiple; False when there is no next bar."""
    from .notelabels import bar_quarters
    if s >= bar_quarters(bar_ts[p.bar]):
        if p.bar + 1 >= len(bar_ts):
            return False
        p.bar, p.pos = p.bar + 1, Fraction(0)
    else:
        p.pos = s
    tot = Fraction(p.total)
    if tot.denominator % 3 == 0:
        p.total = max(Fraction(round(tot * 4), 4), Fraction(1, 4))
    return True


def check_beats(placed: list, onsets: np.ndarray, grid: dict, bar_ts: list[str],
                beat_tol: float = BEAT_TOL, edge: float = EDGE, anchor: float = ANCHOR) -> int:
    """Write straight the beats whose timing fits straight sixteenths better than eighth triplets
    (module docstring). Arguments as check_timing's. Changes ``placed`` in place; returns how many
    notes changed (position or length)."""
    from .notelabels import grid_position
    t, q = np.asarray(grid.get("t", []), np.float64), np.asarray(grid.get("q", []), np.float64)
    bar_q = np.asarray(grid.get("bar_q", []), np.float64)
    if len(t) < 2 or not len(bar_q):
        return 0
    n_bars = min(len(bar_q), len(bar_ts))
    use = [p for p in placed if not p.grace and getattr(p, "src", -1) >= 0 and p.bar < n_bars
           and quarter_beat(bar_ts[p.bar])]
    if not use:
        return 0
    played = grid_position(np.asarray([onsets[p.src] for p in use], np.float64), t, q)[0]
    beats = {}
    for p, qp in zip(use, played):
        beats.setdefault((p.bar, int(Fraction(p.pos))), []).append((p, float(qp - bar_q[p.bar])))
    straight = np.array([0, 0.25, 0.5, 0.75, 1.0])
    thirds, sixths = np.array([0, 1 / 3, 2 / 3, 1.0]), np.arange(7) / 6
    changed = 0
    for (bar, beat), group in beats.items():
        if not any(is_triplet(Fraction(p.pos) - beat) or Fraction(p.total).denominator % 3 == 0 for p, _ in group):
            continue
        x = np.clip(np.array([pp - beat for _, pp in group]), -0.5, 1.5)
        on = [xi for (p, _), xi in zip(group, x) if Fraction(p.pos) == beat]
        if on:                                              # the beat as played: its on-beat notes' median
            x = x - float(np.clip(np.median(on), -anchor, anchor))
        d_s = np.abs(x[:, None] - straight[None]).min(1)
        sext = any(Fraction(p.pos - beat).denominator == 6 for p, _ in group)
        d_t = np.abs(x[:, None] - (sixths if sext else thirds)[None]).min(1)
        if d_s.max() > beat_tol or d_s.mean() + edge > d_t.mean():
            continue
        for (p, pp), xi in zip(group, x):
            pos, tot = Fraction(p.pos), Fraction(p.total)
            if is_triplet(pos - beat):
                s = Fraction(beat) + Fraction(int(np.argmin(np.abs(straight - xi))), 4)
                changed += _straighten(p, s, bar_ts)
            elif tot.denominator % 3 == 0:
                p.total = max(Fraction(round(tot * 4), 4), Fraction(1, 4))
                changed += 1
    return changed
