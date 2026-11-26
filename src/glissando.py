"""Glissandos: found in a performance's notes, written as the score writes them.

A piano glissando is a hand sliding over the white keys (or the black keys). Transcribed, it is a
run of notes going one way, each on the next key of that colour (Transkun misses one now and
then), far faster than written notes are played: 60-110 notes a second in the two of the user's
song (2026-11-26), while 32nds at a fast tempo are 20-25. In 300 ASAP performances the rule fires
21 times in 13 (Liszt's Mephisto Waltz, Prokofiev's Toccata, Balakirev's Islamey: real white-key
sweeps); the one false alarm, a rolled chord (7 notes in 23 ms, black keys among them), is too
short and mixes the colours.

``find_glissandi`` finds them; V2Pipeline (src/pipeline.py, ``glissando=True``) gives the labeller
only each run's first and last note, so the run is not written as a cluster of 64ths, and the
export draws a "gliss." line between the two (``decorate``; MusicXML <glissando>), with the swept
notes put back into the score's MIDI so it sounds like one. No model learns this: none of our
training scores mark glissandos (ASAP's MusicXML has no <glissando> at all), so the scoring
script keeps it off unless asked (scripts/system_scores.py --glissando, system _gl).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

BLACK = frozenset({1, 3, 6, 8, 10})
KEYS = {c: [p for p in range(21, 109) if (p % 12 in BLACK) == (c == "black")] for c in ("white", "black")}
RANK = {c: {p: k for k, p in enumerate(v)} for c, v in KEYS.items()}


def colour(pitch: int) -> str:
    return "black" if int(pitch) % 12 in BLACK else "white"


@dataclass
class Glissando:
    idx: np.ndarray           # the run's notes (indices into the notes given), in order
    direction: int            # +1 up, -1 down
    keys: str                 # "white" or "black"
    t0: float                 # first and last onset (seconds)
    t1: float
    pitches: list = field(default_factory=list)

    @property
    def rate(self) -> float:
        return (len(self.idx) - 1) / max(self.t1 - self.t0, 1e-9)

    def summary(self) -> dict:
        return {"seconds": [round(self.t0, 3), round(self.t1, 3)], "notes": int(len(self.idx)),
                "from": int(self.pitches[0]), "to": int(self.pitches[-1]), "keys": self.keys,
                "direction": "up" if self.direction > 0 else "down", "notes_per_s": round(self.rate, 1)}


def _chains(notes: np.ndarray, direction: int, max_gap: float, max_skip: int, used: set) -> list[list[int]]:
    """Greedy chains in time order: each next note within ``max_gap`` s, of the same key colour,
    1..``max_skip`` keys of that colour further in ``direction``."""
    on, pitch = notes[:, 1].astype(np.float64), notes[:, 0].astype(int)
    out, taken = [], set(used)
    for i in range(len(notes)):
        if i in taken or not 21 <= pitch[i] <= 108:
            continue
        c = colour(pitch[i])
        chain, j = [i], i
        while True:
            nxt = None
            for k in range(j + 1, len(notes)):
                if on[k] - on[j] > max_gap:
                    break
                if k in taken or not 21 <= pitch[k] <= 108 or colour(pitch[k]) != c:
                    continue
                step = (RANK[c][pitch[k]] - RANK[c][pitch[j]]) * direction
                if 1 <= step <= max_skip:
                    nxt = k
                    break
            if nxt is None:
                break
            chain.append(nxt)
            j = nxt
        if len(chain) >= 3:
            taken.update(chain)
            out.append(chain)
    return out


def find_glissandi(notes: np.ndarray, min_notes: int = 8, min_rate: float = 30.0, min_seconds: float = 0.1,
                   max_gap: float = 0.07, max_skip: int = 2, join_gap: float = 0.05) -> list[Glissando]:
    """The glissandos in ``notes`` [N, >=2] (pitch, onset seconds, ...), any order. A run: at least
    ``min_notes`` notes over at least ``min_seconds``, ``min_rate`` notes a second or faster, one
    way, every note on the next key of one colour (or one further: a missed note). Two runs a missed
    note split (the next starting within ``join_gap`` s, further the same way) are one."""
    notes = np.asarray(notes)
    if len(notes) < min_notes:
        return []
    order = np.lexsort((notes[:, 0], notes[:, 1]))
    srt = notes[order]
    on, pitch = srt[:, 1].astype(np.float64), srt[:, 0].astype(int)
    found, used = [], set()
    for d in (+1, -1):
        chains = _chains(srt, d, max_gap, max_skip, used)
        merged = []
        for ch in sorted(chains, key=lambda c: on[c[0]]):
            if merged:
                prev = merged[-1]
                a, b = prev[-1], ch[0]
                c = colour(pitch[a])
                if (colour(pitch[b]) == c and 0 <= on[b] - on[a] <= join_gap
                        and 1 <= (RANK[c][pitch[b]] - RANK[c][pitch[a]]) * d <= max_skip + 2):
                    prev.extend(ch)
                    continue
            merged.append(list(ch))
        for ch in merged:
            t0, t1 = on[ch[0]], on[ch[-1]]
            if len(ch) < min_notes or t1 - t0 < min_seconds or (len(ch) - 1) / (t1 - t0) < min_rate:
                continue
            used.update(ch)
            found.append(Glissando(order[ch], d, colour(pitch[ch[0]]), float(t0), float(t1), pitch[ch].tolist()))
    return sorted(found, key=lambda g: g.t0)


def inner_mask(n: int, glissandi: list[Glissando]) -> np.ndarray:
    """True for the notes to keep: everything but each glissando's notes between its first and last."""
    keep = np.ones(n, bool)
    for g in glissandi:
        keep[g.idx[1:-1]] = False
    return keep


# --------------------------------------------------------------------------------------------
# in the score: the two written notes it joins (src/pipeline.py V2Pipeline.label), the line
# --------------------------------------------------------------------------------------------

def bar_starts(bar_ts: list[str]) -> list[Fraction]:
    from .notelabels import bar_quarters
    out, q = [], Fraction(0)
    for ts in bar_ts:
        out.append(q)
        q += bar_quarters(ts)
    return out + [q]


def attach(glissandi: list[Glissando], placed: list, grid: dict, bar_ts: list[str], tol: float = 1.0) -> list[dict]:
    """Each glissando's first and last note among the placed notes (src/notelabels.py Placed, piece
    bars): the note of that pitch the labeller put nearest to where the grid puts its onset (within
    ``tol`` quarters). The first note's written length then reaches the last (how a glissando is
    notated), unless that is longer than a bar of 4/4 twice over. An end the labeller left out (a
    lone note once the run between is gone looks like a transcription error: the user's song's
    second glissando began on one) is written here, at the sixteenth nearest the grid's place for
    its onset, on the other end's staff (or by its pitch), and appended to ``placed``. Returns what
    the export needs: {"start": (bar, pos, pitch, staff), "end": (...), "inner": pitches, ...,
    "written": how many ends this added}."""
    from .notelabels import Placed, bar_quarters, grid_position
    if not glissandi or len(grid.get("t", [])) < 2:
        return []
    starts = bar_starts(bar_ts)
    gb = np.asarray(grid.get("bar_q", []), np.float64)

    def abs_q(p):
        return starts[p.bar] + p.pos if p.bar < len(starts) else None

    def write(pitch, tq, staff):
        """A note at the sixteenth nearest piece quarter ``tq``, in the bar holding it."""
        b = int(np.clip(np.searchsorted(np.asarray(starts[:-1], np.float64), tq + 1e-9, side="right") - 1,
                        0, len(bar_ts) - 1))
        length = bar_quarters(bar_ts[b])
        pos = min(max(Fraction(round((tq - float(starts[b])) * 4), 4), Fraction(0)), length - Fraction(1, 4))
        p = Placed(b, pos, int(pitch), Fraction(1, 4), int(staff))
        placed.append(p)
        return p

    out = []
    for g in glissandi:
        q = grid_position(np.array([g.t0, g.t1]), grid["t"], grid["q"])[0]
        # grid quarters -> the piece's bars: the grid's bar lines (bar_q) to the written bar starts
        target = []
        for x in q:
            b = max(0, int(np.searchsorted(gb, x + 1e-6, side="right")) - 1) if len(gb) else 0
            b = min(b, len(starts) - 2)
            target.append(float(starts[b]) + float(x - (gb[b] if len(gb) else 0.0)))
        ends = []
        for pitch, tq in ((g.pitches[0], target[0]), (g.pitches[-1], target[1])):
            cands = [p for p in placed if p.pitch == pitch and not p.grace and abs_q(p) is not None
                     and abs(float(abs_q(p)) - tq) <= tol and all(p is not e for e in ends)]
            ends.append(min(cands, key=lambda p: abs(float(abs_q(p)) - tq)) if cands else None)
        a, b = ends
        added = 0
        for k, (pitch, tq) in enumerate(((g.pitches[0], target[0]), (g.pitches[-1], target[1]))):
            if ends[k] is None:
                other = ends[1 - k]
                staff = other.staff if other is not None else (0 if pitch >= 60 else 1)
                ends[k] = write(pitch, tq, staff)
                added += 1
        a, b = ends
        if abs_q(b) <= abs_q(a):   # both on one sixteenth: no room for a line (the notes stay written)
            continue
        span = abs_q(b) - abs_q(a)
        if span <= 8:
            a.total = span
        out.append({"start": (a.bar, a.pos, a.pitch, a.staff), "end": (b.bar, b.pos, b.pitch, b.staff),
                    "inner": [int(x) for x in g.pitches[1:-1]], "written": added, **g.summary()})
    return out


def _find(score, bar: int, pos: Fraction, pitch: int, staff: int):
    """The written note (or the note of a chord) at piece bar ``bar``, position ``pos``, of this
    pitch: on its staff first, else on any."""
    import music21 as m21
    parts = list(score.parts)
    for s in [staff] + [k for k in range(len(parts)) if k != staff]:
        if s >= len(parts):
            continue
        meas = parts[s].measure(bar + 1)
        if meas is None:
            continue
        for el in meas.recurse().notes:
            if el.duration.isGrace or abs(float(el.offset) - float(pos)) > 1e-6:
                continue
            for n in (el.notes if isinstance(el, m21.chord.Chord) else [el]):
                if n.pitch.midi == pitch:
                    return el if isinstance(el, m21.note.Note) else n, s, meas
    return None


def decorate(score, glissandi: list[dict], playback: bool = False) -> int:
    """Draw each glissando's "gliss." line in a music21 score (src/preprocessing/detokenize.py's):
    a wavy line from its first written note to its last. ``playback``: also put the swept notes
    between them, evenly spaced, so the score's MIDI plays the glissando (for the MIDI only: the
    MusicXML writes the line). Returns how many were drawn."""
    import music21 as m21
    drawn = 0
    for g in glissandi or []:
        a = _find(score, g["start"][0], Fraction(g["start"][1]), int(g["start"][2]), int(g["start"][3]))
        b = _find(score, g["end"][0], Fraction(g["end"][1]), int(g["end"][2]), int(g["end"][3]))
        if a is None or b is None:
            continue
        line = m21.spanner.Glissando([a[0], b[0]])
        line.lineType, line.label = "wavy", "gliss."
        score.insert(0, line)
        drawn += 1
        if playback and g.get("inner"):
            part = list(score.parts)[a[1]]
            t0 = float(a[2].offset) + float(Fraction(g["start"][1]))   # the measure's place in the part
            t1 = float(b[2].offset) + float(Fraction(g["end"][1]))
            step = (t1 - t0) / (len(g["inner"]) + 1)
            for k, p in enumerate(g["inner"], start=1):
                n = m21.note.Note(int(p))
                n.quarterLength = step
                n.volume.velocity = 70
                part.insert(t0 + k * step, n)
    return drawn
