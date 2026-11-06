"""Extract notes from a MusicXML score (music21) and turn bars of it into target tokens.

Target token grammar for one bar (all positions/durations in quarter notes, exact fractions):

    bar [ts_3/4] [key_-2]
      pos_0  [ped_on|ped_off ...]  [staff_1] p_60 d_1/2 [tie] [arp] [orn_trill]  p_64 d_1/2 ...
      pos_0 sub_1/2 ...      (position = integer quarter + optional fractional part)

* ``bar`` marks a bar line. A segment that starts mid-bar has no leading ``bar``: it opens with
  ``ts_* key_*`` and its first position token says where in the bar it starts, e.g.
  ``ts_3/4 key_0 pos_1 sub_1/2 p_60 ...`` = "starts on beat 2.5 of a 3/4 bar".
* ``ts_*`` / ``key_*`` are emitted at the start of a segment and whenever they change.
* ``staff_k`` is emitted only when the staff changes within the bar (staff resets to 0 every bar).
* Grace notes use ``grace`` instead of a ``d_*`` token and sit at the position of their main note.
* ``tie`` marks a note that continues a tied note (no new onset in the performance).
* Rests are not tokenized; they are recoverable from positions + durations during score rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import music21 as m21

MAX_DEN = 96

_ORNAMENTS = [
    ("trill", "Trill"),
    ("mordent", "GeneralMordent"),
    ("mordent", "Mordent"),
    ("mordent", "InvertedMordent"),
    ("turn", "Turn"),
    ("tremolo", "Tremolo"),
    ("fermata", "Fermata"),
]
ORNAMENT_CLASSES = [(name, getattr(m21.expressions, cls)) for name, cls in _ORNAMENTS
                    if hasattr(m21.expressions, cls)]


ALLOWED_DENOMINATORS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 32, 48)


def frac(x) -> Fraction:
    """Snap a quarter-length to the nearest fraction with a musically common denominator
    (keeps the position/duration vocabulary small; odd tuplets in cadenzas get approximated)."""
    x = Fraction(x).limit_denominator(MAX_DEN * 8)
    best = None
    for d in ALLOWED_DENOMINATORS:
        cand = Fraction(round(x * d), d)
        if best is None or abs(cand - x) < abs(best - x) - Fraction(1, 10_000):
            best = cand
    return best


def fstr(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


@dataclass
class NoteEvent:
    offset: Fraction          # position within its measure (quarter notes)
    staff: int
    pitch: int
    duration: Fraction        # 0 for grace notes
    grace: bool = False
    tie: bool = False         # continuation of a tie (tie stop / continue)
    arpeggio: bool = False
    ornaments: tuple = ()
    seq: int = 0              # original order, keeps grace-note sequences stable


@dataclass
class Measure:
    length: Fraction
    time_signature: str
    key: int
    events: list = field(default_factory=list)


def _ornaments(el) -> tuple:
    out = []
    for e in getattr(el, "expressions", []):
        for name, cls in ORNAMENT_CLASSES:
            if isinstance(e, cls):
                out.append(name)
                break
    return tuple(sorted(set(out)))


def _is_arpeggio(el) -> bool:
    if any("Arpeggio" in type(e).__name__ for e in getattr(el, "expressions", [])):
        return True
    return any("Arpeggio" in type(s).__name__ for s in el.getSpannerSites())


def load_score(path: str) -> list[Measure]:
    """Parse a MusicXML file into a list of measures indexed by measure *index* (0-based),
    which is what ASAP's ``downbeats_score_map`` refers to."""
    score = m21.converter.parse(str(path))
    parts = list(score.parts)
    staves = [list(p.getElementsByClass("Measure")) for p in parts]
    n = max(len(s) for s in staves)
    measures: list[Measure] = []
    ts, key = "4/4", 0
    for i in range(n):
        ref = staves[0][i] if i < len(staves[0]) else None
        if ref is not None and ref.timeSignature is not None:
            ts = ref.timeSignature.ratioString
        if ref is not None and ref.keySignature is not None:
            key = int(ref.keySignature.sharps)
        length = max((frac(s[i].duration.quarterLength) for s in staves if i < len(s)), default=Fraction(0))
        meas = Measure(length=length, time_signature=ts, key=key)
        seq = 0
        for staff, s in enumerate(staves):
            if i >= len(s):
                continue
            for el in s[i].flatten().notes:
                grace = el.duration.isGrace
                off = frac(el.offset)
                dur = Fraction(0) if grace else frac(el.duration.quarterLength)
                arp, orn = _is_arpeggio(el), _ornaments(el)
                sub = el.notes if isinstance(el, m21.chord.Chord) else [el]
                for nt in sub:
                    tie = nt.tie if nt.tie is not None else el.tie
                    meas.events.append(NoteEvent(
                        offset=off, staff=staff, pitch=int(nt.pitch.midi), duration=dur, grace=grace,
                        tie=tie is not None and tie.type in ("stop", "continue"),
                        arpeggio=arp, ornaments=orn, seq=seq))
                    seq += 1
        measures.append(meas)
    return measures


def bar_length(measures: list[Measure], bar: list[int]) -> Fraction:
    return sum((measures[i].length for i in bar), Fraction(0))


def tokenize_bars(measures: list[Measure], bars: list[list[int]],
                  pedal: list[list[tuple[Fraction, str]]] | None = None,
                  start_pos: float | None = None, end_pos: float | None = None) -> list[str]:
    """Tokenize consecutive bars. ``bars[j]`` lists the measure indices played as one bar
    (usually one, two for bars split across a repeat). ``pedal[j]`` holds (position, token).

    ``start_pos`` / ``end_pos`` (quarters) cut the first / last bar: only events at
    ``start_pos <= pos < end_pos`` are kept. A span starting mid-bar gets no leading ``bar`` token
    (``bar`` always means "a bar line here"), so the token streams of consecutive spans concatenate
    into one valid stream. Pedal items are taken as given."""
    toks: list[str] = []
    prev_ts = prev_key = None
    eps = 1e-9
    for j, bar in enumerate(bars):
        first = measures[bar[0]]
        lo = start_pos if j == 0 and start_pos is not None and start_pos > eps else None
        hi = end_pos if j == len(bars) - 1 else None
        if lo is None:
            toks.append("bar")
        if first.time_signature != prev_ts:
            toks.append(f"ts_{first.time_signature}")
            prev_ts = first.time_signature
        if first.key != prev_key:
            toks.append(f"key_{first.key}")
            prev_key = first.key

        # (offset, priority, staff, seq, pitch, payload); priority: pedal < grace < note
        items = []
        shift = Fraction(0)
        for mi in bar:
            for e in measures[mi].events:
                p = e.offset + shift
                if (lo is not None and float(p) < lo - eps) or (hi is not None and float(p) >= hi - eps):
                    continue
                # canonical order: grace notes keep their written sequence, other notes go by pitch
                items.append((p, 1 if e.grace else 2, e.staff, e.seq if e.grace else 0, e.pitch, e))
            shift += measures[mi].length
        for pos, tok in (pedal[j] if pedal else []):
            items.append((pos, 0, -1, 0, 0, tok))
        items.sort(key=lambda x: x[:5])

        # staff_k is written when the staff changes, assuming staff 0 at a bar line; a span that
        # starts mid-bar states its first staff explicitly so concatenated spans parse correctly
        cur_pos, cur_staff = None, (None if lo is not None else 0)
        for pos, _, staff, _, _, e in items:
            if pos != cur_pos:
                whole = pos.numerator // pos.denominator
                toks.append(f"pos_{whole}")
                if pos != whole:
                    toks.append(f"sub_{fstr(pos - whole)}")
                cur_pos = pos
            if isinstance(e, str):
                toks.append(e)
                continue
            if staff != cur_staff:
                toks.append(f"staff_{staff}")
                cur_staff = staff
            toks.append(f"p_{e.pitch}")
            toks.append("grace" if e.grace else f"d_{fstr(e.duration)}")
            if e.tie:
                toks.append("tie")
            if e.arpeggio:
                toks.append("arp")
            toks += [f"orn_{o}" for o in e.ornaments]
    return toks
