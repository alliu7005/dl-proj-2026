"""Inverse of ``score.tokenize_bars``: token sequence -> music21 Score -> MusicXML.

    tokens_to_musicxml(tokens, "out.musicxml")   # tokens -> file, with export fallbacks

Notes with the same (staff, position, duration) become chords; overlapping chords inside one staff
are spread over voices; gaps are filled with rests. Pedal tokens are parsed (``_Bar.pedal``) but
not rendered into the MusicXML yet; evaluate pedal precision/recall on tokens directly.
"""
from __future__ import annotations

import copy
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

import music21 as m21

MIN_REST = 1 / 24
ORN_CLASSES = {"trill": "Trill", "mordent": "Mordent", "turn": "Turn", "tremolo": "Tremolo",
               "fermata": "Fermata"}


def _frac(s: str) -> Fraction:
    return Fraction(s)


@dataclass
class _Note:
    pos: Fraction
    staff: int
    pitch: int
    dur: Fraction
    grace: bool = False
    tie: bool = False
    arp: bool = False
    orns: list = field(default_factory=list)


@dataclass
class _Bar:
    ts: str
    key: int
    notes: list = field(default_factory=list)
    pedal: list = field(default_factory=list)  # (pos, "ped_on"/"ped_off")


def parse_tokens(tokens: list[str]) -> list[_Bar]:
    """Lenient parser: malformed model output is skipped rather than raising."""
    bars: list[_Bar] = []
    ts, key = "4/4", 0
    pos, staff, cur = Fraction(0), 0, None
    for t in tokens:
        if t == "bar":
            bars.append(_Bar(ts, key))
            pos, staff, cur = Fraction(0), 0, None
            continue
        if not bars:
            bars.append(_Bar(ts, key))
        b = bars[-1]
        kind, _, val = t.partition("_")
        try:
            if kind == "ts":
                ts = b.ts = val
            elif kind == "key":
                key = b.key = int(val)
            elif kind == "pos":
                pos, cur = Fraction(int(val)), None
            elif kind == "sub":
                pos, cur = Fraction(int(pos)) + _frac(val), None
            elif kind == "staff":
                staff = int(val)
            elif kind == "ped":
                b.pedal.append((pos, t))
            elif kind == "p":
                cur = _Note(pos, staff, int(val), Fraction(1, 4))
                b.notes.append(cur)
            elif kind == "d" and cur is not None:
                cur.dur = _frac(val)
            elif t == "grace" and cur is not None:
                cur.grace, cur.dur = True, Fraction(0)
            elif t == "tie" and cur is not None:
                cur.tie = True
            elif t == "arp" and cur is not None:
                cur.arp = True
            elif kind == "orn" and cur is not None:
                cur.orns.append(val)
        except (ValueError, ZeroDivisionError):
            continue
    return bars


# spelling: the tokens carry MIDI pitches only, so the exporter names each note itself
_STEPS = ("C", "D", "E", "F", "G", "A", "B")
_STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_ORDER, _FLAT_ORDER = "FCGDAEB", "BEADGCF"


def _key_alters(key: int) -> dict:
    """The alteration a key signature gives each step (``key``: sharps > 0, flats < 0)."""
    alt = {st: 0 for st in _STEPS}
    for st in (_SHARP_ORDER[:key] if key > 0 else _FLAT_ORDER[:-key]):
        alt[st] = 1 if key > 0 else -1
    return alt


def spell(midi: int, key: int = 0) -> m21.pitch.Pitch:
    """A MIDI pitch named for the key signature: the spelling the signature itself implies if there
    is one (Db and Gb with five flats, Cb with seven, E# with six sharps), else the signature's own
    direction (flats in flat keys, sharps in sharp keys; with none, C# F# G# but Eb Bb, as common
    practice writes them)."""
    pc = midi % 12
    alters = _key_alters(max(-7, min(7, int(key))))
    options = [(st, a) for st in _STEPS for a in (-1, 0, 1) if (_STEP_PC[st] + a) % 12 == pc]
    implied = [o for o in options if alters[o[0]] == o[1]]
    if implied:
        st, a = implied[0]
    else:
        natural = [o for o in options if o[1] == 0]
        if natural:
            st, a = natural[0]
        else:
            want = -1 if key < 0 or (key == 0 and pc in (3, 10)) else 1
            st, a = next(o for o in options if o[1] == want)
    p = m21.pitch.Pitch(st + {-1: "-", 0: "", 1: "#"}[a] + "4")
    p.octave += (midi - p.midi) // 12
    return p


def _make_element(group: list[_Note], key: int = 0, run: int = 0):
    """``run``: how many grace notes come before this position, each written as its own note (a run
    of them as unslashed sixteenths, the way engravers write one; a single one as a slashed eighth)."""
    pitches = sorted({n.pitch for n in group})
    el = (m21.note.Note(spell(pitches[0], key)) if len(pitches) == 1
          else m21.chord.Chord([spell(x, key) for x in pitches]))
    if group[0].grace:
        el.duration = m21.duration.Duration(0.25 if run > 1 else 0.5).getGraceDuration()
        if run > 1:
            el.duration.slash = False
    else:
        el.duration = m21.duration.Duration(group[0].dur)
    if any(n.arp for n in group):
        el.expressions.append(m21.expressions.ArpeggioMark())
    for o in sorted({o for n in group for o in n.orns}):
        cls = getattr(m21.expressions, ORN_CLASSES.get(o, ""), None)
        if cls is not None:
            el.expressions.append(cls())
    tied = {n.pitch for n in group if n.tie}
    return el, tied


def _regrid(bars: list[_Bar], grid: int) -> list[_Bar]:
    for b in bars:
        for n in b.notes:
            n.pos = Fraction(round(n.pos * grid), grid)
            if not n.grace:
                n.dur = max(Fraction(round(n.dur * grid), grid), Fraction(1, grid))
    return bars


def _clip_to_bars(bars: list[_Bar]) -> list[_Bar]:
    """Keep every note inside its bar. Model output can place a note past the bar's end (a wrong
    time signature, a missed bar line) or hold it across the bar line; music21's export then tries
    to carry the overflow into the next measure and fails (KeyError on a voice id, or "object
    already found in this Stream"). Notes starting past the end are dropped, longer ones cut at
    the bar line. Only used as an export fallback (the export normally carries such notes into the
    next bar at the same time instead: ``_carry_overflow``)."""
    for b in bars:
        try:
            bl = Fraction(m21.meter.TimeSignature(b.ts).barDuration.quarterLength)
        except Exception:  # noqa: BLE001
            b.ts, bl = "4/4", Fraction(4)
        kept = []
        for n in b.notes:
            if n.pos >= bl:
                continue
            if n.grace:  # a d_ token after "grace" would otherwise give it a length
                n.dur = Fraction(0)
            elif n.pos + n.dur > bl:
                n.dur = bl - n.pos
            kept.append(n)
        b.notes = kept
        b.pedal = [(p, t) for p, t in b.pedal if p < bl]
    return bars


def tokens_to_score(tokens: list[str], min_staves: int = 2, grid: int | None = None,
                    clip: bool = False, two_voices: bool = False) -> m21.stream.Score:
    """``grid``: optionally snap positions/durations to 1/grid quarter (fallback for export).
    ``clip``: keep notes inside their bars (fallback for malformed model output, see _clip_to_bars).
    ``two_voices``: see bars_to_score."""
    return bars_to_score(parse_tokens(tokens), min_staves, grid, clip, two_voices)


def _bar_length(ts: str) -> Fraction:
    try:
        return Fraction(m21.meter.TimeSignature(ts).barDuration.quarterLength)
    except Exception:  # noqa: BLE001
        return Fraction(4)


def _carry_overflow(bars: list[_Bar]) -> list[_Bar]:
    """Keep every note inside its bar without changing when it sounds. A note's position counts from
    its bar line and a bar lasts its time signature (as the metrics read the tokens): a note past
    the bar's end sounds in the next bar, so it is moved there; a note held across the bar line is
    split there and tied on (bars are added at the end if needed). Otherwise music21's MusicXML
    writer moves the overflow itself and shifts the next bar's notes (with several voices), or the
    bar is written longer than its time signature."""
    out = [_Bar(b.ts, b.key, [], []) for b in bars]
    pending = [(j, n) for j, b in enumerate(bars) for n in b.notes]
    peds = [(j, pos, kind) for j, b in enumerate(bars) for pos, kind in b.pedal]
    while pending:
        j, n = pending.pop(0)
        while j >= len(out):
            out.append(_Bar(out[-1].ts, out[-1].key, [], []))
        L = _bar_length(out[j].ts)
        if n.pos >= L:
            pending.append((j + 1, _Note(n.pos - L, n.staff, n.pitch, n.dur, n.grace, n.tie, n.arp, list(n.orns))))
            continue
        if not n.grace and n.pos + n.dur > L:
            out[j].notes.append(_Note(n.pos, n.staff, n.pitch, L - n.pos, False, n.tie, n.arp, list(n.orns)))
            pending.append((j + 1, _Note(Fraction(0), n.staff, n.pitch, n.pos + n.dur - L, False, True, False, [])))
            continue
        out[j].notes.append(n)
    for j, pos, kind in peds:
        while j < len(out) and pos >= _bar_length(out[j].ts):
            pos, j = pos - _bar_length(out[j].ts), j + 1
        while j >= len(out):
            out.append(_Bar(out[-1].ts, out[-1].key, [], []))
        out[j].pedal.append((pos, kind))
    for b in out:  # the order the tokens give within a position: grace notes first
        b.notes.sort(key=lambda n: (n.pos, not n.grace))
    return out


GRACE_REACH = Fraction(1, 2)  # an orphan grace note joins a main note of its staff this close after it


def _attach_orphan_graces(bar: _Bar):
    """A grace note is written in the voice of the main note it ornaments, which starts where it
    does. One with no main note of its staff there (a system writing it where it was played, just
    before its note) joins the first main note of its staff starting at most GRACE_REACH after it in
    the bar. Left alone, it is followed by a filler rest in its own voice, and viewers count it as a
    note of its written value (OpenSheetMusicDisplay made such a bar an eighth too long, shifting
    every later bar). Grace notes after the last note of a bar (after-graces, as in the scores) have
    no later note and stay."""
    for n in bar.notes:
        if not n.grace:
            continue
        mains = sorted({m.pos for m in bar.notes if not m.grace and m.staff == n.staff})
        if n.pos in mains:
            continue
        later = [p for p in mains if p > n.pos]
        if later and later[0] - n.pos <= GRACE_REACH:
            n.pos = later[0]


def _split_rest(pos: Fraction, length: Fraction) -> list[tuple[Fraction, Fraction]]:
    """A rest of ``length`` quarters at ``pos`` as standard rests: each the longest power-of-two
    value that fits and starts on a multiple of itself (half rests on half-bar beats, quarters on
    beats, ...). For rests with two or more dots, which printed music does not use."""
    out, end = [], pos + length
    while pos < end:
        d = Fraction(4)
        while d > Fraction(1, 64) and (d > end - pos or (pos / d).denominator != 1):
            d /= 2
        if d <= Fraction(1, 64) and d > end - pos:
            d = end - pos
        out.append((pos, d))
        pos += d
    return out


def _grid_rests(v):
    """Rests off the plain grid (a gap that ends on a triplet or quintuplet note) split at the beats
    and into the beat's own units (1/3 for triplets, 1/5 for quintuplets, 1/6 for sixths, ...): music21
    otherwise writes one rest over the gap as an irregular tuplet (5/3 of a quarter as a 6:5 half),
    which MuseScore rejects as corrupted. Rests on the plain grid stay (``_standard_rests``)."""
    import math
    for r in [e for e in v.notesAndRests if isinstance(e, m21.note.Rest)]:
        off = Fraction(r.offset).limit_denominator(10 ** 6)
        end = off + Fraction(r.quarterLength).limit_denominator(10 ** 6)
        if all(_grid_kind(x) <= 1 for x in (off, end)) and not r.duration.tuplets:
            continue
        cuts = [off] + [Fraction(i) for i in range(math.ceil(off), math.floor(end) + 1) if off < i < end] + [end]
        pieces = []
        for a, b in zip(cuts, cuts[1:]):
            dens = [x.denominator for x in (a % 1, b % 1) if x]
            den = math.lcm(*dens) if dens else 1
            k = den
            while k % 2 == 0:
                k //= 2
            if k == 1:
                pieces += _split_rest(a, b - a)
                continue
            unit = Fraction(1, k) if (a * k).denominator == 1 and (b * k).denominator == 1 else Fraction(1, den)
            x = a
            while x < b:
                pieces.append((x, min(unit, b - x)))
                x += unit
        hidden = r.style.hideObjectOnPrint
        v.remove(r)
        for p, d in pieces:
            nr = m21.note.Rest()
            nr.duration = m21.duration.Duration(d)
            nr.style.hideObjectOnPrint = hidden
            v.insert(float(p), nr)


TUPLET_SUB = 4      # _beat_tuplets: a tuplet beat's finest unit is a quarter of one tuplet note (32nd triplets)
TUPLETS = (3, 5, 7)  # the tuplets a beat can take (triplets, quintuplets, septuplets)
MAX_VOICES = 3       # _two_voices: a third voice only for a note on another grid; MuseScore reads four at most
WRITABLE = (16, 12, 8, 6, 4, 3, 2, 1)   # lengths in those units one note can take: plain or dotted values
TYPE_OF = {Fraction(1): "quarter", Fraction(1, 2): "eighth", Fraction(1, 4): "16th", Fraction(1, 8): "32nd",
           Fraction(1, 16): "64th"}


def _tie_pieces(tie: str | None, n: int) -> list:
    """The tie types of n tied pieces replacing one element whose tie was ``tie``."""
    if n == 1:
        return [tie]
    first = "continue" if tie in ("stop", "continue") else "start"
    last = "continue" if tie in ("start", "continue") else "stop"
    return [first] + ["continue"] * (n - 2) + [last]


def _beat_tuplets(v, bar_len: Fraction) -> int:
    """Every beat of a voice that holds a tuplet (a boundary on a triplet, quintuplet, ... grid)
    rewritten in a form notation programs read back exactly: its boundaries snapped to the grid of
    its tuplet (the most common one there; down to a quarter of one tuplet note), each element as
    writable pieces of that grid (a plain or dotted value, tied where it takes more than one), every
    piece carrying the beat's one tuplet (k notes in the time of n, n the power of two below k) with
    the bracket started on the beat's first piece and stopped on its last. Without it music21 writes
    lengths such as 7/12 as 12:7 tuplets, quintuplets beside triplets as 15:8, and slivers of 1/96,
    which MuseScore rebuilds to a different total and rejects ("voice too long", "incomplete
    measure"). A beat on the plain grid is left as it is. Changes the voice in place; returns how
    many beats it rewrote. Beats are quarters (the last one of a bar of eighths a half)."""
    import copy
    import math
    eps = Fraction(1, 10 ** 6)

    def span(el):
        a = Fraction(el.offset).limit_denominator(10 ** 6)
        return a, a + Fraction(el.quarterLength).limit_denominator(10 ** 6)

    def lines():   # the beat lines of the bar
        out, x = [], Fraction(0)
        while x < bar_len:
            out.append(x)
            x += 1
        return out + [bar_len]

    def tuplet_beats():   # the beats holding a boundary on a tuplet grid
        bl = lines()
        out = set()
        for el in v.notesAndRests:
            if el.duration.isGrace:
                continue
            for p in span(el):
                for s_, e_ in zip(bl, bl[1:]):
                    if s_ < p < e_ and _grid_kind((p - s_) / (e_ - s_)) > 1:
                        out.add(s_)
        return out
    # an element across the line of a tuplet beat (a long rest, or a plain note running into one)
    # is cut there first: a rest in two rests, a note in two tied ones
    tb = tuplet_beats()
    cut_at = sorted({x for s_ in tb for x in (s_, min(s_ + 1, bar_len))})
    for el in [e for e in v.notesAndRests if not e.duration.isGrace]:
        a, z = span(el)
        xs = [x for x in cut_at if a < x < z]
        if not xs:
            continue
        ties = {} if el.isRest else {p.pitch.midi: (p.tie.type if p.tie is not None else None) for p in
                                     (el.notes if isinstance(el, m21.chord.Chord) else [el])}
        bounds = [a] + xs + [z]
        v.remove(el)
        for pi, (p0, p1) in enumerate(zip(bounds, bounds[1:])):
            c = copy.deepcopy(el)
            c.duration = m21.duration.Duration(float(p1 - p0))
            if not el.isRest:
                for note in (c.notes if isinstance(c, m21.chord.Chord) else [c]):
                    t = _tie_pieces(ties.get(note.pitch.midi), len(bounds) - 1)[pi]
                    note.tie = m21.tie.Tie(t) if t else None
                if pi > 0:
                    c.expressions = []
            v.insert(float(p0), c)
    els = sorted([e for e in v.notesAndRests if not e.duration.isGrace], key=lambda e: e.offset)
    rewritten = 0
    b = Fraction(0)
    while b < bar_len:
        e_ = min(b + 1, bar_len)
        L = e_ - b
        seg = []
        for el in els:
            a = Fraction(el.offset).limit_denominator(10 ** 6)
            z = a + Fraction(el.quarterLength).limit_denominator(10 ** 6)
            if a >= b - eps and z <= e_ + eps and z > a:
                seg.append([el, a, z])
            elif a < e_ and z > b:   # an element across the beat line: this beat is left as it is
                seg = None
                break
        b_next = e_
        if not seg:
            b = b_next
            continue
        pts = [p for _, a, z in seg for p in (a, z) if b < p < e_]
        kinds = [_grid_kind((p - b) / L) for p in pts]
        kinds = [k for k in kinds if k > 1]
        if not kinds:
            b = b_next
            continue
        # the beat's one tuplet: triplets, quintuplets or septuplets, the one most points are on (a
        # point on a mixed grid, 1/21 or 1/15, counts for each it divides by); the others are snapped
        k = max(TUPLETS, key=lambda c: (sum(x == c for x in kinds), sum(x % c == 0 for x in kinds), -c))
        n = 2 ** int(math.log2(k))
        unit = L / (k * TUPLET_SUB)               # the beat's finest unit
        nominal_unit = L / (n * TUPLET_SUB)       # the plain value one unit stands for
        if L / n not in TYPE_OF:
            b = b_next
            continue
        snap = lambda p: b + round((p - b) / unit) * unit  # noqa: E731
        edges = sorted({snap(a) for _, a, _ in seg} | {e_})
        starts = {}
        for el, a, _ in seg:                      # each element from its snapped start to the next one's
            sa = snap(a)
            if sa not in starts or isinstance(starts[sa], m21.note.Rest):
                starts[sa] = el
        new = []
        for i, sa in enumerate(sorted(starts)):
            if sa >= e_:
                continue
            nxt = next((x for x in edges if x > sa), e_)
            j = round((nxt - sa) / unit)
            pieces = []
            while j > 0:
                w = next(x for x in WRITABLE if x <= j)
                pieces.append(w)
                j -= w
            new.append((starts[sa], sa, pieces))
        for el, _, _ in seg:
            v.remove(el)
        made = []
        for el, sa, pieces in new:
            ties = {p.pitch.midi: (p.tie.type if p.tie is not None else None) for p in
                    (el.notes if isinstance(el, m21.chord.Chord) else [el])} if not el.isRest else {}
            pos = sa
            for pi, w in enumerate(pieces):
                c = copy.deepcopy(el)
                d = m21.duration.Duration(float(w * nominal_unit))
                d.appendTuplet(m21.duration.Tuplet(k, n, durationActual=TYPE_OF[L / n], durationNormal=TYPE_OF[L / n]))
                c.duration = d
                if not el.isRest:
                    for note in (c.notes if isinstance(c, m21.chord.Chord) else [c]):
                        t = _tie_pieces(ties.get(note.pitch.midi), len(pieces))[pi]
                        note.tie = m21.tie.Tie(t) if t else None
                    if pi > 0:   # an arpeggio or ornament belongs to the first piece only
                        c.expressions = []
                v.insert(float(pos), c)
                made.append(c)
                pos += w * unit
        if made:
            made[0].duration.tuplets[0].type = "start" if len(made) > 1 else "startStop"
            if len(made) > 1:
                made[-1].duration.tuplets[0].type = "stop"
        rewritten += 1
        els = sorted([e for e in v.notesAndRests if not e.duration.isGrace], key=lambda e: e.offset)
        b = b_next
    return rewritten


def _standard_rests(v):
    """Rests music21 wrote with two or more dots (a 3.875-quarter rest as a quadruple-dotted half)
    split into standard rests; others (single-dotted, tuplet) stay."""
    for r in [e for e in v.notesAndRests if isinstance(e, m21.note.Rest)]:
        if r.duration.dots < 2 or r.duration.tuplets:
            continue
        off, ql = Fraction(r.offset).limit_denominator(10 ** 6), Fraction(r.quarterLength).limit_denominator(10 ** 6)
        if any(x.denominator & (x.denominator - 1) for x in (off, ql)):  # off the binary grid (after a tuplet)
            continue
        parts = _split_rest(off, ql)
        if len(parts) < 2:
            continue
        v.remove(r)
        for p, d in parts:
            nr = m21.note.Rest()
            nr.duration = m21.duration.Duration(d)
            v.insert(float(p), nr)


def _mean_pitch(notes) -> float:
    return sum(n.pitch for n in notes) / len(notes)


def _one_key_one_note(notes: list[_Note]) -> list[_Note]:
    """A staff's notes in one bar with no pitch sounding twice: a pitch struck again while it is
    held ends where it is struck again (one key cannot sound twice), and the same pitch twice at
    once is one note: the one struck there rather than one tied on from before (a note held over
    the bar line and struck again on the downbeat), else the longer. Grace notes are left alone."""
    out, last = [], {}
    for n in sorted(notes, key=lambda n: (n.pos, n.tie, -n.dur)):
        if n.grace:
            out.append(n)
            continue
        prev = last.get(n.pitch)
        if prev is not None and prev.pos == n.pos:
            continue                                  # a doubled unison: the longer one stays
        if prev is not None and prev.pos + prev.dur > n.pos:
            prev.dur = n.pos - prev.pos
        last[n.pitch] = n
        out.append(n)
    return out


def _grid_kind(x) -> int:
    """The rhythm grid a time point (quarters) is on: 0 on the beat, 1 straight (halves, quarters,
    ... of the beat), else the odd part of its denominator (3: triplets, 5: quintuplets)."""
    f = Fraction(x) % 1
    if f == 0:
        return 0
    d = f.denominator
    while d % 2 == 0:
        d //= 2
    return d


def _beat_kinds(points) -> dict:
    """{beat: {grid kinds}} of these time points (on-the-beat points count for no beat)."""
    out = defaultdict(set)
    for x in points:
        k = _grid_kind(x)
        if k:
            out[int(Fraction(x) // 1)].add(k)
    return out


def _two_voices(notes: list[_Note]) -> list[list[tuple[Fraction, Fraction, list[_Note]]]]:
    """A staff's non-grace notes in one bar -> at most two voices, the upper one first, each a list
    of (pos, dur, notes) that follow one another (a chord per position, rests between them).

    The notes are taken by position, the higher chord first at the same position. A chord goes into
    a voice that is free (its last chord has ended): of two, the one nearer in pitch, unless that
    would put it on the wrong side of the other voice's sounding notes (the upper voice's lowest
    note below the lower voice's highest, or the other way round). A second voice is opened for a
    chord wholly above or below the notes held in the first; one that starts among them (a note
    between two held ones) joins them instead. When no free voice fits, it shares a voice with
    the notes still sounding there (the nearer in pitch): that voice is then written as chords at
    every onset and end, the held notes split and tied (``tie``), so a staff seldom needs a third
    voice and its stems seldom cross. Engravers write two voices to a staff at most in piano music.

    One rhythm grid per beat in a voice: a chord whose onset or end is on another grid than the
    voice's notes in that beat (a straight eighth among triplets, a triplet among quintuplets) goes
    into the other voice if that one fits, else into a third voice. Mixed in one voice, the cuts make
    tuplets no notation program can group (triplet sixteenths around a straight onset, 15:8, 12:7):
    MuseScore rejected those measures as corrupted (the user's song, bars 64 and 65)."""
    groups = defaultdict(list)
    for n in notes:
        groups[(n.pos, n.dur)].append(n)
    order = sorted(groups.items(), key=lambda kv: (kv[0][0], -max(x.pitch for x in kv[1])))
    voices: list[dict] = []   # {"notes": [...], "end": Fraction}

    def sounding(v, pos):   # the voice's notes sounding at pos (started there or held over it)
        return [x for x in v["notes"] if x.pos <= pos < x.pos + x.dur]

    def height(v):
        return _mean_pitch(v["notes"])

    def fits(v, pos, end):   # the chord's onset and end on the grid of the voice's notes in their beats
        g = _beat_kinds((pos, end))
        return all(len(v["kinds"].get(b, set()) | ks) <= 1 for b, ks in g.items())

    def new_voice(group, pos, end):
        voices.append({"notes": list(group), "end": end, "kinds": _beat_kinds((pos, end))})
    for (pos, dur), group in order:
        m = _mean_pitch(group)
        if not voices:
            new_voice(group, pos, pos + dur)
            continue
        main = voices[:2]
        free = [v for v in main if v["end"] <= pos]
        busy = [v for v in main if v["end"] > pos]
        target = None
        if len(main) == 1:   # None: a second voice, when the chord is above or below the notes held
            s = sounding(main[0], pos)
            inside = bool(s) and not (min(x.pitch for x in group) > max(x.pitch for x in s)
                                      or max(x.pitch for x in group) < min(x.pitch for x in s))
            target = main[0] if free or inside else None
        elif len(free) == 2:
            target = min(free, key=lambda v: abs(_mean_pitch(v["notes"][-1:]) - m))
        elif len(free) == 1:
            f, b = free[0], busy[0]
            s = sounding(b, pos)
            upper = height(f) >= height(b)
            lo, hi = min(x.pitch for x in group), max(x.pitch for x in group)
            # in the upper voice its lowest note below the lower voice's highest, or the other way round
            crosses = bool(s) and ((upper and lo < max(x.pitch for x in s)) or (not upper and hi > min(x.pitch for x in s)))
            target = b if crosses else f
        else:
            target = min(main, key=lambda v: min(abs(x.pitch - m) for x in sounding(v, pos) or v["notes"]))
        if target is not None and not fits(target, pos, pos + dur):   # another grid in that beat
            pick = target
            alt = [v for v in main if v is not target and fits(v, pos, pos + dur)]
            target = alt[0] if alt else None
            if target is None and len(main) == 2:   # neither fits: a free third voice of its grid, else a new one
                extra = [v for v in voices[2:] if v["end"] <= pos and fits(v, pos, pos + dur)]
                target = extra[0] if extra else None
                if target is None and len(voices) >= MAX_VOICES:   # no more voices: it shares the nearer one
                    target = pick   # (_beat_tuplets writes the beat on one tuplet grid, straight points exactly)
        if target is None:
            new_voice(group, pos, pos + dur)
        else:
            target["notes"].extend(group)
            target["end"] = max(target["end"], pos + dur)
            for b, ks in _beat_kinds((pos, pos + dur)).items():
                target["kinds"].setdefault(b, set()).update(ks)
    voices.sort(key=height, reverse=True)
    out = []
    for v in voices:   # chords at every onset and end in the voice; a note held across one is tied on
        cuts = sorted({x.pos for x in v["notes"]} | {x.pos + x.dur for x in v["notes"]})
        parts = defaultdict(list)
        for x in v["notes"]:
            inner = [c for c in cuts if x.pos < c < x.pos + x.dur]
            # a piece on a tuplet grid (an end off the plain grid) that crosses a beat line is cut there
            # too, so every tuplet stays inside its beat: an eighth from the last triplet of a beat would
            # otherwise be split by music21 itself, which then splits the next notes to complete tuplets
            # (12:7, triplet sixteenths) and MuseScore rejects the measure
            edges = [x.pos] + inner + [x.pos + x.dur]
            extra = []
            for a, b in zip(edges, edges[1:]):
                if _grid_kind(a) > 1 or _grid_kind(b) > 1:
                    extra += [Fraction(i) for i in range(int(a // 1) + 1, int(-(-b // 1))) if a < i < b]
            inner = sorted(set(inner) | set(extra))
            starts = [x.pos] + inner
            ends = inner + [x.pos + x.dur]
            for k, (a, b) in enumerate(zip(starts, ends)):
                parts[(a, b - a)].append(x if len(starts) == 1 else _Note(
                    a, x.staff, x.pitch, b - a, False, x.tie if k == 0 else True,
                    x.arp if k == 0 else False, list(x.orns) if k == 0 else []))
        out.append(sorted(((p, d, g) for (p, d), g in parts.items()), key=lambda t: t[0]))
    return out


def bars_to_score(bars: list[_Bar], min_staves: int = 2, grid: int | None = None,
                  clip: bool = False, two_voices: bool = False) -> m21.stream.Score:
    """``tokens_to_score`` from parsed bars (changed in place by ``grid`` / ``clip``). Notes past
    their bar's end are carried into the next bar at the same time (``_carry_overflow``) unless
    ``clip``. ``two_voices``: a staff gets two voices at most, the upper one first, with no pitch
    sounding twice (``_two_voices``, ``_one_key_one_note``); without, a note goes into the first
    voice free when it starts, however many that takes."""
    if grid:
        bars = _regrid(bars, grid)
    bars = _clip_to_bars(bars) if clip else _carry_overflow(bars)
    n_staves = max([min_staves] + [n.staff + 1 for b in bars for n in b.notes])
    score = m21.stream.Score()
    parts = [m21.stream.PartStaff(id=f"P1-Staff{s + 1}") for s in range(n_staves)]
    last_el = defaultdict(dict)  # (staff) -> pitch -> last element, to attach tie starts
    prev_ts = prev_key = None
    for bi, bar in enumerate(bars):
        _attach_orphan_graces(bar)
        ts = m21.meter.TimeSignature(bar.ts)
        bar_len = Fraction(ts.barDuration.quarterLength)
        content_end = max([n.pos + n.dur for n in bar.notes] + [bar_len])
        for s, part in enumerate(parts):
            meas = m21.stream.Measure(number=bi + 1)
            # every measure carries its time signature while it is built: without one, music21's rest
            # filling (makeRests -> barDuration) guesses one from the measure's contents, which takes
            # seconds per measure for dense or tuplet-grid content; a repeated one is removed below
            meas.timeSignature = m21.meter.TimeSignature(bar.ts)
            if bar.key != prev_key:
                meas.keySignature = m21.key.KeySignature(bar.key)
            if bi == 0:
                meas.clef = m21.clef.TrebleClef() if s == 0 else m21.clef.BassClef()
            groups = defaultdict(list)
            run = defaultdict(int)  # grace notes before one position: a run, each its own note, in order
            staff_notes = [n for n in bar.notes if n.staff == s]
            plan = None
            if two_voices:   # (pos, grace?, voice, run index, notes), in the order they are written
                staff_notes = _one_key_one_note(staff_notes)
                vs = _two_voices([n for n in staff_notes if not n.grace])
                plan = [(p, False, vi, 0, g) for vi, v in enumerate(vs) for p, d, g in v]
                for n in staff_notes:
                    if n.grace:   # into the voice whose chord at that position is nearest in pitch
                        here = [(abs(_mean_pitch(g) - n.pitch), vi) for p, gr, vi, _, g in plan
                                if not gr and p == n.pos]
                        vi = min(here)[1] if here else 0
                        plan.append((n.pos, True, vi, run[n.pos], [n]))
                        run[n.pos] += 1
                plan.sort(key=lambda t: (t[0], not t[1], t[2], t[3]))
                staff_notes = []
            for n in staff_notes:
                if n.grace:
                    groups[(n.pos, True, n.dur, run[n.pos])].append(n)
                    run[n.pos] += 1
                    continue
                k = 0  # a doubled unison (same pitch twice) goes into a separate element/voice
                while any(x.pitch == n.pitch for x in groups[(n.pos, n.grace, n.dur, k)]):
                    k += 1
                groups[(n.pos, n.grace, n.dur, k)].append(n)
            voices: list[list] = []  # each: list of (pos, end, element)
            if plan is not None:
                items = [((pos, grace, g[0].dur, k), g, vi) for pos, grace, vi, k, g in plan]
                voices = [[] for _ in range(1 + max((vi for *_, vi in items), default=-1))]
            else:
                items = [(key, g, None) for key, g in sorted(groups.items(),
                                                           key=lambda kv: (kv[0][0], not kv[0][1], kv[0][3]))]
            for (pos, grace, dur, _), group, fixed_voice in items:
                el, tied = _make_element(group, bar.key, run=run[pos] if grace else 0)
                for p in tied:  # continuation of a tie from an earlier note
                    prev = last_el[s].get(p)
                    if prev is not None:
                        prev_note = prev if isinstance(prev, m21.note.Note) else next(
                            (x for x in prev.notes if x.pitch.midi == p), None)
                        if prev_note is not None:
                            prev_note.tie = m21.tie.Tie("start" if prev_note.tie is None else "continue")
                    target = el if isinstance(el, m21.note.Note) else next(x for x in el.notes if x.pitch.midi == p)
                    target.tie = m21.tie.Tie("stop")
                end = pos + dur  # a grace note: pos (no length); its main note then joins its voice
                if grace:
                    end = pos
                if fixed_voice is not None:
                    voices[fixed_voice].append((pos, end, el))
                    if not grace:
                        for p in {n.pitch for n in group}:
                            last_el[s][p] = el
                    continue
                for v in voices:
                    # a voice free at this position; also for a grace note: put into a voice still
                    # sounding a longer note, it overlapped that note and shifted everything the
                    # voice wrote after it in the MusicXML
                    if v[-1][1] <= pos:
                        v.append((pos, end, el))
                        break
                else:
                    voices.append([(pos, end, el)])
                if not grace:
                    for p in {n.pitch for n in group}:
                        last_el[s][p] = el
            voices = [v for v in voices if v]
            if not voices:
                r = m21.note.Rest()
                r.duration = m21.duration.Duration(bar_len)
                meas.insert(0, r)
            for vi, v in enumerate(voices):
                voice = m21.stream.Voice(id=str(vi + 1)) if len(voices) > 1 else meas
                for pos, _, el in v:
                    voice.insert(float(pos), el)
                if voice is not meas:
                    meas.insert(0, voice)
            for v in (meas.voices or [meas]):
                v.makeRests(fillGaps=True, refStreamOrTimeRange=(0, float(content_end)), inPlace=True,
                            hideRests=False)
                # slivers left by snapping pos & dur to different grids: absorb into the previous element
                els = sorted(v.notesAndRests, key=lambda e: e.offset)
                for k, r in enumerate(els):
                    if isinstance(r, m21.note.Rest) and r.quarterLength < MIN_REST:
                        prev = next((e for e in reversed(els[:k]) if not e.duration.isGrace
                                     and e.offset + e.quarterLength <= r.offset + 1e-9), None)
                        if prev is not None:
                            prev.quarterLength = prev.quarterLength + r.quarterLength
                        v.remove(r)
                _standard_rests(v)
                if two_voices:
                    _grid_rests(v)
                    _beat_tuplets(v, Fraction(bar_len))
            if bar.ts == prev_ts:  # written only where it changes
                meas.timeSignature = None
            part.append(meas)
        prev_ts, prev_key = bar.ts, bar.key
    for p in parts:
        score.insert(0, p)
    score.insert(0, m21.layout.StaffGroup(parts, symbol="brace"))
    return score


def _fix_measure(meas: m21.stream.Measure):
    """Make a measure exportable: split 'complex' rests into expressible pieces, and drop gap
    rests MusicXML cannot express at all (e.g. 2/45 quarter between a septuplet and a quintuplet),
    letting the previous note sound through the gap instead."""
    for v in (meas.voices or [meas]):
        els = sorted(v.notesAndRests, key=lambda e: e.offset)
        for k, r in enumerate(els):
            if not isinstance(r, m21.note.Rest):
                continue
            if r.duration.type == "complex":
                parts = r.splitAtDurations()
                if all(p.duration.type not in ("complex", "inexpressible") for p in parts):
                    off = r.offset
                    v.remove(r)
                    for p in parts:
                        v.insert(off, p)
                        off += p.quarterLength
                    continue
            if r.duration.type in ("complex", "inexpressible"):
                v.remove(r)


def write_musicxml(score: m21.stream.Score, path: str, decorate=None) -> str:
    """Write MusicXML; repair measures with durations MusicXML can't express and retry once.
    ``decorate(score)``: called on the score first (e.g. src/glissando.py decorate: lines the
    tokens cannot say)."""
    if decorate is not None:
        decorate(score)
    try:
        return score.write("musicxml", fp=str(path))
    except m21.musicxml.xmlObjects.MusicXMLExportException:
        for meas in score.recurse().getElementsByClass(m21.stream.Measure):
            _fix_measure(meas)
        return score.write("musicxml", fp=str(path))


def _exports(bars: list[_Bar], **kw) -> bool:
    """Whether these bars alone make a MusicXML file (written to a temporary file, then removed)."""
    fd, tmp = tempfile.mkstemp(suffix=".musicxml")
    os.close(fd)
    try:
        write_musicxml(bars_to_score(copy.deepcopy(bars), **kw), tmp)
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _fix_bars(bars: list[_Bar], fallback_grid: int, two_voices: bool = False) -> tuple[list[_Bar], dict]:
    """Only the bars that cannot be exported exactly on their own, snapped to 1/``fallback_grid``
    quarter, clipped to the bar, or both (the first that exports): every other bar stays exact.
    Returns the bars (copies) and {bar index: "grid" | "clip" | "clip+grid" | "empty"}."""
    out, fixed = copy.deepcopy(bars), {}
    tv = {"two_voices": two_voices}
    levels = (("grid", {"grid": fallback_grid}), ("clip", {"clip": True}),
              ("clip+grid", {"clip": True, "grid": fallback_grid}))
    for j, b in enumerate(bars):
        if _exports([b], **tv):
            continue
        for name, kw in levels:
            if _exports([b], **kw, **tv):
                one = copy.deepcopy(b)
                if "grid" in kw:
                    _regrid([one], fallback_grid)
                if "clip" in kw:
                    _clip_to_bars([one])
                out[j], fixed[j] = one, name
                break
        else:
            out[j].notes, out[j].pedal = [], []
            fixed[j] = "empty"
    return out, fixed


def tokens_to_musicxml(tokens: list[str], path: str, fallback_grid: int = 48, report: dict | None = None,
                       decorate=None, two_voices: bool = False) -> str:
    """Tokens -> MusicXML file, as exact as the notation allows. Exact rendering first. If that
    fails (mixed tuplet grids making some duration inexpressible, a zero-length note, overfull
    model output), only the bars that fail on their own are snapped to 1/``fallback_grid``
    quarter and / or clipped to their bar (``_fix_bars``), the others staying exact. If the
    score still fails, the whole score is snapped, clipped, or both. Each attempt rebuilds the
    score: a failed export can leave it half-modified. Last resort: the bars that cannot be
    exported even on their own (clipped and snapped) are emptied, so the rest of the score is
    still written. ``report`` (a dict) gets "attempt" (0: exact; 1: per bar, "fixed_bars" saying
    which bars and how; 2-4: the whole score snapped / clipped / both; 5: bars emptied) and
    "emptied_bars" (their indices). ``decorate``: see write_musicxml (applied to every attempt's score).
    ``two_voices``: see bars_to_score (every attempt)."""
    report = {} if report is None else report
    report["emptied_bars"] = []

    def write(score):   # write_musicxml looked up at call time (tests stand in for it); decorate only if given
        return write_musicxml(score, path) if decorate is None else write_musicxml(score, path, decorate)
    tv = {"two_voices": two_voices}
    try:
        out = write(tokens_to_score(tokens, **tv))
        report["attempt"] = 0
        return out
    except Exception as e:  # noqa: BLE001
        last = e
    fixed_bars, fixed = _fix_bars(parse_tokens(tokens), fallback_grid, two_voices)
    try:
        out = write(bars_to_score(fixed_bars, **tv))
        report.update(attempt=1, fixed_bars={int(k): v for k, v in fixed.items()},
                      emptied_bars=sorted(k for k, v in fixed.items() if v == "empty"))
        return out
    except Exception as e:  # noqa: BLE001
        last = e
    attempts = [{"grid": fallback_grid, **tv}, {"clip": True, **tv}, {"clip": True, "grid": fallback_grid, **tv}]
    for k, kw in enumerate(attempts, start=2):
        try:
            out = write(tokens_to_score(tokens, **kw))
            report["attempt"] = k
            return out
        except Exception as e:  # noqa: BLE001
            last = e
    kw = attempts[-1]
    bars = parse_tokens(tokens)
    bad = [j for j, b in enumerate(bars) if not _exports([b], **kw)]  # broken on their own
    for j in bad:
        bars[j].notes, bars[j].pedal = [], []
    for _ in range(64):  # broken only in context (e.g. a tie from the bar before): the first such bar
        if _exports(bars, **kw):
            break
        lo, hi = 1, len(bars)  # the shortest failing prefix
        while lo < hi:
            mid = (lo + hi) // 2
            if _exports(bars[:mid], **kw):
                lo = mid + 1
            else:
                hi = mid
        j = lo - 1
        if not bars[j].notes and not bars[j].pedal:
            break
        bars[j].notes, bars[j].pedal = [], []
        bad.append(j)
    try:
        out = write(bars_to_score(bars, **kw))
    except Exception:  # noqa: BLE001
        raise last
    report.update(attempt=5, emptied_bars=sorted(bad))
    return out
