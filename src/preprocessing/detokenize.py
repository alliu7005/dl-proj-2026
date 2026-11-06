"""Inverse of ``score.tokenize_bars``: token sequence -> music21 Score -> MusicXML.

    tokens_to_musicxml(tokens, "out.musicxml")   # tokens -> file, with export fallbacks

Notes with the same (staff, position, duration) become chords; overlapping chords inside one staff
are spread over voices; gaps are filled with rests. Pedal tokens are parsed (``_Bar.pedal``) but
not rendered into the MusicXML yet; evaluate pedal precision/recall on tokens directly.
"""
from __future__ import annotations

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


def _make_element(group: list[_Note]):
    pitches = sorted({n.pitch for n in group})
    el = m21.note.Note(pitches[0]) if len(pitches) == 1 else m21.chord.Chord(pitches)
    if group[0].grace:
        el.duration = m21.duration.Duration(0.5).getGraceDuration()
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


def tokens_to_score(tokens: list[str], min_staves: int = 2, grid: int | None = None) -> m21.stream.Score:
    """``grid``: optionally snap positions/durations to 1/grid quarter (fallback for export)."""
    bars = parse_tokens(tokens)
    if grid:
        bars = _regrid(bars, grid)
    n_staves = max([min_staves] + [n.staff + 1 for b in bars for n in b.notes])
    score = m21.stream.Score()
    parts = [m21.stream.PartStaff(id=f"P1-Staff{s + 1}") for s in range(n_staves)]
    last_el = defaultdict(dict)  # (staff) -> pitch -> last element, to attach tie starts
    prev_ts = prev_key = None
    for bi, bar in enumerate(bars):
        ts = m21.meter.TimeSignature(bar.ts)
        bar_len = Fraction(ts.barDuration.quarterLength)
        content_end = max([n.pos + n.dur for n in bar.notes] + [bar_len])
        for s, part in enumerate(parts):
            meas = m21.stream.Measure(number=bi + 1)
            if bar.ts != prev_ts:
                meas.timeSignature = m21.meter.TimeSignature(bar.ts)
            if bar.key != prev_key:
                meas.keySignature = m21.key.KeySignature(bar.key)
            if bi == 0:
                meas.clef = m21.clef.TrebleClef() if s == 0 else m21.clef.BassClef()
            groups = defaultdict(list)
            for n in bar.notes:
                if n.staff == s:
                    k = 0  # a doubled unison (same pitch twice) goes into a separate element/voice
                    while any(x.pitch == n.pitch for x in groups[(n.pos, n.grace, n.dur, k)]):
                        k += 1
                    groups[(n.pos, n.grace, n.dur, k)].append(n)
            voices: list[list] = []  # each: list of (pos, end, element)
            for (pos, grace, dur, _), group in sorted(groups.items(), key=lambda kv: (kv[0][0], not kv[0][1], kv[0][3])):
                el, tied = _make_element(group)
                for p in tied:  # continuation of a tie from an earlier note
                    prev = last_el[s].get(p)
                    if prev is not None:
                        prev_note = prev if isinstance(prev, m21.note.Note) else next(
                            (x for x in prev.notes if x.pitch.midi == p), None)
                        if prev_note is not None:
                            prev_note.tie = m21.tie.Tie("start" if prev_note.tie is None else "continue")
                    target = el if isinstance(el, m21.note.Note) else next(x for x in el.notes if x.pitch.midi == p)
                    target.tie = m21.tie.Tie("stop")
                end = pos + dur
                for v in voices:
                    if grace or v[-1][1] <= pos:
                        v.append((pos, end, el))
                        break
                else:
                    voices.append([(pos, end, el)])
                if not grace:
                    for p in {n.pitch for n in group}:
                        last_el[s][p] = el
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


def write_musicxml(score: m21.stream.Score, path: str) -> str:
    """Write MusicXML; repair measures with durations MusicXML can't express and retry once."""
    try:
        return score.write("musicxml", fp=str(path))
    except m21.musicxml.xmlObjects.MusicXMLExportException:
        for meas in score.recurse().getElementsByClass(m21.stream.Measure):
            _fix_measure(meas)
        return score.write("musicxml", fp=str(path))


def tokens_to_musicxml(tokens: list[str], path: str, fallback_grid: int = 48) -> str:
    """Tokens -> MusicXML file. Exact rendering first; if mixed tuplet grids make some duration
    inexpressible, re-render with everything snapped to 1/``fallback_grid`` quarter."""
    try:
        return write_musicxml(tokens_to_score(tokens), path)
    except Exception:  # noqa: BLE001  (a failed export can leave the score half-modified)
        return write_musicxml(tokens_to_score(tokens, grid=fallback_grid), path)
