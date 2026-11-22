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


def _make_element(group: list[_Note], key: int = 0):
    pitches = sorted({n.pitch for n in group})
    el = (m21.note.Note(spell(pitches[0], key)) if len(pitches) == 1
          else m21.chord.Chord([spell(x, key) for x in pitches]))
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


def _clip_to_bars(bars: list[_Bar]) -> list[_Bar]:
    """Keep every note inside its bar. Model output can place a note past the bar's end (a wrong
    time signature, a missed bar line) or hold it across the bar line; music21's export then tries
    to carry the overflow into the next measure and fails (KeyError on a voice id, or "object
    already found in this Stream"). Notes starting past the end are dropped, longer ones cut at
    the bar line. Only used as an export fallback: some real scores have bars fuller than their
    time signature (cadenzas, irregular measures) and export fine as they are."""
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
                    clip: bool = False) -> m21.stream.Score:
    """``grid``: optionally snap positions/durations to 1/grid quarter (fallback for export).
    ``clip``: keep notes inside their bars (fallback for malformed model output, see _clip_to_bars)."""
    return bars_to_score(parse_tokens(tokens), min_staves, grid, clip)


def bars_to_score(bars: list[_Bar], min_staves: int = 2, grid: int | None = None,
                  clip: bool = False) -> m21.stream.Score:
    """``tokens_to_score`` from parsed bars (changed in place by ``grid`` / ``clip``)."""
    if grid:
        bars = _regrid(bars, grid)
    if clip:
        bars = _clip_to_bars(bars)
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
            # every measure carries its time signature while it is built: without one, music21's rest
            # filling (makeRests -> barDuration) guesses one from the measure's contents, which takes
            # seconds per measure for dense or tuplet-grid content; a repeated one is removed below
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
                el, tied = _make_element(group, bar.key)
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


def write_musicxml(score: m21.stream.Score, path: str) -> str:
    """Write MusicXML; repair measures with durations MusicXML can't express and retry once."""
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


def tokens_to_musicxml(tokens: list[str], path: str, fallback_grid: int = 48, report: dict | None = None) -> str:
    """Tokens -> MusicXML file. Exact rendering first; if mixed tuplet grids make some duration
    inexpressible, re-render with everything snapped to 1/``fallback_grid`` quarter; if notes
    overflowing their bars break the export (model output), clip them to the bar and retry.
    Each attempt rebuilds the score: a failed export can leave it half-modified. Last resort: the
    bars that cannot be exported even on their own (clipped and snapped) are emptied, so the rest
    of the score is still written. ``report`` (a dict) gets "attempt" (0-4: which of these
    worked) and "emptied_bars" (their indices)."""
    attempts = [{}, {"grid": fallback_grid}, {"clip": True}, {"clip": True, "grid": fallback_grid}]
    report = {} if report is None else report
    report["emptied_bars"] = []
    for k, kw in enumerate(attempts):
        try:
            out = write_musicxml(tokens_to_score(tokens, **kw), path)
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
        out = write_musicxml(bars_to_score(bars, **kw), path)
    except Exception:  # noqa: BLE001
        raise last
    report.update(attempt=len(attempts), emptied_bars=sorted(bad))
    return out
