"""Playing a written score back as it is written: what music21's MIDI export leaves out.

music21 writes a score's MIDI note by note from the written durations: a grace note has none, so it
is not played at all, and a chord marked rolled (an arpeggio sign) is struck as a block. Users hear
the score through that MIDI (src/pipeline.py engrave -> score.mid, the demo's "The score, played
back"), so ``realize`` prepares a copy of the score for it (never the MusicXML):
  * grace notes are played just before their main note, ``GRACE_S`` each (the main note stays on
    its beat, so the demo's playback bar still lines up with the score); a grace note at the very
    start of the piece, with no room before it, is played on the beat instead;
  * a rolled chord is spread from its lowest note up, ``ROLL_S`` apart, every note held to the
    chord's end. The chords rolled together on both staves at one moment (the sign drawn through
    both) roll as one, from the bass up into the treble. A chord with a tied note is left as it is.
"""
from __future__ import annotations

from collections import defaultdict

GRACE_S = 0.07      # seconds per grace note
ROLL_S = 0.035      # seconds between the notes of a rolled chord


def _rolled(el) -> bool:
    """A chord, or a single note (one staff's part of a roll drawn through both), marked rolled."""
    import music21 as m21
    return any(isinstance(e, m21.expressions.ArpeggioMark) for e in el.expressions)


def _notes(el) -> list:
    import music21 as m21
    return list(el.notes) if isinstance(el, m21.chord.Chord) else [el]


def realize(score, qpm) -> dict:
    """Change ``score`` (a music21 Score, to be written as MIDI) in place so its MIDI plays the grace
    notes and the rolled chords; ``qpm``: the playback tempo (quarters per minute; 120 if unknown),
    or a function of the score offset giving it there (src/tempo.py: a tempo per bar).
    Returns {"graces_played": n, "rolled_chords_played": n}."""
    import music21 as m21

    def qps(off: float) -> float:                   # quarters per second at this offset
        v = qpm(off) if callable(qpm) else qpm
        return (v or 120.0) / 60.0
    parts = list(score.parts)
    graces, rolls = [], defaultdict(list)
    for k, part in enumerate(parts):
        for el in list(part.recurse().notes):
            if el.duration.isGrace:
                graces.append((k, el, float(el.getOffsetInHierarchy(part))))
            elif _rolled(el) and not any(n.tie is not None for n in _notes(el)):
                rolls[round(float(el.getOffsetInHierarchy(part)), 6)].append((k, el))
    # grace notes: the ones before one main note, in their written order, ending on its beat
    by_main = defaultdict(list)
    for k, el, off in graces:
        by_main[(k, round(off, 6))].append(el)
    n_grace = 0
    for (k, off), els in by_main.items():
        step = min(GRACE_S * qps(off), 0.25)
        start = off - step * len(els)
        if start < 0:
            start = off                              # no room before the piece's first beat
        for i, el in enumerate(els):
            pitches = [p.midi for p in el.pitches]
            new = (m21.note.Note(pitches[0]) if len(pitches) == 1 else m21.chord.Chord(pitches))
            new.quarterLength = step
            el.activeSite.remove(el)
            parts[k].insert(start + i * step, new)
            n_grace += 1
    # rolled chords: everything rolled at one moment, lowest note first
    n_roll = 0
    for off, items in rolls.items():
        notes = sorted(((n.pitch.midi, k, el) for k, el in items for n in _notes(el)), key=lambda x: x[0])
        if len(notes) < 2:   # one note marked rolled: nothing to spread
            continue
        end = off + max(float(el.quarterLength) for _, el in items)
        gap = min(ROLL_S * qps(off), (end - off) / (2 * max(len(notes), 1)))
        for _, el in items:
            el.activeSite.remove(el)
        for i, (pitch, k, el) in enumerate(notes):
            n = m21.note.Note(pitch)
            n.quarterLength = max(off + float(el.quarterLength) - (off + i * gap), gap)
            parts[k].insert(off + i * gap, n)
        n_roll += 1
    return {"graces_played": n_grace, "rolled_chords_played": n_roll}
