"""The written score's tempo, from the meter model's grid (v2).

The grid holds, for every tracked beat, its time in the performance (seconds) and its place in the
score (quarters), so it knows the tempo the piece was played at, beat by beat. The score's tokens
carry no tempo, and before this the score MIDI played the whole piece at one tempo (src/pipeline.py
playback_qpm: the score's length over the recording's), which evened out every ritardando and every
section at another speed, and drifted away from the recording through them. ``score_tempo`` gives:

  * the smoothed performer (``played_tempi``), which the printed tempo is read from: each bar's own
    tempo (its quarters over the seconds the performer took),
    smoothed with a running median of ``PLAY_SMOOTH`` bars each side so that a single odd bar (a
    grid error more often than the pianist) is not played as a lurch, and pulled back towards the
    recording by ``CATCH_UP`` of the gap each bar so it does not drift away from it. Within a bar
    it is even: the tracker's beat-to-beat jitter is not played;
  * the tempo marks printed in the MusicXML, from those played tempos: one at the start, and a new
    one only where the tempo changes and stays changed (``CHANGE`` for at least ``RUN`` bars), each
    the median of its section, in the beat unit of its time signature (a dotted quarter in 6/8, a
    half in 2/2), starting at the first bar played at the new tempo;
  * the shorter changes the playback makes, printed as words (``tempo_words``): "rit." where it
    slows more than ``WORD_CHANGE`` below the mark in force, "accel." where it speeds up as much,
    "a tempo" where it comes back;
  * a tempo for every bar, for playing the score back (score.mid), which plays the page
    (``printed_tempi``): each mark held steady, and between a "rit." / "accel." and its "a tempo"
    the smoothed performer's tempo. Everything the playback does is printed, and the reverse. Within
    a section it drifts from the recording by the performer's rubato (under ``WORD_CHANGE``);
  * ``follow``: the score MIDI's clock mapped to the score, for the demo's playback bar.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

MIN_QPM, MAX_QPM = 20.0, 400.0   # a bar's playback tempo is kept within these (quarters per minute)
CHANGE = math.log(1.15)          # a printed mark changes when the tempo moves by more than 15%...
RUN = 4                          # ...for at least this many bars
SMOOTH = 2                       # bars each side of the running median the change is judged on
PLAY_SMOOTH = 1                  # bars each side of the running median the playback tempo takes
CATCH_UP = 0.35                  # share of the playback's lead / lag on the recording made up each bar
WORD_CHANGE = math.log(1.10)     # "rit." / "accel." where the playback is this far off its mark...
WORD_BACK = math.log(1.05)       # ...and "a tempo" once it is back within this


def bar_quarters(ts: str) -> float:
    try:
        n, d = ts.split("/")
        return float(Fraction(int(n) * 4, int(d)))
    except (ValueError, ZeroDivisionError):
        return 4.0


def beat_unit(ts: str) -> float:
    """The note a tempo mark counts in this time signature, in quarters: a dotted quarter in 6/8,
    9/8 and 12/8, a dotted eighth in 6/16..12/16, a half in x/2, an eighth in the other x/8, else a
    quarter."""
    try:
        n, d = (int(x) for x in ts.split("/"))
    except ValueError:
        return 1.0
    compound = n in (6, 9, 12)
    if d == 8:
        return 1.5 if compound else 0.5
    if d == 16:
        return 0.75 if compound else 0.25
    if d == 2:
        return 2.0
    return 1.0


def _clock(grid: dict):
    """The grid's (quarters, seconds) anchors, both strictly increasing; None with fewer than 2."""
    t = np.asarray(grid.get("t", []), np.float64)
    q = np.asarray(grid.get("q", []), np.float64)
    if len(t) < 2 or len(q) != len(t):
        return None
    order = np.argsort(t, kind="stable")
    t, q = t[order], q[order]
    keep = [0]
    for i in range(1, len(t)):
        if t[i] > t[keep[-1]] + 1e-6 and q[i] > q[keep[-1]] + 1e-9:
            keep.append(i)
    if len(keep) < 2:
        return None
    return q[keep], t[keep]


def seconds_at(x, q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The performance's time at score positions ``x`` (quarters on the grid's clock): between the
    anchors around it, else extended with the first / last anchors' tempo."""
    x = np.asarray(x, np.float64)
    out = np.interp(x, q, t)
    lo, hi = x < q[0], x > q[-1]
    out[lo] = t[0] + (x[lo] - q[0]) * (t[1] - t[0]) / (q[1] - q[0])
    out[hi] = t[-1] + (x[hi] - q[-1]) * (t[-1] - t[-2]) / (q[-1] - q[-2])
    return out


def bar_tempi(grid: dict, origin: float, bar_ts: list[str]) -> list[dict] | None:
    """The playback tempo of each of the score's bars: [{"bar": k, "offset": its start in quarters
    from the score's start (``origin`` on the grid's clock: its first bar line, from which a music21
    score counts its offsets, a pickup's notes included), "qpm": its quarters over the seconds the
    performance took for them}]. None without a usable grid."""
    clock = _clock(grid)
    bar_q = np.asarray(grid.get("bar_q", []), np.float64)
    if clock is None or not len(bar_q) or not bar_ts:
        return None
    q, t = clock
    out = []
    for k, ts in enumerate(bar_ts):
        if k >= len(bar_q):
            break
        a = max(float(bar_q[k]), origin)
        b = float(bar_q[k + 1]) if k + 1 < len(bar_q) else float(bar_q[k]) + bar_quarters(ts)
        if b <= a + 1e-9:
            continue
        sa, sb = seconds_at([a, b], q, t)
        qpm = 60.0 * (b - a) / (sb - sa) if sb > sa else (out[-1]["qpm"] if out else 120.0)
        out.append({"bar": k, "offset": round(a - origin, 6), "quarters": b - a,
                    "qpm": float(np.clip(qpm, MIN_QPM, MAX_QPM))})
    return out or None


def _running_median(x: np.ndarray, half: int) -> np.ndarray:
    """Running median, ``half`` values each side, the ends padded with their own value."""
    if half <= 0 or len(x) < 3:
        return x.copy()
    p = np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])])
    return np.array([np.median(p[i: i + 2 * half + 1]) for i in range(len(x))])


def played_tempi(tempi: list[dict], smooth: int = PLAY_SMOOTH, catch_up: float = CATCH_UP) -> list[dict]:
    """The tempo each bar is played back at: the bars' own tempos (``bar_tempi``, the performer's)
    through a running median (``smooth`` bars each side), then each bar's length moved by
    ``catch_up`` of the gap between where the playback would end the bar and where the recording
    does, so the playback stays near the recording (a bar the median changed puts it ahead or
    behind; the gap closes over the next few bars). Same layout as ``bar_tempi``."""
    if not tempi:
        return []
    lt = np.log([b["qpm"] for b in tempi])
    sm = _running_median(lt, smooth)
    out, played, heard = [], 0.0, 0.0      # seconds of the playback / of the recording so far
    for b, s in zip(tempi, sm):
        quarters = b["quarters"]
        heard += quarters * 60.0 / b["qpm"]
        d = quarters * 60.0 / math.exp(s)
        d += catch_up * (heard - (played + d))
        qpm = float(np.clip(60.0 * quarters / d, MIN_QPM, MAX_QPM)) if d > 0 else b["qpm"]
        played += quarters * 60.0 / qpm
        out.append({**b, "qpm": qpm})
    return out


def _round_mark(value: float) -> int:
    """A metronome number: to the nearest 2 from 60 up, else to the nearest 1."""
    return int(2 * round(value / 2)) if value >= 60 else max(1, int(round(value)))


def tempo_marks(tempi: list[dict], bar_ts: list[str], change: float = CHANGE, run: int = RUN,
                smooth: int = SMOOTH) -> list[dict]:
    """The printed marks for these bar tempos: [{"bar", "offset", "qpm", "unit" (quarters),
    "value" (the metronome number in that unit)}]. The log tempos are smoothed with a running median
    (``smooth`` bars each side); a new section starts where the smoothed tempo stays more than
    ``change`` away from the current section's for ``run`` bars, all on the same side. Each section's
    mark is the median of its bars' tempos; a section whose mark reads the same as the one before it
    is merged into it. A section starts at the first bar of its run that is nearer the new tempo than
    the old one: the running median can turn a bar or two early next to odd bars."""
    if not tempi:
        return []
    lt = np.log([b["qpm"] for b in tempi])
    n = len(lt)
    s = np.array([np.median(lt[max(0, i - smooth): i + smooth + 1]) for i in range(n)])
    starts, cur, i = [0], float(np.median(s[:run])), 1
    while i < n:
        d = s[i: i + run] - cur
        if len(d) == run and (np.all(d > change) or np.all(d < -change)):
            new = float(np.median(s[i: i + run]))
            k = i
            while k < min(i + run, n) - 1 and abs(lt[k] - new) >= abs(lt[k] - cur):
                k += 1
            starts.append(k)
            cur = new
            i = k + run
        else:
            i += 1
    marks = []
    for j, a in enumerate(starts):
        b = starts[j + 1] if j + 1 < len(starts) else n
        qpm = float(np.exp(np.median(lt[a:b])))
        k = tempi[a]["bar"]
        unit = beat_unit(bar_ts[k] if k < len(bar_ts) else bar_ts[-1])
        value = _round_mark(qpm / unit)
        if marks and marks[-1]["unit"] == unit and marks[-1]["value"] == value:
            continue
        marks.append({"bar": k, "offset": tempi[a]["offset"], "qpm": round(qpm, 2), "unit": unit, "value": value})
    return marks


def tempo_words(tempi: list[dict], marks: list[dict], change: float = WORD_CHANGE,
                back: float = WORD_BACK) -> list[dict]:
    """The playback's changes the marks do not show, as words: [{"bar", "offset", "text"}].
    Against the mark in force: "rit." where a bar plays more than ``change`` slower, "accel." more
    than ``change`` faster, "a tempo" at the first bar back within ``back`` of it. A new mark ends
    any of them (no "a tempo" needed)."""
    if not tempi or not marks:
        return []
    starts = {m["bar"]: m["qpm"] for m in marks}
    words, ref, state = [], marks[0]["qpm"], 0     # state: -1 slower, 0 at the mark, +1 faster
    for b in tempi:
        if b["bar"] in starts:
            ref, state = starts[b["bar"]], 0
        r = math.log(b["qpm"] / ref)
        new = -1 if r < -change else 1 if r > change else (0 if abs(r) <= back else state)
        if new != state:
            text = {-1: "rit.", 1: "accel.", 0: "a tempo"}[new]
            if b["bar"] in starts and new == 0:
                text = None
            if text:
                words.append({"bar": b["bar"], "offset": b["offset"], "text": text})
            state = new
    return words


def printed_tempi(played: list[dict], marks: list[dict], words: list[dict]) -> list[dict]:
    """What the score MIDI plays: the page. Each bar at the mark in force, except from a "rit." or
    "accel." to its "a tempo" (or the next mark), where the bars take their played tempo
    (``played_tempi``, the smoothed performer). Same layout as ``bar_tempi``."""
    if not marks:
        return played
    starts = {m["bar"]: m["qpm"] for m in marks}
    said = {w["bar"]: w["text"] for w in words}
    out, ref, free = [], marks[0]["qpm"], False
    for b in played:
        if b["bar"] in starts:
            ref, free = starts[b["bar"]], False
        if b["bar"] in said:
            free = said[b["bar"]] != "a tempo"
        out.append({**b, "qpm": b["qpm"] if free else ref})
    return out


def follow(tempi: list[dict], lead: float = 0.0) -> dict:
    """The score MIDI's clock -> the score: {"t": seconds of the score MIDI, "q": quarters} at every
    bar line (piecewise linear between them, as the MIDI plays each bar at one tempo), both counted
    from ``lead`` quarters into the score (the first written note: the demo's playback bar counts
    from there)."""
    ts, qs, sec = [], [], 0.0
    for i, b in enumerate(tempi):
        qs.append(b["offset"])
        ts.append(sec)
        end = tempi[i + 1]["offset"] if i + 1 < len(tempi) else None
        if end is not None:
            sec += (end - b["offset"]) * 60.0 / b["qpm"]
    if tempi:   # the last bar's end, so the map runs to the end of the music
        last = tempi[-1]
        qs.append(last["offset"] + 64.0)
        ts.append(sec + 64.0 * 60.0 / last["qpm"])
    t0 = float(np.interp(lead, qs, ts)) if qs else 0.0
    return {"t": [round(x - t0, 4) for x in ts], "q": [round(x - lead, 4) for x in qs]}


def score_tempo(grid: dict, origin: float, bar_ts: list[str], first: float | None = None,
                marks: bool = True) -> dict | None:
    """Everything above for one score: {"bars": what the score MIDI plays (printed_tempi: the
    marks and words, from the smoothed performer's played_tempi; that alone when ``marks`` is
    False), "marks": tempo_marks, "words": tempo_words (both empty when ``marks`` is False),
    "follow": follow of what it plays}. ``origin``: where the score starts on the grid's clock (its first bar line);
    ``first``: its first written note (default ``origin``; later in a pickup bar). None without a
    usable grid."""
    tempi = bar_tempi(grid, origin, bar_ts)
    if not tempi:
        return None
    played = played_tempi(tempi)
    lead = 0.0 if first is None else max(0.0, first - origin)
    mk = tempo_marks(played, bar_ts) if marks else []
    words = tempo_words(played, mk) if marks else []
    heard = printed_tempi(played, mk, words)   # without marks: the smoothed performer
    return {"bars": heard, "marks": mk, "words": words, "follow": follow(heard, lead)}


def qpm_at(tempi: list[dict], offset: float) -> float:
    """The playback tempo at a score offset (quarters from the score's start)."""
    k = 0
    for i, b in enumerate(tempi):
        if b["offset"] <= offset + 1e-9:
            k = i
        else:
            break
    return tempi[k]["qpm"]


def _measures(score):
    import music21 as m21
    if not score.parts:
        return []
    return list(score.parts[0].getElementsByClass(m21.stream.Measure))


def print_marks(score, marks: list[dict], words: list[dict] | None = None) -> int:
    """The marks, and the words (rit., accel., a tempo), into a music21 score (the MusicXML's),
    above the top staff at the start of their bars. Returns how many were placed."""
    import music21 as m21
    ms = _measures(score)
    n = 0
    for m in marks:
        if m["bar"] >= len(ms):
            continue
        mm = m21.tempo.MetronomeMark(number=m["value"], referent=m21.duration.Duration(m["unit"]))
        mm.placement = "above"
        ms[m["bar"]].insert(0, mm)
        n += 1
    for w in words or []:
        if w["bar"] >= len(ms):
            continue
        te = m21.expressions.TextExpression(w["text"])
        te.style.fontStyle = "italic"
        te.placement = "above"
        ms[w["bar"]].insert(0, te)
        n += 1
    return n


def play_bars(score, tempi: list[dict]) -> int:
    """A tempo at the start of every bar of a music21 score (the score MIDI's copy, never the
    MusicXML's): each bar at its tempo in ``tempi`` (played_tempi). Returns how many were set."""
    import music21 as m21
    ms = _measures(score)
    n = 0
    for b in tempi:
        if b["bar"] >= len(ms):
            continue
        ms[b["bar"]].insert(0, m21.tempo.MetronomeMark(number=round(b["qpm"], 3)))
        n += 1
    return n
