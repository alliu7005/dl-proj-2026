"""Synthetic performances of written scores for the meter model (v2): exact grids from scores that
were never performed or aligned (scripts/build_synthetic_meter.py: PDMX's public-domain MusicXML).

Why: the meter model is v2's bottleneck, and the largest gain it has had came from more data
(Gangqinpu's 17k performances, sixth round). Beyer & Dai name the scarcity of labelled data as the
field's limit. A score gives the grid exactly (every beat, bar line and time signature), so
rendering one as a performance gives a training example with no alignment errors, for any meter
the score writes. That includes the meters the real data has few of (3/8, 6/8, 2/2, 6/4, 12/8),
where the meter model confuses note-value levels most.

``render(measures, rng)``: src/preprocessing/score.py measures -> meter-model performances (the
form scripts/build_meter_data.py writes: notes [N, 4] pitch / onset / offset / velocity, and a
ground-truth grid ``gt`` as src/meter.py annotation_grid gives), one per run of at least
``min_bars`` consecutive regular bars (a bar as long as its time signature, the time signature one
of TS_CLASSES; a pickup or an irregular bar ends a run). The performance:
  tempo     a base tempo (log-normal around 100 quarters per minute), a slow random walk of the log
            tempo per beat (rubato) and sometimes a ritardando into the last bars
  timing    every onset off the time map by a few milliseconds (chords spread a little), grace
            notes just before their beat
  lengths   the written length through the time map, shortened by a per-piece articulation
  velocity  a per-piece level, a mild metrical accent of random strength (0 for some pieces, so
            the model cannot lean on it), the upper staff a little louder, noise
Tied notes are joined into one. The notes are clean MIDI (``beats`` None): training adds the
transcription noise and the simulated tracker to them, as to ASAP's clean performances.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .beats import beat_layout
from .meter import IGNORE, TS_CLASSES, bar_quarters


@dataclass
class RenderConfig:
    min_bars: int = 8
    qpm_mu: float = float(np.log(100.0))
    qpm_sigma: float = 0.35
    qpm_range: tuple = (40.0, 200.0)
    rubato: float = 0.03           # std of the log tempo's step per beat
    rubato_keep: float = 0.92      # AR(1) pull of the log tempo back to the base
    ritard_p: float = 0.3          # share of runs ending in a ritardando
    jitter_s: float = 0.012        # onset noise
    chord_spread_s: float = 0.008
    articulation: tuple = (0.55, 1.0)
    accent: tuple = (0.0, 10.0)    # velocity added on a downbeat (half on other beats)


def _runs(measures) -> list[tuple[int, int]]:
    """(first, last + 1) measure indices of every run of regular bars."""
    runs, start = [], None
    for i, m in enumerate(measures):
        ok = (m.time_signature in TS_CLASSES and m.length > 0
              and Fraction(m.length) == bar_quarters(m.time_signature))
        if ok and start is None:
            start = i
        if not ok and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(measures)))
    return runs


def _notes(measures, lo: int, hi: int, bar_q: list) -> list:
    """(q onset, q length, pitch, staff, grace) of the run's notes, ties joined (a tie continuation
    extends the open note of its pitch and staff that ends where it starts)."""
    events = []
    for i in range(lo, hi):
        for e in measures[i].events:
            events.append((bar_q[i] + Fraction(e.offset), Fraction(e.duration), int(e.pitch), int(e.staff),
                           bool(e.grace), bool(e.tie), e.seq))
    events.sort(key=lambda x: (x[0], x[6]))
    out, open_ = [], {}
    for q, d, p, s, g, tie, _ in events:
        key = (p, s)
        if tie and key in open_:
            j = open_[key]
            if abs(out[j][0] + out[j][1] - q) < Fraction(1, 48):
                out[j][1] += d
                continue
        out.append([q, d, p, s, g])
        if not g:
            open_[key] = len(out) - 1
    return out


def render(measures, rng: np.random.Generator, name: str = "score", cfg: RenderConfig = RenderConfig()) -> list:
    """The score's runs of regular bars as synthetic performances (see the module docstring)."""
    bar_q, acc = [], Fraction(0)
    for m in measures:
        bar_q.append(acc)
        acc += Fraction(m.length)
    perfs = []
    for k, (lo, hi) in enumerate(_runs(measures)):
        if hi - lo < cfg.min_bars:
            continue
        bq, bd, bts, bkey = [], [], [], []
        for i in range(lo, hi):
            ts = measures[i].time_signature
            n, qpb = beat_layout(ts)
            for j in range(n):
                bq.append(float(bar_q[i]) + j * qpb)
                bd.append(j == 0)
                bts.append(ts)
                bkey.append(int(measures[i].key))
        end_q = float(bar_q[hi - 1] + Fraction(measures[hi - 1].length))
        notes = _notes(measures, lo, hi, bar_q)
        if len(notes) < 8 or len(bq) < 8:
            continue
        # the time map: seconds per quarter at each beat (base tempo, rubato, maybe a ritardando)
        base = float(np.clip(np.exp(rng.normal(cfg.qpm_mu, cfg.qpm_sigma)), *cfg.qpm_range))
        w = np.zeros(len(bq))
        for j in range(1, len(bq)):
            w[j] = cfg.rubato_keep * w[j - 1] + rng.normal(0, cfg.rubato)
        if rng.random() < cfg.ritard_p:
            m = min(len(bq), int(rng.integers(4, 13)))
            w[-m:] -= np.linspace(0, rng.uniform(0.1, 0.35), m)
        spq = 60.0 / (base * np.exp(w))
        knots_q = np.array(bq + [end_q])
        t = np.concatenate([[0.0], np.cumsum(np.diff(knots_q) * spq)])
        t0 = float(rng.uniform(0.2, 1.0))
        t += t0

        def at(q):
            q = np.asarray(q, np.float64)
            beyond = np.maximum(q - knots_q[-1], 0) * spq[-1]
            return np.interp(q, knots_q, t) + beyond
        articulation = rng.uniform(*cfg.articulation)
        level = rng.uniform(50, 80)
        accent = rng.uniform(*cfg.accent) if rng.random() < 0.7 else 0.0
        down_q = {round(x, 6) for x, d in zip(bq, bd) if d}
        beat_q = {round(x, 6) for x in bq}
        rows = []
        chord_shift = {}
        for q, d, p, s, g in notes:
            qf = float(q)
            key = round(qf, 6)
            if key not in chord_shift:
                chord_shift[key] = rng.normal(0, cfg.jitter_s)
            on = float(at(qf)) + chord_shift[key] + rng.normal(0, cfg.chord_spread_s)
            if g:
                on -= rng.uniform(0.03, 0.08)
                off = on + 0.06
            else:
                written = float(at(qf + float(d))) - float(at(qf))
                off = on + max(0.03, written * float(np.clip(articulation + rng.normal(0, 0.08), 0.3, 1.05)))
            v = level + (accent if key in down_q else accent / 2 if key in beat_q else 0.0) + (4 if s == 0 else 0) \
                + rng.normal(0, 8)
            rows.append((min(max(p, 21), 108), on, off, float(np.clip(round(v), 1, 127))))
        arr = np.array(sorted(rows, key=lambda r: (r[1], r[0])), np.float32)
        arr[:, 1:3] = np.maximum(arr[:, 1:3], 0.0)
        bt = at(np.array(bq))
        gt = {"t": bt, "q": np.array(bq) - bq[0], "down": np.array(bd, bool),
              "ts": np.array([TS_CLASSES.index(x) if x in TS_CLASSES else IGNORE for x in bts]),
              "key": np.array(bkey), "ts_names": bts}
        perfs.append({"performance": f"synthetic/{name}/{k}", "piece": name, "dataset": "synthetic",
                      "notes": arr, "gt": gt, "beats": None, "qpm": base})
    return perfs
