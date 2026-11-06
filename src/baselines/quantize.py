"""Baseline 1: classical beat-snapping quantization (with oracle beats).

For one evaluation window: map every performed note onset to score time through the annotated beat
grid (the same piecewise-linear performance->score time map used to build the data), snap its
position in the bar to the nearest 16th or 8th-triplet, snap its duration to the nearest common
note value (in log space), and put notes at or above middle C on the upper staff. Bar lines, time
and key signatures come from the annotations too (oracle meter), so this is the classical
"quantize given the beats" setting: an optimistic version of what a quantizer + beat tracker does.
No grace notes, ties, arpeggios or ornaments. Pedal: the performance's sustain pedal, placed the
same way (presses shorter than an eighth note dropped).

``analyze_window`` does the placement and returns per-note arrays (also the feature basis of the
supervised baseline, ``xgb_quantizer``); ``render_window`` turns placed notes into tokens in the
target grammar, directly comparable with the ground truth.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from ..preprocessing.performance import TimeMap
from ..preprocessing.score import Measure, NoteEvent, tokenize_bars
from ..preprocessing.segment import (_locate, _to_quarters, _window_pedal_tokens, bar_timeline,
                                     parse_downbeat_map)

GRIDS = (4, 3)  # 16ths, 8th-note triplets (per quarter)
DURATIONS = [Fraction(x) for x in ("1/4", "1/3", "1/2", "2/3", "3/4", "1", "3/2", "2", "3", "4")]
_LOG_D = np.log([float(d) for d in DURATIONS])


def snap_position(pos: float) -> Fraction:
    cands = [Fraction(round(pos * g), g) for g in GRIDS]
    return min(cands, key=lambda c: (abs(float(c) - pos), c.denominator))


def snap_duration(q: float) -> Fraction:
    return DURATIONS[int(np.argmin(np.abs(_LOG_D - math.log(max(q, 1e-3)))))]


def nominal_bar_len(ts: str) -> Fraction:
    n, d = ts.split("/")
    return Fraction(int(n) * 4, int(d))


class PieceTiming:
    """Bar timeline + time maps for one performance (oracle annotations)."""

    def __init__(self, ann: dict, measures: list[Measure]):
        self.measures = measures
        self.tm = TimeMap(ann["performance_beats"], ann["midi_score_beats"])
        bars = parse_downbeat_map(ann["downbeats_score_map"])
        self.bars, self.starts, self.ends, self.lengths = bar_timeline(bars, measures, ann["midi_score_downbeats"])
        self.cum_q = np.concatenate([[0.0], np.cumsum([float(x) for x in self.lengths])])

    def quarters(self, score_t: float) -> float:
        return _to_quarters(score_t, self.starts, self.ends, self.cum_q)

    def ts(self, k: int) -> str:
        return self.measures[self.bars[k][0]].time_signature


def analyze_window(notes: np.ndarray, meta: dict, timing: PieceTiming) -> dict:
    """Place every in-window note (onset >= 0) on the oracle grid. Returns per-note arrays over the
    in-window notes: ``idx`` (row in ``notes``), ``bar`` (bar index), ``pos_raw`` / ``pos`` (raw and
    snapped position in the bar), ``dur_raw`` (performed duration in quarters), ``dur`` (snapped),
    ``abs`` (snapped onset in quarters from the window's first bar start, nominal bar lengths,
    i.e. the metric's coordinate system)."""
    T, t0 = timing, float(meta["perf_t0"])
    k_lo, start_pos = int(meta["bar_start"]), float(meta.get("start_pos", 0.0))
    idx = np.nonzero(notes[:, 1] >= 0)[0]
    out = {k: [] for k in ("idx", "bar", "pos_raw", "pos", "dur_raw", "dur", "abs")}
    for i in idx:
        s_on = float(T.tm(notes[i, 1] + t0))
        k, pos = _locate(s_on, T.starts, T.ends, T.lengths)
        if k < k_lo or (k == k_lo and pos < start_pos):
            k, pos = k_lo, start_pos
        p = snap_position(pos)
        if p >= T.lengths[k] and k + 1 < len(T.lengths):
            k, p = k + 1, Fraction(0)
        p = min(p, max(T.lengths[k] - Fraction(1, 48), Fraction(0)))
        s_off = float(T.tm(notes[i, 1] + notes[i, 2] + t0))
        dq = T.quarters(s_off) - T.quarters(s_on)
        out["idx"].append(i)
        out["bar"].append(k)
        out["pos_raw"].append(pos)
        out["pos"].append(p)
        out["dur_raw"].append(dq)
        out["dur"].append(snap_duration(dq))
    bar_off = {}
    off = Fraction(0)
    for k in range(k_lo, (max(out["bar"]) if out["bar"] else k_lo) + 1):
        bar_off[k] = off
        off += nominal_bar_len(T.ts(k))
    out["abs"] = [bar_off[k] + p for k, p in zip(out["bar"], out["pos"])]
    out["bar_off"] = bar_off
    return out


def render_window(placed: list[tuple], pedal: np.ndarray, meta: dict, timing: PieceTiming,
                  pedal_grid: int = 4, pedal_min_quarters: float = 0.5) -> list[str]:
    """``placed`` = [(bar, pos, pitch, duration, staff)] -> tokens for the window."""
    T, t0 = timing, float(meta["perf_t0"])
    k_lo, start_pos = int(meta["bar_start"]), float(meta.get("start_pos", 0.0))
    s_lo = -math.inf if meta.get("s_lo") is None else float(meta["s_lo"])
    s_hi = math.inf if meta.get("s_hi") is None else float(meta["s_hi"])
    k_hi = max([k for k, *_ in placed] + [k_lo])
    synth = []
    for k in range(k_lo, k_hi + 1):
        ref = T.measures[T.bars[k][0]]
        m = Measure(length=T.lengths[k], time_signature=ref.time_signature, key=ref.key)
        m.events = [NoteEvent(offset=p, staff=st, pitch=pi, duration=d, seq=n)
                    for n, (kk, p, pi, d, st) in enumerate(placed) if kk == k]
        synth.append(m)
    ped_abs = pedal + t0 if len(pedal) else pedal
    ped_tok = _window_pedal_tokens(ped_abs, T.tm, T.starts, T.ends, T.lengths, k_lo, k_hi, s_lo, s_hi,
                                   start_pos, pedal_grid, pedal_min_quarters)
    return tokenize_bars(synth, [[j] for j in range(len(synth))], ped_tok,
                         start_pos=start_pos if meta.get("s_lo") is not None else None)


def quantize_window(notes: np.ndarray, pedal: np.ndarray, meta: dict, timing: PieceTiming) -> list[str]:
    """Beat-snap baseline: snapped onsets and durations, middle-C staff split."""
    a = analyze_window(notes, meta, timing)
    placed = [(k, p, int(notes[i, 0]), d, 0 if notes[i, 0] >= 60 else 1)
              for i, k, p, d in zip(a["idx"], a["bar"], a["pos"], a["dur"])]
    return render_window(placed, pedal, meta, timing)
