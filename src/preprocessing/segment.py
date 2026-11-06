"""Cut an aligned (performance, score) pair into measure-aligned training segments."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .performance import ONSET, OFFSET, TimeMap
from .score import Measure, bar_length, tokenize_bars


@dataclass
class SegmentConfig:
    bars_per_segment: int = 4
    stride: int = 2                 # bars between segment starts (== bars_per_segment -> no overlap)
    onset_tolerance: float = 0.125  # quarter notes; notes played this early still belong to the next bar
    pedal_targets: bool = True      # add ped_on / ped_off tokens derived from performance CC64
    pedal_grid: int = 4             # pedal positions quantized to 1/pedal_grid quarter note
    pedal_min_quarters: float = 0.5 # drop shorter presses (pedal flutter, not notation)
    max_input_notes: int = 768
    max_target_tokens: int = 1536
    max_bar_quarters: int = 16      # drop segments containing longer bars (unbarred cadenzas)
    min_pitch_f1: float = 0.6       # drop segments whose performed pitches don't match the score's
                                    # (local beat-annotation errors); 0 disables


def pitch_f1(perf_pitches, tokens: list[str]) -> float:
    """F1 between the multiset of performed pitches and onset pitches in the target tokens."""
    score = Counter(int(t[2:]) for i, t in enumerate(tokens)
                    if t.startswith("p_") and not (i + 2 < len(tokens) and tokens[i + 2] == "tie"))
    perf = Counter(int(p) for p in perf_pitches)
    total = sum(score.values()) + sum(perf.values())
    return 2 * sum((score & perf).values()) / total if total else 1.0


def parse_downbeat_map(mp) -> list[list[int]] | None:
    """ASAP ``downbeats_score_map`` -> list of bars, each a list of measure indices.
    Entries are ints or strings like '8-0' (bar split across a repeat). NaN -> None."""
    if not isinstance(mp, list):
        return None
    return [[int(x) for x in str(v).split("-")] for v in mp]


def bar_timeline(bars: list[list[int]], measures: list[Measure], downbeats) -> tuple:
    """Score-time (s) start/end and quarter-length for each bar; prepends a pickup bar if present."""
    db = np.asarray(downbeats, dtype=np.float64)
    lengths = [bar_length(measures, b) for b in bars]
    starts = list(db)
    ends = list(db[1:])
    rate = (db[-1] - db[-2]) / float(lengths[-2]) if len(db) >= 2 and lengths[-2] > 0 else 0.5
    ends.append(db[-1] + float(lengths[-1]) * rate)

    first = bars[0][0]
    if 0 < first <= 2 and all(i < len(measures) for i in range(first)):  # anacrusis
        pick = list(range(first))
        pl = bar_length(measures, pick)
        r0 = (ends[0] - starts[0]) / float(lengths[0]) if lengths[0] > 0 else rate
        bars = [pick] + bars
        lengths = [pl] + lengths
        starts = [starts[0] - float(pl) * r0] + starts
        ends = [starts[1]] + ends
    return bars, np.asarray(starts), np.asarray(ends), lengths


def _to_quarters(s: float, starts, ends, cum_q) -> float:
    """Score time (s) -> quarter notes since the first bar."""
    k = int(np.clip(np.searchsorted(starts, s, side="right") - 1, 0, len(starts) - 1))
    span = ends[k] - starts[k]
    return cum_q[k] + (s - starts[k]) / span * (cum_q[k + 1] - cum_q[k]) if span > 0 else cum_q[k]


def _pedal_tokens(pedal, tm: TimeMap, starts, ends, lengths, j0, j1, lo, hi, cfg) -> list[list]:
    out = [[] for _ in range(j1 - j0)]
    cum_q = np.concatenate([[0.0], np.cumsum([float(x) for x in lengths])])
    for on, off in pedal:
        q_len = (_to_quarters(float(tm(off)), starts, ends, cum_q)
                 - _to_quarters(float(tm(on)), starts, ends, cum_q))
        if q_len < cfg.pedal_min_quarters:
            continue
        for t, tok in ((on, "ped_on"), (off, "ped_off")):
            s = float(tm(t))
            if tok == "ped_on" and s < lo <= float(tm(off)):
                s = max(starts[j0], lo)  # pedal already held when the segment starts
            if not lo <= s < hi:
                continue
            k = int(np.searchsorted(starts, s, side="right")) - 1
            k = min(max(k, j0), j1 - 1)
            span = ends[k] - starts[k]
            pos = (s - starts[k]) / span * float(lengths[k]) if span > 0 else 0.0
            pos = Fraction(round(pos * cfg.pedal_grid), cfg.pedal_grid)
            pos = min(max(pos, Fraction(0)), max(lengths[k] - Fraction(1, cfg.pedal_grid), Fraction(0)))
            out[k - j0].append((pos, tok))
    return out


def segment_pair(notes, pedal, ann: dict, measures: list[Measure], cfg: SegmentConfig,
                 meta: dict) -> tuple[list[dict], dict]:
    """Return (segments, stats) for one performance."""
    stats = {"too_long": 0, "empty": 0, "long_bar": 0, "low_pitch_f1": 0}
    bars = parse_downbeat_map(ann["downbeats_score_map"])
    if bars is None or len(bars) != len(ann["midi_score_downbeats"]) or len(bars) < 2:
        return [], {"skipped": "bad_downbeat_map"}
    if max(i for b in bars for i in b) >= len(measures):
        return [], {"skipped": "map_out_of_range"}
    tm = TimeMap(ann["performance_beats"], ann["midi_score_beats"])
    inv = tm.inverse()
    bars, starts, ends, lengths = bar_timeline(bars, measures, ann["midi_score_downbeats"])

    score_on = tm(notes[:, ONSET])
    spq = np.median((ends - starts) / np.maximum([float(x) for x in lengths], 1e-6))
    tol = cfg.onset_tolerance * spq
    nb, W = len(bars), cfg.bars_per_segment
    seg_starts = list(range(0, max(nb - W, 0) + 1, cfg.stride))
    if seg_starts[-1] + W < nb:
        seg_starts.append(nb - W)

    segs = []
    for j0 in seg_starts:
        j1 = min(j0 + W, nb)
        lo = -math.inf if j0 == 0 else starts[j0] - tol
        hi = math.inf if j1 == nb else ends[j1 - 1] - tol
        if max(lengths[j0:j1]) > cfg.max_bar_quarters:
            stats["long_bar"] += 1
            continue
        idx = np.nonzero((score_on >= lo) & (score_on < hi))[0]
        if len(idx) == 0:
            stats["empty"] += 1
            continue
        seg_notes = notes[idx]
        t0 = min(float(inv(starts[j0] - tol)), float(seg_notes[:, ONSET].min()))
        t1 = max(float(inv(ends[j1 - 1] - tol)), float(seg_notes[:, ONSET].max()))
        out = np.stack([seg_notes[:, 0], seg_notes[:, ONSET] - t0,
                        seg_notes[:, OFFSET] - seg_notes[:, ONSET], seg_notes[:, 3]], axis=1)
        ped = pedal[(pedal[:, 1] > t0) & (pedal[:, 0] < t1)] - t0 if len(pedal) else pedal
        ped_tok = _pedal_tokens(pedal, tm, starts, ends, lengths, j0, j1, lo, hi, cfg) if cfg.pedal_targets else None
        tokens = tokenize_bars(measures, bars[j0:j1], ped_tok)
        if len(out) > cfg.max_input_notes or len(tokens) + 2 > cfg.max_target_tokens:
            stats["too_long"] += 1
            continue
        f1 = pitch_f1(out[:, 0], tokens)
        if f1 < cfg.min_pitch_f1:
            stats["low_pitch_f1"] += 1
            continue
        segs.append({
            **meta,
            "bar_start": j0,
            "n_bars": j1 - j0,
            "window_sec": np.float32(t1 - t0),
            "perf_t0": np.float32(t0),            # absolute performance time of onset 0
            "notes": out.astype(np.float32),     # pitch, onset (s, rel.), duration (s), velocity
            "pedal": np.asarray(ped, dtype=np.float32).reshape(-1, 2),  # (on, off) rel. seconds
            "tokens": tokens,
        })
    stats["segments"] = len(segs)
    return segs, stats
