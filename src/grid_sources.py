"""Other sources of v2's metrical grid than the meter model alone: how much a better grid is worth,
and whether another system can supply one.

v2 labels every note on a grid of the whole piece (src/meter.py: the meter model over the tracked
beats). On the test windows the labeller places 78-86% of onsets right on the true grid and 26-49%
on the meter models' grids: the grid is v2's bottleneck. Three grids for the whole-piece benchmark
(scripts/system_scores.py --grid):

  oracle       the system's own tracked beats with a perfect meter model where the ground truth
               reaches (src/meter.py oracle_logp: each head certain of its target), the model's own
               log-probabilities elsewhere: what fixing the meter model alone could buy (_og)
  oracle_beats the annotated beats themselves where the ground truth reaches, with the perfect
               meter model on them: what fixing the beats and the meter could buy (_ogt)
  refine       not a source but a loop: the labeller's own bar lines (each note it writes on a bar
               line votes for the tracked beat it was played on) added to the meter model's bar
               log-probabilities, the grid decoded again and the notes labelled on it (_rf;
               V2Pipeline(refine=weight)): the two models deciding the bar lines together
  seq2seq      the seq2seq system's score of the same performance aligned to it (src/metrical.py
               fit_alignment): its bar lines and time signatures, timed through the matched notes,
               as v2's grid (_s2g). The seq2seq system writes its bar lines from the notes and is ~2
               MUSTER better on ASAP; v2 writes the notes better given a grid.

The ground truth: ASAP's annotations (data/processed/meter/test.pkl; the real transcriptions share
the ASAP timeline), Gangqinpu's reliable runs of aligned bars (test_gangqinpu.pkl, several per
performance), each a dict of src/meter.py annotation_grid's form.
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-4


def ground_truth(perfs: list) -> dict:
    """{performance: [gt, ...]} from the meter data's performances (a Gangqinpu performance has one
    per reliable stretch)."""
    out: dict = {}
    for p in perfs:
        if p.get("gt") is not None and len(p["gt"]["t"]) >= 2:
            out.setdefault(p["performance"], []).append(p["gt"])
    for v in out.values():
        v.sort(key=lambda g: float(g["t"][0]))
    return out


def spliced_logp(beat_t: np.ndarray, lp: dict, gts: list) -> dict:
    """The model's per-beat log-probabilities with every head certain of its target on the beats a
    ground truth covers (src/meter.py targets: -1 where it does not know), as oracle_logp."""
    from .meter import targets
    out = {k: (np.array(v, np.float64, copy=True) if isinstance(v, np.ndarray) and v.ndim == 2 else v)
           for k, v in lp.items()}
    for gt in gts:
        tg = targets(np.asarray(beat_t, np.float64), gt)
        for head in ("bar", "step", "ts", "pos"):
            if head not in out or head not in tg or not isinstance(out[head], np.ndarray) or out[head].ndim != 2:
                continue
            y = np.asarray(tg[head])
            n = out[head].shape[1]
            rows = np.nonzero((y >= 0) & (y < n))[0]
            if len(rows):
                out[head][rows] = math.log(EPS)
                out[head][rows, y[rows]] = 0.0
    return out


def annotated_beats(beat_t: np.ndarray, beat_d: np.ndarray, gts: list) -> tuple[np.ndarray, np.ndarray]:
    """The tracked beats with those inside a ground truth's span replaced by its annotated beats
    (and downbeats)."""
    from .beats import MATCH_TOL
    t, d = np.asarray(beat_t, np.float64), np.asarray(beat_d, bool)
    keep = np.ones(len(t), bool)
    add_t, add_d = [], []
    for gt in gts:
        lo, hi = float(gt["t"][0]) - MATCH_TOL, float(gt["t"][-1]) + MATCH_TOL
        keep &= ~((t >= lo) & (t <= hi))
        add_t.append(np.asarray(gt["t"], np.float64))
        add_d.append(np.asarray(gt["down"], bool))
    tt = np.concatenate([t[keep]] + add_t)
    dd = np.concatenate([d[keep]] + add_d)
    order = np.argsort(tt, kind="stable")
    tt, dd = tt[order], dd[order]
    ok = np.concatenate([[True], np.diff(tt) > 1e-3])    # no two beats at one time
    return tt[ok], dd[ok]


def score_grid(tokens: list[str], onsets: np.ndarray, pitches: np.ndarray, min_pairs: int = 8) -> dict | None:
    """A written score of the whole performance (tokens) as a grid: its bars and time signatures,
    each quarter timed through the notes matched to the performance (fit_alignment's piecewise-
    linear map), or None when too few notes match. {"t", "q" (quarters from the first bar line, one
    per quarter), "bars", "bar_q", "bar_t", "matched"}."""
    from .metrical import _map_through, fit_alignment
    from .notelabels import bar_quarters, score_notes
    onsets, pitches = np.asarray(onsets, np.float64), np.asarray(pitches)
    _, win = score_notes(tokens)
    if not win.bars:
        return None
    fit = fit_alignment(onsets, pitches, tokens)
    if fit is None:
        return None
    score, sq, live, pairs, pt = fit
    if len(pairs) < min_pairs:
        return None
    bars = [ts for ts, _ in win.bars]
    lens = np.array([float(bar_quarters(ts)) for ts in bars])
    bar_q = np.concatenate([[0.0], np.cumsum(lens)[:-1]])
    q = np.arange(0.0, float(lens.sum()) + 1e-9, 1.0)
    t = _map_through(pairs, q, pt)
    t = np.maximum.accumulate(t) + np.arange(len(t)) * 1e-6   # strictly increasing
    return {"t": t, "q": q, "bars": bars, "bar_q": bar_q, "bar_t": np.interp(bar_q, q, t),
            "matched": len(pairs)}


def labeller_bar_votes(placed: list, grid: dict, onsets: np.ndarray, beat_t: np.ndarray,
                       tol: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Per tracked beat: (notes the labeller wrote on a bar line, notes written anywhere) among the
    notes played within ``tol`` of a beat's length from it. ``placed``: (performed note, bar, pos)."""
    bt = np.asarray(beat_t, np.float64)
    down, on = np.zeros(len(bt)), np.zeros(len(bt))
    if len(bt) < 2:
        return down, on
    lens = np.diff(bt, append=bt[-1] + (bt[-1] - bt[-2]))
    for src, _bar, pos in placed:
        t = float(onsets[src])
        j = int(np.argmin(np.abs(bt - t)))
        if abs(bt[j] - t) > tol * lens[j]:
            continue
        on[j] += 1
        down[j] += float(pos) == 0
    return down, on


def with_votes(lp: dict, votes: tuple, weight: float) -> dict:
    """The meter model's bar log-probabilities with the labeller's votes added (``weight`` x the log
    of the smoothed share of each beat's notes written on a bar line, and off one)."""
    down, on = votes
    out = dict(lp)
    bar = np.array(lp["bar"], np.float64, copy=True)
    n = min(len(bar), len(on))
    m = on[:n] > 0
    if m.any():
        d, o = down[:n][m], on[:n][m]
        bar[:n][m, 1] += weight * np.log((d + 0.5) / (o + 1.0))
        bar[:n][m, 0] += weight * np.log((o - d + 0.5) / (o + 1.0))
    out["bar"] = bar
    return out
