"""The meter model (v2, Phase 2): from tracked beats and the performed notes to the score's metrical
grid of a whole piece (notated beats with bar lines and a time signature per bar), the grid the
note labeller (src/labeller.py) reads every note's position from.

A tracker's beats are close to the notated beats but not the same: Beat This! is at double tempo
on 18% of pieces, sometimes at the eighth notes of a compound meter, misses 12% of beats, adds 13%,
and marks downbeats almost independently per beat (src/beats.py). So the meter model works on the
TRACKED beats and predicts, for each:
  bar      does a bar start on this beat
  step     how many quarter notes it lasts, to the next tracked beat (Q_STEPS): absorbs a doubled
           or tripled tempo, a missed or an extra beat
  ts       the time signature of its bar (TS_CLASSES)
from per-beat features of the notes around it (``beat_features``: where notes fall inside the
beat, how loud / low / long the notes on the beat are, the harmony and its change across it, the
tracker's own downbeat flag, the beat's length) and, with a note encoder (``note_inputs``), a
learned summary of every note in the beat: each note's place in the beat, length, loudness,
pitch and chord are embedded and pooled per beat (mean and max), so the cues the hand-made
features miss are learned. A transformer over the piece's beat sequence makes the predictions; ``decode`` then picks the most probable CONSISTENT grid (a Viterbi search
over time signature x position in the bar: bars add up to their time signature, which changes only
at a bar line), and ``grid_from_states`` turns it into (beat time, quarters from the first bar line)
anchors plus the bars' time signatures.

Training targets (``targets``) come from ASAP's annotations (``annotation_grid``): the annotated
beats with their quarter-note positions (beat_layout of each bar's time signature) and downbeats,
mapped onto the tracked beats by time. PERiScoPe has no beat annotations but every window is
aligned to its score: ``aligned_grids`` turns the score's beats timed through those alignments
(src/beats.py window_grid, merge_performance_bars) into the same ground truth, and
``stitch_notes`` joins the windows' notes back into the performance (6x the works ASAP has).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np

from .beats import MATCH_TOL, beat_layout

TS_CLASSES = ("4/4", "2/4", "3/4", "6/8", "3/8", "5/8", "9/8", "2/2", "12/16", "12/8", "6/4", "3/2", "4/8",
              "5/4", "4/2")
Q_STEPS = tuple(Fraction(x) for x in ("1/4", "1/3", "1/2", "2/3", "3/4", "1", "4/3", "3/2", "2", "3", "4"))
STEP_TOL = 0.2      # a step within 20% of a Q_STEPS value takes it; further off -> no step target
GRID = 12           # positions in the bar are multiples of 1/12 quarter (16ths and triplets)
IGNORE = -1


def bar_quarters(ts: str) -> Fraction:
    n, d = ts.split("/")
    return Fraction(int(n) * 4, int(d))


# ---------------------------------------------------------------------------------------------
# ground truth from the annotations
# ---------------------------------------------------------------------------------------------

def annotation_grid(a: dict) -> dict:
    """ASAP annotations of one performance -> {"t": beat times, "q": quarters from the first beat,
    "down": bar starts, "ts": TS_CLASSES index per beat (-1: a meter outside the classes),
    "key": key signature per beat}. Each beat lasts its bar's beat length (beat_layout)."""
    t = np.asarray(a["performance_beats"], np.float64)
    downs = np.asarray(a["performance_downbeats"], np.float64)
    down = np.isin(np.round(t, 6), np.round(downs, 6))
    tss = sorted((float(k), v[0]) for k, v in a.get("perf_time_signatures", {}).items())
    keys = sorted((float(k), int(v[0])) for k, v in a.get("perf_key_signatures", {}).items())

    def active(changes, x, default):
        cur = changes[0][1] if changes else default
        for when, v in changes:
            if when <= x + 1e-6:
                cur = v
        return cur
    ts = [active(tss, x, "4/4") for x in t]
    qpb = np.array([beat_layout(s)[1] for s in ts])
    q = np.concatenate([[0.0], np.cumsum(qpb[:-1])])
    return {"t": t, "q": q, "down": down, "ts": np.array([TS_CLASSES.index(s) if s in TS_CLASSES else IGNORE
                                                          for s in ts]),
            "key": np.array([active(keys, x, 0) for x in t]), "ts_names": ts}


def quarters_at(t: np.ndarray, gt: dict) -> np.ndarray:
    """Quarter-note position of any times on the annotated grid (linear between beats, extended
    with the end tempos)."""
    bt, bq = gt["t"], gt["q"]
    out = np.interp(t, bt, bq)
    s0 = (bq[1] - bq[0]) / max(bt[1] - bt[0], 1e-6)
    s1 = (bq[-1] - bq[-2]) / max(bt[-1] - bt[-2], 1e-6)
    out = np.where(t < bt[0], bq[0] + (t - bt[0]) * s0, out)
    return np.where(t > bt[-1], bq[-1] + (t - bt[-1]) * s1, out)


def written_tempo(gt: dict, t0: float = -np.inf, t1: float = np.inf) -> float:
    """log quarters per minute of the annotated grid between t0 and t1: the median over its beats
    (nan with fewer than 2)."""
    t, q = np.asarray(gt["t"], np.float64), np.asarray(gt["q"], np.float64)
    keep = (t >= t0 - MATCH_TOL) & (t <= t1 + MATCH_TOL)
    qpm = piece_qpm({"t": t[keep], "q": q[keep]}) if keep.sum() >= 2 else None
    return math.log(qpm) if qpm else float("nan")


N_BAR_POS = 16      # the position-in-bar head: eighth notes from the bar line (0 .. 7.5 quarters)
POS_TOL = 0.3       # eighths: a beat further than this from an eighth-note position has no target


def targets(beat_t: np.ndarray, gt: dict) -> dict:
    """Per tracked beat: bar (1 / 0; -1 outside the annotated span), step (Q_STEPS index; -1 when
    the true length is no Q_STEPS value within STEP_TOL), ts (TS_CLASSES index of the nearest
    annotated beat's bar; -1 outside), pos (eighth notes from the annotated bar line before it,
    N_BAR_POS classes; -1 before the first bar line, outside, or off the eighth-note grid); for the
    piece, tempo: its written tempo over these beats (``written_tempo``, [1], nan when unknown)."""
    k = len(beat_t)
    inside = (beat_t >= gt["t"][0] - MATCH_TOL) & (beat_t <= gt["t"][-1] + MATCH_TOL)
    bar = np.where(inside, 0, IGNORE)
    for d in gt["t"][gt["down"]]:
        j = int(np.argmin(np.abs(beat_t - d))) if k else -1
        if j >= 0 and abs(beat_t[j] - d) <= MATCH_TOL:
            bar[j] = 1
    q = quarters_at(beat_t, gt)
    steps = np.diff(q) if k > 1 else np.zeros(0)
    steps = np.concatenate([steps, steps[-1:]]) if k > 1 else np.zeros(k)
    vals = np.array([float(s) for s in Q_STEPS])
    rel = np.abs(steps[:, None] - vals[None]) / vals[None]
    step = np.where(rel.min(1) <= STEP_TOL, rel.argmin(1), IGNORE)
    step = np.where(inside, step, IGNORE)
    near = np.clip(np.searchsorted(gt["t"], beat_t), 0, len(gt["t"]) - 1)
    prev = np.clip(near - 1, 0, len(gt["t"]) - 1)
    near = np.where(np.abs(gt["t"][prev] - beat_t) < np.abs(gt["t"][near] - beat_t), prev, near)
    ts = np.where(inside, gt["ts"][near], IGNORE)
    tempo = written_tempo(gt, float(beat_t[0]), float(beat_t[-1])) if k else float("nan")
    bq = np.asarray(gt["q"], np.float64)[np.asarray(gt["down"], bool)]
    pos = np.full(k, IGNORE)
    if len(bq) and k:   # a beat within tolerance of the next bar line is that bar's 0
        j = np.searchsorted(bq, q + POS_TOL / 2, side="right") - 1
        e = (q - bq[np.clip(j, 0, None)]) * 2
        r = np.round(e)
        ok = inside & (j >= 0) & (np.abs(e - r) <= POS_TOL) & (r >= 0) & (r < N_BAR_POS)
        pos = np.where(ok, r, IGNORE)
    return {"bar": bar.astype(np.int64), "step": step.astype(np.int64), "ts": ts.astype(np.int64),
            "pos": pos.astype(np.int64), "tempo": np.array([tempo], np.float32)}


def aligned_grids(times, bars, beats, ts_names, min_beats: int = 16) -> list[dict]:
    """Ground truth from a score alignment instead of annotations (PERiScoPe): the merged beats of
    a performance's windows (src/beats.py merge_performance_bars: time, global bar, beat in the
    bar, the bar's time signature) -> ``annotation_grid``-style dicts, one per run of consecutive
    bars (a bar no window timed, e.g. a dropped window, splits the performance: what lies in it is
    unknown) with at least ``min_beats`` beats. Quarter positions come from the bars' lengths, so a
    beat missing inside a bar does not shift the ones after it; q = 0 at the run's first bar line.
    Key signature: 0 (unused)."""
    times, bars, beats = (np.asarray(x) for x in (times, bars, beats))
    out = []
    if not len(times):
        return out
    cut = np.nonzero(np.diff(bars) > 1)[0] + 1
    for seg in np.split(np.arange(len(times)), cut):
        if len(seg) < min_beats:
            continue
        b0, b1 = int(bars[seg[0]]), int(bars[seg[-1]])
        bar_ts = {}
        for i in seg:
            bar_ts.setdefault(int(bars[i]), ts_names[i])
        start, q_of_bar = 0.0, {}
        for b in range(b0, b1 + 1):
            q_of_bar[b] = start
            tsb = bar_ts.setdefault(b, bar_ts.get(b - 1, "4/4"))
            try:
                start += float(bar_quarters(tsb))
            except (ValueError, ZeroDivisionError):
                start += 4.0
        names = [bar_ts[int(bars[i])] for i in seg]
        q = np.array([q_of_bar[int(bars[i])] + int(beats[i]) * beat_layout(n)[1] for i, n in zip(seg, names)])
        out.append({"t": np.asarray(times[seg], np.float64), "q": q, "down": np.asarray(beats[seg] == 0),
                    "ts": np.array([TS_CLASSES.index(n) if n in TS_CLASSES else IGNORE for n in names]),
                    "key": np.zeros(len(seg), np.int64), "ts_names": names})
    return out


def stitch_notes(windows, dedupe_s: float = 0.002) -> np.ndarray:
    """A performance's notes from its windows: ``windows`` = [(perf_t0, notes [n, 4] as stored:
    pitch, onset from the window start, duration, velocity)] -> [N, 4] pitch, onset, offset,
    velocity (load_performance's columns) on the performance's timeline. Notes carried over from
    before a window (onset < 0) are the previous window's; a note in two overlapping windows is
    kept once (same pitch within ``dedupe_s``)."""
    parts = []
    for t0, n in windows:
        n = np.asarray(n, np.float64)
        n = n[n[:, 1] >= 0]
        if len(n):
            on = n[:, 1] + float(t0)
            parts.append(np.stack([n[:, 0], on, on + n[:, 2], n[:, 3]], axis=1))
    if not parts:
        return np.zeros((0, 4), np.float32)
    a = np.concatenate(parts)
    a = a[np.lexsort((a[:, 1], a[:, 0]))]
    keep = np.ones(len(a), bool)
    keep[1:] = ~((a[1:, 0] == a[:-1, 0]) & (a[1:, 1] - a[:-1, 1] < dedupe_s))
    a = a[keep]
    return a[np.argsort(a[:, 1], kind="stable")].astype(np.float32)


# ---------------------------------------------------------------------------------------------
# per-beat features from the notes
# ---------------------------------------------------------------------------------------------

N_PHASE_BINS = 12
AT_BEAT = 0.05      # seconds: a note "on" the beat
N_BEAT_FEATURES = 27 + 12   # the first 27 (26 used) are the first meter model's, then the pitch classes
# a second tracker's beats as evidence on the first one's (``aux``): has one, a beat of it at this
# beat, how many of its beats in the beat's interval (log), their mean place in it, a downbeat of it
# here. After the others, so a model without it (n_features = N_BEAT_FEATURES) reads none of them
N_AUX_FEATURES = 5
N_ALL_FEATURES = N_BEAT_FEATURES + N_AUX_FEATURES


def uses_aux(cfg) -> bool:
    """Whether a meter model (its config: dict or MeterConfig) reads a second tracker's beats."""
    n = cfg["n_features"] if isinstance(cfg, dict) else cfg.n_features
    return n > N_BEAT_FEATURES


def aux_features(beat_t: np.ndarray, aux) -> np.ndarray:
    """[K, N_AUX_FEATURES]: the second tracker's beats ``aux`` = (times, downbeat flags) seen
    from each beat of the first one (its interval: up to the next beat): 1; one of them within
    AT_BEAT (it agrees); how many start inside the interval past that (log2 1 + n: 1 when it beats
    twice as fast); their mean place in it (0..1; 0.5 = on the off-beat: the first tracker off by
    half a beat, or at half the speed); one of them within AT_BEAT marked a downbeat."""
    k = len(beat_t)
    f = np.zeros((k, N_AUX_FEATURES), np.float32)
    if k == 0 or aux is None:
        return f
    t = np.asarray(beat_t, np.float64)
    at, ad = np.asarray(aux[0], np.float64), np.asarray(aux[1], bool)
    order = np.argsort(at, kind="stable")
    at, ad = at[order], ad[order]
    f[:, 0] = 1.0
    if not len(at):
        return f
    ln = np.diff(t)
    ln = np.concatenate([ln, ln[-1:]]) if k > 1 else np.array([0.5])
    ln = np.clip(ln, 0.05, 5.0)
    for j in range(k):
        a0, a1 = np.searchsorted(at, t[j] - AT_BEAT), np.searchsorted(at, t[j] + AT_BEAT)
        if a1 > a0:
            f[j, 1] = 1.0
            f[j, 4] = float(ad[a0:a1].any())
        lo, hi = np.searchsorted(at, t[j] + AT_BEAT), np.searchsorted(at, t[j] + ln[j] - AT_BEAT)
        if hi > lo:
            f[j, 2] = math.log2(1 + hi - lo)
            f[j, 3] = float(np.mean((at[lo:hi] - t[j]) / ln[j]))
    return f


def beat_features(notes: np.ndarray, beat_t: np.ndarray, beat_d: np.ndarray, aux=None) -> np.ndarray:
    """[K, N_ALL_FEATURES] for K tracked beats (the last N_AUX_FEATURES: ``aux_features`` of a
    second tracker's beats ``aux``, zeros without one; a model reads the first n_features). ``notes``: [N, 4] pitch, onset, offset, velocity
    (load_performance's columns), ``beat_d``: the tracker's downbeat flags. Per beat (its interval
    = up to the next beat; the last one repeats the previous length):
      beat length (log2 s / 0.5), log ratio to the previous and next beat, tracker downbeat, 1
      where in the beat the notes start (N_PHASE_BINS fractions) and how many (log)
      the notes on the beat: how many (log), loudest velocity, velocity vs the beat's mean, lowest
        pitch, any below middle-C's octave (bass), their longest duration (log2 s)
      change of harmony across the beat (1 - cosine of the pitch-class profiles of the beat and
        the previous one), lowest pitch of the beat's notes
      the beat's pitch-class profile (12, weighted by duration up to 2 s, summing to 1)"""
    k = len(beat_t)
    f = np.zeros((k, N_ALL_FEATURES), np.float32)
    if k == 0:
        return f
    f[:, N_BEAT_FEATURES:] = aux_features(beat_t, aux)
    t = np.asarray(beat_t, np.float64)
    ln = np.diff(t)
    ln = np.concatenate([ln, ln[-1:]]) if k > 1 else np.array([0.5])
    ln = np.clip(ln, 0.05, 5.0)
    on, pitch, vel = notes[:, 1].astype(np.float64), notes[:, 0].astype(np.int64), notes[:, 3].astype(np.float64)
    dur = np.clip(notes[:, 2] - notes[:, 1], 0.01, 20.0)
    order = np.argsort(on, kind="stable")
    on, pitch, vel, dur = on[order], pitch[order], vel[order], dur[order]
    f[:, 0] = np.log2(ln / 0.5)
    f[:, 1] = np.concatenate([[0.0], np.log2(ln[1:] / ln[:-1])])
    f[:, 2] = np.concatenate([np.log2(ln[1:] / ln[:-1]), [0.0]]) if k > 1 else 0.0
    f[:, 3] = np.asarray(beat_d, np.float32)
    f[:, 4] = 1.0
    prev_chroma = np.zeros(12)
    for j in range(k):
        lo, hi = np.searchsorted(on, t[j] - AT_BEAT), np.searchsorted(on, t[j] + ln[j] - AT_BEAT)
        if hi > lo:
            ph = np.clip((on[lo:hi] - t[j]) / ln[j], 0, 0.999)
            h = np.bincount((ph * N_PHASE_BINS).astype(int), minlength=N_PHASE_BINS)[:N_PHASE_BINS]
            f[j, 5:5 + N_PHASE_BINS] = h / max(hi - lo, 1)
        c = 5 + N_PHASE_BINS
        f[j, c] = math.log1p(hi - lo)
        a0, a1 = np.searchsorted(on, t[j] - AT_BEAT), np.searchsorted(on, t[j] + AT_BEAT)
        if a1 > a0:
            f[j, c + 1] = math.log1p(a1 - a0)
            f[j, c + 2] = vel[a0:a1].max() / 127.0
            f[j, c + 3] = (vel[a0:a1].mean() - (vel[lo:hi].mean() if hi > lo else vel[a0:a1].mean())) / 32.0
            f[j, c + 4] = pitch[a0:a1].min() / 128.0
            f[j, c + 5] = float(pitch[a0:a1].min() < 48)
            f[j, c + 6] = np.log2(dur[a0:a1].max() / 0.5) / 4.0
        chroma = np.bincount(pitch[lo:hi] % 12, minlength=12).astype(float) if hi > lo else np.zeros(12)
        na, nb = np.linalg.norm(chroma), np.linalg.norm(prev_chroma)
        f[j, c + 7] = 1.0 - (chroma @ prev_chroma) / (na * nb) if na > 0 and nb > 0 else 0.0
        f[j, c + 8] = pitch[lo:hi].min() / 128.0 if hi > lo else 0.0
        if hi > lo:
            w = np.bincount(pitch[lo:hi] % 12, weights=np.minimum(dur[lo:hi], 2.0), minlength=12)
            f[j, 27:39] = w / max(w.sum(), 1e-9)  # (index 26 is unused: always 0, as before)
        prev_chroma = chroma
    return f


N_NOTE_FEATURES = 20
CHORD_S = 0.03          # onsets this close form one chord
MAX_NOTES_PER_BEAT = 32


def note_inputs(notes: np.ndarray, beat_t: np.ndarray):
    """Every note's input to the note encoder, in the frame of the tracked beat it starts in (from
    AT_BEAT before the beat to AT_BEAT before the next one, as ``beat_features``): returns
    (features float32 [M, N_NOTE_FEATURES], pitch int64 [M], beat index int64 [M]) for the M notes
    inside the beats (at most MAX_NOTES_PER_BEAT per beat on average). Features: its phase in the
    beat and sin / cos of it at 1, 2, 3, 4, 6 and 12 cycles per beat (where the subdivisions fall),
    its length in beats (log2), velocity and velocity relative to the beat's mean, lowest / highest
    of its chord, the chord's size, its pitch above the beat's lowest note."""
    k = len(beat_t)
    empty = (np.zeros((0, N_NOTE_FEATURES), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64))
    if k == 0 or not len(notes):
        return empty
    t = np.asarray(beat_t, np.float64)
    ln = np.diff(t)
    ln = np.concatenate([ln, ln[-1:]]) if k > 1 else np.array([0.5])
    ln = np.clip(ln, 0.05, 5.0)
    on = notes[:, 1].astype(np.float64)
    order = np.argsort(on, kind="stable")
    on, pitch = on[order], notes[order, 0].astype(np.int64)
    dur = np.clip(notes[order, 2] - notes[order, 1], 0.01, 20.0).astype(np.float64)
    vel = notes[order, 3].astype(np.float64)
    j = np.searchsorted(t, on + AT_BEAT, side="right") - 1
    inside = (j >= 0) & (on < t[-1] + ln[-1] - AT_BEAT)
    if not inside.any():
        return empty
    on, pitch, dur, vel, j = on[inside], pitch[inside], dur[inside], vel[inside], j[inside]
    cap = MAX_NOTES_PER_BEAT * k
    if len(on) > cap:  # an absurdly dense input: keep an even sample
        pick = np.linspace(0, len(on) - 1, cap).round().astype(int)
        on, pitch, dur, vel, j = on[pick], pitch[pick], dur[pick], vel[pick], j[pick]
    m = len(on)
    ph = np.clip((on - t[j]) / ln[j], -0.25, 1.0)
    f = np.zeros((m, N_NOTE_FEATURES), np.float32)
    f[:, 0] = ph
    for c, cyc in enumerate((1, 2, 3, 4, 6, 12)):
        f[:, 1 + 2 * c] = np.sin(2 * np.pi * cyc * ph)
        f[:, 2 + 2 * c] = np.cos(2 * np.pi * cyc * ph)
    f[:, 13] = np.clip(np.log2(dur / ln[j]), -6, 4) / 4.0
    f[:, 14] = vel / 127.0
    cnt = np.bincount(j, minlength=k)
    mean_v = np.bincount(j, weights=vel, minlength=k) / np.maximum(cnt, 1)
    f[:, 15] = (vel - mean_v[j]) / 32.0
    chord = np.concatenate([[0], np.cumsum(np.diff(on) > CHORD_S)])
    lo_p = np.full(chord[-1] + 1, 999)
    hi_p = np.full(chord[-1] + 1, -1)
    np.minimum.at(lo_p, chord, pitch)
    np.maximum.at(hi_p, chord, pitch)
    size = np.bincount(chord)
    f[:, 16] = pitch == lo_p[chord]
    f[:, 17] = pitch == hi_p[chord]
    f[:, 18] = np.log1p(size[chord]) / 2.0
    low = np.full(k, 999)
    np.minimum.at(low, j, pitch)
    f[:, 19] = (pitch - low[j]) / 24.0
    return f, np.clip(pitch, 0, 127), j.astype(np.int64)


# ---------------------------------------------------------------------------------------------
# consistent decoding: Viterbi over (time signature, position in the bar)
# ---------------------------------------------------------------------------------------------

@dataclass
class DecodeConfig:
    ts_change: float = math.log(1e-3)     # log-prob of a new time signature at a bar line
    mid_step_bar: float = math.log(1e-2)  # log-prob of a bar line falling between two tracked beats
    ts_weight: float = 1.0                # weight of the time-signature head's log-probs
    bar_weight: float = 1.0               # weight of the bar head's log-probs
    pos_weight: float = 1.0               # weight of the position-in-bar head's log-probs (a model with one)
    # a prior on the written tempo each step implies (quarters per minute between two tracked beats):
    # log-normal, centred on the training scores' (median 122, log-mean 116 quarters per minute over
    # 756 ASAP performances, log-std 0.51). Off by default (weight 0); --tune --tempo-prior sets it.
    # It weighs steps that differ by a factor of 2 (3/8 against 3/4, 2/4 against 2/2): beat and
    # downbeat F cannot see that choice, and the step head gets it wrong most often.
    tempo_weight: float = 0.0
    tempo_mu: float = math.log(116.0)
    tempo_sigma: float = 0.5
    # the note-value level of the whole piece (choose_level), after decoding: the grid as decoded,
    # or the same bar lines at halved / doubled note values (3/8 <-> 3/4, 2/4 <-> 2/2), whichever
    # scores best on the piece's written tempo: level_weight x a log-normal prior (level_mu /
    # level_sigma: the training scores'), level_model_weight x the meter model's own estimate of
    # this piece's written tempo (MeterConfig.tempo_head), plus level_stay for keeping the decoded
    # one. All 0 (the default): the grid as decoded. --tune --level sets them on validation.
    level_weight: float = 0.0
    level_mu: float = math.log(116.0)
    level_sigma: float = 0.51
    level_model_weight: float = 0.0
    level_stay: float = 0.0
    # the grid's beat times moved onto the notes after decoding (refine_grid): an anchor (a tracked
    # beat) onto the onset of a chord within snap_s seconds, and a notated beat no tracked beat
    # times (a missed beat, a step of 2 quarters) added at a chord onset within fill_tol beats of
    # where the anchors around it put it. 0 (the default): the grid as decoded.
    snap_s: float = 0.0
    fill_tol: float = 0.0
    # local tempo consistency (local_tempo_logprior): the grid decoded once, each step's written
    # tempo compared with the median of the steps around it (LOCAL_WINDOW each side), and the
    # decoder run again with local_weight x a log-normal (local_sigma) around that median added to
    # the step head. The decoder alone takes a short bar as a jump (three quarters in one beat's
    # time: the user's bars 47 and 51) because a new time signature costs more than an implausible
    # tempo for one step; this weighs the step against its neighbours. 0 (the default): off;
    # scripts/eval_meter.py --local-tempo tunes it on validation (ASAP, real, PERiScoPe, Gangqinpu).
    local_weight: float = 0.0
    local_sigma: float = 0.25

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None):
        return cls(**{k: float(v) for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def _states(ts_idx: tuple[int, ...]):
    """All (ts, position in GRID units) states, and each ts's bar length in GRID units."""
    lens = [int(bar_quarters(TS_CLASSES[c]) * GRID) for c in ts_idx]
    states = [(c, p) for c, L in zip(ts_idx, lens) for p in range(L)]
    return states, dict(zip(ts_idx, lens))


@lru_cache(maxsize=8)
def _machine(ts_idx: tuple[int, ...]):
    """The decoder's transitions for a set of time signatures: the states, and for every state the
    moves into it (source state, step, passes the bar line?), padded with source n (a dummy state
    scored -inf). A state's moves are listed by descending (source, step), so ``argmax`` breaks
    ties as the first decoder did (the last of equal candidates)."""
    states, lens = _states(ts_idx)
    n = len(states)
    index = {s: i for i, s in enumerate(states)}
    steps = [int(s * GRID) for s in Q_STEPS]
    inc = [[] for _ in range(n)]
    for i, (c, p) in enumerate(states):
        for j, s in enumerate(steps):
            q = p + s
            over = False
            if q >= lens[c]:
                over = q > lens[c]
                q = (q - lens[c]) % lens[c]
            inc[index[(c, q)]].append((i, j, over))
    m = max(len(x) for x in inc)
    src = np.full((n, m), n, np.int64)
    stp = np.zeros((n, m), np.int64)
    mid = np.zeros((n, m), bool)
    for t, lst in enumerate(inc):
        for r, (i, j, over) in enumerate(reversed(lst)):
            src[t, r], stp[t, r], mid[t, r] = i, j, over
    st_c = np.array([c for c, _ in states])
    st_p = np.array([p for _, p in states])
    return states, src, stp, mid, st_c, st_p == 0


def decode(logp_bar: np.ndarray, logp_step: np.ndarray, logp_ts: np.ndarray, cfg: DecodeConfig = DecodeConfig(),
           ts_idx: tuple[int, ...] | None = None, return_steps: bool = False, logp_pos: np.ndarray | None = None):
    """Most probable consistent (time signature, position in the bar) per tracked beat.
    logp_bar [K, 2] (no / yes), logp_step [K, len(Q_STEPS)], logp_ts [K, len(TS_CLASSES)].
    Beat k+1's position = beat k's + its step; reaching the bar length exactly starts a bar on beat
    k+1 (where the time signature may change), passing it puts the bar line between the two
    (penalised: a tracker that missed the downbeat's beat). ``cfg`` weighs the bar and
    time-signature heads and sets the two penalties (tuned on validation by scripts/eval_meter.py
    --tune). ``return_steps``: also the Q_STEPS index chosen between each pair of beats (a step can
    span more than a bar, which the states alone do not show: give both to ``grid_from_states``).
    ``logp_pos`` [K, N_BAR_POS] (a model with a position-in-bar head): scored at each state's
    eighth note (``pos_weight``; a state between eighths, a triplet, at the one below it - 1)."""
    ts_idx = tuple(range(len(TS_CLASSES))) if ts_idx is None else tuple(ts_idx)
    states, src, stp, mid, st_c, at0 = _machine(ts_idx)
    n, k = len(states), len(logp_bar)
    pen = np.where(mid, cfg.mid_step_bar, 0.0)
    rows = np.arange(n)
    starts = np.nonzero(at0)[0]

    if logp_pos is not None:
        st_p = np.array([p for _, p in states])
        st_e = np.minimum(st_p // (GRID // 2), N_BAR_POS - 1)
        off_e = np.where(st_p % (GRID // 2) == 0, 0.0, -1.0)

    def emit(j):
        e = cfg.bar_weight * np.where(at0, logp_bar[j, 1], logp_bar[j, 0]) + cfg.ts_weight * logp_ts[j, st_c]
        if logp_pos is not None:
            e = e + cfg.pos_weight * (logp_pos[j, st_e] + off_e)
        return e
    score = emit(0)
    back = np.zeros((k, n), np.int64)
    back_step = np.zeros((k, n), np.int64)
    for j in range(1, k):
        ext = np.append(score, -np.inf)
        cand = ext[src] + logp_step[j - 1][stp] + pen               # [n, moves]
        a = cand.argmax(1)
        new = cand[rows, a]
        arg = src[rows, a]
        step = stp[rows, a]
        # a new time signature at a bar line: from the best bar start of any other time signature
        best = starts[np.argmax(new[starts])]
        alt = new[best] + cfg.ts_change
        better = alt > new[starts]
        new[starts[better]] = alt
        arg[starts[better]] = arg[best]
        step[starts[better]] = step[best]
        back[j], back_step[j] = arg, step
        score = new + emit(j)
    path = [int(np.argmax(score))]
    steps = []
    for j in range(k - 1, 0, -1):
        steps.append(int(back_step[j, path[-1]]))
        path.append(int(back[j, path[-1]]))
    path.reverse()
    steps.reverse()
    out = [states[i] for i in path]
    return (out, steps) if return_steps else out


def grid_from_states(beat_t: np.ndarray, states: list[tuple[int, int]], steps: list[int] | None = None) -> dict:
    """Decoded states -> the grid: {"t": beat times, "q": quarters from the first bar line (the bar
    the first beat is in; a pickup's first beat has q > 0), "bars": time signature of each bar,
    "bar_q": each bar line's q, "bar_t": its time (interpolated where it falls between beats)}.
    ``steps``: the Q_STEPS index decoded between each pair of beats (``decode(...,
    return_steps=True)``): a step longer than the bar passes one bar line or more, all of them
    written. Without, consecutive beats of one time signature are taken as ``(p' - p) mod bar
    length`` apart (a whole bar when equal), which loses the bars of such a step. A new time
    signature starts at its bar line."""
    t = np.asarray(beat_t, np.float64)
    if not states:
        return {"t": t, "q": np.zeros(0), "bars": [], "bar_q": np.zeros(0), "bar_t": np.zeros(0)}
    lens = {c: int(bar_quarters(TS_CLASSES[c]) * GRID) for c, _ in states}
    c0, p0 = states[0]
    acc = [p0]                                   # GRID units from the first bar line
    bars, bar_q = [TS_CLASSES[c0]], [0]
    units = [int(Q_STEPS[k] * GRID) for k in steps] if steps is not None else None
    for j, ((pc, pp), (c, p)) in enumerate(zip(states[:-1], states[1:])):
        L = lens[pc]
        if units is not None:
            d = units[j]
        elif c == pc:
            d = (p - pp) % L or L
        else:
            d = L - pp + p
        acc.append(acc[-1] + d)
        line = acc[-1] - p                       # the bar line of beat j + 1's bar
        while bar_q[-1] + L < line:              # bar lines a long step passes on the way
            bar_q.append(bar_q[-1] + L)
            bars.append(TS_CLASSES[pc])
        if line > bar_q[-1]:
            bar_q.append(line)
            bars.append(TS_CLASSES[c])
    q = np.array(acc, np.float64) / GRID
    bq = np.array(bar_q, np.float64) / GRID
    bar_t = np.interp(bq, q, t) if len(t) > 1 else np.zeros(len(bq))
    if len(t) > 1:  # bar lines before the first beat: extend with the first tempo
        s0 = (t[1] - t[0]) / max(q[1] - q[0], 1e-6)
        bar_t = np.where(bq < q[0], t[0] + (bq - q[0]) * s0, bar_t)
    return {"t": t, "q": q, "bars": bars, "bar_q": bq, "bar_t": bar_t}


def cover_notes(grid: dict, t_first: float, tol: float = 1 / 24) -> dict:
    """The grid with bars of its first time signature added before its first bar line until the
    note at ``t_first`` (seconds) falls inside its bars. A grid starts at the bar line of the first
    tracked beat's bar, so notes played before it (a pickup the tracker had no beat for) would sit
    at negative positions of bar 0, on top of its downbeat; with the bar in front they are a
    partial first bar. Times before the first beat are extended with the first tempo. Applied to
    the grids the labeller gets (scripts/eval_meter.py saves them, src/pipeline.py V2Pipeline),
    after the grid is scored: the scores stay those of the decoded beats."""
    t, q = np.asarray(grid["t"], np.float64), np.asarray(grid["q"], np.float64)
    if len(t) < 2 or not grid["bars"] or t_first >= t[0]:
        return grid
    spq = max(t[1] - t[0], 1e-6) / max(q[1] - q[0], 1e-6)       # seconds per quarter at the start
    qf = q[0] + (t_first - t[0]) / spq
    if qf >= -tol:
        return grid
    L = float(bar_quarters(grid["bars"][0]))
    k = int(math.ceil((-qf - tol) / L))
    shift = k * L
    new_q = np.arange(k, dtype=np.float64) * L
    new_t = t[0] + (new_q - (q[0] + shift)) * spq
    out = dict(grid)
    out["q"] = q + shift
    out["bar_q"] = np.concatenate([new_q, np.asarray(grid["bar_q"], np.float64) + shift])
    out["bar_t"] = np.concatenate([new_t, np.asarray(grid["bar_t"], np.float64)])
    out["bars"] = [grid["bars"][0]] * k + list(grid["bars"])
    return out


CHORD_S = 0.035     # onsets this close together are one chord (refine_grid)


def chord_onsets(onsets: np.ndarray, gap: float = CHORD_S) -> np.ndarray:
    """One time per chord: the onsets grouped wherever consecutive ones are within ``gap``
    seconds, each group's median (a played chord is spread over a few tens of ms)."""
    on = np.sort(np.asarray(onsets, np.float64))
    if not len(on):
        return on
    cut = np.nonzero(np.diff(on) > gap)[0] + 1
    return np.array([float(np.median(g)) for g in np.split(on, cut)])


def _notated_qs(grid: dict) -> np.ndarray:
    """The quarter positions of every notated beat of the grid's bars (beat_layout of each bar's
    time signature), within its anchors' range."""
    q = np.asarray(grid["q"], np.float64)
    bq = list(np.asarray(grid["bar_q"], np.float64)) + [np.inf]
    out = []
    for k, ts in enumerate(grid["bars"]):
        n, qpb = beat_layout(ts)
        for j in range(n):
            x = bq[k] + j * qpb
            if x < bq[k + 1] - 1e-9 and q[0] - 1e-9 <= x <= q[-1] + 1e-9:
                out.append(x)
    return np.array(out, np.float64)


def refine_grid(grid: dict, onsets: np.ndarray, cfg: DecodeConfig) -> dict:
    """The decoded grid with its beat times taken from the notes where they say more than the
    tracker did (Transkun's onsets are within ~5 ms; a tracked beat jitters by ~18 ms, and 12% of
    the beats are missed, the grid then interpolating over two beats under rubato):
      snap   (cfg.snap_s > 0) every anchor (a tracked beat with its quarter position) onto the
             nearest chord (chord_onsets) within snap_s seconds;
      fill   (cfg.fill_tol > 0) every notated beat strictly between two anchors (a missed beat, a
             step longer than a beat) gets an anchor at the nearest chord within fill_tol beats of
             the time the anchors around it give it (nothing played there: no anchor).
    Anchor times stay strictly increasing (a move or an addition that would break that is not
    made); bar times are re-read from the anchors as grid_from_states does. Everything else (bars,
    time signatures, quarter positions of the anchors, the level) is the grid's."""
    t0, q0 = np.asarray(grid["t"], np.float64), np.asarray(grid["q"], np.float64)
    if (cfg.snap_s <= 0 and cfg.fill_tol <= 0) or len(t0) < 2 or len(q0) != len(t0):
        return grid
    chords = chord_onsets(onsets)
    if not len(chords):
        return grid

    def nearest(x: float) -> float:
        j = int(np.searchsorted(chords, x))
        cand = [chords[i] for i in (j - 1, j) if 0 <= i < len(chords)]
        return min(cand, key=lambda c: abs(c - x))

    t = t0.copy()
    moved = 0
    if cfg.snap_s > 0:
        for i in range(len(t)):
            c = nearest(t[i])
            lo = t[i - 1] if i else -np.inf
            hi = t0[i + 1] if i + 1 < len(t) else np.inf
            if abs(c - t[i]) <= cfg.snap_s and lo < c < hi and c != t[i]:
                t[i] = c
                moved += 1
    added_q, added_t = [], []
    if cfg.fill_tol > 0:
        nq = _notated_qs(grid)
        for i in range(len(t) - 1):
            inside = nq[(nq > q0[i] + 1e-6) & (nq < q0[i + 1] - 1e-6)]
            if not len(inside):
                continue
            spq = (t[i + 1] - t[i]) / max(q0[i + 1] - q0[i], 1e-9)     # seconds per quarter here
            prev = t[i]
            for x in inside:
                k = int(np.searchsorted(grid["bar_q"], x, side="right")) - 1
                qpb = beat_layout(grid["bars"][max(k, 0)])[1]
                guess = t[i] + (x - q0[i]) * spq
                c = nearest(guess)
                if abs(c - guess) <= cfg.fill_tol * qpb * spq and prev < c < t[i + 1]:
                    added_q.append(x)
                    added_t.append(c)
                    prev = c
    q = np.concatenate([q0, added_q])
    t = np.concatenate([t, added_t])
    order = np.argsort(q, kind="stable")
    q, t = q[order], t[order]
    out = dict(grid)
    out["t"], out["q"] = t, q
    bq = np.asarray(grid["bar_q"], np.float64)
    bar_t = np.interp(bq, q, t)
    s0 = (t[1] - t[0]) / max(q[1] - q[0], 1e-6)
    out["bar_t"] = np.where(bq < q[0], t[0] + (bq - q[0]) * s0, bar_t)
    out["refined"] = {"snapped": moved, "filled": len(added_q)}
    return out


# ---------------------------------------------------------------------------------------------
# training samples, the model, its loss, prediction and the grid's scores
# ---------------------------------------------------------------------------------------------

def perf_key(perf: dict) -> str:
    """A meter-model performance's key: the performance, #segment when it has several (the key of
    its grids in scripts/eval_meter.py and of its tracked beats in meter/tracked_*.npz)."""
    return perf["performance"] if perf.get("segments", 1) <= 1 else f"{perf['performance']}#{perf['segment']}"


def is_clean(perf: dict) -> bool:
    """Clean performance MIDI (ASAP): its notes get the synthetic transcription noise. Real
    transcriptions (a tracker's beats, or PERiScoPe's ``real_notes``) never do."""
    return perf.get("beats") is None and not perf.get("real_notes", False)


@dataclass
class Augment:
    """Training-time augmentation of a whole performance (as the labeller's src/dataset.py
    AugmentConfig): with only a few hundred pieces a model learns to recognise a piece by its exact
    tempo and pitches instead of reading its meter, so each sample is stretched in time (the notes,
    the ground-truth beats and any tracked beats together), transposed (keeping the notes on the
    keyboard) and its velocities scaled."""
    tempo: tuple = (0.85, 1.18)
    transpose: tuple = (-6, 6)
    velocity: tuple = (0.8, 1.2)
    prob: float = 0.9

    def __call__(self, perf: dict, rng: np.random.Generator) -> dict:
        if rng.random() >= self.prob:
            return perf
        s = float(np.exp(rng.uniform(np.log(self.tempo[0]), np.log(self.tempo[1]))))
        notes = np.array(perf["notes"], np.float64, copy=True)
        if len(notes):
            lo = max(self.transpose[0], int(21 - notes[:, 0].min()))   # shifts keeping every note on the keyboard
            hi = min(self.transpose[1], int(108 - notes[:, 0].max()))
            k = int(rng.integers(lo, hi + 1)) if lo <= hi else 0
            notes[:, 0] += k
            notes[:, 1:3] *= s
            notes[:, 3] = np.clip(np.round(notes[:, 3] * rng.uniform(*self.velocity)), 1, 127)
        out = dict(perf, notes=notes.astype(np.float32), gt=dict(perf["gt"], t=np.asarray(perf["gt"]["t"]) * s))
        for key in ("beats", "aux"):   # tracked beats stretch with the notes
            if perf.get(key) is not None:
                out[key] = (np.asarray(perf[key][0]) * s, perf[key][1])
        return out


def sample(perf: dict, rng: np.random.Generator, noise=None, chunk: int = 0, note_noise=None, augment=None,
           aux_noise=None):
    """One training / validation sample of a performance: its tracked beats (the real tracker's if
    it has them, else the annotated beats corrupted by ``noise``, a src/beats.py BeatNoise; exact
    without), their features and targets, cut to a random ``chunk`` of beats (0 = all).
    ``note_noise``: a src/preprocessing/noise.py NoiseModel for the notes of a clean performance
    (clean MIDI -> like a transcription; ``is_clean``); ``augment``: an ``Augment`` (training);
    ``aux_noise``: a second tracker (a BeatNoise) whose beats join as evidence (``aux_features``):
    the performance's own ``aux`` beats if it has them (a real second tracker), else the
    ground truth corrupted by it; None: no second tracker.
    Returns (features [K, F], targets {bar, step, ts}, beat times [K], note inputs: ``note_inputs``)."""
    if augment is not None:
        perf = augment(perf, rng)
    gt, notes = perf["gt"], perf["notes"]
    if perf.get("beats") is not None:
        bt, bd = perf["beats"]
    elif noise is not None:
        bt, bd = noise(gt["t"], gt["down"], rng)
    else:
        bt, bd = gt["t"], gt["down"]
    bt, bd = np.asarray(bt, np.float64), np.asarray(bd, bool)
    aux = None
    if aux_noise is not None:
        aux = perf["aux"] if perf.get("aux") is not None else aux_noise(gt["t"], gt["down"], rng)
    if note_noise is not None and len(notes) and is_clean(perf):
        store = np.stack([notes[:, 0], notes[:, 1], notes[:, 2] - notes[:, 1], notes[:, 3]], axis=1).astype(np.float32)
        noisy, _ = note_noise(store, np.zeros((0, 2), np.float32), rng)
        notes = np.stack([noisy[:, 0], noisy[:, 1], noisy[:, 1] + noisy[:, 2], noisy[:, 3]], axis=1)
    if chunk and len(bt) > chunk:
        a = int(rng.integers(0, len(bt) - chunk + 1))
        bt, bd = bt[a:a + chunk], bd[a:a + chunk]
    return beat_features(notes, bt, bd, aux), targets(bt, gt), bt, note_inputs(notes, bt)


def collate(samples: list, device="cpu"):
    """[(features [K, F], targets or None, beat times, note inputs)] -> (x [B, K, F], mask [B, K],
    notes (features [B, M, NF], pitch [B, M], beat [B, M]; beat -1 = padding), targets {head:
    [B, K]}, -1 where there is none; tempo: [B], nan where there is none)."""
    import torch
    b = len(samples)
    k = max(len(s[0]) for s in samples)
    m = max(max(len(s[3][0]) for s in samples), 1)
    x = torch.zeros(b, k, samples[0][0].shape[1])
    mask = torch.zeros(b, k, dtype=torch.bool)
    nf = torch.zeros(b, m, N_NOTE_FEATURES)
    npitch = torch.zeros(b, m, dtype=torch.long)
    nbeat = torch.full((b, m), -1, dtype=torch.long)
    tg = {h: torch.full((b, k), IGNORE, dtype=torch.long) for h in ("bar", "step", "ts", "pos")}
    tg["tempo"] = torch.full((b,), float("nan"))
    for i, (f, t, _, (a, p, j)) in enumerate(samples):
        x[i, :len(f)] = torch.from_numpy(f)
        mask[i, :len(f)] = True
        nf[i, :len(a)] = torch.from_numpy(a)
        npitch[i, :len(p)] = torch.from_numpy(p)
        nbeat[i, :len(j)] = torch.from_numpy(j)
        if t is not None:
            for h in ("bar", "step", "ts", "pos"):
                if h in t:
                    tg[h][i, :len(f)] = torch.from_numpy(t[h])
            if "tempo" in t:
                tg["tempo"][i] = float(t["tempo"][0])
    notes = (nf.to(device), npitch.to(device), nbeat.to(device))
    return x.to(device), mask.to(device), notes, {h: v.to(device) for h, v in tg.items()}


@dataclass
class MeterConfig:
    n_features: int = N_BEAT_FEATURES
    d_model: int = 256
    n_heads: int = 8
    layers: int = 6
    ff: int = 1024
    dropout: float = 0.1
    n_steps: int = len(Q_STEPS)
    n_ts: int = len(TS_CLASSES)
    n_note_features: int = 0     # > 0: the note encoder over note_inputs (N_NOTE_FEATURES); 0 = none
    d_note: int = 64
    tempo_head: bool = False     # a piece-level head: this piece's written tempo (log quarters per minute)
    n_pos: int = 0               # a position-in-bar head (N_BAR_POS classes), scored by the decoder; 0 = none

    def to_dict(self):
        return asdict(self)


TEMPO_CENTRE = math.log(116.0)   # the tempo head's estimates start at the training scores' log-mean


def build_meter_model(cfg: dict):
    """The beat-sequence transformer (Scoreformer encoder layers, rotary positions over the beat
    index) with the bar / step / ts heads. With ``n_note_features``, a note encoder (a per-note
    MLP over ``note_inputs`` + a pitch embedding) pooled per beat (mean and max) joins each beat's
    features. Beat features beyond ``n_features`` are ignored (the first meter model had 27).
    With ``tempo_head``, ``"tempo"`` [B, 2]: the piece's written tempo as a log-normal (mean and
    log std of log quarters per minute) from the mean of its beats' states (choose_level)."""
    import torch
    import torch.nn as nn

    from .scoreformer import EncoderLayer, ScoreformerConfig, rope_angles

    c = MeterConfig(**cfg)

    class MeterModel(nn.Module):
        arch = "meter"

        def __init__(self):
            super().__init__()
            self.cfg = c
            enc = ScoreformerConfig(vocab_size=1, d_model=c.d_model, n_heads=c.n_heads, enc_layers=c.layers,
                                    dec_layers=0, ff=c.ff, dropout=c.dropout)
            extra = 2 * c.d_note if c.n_note_features else 0
            if c.n_note_features:
                self.note_in = nn.Linear(c.n_note_features, c.d_note)
                self.pitch = nn.Embedding(128, c.d_note)
                self.note_mlp = nn.Sequential(nn.GELU(), nn.Linear(c.d_note, c.d_note), nn.GELU(),
                                              nn.Linear(c.d_note, c.d_note))
            self.inp = nn.Sequential(nn.Linear(c.n_features + extra, c.d_model), nn.LayerNorm(c.d_model),
                                     nn.Dropout(c.dropout))
            self.layers = nn.ModuleList(EncoderLayer(enc) for _ in range(c.layers))
            self.norm = nn.LayerNorm(c.d_model)
            self.head = nn.Linear(c.d_model, 2 + c.n_steps + c.n_ts)
            if c.n_pos:  # its own layer: a checkpoint without it (--init) loads, the head starts fresh
                self.pos = nn.Linear(c.d_model, c.n_pos)
            if c.tempo_head:  # mean of the beats' states -> (log-qpm offset from the prior's centre, log std)
                self.tempo = nn.Sequential(nn.Linear(c.d_model, c.d_model), nn.GELU(), nn.Linear(c.d_model, 2))
            dh = c.d_model // c.n_heads
            self.register_buffer("freq", 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh)), persistent=False)

        def pool_notes(self, notes, b, k):
            """Per-note embeddings pooled into their beats: [B, K, 2 d_note] (mean, max; 0 for a
            beat without notes)."""
            feat, pitch, beat = notes
            e = self.note_mlp(self.note_in(feat) + self.pitch(pitch))
            valid = beat >= 0
            idx = (beat.clamp(min=0) + torch.arange(b, device=beat.device)[:, None] * k)[valid]
            v = e[valid]
            d = e.shape[-1]
            tot = torch.zeros(b * k, d, device=e.device, dtype=e.dtype).index_add_(0, idx, v)
            cnt = torch.zeros(b * k, device=e.device, dtype=e.dtype).index_add_(0, idx, torch.ones_like(v[:, 0]))
            mx = torch.zeros(b * k, d, device=e.device, dtype=e.dtype).scatter_reduce_(
                0, idx[:, None].expand(-1, d), v, reduce="amax", include_self=False)
            return torch.cat([tot / cnt.clamp(min=1)[:, None], mx], -1).view(b, k, 2 * d)

        def forward(self, x, mask, notes=None):
            pos = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
            cos, sin = rope_angles(pos, self.freq)
            x = x[..., :c.n_features]
            if c.n_note_features:
                if notes is None:
                    raise ValueError("this meter model has a note encoder: pass the note inputs")
                x = torch.cat([x, self.pool_notes(notes, x.shape[0], x.shape[1])], -1)
            h = self.inp(x)
            m = mask[:, None, None, :]
            for layer in self.layers:
                h = layer(h, cos, sin, m)
            h = self.norm(h)
            o = self.head(h)
            out = {"bar": o[..., :2], "step": o[..., 2:2 + c.n_steps], "ts": o[..., 2 + c.n_steps:]}
            if c.n_pos:
                out["pos"] = self.pos(h)
            if c.tempo_head:
                w = mask[..., None].to(h.dtype)
                r = self.tempo((h * w).sum(1) / w.sum(1).clamp(min=1.0))
                out["tempo"] = torch.stack([TEMPO_CENTRE + r[:, 0], r[:, 1].clamp(-4.0, 1.0) - 0.7], -1)
            return out

    return MeterModel()


def meter_loss(out: dict, tg: dict, tempo_weight: float = 1.0):
    """Cross-entropy of the three heads over the beats each has a target for; with a tempo head,
    plus ``tempo_weight`` x its Gaussian negative log-likelihood of the piece's written tempo (log
    quarters per minute) over the samples that have one."""
    import torch
    import torch.nn.functional as F
    total, parts = 0.0, {}
    for k in [h for h in ("bar", "step", "ts", "pos") if h in out and h in tg]:
        y = tg[k]
        m = y >= 0
        loss = F.cross_entropy(out[k][m].float(), y[m]) if m.any() else out[k].sum() * 0.0
        parts[k] = float(loss.detach())
        total = total + loss
    if "tempo" in out and "tempo" in tg:
        y = tg["tempo"].float()
        m = torch.isfinite(y)
        mu, logsd = out["tempo"][:, 0].float(), out["tempo"][:, 1].float()
        nll = (0.5 * ((y - mu) / logsd.exp()) ** 2 + logsd)[m].mean() if m.any() else mu.sum() * 0.0
        parts["tempo"] = float(nll.detach())
        total = total + tempo_weight * nll
    return total, parts


def predict_logp(model, notes: np.ndarray, beat_t: np.ndarray, beat_d: np.ndarray, device="cpu",
                 aux=None) -> dict | None:
    """The model's per-beat log-probabilities {bar, step, ts} for a whole performance (None with
    fewer than 2 beats). ``aux``: a second tracker's (times, downbeats), for a model that reads
    one (``uses_aux``). A ``MeterEnsemble``: every member on every one of its views of the
    performance (``view_of``), their probabilities averaged (``mean_logp``); a single model is
    the one member and the one view (the performance as it is)."""
    import torch
    beat_t = np.asarray(beat_t, np.float64)
    if len(beat_t) < 2:
        return None
    members = list(getattr(model, "members", None) or [model])
    views = tuple(getattr(model, "views", None) or (IDENTITY_VIEW,))
    outs = []
    with torch.no_grad():
        for view in views:
            vn, vt, va = view_of(notes, beat_t, aux, view)
            feats = beat_features(vn, vt, beat_d, va)
            x, mask, nin, _ = collate([(feats, None, vt, note_inputs(vn, vt))], device)
            for m in members:
                m.eval()
                outs.append(logp_of(m(x, mask, nin)))
    return outs[0] if len(outs) == 1 else mean_logp(outs)


# ---------------------------------------------------------------------------------------------
# ensembles: several meter models, and the same performance seen several ways (test-time views)
# ---------------------------------------------------------------------------------------------
# Meter models of one recipe differ a lot from seed to seed (third round: time-signature accuracy
# 0.57-0.64 on PERiScoPe, up to 0.09 apart on the real test set; eighth round: up to ~0.9 MUSTER on
# whole pieces), and a larger model overfits. Averaging the probabilities of several seeds is the
# usual answer to that variance. A view is the performance transposed and stretched in time as the
# training augmentation does (Augment), so the model's answers on it are as valid as on the input;
# averaging over views is a cheaper ensemble of the same model.

IDENTITY_VIEW = (0, 1.0)            # (semitones, time factor): the performance as it is
TTA_VIEWS = ((0, 1.0), (-3, 1.0), (3, 1.0), (0, 0.9), (0, 1.1))
ENSEMBLE_FILE = "ensemble.json"     # in a run folder in place of best.pt: the members and views


def view_of(notes: np.ndarray, beat_t: np.ndarray, aux, view) -> tuple:
    """(notes, beat times, aux) of the performance transposed by ``view[0]`` semitones (kept on the
    keyboard, A0-C8: the shift is clipped as Augment does) and every time multiplied by
    ``view[1]``; the identity view returns the inputs themselves. The beats stay the same beats in
    the same order, so the model's per-beat outputs line up with the original's."""
    k, s = int(view[0]), float(view[1])
    if k == 0 and s == 1.0:
        return notes, beat_t, aux
    n = np.array(notes, np.float64, copy=True)
    if len(n) and k:
        k = min(max(k, int(21 - n[:, 0].min())), int(108 - n[:, 0].max()))
        n[:, 0] += k
    n[:, 1:3] *= s
    a = None if aux is None else (np.asarray(aux[0], np.float64) * s, aux[1])
    return n.astype(np.float32), np.asarray(beat_t, np.float64) * s, a


def mean_logp(lps: list) -> dict:
    """The log of the mean probability of every head over several ``logp_of`` outputs (a mixture
    of the members and views); with a tempo head in all of them, the mixture's mean and std of
    the piece's log written tempo."""
    out = {}
    for h in lps[0]:   # the heads every member has (a position-in-bar head only when all have one)
        if h == "tempo" or any(h not in lp for lp in lps):
            continue
        a = np.stack([lp[h] for lp in lps])
        m = a.max(0)
        out[h] = m + np.log(np.exp(a - m).mean(0))
    if all("tempo" in lp for lp in lps):
        mu = np.array([lp["tempo"][0] for lp in lps], np.float64)
        sd = np.array([lp["tempo"][1] for lp in lps], np.float64)
        out["tempo"] = np.array([mu.mean(), math.sqrt(float((sd ** 2).mean() + mu.var()))])
    return out


class MeterEnsemble:
    """Several meter models read together (``predict_logp``): ``members`` (built, loaded, in eval
    mode), ``views`` ((semitones, time factor) pairs). ``cfg`` is the first member's: every member
    must read the same evidence (all or none a second tracker: ``uses_aux``). Quacks like a model
    where the pipeline and the scripts use one (.cfg, .eval(), .to(), .parameters())."""

    def __init__(self, members: list, views=(IDENTITY_VIEW,), names=None):
        if not members:
            raise ValueError("an ensemble needs at least one member")
        if len({uses_aux(m.cfg) for m in members}) > 1:
            raise ValueError("the members do not read the same evidence (some read a second tracker)")
        self.members = list(members)
        self.views = tuple((int(k), float(s)) for k, s in views)
        self.names = list(names or [])
        self.cfg = members[0].cfg

    def eval(self):
        for m in self.members:
            m.eval()
        return self

    def to(self, device):
        self.members = [m.to(device) for m in self.members]
        return self

    def parameters(self):
        for m in self.members:
            yield from m.parameters()

    @staticmethod
    def spec(path) -> dict:
        import json
        from pathlib import Path
        d = json.loads(Path(path).read_text())
        if not d.get("members"):
            raise ValueError(f"{path}: no members")
        return d


def load_meter(ckpt, device="cpu"):
    """A meter model from its checkpoint, built and in eval mode; a run folder with an
    ``ensemble.json`` and no checkpoint (scripts/make_ensemble.py) gives a ``MeterEnsemble``: its
    members' checkpoints (paths relative to the runs folder, the ensemble folder's parent) and its
    views."""
    from pathlib import Path

    import torch
    ckpt = Path(ckpt)
    ens = ckpt.parent / ENSEMBLE_FILE
    if ens.exists() and not ckpt.exists():
        spec = MeterEnsemble.spec(ens)
        runs = ckpt.parent.parent
        members = [load_meter(runs / p, device) for p in spec["members"]]
        if any(isinstance(m, MeterEnsemble) for m in members):
            raise ValueError(f"{ens}: a member is an ensemble itself (list its members instead)")
        return MeterEnsemble(members, spec.get("views") or (IDENTITY_VIEW,), spec["members"])
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_meter_model(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


def logp_of(out: dict, b: int = 0) -> dict:
    """Item ``b`` of the model's output: each head's log-probabilities per beat, and with a tempo
    head ``"tempo"`` = (mean, std) of the piece's log written tempo."""
    import torch
    lp = {k: torch.log_softmax(v[b].float(), -1).cpu().numpy() for k, v in out.items() if k != "tempo"}
    if "tempo" in out:
        mu, logsd = out["tempo"][b].float().cpu().numpy()
        lp["tempo"] = np.array([mu, np.exp(logsd)])
    return lp


def oracle_logp(beat_t: np.ndarray, gt: dict, eps: float = 1e-4) -> dict | None:
    """The log-probabilities of a perfect meter model on these (tracked) beats: each head certain
    of its training target (``targets``), uniform where the target is unknown (a beat whose
    length is no Q_STEPS value, or outside the annotated span). Decoded like a model's, it gives
    the best grid the tracked beats allow: what a meter model could reach at most (None with
    fewer than 2 beats)."""
    beat_t = np.asarray(beat_t, np.float64)
    if len(beat_t) < 2:
        return None
    tg = targets(beat_t, gt)

    def onehot(y, n):
        lp = np.full((len(y), n), math.log(eps))
        lp[y >= 0, y[y >= 0]] = 0.0
        lp[y < 0] = math.log(1.0 / n)
        return lp
    return {"bar": onehot(tg["bar"], 2), "step": onehot(tg["step"], len(Q_STEPS)),
            "ts": onehot(tg["ts"], len(TS_CLASSES))}


def tempo_logprior(beat_t: np.ndarray, cfg: DecodeConfig) -> np.ndarray:
    """[K, len(Q_STEPS)]: the tempo prior's log-weight of each step between beat k and k + 1 (the
    written tempo it implies, 60 * step / seconds, against cfg's log-normal; the last row is 0, as
    the step head's last row is unused)."""
    t = np.asarray(beat_t, np.float64)
    out = np.zeros((len(t), len(Q_STEPS)))
    if len(t) < 2 or cfg.tempo_weight <= 0:
        return out
    dt = np.clip(np.diff(t), 1e-3, None)
    qpm = 60.0 * np.asarray(Q_STEPS, np.float64)[None, :] / dt[:, None]
    out[:-1] = -0.5 * cfg.tempo_weight * ((np.log(qpm) - cfg.tempo_mu) / cfg.tempo_sigma) ** 2
    return out


LOCAL_WINDOW = 8   # local_tempo_logprior: steps on each side whose median written tempo is the reference


def local_tempo_logprior(beat_t: np.ndarray, q: np.ndarray, cfg: DecodeConfig, window: int = LOCAL_WINDOW) -> np.ndarray:
    """[K, len(Q_STEPS)]: the log-weight of each step between beat k and k + 1 by how far the
    written tempo it implies is from the median of the decoded steps around it (``q``: the first
    decoding's quarters per beat; itself excluded; at least 3 neighbours, else 0)."""
    t, q = np.asarray(beat_t, np.float64), np.asarray(q, np.float64)
    out = np.zeros((len(t), len(Q_STEPS)))
    if len(t) < 4 or cfg.local_weight <= 0 or len(q) != len(t):
        return out
    dt = np.clip(np.diff(t), 1e-3, None)
    dq = np.diff(q)
    qpm_dec = np.where(dq > 0, 60.0 * dq / dt, np.nan)
    steps = np.asarray(Q_STEPS, np.float64)
    for j in range(len(dt)):
        lo, hi = max(0, j - window), min(len(dt), j + window + 1)
        nb = np.concatenate([qpm_dec[lo:j], qpm_dec[j + 1:hi]])
        nb = nb[np.isfinite(nb)]
        if len(nb) < 3:
            continue
        ref = np.log(np.median(nb))
        out[j] = -0.5 * cfg.local_weight * ((np.log(60.0 * steps / dt[j]) - ref) / cfg.local_sigma) ** 2
    return out


def grid_from_logp(beat_t: np.ndarray, lp: dict | None, cfg: DecodeConfig = DecodeConfig()) -> dict:
    """``predict_logp``'s output decoded into a grid (a single bar without it), with cfg's tempo
    prior added to the step head's log-probabilities when its weight is above 0 (and with
    cfg.local_weight above 0, decoded a second time with local_tempo_logprior added), then the
    piece's note-value level chosen (``choose_level``; with cfg's level weights at 0, the grid as
    decoded)."""
    if lp is None:
        return grid_from_states(beat_t, [(0, 0)] * len(beat_t))
    step = lp["step"] + tempo_logprior(beat_t, cfg) if cfg.tempo_weight > 0 else lp["step"]
    states, steps = decode(lp["bar"], step, lp["ts"], cfg, return_steps=True, logp_pos=lp.get("pos"))
    if cfg.local_weight > 0 and len(beat_t) >= 4:
        first = grid_from_states(beat_t, states, steps)
        step = step + local_tempo_logprior(beat_t, first["q"], cfg)
        states, steps = decode(lp["bar"], step, lp["ts"], cfg, return_steps=True, logp_pos=lp.get("pos"))
    return choose_level(grid_from_states(beat_t, states, steps), cfg, lp.get("tempo"))


# ---------------------------------------------------------------------------------------------
# the note-value level of a whole piece
# ---------------------------------------------------------------------------------------------
# The same music can be written at two metrical levels: 3/4 or 3/8, 2/2 or 2/4, 6/4 or 6/8, the
# bar lines in the same places and every note value halved. Scores of one piece write one of them,
# and the decoder gets it wrong for whole pieces (ASAP test: 5 of 14 pieces, among them Chopin's
# first Ballade, 6/4 written 6/8): beat and downbeat F cannot see it, the time signature and every
# onset and length in quarters can. The decoder cannot weigh it with its own heads (the grid it
# returns is the best under them, the other level included), so the choice needs other evidence:
# a piece-level estimate of the written tempo.

LEVELS = (Fraction(1), Fraction(1, 2), Fraction(2))   # as decoded first: ties keep it
LEVEL_TOL = 0.4                                       # |log2(decoded / true quarters)| this close = the right level


def scaled_ts(ts: str, f) -> str | None:
    """The time signature of a bar with every note value multiplied by ``f`` (2: 3/8 -> 3/4; 1/2:
    3/4 -> 3/8), None when that is not one of TS_CLASSES (the labeller and the export know those)."""
    n, d = ts.split("/")
    d2 = Fraction(int(d)) / Fraction(f)
    if d2.denominator != 1 or d2 < 1:
        return None
    s = f"{n}/{int(d2)}"
    return s if s in TS_CLASSES else None


def scale_grid(grid: dict, f) -> dict | None:
    """The grid with the same bar lines and beat times and every note value multiplied by ``f``
    (its quarters, bar positions and time signatures), None when a bar's time signature has no
    counterpart (``scaled_ts``). ``level`` records the factor against the decoded grid."""
    f = Fraction(f)
    if f == 1:
        return grid
    bars = [scaled_ts(b, f) for b in grid["bars"]]
    if not bars or any(b is None for b in bars):
        return None
    return dict(grid, q=np.asarray(grid["q"], np.float64) * float(f),
                bar_q=np.asarray(grid["bar_q"], np.float64) * float(f), bars=bars,
                level=float(Fraction(grid.get("level", 1.0)).limit_denominator(8) * f))


def piece_qpm(grid: dict) -> float | None:
    """The grid's written tempo: the median over its beats of quarters per minute (beats at least
    50 ms apart; None with fewer than 2)."""
    t, q = np.asarray(grid["t"], np.float64), np.asarray(grid["q"], np.float64)
    if len(t) < 2:
        return None
    dt, dq = np.diff(t), np.diff(q)
    ok = (dt > 0.05) & (dq > 0)
    return float(np.median(60.0 * dq[ok] / dt[ok])) if ok.any() else None


def level_ratio(grid: dict, gt: dict) -> float | None:
    """Decoded quarters per true quarter over the span both cover (about 1: the right level; 1/2:
    every note value halved; None when fewer than 4 decoded beats fall inside the annotations)."""
    t = np.asarray(grid["t"], np.float64)
    inside = (t >= gt["t"][0] - MATCH_TOL) & (t <= gt["t"][-1] + MATCH_TOL)
    if inside.sum() < 4:
        return None
    tq = quarters_at(t[inside], gt)
    dq = np.asarray(grid["q"], np.float64)[inside]
    true_span = tq[-1] - tq[0]
    return float((dq[-1] - dq[0]) / true_span) if true_span > 1e-6 else None


def level_scores(grid: dict, cfg: DecodeConfig, tempo=None) -> dict:
    """{factor: score} of every level the grid can take (``LEVELS``, ``scale_grid``): the written
    tempo it implies (``piece_qpm`` x factor) under the log-normal prior (cfg.level_weight) and under
    the model's estimate for this piece (``tempo`` = (log-qpm mean, std) from the tempo head;
    cfg.level_model_weight), plus cfg.level_stay for the decoded one."""
    qpm = piece_qpm(grid)
    out = {}
    if qpm is None:
        return {Fraction(1): 0.0}
    for f in LEVELS:
        if scale_grid(grid, f) is None:
            continue
        x = math.log(qpm * float(f))
        s = -0.5 * cfg.level_weight * ((x - cfg.level_mu) / cfg.level_sigma) ** 2
        if tempo is not None and cfg.level_model_weight > 0:
            mu, sd = float(tempo[0]), max(float(tempo[1]), 1e-3)
            s += cfg.level_model_weight * (-0.5 * ((x - mu) / sd) ** 2)
        if f == 1:
            s += cfg.level_stay
        out[f] = s
    return out


def choose_level(grid: dict, cfg: DecodeConfig, tempo=None) -> dict:
    """The grid at the level ``level_scores`` puts first (the decoded one on a tie, and always with
    cfg's level weights at 0); ``level`` records the factor (1.0: as decoded)."""
    if cfg.level_weight <= 0 and (cfg.level_model_weight <= 0 or tempo is None):
        return dict(grid, level=1.0)
    scores = level_scores(grid, cfg, tempo)
    best = max(scores, key=lambda f: (scores[f], f == 1))
    return dict(scale_grid(grid, best), level=float(best)) if best != 1 else dict(grid, level=1.0)


def predict_grid(model, notes: np.ndarray, beat_t: np.ndarray, beat_d: np.ndarray, device="cpu",
                 cfg: DecodeConfig = DecodeConfig(), aux=None) -> dict:
    """A whole performance: the model's per-beat log-probabilities, decoded into a consistent grid
    (``decode``, ``grid_from_states``)."""
    return grid_from_logp(beat_t, predict_logp(model, notes, beat_t, beat_d, device, aux), cfg)


def grid_beats(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    """The grid's notated beats: (times, bar start?) of every beat of every bar (beat_layout of
    its time signature), timed through the grid's (t, q) anchors."""
    q, t = grid["q"], grid["t"]
    if len(q) < 2:
        return np.zeros(0), np.zeros(0, bool)
    bq = list(grid["bar_q"]) + [np.inf]  # the last bar runs to the last beat (the q range below)
    qs, down = [], []
    for k, ts in enumerate(grid["bars"]):
        n, qpb = beat_layout(ts)
        for j in range(n):
            x = bq[k] + j * qpb
            if x < bq[k + 1] - 1e-9 and q[0] - 1e-9 <= x <= q[-1] + 1e-9:
                qs.append(x)
                down.append(j == 0)
    return np.interp(np.array(qs), q, t), np.array(down, bool)


def grid_scores(grid: dict, gt: dict) -> dict:
    """How close a decoded grid is to the annotated one: beat and downbeat F-measure of its
    notated beats (``grid_beats``) against the annotated beats and bar starts, and the share of
    annotated bars whose time signature it has right (the bar its downbeat falls in)."""
    from .beats import f_measure
    bt, bd = grid_beats(grid)
    lo, hi = gt["t"][0] - MATCH_TOL, gt["t"][-1] + MATCH_TOL
    keep = (bt >= lo) & (bt <= hi)
    bt, bd = bt[keep], bd[keep]
    downs = gt["t"][gt["down"]]
    ts_ok, n = 0, 0
    bar_t = np.asarray(grid["bar_t"])
    for k in np.nonzero(gt["down"])[0]:
        j = int(np.searchsorted(bar_t, gt["t"][k] + MATCH_TOL)) - 1
        if 0 <= j < len(grid["bars"]):
            ts_ok += grid["bars"][j] == gt["ts_names"][k]
        n += 1
    r = level_ratio(grid, gt)
    return {"beat_f": f_measure(gt["t"], bt), "downbeat_f": f_measure(downs, bt[bd]),
            "ts_acc": ts_ok / max(n, 1),
            # the note-value level (level_ratio): 1 right, 0 halved or doubled, nan when it cannot be told
            "level_ok": float(abs(math.log2(r)) <= LEVEL_TOL) if r and r > 0 else float("nan")}
