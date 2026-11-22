"""Beats as an input: ground-truth beat grids, beat-tracker output, tracker errors, note features.

The score model's remaining gap is rhythm placement: even given the true previous bars it puts ~40%
of onsets at the wrong position, while the beat-snap + XGBoost baseline, given the annotated beats,
reaches onset F1 0.74. So the model gets the beats too: from a beat tracker at inference (Beat This!,
on the audio, scripts/track_beats.py), and in training from the ground truth corrupted like the
tracker's real errors (``BeatNoise``, fitted by scripts/fit_beat_noise.py).

A beat list is (times [K] seconds ascending, down [K] bool: the beat starts a bar). Every source
gives the same form, so features never depend on where the beats came from:

  ground truth   ``window_grid``: the score's beats of one window, timed by the same note alignment
                 as the metrical labels (src/metrical.py fit_alignment): each bar of the target gets
                 its beats (``beat_layout``: quarter notes in 4/4, dotted quarters in 6/8, ...), and
                 the alignment's piecewise-linear time map gives their performed times.
                 ``merge_performance`` joins a performance's windows by global bar number, keeping
                 for every beat the window that timed it best (inside its aligned notes, not
                 extrapolated). scripts/build_beats.py stores one list per performance and split.
  tracker        scripts/track_beats.py: Beat This! on the MAESTRO audio, shifted onto the ASAP
                 timeline like the transcriptions.

Per-note features (``beat_features``, ``BEAT_FEATURES``), appended to the note features when a
model is trained with beats:
  beat_sin, beat_cos   the onset's phase between the surrounding beats, on the unit circle
  bar_sin, bar_cos     its position in the bar (beats since the downbeat + phase) / beats per bar;
                       0, 0 when the downbeats are unknown
  log_ibi              local beat period: log2(seconds per beat / 0.5) / 2, clipped to [-1, 1]
  beats_per_bar        beats between downbeats / 4 (0 when unknown)
  has_beats            1 where the note lies inside the beat list (a gap of more than 3 median beat
                       periods counts as no beats), else every beat feature is 0

Storage: ``<split>/beats_<source>.npz`` (``save_beats`` / ``load_beats``), performance -> list, in
absolute performance time (seconds), the frame of the stored notes plus ``perf_t0``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .metrical import fit_alignment
from .preprocessing.detokenize import parse_tokens

BEAT_FEATURES = ["beat_sin", "beat_cos", "bar_sin", "bar_cos", "log_ibi", "beats_per_bar", "has_beats"]
N_BEAT_FEATURES = len(BEAT_FEATURES)
MIN_PAIRS = 4          # aligned notes needed before a window's beat grid is trusted
GAP = 3.0              # a beat interval longer than this many median periods = no beats there
MATCH_TOL = 0.07       # seconds: the standard beat-tracking tolerance (F-measure, MIREX)
SOURCES = ("gt", "tracked")


# ---------------------------------------------------------------------------------------------
# ground truth from the score
# ---------------------------------------------------------------------------------------------

def beat_layout(ts: str) -> tuple[int, float]:
    """(beats per bar, quarter notes per beat) of a time signature, as ASAP's beat annotations
    count them (all 1,067 annotated performances follow it): compound meters, a numerator that is
    a multiple of 3 above 3, count dotted beats (6/8 -> 2, 9/8 -> 3, 12/8 -> 4, 6/4 -> 2,
    24/16 -> 8); every other meter counts its numerator (4/4 -> 4, 3/4 -> 3, 3/8 -> 3, 2/2 -> 2)."""
    try:
        n, d = (int(x) for x in ts.split("/"))
    except (ValueError, AttributeError):
        return 4, 1.0
    if n <= 0 or d <= 0:
        return 4, 1.0
    if n % 3 == 0 and n > 3:
        return n // 3, 3 * 4 / d
    return n, 4 / d


def _time_map(pairs, sq, pt, max_ratio: float = 3.0):
    """Knots (q, t) of the piecewise-linear non-decreasing map through the matches, and the
    overall slope used beyond them (seconds per quarter), as in src/metrical.py _map_through, but
    robust: a wrong pairing bends the map into an impossible tempo (one quarter note in 40 ms
    next to quarters of 0.4 s), so knots whose two neighbouring segments are both more than
    ``max_ratio`` times faster or slower than the median tempo are removed one at a time, worst
    first. (The metrical labels only need the pairs to be right; the beat grid needs the map.)"""
    mq = np.array([sq[s] for s in pairs.values()])
    mt = np.array([pt[k] for k in pairs])
    uq, inv = np.unique(mq, return_inverse=True)
    ut = np.array([mt[inv == k].mean() for k in range(len(uq))])
    while len(uq) > 3:
        slope = np.diff(ut) / np.diff(uq)
        med = np.median(slope[slope > 0]) if (slope > 0).any() else 0.5
        dev = np.abs(np.log(np.maximum(slope, 1e-4) / med))  # tempo deviation of each segment
        bad = dev > np.log(max_ratio)
        # a knot is suspect when both segments around it are off (an end knot: its one segment)
        score = np.zeros(len(uq))
        score[0], score[-1] = dev[0] * bad[0], dev[-1] * bad[-1]
        score[1:-1] = np.where(bad[:-1] & bad[1:], dev[:-1] + dev[1:], 0.0)
        worst = int(np.argmax(score))
        if score[worst] <= 0:
            break
        uq, ut = np.delete(uq, worst), np.delete(ut, worst)
    ut = np.maximum.accumulate(ut)
    slope = max((ut[-1] - ut[0]) / (uq[-1] - uq[0]), 1e-3) if len(uq) > 1 else 0.5
    return uq, ut, slope


def window_grid(onsets: np.ndarray, pitches: np.ndarray, tokens: list[str], max_extrap: float = 1.0) -> dict | None:
    """The score's beats of one window, timed in the window's frame. Returns arrays bar (window bar
    index, 0 = the target's first bar), beat (index in its bar), t (seconds), extrap (quarters
    outside the aligned notes; 0 = interpolated), ts (its bar's time signature), or None if fewer
    than MIN_PAIRS notes align. Beats more than ``max_extrap`` quarter notes outside the aligned notes are left out."""
    fit = fit_alignment(np.asarray(onsets, np.float64), np.asarray(pitches), tokens)
    if fit is None or len(fit[3]) < MIN_PAIRS:
        return None
    _, sq, _, pairs, pt = fit
    uq, ut, slope = _time_map(pairs, sq, pt)
    bars, beats, qs, tss = [], [], [], []
    start = 0.0
    for j, b in enumerate(parse_tokens(tokens)):
        n, qpb = beat_layout(b.ts)
        for k in range(n):
            bars.append(j)
            beats.append(k)
            qs.append(start + k * qpb)
            tss.append(b.ts or "4/4")
        try:
            num, den = (int(x) for x in b.ts.split("/"))
            start += num * 4 / den
        except (ValueError, AttributeError, ZeroDivisionError):
            start += 4.0
    q = np.asarray(qs, np.float64)
    t = np.interp(q, uq, ut)
    lo, hi = q < uq[0], q > uq[-1]
    t[lo] = ut[0] + slope * (q[lo] - uq[0])
    t[hi] = ut[-1] + slope * (q[hi] - uq[-1])
    extrap = np.maximum(uq[0] - q, 0) + np.maximum(q - uq[-1], 0)
    keep = extrap <= max_extrap  # beyond the aligned notes the map is a guess: leave those out
    return {"bar": np.asarray(bars, np.int64)[keep], "beat": np.asarray(beats, np.int64)[keep],
            "t": t[keep], "extrap": extrap[keep], "ts": np.asarray(tss, object)[keep]}


def merge_performance(windows) -> tuple[np.ndarray, np.ndarray]:
    """One beat list for a performance from its windows' grids: ``windows`` = [(bar_start,
    perf_t0, grid or None)]. Each beat (global bar, beat in bar) comes from the window that timed it
    with the least extrapolation (windows overlap at their edges in training splits); times must
    increase, so a beat that does not is dropped. Returns (times, down)."""
    times, down, _, _, _ = merge_performance_bars(windows)
    return times, down


def merge_performance_bars(windows):
    """``merge_performance`` with every beat's place in the score: (times, down, global bar, beat
    in its bar, its bar's time signature; "4/4" from a grid without them)."""
    best = {}
    for bar_start, t0, g in windows:
        if g is None:
            continue
        ts = g.get("ts")
        ts = ["4/4"] * len(g["t"]) if ts is None else ts
        for b, k, t, e, s in zip(g["bar"], g["beat"], g["t"], g["extrap"], ts):
            key = (int(bar_start) + int(b), int(k))
            if key not in best or e < best[key][1]:
                best[key] = (float(t0) + float(t), float(e), str(s))
    times, down, bars, beats, tss, last = [], [], [], [], [], -np.inf
    for key in sorted(best):
        t = best[key][0]
        if t <= last:
            continue
        times.append(t)
        down.append(key[1] == 0)
        bars.append(key[0])
        beats.append(key[1])
        tss.append(best[key][2])
        last = t
    return (np.asarray(times, np.float64), np.asarray(down, bool), np.asarray(bars, np.int64),
            np.asarray(beats, np.int64), tss)


# ---------------------------------------------------------------------------------------------
# per-note features
# ---------------------------------------------------------------------------------------------

def _bar_positions(down: np.ndarray):
    """(beats since the last downbeat, beats per bar) for every beat; -1 where unknown. Beats
    before the first downbeat count back from it with the first bar's length; the last bar takes
    the previous bar's length."""
    n = len(down)
    pos = np.full(n, -1, np.int64)
    length = np.full(n, -1, np.int64)
    d = np.nonzero(down)[0]
    if len(d) < 2:
        return pos, length
    for a, b in zip(d[:-1], d[1:]):
        pos[a:b] = np.arange(b - a)
        length[a:b] = b - a
    last = d[-1]
    pos[last:] = np.arange(n - last)
    length[last:] = length[d[-2]]
    first_len = d[1] - d[0]
    pos[:d[0]] = (np.arange(d[0]) - d[0]) % first_len
    length[:d[0]] = first_len
    over = pos >= length  # a final "bar" running on past its length: position unknown
    pos[over], length[over] = -1, -1
    return pos, length


def beat_features(onsets: np.ndarray, times: np.ndarray, down: np.ndarray) -> np.ndarray:
    """[N, N_BEAT_FEATURES] for notes at ``onsets`` (same time frame as ``times``)."""
    onsets = np.asarray(onsets, np.float64)
    out = np.zeros((len(onsets), N_BEAT_FEATURES), np.float32)
    times = np.asarray(times, np.float64)
    down = np.asarray(down, bool)
    if len(times) < 2 or not len(onsets):
        return out
    ibi = np.diff(times)
    med = float(np.median(ibi))
    if med <= 0:
        return out
    bx = np.concatenate([[times[0] - ibi[0]], times, [times[-1] + ibi[-1]]])  # one beat past each end
    dx = np.concatenate([[False], down, [False]])
    pos, length = _bar_positions(dx)
    if len(np.nonzero(down)[0]) >= 2:  # the extrapolated first beat precedes beat 0
        first = np.nonzero(dx)[0][0]
        if pos[0] < 0 and length[first] > 0:
            pos[0], length[0] = (0 - first) % length[first], length[first]
    k = np.searchsorted(bx, onsets, side="right") - 1
    inside = (k >= 0) & (k < len(bx) - 1)
    k = np.clip(k, 0, len(bx) - 2)
    span = bx[k + 1] - bx[k]
    ok = inside & (span > 0) & (span <= GAP * med)
    phase = np.where(ok, (onsets - bx[k]) / np.where(span > 0, span, 1), 0.0)
    has_bar = ok & (pos[k] >= 0) & (length[k] > 0)
    bar = np.where(has_bar, (pos[k] + phase) / np.where(length[k] > 0, length[k], 1), 0.0)
    out[:, 0] = np.sin(2 * np.pi * phase)
    out[:, 1] = np.cos(2 * np.pi * phase)
    out[:, 2] = np.where(has_bar, np.sin(2 * np.pi * bar), 0.0)
    out[:, 3] = np.where(has_bar, np.cos(2 * np.pi * bar), 0.0)
    out[:, 4] = np.clip(np.log2(np.maximum(span, 1e-3) / 0.5), -2, 2) / 2
    out[:, 5] = np.where(has_bar, np.minimum(length[k], 8) / 4, 0.0)
    out[:, 6] = 1.0
    out[~ok] = 0.0
    return out


def window_beats(times: np.ndarray, down: np.ndarray, t0: float, lo: float, hi: float, margin: float = 4.0):
    """The part of a performance's beat list around one window, in the window's frame (seconds
    from ``t0``): beats from ``lo - margin`` to ``hi + margin`` (absolute seconds), so notes near
    the edges still lie between two beats."""
    if times is None or not len(times):
        return np.zeros(0), np.zeros(0, bool)
    sel = (times >= lo - margin) & (times <= hi + margin)
    return times[sel] - t0, down[sel]


# ---------------------------------------------------------------------------------------------
# tracker errors
# ---------------------------------------------------------------------------------------------

DOWN, HALF, OTHER, EXTRA = 0, 1, 2, 3  # a beat's place in its bar (bar_classes); EXTRA: not a true beat
PHASE_BINS = 10               # histogram of where spurious beats fall between two true beats
PROFILE = ("miss", "extra", "offset", "jitter", "down_hit", "down_half", "down_other", "down_extra")
FACTORS = (2, 3, 4)           # tempo octave errors: beats k times too fast / too slow


def bar_length(down: np.ndarray) -> int:
    """Median beats per bar of a beat list (0: fewer than two downbeats)."""
    d = np.nonzero(np.asarray(down, bool))[0]
    return int(np.median(np.diff(d))) if len(d) >= 2 else 0


def bar_classes(down: np.ndarray) -> np.ndarray:
    """Each beat's place in its bar: DOWN (a downbeat), HALF (the middle beat of a bar with an even
    number of beats: beat 3 in 4/4, beat 2 in 2/4) or OTHER. A bar runs from one downbeat to the
    next; before the first and after the last, the median bar length is assumed. Without
    downbeats every beat is OTHER."""
    down = np.asarray(down, bool)
    n = len(down)
    out = np.full(n, OTHER, np.int64)
    d = np.nonzero(down)[0]
    if not len(d):
        return out
    out[d] = DOWN
    med = int(np.median(np.diff(d))) if len(d) >= 2 else 0
    i = np.arange(n)
    p = np.searchsorted(d, i, side="right") - 1          # the bar's downbeat (-1: before the first)
    after = p >= 0
    k = np.where(after, i - d[np.maximum(p, 0)], med - (d[0] - i))  # beats since the downbeat
    nxt = np.minimum(p + 1, len(d) - 1)
    bar = np.where(after & (p + 1 < len(d)), d[nxt] - d[np.maximum(p, 0)], med)
    half = ~down & (bar > 0) & (bar % 2 == 0) & (k == bar // 2)
    out[half] = HALF
    return out


@dataclass
class BeatNoise:
    """The errors a beat tracker makes, applied to a ground-truth beat list (training, and the
    simulated-tracker test condition). Fitted by scripts/fit_beat_noise.py from Beat This! against
    the ASAP annotations (``compare`` per performance, ``fit_noise``).
      p_double / p_half   tempo octave errors, per beat list: too fast (k beats per true beat) /
                          too slow (every k-th beat) over one stretch of the list. With
                          octave_profiles (fitted: one per real wrong-tempo performance, its kind,
                          factor k, the fraction of the piece it covers, its miss rate and its
                          four downbeat rates below) a random profile of that kind is replayed,
                          with no spurious beats on top (they are the error); without, k is
                          drawn from double_k / half_k (probabilities of k = FACTORS; Beat This!
                          also tracks the eighth notes of compound meters, k = 3), over the whole
                          list, with the octave_down rates
      p_offbeat           beats on the off-beats (half a period late), per beat list
      miss                share of the true beats it misses
      extra               share of its beats that are spurious; they fall between two true beats
                          at a phase drawn from ``extra_phase`` (PHASE_BINS probabilities; None =
                          uniform): Beat This! adds them mostly mid-beat, i.e. local subdivision
      offset, jitter      timing of the beats it finds: mean and standard deviation (seconds)
      down_hit            chance a true downbeat is marked as a downbeat
      down_half           ... a half-bar beat (beat 3 of 4, 2 of 2)    } Beat This! (without its
      down_other          ... any other beat                           } DBN) imposes no bar
      down_extra          ... a spurious beat                          } length: each is its own peak
      octave_down         the same four rates when the tempo octave is wrong (None: as above). It
                          keeps finding true downbeats then; each beat is classed by the true bar,
                          and the beats a too-fast tracker adds count as spurious
      octave_profiles     see p_double / p_half (None: not fitted)
      profiles            per-performance values of PROFILE (the fitted right-tempo
                          performances, with their bar length "bar"), or None. When set, each beat
                          list draws one profile for all of the above but the octave errors, from
                          the performances with its bar length if there are any, so a simulated
                          tracker is as good or as bad on a piece as the real one was (the rates
                          vary a lot between pieces, and with the meter); the scalars are then the
                          profiles' means. Octave profiles are drawn the same way."""
    p_double: float = 0.18
    p_half: float = 0.06
    double_k: list = field(default_factory=lambda: [1.0, 0.0, 0.0])
    half_k: list = field(default_factory=lambda: [1.0, 0.0, 0.0])
    p_offbeat: float = 0.0
    miss: float = 0.12
    extra: float = 0.13
    offset: float = 0.0
    jitter: float = 0.02
    down_hit: float = 0.88
    down_half: float = 0.38
    down_other: float = 0.20
    down_extra: float = 0.34
    octave_down: list | None = None
    octave_profiles: list | None = None
    extra_phase: list | None = None
    profiles: list | None = None
    fitted: bool = False
    note: str = "defaults (Beat This! on ASAP / MAESTRO, rounded), not fitted"

    @classmethod
    def from_dict(cls, d: dict) -> "BeatNoise":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def from_json(cls, path) -> "BeatNoise":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path, extra: dict | None = None):
        d = asdict(self)
        d.update(extra or {})
        Path(path).write_text(json.dumps(d, indent=1))

    @staticmethod
    def _factor(probs, rng) -> int:
        p = np.asarray(probs if probs else [1.0], np.float64)
        return int(FACTORS[int(rng.choice(len(p), p=p / p.sum()))])

    @staticmethod
    def _pick(profiles: list, bar: int, rng):
        """A random profile, from those of this bar length if there are any."""
        same = [q for q in profiles if q.get("bar") == bar]
        pool = same or profiles
        return pool[int(rng.integers(len(pool)))]

    def _octave(self, kind: str, bar: int, rng):
        """Factor, fraction of the list, downbeat rates and miss rate of one tempo octave error
        (rates / miss None: the right-tempo ones)."""
        profs = [q for q in (self.octave_profiles or []) if q.get("octave") == kind]
        if profs:
            q = self._pick(profs, bar, rng)
            return (int(q["factor"]), float(q["frac"]),
                    [float(q[k]) for k in ("down_hit", "down_half", "down_other", "down_extra")],
                    float(q.get("miss", 0.0)))
        return self._factor(self.double_k if kind == "double" else self.half_k, rng), 1.0, self.octave_down, None

    def rates(self, rng: np.random.Generator, bar: int = 0) -> dict:
        """The PROFILE values for one beat list of ``bar`` beats per bar: a random fitted profile
        (of that bar length if there is one), or the scalars."""
        if self.profiles:
            prof = self._pick(self.profiles, bar, rng)
            return {k: float(prof.get(k, getattr(self, k))) for k in PROFILE}
        return {k: float(getattr(self, k)) for k in PROFILE}

    def __call__(self, times: np.ndarray, down: np.ndarray, rng: np.random.Generator):
        times = np.asarray(times, np.float64).copy()
        down = np.asarray(down, bool).copy()
        if len(times) < 3:
            return times, down
        bar = bar_length(down)
        r = self.rates(rng, bar)
        p_class = np.array([r["down_hit"], r["down_half"], r["down_other"], r["down_extra"]], np.float64)
        cls = bar_classes(down)  # each beat's place in the true bar, carried through the tempo errors
        u = rng.random()
        kind = "double" if u < self.p_double else "half" if u < self.p_double + self.p_half else None
        if kind is not None:
            k, frac, rates, miss = self._octave(kind, bar, rng)
            times, cls = octave_stretch(times, cls, kind, k, frac, rng)
            if rates is not None:
                p_class = np.asarray(rates, np.float64)
            if miss is not None:  # a replayed wrong-tempo performance: its own misses, and its
                r["miss"], r["extra"] = miss, 0.0  # spurious beats are the octave error itself
        if len(times) >= 3 and rng.random() < self.p_offbeat:
            times = (times[:-1] + times[1:]) / 2
            cls = np.full(len(times), EXTRA)
            if self.octave_down:
                p_class = np.asarray(self.octave_down, np.float64)
        # downbeats: each beat marked independently, by its place in the true bar
        down = rng.random(len(times)) < p_class[cls]
        if r["miss"] > 0 and len(times) > 2:
            keep = rng.random(len(times)) >= r["miss"]
            if keep.sum() >= 2:
                times, down = times[keep], down[keep]
        if r["extra"] > 0 and len(times) >= 2:
            e = min(r["extra"], 0.9)
            n = int(rng.binomial(len(times), e / (1 - e)))  # so that a share `extra` of the beats is spurious
            if n:
                j = rng.integers(len(times) - 1, size=n)
                if self.extra_phase:
                    p = np.asarray(self.extra_phase, np.float64)
                    b = rng.choice(len(p), size=n, p=p / p.sum())
                    phase = (b + rng.random(n)) / len(p)
                else:
                    phase = rng.random(n)
                gap = times[j + 1] - times[j]
                edge = np.minimum(0.5, 0.051 / np.maximum(gap, 1e-9))  # never within 50 ms of a beat
                phase = np.clip(phase, edge, 1 - edge)
                times = np.concatenate([times, times[j] + phase * gap])
                down = np.concatenate([down, rng.random(n) < p_class[EXTRA]])
        times = times + r["offset"]
        if r["jitter"] > 0:
            times = times + rng.normal(0, r["jitter"], len(times))
        order = np.argsort(times, kind="stable")
        times, down = times[order], down[order]
        keep = np.concatenate([[True], np.diff(times) > 0.05])  # no two beats within 50 ms
        return times[keep], down[keep]


def octave_stretch(times: np.ndarray, cls: np.ndarray, kind: str, k: int, frac: float, rng):
    """A tempo octave error over one contiguous stretch covering ``frac`` of the beat intervals:
    "double" puts k - 1 more beats evenly into each interval (class EXTRA), "half" keeps every k-th
    beat. Returns the new times and classes, in time order."""
    n = len(times)
    if n < 2:
        return times, cls
    m = int(np.clip(round(frac * (n - 1)), 1, n - 1))   # intervals affected
    a = int(rng.integers(0, n - m)) if n - m > 0 else 0
    if kind == "double":
        seg = times[a:a + m + 1]
        new = (seg[:-1, None] + (np.arange(1, k) / k) * np.diff(seg)[:, None]).ravel()
        times = np.concatenate([times, new])
        cls = np.concatenate([cls, np.full(len(new), EXTRA)])
        order = np.argsort(times, kind="stable")
        return times[order], cls[order]
    keep = np.ones(n, bool)
    idx = np.arange(a, a + m + 1)
    keep[idx[(idx - a) % k != 0]] = False
    return times[keep], cls[keep]


def match(ref: np.ndarray, est: np.ndarray, tol: float = MATCH_TOL) -> list[tuple[int, int]]:
    """One-to-one matching of estimated to reference times within ``tol`` (nearest first)."""
    if not len(ref) or not len(est):
        return []
    d = np.abs(ref[:, None] - est[None, :])
    cand = np.argwhere(d <= tol)
    cand = cand[np.argsort(d[cand[:, 0], cand[:, 1]], kind="stable")]
    used_r, used_e, out = set(), set(), []
    for r, e in cand:
        if r in used_r or e in used_e:
            continue
        used_r.add(int(r))
        used_e.add(int(e))
        out.append((int(r), int(e)))
    return out


def f_measure(ref: np.ndarray, est: np.ndarray, tol: float = MATCH_TOL) -> float:
    """Beat-tracking F-measure (MIREX): matches within ``tol`` seconds."""
    if not len(ref) and not len(est):
        return 1.0
    if not len(ref) or not len(est):
        return 0.0
    m = len(match(ref, est, tol))
    return 2 * m / (len(ref) + len(est))


DOWNBEAT_WEIGHT = 0.5  # grid_agreement: the tracker's downbeats are less reliable than its beats


def grid_agreement(onsets: np.ndarray, pitches: np.ndarray, tokens: list[str], beat_t: np.ndarray,
                   beat_d: np.ndarray) -> float | None:
    """How well a decoded window's beats agree with the beats the tracker found: the window's
    written beats (``beat_layout`` of each bar's time signature, so a 6/8 reading puts its beats
    on dotted quarters and a 3/4 one on quarters) timed through the alignment of its notes to the
    performed ones (``window_grid``), against the tracked beats, over the span of the window's
    notes: beat F-measure + DOWNBEAT_WEIGHT x downbeat F-measure (bar lines vs tracked downbeats).
    None when it cannot be told (no beats there, too few aligned notes). Window frame seconds;
    ``onsets`` / ``pitches`` = the window's own notes (carried-over ones have onsets < 0)."""
    beat_t, beat_d = np.asarray(beat_t, np.float64), np.asarray(beat_d, bool)
    on = np.asarray(onsets, np.float64)
    live = on[on >= 0]
    if len(live) < MIN_PAIRS or len(beat_t) < 2:
        return None
    lo, hi = float(live.min()) - MATCH_TOL, float(live.max()) + MATCH_TOL
    inside = (beat_t >= lo) & (beat_t <= hi)
    if inside.sum() < 2:
        return None
    grid = window_grid(on, np.asarray(pitches), tokens)
    if grid is None:
        return None
    g_in = (grid["t"] >= lo) & (grid["t"] <= hi)
    gt_, gd = grid["t"][g_in], (grid["beat"] == 0)[g_in]
    tt, td = beat_t[inside], beat_d[inside]
    score = f_measure(tt, gt_)
    if td.any():  # no tracked downbeats here: the bar lines cannot be checked (same for every reading)
        score += DOWNBEAT_WEIGHT * f_measure(tt[td], gt_[gd])
    return float(score)


def compare(gt_t, gt_d, tr_t, tr_d) -> dict:
    """How a tracked beat list differs from the ground truth, in BeatNoise's terms: the tempo octave
    (and its factor), how often each kind of beat (bar_classes; spurious ones too) was marked as a
    downbeat ("down": class -> [marked, count]), and at the right tempo missed / spurious beats,
    timing errors and where the spurious beats fall. Only the tracked beats within the annotated span count:
    an ASAP performance can be one piece of a longer MAESTRO recording (101 of the 418 hold two or
    more, e.g. a prelude and its fugue), and the tracker hears all of it."""
    gt_t, tr_t = np.asarray(gt_t, np.float64), np.asarray(tr_t, np.float64)
    gt_d, tr_d = np.asarray(gt_d, bool), np.asarray(tr_d, bool)
    if len(gt_t):
        span = (tr_t >= gt_t[0] - MATCH_TOL) & (tr_t <= gt_t[-1] + MATCH_TOL)
        tr_t, tr_d = tr_t[span], tr_d[span]
    out = {"beat_f": f_measure(gt_t, tr_t), "downbeat_f": f_measure(gt_t[gt_d], tr_t[tr_d]),
           "bar": bar_length(gt_d)}
    if len(gt_t) < 3 or len(tr_t) < 3:
        out["octave"] = "none"
        return out
    lo, hi = max(gt_t[0], tr_t[0]), min(gt_t[-1], tr_t[-1])  # compare where both have beats
    g = (gt_t >= lo - MATCH_TOL) & (gt_t <= hi + MATCH_TOL)
    e = (tr_t >= lo - MATCH_TOL) & (tr_t <= hi + MATCH_TOL)
    gt_t, gt_d, tr_t, tr_d = gt_t[g], gt_d[g], tr_t[e], tr_d[e]
    if len(gt_t) < 3 or len(tr_t) < 3:
        out["octave"] = "none"
        return out
    ratio = float(np.median(np.diff(tr_t)) / np.median(np.diff(gt_t)))
    out["ratio"], out["density"] = ratio, len(tr_t) / len(gt_t)
    pairs = match(gt_t, tr_t)
    out["found"] = len(pairs) / len(gt_t)  # share of the true beats it has a beat on (any octave)
    cls = bar_classes(gt_d)
    counts = {"hit": [0, 0], "half": [0, 0], "other": [0, 0], "extra": [0, 0]}
    for a, b in pairs:
        c = counts[("hit", "half", "other")[cls[a]]]
        c[0] += int(tr_d[b])
        c[1] += 1
    matched = np.zeros(len(tr_t), bool)
    matched[[b for _, b in pairs]] = True
    spurious = np.nonzero(~matched)[0]
    counts["extra"] = [int(tr_d[spurious].sum()), int(len(spurious))]
    out["down"] = counts
    if ratio < 0.7:
        out["octave"], out["factor"] = "double", int(np.clip(np.floor(1 / ratio + 0.5), FACTORS[0], FACTORS[-1]))
        return out
    if ratio > 1.4:
        out["octave"], out["factor"] = "half", int(np.clip(np.floor(ratio + 0.5), FACTORS[0], FACTORS[-1]))
        return out
    mids = (gt_t[:-1] + gt_t[1:]) / 2
    if len(pairs) < 0.5 * len(gt_t) and len(match(mids, tr_t)) >= 0.5 * len(mids):
        out["octave"] = "offbeat"
        return out
    out["octave"] = "normal"
    out["miss"] = 1 - len(pairs) / len(gt_t)
    out["extra"] = 1 - len(pairs) / len(tr_t)
    out["errors"] = [float(tr_t[b] - gt_t[a]) for a, b in pairs]
    j = np.searchsorted(gt_t, tr_t[spurious]) - 1
    ok = (j >= 0) & (j + 1 < len(gt_t))
    j, ts = j[ok], tr_t[spurious][ok]
    out["extra_phase"] = [float(x) for x in (ts - gt_t[j]) / (gt_t[j + 1] - gt_t[j])]
    return out


def downbeat_rates(comparisons: list[dict]) -> dict:
    """Pooled downbeat rates of these comparisons: how often a true downbeat, a half-bar beat,
    another beat and a spurious beat were marked as downbeats (None: no such beats)."""
    out = {}
    for key, name in (("hit", "down_hit"), ("half", "down_half"), ("other", "down_other"), ("extra", "down_extra")):
        m = sum(c["down"][key][0] for c in comparisons if "down" in c)
        n = sum(c["down"][key][1] for c in comparisons if "down" in c)
        out[name] = m / n if n else None
    return out


def fit_noise(comparisons: list[dict], note: str = "") -> BeatNoise:
    """BeatNoise from ``compare`` results, one per performance: the share of performances with each
    tempo octave error and its factors, the pooled downbeat rates at a wrong tempo, one octave
    profile per wrong-tempo performance (kind, factor, the fraction of the piece it covers from
    the beat density, its downbeat rates), one profile per
    right-tempo performance (its miss / extra rates, timing and downbeat rates; a downbeat rate it
    has no beats for -> the pooled rate), the scalars as the profiles' means, and the pooled phase
    histogram of the spurious beats."""
    n = len(comparisons)
    if n == 0:
        raise ValueError("no performances to fit the beat noise on")
    cnt = lambda val: sum(c.get("octave") == val for c in comparisons) / n  # noqa: E731
    normal = [c for c in comparisons if c.get("octave") == "normal"]
    pooled = {k: (0.0 if v is None else v) for k, v in downbeat_rates(normal).items()}
    wrong = downbeat_rates([c for c in comparisons if c.get("octave") in ("double", "half", "offbeat")])
    octave_down = None
    if any(v is not None for v in wrong.values()):
        octave_down = [round(float(pooled[k] if wrong[k] is None else wrong[k]), 5)
                       for k in ("down_hit", "down_half", "down_other", "down_extra")]
    errs = np.concatenate([np.asarray(c["errors"]) for c in normal]) if normal else np.zeros(0)
    jit = float(np.std(errs)) if len(errs) > 1 else 0.0
    profiles = []
    for c in normal:
        e = np.asarray(c["errors"])
        prof = {"bar": int(c.get("bar", 0)), "miss": c["miss"], "extra": c["extra"],
                "offset": float(e.mean()) if len(e) else 0.0, "jitter": float(e.std()) if len(e) > 1 else jit}
        for key, name in (("hit", "down_hit"), ("half", "down_half"), ("other", "down_other"),
                          ("extra", "down_extra")):
            m, k = c["down"][key]
            prof[name] = m / k if k else pooled[name]
        profiles.append({k: v if k == "bar" else round(float(v), 5) for k, v in prof.items()})
    mean = {k: float(np.mean([p[k] for p in profiles])) if profiles else 0.0 for k in PROFILE}
    phases = np.concatenate([np.asarray(c.get("extra_phase", []), np.float64) for c in normal]) if normal else []
    hist = np.histogram(phases, bins=PHASE_BINS, range=(0, 1))[0].astype(np.float64) if len(phases) else None
    def factors(octave):
        ks = [c.get("factor", 2) for c in comparisons if c.get("octave") == octave]
        return [round(ks.count(k) / len(ks), 5) for k in FACTORS] if ks else [1.0, 0.0, 0.0]
    octave_profiles = []
    for c in comparisons:
        if c.get("octave") not in ("double", "half") or "down" not in c:
            continue
        # beats per true beat: (1 - miss) + (k - 1) f too fast (the true beats it missed are its
        # misses); 1 - (1 - 1/k) f too slow (the true beats it drops are the error itself)
        k, dens = c["factor"], c["density"]
        if c["octave"] == "double":
            miss = 1 - c["found"]
            frac = (dens - (1 - miss)) / (k - 1)
        else:
            miss, frac = 0.0, (1 - dens) / (1 - 1 / k)
        q = {"octave": c["octave"], "bar": int(c.get("bar", 0)), "factor": int(k),
             "frac": round(float(np.clip(frac, 0.05, 1.0)), 5),
             "miss": round(float(np.clip(miss, 0.0, 0.9)), 5)}
        for key, name in (("hit", "down_hit"), ("half", "down_half"), ("other", "down_other"),
                          ("extra", "down_extra")):
            m_, n_ = c["down"][key]
            fallback = octave_down[("down_hit", "down_half", "down_other", "down_extra").index(name)] \
                if octave_down else pooled[name]
            q[name] = round(m_ / n_, 5) if n_ else fallback
        octave_profiles.append(q)
    return BeatNoise(
        p_double=cnt("double"), p_half=cnt("half"), double_k=factors("double"), half_k=factors("half"),
        p_offbeat=cnt("offbeat"), **mean, octave_down=octave_down, octave_profiles=octave_profiles or None,
        extra_phase=[round(float(x), 5) for x in hist / hist.sum()] if hist is not None and hist.sum() else None,
        profiles=profiles or None, fitted=True, note=note or f"fitted on {n} performances")


# ---------------------------------------------------------------------------------------------
# storage: one beat list per performance, per split and source
# ---------------------------------------------------------------------------------------------

def beats_path(split_dir, source: str) -> Path:
    if source not in SOURCES:
        raise ValueError(f"beat source must be one of {SOURCES}, not {source!r}")
    return Path(split_dir) / f"beats_{source}.npz"


def save_beats(path, beats: dict) -> None:
    """``beats``: performance -> (times, down), absolute performance seconds."""
    perfs = sorted(beats)
    times = [np.asarray(beats[p][0], np.float64) for p in perfs]
    downs = [np.asarray(beats[p][1], bool) for p in perfs]
    offsets = np.cumsum([0] + [len(t) for t in times])
    np.savez(path, perfs=np.array(json.dumps(perfs)), offsets=offsets,
             times=np.concatenate(times) if times else np.zeros(0),
             down=np.concatenate(downs) if downs else np.zeros(0, bool))


def load_beats(path) -> dict:
    """performance -> (times, down) from ``save_beats``; {} if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    z = np.load(path)
    perfs = json.loads(str(z["perfs"]))
    off, times, down = z["offsets"], z["times"], z["down"]
    return {p: (times[off[k]:off[k + 1]], down[off[k]:off[k + 1]]) for k, p in enumerate(perfs)}
