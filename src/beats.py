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
from dataclasses import asdict, dataclass, fields
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
    outside the aligned notes; 0 = interpolated), or None if fewer than MIN_PAIRS notes align.
    Beats more than ``max_extrap`` quarter notes outside the aligned notes are left out."""
    fit = fit_alignment(np.asarray(onsets, np.float64), np.asarray(pitches), tokens)
    if fit is None or len(fit[3]) < MIN_PAIRS:
        return None
    _, sq, _, pairs, pt = fit
    uq, ut, slope = _time_map(pairs, sq, pt)
    bars, beats, qs = [], [], []
    start = 0.0
    for j, b in enumerate(parse_tokens(tokens)):
        n, qpb = beat_layout(b.ts)
        for k in range(n):
            bars.append(j)
            beats.append(k)
            qs.append(start + k * qpb)
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
            "t": t[keep], "extrap": extrap[keep]}


def merge_performance(windows) -> tuple[np.ndarray, np.ndarray]:
    """One beat list for a performance from its windows' grids: ``windows`` = [(bar_start,
    perf_t0, grid or None)]. Each beat (global bar, beat in bar) comes from the window that timed it
    with the least extrapolation (windows overlap at their edges in training splits); times must
    increase, so a beat that does not is dropped. Returns (times, down)."""
    best = {}
    for bar_start, t0, g in windows:
        if g is None:
            continue
        for b, k, t, e in zip(g["bar"], g["beat"], g["t"], g["extrap"]):
            key = (int(bar_start) + int(b), int(k))
            if key not in best or e < best[key][1]:
                best[key] = (float(t0) + float(t), float(e))
    keys = sorted(best)
    times, down, last = [], [], -np.inf
    for key in keys:
        t = best[key][0]
        if t <= last:
            continue
        times.append(t)
        down.append(key[1] == 0)
        last = t
    return np.asarray(times, np.float64), np.asarray(down, bool)


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

@dataclass
class BeatNoise:
    """The errors a beat tracker makes, applied to a ground-truth beat list (training, and the
    simulated-tracker test condition). Probabilities are per beat list (a window, or a piece when
    fitted); fitted by scripts/fit_beat_noise.py from Beat This! against the ground truth.
      p_double / p_half   tempo octave errors: every half beat / every other beat
      p_offbeat           beats on the off-beats (half a period late)
      miss, extra         share of beats missed / spurious beats added (as a share of the beats)
      jitter              timing error of the beats it finds (seconds, standard deviation)
      p_no_downbeats      no downbeats at all
      p_downbeat_shift    the right bar length, downbeats on the wrong beat of the bar
      p_meter             the wrong bar length (downbeats every 2 / 3 / 4 beats instead)"""
    p_double: float = 0.05
    p_half: float = 0.10
    p_offbeat: float = 0.05
    miss: float = 0.08
    extra: float = 0.05
    jitter: float = 0.02
    p_no_downbeats: float = 0.05
    p_downbeat_shift: float = 0.15
    p_meter: float = 0.10
    fitted: bool = False
    note: str = "defaults (typical Beat This! errors on classical piano), not fitted"

    @classmethod
    def from_json(cls, path) -> "BeatNoise":
        d = json.loads(Path(path).read_text())
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_json(self, path, extra: dict | None = None):
        d = asdict(self)
        d.update(extra or {})
        Path(path).write_text(json.dumps(d, indent=1))

    def __call__(self, times: np.ndarray, down: np.ndarray, rng: np.random.Generator):
        times = np.asarray(times, np.float64).copy()
        down = np.asarray(down, bool).copy()
        if len(times) < 3:
            return times, down
        u = rng.random()
        if u < self.p_double:  # a beat between every two
            mid = (times[:-1] + times[1:]) / 2
            times = np.insert(times, np.arange(1, len(times)), mid)
            down = np.insert(down, np.arange(1, len(down)), False)
        elif u < self.p_double + self.p_half:  # every other beat; a dropped downbeat moves to the next kept beat
            r = int(rng.integers(2))
            keep = np.arange(r, len(times), 2)
            moved = np.zeros(len(keep), bool)
            if r == 1:
                moved |= down[keep - 1]
            nxt = keep + 1
            moved |= np.where(nxt < len(down), down[np.minimum(nxt, len(down) - 1)], False) & (r == 0)
            times, down = times[keep], down[keep] | moved
        if len(times) >= 3 and rng.random() < self.p_offbeat:
            times = (times[:-1] + times[1:]) / 2
            down = down[:-1]
        d = np.nonzero(down)[0]
        bar_len = int(np.median(np.diff(d))) if len(d) >= 2 else 0
        u = rng.random()
        if u < self.p_no_downbeats:
            down[:] = False
        elif u < self.p_no_downbeats + self.p_downbeat_shift and bar_len >= 2:
            s = int(rng.integers(1, bar_len))
            down = np.roll(down, s)
            down[:s] = False
        elif u < self.p_no_downbeats + self.p_downbeat_shift + self.p_meter:
            new = int(rng.choice([m for m in (2, 3, 4) if m != bar_len]))
            off = int(rng.integers(new))
            down = (np.arange(len(times)) - off) % new == 0
        if self.miss > 0 and len(times) > 2:
            keep = rng.random(len(times)) >= self.miss
            if keep.sum() >= 2:
                times, down = times[keep], down[keep]
        if self.extra > 0 and len(times) >= 2:
            n = int(rng.binomial(len(times), min(self.extra, 1.0)))
            if n:
                new_t = rng.uniform(times[0], times[-1], n)
                times = np.concatenate([times, new_t])
                down = np.concatenate([down, np.zeros(n, bool)])
        if self.jitter > 0:
            times = times + rng.normal(0, self.jitter, len(times))
        order = np.argsort(times, kind="stable")
        times, down = times[order], down[order]
        keep = np.concatenate([[True], np.diff(times) > 0.05])  # no two beats within 50 ms
        return times[keep], down[keep]


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


def _bar_length(times, down) -> int:
    d = np.nonzero(down)[0]
    return int(np.median(np.diff(d))) if len(d) >= 2 else 0


def compare(gt_t, gt_d, tr_t, tr_d) -> dict:
    """How a tracked beat list differs from the ground truth, in BeatNoise's terms."""
    gt_t, tr_t = np.asarray(gt_t, np.float64), np.asarray(tr_t, np.float64)
    gt_d, tr_d = np.asarray(gt_d, bool), np.asarray(tr_d, bool)
    out = {"beat_f": f_measure(gt_t, tr_t), "downbeat_f": f_measure(gt_t[gt_d], tr_t[tr_d])}
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
    out["ratio"] = ratio
    if ratio < 0.7:
        out["octave"] = "double"
        return out
    if ratio > 1.4:
        out["octave"] = "half"
        return out
    pairs = match(gt_t, tr_t)
    mids = (gt_t[:-1] + gt_t[1:]) / 2
    if len(pairs) < 0.5 * len(gt_t) and len(match(mids, tr_t)) >= 0.5 * len(mids):
        out["octave"] = "offbeat"
        return out
    out["octave"] = "normal"
    out["miss"] = 1 - len(pairs) / len(gt_t)
    out["extra"] = 1 - len(pairs) / len(tr_t)
    out["errors"] = [float(tr_t[b] - gt_t[a]) for a, b in pairs]
    gl, tl = _bar_length(gt_t, gt_d), _bar_length(tr_t, tr_d)
    if not tr_d.any():
        out["downbeats"] = "none"
    elif gl and tl and tl != gl:
        out["downbeats"] = "meter"
    else:
        hit = len(match(gt_t[gt_d], tr_t[tr_d]))
        out["downbeats"] = "correct" if hit >= 0.5 * max(tr_d.sum(), 1) else "shift"
    return out


def fit_noise(comparisons: list[dict], note: str = "") -> BeatNoise:
    """BeatNoise from ``compare`` results, one per performance: the share of performances with
    each kind of error, and the beat-level rates and jitter pooled over the normal ones."""
    n = len(comparisons)
    if n == 0:
        raise ValueError("no performances to fit the beat noise on")
    cnt = lambda key, val: sum(c.get(key) == val for c in comparisons) / n  # noqa: E731
    normal = [c for c in comparisons if c.get("octave") == "normal"]
    errs = np.concatenate([np.asarray(c["errors"]) for c in normal]) if normal else np.zeros(0)
    nd = max(len(normal), 1)
    return BeatNoise(
        p_double=cnt("octave", "double"), p_half=cnt("octave", "half"), p_offbeat=cnt("octave", "offbeat"),
        miss=float(np.mean([c["miss"] for c in normal])) if normal else 0.0,
        extra=float(np.mean([c["extra"] for c in normal])) if normal else 0.0,
        jitter=float(np.std(errs)) if len(errs) > 1 else 0.0,
        p_no_downbeats=sum(c.get("downbeats") == "none" for c in normal) / nd,
        p_downbeat_shift=sum(c.get("downbeats") == "shift" for c in normal) / nd,
        p_meter=sum(c.get("downbeats") == "meter" for c in normal) / nd,
        fitted=True, note=note or f"fitted on {n} performances")


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
