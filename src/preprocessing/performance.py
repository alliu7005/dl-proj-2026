"""Load performance MIDI (notes + sustain pedal) and map performance time to score time."""
from __future__ import annotations

import numpy as np
import pretty_midi

# Column layout of a note array: pitch, onset (s), offset (s), velocity
PITCH, ONSET, OFFSET, VEL = 0, 1, 2, 3
SUSTAIN_CC = 64


def load_performance(path, pedal_threshold: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return (notes [N,4] float32 sorted by onset, pedal [K,2] float32 of (on, off) times).
    ``path`` may also be the raw bytes of a MIDI file."""
    if isinstance(path, (bytes, bytearray)):
        import io
        pm = pretty_midi.PrettyMIDI(io.BytesIO(path))
    else:
        pm = pretty_midi.PrettyMIDI(str(path))
    notes, ccs = [], []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes += [(n.pitch, n.start, n.end, n.velocity) for n in inst.notes]
        ccs += [(c.time, c.value) for c in inst.control_changes if c.number == SUSTAIN_CC]
    arr = np.asarray(notes, dtype=np.float32).reshape(-1, 4)
    arr = arr[np.lexsort((arr[:, PITCH], arr[:, ONSET]))]
    end_time = float(arr[:, OFFSET].max()) if len(arr) else 0.0
    return arr, pedal_intervals(sorted(ccs), pedal_threshold, end_time)


def estimate_offset(ref: np.ndarray, other: np.ndarray, tol: float = 0.05, n_ref: int = 200,
                    n_cand: int = 20) -> tuple[float, float]:
    """Constant time offset between two versions of one performance (e.g. a transcription vs. a
    trimmed/aligned copy). ``ref`` / ``other``: arrays whose columns 0, 1 are pitch, onset.
    Returns (d, fraction of the first ``n_ref`` ref notes matched) with other_onset - d ~= ref_onset."""
    a = ref[:n_ref]
    if not len(a) or not len(other):
        return 0.0, 0.0
    by_pitch = {p: np.sort(other[other[:, 0] == p, 1]) for p in np.unique(a[:, 0])}
    cands = np.concatenate([by_pitch[a[k, 0]] - a[k, 1] for k in range(min(n_cand, len(a)))])

    def score(d):
        hits = 0
        for p, t in zip(a[:, 0], a[:, 1]):
            arr = by_pitch[p]
            j = np.searchsorted(arr, t + d)
            hits += any(abs(arr[jj] - (t + d)) <= tol for jj in (j - 1, j) if 0 <= jj < len(arr))
        return hits / len(a)

    best = max(cands, key=score, default=0.0)
    m = [o - t for p, t in zip(a[:, 0], a[:, 1]) for o in by_pitch[p] if abs(o - t - best) <= tol]
    d = float(np.median(m)) if m else float(best)
    return d, score(d)


def pedal_intervals(ccs, threshold: int, end_time: float) -> np.ndarray:
    """Turn a sorted list of (time, cc64 value) into (on, off) intervals."""
    out, on = [], None
    for t, v in ccs:
        if v >= threshold and on is None:
            on = t
        elif v < threshold and on is not None:
            if t > on:
                out.append((on, t))
            on = None
    if on is not None:
        out.append((on, max(end_time, on)))
    arr = np.asarray(out, dtype=np.float32).reshape(-1, 2)
    return arr[arr[:, 1] > arr[:, 0]]  # drop presses that collapse to zero length in float32


class TimeMap:
    """Piecewise-linear map between two aligned beat grids, linearly extrapolated at both ends."""

    def __init__(self, src_beats, dst_beats):
        src = np.asarray(src_beats, dtype=np.float64)
        dst = np.asarray(dst_beats, dtype=np.float64)
        if len(src) != len(dst) or len(src) < 2:
            raise ValueError(f"beat grids must align and have >=2 beats ({len(src)} vs {len(dst)})")
        keep = np.concatenate([[True], np.diff(src) > 0])  # drop duplicate/backwards beats
        self.src, self.dst = src[keep], dst[keep]
        self._lo = (self.dst[1] - self.dst[0]) / (self.src[1] - self.src[0])
        self._hi = (self.dst[-1] - self.dst[-2]) / (self.src[-1] - self.src[-2])

    def __call__(self, t):
        t = np.asarray(t, dtype=np.float64)
        y = np.interp(t, self.src, self.dst)
        y = np.where(t < self.src[0], self.dst[0] + (t - self.src[0]) * self._lo, y)
        y = np.where(t > self.src[-1], self.dst[-1] + (t - self.src[-1]) * self._hi, y)
        return y

    def inverse(self) -> "TimeMap":
        return TimeMap(self.dst, self.src)
