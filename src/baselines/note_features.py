"""Per-note context features for the note-level error detector (XGBoost baseline).

Input: one segment's notes [N,4] (pitch, onset, duration, velocity) + pedal [K,2].
Output: float32 [N, F] with column names in ``FEATURE_NAMES``.

Encodes the cues the proposal hypothesizes for transcription errors: hallucinated notes sit at
overtone-like intervals from a louder simultaneous note, missing/extra notes cluster in dense
chords, sustain pedal smears durations, and wrong notes clash with the local key.
"""
from __future__ import annotations

import numpy as np

CO_ONSET = 0.05        # s, "same chord"
WINDOW = 1.0           # s, local context
INTERVALS = (12, 19, 24, 7, 5, 4, 3, 2, 1, -1, -2, -3, -4, -5, -7, -12)

FEATURE_NAMES = (
    ["pitch", "velocity", "log_dur", "pedal_on", "pedal_off", "dt_prev", "dt_next",
     "n_co", "vel_rel_chord_max", "vel_minus_chord_mean", "pitch_above_chord_min",
     "pitch_below_chord_max", "is_lowest", "is_highest",
     "n_sounding", "sounding_octave_below", "density", "pc_frac", "pc_clash_frac",
     "pitch_z", "vel_z", "dt_same_pitch", "same_pitch_overlap", "dur_rel_median"]
    + [f"src_{k}" for k in INTERVALS] + [f"louder_src_{k}" for k in INTERVALS]
)


def _pedal_at(t: np.ndarray, pedal: np.ndarray) -> np.ndarray:
    if len(pedal) == 0:
        return np.zeros_like(t)
    return ((t[:, None] >= pedal[None, :, 0]) & (t[:, None] < pedal[None, :, 1])).any(1).astype(np.float32)


def note_context_features(notes: np.ndarray, pedal: np.ndarray) -> np.ndarray:
    n = len(notes)
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32)
    p, on, dur, vel = (notes[:, i].astype(np.float64) for i in range(4))
    off = on + dur
    eye = np.eye(n, dtype=bool)
    dt = on[None, :] - on[:, None]                 # dt[i, j] = on_j - on_i
    dp = p[:, None] - p[None, :]                   # dp[i, j] = p_i - p_j
    co = (np.abs(dt) <= CO_ONSET) & ~eye
    win = (np.abs(dt) <= WINDOW) & ~eye
    sounding = (on[None, :] < on[:, None] - CO_ONSET) & (off[None, :] > on[:, None])

    n_co = co.sum(1)
    chord = co | eye
    vmax = np.where(chord, vel[None, :], -np.inf).max(1)
    vmean_co = np.where(co, vel[None, :], 0).sum(1) / np.maximum(n_co, 1)
    pmin = np.where(chord, p[None, :], np.inf).min(1)
    pmax = np.where(chord, p[None, :], -np.inf).max(1)

    order = np.argsort(on, kind="stable")
    rank = np.empty(n, int)
    rank[order] = np.arange(n)
    on_sorted = on[order]
    dt_prev = np.where(rank > 0, on - on_sorted[np.maximum(rank - 1, 0)], 5.0)
    dt_next = np.where(rank < n - 1, on_sorted[np.minimum(rank + 1, n - 1)] - on, 5.0)

    n_win = win.sum(1)
    pc = (p % 12).astype(int)
    pc_eq = (pc[:, None] == pc[None, :])
    pc_clash = ((pc[:, None] - pc[None, :]) % 12 == 1) | ((pc[:, None] - pc[None, :]) % 12 == 11)
    wsum = np.maximum(n_win, 1)
    pc_frac = (win & pc_eq).sum(1) / wsum
    pc_clash_frac = (win & pc_clash).sum(1) / wsum
    wp = np.where(win, p[None, :], np.nan)
    wv = np.where(win, vel[None, :], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), _quiet():
        pitch_z = np.nan_to_num((p - np.nanmean(wp, 1)) / (np.nanstd(wp, 1) + 1.0))
        vel_z = np.nan_to_num((vel - np.nanmean(wv, 1)) / (np.nanstd(wv, 1) + 1.0))
        dur_rel = np.nan_to_num(np.log(np.maximum(dur, 1e-3) /
                                       np.maximum(np.nanmedian(np.where(win, dur[None, :], np.nan), 1), 1e-3)))

    same = (dp == 0) & ~eye & (dt < 0)              # earlier notes with the same pitch
    last_same = np.where(same, on[None, :], -np.inf).max(1)
    dt_same = np.minimum(on - last_same, 5.0)
    same_overlap = (same & (off[None, :] > on[:, None])).any(1)

    feats = [p, vel, np.log(np.maximum(dur, 1e-3)), _pedal_at(on, pedal), _pedal_at(off, pedal),
             np.minimum(dt_prev, 5.0), np.minimum(dt_next, 5.0),
             n_co, vel / np.maximum(vmax, 1), vel - vmean_co, p - pmin, pmax - p,
             (p == pmin).astype(float), (p == pmax).astype(float),
             sounding.sum(1), (sounding & (dp == 12)).any(1).astype(float),
             n_win / (2 * WINDOW), pc_frac, pc_clash_frac, pitch_z, vel_z, dt_same,
             same_overlap.astype(float), dur_rel]
    near = co | sounding
    for k in INTERVALS:
        feats.append((near & (dp == k)).any(1).astype(float))
    for k in INTERVALS:
        feats.append((near & (dp == k) & (vel[None, :] >= vel[:, None])).any(1).astype(float))
    return np.stack(feats, axis=1).astype(np.float32)


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)
