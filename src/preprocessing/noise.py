"""Synthetic transcription-error injection, applied on the fly to clean performance segments.

Defaults are placeholders. Run ``scripts/fit_noise.py`` on transcribed MAESTRO test/validation
files vs. ground-truth MIDI to fit them, then pass the resulting JSON to ``NoiseModel.from_json``.
The committed ``configs/noise_stats.json`` is fitted on **Kong et al.** transcriptions; to match
Transkun V2 (PERiScoPe) instead, refit on ``data/transcribed_transkun/pairs.csv``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

# Note columns (segment format): pitch, onset, duration, velocity
P, ON, DUR, VEL = 0, 1, 2, 3


@dataclass
class NoiseConfig:
    # missing notes: p = drop_base + drop_per_chord_note * (chord_size - 1), chord = onsets within chord_window
    drop_base: float = 0.02
    drop_per_chord_note: float = 0.01
    chord_window: float = 0.03
    # hallucinated notes, per real note; interval (semitones above source) distribution
    halluc_rate: float = 0.04
    halluc_intervals: dict = field(default_factory=lambda: {
        "12": 0.45, "19": 0.2, "24": 0.1, "7": 0.05, "-12": 0.05, "1": 0.05, "-1": 0.05, "28": 0.05})
    halluc_vel_ratio: tuple = (0.3, 0.7)        # uniform range, relative to source velocity
    halluc_dur_ratio: tuple = (0.2, 1.0)
    halluc_onset_std: float = 0.01
    # timing / duration errors on real notes
    onset_std: float = 0.008                    # seconds
    offset_log_std: float = 0.15                # multiplicative log-normal noise on duration
    pedal_extend_prob: float = 0.5              # note released under sustain -> lasts until pedal up
    velocity_std: float = 4.0
    # pedal transcription errors
    pedal_time_std: float = 0.03
    pedal_drop_prob: float = 0.05
    # probability that a segment is left clean (keeps the model good on clean MIDI too)
    clean_prob: float = 0.1
    # --- empirical mode (written by scripts/fit_noise.py); overrides the parametric fields above ---
    # miss rate per "chordsize|velbin" (chord size capped at 8, see vel_bin)
    drop_table: dict | None = None
    # hallucinations spawned per real note, per velocity bin of the source note
    halluc_rate_by_vel: dict | None = None
    # real hallucinations relative to their source note: [interval, delay_s, vel_ratio, dur_s, src_vel_bin]
    halluc_bank: list | None = None


VEL_EDGES = (40, 55, 70)
SOURCE_TAU = 0.3      # s; recency weighting when attributing a hallucination to a source note
SOURCE_LOOKBACK = 2.0


def vel_bin(v):
    return np.searchsorted(VEL_EDGES, v, side="right")


def chord_sizes(onsets: np.ndarray, window: float) -> np.ndarray:
    order = np.argsort(onsets, kind="stable")
    group = np.concatenate([[0], np.cumsum(np.diff(onsets[order]) > window)])
    size = np.empty(len(onsets), dtype=np.int64)
    size[order] = np.bincount(group)[group]
    return size


def attribute_source(t: float, on: np.ndarray, end: np.ndarray, vel: np.ndarray) -> int:
    """Most likely source of a hallucination at time t: a recent, loud note still sounding
    (``end`` = pedal-extended offset). Returns -1 if nothing is sounding."""
    cand = np.nonzero((on <= t + 0.03) & (on >= t - SOURCE_LOOKBACK) & (end >= t - 0.03))[0]
    if len(cand) == 0:
        return -1
    w = vel[cand] * np.exp(-np.maximum(t - on[cand], 0) / SOURCE_TAU)
    return int(cand[np.argmax(w)])


class NoiseModel:
    def __init__(self, cfg: NoiseConfig | None = None):
        self.cfg = cfg or NoiseConfig()
        iv = self.cfg.halluc_intervals
        self._iv = np.array([int(k) for k in iv], dtype=np.int64)
        p = np.array(list(iv.values()), dtype=np.float64)
        self._ivp = p / p.sum()
        self._bank = None
        if self.cfg.halluc_bank:
            b = np.asarray(self.cfg.halluc_bank, dtype=np.float64)
            self._bank = {vb: b[b[:, 4] == vb] for vb in range(len(VEL_EDGES) + 1)}
            for vb, rows in self._bank.items():  # fall back to the whole bank for empty bins
                if len(rows) == 0:
                    self._bank[vb] = b

    @property
    def empirical(self) -> bool:
        return self._bank is not None

    @classmethod
    def from_json(cls, path: str) -> "NoiseModel":
        with open(path) as f:
            d = json.load(f)
        known = NoiseConfig.__dataclass_fields__
        return cls(NoiseConfig(**{k: v for k, v in d.items() if k in known}))

    def to_json(self, path: str):
        d = asdict(self.cfg)
        bank = d.pop("halluc_bank")
        text = json.dumps(d, indent=2)
        if bank:  # one bank row per line keeps the file compact and diffable
            rows = ",\n    ".join(json.dumps(r) for r in bank)
            text = text[:-2] + ',\n  "halluc_bank": [\n    ' + rows + "\n  ]\n}"
        with open(path, "w") as f:
            f.write(text)

    def _parametric_hallucinations(self, n, rng):
        """Returns (hallucinations, carried flags): carried if the source note is carried-over."""
        c = self.cfg
        src = n[rng.random(len(n)) < c.halluc_rate]
        if not len(src):
            return n[:0], np.zeros(0, dtype=bool)
        h = src.copy()
        h[:, P] = src[:, P] + rng.choice(self._iv, size=len(h), p=self._ivp)
        h[:, ON] += rng.normal(0, c.halluc_onset_std, len(h))
        h[:, DUR] *= rng.uniform(*c.halluc_dur_ratio, len(h))
        h[:, VEL] *= rng.uniform(*c.halluc_vel_ratio, len(h))
        keep = (h[:, P] >= 21) & (h[:, P] <= 108)
        return h[keep], (src[:, ON] < 0)[keep]

    def _bank_hallucinations(self, n, rng):
        """Each real note spawns Poisson(rate[vel bin]) hallucinations; each copies a real one's
        (interval, delay, velocity ratio, duration) relative to its source note."""
        c = self.cfg
        vb = vel_bin(n[:, VEL])
        rate = np.array([c.halluc_rate_by_vel.get(str(v), c.halluc_rate) for v in vb])
        counts = rng.poisson(rate)
        out = []
        for i in np.nonzero(counts)[0]:
            bank = self._bank[int(vb[i])]
            for row in bank[rng.integers(len(bank), size=counts[i])]:
                k, delay, vr, dur = row[:4]
                out.append((n[i, P] + k, n[i, ON] + delay, dur, n[i, VEL] * vr))
        if not out:
            return n[:0]
        h = np.asarray(out, dtype=n.dtype)
        return h[(h[:, P] >= 21) & (h[:, P] <= 108)]

    def __call__(self, notes: np.ndarray, pedal: np.ndarray, rng: np.random.Generator,
                 return_labels: bool = False):
        """Returns (noisy notes, noisy pedal), plus per-note labels (1 = hallucinated) if
        ``return_labels``."""
        c = self.cfg
        if len(notes) == 0 or rng.random() < c.clean_prob:
            out = (notes.copy(), pedal.copy())
            return out + (np.zeros(len(notes), dtype=np.int8),) if return_labels else out
        n = notes.copy()

        # 1) sustain pedal lengthens notes released while the pedal is down
        if len(pedal) and c.pedal_extend_prob > 0:
            off = n[:, ON] + n[:, DUR]
            for on_t, off_t in pedal:
                hit = (off > on_t) & (off < off_t) & (rng.random(len(n)) < c.pedal_extend_prob)
                n[hit, DUR] = off_t - n[hit, ON]

        # 2) missing notes, more likely in dense chords (and for quiet notes, in empirical mode)
        size = chord_sizes(n[:, ON], c.chord_window)
        if c.drop_table:
            vb = vel_bin(n[:, VEL])
            p_drop = np.array([c.drop_table.get(f"{min(sz, 8)}|{v}", c.drop_base)
                               for sz, v in zip(size, vb)])
        else:
            p_drop = c.drop_base + c.drop_per_chord_note * (size - 1)
        keep = rng.random(len(n)) >= p_drop
        if not keep.any():
            keep[rng.integers(len(n))] = True

        # 3) hallucinated notes. A hallucination is "carried" (struck before the segment, onset < 0)
        #    if its source is carried (parametric) or its replayed onset lands before t=0 (bank)
        if self.empirical:
            h = self._bank_hallucinations(n, rng)
            h_carried = h[:, ON] < 0
        else:
            h, h_carried = self._parametric_hallucinations(n, rng)
        n = n[keep]
        n_carried = n[:, ON] < 0

        # 4) timing / duration / velocity jitter on real notes
        n[:, ON] += rng.normal(0, c.onset_std, len(n))
        n[:, DUR] *= np.exp(rng.normal(0, c.offset_log_std, len(n)))
        n[:, VEL] += rng.normal(0, c.velocity_std, len(n))

        # negative onset = struck before the segment and still held (see segment.carry_notes):
        # jitter must not move notes across t=0 in either direction
        out = np.concatenate([n, h])
        labels = np.concatenate([np.zeros(len(n), np.int8), np.ones(len(h), np.int8)])
        carried = np.concatenate([n_carried, h_carried])
        out[:, ON] = np.where(carried, np.minimum(out[:, ON], -1e-3), np.maximum(out[:, ON], 0))
        out[:, DUR] = np.maximum(out[:, DUR], 0.01)
        out[:, VEL] = np.clip(np.round(out[:, VEL]), 1, 127)
        order = np.lexsort((out[:, P], out[:, ON]))
        out, labels = out[order], labels[order]

        ped = pedal.copy()
        if len(ped):
            ped = ped[rng.random(len(ped)) >= c.pedal_drop_prob]
            held = ped[:, 0] < 0  # pressed before the segment: keep that visible
            ped = ped + rng.normal(0, c.pedal_time_std, ped.shape).astype(np.float32)
            ped[:, 0] = np.where(held, np.minimum(ped[:, 0], -1e-3), np.maximum(ped[:, 0], 0))
            ped = ped[ped[:, 1] > ped[:, 0]]
        if return_labels:
            return out.astype(np.float32), ped.astype(np.float32), labels
        return out.astype(np.float32), ped.astype(np.float32)
