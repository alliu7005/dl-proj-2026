"""Synthetic transcription-error injection, applied on the fly to clean performance segments.

Defaults are placeholders. Run ``scripts/fit_noise.py`` on Kong-transcribed MAESTRO *test* files
vs. ground-truth MIDI to fit them, then pass the resulting JSON to ``NoiseModel.from_json``.
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


class NoiseModel:
    def __init__(self, cfg: NoiseConfig | None = None):
        self.cfg = cfg or NoiseConfig()
        iv = self.cfg.halluc_intervals
        self._iv = np.array([int(k) for k in iv], dtype=np.int64)
        p = np.array(list(iv.values()), dtype=np.float64)
        self._ivp = p / p.sum()

    @classmethod
    def from_json(cls, path: str) -> "NoiseModel":
        with open(path) as f:
            d = json.load(f)
        known = NoiseConfig.__dataclass_fields__
        return cls(NoiseConfig(**{k: v for k, v in d.items() if k in known}))

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

    def __call__(self, notes: np.ndarray, pedal: np.ndarray, rng: np.random.Generator):
        c = self.cfg
        if len(notes) == 0 or rng.random() < c.clean_prob:
            return notes.copy(), pedal.copy()
        n = notes.copy()

        # 1) sustain pedal lengthens notes released while the pedal is down
        if len(pedal) and c.pedal_extend_prob > 0:
            off = n[:, ON] + n[:, DUR]
            for on_t, off_t in pedal:
                hit = (off > on_t) & (off < off_t) & (rng.random(len(n)) < c.pedal_extend_prob)
                n[hit, DUR] = off_t - n[hit, ON]

        # 2) missing notes, more likely in dense chords
        order = np.argsort(n[:, ON], kind="stable")
        on_sorted = n[order, ON]
        group = np.concatenate([[0], np.cumsum(np.diff(on_sorted) > c.chord_window)])
        size = np.bincount(group)[group]
        p_drop = np.empty(len(n))
        p_drop[order] = c.drop_base + c.drop_per_chord_note * (size - 1)
        keep = rng.random(len(n)) >= p_drop
        if not keep.any():
            keep[rng.integers(len(n))] = True

        # 3) hallucinated notes (mostly overtones of real notes)
        src = n[rng.random(len(n)) < c.halluc_rate]
        if len(src):
            h = src.copy()
            h[:, P] = src[:, P] + rng.choice(self._iv, size=len(h), p=self._ivp)
            h[:, ON] += rng.normal(0, c.halluc_onset_std, len(h))
            h[:, DUR] *= rng.uniform(*c.halluc_dur_ratio, len(h))
            h[:, VEL] *= rng.uniform(*c.halluc_vel_ratio, len(h))
            h = h[(h[:, P] >= 21) & (h[:, P] <= 108)]
        else:
            h = n[:0]
        n = n[keep]

        # 4) timing / duration / velocity jitter on real notes
        n[:, ON] += rng.normal(0, c.onset_std, len(n))
        n[:, DUR] *= np.exp(rng.normal(0, c.offset_log_std, len(n)))
        n[:, VEL] += rng.normal(0, c.velocity_std, len(n))

        out = np.concatenate([n, h])
        out[:, ON] = np.maximum(out[:, ON], 0)
        out[:, DUR] = np.maximum(out[:, DUR], 0.01)
        out[:, VEL] = np.clip(np.round(out[:, VEL]), 1, 127)
        out = out[np.lexsort((out[:, P], out[:, ON]))]

        ped = pedal.copy()
        if len(ped):
            ped = ped[rng.random(len(ped)) >= c.pedal_drop_prob]
            ped = ped + rng.normal(0, c.pedal_time_std, ped.shape).astype(np.float32)
            ped[:, 0] = np.maximum(ped[:, 0], 0)
            ped = ped[ped[:, 1] > ped[:, 0]]
        return out.astype(np.float32), ped.astype(np.float32)
