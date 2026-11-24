"""Performance MIDI -> audio, for beat tracking a MIDI input (Beat This! hears audio only).

    wav = render(notes)                     # notes [N, 4]: pitch, onset s, offset s, velocity
    write_wav("perf.wav", wav)              # 16-bit mono, SR Hz (stdlib wave: no extra package)

A small additive piano-like synthesizer, deterministic and dependency-free: each key is a fixed
template (6 slightly inharmonic partials, a 4 ms attack and a pitch-dependent exponential decay:
low strings ring longer), scaled by velocity and cut with a short release at the note's offset.
Beat This! only needs clear onsets and a plausible spectrum; the templates are rendered once per
pitch, so a 10-minute piece takes about a second.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 22050          # Beat This! resamples to 22.05 kHz anyway
MAX_RING = 3.0      # seconds a note can sound (the decay makes later samples inaudible)
RELEASE = 0.08      # seconds of fade after the offset
ATTACK = 0.004
N_PARTIALS = 6
INHARMONIC = 4e-4   # stiffness: partial k sits at k * f0 * sqrt(1 + B k^2)


def _template(pitch: int, sr: int = SR) -> np.ndarray:
    """One key struck at full velocity, MAX_RING seconds long, peak about 1."""
    f0 = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
    n = int(MAX_RING * sr)
    t = np.arange(n) / sr
    tau = float(np.interp(pitch, [21, 60, 108], [2.2, 0.9, 0.18]))  # decay time constant, seconds
    out = np.zeros(n)
    for k in range(1, N_PARTIALS + 1):
        f = k * f0 * np.sqrt(1.0 + INHARMONIC * k * k)
        if f >= sr / 2:
            break
        out += np.sin(2 * np.pi * f * t) * np.exp(-t * k ** 0.5 / tau) / k ** 1.3
    out *= np.minimum(1.0, t / ATTACK)
    return out / max(np.abs(out).max(), 1e-9)


def render(notes: np.ndarray, sr: int = SR, pad: float = 1.0) -> np.ndarray:
    """Mono float32 audio of the notes (time 0 = 0 s; onsets before 0 are clipped to 0), ``pad``
    seconds of silence after the last note, normalised to peak 0.9."""
    notes = np.asarray(notes, np.float64).reshape(-1, 4)
    if len(notes) == 0:
        return np.zeros(int(pad * sr), np.float32)
    on = np.maximum(notes[:, 1], 0.0)
    off = np.maximum(notes[:, 2], on + 0.02)
    total = int((min(off.max() + RELEASE, on.max() + MAX_RING) + pad) * sr) + 1
    out = np.zeros(total)
    cache = {}
    rel = np.cos(np.linspace(0, np.pi / 2, int(RELEASE * sr))) ** 2
    for p, a, b, v in zip(notes[:, 0].astype(int), on, off, notes[:, 3]):
        p = int(np.clip(p, 21, 108))
        if p not in cache:
            cache[p] = _template(p, sr)
        tpl = cache[p]
        i0 = int(round(a * sr))
        n = min(len(tpl), int(round((b - a) * sr)) + len(rel), total - i0)
        if n <= 0:
            continue
        seg = tpl[:n] * (max(float(v), 1.0) / 127.0) ** 1.6
        hold = max(0, n - len(rel))
        seg[hold:] *= rel[: n - hold]
        out[i0:i0 + n] += seg
    peak = np.abs(out).max()
    return (out * (0.9 / peak) if peak > 0 else out).astype(np.float32)


def write_wav(path, audio: np.ndarray, sr: int = SR) -> Path:
    """16-bit PCM mono WAV."""
    path = Path(path)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path
