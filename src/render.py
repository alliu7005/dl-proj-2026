"""Performances rendered to audio with a sampled piano: FluidSynth playing a SoundFont.

The meter model trains on a beat tracker's beats. Most of its performances have no recording (ASAP's
MIDI, PERiScoPe's and most of Gangqinpu's transcriptions), so their tracker errors were simulated
(src/beats.py BeatNoise). Rendering them and running the real tracker gives the errors the music
itself causes. A sampled piano is close enough to a recording for that; the additive tone of
src/synth.py is not. Beat This! (held-out fold models) on 10 ASAP performances, MAESTRO's
recordings against renders of the same performances' MIDI with MuseScore's "MS Basic.sf3":

    |                 | beat F | downbeat F | wrong tempo octave | missed / extra beats |
    | recording       | 0.793  | 0.606      | 2 of 10            | 17% / 11%            |
    | FluidSynth      | 0.784  | 0.618      | 2 of 10            | 17% / 12%            |
    | src/synth.py    | 0.652  | 0.470      | 4 of 10            | 27% / 29%            |

Where things are: the FluidSynth program from ``FLUIDSYNTH`` (else on the PATH), the SoundFonts
from ``SOUNDFONTS`` (paths separated by os.pathsep), or given. With several SoundFonts each
performance gets one, chosen from its key (the same one every run), so the data holds several pianos.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import zlib
from pathlib import Path

import numpy as np

SR = 22050       # Beat This! resamples to 22.05 kHz mono anyway
GAIN = 0.8       # FluidSynth's default (0.2) renders a piano at about -19 dBFS peak


def find_fluidsynth(path: str | None = None) -> str:
    """The FluidSynth program: ``path``, else $FLUIDSYNTH, else on the PATH."""
    cand = path or os.environ.get("FLUIDSYNTH") or shutil.which("fluidsynth")
    if not cand or not Path(cand).exists():
        raise FileNotFoundError("FluidSynth not found: install it (conda install -c conda-forge fluidsynth) "
                                "or set FLUIDSYNTH to the program")
    return str(cand)


def find_soundfonts(spec: str | list | None = None) -> list[Path]:
    """SoundFont files: ``spec`` (a list, or paths joined by os.pathsep), else $SOUNDFONTS."""
    if spec is None:
        spec = os.environ.get("SOUNDFONTS", "")
    items = spec if isinstance(spec, (list, tuple)) else [s for s in str(spec).split(os.pathsep) if s]
    fonts = [Path(s) for s in items]
    missing = [str(f) for f in fonts if not f.exists()]
    if not fonts or missing:
        raise FileNotFoundError(f"SoundFonts not found: {missing or 'none given'} (--soundfonts or SOUNDFONTS)")
    return fonts


def write_midi(notes: np.ndarray, path, pedal: np.ndarray | None = None) -> Path:
    """Notes [N, 4] (pitch, onset, offset, velocity in seconds, as load_performance gives them) and
    sustain-pedal intervals [K, 2] -> a one-track piano MIDI file."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for p, on, off, v in np.asarray(notes, np.float64).reshape(-1, 4):
        on = max(float(on), 0.0)
        inst.notes.append(pretty_midi.Note(velocity=int(np.clip(round(v), 1, 127)), pitch=int(np.clip(p, 0, 127)),
                                           start=on, end=max(float(off), on + 0.01)))
    for a, b in (np.asarray(pedal, np.float64).reshape(-1, 2) if pedal is not None else []):
        inst.control_changes += [pretty_midi.ControlChange(64, 127, max(float(a), 0.0)),
                                 pretty_midi.ControlChange(64, 0, max(float(b), 0.0))]
    pm.instruments.append(inst)
    pm.write(str(path))
    return Path(path)


class Renderer:
    """``renderer(source, wav, key)``: ``source`` a MIDI file, or notes [N, 4] (optionally (notes,
    pedal)), rendered to ``wav`` with the SoundFont chosen for ``key``. Returns the SoundFont used."""

    def __init__(self, soundfonts=None, fluidsynth: str | None = None, gain: float = GAIN, sr: int = SR):
        self.fonts = find_soundfonts(soundfonts)
        self.exe = find_fluidsynth(fluidsynth)
        self.gain, self.sr = float(gain), int(sr)

    def font_for(self, key: str) -> Path:
        return self.fonts[zlib.crc32(str(key).encode()) % len(self.fonts)]

    def __call__(self, source, wav, key: str = "") -> Path:
        wav = Path(wav)
        font = self.font_for(key)
        midi = source
        if not isinstance(source, (str, Path)):
            notes, pedal = source if isinstance(source, tuple) else (source, None)
            midi = write_midi(notes, wav.with_suffix(".mid"), pedal)
        r = subprocess.run([self.exe, "-ni", "-q", "-g", str(self.gain), "-r", str(self.sr), "-F", str(wav),
                            str(font), str(midi)], capture_output=True, text=True)
        if r.returncode != 0 or not wav.exists():
            raise RuntimeError(f"FluidSynth failed on {midi}: {(r.stderr or r.stdout).strip()[:300]}")
        if midi is not source:
            Path(midi).unlink(missing_ok=True)
        return font
