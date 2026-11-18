"""End-to-end inference: piano recording (or performance MIDI) -> engraved score.

    audio --Transkun V2--> performance MIDI --note windows--> Scoreformer (chained) --> tokens
          --> MusicXML + score MIDI

Used by the demo (demo/app.py) and by scripts/audio_to_score.py. Windows are cut exactly like the
test sets (128 notes, cut at the largest onset gap nearby, carried-over notes and pedal included),
and each window is decoded with the model's own previous output as its prompt ("chained" mode in
scripts/evaluate_pieces.py), so the demo runs the system the way it is evaluated.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .dataset import note_features
from .preprocessing.performance import OFFSET, ONSET, load_performance
from .preprocessing.segment import SegmentConfig, cut_windows, prompt_from_output
from .preprocessing.vocab import BOS, EOS, SEP, Vocab
from .scoreformer import load_model

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif"}
MIDI_EXT = {".mid", ".midi"}
CARRY_MIN_SEC = 0.125  # training uses 1/4 quarter note past the cut; ~0.125 s at 120 bpm

Progress = Callable[[str, float, str], None]  # (stage, fraction 0..1, message)


def _noop(stage: str, frac: float, msg: str):
    pass


def pedal_down(tokens: list[str]) -> bool:
    for t in reversed(tokens):
        if t in ("ped_on", "ped_off"):
            return t == "ped_on"
    return False


def stitch(history: list[str], window: list[str]) -> list[str]:
    """Append a window's tokens, dropping its opening ``ped_on`` if the pedal is already down."""
    if pedal_down(history):
        first = next((k for k, t in enumerate(window) if t.startswith("ped_")), None)
        if first is not None and window[first] == "ped_on":
            window = window[:first] + window[first + 1:]
    return history + window


def performance_windows(notes: np.ndarray, pedal: np.ndarray, size: int = 128,
                        search: float | None = None) -> list[dict]:
    """Cut a performance (no score) into model inputs, mirroring segment.segment_windows at eval
    time: cut points at the midpoint of the largest onset gap near every ``size`` notes, onsets
    relative to the cut, notes struck before the cut but still held past it carried over with a
    negative onset, pedal intervals clipped to the window. Returns raw [N,4] notes and pedal."""
    if search is None:
        search = SegmentConfig().cut_search
    on = notes[:, ONSET].astype(np.float64)
    off = notes[:, OFFSET].astype(np.float64)
    cuts = cut_windows(on, lambda: size, 0.0, search)
    out = []
    for a, b in cuts:
        t0 = float(on[a]) if a == 0 else float(on[a - 1] + on[a]) / 2
        t1 = float(off[a:b].max()) if b == len(on) else float(on[b - 1] + on[b]) / 2
        carried = np.nonzero((on[:a] < t0) & (off[:a] >= t0 + CARRY_MIN_SEC))[0] if a else np.zeros(0, int)
        sel = np.concatenate([notes[carried], notes[a:b]])
        w = np.stack([sel[:, 0], sel[:, ONSET] - t0, sel[:, OFFSET] - sel[:, ONSET], sel[:, 3]], axis=1)
        w[: len(carried), 1] = np.minimum(w[: len(carried), 1], -1e-3)
        ped = pedal[(pedal[:, 1] > t0) & (pedal[:, 0] < t1)] - t0 if len(pedal) else pedal.reshape(-1, 2)
        out.append({"notes": w.astype(np.float32), "pedal": np.asarray(ped, np.float32).reshape(-1, 2),
                    "t0": t0, "t1": t1, "n": int(b - a), "carried": int(len(carried))})
    return out


@dataclass
class Result:
    tokens: list[str]
    files: dict = field(default_factory=dict)      # kind -> path
    stats: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


class ScorePipeline:
    """Loads the score model (and, on first audio input, Transkun) once; ``run`` per file."""

    def __init__(self, ckpt: str | Path, vocab: str | Path | None = None, device: str = "auto",
                 window_notes: int = 128):
        ckpt = Path(ckpt)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = load_model(ckpt, self.device)
        self.cfg = self.model.cfg
        self.amp = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.window_notes = window_notes
        vocab_path = self._find(vocab, ckpt, "vocab.json")
        self.vocab = Vocab.load(str(vocab_path))
        if len(self.vocab) != self.cfg.vocab_size:  # a different data build orders tokens differently
            raise ValueError(f"{vocab_path} has {len(self.vocab)} tokens but the checkpoint expects "
                             f"{self.cfg.vocab_size}: use the vocab.json of the data the model was trained on "
                             f"(copy it next to {ckpt.name})")
        mean, std = getattr(self.cfg, "feat_mean", None), getattr(self.cfg, "feat_std", None)
        if mean is None or not any(mean):  # Score2Seq checkpoints don't carry the stats
            st = json.loads(self._find(None, ckpt, "feature_stats.json").read_text())
            mean, std = st["mean"], st["std"]
        self.scale = (np.array([1, 2, 3, 4]), np.asarray(mean, np.float32), np.asarray(std, np.float32))
        self.bos, self.sep, self.eos = (self.vocab.stoi[t] for t in (BOS, SEP, EOS))
        self.ckpt = ckpt
        self._transkun = None
        n_params = sum(p.numel() for p in self.model.parameters())
        self.info = {"checkpoint": str(ckpt), "arch": type(self.model).__name__,
                     "params_M": round(n_params / 1e6, 1), "device": str(self.device)}

    @staticmethod
    def _find(given, ckpt: Path, name: str) -> Path:
        """vocab.json / feature_stats.json: explicit path, next to the checkpoint, or data/processed."""
        for p in ([Path(given)] if given else []) + [ckpt.parent / name, Path("data/processed") / name]:
            if p.exists():
                return p
        raise FileNotFoundError(f"{name} not found next to {ckpt} or in data/processed")

    # ------------------------------------------------------------------ steps
    def transcribe(self, audio: Path, midi_out: Path) -> Path:
        if self._transkun is None:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from scripts.transcribe import Transcriber
            self._transkun = Transcriber("cuda" if self.device.type == "cuda" else "cpu")
        self._transkun(str(audio), str(midi_out))
        return midi_out

    def features(self, w: dict) -> torch.Tensor:
        f = note_features(w["notes"], w["pedal"])
        cols, mean, std = self.scale
        f[:, cols] = (f[:, cols] - mean) / std
        return torch.from_numpy(f[: self.cfg.max_notes])

    def prefix(self, prompt: list[str]) -> torch.Tensor:
        if not prompt:
            return torch.tensor([self.bos])
        return torch.tensor([self.bos] + self.vocab.encode(prompt, add_special=False) + [self.sep])

    @torch.no_grad()
    def decode_window(self, feats: torch.Tensor, prefix: torch.Tensor) -> list[str]:
        notes = feats[None].to(self.device)
        mask = torch.ones(1, len(feats), dtype=torch.bool, device=self.device)
        # test windows average 3.4 target tokens per input note (99th percentile 4.7): the cap only
        # stops a degenerate loop from running to max_tokens on every window
        max_new = max(16, min(self.cfg.max_tokens - len(prefix), 6 * len(feats) + 64))
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            ids = self.model.generate(notes, mask, [prefix], self.eos, max_new=max_new)[0]
        return self.vocab.decode(ids.tolist() if torch.is_tensor(ids) else ids)

    def decode(self, windows: list[dict], mode: str = "chained", progress: Progress = _noop) -> list[str]:
        history: list[str] = []
        for j, w in enumerate(windows):
            prompt = prompt_from_output(history) if (mode == "chained" and j) else []
            out = self.decode_window(self.features(w), self.prefix(prompt))
            history = stitch(history, out)
            progress("score", (j + 1) / len(windows), f"window {j + 1} of {len(windows)}")
        return history

    # ------------------------------------------------------------------ all of it
    def run(self, src: str | Path, out_dir: str | Path, mode: str = "chained",
            progress: Progress = _noop) -> Result:
        src, out = Path(src), Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        timing, files = {}, {}
        t = time.perf_counter()
        ext = src.suffix.lower()
        if ext in MIDI_EXT:
            midi = src
        elif ext in AUDIO_EXT:
            progress("transcribe", 0.0, "transcribing audio with Transkun")
            midi = self.transcribe(src, out / "transcription.mid")
            timing["transcribe_s"] = round(time.perf_counter() - t, 1)
        else:
            raise ValueError(f"unsupported file type {ext!r} (audio: {sorted(AUDIO_EXT)}, or .mid)")
        files["transcription_midi"] = midi

        notes, pedal = load_performance(midi)
        if len(notes) == 0:
            raise ValueError("no notes found (silent recording, or not a piano?)")
        windows = performance_windows(notes, pedal, self.window_notes)
        progress("score", 0.0, f"window 0 of {len(windows)}")
        t = time.perf_counter()
        tokens = self.decode(windows, mode, progress)
        timing["score_s"] = round(time.perf_counter() - t, 1)
        (out / "tokens.txt").write_text(" ".join(tokens))
        files["tokens"] = out / "tokens.txt"

        progress("engrave", 0.0, "writing MusicXML")
        res = Result(tokens, files)
        try:
            from .preprocessing.detokenize import tokens_to_musicxml, tokens_to_score
            files["musicxml"] = Path(tokens_to_musicxml(tokens, str(out / "score.musicxml")))
            for clip in (False, True):  # clip notes to their bars if the plain export fails
                try:
                    tokens_to_score(tokens, clip=clip).write("midi", fp=str(out / "score.mid"))
                    files["score_midi"] = out / "score.mid"
                    break
                except Exception as e:  # noqa: BLE001
                    if clip:
                        res.errors.append(f"score MIDI: {type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            res.errors.append(f"MusicXML: {type(e).__name__}: {e}")

        dur = float(notes[:, OFFSET].max() - notes[:, ONSET].min())
        res.stats = {**self.info, "mode": mode, "duration_s": round(dur, 1), "notes": int(len(notes)),
                     "pedal_presses": int(len(pedal)), "windows": len(windows),
                     "bars": tokens.count("bar"), "tokens": len(tokens), **timing}
        (out / "result.json").write_text(json.dumps({"stats": res.stats, "errors": res.errors,
                                                     "files": {k: str(v) for k, v in files.items()}}, indent=1))
        return res
