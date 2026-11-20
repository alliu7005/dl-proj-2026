"""End-to-end inference: piano recording (or performance MIDI) -> engraved score.

    audio --Transkun V2--> performance MIDI --note windows--> score model (meter) --> tokens
          --> MusicXML + score MIDI

Used by the demo (demo/app.py) and by scripts/audio_to_score.py. Windows are cut exactly like the
test sets (128 notes, cut at the largest onset gap nearby, carried-over notes and pedal included)
and decoded by src/decoding.py, the same code scripts/evaluate_pieces.py scores, with the same
per-window output budget. Modes:
  meter    (default) guarded, plus consistent time signatures: a window whose time signature
           disagrees with a clear majority of its neighbours is decoded again with that one forced,
           and the piece re-decoded from there (decoding.decode_pieces_meter). The best mode in the
           evaluation (outputs/compare/summary.md): for sf_base_reg_cab (the default model) it beats
           guarded on bar F1 on every full test set (+0.003 to +0.015), onset F1 within 0.006
  guarded  each window's context is the model's own previous output, but a window whose output
           clearly disagrees with its input notes (a degenerate window) is replaced by the window
           decoded without context, so one bad window cannot derail the rest of the piece
  piece    meter, with one time signature for the whole piece unless a different one holds
           for a stretch as long as the rest
  chained  the model's own previous output as context, unguarded
  none     every window decoded on its own
  tsearch  the piece decoded under its 3 likeliest time signatures (forced in every window, guarded),
           keeping the one the model finds most probable (decoding.decode_pieces_tsearch)
  tsearch_beat  the same candidates, keeping the one whose written beats agree best with Beat This!'s
           beats (src/beats.py grid_agreement); as tsearch without beats (MIDI input, non-beat model)

Beats: a model trained with beat features (src/beats.py) gets the beats of the recording from
Beat This! (scripts/track_beats.py Tracker, checkpoint final0) after transcription. MIDI input, or
an environment without the beat-this package, gives the "no beats" features the model was also
trained on (``beat_dropout``), and the result says which it was (stats "beats").
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .dataset import assemble_notes, note_features
from .beats import beat_features, grid_agreement, window_beats
from .decoding import (DEFAULT_GUARD_MARGIN, Scored, decode_pieces, decode_pieces_meter, decode_pieces_tsearch,
                       max_new_tokens)
from .eval.plan import PIECE_MAX_BLIP
from .preprocessing.performance import OFFSET, ONSET, load_performance
from .preprocessing.segment import SegmentConfig, cut_windows
from .preprocessing.vocab import BOS, EOS, SEP, Vocab
from .scoreformer import load_model

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif"}
MIDI_EXT = {".mid", ".midi"}
MODES = ("meter", "piece", "tsearch", "tsearch_beat", "guarded", "chained", "none")  # the first is the default
GUARD_MARGIN = DEFAULT_GUARD_MARGIN  # the same default as scripts/evaluate_pieces.py
CARRY_MIN_SEC = 0.125  # training uses 1/4 quarter note past the cut; ~0.125 s at 120 bpm

Progress = Callable[[str, float, str], None]  # (stage, fraction 0..1, message)


def _noop(stage: str, frac: float, msg: str):
    pass


def performance_windows(notes: np.ndarray, pedal: np.ndarray, size: int = 128,
                        search: float | None = None, context_before: int = 0,
                        context_after: int = 0, beats: tuple | None = None) -> list[dict]:
    """Cut a performance (no score) into model inputs, mirroring segment.segment_windows at eval
    time: cut points at the midpoint of the largest onset gap near every ``size`` notes, onsets
    relative to the cut, notes struck before the cut but still held past it carried over with a
    negative onset, pedal intervals clipped to the window. Returns raw [N,4] notes and pedal.
    With ``context_before`` / ``context_after``, each window also gets that many of the performed
    notes just before / after it ("before" / "after", same time frame; the carried-over notes are
    not repeated), as src/dataset.py gives the model during training. ``beats`` = (times, down)
    in the performance's seconds: each window gets the beats around it ("beat_t" / "beat_d", in its
    own time frame), as src/dataset.py does."""
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
        lo, hi = t0, t1
        before = after = np.zeros((0, 4), np.float32)
        if context_before and a:
            prev = np.setdiff1d(np.arange(a), carried)[-context_before:]
            before = np.stack([notes[prev, 0], on[prev] - t0, off[prev] - on[prev], notes[prev, 3]], axis=1)
            lo = float(on[prev].min()) if len(prev) else t0
        if context_after and b < len(on):
            nxt = np.arange(b, min(len(on), b + context_after))
            after = np.stack([notes[nxt, 0], on[nxt] - t0, off[nxt] - on[nxt], notes[nxt, 3]], axis=1)
            hi = max(t1, float(off[nxt].max()))
        ped = pedal[(pedal[:, 1] > lo) & (pedal[:, 0] < hi)] - t0 if len(pedal) else pedal.reshape(-1, 2)
        bt, bd = window_beats(beats[0], beats[1], t0, lo, hi) if beats is not None else (np.zeros(0), np.zeros(0, bool))
        out.append({"notes": w.astype(np.float32), "pedal": np.asarray(ped, np.float32).reshape(-1, 2),
                    "before": np.asarray(before, np.float32), "after": np.asarray(after, np.float32),
                    "t0": t0, "t1": t1, "n": int(b - a), "carried": int(len(carried)),
                    "beat_t": bt, "beat_d": bd})
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
        self._tracker = None
        n_params = sum(p.numel() for p in self.model.parameters())
        self.info = {"checkpoint": str(ckpt), "arch": type(self.model).__name__,
                     "params_M": round(n_params / 1e6, 1), "device": str(self.device),
                     "beats": bool(getattr(self.cfg, "beat_features", False))}

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

    def track_beats(self, audio: Path):
        """(times, down) from Beat This!, or None if it is not installed."""
        if self._tracker is None:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from scripts.track_beats import Tracker
            try:
                self._tracker = Tracker("cuda" if self.device.type == "cuda" else "cpu")
            except ImportError:
                self._tracker = False
        return self._tracker(audio) if self._tracker else None

    def features(self, w: dict) -> torch.Tensor:
        """Model input for a window: its notes, plus context notes and beat features if the model
        was trained with them (the same layout as src/dataset.py)."""
        if int(getattr(self.cfg, "context_before", 0)) or int(getattr(self.cfg, "context_after", 0)):
            notes, roles = assemble_notes(w["notes"], w.get("before", np.zeros((0, 4))), w.get("after", np.zeros((0, 4))))
            f = note_features(notes, w["pedal"], roles)
        else:
            notes = w["notes"]
            f = note_features(notes, w["pedal"])
        cols, mean, std = self.scale
        f[:, cols] = (f[:, cols] - mean) / std
        if getattr(self.cfg, "beat_features", False):
            beats = beat_features(notes[:, 1], w.get("beat_t", np.zeros(0)), w.get("beat_d", np.zeros(0, bool)))
            f = np.concatenate([f, beats], axis=1)
        return torch.from_numpy(f[: self.cfg.max_notes])

    def prefix(self, prompt: list[str], start: list[str] = ()) -> torch.Tensor:
        """<bos> [prompt <sep>] [start]: ``start`` = tokens the output is forced to begin with."""
        head = [self.bos] + (self.vocab.encode(prompt, add_special=False) + [self.sep] if prompt else [])
        return torch.tensor(head + self.vocab.encode(list(start), add_special=False))

    @torch.no_grad()
    def decode_window(self, feats: torch.Tensor, prefix: torch.Tensor, n_notes: int | None = None,
                      ban_ts: bool = False) -> Scored:
        """``n_notes``: the window's own notes (without context notes), for the output budget.
        ``ban_ts``: generate no time signature (the window's own was forced in the prefix). The
        output carries the log-probability of its generated tokens (tsearch ranks by it)."""
        notes = feats[None].to(self.device)
        mask = torch.ones(1, len(feats), dtype=torch.bool, device=self.device)
        max_new = max_new_tokens(len(feats) if n_notes is None else n_notes, len(prefix), self.cfg.max_tokens)
        banned = None
        if ban_ts:
            banned = torch.zeros(1, len(self.vocab.itos), dtype=torch.bool)
            banned[0, [i for i, t in enumerate(self.vocab.itos) if t.startswith("ts_")]] = True
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            ids, lps = self.model.generate(notes, mask, [prefix], self.eos, max_new=max_new, banned=banned,
                                           return_logp=True)
        ids, lp = list(ids[0]), lps[0]
        lp = lp[:max_new] if len(ids) > max_new else lp
        return Scored(self.vocab.decode(ids[:max_new]), sum(lp), len(lp))

    def decode(self, windows: list[dict], mode: str = MODES[0], progress: Progress = _noop,
               guard_margin: float = GUARD_MARGIN) -> tuple[list[str], dict]:
        """Decode a piece window by window (see the module docstring for the modes). Returns the
        stitched tokens and {"guard_fallbacks": windows replaced by their no-context output,
        "meter_fixes": windows whose time signature meter mode corrected, and for tsearch modes
        "time_signature": the one kept and "ts_candidates": the ones tried}."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")

        def decode(ws, contexts, starts):
            out = []
            for w, c, s in zip(ws, contexts, starts):
                o = self.decode_window(self.features(w), self.prefix(c, s), len(w["notes"]),
                                       ban_ts=any(t.startswith("ts_") for t in s))
                out.append(Scored(list(s) + o, o.logp, o.n))
            return out

        kw = dict(contiguous=(lambda prev, w: mode != "none"), pitches=lambda w: w["notes"][:, 0].tolist(),
                  progress=lambda j, total: progress("score", j / total, f"window {j} of {total}"))
        forced = {}
        if mode in ("tsearch", "tsearch_beat"):
            alone = {}
            first = decode_pieces({0: windows}, decode, guard_margin=guard_margin, alone=alone, **kw)

            def beat_score(w, tokens):
                if len(w.get("beat_t", ())) < 2:
                    return None
                return grid_agreement(w["notes"][:, 1], w["notes"][:, 0], tokens, w["beat_t"], w["beat_d"])
            res, cands = decode_pieces_tsearch({0: windows}, decode, guard_margin=guard_margin, first_pass=first,
                                               alone=alone, beat_score=beat_score, **kw)
            _, history, fallbacks, chosen = res[mode]
            return history[0], {"guard_fallbacks": len(fallbacks), "meter_fixes": 0,
                                "time_signature": chosen[0].replace("ts_", ""),
                                "ts_candidates": [c.replace("ts_", "") for c in cands[0]]}
        if mode in ("meter", "piece"):
            _, history, fallbacks, forced = decode_pieces_meter(
                {0: windows}, decode, guard_margin=guard_margin,
                max_blip=PIECE_MAX_BLIP if mode == "piece" else 2, **kw)
        else:
            _, history, fallbacks = decode_pieces({0: windows}, decode,
                                                  guard_margin=guard_margin if mode == "guarded" else None, **kw)
        return history[0], {"guard_fallbacks": len(fallbacks), "meter_fixes": len(forced)}

    # ------------------------------------------------------------------ all of it
    def run(self, src: str | Path, out_dir: str | Path, mode: str = MODES[0],
            progress: Progress = _noop, guard_margin: float = GUARD_MARGIN) -> Result:
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
        beats, beat_info = None, "not used by this model"
        if getattr(self.cfg, "beat_features", False):
            if ext in AUDIO_EXT:
                progress("beats", 0.0, "tracking beats with Beat This!")
                t = time.perf_counter()
                beats = self.track_beats(src)
                timing["beats_s"] = round(time.perf_counter() - t, 1)
                beat_info = (f"Beat This!: {len(beats[0])} beats, {int(beats[1].sum())} downbeats" if beats is not None
                             else "none (Beat This! not installed: pip install beat-this rotary-embedding-torch einops)")
            else:
                beat_info = "none (MIDI input)"
        windows = performance_windows(notes, pedal, self.window_notes,
                                      context_before=int(getattr(self.cfg, "context_before", 0)),
                                      context_after=int(getattr(self.cfg, "context_after", 0)), beats=beats)
        progress("score", 0.0, f"window 0 of {len(windows)}")
        t = time.perf_counter()
        tokens, counts = self.decode(windows, mode, progress, guard_margin)
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
                     "pedal_presses": int(len(pedal)), "windows": len(windows), "beats": beat_info, **counts,
                     "bars": tokens.count("bar"), "tokens": len(tokens), **timing}
        (out / "result.json").write_text(json.dumps({"stats": res.stats, "errors": res.errors,
                                                     "files": {k: str(v) for k, v in files.items()}}, indent=1))
        return res
