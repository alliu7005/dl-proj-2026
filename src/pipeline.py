"""End-to-end inference: piano recording (or performance MIDI) -> engraved score.

    ScorePipeline (seq2seq):
    audio --Transkun V2--> performance MIDI --note windows--> score model (meter) --> tokens
          --> MusicXML + score MIDI
    V2Pipeline (v2, at the end of this file):
    audio --Transkun V2--> performance MIDI, --Beat This!--> beats --meter model--> the piece's grid
          --note labeller, every window in its stretch of the grid--> one score --> MusicXML + MIDI

Used by the demo (demo/app.py), scripts/system_scores.py and scripts/audio_to_score.py. Windows
are cut exactly like the test sets (128 notes, cut at the largest onset gap nearby, carried-over
notes and pedal included). ScorePipeline decodes them with src/decoding.py, the same code
scripts/evaluate_pieces.py scores, with the same per-window output budget. Modes:
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
                        context_after: int = 0, beats: tuple | None = None,
                        drop: np.ndarray | None = None) -> list[dict]:
    """Cut a performance (no score) into model inputs, mirroring segment.segment_windows at eval
    time: cut points at the midpoint of the largest onset gap near every ``size`` notes, onsets
    relative to the cut, notes struck before the cut but still held past it carried over with a
    negative onset, pedal intervals clipped to the window. Returns raw [N,4] notes and pedal.
    With ``context_before`` / ``context_after``, each window also gets that many of the performed
    notes just before / after it ("before" / "after", same time frame; the carried-over notes are
    not repeated), as src/dataset.py gives the model during training. ``beats`` = (times, down)
    in the performance's seconds: each window gets the beats around it ("beat_t" / "beat_d", in its
    own time frame), as src/dataset.py does. ``drop`` (bool [N]): notes left out of every window's
    input, after the cuts were made on all the notes. "idx", "idx_before", "idx_after": the
    performance's indices of "notes", "before", "after"."""
    if search is None:
        search = SegmentConfig().cut_search
    on = notes[:, ONSET].astype(np.float64)
    off = notes[:, OFFSET].astype(np.float64)
    cuts = cut_windows(on, lambda: size, 0.0, search)
    keep = np.ones(len(notes), bool) if drop is None else ~np.asarray(drop, bool)
    out = []
    for a, b in cuts:
        t0 = float(on[a]) if a == 0 else float(on[a - 1] + on[a]) / 2
        t1 = float(off[a:b].max()) if b == len(on) else float(on[b - 1] + on[b]) / 2
        carried = np.nonzero((on[:a] < t0) & (off[:a] >= t0 + CARRY_MIN_SEC) & keep[:a])[0] if a else np.zeros(0, int)
        own = np.arange(a, b)[keep[a:b]]
        sel = np.concatenate([notes[carried], notes[own]])
        w = np.stack([sel[:, 0], sel[:, ONSET] - t0, sel[:, OFFSET] - sel[:, ONSET], sel[:, 3]], axis=1)
        w[: len(carried), 1] = np.minimum(w[: len(carried), 1], -1e-3)
        lo, hi = t0, t1
        before = after = np.zeros((0, 4), np.float32)
        if context_before and a:
            prev = np.setdiff1d(np.arange(a)[keep[:a]], carried)[-context_before:]
            before = np.stack([notes[prev, 0], on[prev] - t0, off[prev] - on[prev], notes[prev, 3]], axis=1)
            lo = float(on[prev].min()) if len(prev) else t0
        if context_after and b < len(on):
            nxt = np.arange(b, len(on))[keep[b:]][:context_after]
            after = np.stack([notes[nxt, 0], on[nxt] - t0, off[nxt] - on[nxt], notes[nxt, 3]], axis=1)
            hi = max(t1, float(off[nxt].max()))
        ped = pedal[(pedal[:, 1] > lo) & (pedal[:, 0] < hi)] - t0 if len(pedal) else pedal.reshape(-1, 2)
        bt, bd = window_beats(beats[0], beats[1], t0, lo, hi) if beats is not None else (np.zeros(0), np.zeros(0, bool))
        out.append({"notes": w.astype(np.float32), "pedal": np.asarray(ped, np.float32).reshape(-1, 2),
                    "before": np.asarray(before, np.float32), "after": np.asarray(after, np.float32),
                    "idx": np.concatenate([carried, own]).astype(np.int64),
                    "idx_before": np.asarray(prev if context_before and a else [], np.int64),
                    "idx_after": np.asarray(nxt if context_after and b < len(on) else [], np.int64),
                    "t0": t0, "t1": t1, "n": int(len(own)), "carried": int(len(carried)),
                    "beat_t": bt, "beat_d": bd})
    return out


def playback_qpm(score, seconds: float | None) -> float | None:
    """The tempo (quarter notes per minute) at which a music21 score sounds as long as the
    performance: from its first note's start to its last note's end, in quarters, over the
    performance's first onset to last offset in seconds, within 20-300. The score MIDI is played at
    it; without, music21 plays every score at 120 (a slow piece twice too fast)."""
    if not seconds or seconds <= 0:
        return None
    notes = [n for n in score.flatten().notes if not n.duration.isGrace]
    if not notes:
        return None
    first = min(float(n.offset) for n in notes)
    last = max(float(n.offset + n.quarterLength) for n in notes)
    if last <= first:
        return None
    return float(np.clip(60.0 * (last - first) / seconds, 20.0, 300.0))


def engrave(tokens: list[str], out: Path, res: "Result", seconds: float | None = None,
            glissandi: list | None = None, stems: bool = False) -> dict:
    """Tokens -> ``out``/score.musicxml and score.mid (``res.files``; failures in ``res.errors``).
    ``seconds``: the performance's length, which sets the score MIDI's tempo (``playback_qpm``; the
    MusicXML carries no tempo, as the scores the comparison rates). Returns how the export went
    (src/preprocessing/detokenize.py tokens_to_musicxml: the fallback that worked, the bars it had
    to empty) and "playback_qpm". ``glissandi`` (src/glissando.py attach): a "gliss." line each in
    the MusicXML, the swept notes in the score MIDI ("glissandi_drawn": how many lines). The score
    MIDI plays grace notes and rolled chords as written (src/playback.py: "graces_played",
    "rolled_chords_played"), which music21's MIDI export alone does not. ``stems``: the higher
    voice's stems up, the others' down where a staff has two voices (src/readable.py stem_voices;
    "stems_set": how many measures)."""
    import music21 as m21
    from .preprocessing.detokenize import tokens_to_musicxml, tokens_to_score
    report = {}
    deco = None
    if glissandi or stems:
        def deco(score):
            if glissandi:
                from .glissando import decorate
                report["glissandi_drawn"] = decorate(score, glissandi)
            if stems:
                from .readable import stem_voices
                report["stems_set"] = stem_voices(score)
    try:
        res.files["musicxml"] = Path(tokens_to_musicxml(tokens, str(out / "score.musicxml"), report=report,
                                                        decorate=deco))
        for clip in (False, True):  # clip notes to their bars if the plain export fails
            try:
                score = tokens_to_score(tokens, clip=clip)
                qpm = playback_qpm(score, seconds)
                from .playback import realize   # grace notes and rolled chords as written (src/playback.py)
                report.update(realize(score, qpm))
                if glissandi:
                    from .glissando import decorate
                    decorate(score, glissandi, playback=True)
                report["playback_qpm"] = None if qpm is None else round(qpm, 1)
                if qpm:
                    first = score.parts[0].getElementsByClass(m21.stream.Measure).first()
                    first.insert(0, m21.tempo.MetronomeMark(number=round(qpm, 1)))
                score.write("midi", fp=str(out / "score.mid"))
                res.files["score_midi"] = out / "score.mid"
                break
            except Exception as e:  # noqa: BLE001
                if clip:
                    res.errors.append(f"score MIDI: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"MusicXML: {type(e).__name__}: {e}")
    if report.get("emptied_bars"):
        res.errors.append(f"MusicXML: {len(report['emptied_bars'])} bars could not be engraved and were left "
                          f"empty: {report['emptied_bars'][:20]}")
    return report


@dataclass
class Result:
    tokens: list[str]
    files: dict = field(default_factory=dict)      # kind -> path
    stats: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


BEAT_SOURCES = ("beat_this", "pm2s")


class _FrontEnd:
    """Audio -> performance MIDI (Transkun) and beats (Beat This!, or PM2S from the notes), loaded
    on first use. Needs ``self.device``."""
    _transkun = None
    _tracker = None
    _pm2s = None

    def transcribe(self, audio: Path, midi_out: Path) -> Path:
        if self._transkun is None:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from scripts.transcribe import Transcriber
            self._transkun = Transcriber("cuda" if self.device.type == "cuda" else "cpu")
        self._transkun(str(audio), str(midi_out))
        return midi_out

    def track_beats(self, audio: Path, checkpoint: str = "final0"):
        """(times, down) from Beat This! (``checkpoint``: final0, or the cross-validation model
        fold<k> that never heard a test piece), or None if it is not installed."""
        if self._tracker is None:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from scripts.track_beats import Tracker
            try:
                self._tracker = Tracker("cuda" if self.device.type == "cuda" else "cpu")
            except ImportError:
                self._tracker = False
        return self._tracker(audio, checkpoint) if self._tracker else None

    def pm2s_beats(self, notes: np.ndarray):
        """(times, down) from PM2S on the notes themselves (src/pm2s_beats.py; no audio needed):
        for a MIDI input, the alternative to Beat This! on the MIDI rendered to audio."""
        if self._pm2s is None:
            from .pm2s_beats import PM2SBeats
            self._pm2s = PM2SBeats(device="cuda" if self.device.type == "cuda" else "cpu")
        return self._pm2s(notes)

    def track_midi_beats(self, notes: np.ndarray, checkpoint: str = "final0"):
        """Beats of a MIDI input: Beat This! on the notes rendered to audio (src/synth.py), or None
        without Beat This!."""
        import tempfile
        from .synth import render, write_wav
        with tempfile.TemporaryDirectory() as tmp:
            wav = write_wav(Path(tmp) / "performance.wav", render(notes))
            return self.track_beats(wav, checkpoint)


class ScorePipeline(_FrontEnd):
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
            progress: Progress = _noop, guard_margin: float = GUARD_MARGIN, beats: tuple | None = None,
            midi_beats: bool = False) -> Result:
        """``beats``: (times, down) in the input's seconds, given instead of tracking them (e.g. the
        tracked beats of a transcription already made; scripts/system_scores.py). Only used by a
        model trained with beats. ``midi_beats``: a MIDI input without given beats gets Beat This!'s
        beats of the notes rendered to audio (src/synth.py) instead of none."""
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
        given, beats, beat_info = beats, None, "not used by this model"
        if getattr(self.cfg, "beat_features", False):
            if given is not None:
                beats = (np.asarray(given[0], np.float64), np.asarray(given[1], bool))
                beat_info = f"given: {len(beats[0])} beats, {int(beats[1].sum())} downbeats"
            elif ext in AUDIO_EXT:
                progress("beats", 0.0, "tracking beats with Beat This!")
                t = time.perf_counter()
                beats = self.track_beats(src)
                timing["beats_s"] = round(time.perf_counter() - t, 1)
                beat_info = (f"Beat This!: {len(beats[0])} beats, {int(beats[1].sum())} downbeats" if beats is not None
                             else "none (Beat This! not installed: pip install beat-this rotary-embedding-torch einops)")
            elif midi_beats:
                progress("beats", 0.0, "tracking beats with Beat This! (MIDI rendered to audio)")
                t = time.perf_counter()
                beats = self.track_midi_beats(notes)
                timing["beats_s"] = round(time.perf_counter() - t, 1)
                beat_info = (f"Beat This! on the MIDI rendered to audio: {len(beats[0])} beats, "
                             f"{int(beats[1].sum())} downbeats" if beats is not None
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
        dur = float(notes[:, OFFSET].max() - notes[:, ONSET].min())
        export = engrave(tokens, out, res, dur)

        res.stats = {**self.info, "mode": mode, "duration_s": round(dur, 1), "notes": int(len(notes)),
                     "pedal_presses": int(len(pedal)), "windows": len(windows), "beats": beat_info, **counts,
                     "bars": tokens.count("bar"), "tokens": len(tokens), "export": export,
                     "playback_qpm": export.get("playback_qpm"), **timing}
        (out / "result.json").write_text(json.dumps({"stats": res.stats, "errors": res.errors,
                                                     "files": {k: str(v) for k, v in files.items()}}, indent=1))
        return res


# ---------------------------------------------------------------------------------------------
# v2: the metrical grid of the whole piece first, then every note labelled against it
# ---------------------------------------------------------------------------------------------

# A third or fourth staff is written only when at least this share of the piece's notes is on it.
# In the target scores such staves hold 28-100% of a piece's notes, but for one piece a single short
# passage holds 0.6%. Below this share they are stray predictions (one note of 1,221 in a demo piece
# opened two staves for all 73 bars), and those notes go on their likeliest of the staves kept.
MIN_EXTRA_STAFF = 0.02


def piece_staves(staffs: list[int]) -> int:
    """How many staves a piece's score gets from its notes' predicted staves: always the first two,
    and a higher one (and those below it) when it holds at least MIN_EXTRA_STAFF of the notes."""
    if not staffs:
        return 2
    count = np.bincount(np.asarray(staffs, np.int64))
    kept = [s for s in range(2, len(count)) if count[s] >= MIN_EXTRA_STAFF * len(staffs)]
    return max([2] + [s + 1 for s in kept])


DELETE_ABOVE = 0.5     # v2 leaves a note out when the labeller's p(delete) is above this (as evaluated)
GRACE_REACH = 1        # quarters: a grace note joins a main note of its staff at most this far after it
GRACE_AS_NOTE = 0.25   # quarters: the longest written length of a "grace" note with no main note to join


TIDY_TOL = 0.125       # quarters: a written note end at most this far past the next onset of its staff stops there


def _bar_starts(placed: list, bar_ts: list[str]) -> list:
    """Each bar's start in quarters from the first bar line (Fractions), the last bar's length
    repeated past the grid for notes placed after it."""
    from fractions import Fraction
    from .meter import bar_quarters
    lens = [Fraction(bar_quarters(ts)) for ts in bar_ts] or [Fraction(4)]
    top = max([p.bar for p in placed] + [0]) + 2
    lens += [lens[-1]] * max(0, top - len(lens))
    starts = [Fraction(0)]
    for L in lens:
        starts.append(starts[-1] + L)
    return starts


def tidy_lengths(placed: list, bar_ts: list[str], tol: float = TIDY_TOL) -> int:
    """Written lengths that run at most ``tol`` quarters past the next onset of their staff stop
    there. The labeller predicts each note's written length on its own, from a performed duration
    that the pedal and legato playing lengthen; an end a little past the next note puts the two in
    separate voices. A note held over moving notes (its end far past the next onset) is left alone,
    and so are gaps: closing them (a note ending a 16th early meets the next) removes many short
    rests but disagrees with the scores, which write them (clean ASAP test windows, true grid:
    duration accuracy 0.849 as predicted, 0.827 with gaps up to a 16th closed, 0.853 with overlaps up
    to a 32nd trimmed, 0.848 up to a 16th). Changes ``placed`` in place; returns how many changed."""
    from bisect import bisect_right
    from fractions import Fraction
    starts = _bar_starts(placed, bar_ts)
    at = lambda p: starts[p.bar] + Fraction(p.pos)  # noqa: E731
    onsets = {}
    for p in placed:
        if not p.grace:
            onsets.setdefault(p.staff, set()).add(at(p))
    onsets = {s: sorted(v) for s, v in onsets.items()}
    tol, changed = Fraction(tol).limit_denominator(1000), 0
    for p in placed:
        if p.grace or not p.total:
            continue
        q = at(p)
        on = onsets[p.staff]
        k = bisect_right(on, q)
        if k >= len(on):
            continue
        over = q + Fraction(p.total) - on[k]
        if 0 < over <= tol:
            p.total = on[k] - q
            changed += 1
    return changed


def attach_graces(placed: list, bar_ts: list[str]) -> int:
    """The labeller writes a grace note where it was played: just before its main note (1/8 quarter
    before it in a demo piece, 6 of its 7 grace notes), sometimes across the bar line. A score
    writes it at its main note's position, in its voice; elsewhere it has no main note and viewers
    misread the bar. Each grace note joins the first main note of its staff starting at or after it
    within GRACE_REACH quarters (that note's bar and position). One with no such note was a short
    note the labeller called a grace note: it is written as a note, lasting until the next note of
    its staff (at most GRACE_AS_NOTE). Changes ``placed`` in place; returns how many moved or became
    notes."""
    from fractions import Fraction
    if not any(p.grace for p in placed):
        return 0
    starts = _bar_starts(placed, bar_ts)
    at = lambda p: starts[p.bar] + Fraction(p.pos)  # noqa: E731
    mains = {}
    for p in placed:
        if not p.grace:
            mains.setdefault(p.staff, []).append(p)
    for v in mains.values():
        v.sort(key=at)
    reach, short, changed = Fraction(GRACE_REACH), Fraction(GRACE_AS_NOTE), 0
    for p in placed:
        if not p.grace:
            continue
        q = at(p)
        nxt = next((m for m in mains.get(p.staff, []) if at(m) >= q), None)
        if nxt is not None and at(nxt) == q:
            continue
        if nxt is not None and at(nxt) - q <= reach:
            p.bar, p.pos = nxt.bar, nxt.pos
        else:
            gap = at(nxt) - q if nxt is not None else short
            p.grace, p.total = False, min(gap, short)
        changed += 1
    return changed


def follow_map(grid: dict, placed: list) -> dict | None:
    """Where the music is in the score at each moment of the performance, for the demo's playback
    bar: {"t": the grid's beat times (s, the recording's / transcription's clock), "q": the score
    position there, in quarters from the first written note}. The first note anchors it because a
    viewer counts its time from the score's first written beat (a pickup bar starts there), not from
    the grid's first bar line. None without a grid or notes."""
    t, q, bar_q = (np.asarray(grid.get(k, []), np.float64) for k in ("t", "q", "bar_q"))
    if len(t) < 2 or not len(placed) or not len(bar_q):
        return None
    first = min(float(bar_q[min(p.bar, len(bar_q) - 1)]) + float(p.pos) for p in placed)
    order = np.argsort(t, kind="stable")
    return {"t": [round(float(x), 3) for x in t[order]], "q": [round(float(x) - first, 4) for x in q[order]]}


class V2Pipeline(_FrontEnd):
    """v2 (docs/experiments.md, "v2: grid first, then label every note"): the piece's beats (Beat This! on the
    audio; a MIDI input rendered to audio first, src/synth.py), the meter model's grid of the whole
    piece over them (src/meter.py: bars, time signatures; its tuned decoder, decode.json) and a key
    signature from the notes, then the note labeller on every window cut as the test sets are,
    each in its stretch of the grid (src/notelabels.py window_frame), and all the labelled notes
    rendered as ONE score in the grid's bars: no window guesses where it starts, nothing is
    stitched.

        pipe = V2Pipeline("outputs/runs/nl_base/best.pt", "outputs/runs/meter_v3/best.pt")
        res = pipe.run("recording.mp3", "outputs/demo/x")     # or a .mid
    """

    def __init__(self, labeller_ckpt: str | Path, meter_ckpt: str | Path, vocab: str | Path | None = None,
                 device: str = "auto", window_notes: int = 128, delete_above: float = DELETE_ABOVE,
                 tidy: bool = False, glissando: bool = False, timing_check: bool = False,
                 readable: bool = False, fix_jumps: bool = False, timing_beats: bool = False,
                 beat_decode: bool = False):
        """``glissando``: find glissandos in the notes (src/glissando.py) and write each as its first
        and last note joined by a "gliss." line, not as the run of notes. ``timing_check``: a note
        written on a triplet position but played on a straight eighth or sixteenth is moved there
        (src/timing.py). ``readable``: rests of a 16th or less closed, overlaps of a 16th or less
        trimmed, stems by voice (src/readable.py). ``fix_jumps``: a bar the meter model's decoder
        stretched with a jump is shortened instead (src/readable.py fix_jumps). ``timing_beats``: a
        beat whose timing fits straight sixteenths better than the labeller's triplets is written
        straight (src/timing.py check_beats). ``beat_decode``: the labeller's splits of the quarter
        decoded a beat at a time, fused with the timing (src/beat_decode.py), with the parameters
        tuned for the checkpoint (beat_decode.json next to it; without one, each note's likeliest
        split as before, and the stats say so). All off by default:
        the scoring script measures each as its own system."""
        from .labeller import load_labeller
        from .meter import DecodeConfig, load_meter
        from .notelabels import LabelSpace
        labeller_ckpt, meter_ckpt = Path(labeller_ckpt), Path(meter_ckpt)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.amp = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.labeller = load_labeller(labeller_ckpt, self.device)
        self.cfg = self.labeller.cfg
        self.meter = load_meter(meter_ckpt, self.device)   # an ensemble run: its members and views (src/meter.py)
        dfile = meter_ckpt.parent / "decode.json"
        self.decode_cfg = (DecodeConfig.from_dict(json.loads(dfile.read_text())["decode"]) if dfile.exists()
                           else DecodeConfig())
        vocab_path = ScorePipeline._find(vocab, labeller_ckpt, "vocab.json")
        self.vocab = Vocab.load(str(vocab_path))
        self.space = LabelSpace(self.vocab.itos)
        if (self.space.n_sub, self.space.n_dur) != (self.cfg.n_sub, self.cfg.n_dur):
            raise ValueError(f"{vocab_path} gives {self.space.n_sub} sub-positions / {self.space.n_dur} lengths but "
                             f"the labeller has {self.cfg.n_sub} / {self.cfg.n_dur}: put the vocab.json of the data it "
                             f"trained on next to it ({Path(labeller_ckpt).parent / 'vocab.json'}; on PACE: "
                             "data/processed/vocab.json), or give --vocab")
        self.scale = (np.array([1, 2, 3, 4]), np.asarray(self.cfg.feat_mean, np.float32),
                      np.asarray(self.cfg.feat_std, np.float32))
        self.window_notes = window_notes
        n_lab = sum(p.numel() for p in self.labeller.parameters())
        n_met = sum(p.numel() for p in self.meter.parameters())
        self.delete_above = float(delete_above)
        self.tidy = bool(tidy)
        self.glissando = bool(glissando)
        self.timing_check = bool(timing_check)
        self.readable = bool(readable)
        self.timing_beats = bool(timing_beats)
        self.fix_jumps = bool(fix_jumps)
        from .beat_decode import BeatDecode
        self.beat_params = BeatDecode.load(labeller_ckpt.parent)   # None: not tuned for this labeller
        self.beat_decode = bool(beat_decode)
        self.info = {"system": "v2", "labeller": str(labeller_ckpt), "meter": str(meter_ckpt),
                     "params_M": round((n_lab + n_met) / 1e6, 1), "device": str(self.device),
                     "decode": self.decode_cfg.to_dict()}

    # ------------------------------------------------------------------ steps
    def grid(self, notes: np.ndarray, beats: tuple, fix_jumps: bool | None = None) -> dict:
        """The meter model's grid of the whole performance on these beats, with its key signature
        (``fix_jumps``: the pipeline's setting if None; the count in "jumps_fixed")."""
        from .meter import cover_notes, predict_grid, refine_grid, uses_aux
        from .notelabels import estimate_key
        # a meter model trained with a second tracker (train_meter.py --aux pm2s) reads PM2S's beats of the notes too
        aux = self.pm2s_beats(notes) if uses_aux(self.meter.cfg) else None
        g = predict_grid(self.meter, notes, np.asarray(beats[0], np.float64), np.asarray(beats[1], bool),
                         self.device, self.decode_cfg, aux=aux)
        g = refine_grid(g, notes[:, ONSET], self.decode_cfg)  # beat times onto the notes (off unless decode.json says)
        g = cover_notes(g, float(notes[:, ONSET].min()))  # a pickup before the first tracked beat's bar
        g["jumps_fixed"] = 0
        if self.fix_jumps if fix_jumps is None else fix_jumps:
            from .readable import fix_jumps as fix
            g["jumps_fixed"] = fix(g)
        g["key"] = estimate_key(notes)
        return g

    def items(self, windows: list[dict], grid: dict) -> tuple[list[dict], list]:
        """The labeller's input for every window (the layout of src/dataset.py with notelabels and
        context notes: note features, standardised, then where the grid puts each note), and each
        window's frame in the grid."""
        from .dataset import ROLE_CARRIED, ROLE_WINDOW, assembly
        from .notelabels import grid_features, window_frame
        items, frames = [], []
        cols, mean, std = self.scale
        context = bool(int(self.cfg.context_before) or int(self.cfg.context_after))
        for w in windows:
            if context:  # the layout of src/dataset.py with context notes (role columns)
                notes, roles, order = assembly(w["notes"], w["before"], w["after"])
                f = note_features(notes, w["pedal"], roles)
                if "idx" in w:
                    w["src"] = np.concatenate([w["idx"], w["idx_before"], w["idx_after"]])[order]
            else:  # and without
                notes = w["notes"]
                roles = np.where(notes[:, 1] < 0, ROLE_CARRIED, ROLE_WINDOW).astype(np.int8)
                f = note_features(notes, w["pedal"])
                if "idx" in w:
                    w["src"] = w["idx"]
            f[:, cols] = (f[:, cols] - mean) / std
            g_t, g_q, win = window_frame(grid, w["t0"], w["notes"][:, 1], int(grid.get("key", 0)))
            gf, bar_est, pos_est = grid_features(notes[:, 1], g_t, g_q, win)
            f = np.concatenate([f, gf], axis=1)[: self.cfg.max_notes]
            n = len(f)
            items.append({"notes": torch.from_numpy(f.astype(np.float32)), "roles": torch.from_numpy(roles[:n]),
                          "target": torch.zeros(1, dtype=torch.long), "prompt_len": 0,
                          "bar_est": torch.from_numpy(bar_est[:n].astype(np.int64)),
                          "pos_est": torch.from_numpy(pos_est[:n].astype(np.float32))})
            frames.append(win)
        return items, frames

    def label(self, notes: np.ndarray, pedal: np.ndarray, grid: dict,
              progress: Progress = _noop, delete_above: float | None = None,
              tidy: bool | None = None, glissandi: list | None = None,
              drop: np.ndarray | None = None, timing_check: bool | None = None,
              readable: bool | None = None, timing_beats: bool | None = None,
              beat_decode: bool | None = None) -> tuple[list[str], dict]:
        """Every note labelled in its window, all rendered in the grid's bars as one score.
        ``delete_above``: leave a note out only above this p(delete); ``tidy``: written lengths stop
        at the next onset of their staff when they run just past it (tidy_lengths). Defaults: the
        pipeline's. ``glissandi`` (src/glissando.py find_glissandi): those two found among the
        labelled notes (by the performed note each came from), the first written to reach the last
        ("glissandi" in the result, for the export's line). ``drop``: notes not to write (the
        glissandos' middles): the labeller still gets every note, exactly as without them, so the rest
        of the piece is written exactly as it would be, and their labels are left out."""
        delete_above = self.delete_above if delete_above is None else float(delete_above)
        tidy = self.tidy if tidy is None else bool(tidy)
        import sys
        from fractions import Fraction
        from .labeller import predict_placed
        from .notelabels import Window, grid_position, render
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.train import TruncatingCollate
        windows = performance_windows(notes, pedal, self.window_notes, context_before=int(self.cfg.context_before),
                                      context_after=int(self.cfg.context_after))
        progress("score", 0.0, f"labelling {len(windows)} windows")
        items, frames = self.items(windows, grid)
        coll = TruncatingCollate(self.cfg.max_notes, 10 ** 9)
        bd = self.beat_decode if beat_decode is None else bool(beat_decode)
        bparams = self.beat_params if bd else None   # None also when not tuned
        placed = predict_placed(self.labeller, items, coll, self.device, self.amp, self.space, carried=False,
                                delete_above=delete_above, beat_decode=bparams)
        staves = piece_staves([p.staff for pl in placed for p in pl])
        folded = sum(p.staff >= staves for pl in placed for p in pl)
        if folded:  # a stray third / fourth staff: those notes on their likeliest of the staves kept
            placed = predict_placed(self.labeller, items, coll, self.device, self.amp, self.space, carried=False,
                                    staves=staves, delete_above=delete_above, beat_decode=bparams)
        notes_out = []
        skip = np.asarray(drop, bool) if drop is not None else np.zeros(len(notes), bool)
        for w, win, pl in zip(windows, frames, placed):
            for p in pl:
                p.bar += win.first_bar
                p.src = int(w["src"][p.src]) if p.src >= 0 and "src" in w else -1   # the performance's note
                if p.src >= 0 and skip[p.src]:

                    continue
                notes_out.append(p)
        timing = self.timing_check if timing_check is None else bool(timing_check)
        fixed = 0
        if timing:   # triplets the timing contradicts, before graces join their notes and lengths are tidied
            from .timing import check_timing
            fixed = check_timing(notes_out, notes[:, ONSET], grid, list(grid["bars"]))
        tb = self.timing_beats if timing_beats is None else bool(timing_beats)
        beat_fixed = 0
        if tb:       # whole beats the timing says are straight sixteenths, not the labeller's triplets
            from .timing import check_beats
            beat_fixed = check_beats(notes_out, notes[:, ONSET], grid, list(grid["bars"]))
        graces = attach_graces(notes_out, list(grid["bars"]))
        tidied = tidy_lengths(notes_out, list(grid["bars"])) if tidy else 0
        rd = self.readable if readable is None else bool(readable)
        closed = trimmed = 0
        if rd:   # short rests closed, small overlaps trimmed (before a glissando's start is set to reach its end)
            from .readable import readable_lengths
            closed, trimmed = readable_lengths(notes_out, list(grid["bars"]))
        key = int(grid.get("key", 0))
        bars = [(ts, key) for ts in grid["bars"]] or [("4/4", key)]
        opens = True
        if len(grid["t"]) >= 2:
            # a pickup: the first bar is written from the first note on
            q0, q1 = grid_position(np.array([float(notes[:, ONSET].min()), float(notes[:, OFFSET].max())]),
                                   grid["t"], grid["q"])[0]
            opens = float(q0) < 1 / 24
            # the score ends with the bar the last note ends in (beats past the music add no bars)
            last = int(np.searchsorted(np.asarray(grid["bar_q"], np.float64), q1 - 1e-6, side="right")) - 1
            last = max([last] + [p.bar for p in notes_out])
            bars = bars[: max(last, 0) + 1]
        gl = []
        if glissandi:
            from .glissando import attach
            gl = attach(glissandi, notes_out, grid, [ts for ts, _ in bars])
        piece = Window(bars=bars, opens_with_bar=opens, start_pos=Fraction(0) if opens else None)
        tokens = render(piece, notes_out)
        own = int(sum(int((it["roles"] == 0).sum()) for it in items)) - int(skip.sum())
        kept = len(notes_out) - sum(g.get("written", 0) for g in gl)
        return tokens, {"windows": len(windows), "notes_kept": kept, "notes_deleted": own - kept,
                        "bar_ts": [ts for ts, _ in bars], "staves": max([2] + [p.staff + 1 for p in notes_out]),
                        "staff_folded": int(folded), "graces_attached": graces, "delete_above": delete_above,
                        "tidy": tidy, "lengths_tidied": tidied, "glissandi": gl,
                        "timing_check": timing, "timing_fixed": fixed,
                        "timing_beats": tb, "timing_beats_fixed": beat_fixed,
                        "beat_decode": ((self.beat_params.to_dict() if self.beat_params else "not tuned") if bd
                                        else False),
                        "readable": rd, "rests_closed": closed, "overlaps_trimmed": trimmed,
                        "follow": follow_map(grid, notes_out)}

    # ------------------------------------------------------------------ all of it
    def run(self, src: str | Path, out_dir: str | Path, progress: Progress = _noop, beats: tuple | None = None,
            beat_checkpoint: str = "final0", delete_above: float | None = None, tidy: bool | None = None,
            beat_source: str = "beat_this", glissando: bool | None = None, timing_check: bool | None = None,
            readable: bool | None = None, fix_jumps: bool | None = None, timing_beats: bool | None = None,
            beat_decode: bool | None = None, **_) -> Result:
        """``beats``: (times, down) in the input's seconds, given instead of tracking them.
        ``beat_checkpoint``: the Beat This! model that tracks them (scripts/system_scores.py gives
        each ASAP test piece the cross-validation model that never heard it). ``beat_source``:
        beat_this (the recording; a MIDI input rendered to audio first) or pm2s (PM2S on the notes:
        the MIDI, or the transcription of a recording). ``glissando``, ``timing_check``, ``timing_beats``, ``beat_decode``, ``readable``,
        ``fix_jumps``: the pipeline's settings if None."""
        if beat_source not in BEAT_SOURCES:
            raise ValueError(f"beat_source must be one of {BEAT_SOURCES}, not {beat_source!r}")
        from collections import Counter
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
        t = time.perf_counter()
        if beats is not None:
            beats = (np.asarray(beats[0], np.float64), np.asarray(beats[1], bool))
            beat_info = f"given: {len(beats[0])} beats, {int(beats[1].sum())} downbeats"
        elif beat_source == "pm2s":
            progress("beats", 0.0, "tracking beats from the notes with PM2S")
            beats = self.pm2s_beats(notes)
            beat_info = f"PM2S on the notes: {len(beats[0])} beats, {int(beats[1].sum())} downbeats"
        else:
            if ext in AUDIO_EXT:
                progress("beats", 0.0, "tracking beats with Beat This!")
                beats = self.track_beats(src, beat_checkpoint)
                beat_info = "Beat This!"
            else:
                progress("beats", 0.0, "tracking beats with Beat This! (MIDI rendered to audio)")
                beats = self.track_midi_beats(notes, beat_checkpoint)
                beat_info = "Beat This! on the MIDI rendered to audio"
            if beats is not None:
                beat_info += f" ({beat_checkpoint}): {len(beats[0])} beats, {int(beats[1].sum())} downbeats"
        timing["beats_s"] = round(time.perf_counter() - t, 1)
        res = Result([], files)
        if beats is not None:  # only the beats of the music (a recording can hold more than this piece)
            span = (beats[0] >= float(notes[:, ONSET].min()) - 1.0) & (beats[0] <= float(notes[:, OFFSET].max()) + 1.0)
            beats = (beats[0][span], beats[1][span])
        if beats is None or len(beats[0]) < 2:
            res.errors.append("v2 needs the beats: " + ("Beat This! is not installed (pip install beat-this "
                                                        "rotary-embedding-torch einops)" if beats is None
                                                        else "fewer than 2 found"))
            res.stats = {**self.info, "beats": "none", "notes": int(len(notes)), "pedal_presses": int(len(pedal)),
                         "duration_s": round(float(notes[:, OFFSET].max() - notes[:, ONSET].min()), 1), **timing}
            return res
        progress("grid", 0.0, "the metrical grid of the piece")
        t = time.perf_counter()
        grid = self.grid(notes, beats, fix_jumps=fix_jumps)
        timing["grid_s"] = round(time.perf_counter() - t, 1)
        t = time.perf_counter()
        found = []
        if self.glissando if glissando is None else glissando:
            from .glissando import find_glissandi, inner_mask
            found = find_glissandi(notes)
        drop = ~inner_mask(len(notes), found) if found else None
        tokens, counts = self.label(notes, pedal, grid, progress, delete_above=delete_above, tidy=tidy,
                                    glissandi=found, drop=drop, timing_check=timing_check, readable=readable,
                                    timing_beats=timing_beats, beat_decode=beat_decode)
        gl = counts.pop("glissandi")
        timing["score_s"] = round(time.perf_counter() - t, 1)
        res.tokens = tokens
        (out / "tokens.txt").write_text(" ".join(tokens))
        files["tokens"] = out / "tokens.txt"
        progress("engrave", 0.0, "writing MusicXML")
        export = engrave(tokens, out, res, float(notes[:, OFFSET].max() - notes[:, ONSET].min()), glissandi=gl,
                         stems=counts["readable"])
        written = counts.pop("bar_ts")  # the score's bars (the grid's, up to the bar the music ends in)
        ts = Counter(written)
        dur = float(notes[:, OFFSET].max() - notes[:, ONSET].min())
        res.stats = {**self.info, "duration_s": round(dur, 1), "notes": int(len(notes)),
                     "pedal_presses": int(len(pedal)), "beats": beat_info, "bars": len(written),
                     "grid_bars": len(grid["bars"]),
                     "level": float(grid.get("level", 1.0)),   # the note-value level choose_level picked
                     "time_signature": ts.most_common(1)[0][0] if ts else None,
                     "time_signatures": dict(ts.most_common()), "key_signature": int(grid.get("key", 0)),
                     **counts, "tokens": len(tokens), "export": export,
                     "glissando": bool(self.glissando if glissando is None else glissando),
                     "jumps_fixed": int(grid.get("jumps_fixed", 0)),
                     "glissandi_found": len(found),
                     "glissandi": [{k: v for k, v in g.items() if k not in ("start", "end", "inner")}
                                   | {"bar": int(g["start"][0]) + 1} for g in gl],
                     "playback_qpm": export.get("playback_qpm"), **timing}
        (out / "result.json").write_text(json.dumps({"stats": res.stats, "errors": res.errors,
                                                     "files": {k: str(v) for k, v in files.items()}}, indent=1))
        return res
