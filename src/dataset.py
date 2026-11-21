"""PyTorch dataset over preprocessed segments, with on-the-fly transcription-noise injection.

    ds = ScoreSegmentDataset("data/processed/train", "data/processed/vocab.json",
                             noise=NoiseModel.from_json("configs/noise_stats.json"),
                             augment=AugmentConfig())          # train only; omit for val/test
    loader = DataLoader(ds, batch_size=32, sampler=ds.sampler(), collate_fn=collate, num_workers=4)
    for epoch in ...:
        ds.set_epoch(epoch)   # new noise each epoch (with num_workers>0 use persistent_workers=False)

Each item:
    notes       float32 [N, F]  per-note features (see ``note_features``), sorted by onset; F = 8, or
                                10 with context notes
    target      int64   [T]     <bos> [prompt <sep>] score tokens <eos>
    prompt_len  int             tokens before the first one to predict (<bos> + prompt + <sep>)
    roles       int8    [N]     per note: 0 in the window, 1 carried over into it, 2 context before,
                                3 context after (``ROLE_*``)
    downbeat, phase  int8 [N]   with ``metrical=True``: each note's metrical labels for the auxiliary
                                target (src/metrical.py), -1 where unknown (context, carried, ...)

Beats (``beats`` = "gt" / "tracked" / "none"): 7 more features per note (src/beats.py
``beat_features``) from the performance's beat list in ``<split>/beats_<source>.npz``: ground
truth (scripts/build_beats.py) or a beat tracker's output (scripts/track_beats.py); "none" gives
the all-zero "no beats" features (MIDI input without a tracker). ``beat_noise`` corrupts each
window's beats like a real tracker (src.beats.BeatNoise; the simulated-tracker test condition uses
it too, with a fixed seed), and when training ``beat_dropout`` of the windows get no beats at all,
so the model also works without them. Beat times follow the tempo augmentation.

Context notes (``context_before`` / ``context_after`` > 0): the encoder also sees up to that many
performed notes just before and just after the window, flagged by two extra features, so it can
read the rhythm and meter around the window instead of guessing where the bar starts from ~128
notes alone. The target is unchanged: the model writes the window's notes only. The notes come from
a pool of every stored window of the same performance in absolute time (no data rebuild needed),
get the same augmentation as the window and their own synthetic noise, and are assembled by
``assemble_notes`` exactly as src/pipeline.py does at inference.

The prompt is the score just before the window (see segment.segment_windows); at inference it is
the tail of the previous window's output, from its last ``bar`` token (pedal tokens removed).
Train only on target tokens: ``collate`` returns ``loss_mask`` for that.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .beats import N_BEAT_FEATURES, BeatNoise, beat_features, beats_path, load_beats, window_beats
from .preprocessing.noise import NoiseModel
from .preprocessing.store import SegmentStore
from .preprocessing.vocab import BOS, EOS, ORNAMENT_PREFIXES, SEP, Vocab

N_FEATURES = 8           # without context notes
N_FEATURES_CONTEXT = 10  # with context notes: + "context before", "context after" flags
BEAT_SOURCES = ("gt", "tracked", "none")


def n_features(context: bool, beats: bool) -> int:
    """Per-note input width: 8, 10 with context notes, + 7 with beats (src/beats.py)."""
    return (N_FEATURES_CONTEXT if context else N_FEATURES) + (N_BEAT_FEATURES if beats else 0)
SCALED = [1, 2, 3, 4]  # onset, inter-onset gap, log duration, velocity: standardized with train stats
FEATURE_NAMES = ["pitch", "onset", "ioi", "log_dur", "velocity", "ped_at_onset", "ped_at_offset", "carried",
                 "context_before", "context_after"]
ROLE_WINDOW, ROLE_CARRIED, ROLE_BEFORE, ROLE_AFTER = 0, 1, 2, 3


def note_features(notes: np.ndarray, pedal: np.ndarray, roles: np.ndarray | None = None) -> np.ndarray:
    """[N,4] (pitch, onset, dur, vel) -> [N,8], or [N,10] if ``roles`` is given:
    pitch (int, keep as index for an embedding), onset (s), inter-onset delta (s), log duration,
    velocity / 127, sustain-pedal-down at onset (0/1), sustain-pedal-down at offset (0/1),
    carried over (0/1: struck before the segment and still held into it), and with roles: context
    before / context after (0/1). Without roles, "carried" is onset < 0 (no context notes)."""
    on, dur = notes[:, 1], notes[:, 2]
    delta = np.diff(on, prepend=on[:1]) if len(on) else on

    def ped_at(t):
        if len(pedal) == 0:
            return np.zeros_like(t)
        return ((t[:, None] >= pedal[None, :, 0]) & (t[:, None] < pedal[None, :, 1])).any(1).astype(np.float32)

    cols = [notes[:, 0], on, delta, np.log(np.maximum(dur, 1e-3)), notes[:, 3] / 127.0, ped_at(on), ped_at(on + dur)]
    if roles is None:
        cols.append((on < 0).astype(np.float32))
    else:
        cols += [(roles == r).astype(np.float32) for r in (ROLE_CARRIED, ROLE_BEFORE, ROLE_AFTER)]
    return np.stack(cols, axis=1).astype(np.float32)


def assembly(window: np.ndarray, before: np.ndarray, after: np.ndarray):
    """``assemble_notes`` plus the permutation it applied to [window, before, after] (concatenated),
    so per-note arrays can follow the notes."""
    parts = [window, before, after]
    roles = np.concatenate([np.where(window[:, 1] < 0, ROLE_CARRIED, ROLE_WINDOW),
                            np.full(len(before), ROLE_BEFORE), np.full(len(after), ROLE_AFTER)]).astype(np.int8)
    notes = np.concatenate([np.asarray(x, np.float32).reshape(-1, 4) for x in parts])
    order = np.lexsort((notes[:, 0], notes[:, 1]))
    return notes[order], roles[order], order


def assemble_notes(window: np.ndarray, before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Window notes (carried-over ones have onset < 0) plus context notes, all [K,4] in the
    window's time frame -> (notes sorted by onset then pitch, roles). Shared by the Dataset and the
    inference pipeline so both build identical inputs."""
    notes, roles, _ = assembly(window, before, after)
    return notes, roles


def compute_feature_stats(split_dir: str, out_path: str, n: int = 5000, seed: int = 0) -> dict:
    """Mean / std of the continuous note features over (up to ``n``) training segments, after the
    carried-over filtering the Dataset applies. Saved as JSON and applied by ScoreSegmentDataset."""
    store = SegmentStore(split_dir)
    idx = np.random.default_rng(seed).choice(len(store), size=min(n, len(store)), replace=False)
    feats = []
    for i in idx:
        notes, pedal, _ = store.get(int(i))
        notes = drop_released_carried(notes, store.meta[int(i)].get("carry_min_sec"))
        feats.append(note_features(notes, pedal)[:, SCALED])
    f = np.concatenate(feats).astype(np.float64)
    stats = {"columns": [FEATURE_NAMES[c] for c in SCALED], "index": SCALED,
             "mean": f.mean(0).tolist(), "std": np.maximum(f.std(0), 1e-6).tolist(), "n_notes": len(f)}
    Path(out_path).write_text(json.dumps(stats, indent=1))
    return stats


def released_mask(notes: np.ndarray, min_end: float | None) -> np.ndarray:
    """Which notes ``drop_released_carried`` keeps."""
    if min_end is None or not len(notes):
        return np.ones(len(notes), bool)
    return (notes[:, 1] >= 0) | (notes[:, 1] + notes[:, 2] >= min_end)


def drop_released_carried(notes: np.ndarray, min_end: float | None) -> np.ndarray:
    """Carried-over candidates (onset < 0) are kept only if the key is still down at ``min_end``
    (segment seconds). Applied after noise, so pedal lengthening in the noise model can turn a
    pedal-sustained note into a carried one, as in real transcriptions."""
    return notes[released_mask(notes, min_end)]


@dataclass
class AugmentConfig:
    """Training-time augmentation (applied to the clean segment, before noise injection)."""
    transpose: tuple = (-6, 6)        # semitones, inclusive; only shifts keeping all notes on the keyboard
    tempo: tuple = (0.85, 1.18)       # time-stretch factor for onsets / durations / pedal
    velocity: tuple = (0.8, 1.2)      # velocity scale
    prob: float = 0.9                 # probability a segment is augmented at all


class _Transposer:
    """Shifts p_* target tokens by k semitones and re-spells key_* (circle of fifths)."""

    def __init__(self, vocab: Vocab):
        n = len(vocab)
        self.pitch = np.full(n, -1)
        self.key = np.full(n, 99)
        for i, t in enumerate(vocab.itos):
            if t.startswith("p_"):
                self.pitch[i] = int(t[2:])
            elif t.startswith("key_"):
                self.key[i] = int(t[4:])
        self.p2id = {int(t[2:]): i for i, t in enumerate(vocab.itos) if t.startswith("p_")}
        self.k2id = {int(t[4:]): i for i, t in enumerate(vocab.itos) if t.startswith("key_")}

    def pitch_range(self, ids):
        p = self.pitch[ids]
        p = p[p >= 0]
        return (int(p.min()), int(p.max())) if len(p) else (60, 60)

    def __call__(self, ids: np.ndarray, k: int) -> np.ndarray:
        out = ids.copy()
        pm = self.pitch[ids] >= 0
        out[pm] = [self.p2id[p + k] for p in self.pitch[ids[pm]]]
        km = self.key[ids] != 99
        # +1 semitone = +7 sharps (mod 12), kept in -6..5 (enharmonic choice)
        out[km] = [self.k2id[(kk + 7 * k + 6) % 12 - 6] for kk in self.key[ids[km]]]
        return out


@dataclass
class ContextNoise:
    """Training-time corruption of the context (prompt), so the model learns not to trust it
    blindly. At inference the context is the model's own output for the previous window, and its
    mistakes carry over: in chained evaluation every model scored below decoding with no context
    at all. ``prob`` is the fraction of training windows (with a context) that get corrupted; each
    corrupted context gets one or more of the errors below (at least one)."""
    prob: float = 0.0
    ts: float = 0.25           # wrong time signature (one of the common ones)
    shift: float = 0.35        # every position moved by 1-2 quarter notes (misplaced bar phase)
    drop_leading: float = 0.2  # bar line placed too late: the first positions of the fragment are gone
    drop_notes: float = 0.35   # notes deleted, each with probability ``note_p``
    durations: float = 0.3     # written durations replaced, each with probability ``note_p``
    staff: float = 0.15        # staff changes flipped (0 <-> 1), each with probability 1/2
    note_p: float = 0.2


_NOTE_ATTR = ("d_", "tie", "arp", "orn_", "grace")


class _ContextCorrupter:
    """Applies ``ContextNoise`` to a context given as token ids (``bar [ts] [key]`` then position
    groups ``pos_k [sub_x] ([staff_s] p_n d_x|grace [tie] [arp] [orn_*])...``, no pedal). Every
    output token is in the vocabulary; empty position groups are removed, as in
    ``segment.prompt_from_output``, so corrupted contexts look like real model output."""

    def __init__(self, vocab: Vocab, cfg: ContextNoise):
        self.cfg, self.itos, self.stoi = cfg, vocab.itos, vocab.stoi
        counts = vocab.counts or {}

        def common(prefix, n):
            toks = [t for t in vocab.itos if t.startswith(prefix)]
            return sorted(toks, key=lambda t: -counts.get(t, 0))[:n]
        self.ts_choices = common("ts_", 8)       # 4/4 6/8 3/4 2/4 12/8 2/2 3/8 4/2 on our data
        self.dur_choices = common("d_", 12)
        self.max_pos = max(int(t[4:]) for t in vocab.itos if t.startswith("pos_"))

    def _pick(self, rng, choices, avoid):
        c = [x for x in choices if x != avoid]
        return c[int(rng.integers(len(c)))] if c else avoid

    def __call__(self, ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        for _ in range(4):  # retry the rare draw that leaves the context unchanged
            out = self._corrupt(ids, rng)
            if not np.array_equal(out, ids):
                return out
        return out

    def _bar_positions(self, head: list[str]) -> int:
        """Highest whole-quarter position inside the bar (from the context's time signature)."""
        ts = next((t[3:] for t in head if t.startswith("ts_")), "4/4")
        try:
            n, d = (int(x) for x in ts.split("/"))
            return max(0, min(self.max_pos, math.ceil(n * 4 / d) - 1))
        except (ValueError, ZeroDivisionError):
            return 3

    def _corrupt(self, ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.cfg
        toks = [self.itos[i] for i in ids]
        k = next((j for j, t in enumerate(toks) if t.startswith("pos_")), len(toks))
        head, groups = toks[:k], []
        for t in toks[k:]:
            if t.startswith("pos_") or not groups:
                groups.append([])
            groups[-1].append(t)
        ops = [op for op, p in (("ts", c.ts), ("shift", c.shift), ("drop_leading", c.drop_leading),
                                ("drop_notes", c.drop_notes), ("durations", c.durations),
                                ("staff", c.staff)) if rng.random() < p]
        if not ops:
            ops = ["shift"]
        # positions stay inside the (true) bar, or within the fragment's own range for the few
        # scores whose bars hold more than their time signature (cadenzas)
        top = max([self._bar_positions(head)] + [int(t[4:]) for t in toks[k:] if t.startswith("pos_")])
        if "ts" in ops:
            head = [self._pick(rng, self.ts_choices, t) if t.startswith("ts_") else t for t in head]
        if "drop_leading" in ops and len(groups) >= 2:
            groups = groups[int(rng.integers(1, len(groups) // 2 + 1)):]
        if "shift" in ops:
            pos = [int(t[4:]) for g in groups for t in g if t.startswith("pos_")]
            d = int(rng.choice([-2, -1, 1, 2]))
            if pos and (min(pos) + d < 0 or max(pos) + d > top):
                d = -d  # the other direction, so the shift is not clipped away
            groups = [[f"pos_{min(max(int(t[4:]) + d, 0), top)}" if t.startswith("pos_") else t
                       for t in g] for g in groups]
        if "drop_notes" in ops:
            kept = []
            for g in groups:
                out, skip = [], False
                for t in g:
                    if t.startswith("p_"):
                        skip = rng.random() < c.note_p
                    elif not t.startswith(_NOTE_ATTR):
                        skip = False
                    if not skip:
                        out.append(t)
                kept.append(out)
            groups = kept
        if "durations" in ops:
            groups = [[self._pick(rng, self.dur_choices, t) if t.startswith("d_") and rng.random() < c.note_p
                       else t for t in g] for g in groups]
        if "staff" in ops:
            flip = {"staff_0": "staff_1", "staff_1": "staff_0"}
            groups = [[flip[t] if t in flip and rng.random() < 0.5 else t for t in g] for g in groups]
        body = [t for g in groups if any(x.startswith("p_") for x in g) for t in g]
        return np.asarray([self.stoi[t] for t in head + body], dtype=ids.dtype)


class ScoreSegmentDataset(Dataset):
    def __init__(self, split_dir: str, vocab_path: str, noise: NoiseModel | None = None,
                 seed: int = 0, ornament_oversample: float = 3.0, augment: AugmentConfig | None = None,
                 use_prompt: bool = True, prompt_dropout: float = 0.1, feature_stats: str | None = "auto",
                 context_noise: ContextNoise | None = None, context_before: int = 0, context_after: int = 0,
                 metrical: bool = False, beats: str | None = None, beat_noise: BeatNoise | None = None,
                 beat_dropout: float = 0.0, notelabels: bool = False):
        """``prompt_dropout``: chance (when training, i.e. with noise or augmentation) of hiding the
        prompt, so the model also works without context (first window of a piece, or after a bad
        window). ``feature_stats``: JSON from ``compute_feature_stats`` (written by build_dataset.py
        next to vocab.json; "auto" = use it if present, None = leave features unscaled).
        ``context_noise``: corrupt the prompt of this fraction of training windows (see
        ``ContextNoise``); like prompt dropout, only when training. ``context_before`` /
        ``context_after``: context notes the encoder also sees (module docstring); 0 = none.
        ``metrical``: add per-note metrical labels (src/metrical.py), read from the split's
        metrical.npy if scripts/build_metrical.py made it, else aligned on the fly.
        ``beats`` / ``beat_noise`` / ``beat_dropout``: beat features (module docstring); None = off.
        ``notelabels``: the note labeller's inputs and targets (src/notelabels.py, from the split's
        notelabels.npz made by scripts/build_notelabels.py): the grid features of every note are
        appended to its features (the grid scaled with the tempo augmentation), and the item gets
        "labels" [N, TARGET_COLS], "flags" [N, N_FLAGS] (-1 = nothing to learn), "bar_est" and
        "pos_est" (where the grid puts each note), the labels following the notes through the
        synthetic noise (``transfer_rows``)."""
        self.store = SegmentStore(split_dir)
        self.vocab = Vocab.load(vocab_path)
        self.noise = noise
        self.augment = augment
        self._tr = _Transposer(self.vocab) if augment else None
        self.seed = seed
        self.epoch = 0
        self.use_prompt, self.prompt_dropout = use_prompt, prompt_dropout
        self.context_noise = context_noise if context_noise and context_noise.prob > 0 else None
        self._ctx = _ContextCorrupter(self.vocab, context_noise) if self.context_noise else None
        self.context_before, self.context_after = int(context_before), int(context_after)
        self.with_context = self.context_before > 0 or self.context_after > 0
        self._pools: dict = {}  # performance -> (notes, keys, pedal) in absolute time, built lazily
        self._by_perf = None
        if beats is not None and beats not in BEAT_SOURCES:
            raise ValueError(f"beats must be one of {BEAT_SOURCES} or None, not {beats!r}")
        self.beats, self.beat_noise, self.beat_dropout = beats, beat_noise, float(beat_dropout)
        # evaluation: also return each item's raw note onsets (window seconds) / pitches and its beat
        # list ("raw_onsets", "raw_pitches", "beat_t", "beat_d"), e.g. for src/beats.py grid_agreement
        self.keep_raw = False
        self._beats = {}
        if beats in ("gt", "tracked"):
            path = beats_path(split_dir, beats)
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: run scripts/build_beats.py (gt) or "
                                        f"scripts/track_beats.py (tracked) first")
            self._beats = load_beats(path)
        self.notelabels = bool(notelabels)
        self._nl = None
        if self.notelabels:
            from .notelabels import LabelSpace, load_split
            self._nl = load_split(split_dir, len(self.store.notes), len(self.store))
            if self._nl is None:
                raise FileNotFoundError(f"{split_dir}/notelabels.npz missing or out of date: run "
                                        f"scripts/build_notelabels.py first")
            self.label_space = LabelSpace(self.vocab.itos)
        # evaluation with an estimated grid: {window index: (grid_t, grid_q, src.notelabels.Window)}
        # replaces the stored true grid and the target's bars (scripts/eval_labeller.py --grid meter:...)
        self.grid_override: dict | None = None
        self.metrical = bool(metrical)
        self._mfile = None
        self._mcache: dict = {}
        if self.metrical:
            from .metrical import load_split_labels
            self._mfile = load_split_labels(split_dir, len(self.store.notes))
        if feature_stats == "auto":
            auto = Path(vocab_path).with_name("feature_stats.json")
            feature_stats = str(auto) if auto.exists() else None
        self.scale = None
        if feature_stats:
            st = json.loads(Path(feature_stats).read_text())
            self.scale = (np.asarray(st["index"]), np.asarray(st["mean"], dtype=np.float32),
                          np.asarray(st["std"], dtype=np.float32))
        self.bos, self.eos, self.sep = self.vocab.stoi[BOS], self.vocab.stoi[EOS], self.vocab.stoi[SEP]
        rare = np.array([self.vocab.itos[i].startswith(ORNAMENT_PREFIXES[:3]) for i in range(len(self.vocab))])
        idx = self.store.index
        cs = np.concatenate([[0], np.cumsum(rare[np.asarray(self.store.tokens)])])
        has_rare = (cs[idx[:, 5]] - cs[idx[:, 6]]) > 0  # rare tokens in the target (not the prompt)
        self.has_ornament = has_rare
        self.weights = np.where(has_rare, ornament_oversample, 1.0)

    def set_epoch(self, epoch: int):
        """Call each epoch so the injected noise differs between epochs but stays reproducible."""
        self.epoch = epoch

    @staticmethod
    def _keys(pitch: np.ndarray, abs_on: np.ndarray) -> np.ndarray:
        """One integer per performed note: pitch and absolute onset in milliseconds."""
        return pitch.astype(np.int64) * 10_000_000_000 + np.round(abs_on * 1000).astype(np.int64)

    def _pool(self, perf: str):
        """All notes and pedal intervals of one performance, in absolute seconds, from every stored
        window of it (windows overlap during training, so notes are de-duplicated)."""
        if perf not in self._pools:
            if self._by_perf is None:
                self._by_perf = {}
                for j, m in enumerate(self.store.meta):
                    self._by_perf.setdefault(m["performance"], []).append(j)
            notes, peds = [], []
            for j in self._by_perf[perf]:
                n, p, _ = self.store.get(j)
                t0 = float(self.store.meta[j]["perf_t0"])
                a = n.astype(np.float64).copy()
                a[:, 1] += t0
                notes.append(a)
                if len(p):
                    peds.append(p.astype(np.float64) + t0)
            a = np.concatenate(notes)
            keys = self._keys(a[:, 0], a[:, 1])
            keys, first = np.unique(keys, return_index=True)
            a = a[first]
            order = np.argsort(a[:, 1], kind="stable")
            ped = np.concatenate(peds) if peds else np.zeros((0, 2))
            if len(ped):
                _, pf = np.unique(np.round(ped * 1000).astype(np.int64), axis=0, return_index=True)
                ped = ped[np.sort(pf)]
            self._pools[perf] = (a[order], keys[order], ped)
        return self._pools[perf]

    def _context(self, i: int, window: np.ndarray):
        """(before [<=context_before, 4], after [<=context_after, 4], pedal [K,2]) for window ``i``,
        in its time frame (seconds from perf_t0); notes of the window itself are left out."""
        m = self.store.meta[i]
        t0 = float(m["perf_t0"])
        pool, keys, ped = self._pool(m["performance"])
        own = self._keys(window[:, 0], window[:, 1].astype(np.float64) + t0)
        free = ~np.isin(keys, own)
        before = pool[free & (pool[:, 1] < t0)]
        before = before[-self.context_before:] if self.context_before else before[:0]  # ([-0:] = all)
        after = pool[free & (pool[:, 1] >= t0)][:self.context_after]
        lo = float(before[:, 1].min()) if len(before) else t0
        hi = t0 + float(m["window_sec"])
        if len(after):
            hi = max(hi, float((after[:, 1] + after[:, 2]).max()))
        ped = ped[(ped[:, 1] > lo) & (ped[:, 0] < hi)] if len(ped) else ped
        shift = np.array([0, t0, 0, 0])
        return ((before - shift).astype(np.float32), (after - shift).astype(np.float32),
                (ped - t0).astype(np.float32).reshape(-1, 2))

    def _metrical(self, i: int, notes: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """[n, 2] (downbeat, phase) for the stored notes of window ``i``."""
        if self._mfile is not None:
            a0, a1 = self.store.index[i, 0], self.store.index[i, 1]
            return np.array(self._mfile[a0:a1])
        if i not in self._mcache:
            from .metrical import align
            d, ph = align(notes[:, 1], notes[:, 0], self.vocab.decode(tokens.tolist()))
            self._mcache[i] = np.stack([d, ph], axis=1)
        return self._mcache[i]

    def _window_beats(self, i: int, lo: float, hi: float, rng, training: bool):
        """Beat list of window ``i`` in its time frame (before tempo augmentation), from window-frame
        ``lo`` to ``hi`` seconds (its notes and context notes), corrupted / dropped as configured."""
        m = self.store.meta[i]
        t0 = float(m["perf_t0"])
        times, down = self._beats.get(m["performance"], (None, None))
        bt, bd = window_beats(times, down, t0, t0 + lo, t0 + hi)
        brng = np.random.default_rng((self.seed, self.epoch if training else 0, i, 13))  # own stream
        if training and self.beat_dropout > 0 and brng.random() < self.beat_dropout:
            return np.zeros(0), np.zeros(0, bool)
        if self.beat_noise is not None and len(bt):
            bt, bd = self.beat_noise(bt, bd, brng)
        return bt, bd

    def sampler(self) -> WeightedRandomSampler:
        """Oversamples segments containing grace notes / arpeggios / ornaments."""
        return WeightedRandomSampler(torch.as_tensor(self.weights, dtype=torch.double),
                                     num_samples=len(self), replacement=True)

    def __len__(self):
        return len(self.store)

    def _augment(self, notes, pedal, tokens, prompt, rng):
        """Returns (notes, pedal, tokens, prompt, tempo factor). Transposition shifts input pitches,
        target AND prompt tokens together; the tempo factor also rescales time thresholds."""
        a = self.augment
        if rng.random() >= a.prob:
            return notes, pedal, tokens, prompt, 1.0
        notes, pedal = notes.copy(), pedal.copy()
        lo_t, hi_t = self._tr.pitch_range(np.concatenate([tokens, prompt]))
        lo = min(int(notes[:, 0].min()), lo_t) if len(notes) else lo_t
        hi = max(int(notes[:, 0].max()), hi_t) if len(notes) else hi_t
        k_lo, k_hi = max(a.transpose[0], 21 - lo), min(a.transpose[1], 108 - hi)
        if k_lo <= k_hi:
            k = int(rng.integers(k_lo, k_hi + 1))
            if k:
                notes[:, 0] += k
                tokens = self._tr(tokens, k)
                prompt = self._tr(prompt, k) if len(prompt) else prompt
        f = float(rng.uniform(*a.tempo))
        notes[:, 1:3] *= f
        pedal *= f
        notes[:, 3] = np.clip(np.round(notes[:, 3] * rng.uniform(*a.velocity)), 1, 127)
        return notes, pedal, tokens, prompt, f

    def __getitem__(self, i):
        notes, pedal, tokens = self.store.get(i)
        prompt = self.store.get_prompt(i) if self.use_prompt else np.zeros(0, dtype=np.int64)
        rng = np.random.default_rng((self.seed, self.epoch, i))
        tempo = 1.0
        lab = self._metrical(i, notes, tokens) if self.metrical else None  # rows follow the window notes
        nl = None
        if self.notelabels:  # label rows of the window's stored notes
            a0, a1 = self.store.index[i, 0], self.store.index[i, 1]
            nl = np.array(self._nl["labels"][a0:a1])
        if self.with_context:
            before, after, cped = self._context(i, notes)
            nb, nw, npw = len(before), len(notes), len(pedal)
            notes = np.concatenate([before, notes, after])  # augmented together: same key and tempo
            pedal = np.concatenate([pedal.reshape(-1, 2), cped])
        training = self.noise is not None or self.augment is not None
        if self.beats in ("gt", "tracked"):  # the span of every note the encoder will see
            span = notes[:, 1] if len(notes) else np.zeros(1)
            ends = notes[:, 1] + notes[:, 2] if len(notes) else np.zeros(1)
            bt, bd = self._window_beats(i, float(span.min()), float(max(ends.max(), self.store.meta[i]["window_sec"])),
                                        rng, training)
        else:
            bt, bd = np.zeros(0), np.zeros(0, bool)
        if self.augment is not None:
            notes, pedal, tokens, prompt, tempo = self._augment(notes, pedal, tokens, prompt, rng)
            bt = bt * tempo  # the beats follow the tempo change
        if self.with_context:
            before, notes, after = notes[:nb], notes[nb:nb + nw], notes[nb + nw:]
            pedal, cped = pedal[:npw], pedal[npw:]
        # synthetic noise only on clean performances: real transcriptions are noisy already
        clean = self.store.meta[i].get("source", "clean") == "clean"
        if self.noise is not None and clean:
            ref = notes
            notes, pedal = self.noise(notes, pedal, rng)
            if lab is not None:  # noise moved, dropped and added notes: carry the labels across
                from .metrical import transfer
                lab = np.stack(transfer((lab[:, 0], lab[:, 1]), ref, notes), axis=1)
            if nl is not None:
                from .notelabels import transfer_rows
                nl = transfer_rows(nl, ref, notes)
            if self.with_context:  # context notes come from the same transcriber: noise them too
                nrng = np.random.default_rng((self.seed, self.epoch, i, 11))
                before = self.noise(before, cped, nrng)[0] if len(before) else before
                after = self.noise(after, cped, nrng)[0] if len(after) else after
        if training and rng.random() < self.prompt_dropout:
            prompt = prompt[:0]
        if training and self._ctx is not None and len(prompt):
            # separate stream: with context noise off, every other random draw is unchanged
            crng = np.random.default_rng((self.seed, self.epoch, i, 7))
            if crng.random() < self.context_noise.prob:
                prompt = self._ctx(prompt, crng)
        min_end = self.store.meta[i].get("carry_min_sec")
        keep = released_mask(notes, None if min_end is None else min_end * tempo)
        notes = notes[keep]
        if lab is not None:
            lab = lab[keep]
        if nl is not None:
            nl = nl[keep]
        head = np.concatenate([[self.bos], prompt, [self.sep]]) if len(prompt) else np.array([self.bos])
        target = np.concatenate([head, tokens, [self.eos]])
        if self.with_context:
            nb_, na_ = len(before), len(after)
            notes, roles, order = assembly(notes, before, after)
            feats = note_features(notes, np.concatenate([pedal.reshape(-1, 2), cped]), roles)
            if lab is not None:
                lab = np.concatenate([lab, np.full((nb_ + na_, 2), -1, np.int8)])[order]
            if nl is not None:  # context notes: features only, nothing to learn
                nl = np.concatenate([nl, np.full((nb_ + na_, nl.shape[1]), -1, np.int32)])[order]
        else:
            roles = np.where(notes[:, 1] < 0, ROLE_CARRIED, ROLE_WINDOW).astype(np.int8)
            feats = note_features(notes, pedal)
        if self.scale is not None:
            cols, mean, std = self.scale
            feats[:, cols] = (feats[:, cols] - mean) / std
        if self.beats is not None:
            feats = np.concatenate([feats, beat_features(notes[:, 1], bt, bd)], axis=1)
        extra = {}
        if nl is not None:
            from .notelabels import grid_features, score_notes, targets
            if self.grid_override is not None and i in self.grid_override:
                g_t, g_q, win = self.grid_override[i]
                g_t = np.asarray(g_t) * tempo
            else:
                g0, g1 = self._nl["grid_index"][i]
                win = score_notes(self.vocab.decode(tokens.tolist()))[1]
                g_t, g_q = self._nl["grid_t"][g0:g1] * tempo, self._nl["grid_q"][g0:g1]
            gf, bar_est, pos_est = grid_features(notes[:, 1], g_t, g_q, win)
            feats = np.concatenate([feats, gf], axis=1)
            tg, fl = targets(nl, bar_est, self.label_space)
            extra = {"labels": torch.from_numpy(tg), "flags": torch.from_numpy(fl),
                     "bar_est": torch.from_numpy(bar_est.astype(np.int64)),
                     "pos_est": torch.from_numpy(pos_est.astype(np.float32))}
        item = {"notes": torch.from_numpy(feats), "roles": torch.from_numpy(roles),
                "target": torch.from_numpy(target), "prompt_len": len(head), **extra}
        if self.keep_raw:  # rows follow "notes" / "roles"
            item.update(raw_onsets=notes[:, 1].astype(np.float64), raw_pitches=notes[:, 0].astype(np.int64),
                        beat_t=np.asarray(bt, np.float64), beat_d=np.asarray(bd, bool))
        if lab is not None:
            lab = np.where((roles <= ROLE_CARRIED)[:, None] & (notes[:, 1] >= 0)[:, None], lab, -1)
            item["downbeat"] = torch.from_numpy(lab[:, 0].astype(np.int64))
            item["phase"] = torch.from_numpy(lab[:, 1].astype(np.int64))
        return item


def collate(batch, pad_id: int = 0):
    """Pads to the longest item. Returns notes [B,N,F], note_mask [B,N] (True = real),
    target [B,T] padded with ``pad_id`` (index of <pad> in the vocab, 0 by construction), and
    loss_mask [B,T]: True for the tokens the model should learn to predict (target + <eos>;
    not <bos>, the prompt, <sep> or padding). With teacher forcing, feed target[:, :-1] and score
    the logits against target[:, 1:] where loss_mask[:, 1:]."""
    B = len(batch)
    n = max(len(b["notes"]) for b in batch)
    t = max(len(b["target"]) for b in batch)
    notes = torch.zeros(B, n, batch[0]["notes"].shape[1])
    mask = torch.zeros(B, n, dtype=torch.bool)
    target = torch.full((B, t), pad_id, dtype=torch.long)
    loss_mask = torch.zeros(B, t, dtype=torch.bool)
    for k, b in enumerate(batch):
        notes[k, : len(b["notes"])] = b["notes"]
        mask[k, : len(b["notes"])] = True
        target[k, : len(b["target"])] = b["target"]
        loss_mask[k, b["prompt_len"]: len(b["target"])] = True
    out = {"notes": notes, "note_mask": mask, "target": target, "loss_mask": loss_mask}
    for key in ("downbeat", "phase"):  # metrical labels, padded with -1 (ignored)
        if key in batch[0]:
            lab = torch.full((B, n), -1, dtype=torch.long)
            for k, b in enumerate(batch):
                lab[k, : len(b[key])] = b[key]
            out[key] = lab
    for key, fill in (("labels", -1), ("flags", -1), ("bar_est", 0), ("pos_est", 0.0)):  # note labeller
        if key in batch[0]:
            shape = (B, n) + tuple(batch[0][key].shape[1:])
            t_ = torch.full(shape, fill, dtype=batch[0][key].dtype)
            for k, b in enumerate(batch):
                t_[k, : len(b[key])] = b[key]
            out[key] = t_
    return out


class MixedTrainDataset(Dataset):
    """Clean segments + synthetic noise, mixed with REAL Kong transcriptions (train_transcribed).

    ``real_fraction`` of the samples drawn by ``sampler()`` come from the real transcriptions
    (which get augmentation but no synthetic noise, since they're already noisy).

        train = MixedTrainDataset("data/processed/train", "data/processed/train_transcribed",
                                  "data/processed/vocab.json", noise=NoiseModel.from_json(...),
                                  augment=AugmentConfig(), real_fraction=0.3)
        dl = DataLoader(train, batch_size=32, sampler=train.sampler(), collate_fn=collate)
    """

    def __init__(self, clean_dir: str, real_dir: str, vocab_path: str, noise: NoiseModel | None,
                 augment: AugmentConfig | None = None, real_fraction: float = 0.3, seed: int = 0,
                 ornament_oversample: float = 3.0, context_noise: ContextNoise | None = None,
                 context_before: int = 0, context_after: int = 0, metrical: bool = False,
                 beats: dict | None = None, real_beats: dict | None = None, notelabels: bool = False):
        """``beats`` / ``real_beats``: beat arguments of ScoreSegmentDataset (beats, beat_noise,
        beat_dropout) for the clean and the real-transcription part; None = no beat features."""
        ctx = dict(context_noise=context_noise, context_before=context_before, context_after=context_after,
                   metrical=metrical, notelabels=notelabels)
        self.synthetic = ScoreSegmentDataset(clean_dir, vocab_path, noise=noise, augment=augment,
                                             seed=seed, ornament_oversample=ornament_oversample, **ctx,
                                             **(beats or {}))
        self.real = ScoreSegmentDataset(real_dir, vocab_path, noise=None, augment=augment,
                                        seed=seed + 1, ornament_oversample=ornament_oversample, **ctx,
                                        **(real_beats if real_beats is not None else beats or {}))
        self.vocab = self.synthetic.vocab
        self.real_fraction = real_fraction
        self.n_syn = len(self.synthetic)

    def set_epoch(self, epoch: int):
        self.synthetic.set_epoch(epoch)
        self.real.set_epoch(epoch)

    def sampler(self) -> WeightedRandomSampler:
        ws = self.synthetic.weights / self.synthetic.weights.sum() * (1 - self.real_fraction)
        wr = self.real.weights / self.real.weights.sum() * self.real_fraction
        return WeightedRandomSampler(torch.as_tensor(np.concatenate([ws, wr]), dtype=torch.double),
                                     num_samples=self.n_syn, replacement=True)

    def __len__(self):
        return self.n_syn + len(self.real)

    def __getitem__(self, i):
        return self.synthetic[i] if i < self.n_syn else self.real[i - self.n_syn]
