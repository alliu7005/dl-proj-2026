"""PyTorch dataset over preprocessed segments, with on-the-fly transcription-noise injection.

    ds = ScoreSegmentDataset("data/processed/train", "data/processed/vocab.json",
                             noise=NoiseModel.from_json("configs/noise_stats.json"),
                             augment=AugmentConfig())          # train only; omit for val/test
    loader = DataLoader(ds, batch_size=32, sampler=ds.sampler(), collate_fn=collate, num_workers=4)
    for epoch in ...:
        ds.set_epoch(epoch)   # new noise each epoch (with num_workers>0 use persistent_workers=False)

Each item:
    notes       float32 [N, 8]  per-note features (see ``note_features``), sorted by onset
    target      int64   [T]     <bos> [prompt <sep>] score tokens <eos>
    prompt_len  int             tokens before the first one to predict (<bos> + prompt + <sep>)

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

from .preprocessing.noise import NoiseModel
from .preprocessing.store import SegmentStore
from .preprocessing.vocab import BOS, EOS, ORNAMENT_PREFIXES, SEP, Vocab

N_FEATURES = 8
SCALED = [1, 2, 3, 4]  # onset, inter-onset gap, log duration, velocity: standardized with train stats
FEATURE_NAMES = ["pitch", "onset", "ioi", "log_dur", "velocity", "ped_at_onset", "ped_at_offset", "carried"]


def note_features(notes: np.ndarray, pedal: np.ndarray) -> np.ndarray:
    """[N,4] (pitch, onset, dur, vel) -> [N,8]:
    pitch (int, keep as index for an embedding), onset (s), inter-onset delta (s), log duration,
    velocity / 127, sustain-pedal-down at onset (0/1), sustain-pedal-down at offset (0/1),
    carried over (0/1: struck before the segment and still held into it; onset < 0)."""
    on, dur = notes[:, 1], notes[:, 2]
    delta = np.diff(on, prepend=on[:1]) if len(on) else on

    def ped_at(t):
        if len(pedal) == 0:
            return np.zeros_like(t)
        return ((t[:, None] >= pedal[None, :, 0]) & (t[:, None] < pedal[None, :, 1])).any(1).astype(np.float32)

    return np.stack([notes[:, 0], on, delta, np.log(np.maximum(dur, 1e-3)), notes[:, 3] / 127.0,
                     ped_at(on), ped_at(on + dur), (on < 0).astype(np.float32)], axis=1).astype(np.float32)


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


def drop_released_carried(notes: np.ndarray, min_end: float | None) -> np.ndarray:
    """Carried-over candidates (onset < 0) are kept only if the key is still down at ``min_end``
    (segment seconds). Applied after noise, so pedal lengthening in the noise model can turn a
    pedal-sustained note into a carried one, as in real transcriptions."""
    if min_end is None or not len(notes):
        return notes
    return notes[(notes[:, 1] >= 0) | (notes[:, 1] + notes[:, 2] >= min_end)]


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
                 context_noise: ContextNoise | None = None):
        """``prompt_dropout``: chance (when training, i.e. with noise or augmentation) of hiding the
        prompt, so the model also works without context (first window of a piece, or after a bad
        window). ``feature_stats``: JSON from ``compute_feature_stats`` (written by build_dataset.py
        next to vocab.json; "auto" = use it if present, None = leave features unscaled).
        ``context_noise``: corrupt the prompt of this fraction of training windows (see
        ``ContextNoise``); like prompt dropout, only when training."""
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
        if self.augment is not None:
            notes, pedal, tokens, prompt, tempo = self._augment(notes, pedal, tokens, prompt, rng)
        # synthetic noise only on clean performances: real transcriptions are noisy already
        if self.noise is not None and self.store.meta[i].get("source", "clean") == "clean":
            notes, pedal = self.noise(notes, pedal, rng)
        training = self.noise is not None or self.augment is not None
        if training and rng.random() < self.prompt_dropout:
            prompt = prompt[:0]
        if training and self._ctx is not None and len(prompt):
            # separate stream: with context noise off, every other random draw is unchanged
            crng = np.random.default_rng((self.seed, self.epoch, i, 7))
            if crng.random() < self.context_noise.prob:
                prompt = self._ctx(prompt, crng)
        min_end = self.store.meta[i].get("carry_min_sec")
        notes = drop_released_carried(notes, None if min_end is None else min_end * tempo)
        head = np.concatenate([[self.bos], prompt, [self.sep]]) if len(prompt) else np.array([self.bos])
        target = np.concatenate([head, tokens, [self.eos]])
        feats = note_features(notes, pedal)
        if self.scale is not None:
            cols, mean, std = self.scale
            feats[:, cols] = (feats[:, cols] - mean) / std
        return {"notes": torch.from_numpy(feats),
                "target": torch.from_numpy(target), "prompt_len": len(head)}


def collate(batch, pad_id: int = 0):
    """Pads to the longest item. Returns notes [B,N,F], note_mask [B,N] (True = real),
    target [B,T] padded with ``pad_id`` (index of <pad> in the vocab, 0 by construction), and
    loss_mask [B,T]: True for the tokens the model should learn to predict (target + <eos>;
    not <bos>, the prompt, <sep> or padding). With teacher forcing, feed target[:, :-1] and score
    the logits against target[:, 1:] where loss_mask[:, 1:]."""
    B = len(batch)
    n = max(len(b["notes"]) for b in batch)
    t = max(len(b["target"]) for b in batch)
    notes = torch.zeros(B, n, N_FEATURES)
    mask = torch.zeros(B, n, dtype=torch.bool)
    target = torch.full((B, t), pad_id, dtype=torch.long)
    loss_mask = torch.zeros(B, t, dtype=torch.bool)
    for k, b in enumerate(batch):
        notes[k, : len(b["notes"])] = b["notes"]
        mask[k, : len(b["notes"])] = True
        target[k, : len(b["target"])] = b["target"]
        loss_mask[k, b["prompt_len"]: len(b["target"])] = True
    return {"notes": notes, "note_mask": mask, "target": target, "loss_mask": loss_mask}


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
                 ornament_oversample: float = 3.0, context_noise: ContextNoise | None = None):
        self.synthetic = ScoreSegmentDataset(clean_dir, vocab_path, noise=noise, augment=augment,
                                             seed=seed, ornament_oversample=ornament_oversample,
                                             context_noise=context_noise)
        self.real = ScoreSegmentDataset(real_dir, vocab_path, noise=None, augment=augment,
                                        seed=seed + 1, ornament_oversample=ornament_oversample,
                                        context_noise=context_noise)
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
