"""PyTorch dataset over preprocessed segments, with on-the-fly transcription-noise injection.

    ds = ScoreSegmentDataset("data/processed/train", "data/processed/vocab.json",
                             noise=NoiseModel.from_json("configs/noise_stats.json"),
                             augment=AugmentConfig())          # train only; omit for val/test
    loader = DataLoader(ds, batch_size=32, sampler=ds.sampler(), collate_fn=collate, num_workers=4)
    for epoch in ...:
        ds.set_epoch(epoch)   # new noise each epoch (with num_workers>0 use persistent_workers=False)

Each item:
    notes   float32 [N, 7]  per-note features (see ``note_features``), sorted by onset
    target  int64   [T]     <bos> score tokens <eos>
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .preprocessing.noise import NoiseModel
from .preprocessing.store import SegmentStore
from .preprocessing.vocab import BOS, EOS, ORNAMENT_PREFIXES, Vocab

N_FEATURES = 7


def note_features(notes: np.ndarray, pedal: np.ndarray) -> np.ndarray:
    """[N,4] (pitch, onset, dur, vel) -> [N,7]:
    pitch (int, keep as index for an embedding), onset (s), inter-onset delta (s), log duration,
    velocity / 127, sustain-pedal-down at onset (0/1), sustain-pedal-down at offset (0/1)."""
    on, dur = notes[:, 1], notes[:, 2]
    delta = np.diff(on, prepend=on[:1]) if len(on) else on

    def ped_at(t):
        if len(pedal) == 0:
            return np.zeros_like(t)
        return ((t[:, None] >= pedal[None, :, 0]) & (t[:, None] < pedal[None, :, 1])).any(1).astype(np.float32)

    return np.stack([notes[:, 0], on, delta, np.log(np.maximum(dur, 1e-3)), notes[:, 3] / 127.0,
                     ped_at(on), ped_at(on + dur)], axis=1).astype(np.float32)


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


class ScoreSegmentDataset(Dataset):
    def __init__(self, split_dir: str, vocab_path: str, noise: NoiseModel | None = None,
                 seed: int = 0, ornament_oversample: float = 3.0, augment: AugmentConfig | None = None):
        self.store = SegmentStore(split_dir)
        self.vocab = Vocab.load(vocab_path)
        self.noise = noise
        self.augment = augment
        self._tr = _Transposer(self.vocab) if augment else None
        self.seed = seed
        self.epoch = 0
        self.bos, self.eos = self.vocab.stoi[BOS], self.vocab.stoi[EOS]
        rare = np.array([self.vocab.itos[i].startswith(ORNAMENT_PREFIXES[:3]) for i in range(len(self.vocab))])
        tok = np.asarray(self.store.tokens)
        idx = self.store.index
        has_rare = np.add.reduceat(rare[tok].astype(np.int64), idx[:, 4]) > 0 if len(idx) else np.zeros(0, bool)
        has_rare &= idx[:, 5] > idx[:, 4]
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

    def _augment(self, notes, pedal, tokens, rng):
        a = self.augment
        if rng.random() >= a.prob:
            return notes, pedal, tokens
        notes, pedal = notes.copy(), pedal.copy()
        lo = min(int(notes[:, 0].min()), self._tr.pitch_range(tokens)[0]) if len(notes) else 60
        hi = max(int(notes[:, 0].max()), self._tr.pitch_range(tokens)[1]) if len(notes) else 60
        k_lo, k_hi = max(a.transpose[0], 21 - lo), min(a.transpose[1], 108 - hi)
        if k_lo <= k_hi:
            k = int(rng.integers(k_lo, k_hi + 1))
            if k:
                notes[:, 0] += k
                tokens = self._tr(tokens, k)
        f = rng.uniform(*a.tempo)
        notes[:, 1:3] *= f
        pedal *= f
        notes[:, 3] = np.clip(np.round(notes[:, 3] * rng.uniform(*a.velocity)), 1, 127)
        return notes, pedal, tokens

    def __getitem__(self, i):
        notes, pedal, tokens = self.store.get(i)
        rng = np.random.default_rng((self.seed, self.epoch, i))
        if self.augment is not None:
            notes, pedal, tokens = self._augment(notes, pedal, tokens, rng)
        if self.noise is not None:
            notes, pedal = self.noise(notes, pedal, rng)
        target = np.concatenate([[self.bos], tokens, [self.eos]])
        return {"notes": torch.from_numpy(note_features(notes, pedal)),
                "target": torch.from_numpy(target)}


def collate(batch, pad_id: int = 0):
    """Pads to the longest item. Returns notes [B,N,F], note_mask [B,N] (True = real),
    target [B,T] padded with ``pad_id`` (index of <pad> in the vocab, 0 by construction)."""
    B = len(batch)
    n = max(len(b["notes"]) for b in batch)
    t = max(len(b["target"]) for b in batch)
    notes = torch.zeros(B, n, N_FEATURES)
    mask = torch.zeros(B, n, dtype=torch.bool)
    target = torch.full((B, t), pad_id, dtype=torch.long)
    for k, b in enumerate(batch):
        notes[k, : len(b["notes"])] = b["notes"]
        mask[k, : len(b["notes"])] = True
        target[k, : len(b["target"])] = b["target"]
    return {"notes": notes, "note_mask": mask, "target": target}


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
                 ornament_oversample: float = 3.0):
        self.synthetic = ScoreSegmentDataset(clean_dir, vocab_path, noise=noise, augment=augment,
                                             seed=seed, ornament_oversample=ornament_oversample)
        self.real = ScoreSegmentDataset(real_dir, vocab_path, noise=None, augment=augment,
                                        seed=seed + 1, ornament_oversample=ornament_oversample)
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
