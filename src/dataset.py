"""PyTorch dataset over preprocessed segments, with on-the-fly transcription-noise injection.

    ds = ScoreSegmentDataset("data/processed/train", "data/processed/vocab.json",
                             noise=NoiseModel.from_json("configs/noise_stats.json"))
    loader = DataLoader(ds, batch_size=32, sampler=ds.sampler(), collate_fn=collate, num_workers=4)
    for epoch in ...:
        ds.set_epoch(epoch)   # new noise each epoch (with num_workers>0 use persistent_workers=False)

Each item:
    notes   float32 [N, 7]  per-note features (see ``note_features``), sorted by onset
    target  int64   [T]     <bos> score tokens <eos>
"""
from __future__ import annotations

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


class ScoreSegmentDataset(Dataset):
    def __init__(self, split_dir: str, vocab_path: str, noise: NoiseModel | None = None,
                 seed: int = 0, ornament_oversample: float = 3.0):
        self.store = SegmentStore(split_dir)
        self.vocab = Vocab.load(vocab_path)
        self.noise = noise
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

    def __getitem__(self, i):
        notes, pedal, tokens = self.store.get(i)
        if self.noise is not None:
            rng = np.random.default_rng((self.seed, self.epoch, i))
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
