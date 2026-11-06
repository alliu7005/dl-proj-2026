"""PyTorch dataset over preprocessed segments, with on-the-fly transcription-noise injection.

    ds = ScoreSegmentDataset("data/processed/train", "data/processed/vocab.json",
                             noise=NoiseModel.from_json("data/noise_stats.json"))
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


class ScoreSegmentDataset(Dataset):
    def __init__(self, split_dir: str, vocab_path: str, noise: NoiseModel | None = None,
                 seed: int = 0, ornament_oversample: float = 3.0, use_prompt: bool = True,
                 prompt_dropout: float = 0.1, feature_stats: str | None = "auto"):
        """``prompt_dropout``: chance (with noise, i.e. training) of hiding the prompt, so the
        model also works without context (first window of a piece, or after a bad window).
        ``feature_stats``: JSON from ``compute_feature_stats`` (written by build_dataset.py next to
        vocab.json; "auto" = use it if present, None = leave features unscaled)."""
        self.store = SegmentStore(split_dir)
        self.vocab = Vocab.load(vocab_path)
        self.noise = noise
        self.seed = seed
        self.epoch = 0
        self.use_prompt, self.prompt_dropout = use_prompt, prompt_dropout
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

    def __getitem__(self, i):
        notes, pedal, tokens = self.store.get(i)
        prompt = self.store.get_prompt(i) if self.use_prompt else np.zeros(0, dtype=np.int64)
        if self.noise is not None:  # training
            rng = np.random.default_rng((self.seed, self.epoch, i))
            if self.store.meta[i].get("source", "clean") == "clean":  # real transcriptions are noisy already
                notes, pedal = self.noise(notes, pedal, rng)
            if rng.random() < self.prompt_dropout:
                prompt = prompt[:0]
        notes = drop_released_carried(notes, self.store.meta[i].get("carry_min_sec"))
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
