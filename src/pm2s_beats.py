"""PM2S's beat tracker on performance MIDI (Liu, Kong, Morfi & Benetos, "Performance MIDI-to-score
conversion by neural beat tracking", ISMIR 2022; https://github.com/cheriell/PM2S, commit 9586f91):
beats and downbeats from the notes themselves, no audio.

    tracker = PM2SBeats()                     # weights downloaded once (Zenodo), then cached
    times, down = tracker(notes)              # notes [N, 4]: pitch, onset s, offset s, velocity

Why: v2 on a MIDI input gets Beat This!'s beats of the MIDI rendered by src/synth.py, which are
poor (beat F 0.58 against 0.71 on the real recordings; 2.8 MUSTER points on clean MIDI). PM2S reads
the notes directly. Its released weights were trained on all of ASAP (``--full_train``), our test
performances included: numbers on the ASAP test set are optimistic until it is retrained without
them (``weights=`` / PM2S_WEIGHTS takes such a checkpoint).

The model (a CNN over one-hot note features, then GRUs for beat / downbeat / tempo) and its
post-processing (dynamic thresholds, merging close beats, filling missing beats by dynamic
programming) are PM2S's ``RNNJointBeatModel`` and ``RNNJointBeatProcessor``, copied here so it runs
on our PyTorch without its pinned 2022 environment. MIT licence:

    Copyright (c) 2022 Lele Liu
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions: The above copyright notice
    and this permission notice shall be included in all copies or substantial portions of the
    Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

Differences from PM2S: inputs are our [N, 4] notes (offsets, converted to its durations); the
downbeats come back as a flag on the beat each one sits on (as Beat This!'s do); fewer than three
beats skip the gap filling (where PM2S fails).
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

WEIGHTS_URL = "https://zenodo.org/records/10520196/files/RNNJointBeatModel.pth?download=1"
WEIGHTS_SHA256 = "939e0181f119a473200aece6307906e3dee69d4e2c44abcc878755bf6a18beb6"
RESOLUTION = 0.01     # onset shift one-hot resolution, seconds
TOLERANCE = 0.07      # beats closer than 2 x this are one beat
MIN_BPM = 40
IBI_VOCAB = int(4 / RESOLUTION) + 1


def default_weights_path() -> Path:
    """$PM2S_WEIGHTS, else the torch cache ($TORCH_HOME, on PACE on scratch)/pm2s/."""
    if os.environ.get("PM2S_WEIGHTS"):
        return Path(os.environ["PM2S_WEIGHTS"])
    root = Path(os.environ.get("TORCH_HOME") or Path.home() / ".cache" / "torch")
    return root / "pm2s" / "RNNJointBeatModel.pth"


def fetch_weights(path: Path) -> Path:
    """The released weights at ``path``, downloaded and checked against their SHA-256 if missing."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        tmp.unlink()
        raise RuntimeError(f"PM2S weights: SHA-256 {digest} is not the released file's {WEIGHTS_SHA256}")
    tmp.replace(path)
    return path


def _model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    in_features = 128 + IBI_VOCAB + 1 + 1     # pitch one-hot, onset shift one-hot, duration, velocity

    def encode(note_seq):
        pitch = F.one_hot(note_seq[:, :, 0].long(), 128).float()
        onset = note_seq[:, :, 1]
        shift = onset[:, 1:] - onset[:, :-1]
        shift = torch.cat([torch.zeros_like(shift[:, 0:1]), shift], dim=1).clamp(0, 4)
        shift = F.one_hot(torch.round(shift / RESOLUTION).long(), IBI_VOCAB).float()
        return torch.cat([pitch, shift, note_seq[:, :, 2].float().unsqueeze(2),
                          note_seq[:, :, 3].float().unsqueeze(2) / 127.0], dim=2)

    class ConvBlock(nn.Module):
        def __init__(self, in_features, hidden_size=512, kernel_size=9, dropout=0.15):
            super().__init__()
            h, k = hidden_size, kernel_size
            self.conv = nn.Sequential(
                nn.Conv2d(1, h // 4, kernel_size=(k, in_features), padding=(k // 2, 0)),
                nn.BatchNorm2d(h // 4), nn.ELU(), nn.Dropout(p=dropout),
                nn.Conv2d(h // 4, h // 2, kernel_size=(k, 1), padding=(k // 2, 0)),
                nn.BatchNorm2d(h // 2), nn.ELU(), nn.Dropout(p=dropout),
                nn.Conv2d(h // 2, h, kernel_size=(k, 1), padding=(k // 2, 0)),
                nn.BatchNorm2d(h), nn.ELU(), nn.Dropout(p=dropout))

        def forward(self, x):
            return self.conv(x.unsqueeze(1)).squeeze(3).transpose(1, 2)

    class GRUBlock(nn.Module):
        def __init__(self, in_features, hidden_size=512, gru_layers=2, dropout=0.15):
            super().__init__()
            self.grus_beat = nn.GRU(input_size=in_features, hidden_size=hidden_size, num_layers=gru_layers,
                                    batch_first=True, dropout=dropout, bidirectional=True)
            self.linear = nn.Linear(hidden_size * 2, hidden_size)

        def forward(self, x):
            return self.linear(self.grus_beat(x)[0])

    class LinearOutput(nn.Module):
        def __init__(self, in_features, out_features, activation_type="sigmoid", dropout=0.15):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            self.linear = nn.Linear(in_features, out_features)
            self.activation = nn.Sigmoid() if activation_type == "sigmoid" else nn.LogSoftmax(dim=2)

        def forward(self, x):
            return self.activation(self.linear(self.dropout(x)))

    class RNNJointBeatModel(nn.Module):
        def __init__(self, hidden_size=512):
            super().__init__()
            self.convs = ConvBlock(in_features=in_features)
            self.gru_beat = GRUBlock(in_features=hidden_size)
            self.gru_downbeat = GRUBlock(in_features=hidden_size)
            self.gru_tempo = GRUBlock(in_features=hidden_size)
            self.out_beat = LinearOutput(hidden_size, 1, "sigmoid")
            self.out_downbeat = LinearOutput(hidden_size, 1, "sigmoid")
            self.out_tempo = LinearOutput(hidden_size, IBI_VOCAB, "softmax")

        def forward(self, x):
            x = self.convs(encode(x))
            xb = self.gru_beat(x)
            xd = self.gru_downbeat(xb)
            return self.out_beat(xb).squeeze(2), self.out_downbeat(xd).squeeze(2)

    return RNNJointBeatModel()


def _dedupe(times: np.ndarray) -> np.ndarray:
    """PM2S: runs of times closer than 2 x TOLERANCE -> the mean of each run's first and last."""
    if not len(times):
        return times
    first = times[np.concatenate([[True], np.abs(np.diff(times)) > TOLERANCE * 2])]
    last = times[::-1][np.concatenate([[True], np.abs(np.diff(times[::-1])) > TOLERANCE * 2])][::-1]
    return np.mean([first, last], axis=0)


def postprocess(beat_probs: np.ndarray, downbeat_probs: np.ndarray, onsets: np.ndarray, prob_thresh: float = 0.5,
                penalty: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """PM2S's RNNJointBeatProcessor.pps: per-note probabilities -> (beat times, downbeat times)."""
    n = len(beat_probs)
    wb, wd = (60.0 / MIN_BPM) * 4, (60.0 / MIN_BPM) * 8      # threshold windows, seconds
    tb, td = np.ones(n) * prob_thresh, np.ones(n) * prob_thresh
    lb = rb = ld = rd = 0
    for i, on in enumerate(onsets):
        while onsets[lb] < on - wb / 2:
            lb += 1
        while rb < n and onsets[rb] < on + wb / 2:
            rb += 1
        while onsets[ld] < on - wd / 2:
            ld += 1
        while rd < n and onsets[rd] < on + wd / 2:
            rd += 1
        tb[i] = np.max(beat_probs[lb:rb]) * prob_thresh
        td[i] = np.max(downbeat_probs[ld:rd]) * prob_thresh
    beats = _dedupe(onsets[beat_probs > tb])
    downbeats = _dedupe(onsets[downbeat_probs > td])
    extra = [d for d in downbeats if not len(beats) or np.min(np.abs(beats - d)) > TOLERANCE * 2]
    beats = np.sort(np.concatenate([beats, extra]))
    if len(beats) >= 3:
        beats = complement(beats, penalty)
    return beats, downbeats


def complement(beats: np.ndarray, penalty: float = 1.0) -> np.ndarray:
    """PM2S's beat_complement_dp: beats added where an interval is 2-4 times its neighbours', then
    insertions chosen by dynamic programming (steady inter-beat intervals, a penalty per insertion)."""
    wlen = 5
    ibis = np.diff(beats)
    filled = []
    for i in range(len(beats) - 1):
        filled.append(beats[i])
        ibi = ibis[i]
        near = np.median(ibis[max(0, i - wlen):min(len(ibis), i + wlen + 1)])
        for ratio in (2, 3, 4):
            if abs(ibi / near - ratio) / ratio < 0.15:
                for x in range(1, ratio):
                    filled.append(beats[i] + x * ibi / ratio)
    filled.append(beats[-1])     # PM2S drops the last beat here (it loops to len - 1); kept, as the DP needs it
    beats = np.sort(np.array(filled))
    paths = [[beats[0], beats[1]] for _ in range(4)]
    cost = [0.0] * 4
    for i in range(2, len(beats)):
        new_paths, new_cost = [None] * 4, [0.0] * 4
        for x in range(4):
            ibi = (beats[i] - beats[i - 1]) / (x + 1)
            objs = [cost[p] + abs(np.log(ibi / (paths[p][-1] - paths[p][-2]))) + penalty * x for p in range(4)]
            best = int(np.argmin(objs))
            new_paths[x] = paths[best] + [beats[i - 1] + ibi * k for k in range(1, x + 1)] + [beats[i]]
            new_cost[x] = objs[best]
        paths, cost = new_paths, new_cost
    return np.array(paths[int(np.argmin(cost))])


class PM2SBeats:
    """PM2S's beat tracker, loaded once: ``tracker(notes)`` -> (beat times, downbeat flags)."""

    def __init__(self, weights: str | Path | None = None, device: str = "cpu"):
        import torch
        path = fetch_weights(Path(weights) if weights else default_weights_path())
        self.model = _model()
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model.to(device).eval()
        self.device, self.weights = device, str(path)

    def probabilities(self, notes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(beat, downbeat probability per note, onsets) for our [N, 4] notes, sorted by onset."""
        import torch
        n = np.asarray(notes, np.float64)
        n = n[np.argsort(n[:, 1], kind="stable")]
        seq = np.stack([n[:, 0], n[:, 1], np.maximum(n[:, 2] - n[:, 1], 0.0), n[:, 3]], axis=1).astype(np.float32)
        with torch.no_grad():
            b, d = self.model(torch.from_numpy(seq).unsqueeze(0).to(self.device))
        return b[0].float().cpu().numpy(), d[0].float().cpu().numpy(), seq[:, 1].astype(np.float64)

    def __call__(self, notes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(notes) < 2:
            return np.zeros(0), np.zeros(0, bool)
        b, d, onsets = self.probabilities(notes)
        beats, downbeats = postprocess(b, d, onsets)
        beats = np.asarray(beats, np.float64)
        down = np.zeros(len(beats), bool)
        if len(beats) and len(downbeats):   # each downbeat flags the beat it sits on (a merged one is within 2 x TOLERANCE)
            j = np.abs(beats[:, None] - np.asarray(downbeats)[None, :]).argmin(0)
            ok = np.abs(beats[j] - downbeats) <= TOLERANCE * 2
            down[j[ok]] = True
        return beats, down
