"""PKSpell (Foscarin, Audebert & Fournier-S'niehotta, "PKSpell: Data-Driven Pitch Spelling and Key
Signature Estimation", ISMIR 2021; https://github.com/fosfrancesco/pkspell, commit 477e668, MIT
licence): the key signature and every note's spelling of a score, from each note's pitch class and
written length only. Vendored like src/pm2s_beats.py: the model below is its PKSpell class (two
bidirectional GRUs: spelling from the first, key signature from a second over it); the released
weights are downloaded once into the torch cache and checked against their SHA-256.

Why: v2 spells every note from one key signature (src/preprocessing/detokenize.py spell: the
signature's direction, no context), and the key comes from a pitch-class profile match
(src/notelabels.py estimate_key). The whole-piece benchmark puts v2's spelling error at ~12% against
Beyer & Dai's 8.6%, and on Chopin's Mazurka op. 6 no. 2 (four sharps) the demo wrote four flats.

    respell(score)        # a music21 score: spellings and key signature set in place

PKSpell's released model was trained on ASAP's scores, which include our ASAP test pieces: on the
ASAP benchmark it is optimistic (as the released PM2S was); Gangqinpu's test set is music it never saw.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

COMMIT = "477e6687a7114f1991492e73f5c66edcb6241899"
WEIGHTS_URL = f"https://raw.githubusercontent.com/fosfrancesco/pkspell/{COMMIT}/models/pkspell_statedict.pt"
WEIGHTS_SHA256 = "5d678f31fb90279003a555458068f01801452eb223748662d0e4eeb03b9eb717"
# its vocabularies (src/utils/constants.py, src/data/pytorch_datasets.py)
PITCHES = {0: ["C", "B#", "D--"], 1: ["C#", "B##", "D-"], 2: ["D", "C##", "E--"], 3: ["D#", "E-", "F--"],
           4: ["E", "D##", "F-"], 5: ["F", "E#", "G--"], 6: ["F#", "E##", "G-"], 7: ["G", "F##", "A--"],
           8: ["G#", "A-"], 9: ["A", "G##", "B--"], 10: ["A#", "B-", "C--"], 11: ["B", "A##", "C-"]}
ACCEPTED = [x for v in PITCHES.values() for x in v]
KEY_SIGNATURES = list(range(-7, 8))
N_PC = 13            # 12 pitch classes + its padding symbol, one-hot
N_DUR = 4            # written lengths in 4 classes per piece (1-D k-means)


def weights_path() -> Path:
    root = Path(os.environ.get("TORCH_HOME") or Path.home() / ".cache" / "torch")
    return root / "pkspell" / "pkspell_statedict.pt"


def fetch_weights(path: Path | None = None) -> Path:
    path = Path(path or weights_path())
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        tmp.unlink()
        raise RuntimeError(f"PKSpell weights: SHA-256 {digest} is not the released file's {WEIGHTS_SHA256}")
    tmp.replace(path)
    return path


def kmeans_1d(x, k: int = N_DUR) -> np.ndarray:
    """Exact 1-D k-means cluster labels (ordered by value), as the kmeans1d package PKSpell used:
    an optimal partition of the sorted distinct values (dynamic programming over their counts;
    equal values always share a cluster in an optimum). Fewer distinct values than ``k``: one
    cluster each."""
    x = np.asarray(x, np.float64)
    if not len(x):
        return np.zeros(0, np.int64)
    vals, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    m = len(vals)
    if m <= k:
        return inv.astype(np.int64)
    w, wx, wxx = (np.concatenate([[0.0], np.cumsum(a)]) for a in (cnt, cnt * vals, cnt * vals * vals))

    def cost(i, j):   # values i..j-1 in one cluster
        n, s = w[j] - w[i], wx[j] - wx[i]
        return (wxx[j] - wxx[i]) - s * s / n
    inf = float("inf")
    dp = np.full((k + 1, m + 1), inf)
    arg = np.zeros((k + 1, m + 1), np.int64)
    dp[0, 0] = 0.0
    for c in range(1, k + 1):
        for j in range(c, m + 1):
            best, bi = inf, c - 1
            for i in range(c - 1, j):
                v = dp[c - 1, i] + cost(i, j)
                if v < best:
                    best, bi = v, i
            dp[c, j], arg[c, j] = best, bi
    labels = np.zeros(m, np.int64)
    j = m
    for c in range(k, 0, -1):
        i = arg[c, j]
        labels[i:j] = c - 1
        j = i
    return labels[inv]


def build_model():
    import torch.nn as nn

    class PKSpell(nn.Module):
        def __init__(self, input_dim=N_PC + N_DUR, hidden_dim=300, hidden_dim2=24):
            super().__init__()
            self.rnn = nn.GRU(input_dim, hidden_dim // 2, bidirectional=True)
            self.rnn2 = nn.GRU(hidden_dim, hidden_dim2 // 2, bidirectional=True)
            self.top_layer_pitch = nn.Linear(hidden_dim, len(ACCEPTED) + 1)
            self.top_layer_ks = nn.Linear(hidden_dim2, len(KEY_SIGNATURES) + 1)

        def forward(self, x):          # x [T, 1, input_dim] -> (spelling logits [T, 1, 36], key logits [T, 1, 16])
            h, _ = self.rnn(x)
            k, _ = self.rnn2(h)
            return self.top_layer_pitch(h), self.top_layer_ks(k)
    return PKSpell()


_MODEL = None


def model(device="cpu"):
    global _MODEL
    if _MODEL is None:
        import torch
        m = build_model()
        m.load_state_dict(torch.load(fetch_weights(), map_location="cpu", weights_only=True))
        _MODEL = m.eval()
    return _MODEL.to(device)


def predict(pitch_classes, durations, device="cpu") -> tuple[list[str], list[int]]:
    """Each note's spelling (A#, B-, ...; '-' = flat, music21's) and key signature (sharps > 0),
    from the notes in score order (pitch classes 0-11, written lengths in any unit)."""
    import torch
    pc = np.asarray(pitch_classes, np.int64) % 12
    if not len(pc):
        return [], []
    x = np.zeros((len(pc), N_PC + N_DUR), np.float32)
    x[np.arange(len(pc)), pc] = 1.0
    x[np.arange(len(pc)), N_PC + kmeans_1d(durations)] = 1.0
    mdl = model(device)
    with torch.no_grad():
        sp, ks = mdl(torch.from_numpy(x)[:, None, :].to(device))
    sp = sp[:, 0, :len(ACCEPTED)].argmax(-1).cpu().numpy()   # never the padding symbol
    ks = ks[:, 0, :len(KEY_SIGNATURES)].argmax(-1).cpu().numpy()
    names = [ACCEPTED[i] for i in sp]
    for i, (n, p) in enumerate(zip(names, pc)):   # a spelling of another pitch class: keep the pitch
        if n not in PITCHES[int(p)]:
            names[i] = PITCHES[int(p)][0]
    return names, [KEY_SIGNATURES[i] for i in ks]


def respell(score, device="cpu") -> dict:
    """PKSpell on a music21 score, in place: every note named as it says (the octave kept: B#3 is
    C4's pitch), and every key signature set to the one it gives most notes. Returns {"key": the
    key signature, "respelled": notes whose name changed, "notes": notes seen}."""
    import music21 as m21
    flat = score.flatten().notes
    pitches = [p for n in flat for p in n.pitches]
    durs = [float(n.duration.quarterLength) for n in flat for _ in n.pitches]
    names, keys = predict([p.midi % 12 for p in pitches], durs, device)
    changed = 0
    for p, name in zip(pitches, names):
        if p.name == name:
            continue
        midi = p.midi
        new = m21.pitch.Pitch(name)
        new.octave = p.octave if p.octave is not None else 4
        new.octave += (midi - new.midi) // 12
        while new.midi < midi:
            new.octave += 1
        while new.midi > midi:
            new.octave -= 1
        p.name, p.octave = new.name, new.octave
        changed += 1
    key = max(set(keys), key=keys.count) if keys else 0
    for ks in score.recurse().getElementsByClass(m21.key.KeySignature):
        ks.sharps = key
    return {"key": int(key), "respelled": changed, "notes": len(pitches)}
