"""Target-token vocabulary built from the training split."""
from __future__ import annotations

import json
from collections import Counter

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
# Rare-but-important tokens the proposal wants up-weighted in the loss / oversampled.
ORNAMENT_PREFIXES = ("grace", "arp", "orn_", "ped_")

# Structural tokens are fixed by the grammar (so e.g. a time signature that only occurs in val/test
# is still in-vocabulary). Only sub-beat positions (sub_*) and durations (d_*) come from the data.
GRAMMAR = (
    ["bar", "grace", "tie", "arp", "ped_on", "ped_off"]
    + [f"orn_{o}" for o in ("fermata", "mordent", "tremolo", "trill", "turn")]
    + [f"staff_{i}" for i in range(4)]
    + [f"key_{k}" for k in range(-7, 8)]
    + [f"p_{p}" for p in range(21, 109)]
    + [f"pos_{i}" for i in range(16)]
    + [f"ts_{n}/{d}" for d in (2, 4, 8, 16) for n in range(1, 17)]
)


class Vocab:
    def __init__(self, tokens: list[str], counts: dict | None = None):
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.counts = counts or {}

    @classmethod
    def build(cls, token_lists, min_count: int = 2) -> "Vocab":
        return cls.from_counts(Counter(t for toks in token_lists for t in toks), min_count)

    @classmethod
    def from_counts(cls, c: Counter, min_count: int = 2) -> "Vocab":
        kept = set(GRAMMAR) | {t for t, n in c.items() if n >= min_count}
        kept = sorted(kept, key=lambda t: (t.split("_")[0], t))
        return cls(SPECIALS + kept, dict(c))

    def encode(self, tokens: list[str], add_special: bool = True) -> list[int]:
        unk = self.stoi[UNK]
        ids = [self.stoi.get(t, unk) for t in tokens]
        return [self.stoi[BOS]] + ids + [self.stoi[EOS]] if add_special else ids

    def decode(self, ids) -> list[str]:
        return [self.itos[i] for i in ids if self.itos[i] not in (PAD, BOS, EOS)]

    def loss_weights(self, ornament_weight: float = 5.0) -> list[float]:
        """Per-token CE weights: ornament/pedal tokens get ``ornament_weight``, PAD gets 0."""
        w = [ornament_weight if t.startswith(ORNAMENT_PREFIXES) else 1.0 for t in self.itos]
        w[self.stoi[PAD]] = 0.0
        return w

    def __len__(self):
        return len(self.itos)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"itos": self.itos, "counts": self.counts}, f, indent=0)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path) as f:
            d = json.load(f)
        return cls(d["itos"], d.get("counts"))
