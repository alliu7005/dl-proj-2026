"""Target-token vocabulary built from the training split."""
from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
# Rare-but-important tokens the proposal wants up-weighted in the loss / oversampled.
ORNAMENT_PREFIXES = ("grace", "arp", "orn_", "ped_")

# Structural tokens are fixed by the grammar (so e.g. a time signature that only occurs in val/test
# is still in-vocabulary). Only sub-beat positions (sub_*) and durations (d_*) come from the data.
# Positions/durations are snapped to these denominators (see score.frac)
ALLOWED_DENOMINATORS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 32, 48)


def _fstr(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def _fractions(lo: Fraction, hi: Fraction, dens) -> list[str]:
    vals = {Fraction(k, d) for d in dens for k in range(int(lo * d), int(hi * d) + 1)}
    return sorted(_fstr(v) for v in vals if lo < v <= hi)


GRAMMAR = (
    ["bar", "grace", "tie", "arp", "ped_on", "ped_off", "d_0"]
    # every snapped sub-beat position, every snapped duration up to a half note,
    # and common longer durations (so test-time durations are never <unk>)
    + [f"sub_{f}" for f in _fractions(Fraction(0), Fraction(1), ALLOWED_DENOMINATORS) if f != "1"]
    + [f"d_{f}" for f in _fractions(Fraction(0), Fraction(2), ALLOWED_DENOMINATORS)]
    + [f"d_{f}" for f in _fractions(Fraction(2), Fraction(16), (1, 2, 3, 4, 8))]
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
            json.dump({"itos": self.itos, "counts": dict(sorted(self.counts.items()))}, f, indent=0)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path) as f:
            d = json.load(f)
        return cls(d["itos"], d.get("counts"))
