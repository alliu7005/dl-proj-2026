"""Beat-level decoding of the note labeller's position heads: one subdivision per beat, the model's
own probabilities fused with when each note was played.

The labeller (src/labeller.py) predicts each note's position as a whole quarter (head ``pos``) and
a split of it (head ``sub``: 1/2, 1/4, 1/3, 1/6, ...), and placed_notes takes each note's likeliest
split on its own. Two weaknesses showed in the user's song (2026-11-26), bars 64-71:

  * nothing ties the notes of one beat together: a beat played as the straight 16th syncopation
    (1/2, 3/4 of the beat; then 1/4, 3/4) came out with some notes on the straight grid and others
    on the triplet one (2/3; 1/3, 2/3), which no score writes;
  * the played timing is one input among many: the model saw the onsets, yet its position head
    leaned on the window's rhythm (a window that tips to triplets writes triplets).

``decode_subs`` replaces the per-note choice with a constrained, fused one:

  score_i(s) = log p_model(sub_i = s) - weight * (x_i - s)^2

where x_i is where in its quarter the note was played on the grid (V2Pipeline's position
estimate; ``anchor``: measured from where the beat's on-the-quarter notes were played, up to that
much early or late), and with ``consistent`` every beat (its notes in one bar and quarter, and on
one staff with ``by_staff``) takes ONE family of splits, the one whose best splits score highest
together:

  straight  0 and the splits whose denominator is a power of 2 (1/2, 1/4, 3/4, 1/8, ...)
  triplet   0, 1/2 and the splits whose denominator is a multiple of 3 (1/3, 2/3, 1/6, 5/6, ...);
            1/2 belongs to both (a sextuplet run passes the half beat)
  other     0 and the rest (quintuplets, septuplets, ...)

Each note then takes its best split within its beat's family; a note's length that is a triplet
length in a beat decoded straight takes the model's likeliest straight length instead.

With ``prior`` > 0 (and ``consistent``) the beats' families are not chosen one by one but as a
sequence (Viterbi over each window's beats in time order, a chain per staff with ``by_staff``):

  total(k_1..k_T) = sum_t [ beat_t's score in family k_t + prior * log P(k_t | k_{t-1}) ]

with P the rhythm prior (``table``: log_init [3], log_trans [3][3]), learned from the written
scores alone (scripts/fit_rhythm_prior.py): how often a beat is straight, triplet or other, and
which follows which. A lone triplet beat among straight ones then costs what the scores say it
should; runs the labeller and the timing agree on still come through.

``weight`` 0 with ``consistent`` off is exactly the labeller's own choice. The parameters are not
hand-set: scripts/tune_beat_decode.py picks them on the validation windows (ASAP and Gangqinpu,
the tracker's timing errors on the grid, as the labeller was validated in training) and writes
them next to the checkpoint (``beat_decode.json``), which V2Pipeline reads. This is the model's
decision, made consistently: the rules of src/timing.py (check_timing, check_beats) judge the
written result afterwards and are measured against it (scripts/system_scores.py: _bd against
_tc / _tb).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

FILE = "beat_decode.json"
STRAIGHT, TRIPLET, OTHER = 0, 1, 2


@dataclass
class BeatDecode:
    weight: float = 0.0        # the timing term's weight (per squared quarter)
    consistent: bool = True    # one family of splits per beat
    by_staff: bool = False     # a beat per staff (else both hands share it)
    anchor: float = 0.1        # the most a beat's on-the-quarter notes may shift its timing
    prior: float = 0.0         # the rhythm prior's weight (0: each beat's family on its own)
    table: dict | None = None  # the rhythm prior: {"log_init": [3], "log_trans": [[3] x 3]}

    @property
    def active(self) -> bool:
        return self.weight > 0 or self.consistent

    @property
    def sequential(self) -> bool:
        return self.consistent and self.prior > 0 and self.table is not None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path) -> "BeatDecode | None":
        """The tuned parameters in ``path`` (a beat_decode.json, or the directory holding it), else None."""
        p = Path(path)
        p = p / FILE if p.is_dir() else p
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        d = d.get("best", d)
        return cls(**{k: d[k] for k in ("weight", "consistent", "by_staff", "anchor", "prior", "table") if k in d})


def family_of(f: Fraction) -> int:
    d = Fraction(f).denominator
    if d & (d - 1) == 0:
        return STRAIGHT
    return TRIPLET if d % 3 == 0 else OTHER


class Families:
    """The split classes of a label space by family (masks over the ``sub`` head's classes, 0 = on
    the quarter, in every family) and the straight lengths (a mask over the ``dur`` head's)."""

    def __init__(self, space):
        vals = [Fraction(0)] + list(space.subs)
        self.values = np.array([float(v) for v in vals])
        fam = np.array([-1] + [family_of(v) for v in space.subs])
        self.masks = np.zeros((3, len(vals)), bool)
        for k in range(3):
            self.masks[k] = (fam == k) | (fam == -1)
        half = [i for i, v in enumerate(vals) if v == Fraction(1, 2)]
        self.masks[TRIPLET, half] = True
        self.fam = fam
        self.straight_dur = np.array([Fraction(d).denominator & (Fraction(d).denominator - 1) == 0
                                      for d in space.durs])
        self.triplet_dur = np.array([Fraction(d).denominator % 3 == 0 for d in space.durs])


def viterbi(emit: np.ndarray, log_init: np.ndarray, log_trans: np.ndarray, w: float) -> np.ndarray:
    """The best family sequence: ``emit`` [T, K] each beat's score per family, the prior weighted by ``w``."""
    T, K = emit.shape
    v = emit[0] + w * log_init
    back = np.zeros((T, K), np.int64)
    for t in range(1, T):
        cand = v[:, None] + w * log_trans          # [from, to]
        back[t] = cand.argmax(0)
        v = cand.max(0) + emit[t]
    out = np.zeros(T, np.int64)
    out[-1] = int(v.argmax())
    for t in range(T - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def decode_subs(sub_logp: np.ndarray, x: np.ndarray, groups: np.ndarray, fams: Families,
                p: BeatDecode, order: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The split each note takes (module docstring). ``sub_logp`` [n, n_sub] log-probabilities;
    ``x`` [n] where in its quarter each note was played (NaN: unknown, no timing term); ``groups``
    [n] its beat (an id; -1: decoded alone); ``order`` {beat id: (chain, time)} for the sequential
    decoding (``p.prior``; without it the ids' order is the time order, one chain). Returns (split
    class per note, family of its beat: -1 where decoded alone or nothing was constrained)."""
    n = len(sub_logp)
    score = np.array(sub_logp, np.float64, copy=True)
    x = np.asarray(x, np.float64).copy()
    groups = np.asarray(groups)
    if p.anchor > 0 and n:
        own = score.argmax(1) == 0
        for g in np.unique(groups[groups >= 0]):
            m = groups == g
            on = x[m & own & np.isfinite(x)]
            if len(on):
                x[m] -= float(np.clip(np.median(on), -p.anchor, p.anchor))
    if p.weight > 0:
        d = x[:, None] - fams.values[None, :]
        t = -p.weight * d * d
        score += np.where(np.isfinite(t), t, 0.0)
    pick = score.argmax(1)
    beat_fam = np.full(n, -1)
    if not p.consistent:
        return pick, beat_fam
    ids = np.unique(groups[groups >= 0])
    tots = {}
    for g in ids:
        sc = score[groups == g]
        tots[int(g)] = np.array([np.where(fams.masks[k][None], sc, -np.inf).max(1).sum() for k in range(3)])
    choice = {g: int(np.argmax(t)) for g, t in tots.items()}
    if p.sequential and len(ids) > 1:
        li = np.asarray(p.table["log_init"], np.float64)
        lt = np.asarray(p.table["log_trans"], np.float64)
        chains = {}
        for g in tots:
            c, t = (order or {}).get(g, (0, g))
            chains.setdefault(c, []).append((t, g))
        for beats in chains.values():
            seq = [g for _, g in sorted(beats)]
            emit = np.stack([np.where(np.isfinite(tots[g]), tots[g], -1e9) for g in seq])
            for g, k in zip(seq, viterbi(emit, li, lt, p.prior)):
                choice[g] = int(k)
    for g, k in choice.items():
        idx = np.flatnonzero(groups == g)
        pick[idx] = np.where(fams.masks[k][None], score[idx], -np.inf).argmax(1)
        beat_fam[idx] = k
    return pick, beat_fam


def straight_lengths(dur_logits: np.ndarray, dur_pick: np.ndarray, beat_fam: np.ndarray, fams: Families) -> np.ndarray:
    """A note in a beat decoded straight whose likeliest length is a triplet length takes its
    likeliest straight length."""
    out = np.asarray(dur_pick).copy()
    fix = (beat_fam == STRAIGHT) & fams.triplet_dur[out]
    if fix.any():
        out[fix] = np.where(fams.straight_dur[None], dur_logits[fix], -np.inf).argmax(1)
    return out


def beat_groups(bar: np.ndarray, pos: np.ndarray, staff: np.ndarray, use: np.ndarray, by_staff: bool) -> np.ndarray:
    """A beat id per note (bar, quarter, and staff with ``by_staff``) for the ``use`` notes, -1 for the rest."""
    g = np.full(len(bar), -1)
    keys = {}
    for i in np.flatnonzero(use):
        k = (int(bar[i]), int(pos[i]), int(staff[i]) if by_staff else 0)
        g[i] = keys.setdefault(k, len(keys))
    return g


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, np.float64)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def decode_item(sub_logits: np.ndarray, dur_logits: np.ndarray, pick: dict, bar_est, pos_est, use: np.ndarray,
                fams: Families, p: BeatDecode) -> tuple[np.ndarray, np.ndarray]:
    """One window's notes: (split class, length class) per note, decoded (module docstring).
    ``pick``: the heads' argmax per note ("delta", "pos", "sub", "dur", "staff"); ``bar_est`` /
    ``pos_est``: the grid's bar and position of each note; ``use``: the notes rendered (the others
    keep their argmax and join no beat). The timing term applies where the predicted bar is the
    grid's (delta 0) and the note was played within 1.5 quarters of its predicted quarter."""
    n = len(sub_logits)
    bars = np.maximum(0, np.asarray(bar_est[:n], np.int64) + pick["delta"][:n] - 1)
    x = np.asarray(pos_est[:n], np.float64) - pick["pos"][:n]
    x = np.where((pick["delta"][:n] == 1) & (np.abs(x) < 1.5), x, np.nan)
    groups = beat_groups(bars, pick["pos"][:n], pick["staff"][:n], np.asarray(use[:n], bool), p.by_staff)
    order = {}
    for i in np.flatnonzero(groups >= 0):   # a chain per staff with by_staff, in time order
        order.setdefault(int(groups[i]), (int(pick["staff"][i]) if p.by_staff else 0,
                                          (int(bars[i]), int(pick["pos"][i]))))
    sub, fam = decode_subs(log_softmax(sub_logits), x, groups, fams, p, order)
    sub = np.where(groups >= 0, sub, pick["sub"][:n])
    return sub, straight_lengths(dur_logits, pick["dur"][:n], fam, fams)
