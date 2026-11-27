"""A learned chooser for each note's written length, among the labeller's own top-k candidates.

The labeller predicts every note's length in one parallel pass: a note does not see where its
neighbours were placed, so it cannot tell that a quarter note is followed by its staff's next onset
a quarter later. The headroom test (scripts/headroom.py labeller) found the right length among the
dur head's top 2 for ~90% of notes and its top 4 for ~95%, against ~80% for its likeliest, and the
windows' rendered duration accuracy 0.87 -> 0.97 with a perfect pick among the top 4 (true grid).

This module scores each candidate length from what the labeller lacked, once the notes are placed
(the idea of Nakamura, Yoshii & Dixon's note-value model: a note's length is read from the onsets
around it):
  * the labeller's own log-probability of it and its rank;
  * the written gap to the next onset on the note's staff, and on any staff: the candidate's length
    against them (equal, shorter, longer, log ratio);
  * the performed duration against the performed gap to that next onset on the staff (seconds), so
    a held note and a detached one differ;
  * where the candidate ends: on a beat (quarter), on the bar line, past the bar line;
  * how many notes of the same chord (same position and staff) the labeller gave that length.
A small MLP scores each candidate (shared over the k), softmax over the k; trained with cross-entropy
on held-out windows where the true length is among the candidates (scripts/train_dur_chooser.py).

``DurChooser.apply(placed, frame, ...)``: the notes' lengths replaced by the chooser's picks
(placed_notes calls it, src/labeller.py, when given a chooser and the window's frame).
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .notelabels import bar_quarters

K = 4
FILE = "dur_chooser.pt"
CLIP = 4.0
EPS = 1e-6
FEATURES = ("lp", "lp_gap", "rank0", "rank1", "rank2", "rank3", "log_d",
            "has_next_s", "ratio_s", "eq_s", "lt_s", "gt_s",
            "has_next_a", "ratio_a", "eq_a",
            "has_perf", "perf_ratio", "perf_gap",
            "end_quarter", "end_bar", "past_bar",
            "has_chord", "chord_agree")
N_FEATURES = len(FEATURES)


def _log2(x: np.ndarray) -> np.ndarray:
    return np.clip(np.log2(np.maximum(x, EPS)), -CLIP, CLIP)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def candidates(dur_logits: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """[n, n_dur] logits -> (classes [n, k], log-probabilities [n, k]), likeliest first."""
    lp = log_softmax(np.asarray(dur_logits, np.float64))
    k = min(k, lp.shape[1])
    idx = np.argsort(-lp, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(lp, idx, 1)


def features(placed: list, frame, dur_logits: np.ndarray, onset_s: np.ndarray, dur_s: np.ndarray,
             space, k: int = K) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each placed note that is not a grace note: (its index in ``placed`` [m], candidate classes
    [m, k], features [m, k, N_FEATURES]). ``dur_logits`` / ``onset_s`` / ``dur_s``: per note of the
    item (``Placed.src``), the dur head's logits and the performed onset and duration in seconds.
    ``frame``: the window's bars (src/notelabels.py Window)."""
    use = [j for j, p in enumerate(placed) if not p.grace and 0 <= p.src < len(dur_logits)]
    m = len(use)
    if m == 0:
        return np.zeros(0, int), np.zeros((0, k), int), np.zeros((0, k, N_FEATURES), np.float32)
    lens = [float(bar_quarters(ts)) for ts, _ in frame.bars] or [4.0]
    starts = np.concatenate([[0.0], np.cumsum(lens)])
    bar = np.array([min(max(placed[j].bar, 0), len(lens) - 1) for j in use])
    A = starts[bar] + np.array([float(placed[j].pos) for j in use])
    staff = np.array([placed[j].staff for j in use])
    src = np.array([placed[j].src for j in use])
    cls, lp = candidates(dur_logits[src], k)
    kk = cls.shape[1]
    pick = cls[:, 0]
    later = A[None, :] > A[:, None] + EPS                         # [i, j]: j starts after i
    same = later & (staff[None, :] == staff[:, None])
    big = 1e9
    nxt_s = np.where(same, A[None, :], big).min(1)
    nxt_a = np.where(later, A[None, :], big).min(1)
    has_s, has_a = nxt_s < big, nxt_a < big
    ioi_s, ioi_a = np.where(has_s, nxt_s - A, 1.0), np.where(has_a, nxt_a - A, 1.0)
    # the performed gap to the staff's next onset: its earliest-played note
    cand_j = np.where(same & (np.abs(A[None, :] - nxt_s[:, None]) < EPS), onset_s[src][None, :], big)
    perf_next = cand_j.min(1)
    perf_ioi = perf_next - onset_s[src]
    has_perf = has_s & (perf_ioi > 0.02) & (perf_next < big)
    perf_ratio = np.where(has_perf, _log2(dur_s[src] / np.where(has_perf, perf_ioi, 1.0)), 0.0)
    chord = (np.abs(A[None, :] - A[:, None]) < EPS) & (staff[None, :] == staff[:, None])
    np.fill_diagonal(chord, False)
    n_chord = chord.sum(1)
    X = np.zeros((m, kk, N_FEATURES), np.float32)
    vals = np.array([[float(space.dur_value(c)) for c in row] for row in cls])
    bar_end = starts[bar + 1]
    for r in range(kk):
        d = vals[:, r]
        rs, ra = _log2(d / ioi_s), _log2(d / ioi_a)
        end = A + d
        X[:, r, 0] = lp[:, r]
        X[:, r, 1] = lp[:, r] - lp[:, 0]
        X[:, r, 2 + r] = 1.0
        X[:, r, 6] = _log2(d)
        X[:, r, 7] = has_s
        X[:, r, 8] = np.where(has_s, rs, 0.0)
        X[:, r, 9] = has_s & (np.abs(d - ioi_s) < EPS)
        X[:, r, 10] = has_s & (d < ioi_s - EPS)
        X[:, r, 11] = has_s & (d > ioi_s + EPS)
        X[:, r, 12] = has_a
        X[:, r, 13] = np.where(has_a, ra, 0.0)
        X[:, r, 14] = has_a & (np.abs(d - ioi_a) < EPS)
        X[:, r, 15] = has_perf
        X[:, r, 16] = perf_ratio
        X[:, r, 17] = np.where(has_perf, np.clip(rs - perf_ratio, -CLIP, CLIP), 0.0)
        X[:, r, 18] = np.abs(end - np.round(end)) < EPS
        X[:, r, 19] = np.abs(end - bar_end) < EPS
        X[:, r, 20] = end > bar_end + EPS
        X[:, r, 21] = n_chord > 0
        agree = (chord & (pick[None, :] == cls[:, r][:, None])).sum(1)
        X[:, r, 22] = np.where(n_chord > 0, agree / np.maximum(n_chord, 1), 0.0)
    return np.asarray(use), cls, X


class DurChooser(nn.Module):
    """Candidate scorer: features [..., k, F] -> scores [..., k] (softmax over k by the caller)."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64):
        super().__init__()
        self.n_features, self.hidden = n_features, hidden
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[..., 0]

    @torch.no_grad()
    def choose(self, X: np.ndarray) -> np.ndarray:
        """[m, k, F] -> the chosen candidate per note [m]."""
        if len(X) == 0:
            return np.zeros(0, int)
        p = next(self.parameters())
        s = self(torch.from_numpy(np.asarray(X, np.float32)).to(p.device))
        return s.argmax(-1).cpu().numpy()

    def apply(self, placed: list, frame, dur_logits: np.ndarray, onset_s: np.ndarray, dur_s: np.ndarray,
              space, keep: np.ndarray | None = None) -> int:
        """Replace the placed notes' lengths by the chooser's picks; ``keep`` [n] (per item note):
        notes whose length another step set already (the beat decoding) are left alone. Returns how
        many lengths changed."""
        use, cls, X = features(placed, frame, dur_logits, onset_s, dur_s, space)
        if len(use) == 0:
            return 0
        best = self.choose(X)
        changed = 0
        for j, c in zip(use, cls[np.arange(len(use)), best]):
            p = placed[j]
            if keep is not None and keep[p.src]:
                continue
            v = space.dur_value(int(c))
            if v != p.total:
                p.total = Fraction(v)
                changed += 1
        return changed

    def save(self, path, meta: dict | None = None) -> None:
        torch.save({"state": self.state_dict(), "n_features": self.n_features, "hidden": self.hidden,
                    "features": FEATURES, "k": K, "meta": meta or {}}, path)

    @classmethod
    def load(cls, path, device="cpu") -> "DurChooser | None":
        """The chooser saved next to a labeller checkpoint (``path`` a file or its folder), or None."""
        path = Path(path)
        if path.is_dir():
            path = path / FILE
        if not path.exists():
            return None
        ck = torch.load(path, map_location=device, weights_only=False)
        if tuple(ck.get("features", ())) != FEATURES:
            raise ValueError(f"{path}: made with other features ({len(ck.get('features', ()))}); retrain it")
        m = cls(ck["n_features"], ck["hidden"])
        m.load_state_dict(ck["state"])
        m.meta = ck.get("meta", {})
        return m.to(device).eval()


def train_chooser(X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray, epochs: int = 30,
                  lr: float = 2e-3, batch: int = 4096, seed: int = 0, device="cpu", log=print) -> tuple:
    """Fit on candidates ``X`` [n, k, F] with the true candidate ``y`` [n]; keep the epoch with the
    best accuracy on (Xv, yv). Returns (model, {"val_acc", "val_acc_top1", "epoch"})."""
    torch.manual_seed(seed)
    model = DurChooser().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64))
    Xvt = torch.from_numpy(Xv.astype(np.float32)).to(device)
    base = float((yv == 0).mean()) if len(yv) else 0.0
    best, best_state, best_ep = -1.0, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(Xt))
        for lo in range(0, len(order), batch):
            idx = torch.from_numpy(order[lo:lo + batch])
            loss = nn.functional.cross_entropy(model(Xt[idx].to(device)), yt[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(Xvt).argmax(-1).cpu().numpy() == yv).mean()) if len(yv) else 0.0
        if acc > best:
            best, best_ep = acc, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"epoch {ep}: val candidate accuracy {acc:.4f} (labeller's likeliest {base:.4f})")
    model.load_state_dict(best_state)
    return model.eval(), {"val_acc": best, "val_acc_top1": base, "epoch": best_ep}
