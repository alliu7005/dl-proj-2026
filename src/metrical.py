"""Per-note metrical labels for the auxiliary training target: where in the bar each performed
note falls, obtained by aligning the window's performed notes to its score (the target tokens).

The decoder has to place every note at an exact position in the bar, and its biggest failure is
the bar phase: rhythms that are right but shifted (onset-shift F1 ~0.4 vs onset F1 ~0.2). An
auxiliary head on the encoder (``--aux-metrical`` in scripts/train_model.py) is trained to predict,
for every performed note of the window,
  downbeat  1 if the note starts a bar (position 0), else 0
  phase     its position within the beat: 0 on a quarter-note beat, 1 an eighth after one,
            2 a triplet position (1/3, 2/3), 3 a sixteenth (1/4, 3/4), 4 anything finer
so the encoder has to work out the meter and bar phase from the timing, which is exactly what the
decoder needs. Notes without a reliable label get -1 (ignored): carried-over and context notes,
grace notes, ornaments the score writes as one note, and notes the alignment cannot place.

Alignment (``align``): score notes that start a sound (not tie continuations, not grace notes) get
a position q in quarter notes from the window's first bar line. Written and performed onsets are
nearly one-to-one, so a first time map pairs them by rank, which follows any rubato without a
tempo model; notes are then matched per pitch with an order-preserving alignment (never crossing
repeated notes) within half the local quarter note, and the map is refitted through the matches
(piecewise linear) for two more passes.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .preprocessing.detokenize import parse_tokens

N_PHASE = 5
IGNORE = -1
_PHASE = {Fraction(0): 0, Fraction(1, 2): 1, Fraction(1, 3): 2, Fraction(2, 3): 2, Fraction(1, 4): 3,
          Fraction(3, 4): 3}


def phase_class(pos: Fraction) -> int:
    """Position within the quarter-note beat -> 0 beat, 1 eighth, 2 triplet, 3 sixteenth, 4 finer."""
    return _PHASE.get(pos - (pos.numerator // pos.denominator), 4)


def _bar_quarters(ts: str) -> Fraction:
    try:
        n, d = ts.split("/")
        return Fraction(int(n) * 4, int(d))
    except (ValueError, ZeroDivisionError):
        return Fraction(4)


def score_onsets(tokens: list[str]) -> list[tuple[float, int, int, int]]:
    """Written notes that start a sound: (q quarters from the first bar line, pitch, downbeat, phase)."""
    out, offset = [], Fraction(0)
    for b in parse_tokens(tokens):
        for n in b.notes:
            if not n.tie and not n.grace:
                out.append((float(offset + n.pos), n.pitch, int(n.pos == 0), phase_class(n.pos)))
        offset += _bar_quarters(b.ts)
    return out


def _match(perf_t, perf_p, score_t, score_p, tol):
    """Per pitch, greedily pair performed and predicted times nearest-first within ``tol``.
    Returns {perf index: score index}."""
    pairs = {}
    for p in np.unique(perf_p):
        pi = np.nonzero(perf_p == p)[0]
        si = np.nonzero(score_p == p)[0]
        if len(si) == 0:
            continue
        d = np.abs(perf_t[pi][:, None] - score_t[si][None, :])
        order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
        used_p, used_s = set(), set()
        for a, b in order:
            if d[a, b] > tol:
                break
            if a in used_p or b in used_s:
                continue
            used_p.add(a)
            used_s.add(b)
            pairs[int(pi[a])] = int(si[b])
    return pairs


def _local_tol(sq, pred, frac):
    """Matching tolerance per written note: ``frac`` of the local quarter-note duration of the
    time map (from its slope around the note), at least 30 ms."""
    order = np.argsort(sq, kind="stable")
    q, t = sq[order], pred[order]
    uq, inv = np.unique(q, return_inverse=True)
    ut = np.array([t[inv == k].mean() for k in range(len(uq))])
    if len(uq) < 2:
        spq = np.full(len(sq), 0.5)
    else:
        g = np.gradient(ut, uq)
        g = np.clip(np.where(np.isfinite(g) & (g > 0), g, np.median(g[g > 0]) if (g > 0).any() else 0.5), 0.05, 5.0)
        spq = np.empty(len(sq))
        spq[order] = g[inv]
    return np.maximum(frac * spq, 0.03)


def _match_tol(perf_t, perf_p, score_t, score_p, tol):
    """Per pitch, an order-preserving alignment of performed and written notes (an edit-distance
    dynamic program): pairing costs |time difference| / tol, skipping a note costs 1, and pairs
    further apart than ``tol`` are not allowed. Unlike nearest-first matching it never crosses two
    repeated notes of one pitch, so a generous tolerance is safe. ``tol`` is per written note.
    Returns {perf index: score index}."""
    pairs = {}
    for p in np.unique(perf_p):
        pi = np.nonzero(perf_p == p)[0]
        si = np.nonzero(score_p == p)[0]
        if len(si) == 0:
            continue
        pi = pi[np.argsort(perf_t[pi], kind="stable")]
        si = si[np.argsort(score_t[si], kind="stable")]
        a, b = perf_t[pi], score_t[si]
        n, m = len(a), len(b)
        cost = np.abs(a[:, None] - b[None, :]) / tol[si][None, :]
        dp = np.zeros((n + 1, m + 1))
        dp[:, 0] = np.arange(n + 1)
        dp[0, :] = np.arange(m + 1)
        move = np.zeros((n + 1, m + 1), np.int8)  # 0 pair, 1 skip performed, 2 skip written
        move[1:, 0], move[0, 1:] = 1, 2
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                c = cost[i - 1, j - 1]
                best, mv = dp[i - 1, j] + 1, 1
                if dp[i, j - 1] + 1 < best:
                    best, mv = dp[i, j - 1] + 1, 2
                if c <= 1 and dp[i - 1, j - 1] + c < best:
                    best, mv = dp[i - 1, j - 1] + c, 0
                dp[i, j], move[i, j] = best, mv
        i, j = n, m
        while i > 0 and j > 0:
            if move[i, j] == 0:
                pairs[int(pi[i - 1])] = int(si[j - 1])
                i, j = i - 1, j - 1
            elif move[i, j] == 1:
                i -= 1
            else:
                j -= 1
    return pairs


def _map_through(pairs, sq, pt):
    """Piecewise-linear, non-decreasing time map q -> t through matched (written, performed)
    pairs, extended at both ends with the overall slope. Returns predicted t for every q in ``sq``."""
    mq = np.array([sq[s] for s in pairs.values()])
    mt = np.array([pt[k] for k in pairs])
    uq, inv = np.unique(mq, return_inverse=True)
    ut = np.maximum.accumulate(np.array([mt[inv == k].mean() for k in range(len(uq))]))
    if len(uq) < 2:
        return np.full(len(sq), ut[0])
    slope = max((ut[-1] - ut[0]) / (uq[-1] - uq[0]), 1e-3)
    pred = np.interp(sq, uq, ut)
    lo, hi = sq < uq[0], sq > uq[-1]
    pred[lo] = ut[0] + slope * (sq[lo] - uq[0])
    pred[hi] = ut[-1] + slope * (sq[hi] - uq[-1])
    return pred


def align(onsets: np.ndarray, pitches: np.ndarray, tokens: list[str]):
    """Label performed notes (window time frame; onsets < 0 = carried over, not labelled).
    Returns (downbeat [N] int8, phase [N] int8), IGNORE where unknown.

    Written and performed onsets are nearly one-to-one, so the first time map pairs them by rank
    (the k-th written onset with the performed onset at the same relative rank), which follows any
    tempo change without assuming one; order-preserving matches per pitch within half the local
    quarter note then give a piecewise-linear map, refitted twice."""
    n = len(onsets)
    down = np.full(n, IGNORE, np.int8)
    phase = np.full(n, IGNORE, np.int8)
    score = score_onsets(tokens)
    live = np.nonzero(onsets >= 0)[0]
    if len(score) < 2 or len(live) < 2:
        return down, phase
    sq = np.array([x[0] for x in score])
    sp = np.array([x[1] for x in score])
    pt, pp = onsets[live].astype(np.float64), pitches[live].astype(np.int64)

    # 1. rank map: written onset k <-> performed onset at the same relative rank (chords averaged)
    ts = np.sort(pt)
    order = np.argsort(sq, kind="stable")
    rank = np.empty(len(sq))
    rank[order] = np.arange(len(sq)) * (len(ts) - 1) / max(len(sq) - 1, 1)
    pred = np.interp(rank, np.arange(len(ts)), ts)
    uq, inv = np.unique(sq, return_inverse=True)
    pred = np.array([pred[inv == k].mean() for k in range(len(uq))])[inv]  # one time per chord
    pairs = _match_tol(pt, pp, pred, sp, _local_tol(sq, pred, 1 / 2))
    # 2. refit through the matches, twice (the order-preserving matching keeps half a beat safe)
    for frac in (1 / 2, 1 / 2):
        if len(pairs) < 2:
            break
        pred = _map_through(pairs, sq, pt)
        pairs = _match_tol(pt, pp, pred, sp, _local_tol(sq, pred, frac))
    for k, s_ in pairs.items():
        down[live[k]] = score[s_][2]
        phase[live[k]] = score[s_][3]
    return down, phase


def transfer(labels: tuple[np.ndarray, ...], ref: np.ndarray, notes: np.ndarray, tol: float = 0.08):
    """Carry per-note labels from ``ref`` notes to ``notes`` (e.g. after synthetic noise moved,
    dropped and added notes): each note takes the label of the nearest-onset reference note of the
    same pitch within ``tol`` seconds (each reference note used once); others get IGNORE."""
    out = tuple(np.full(len(notes), IGNORE, np.int8) for _ in labels)
    pairs = _match(notes[:, 1].astype(np.float64), notes[:, 0].astype(np.int64),
                   ref[:, 1].astype(np.float64), ref[:, 0].astype(np.int64), tol)
    for k, r in pairs.items():
        for o, lab in zip(out, labels):
            o[k] = lab[r]
    return out


# ---------------------------------------------------------------------------------------------
# labels for a whole split, stored next to it (built once by scripts/build_metrical.py)
# ---------------------------------------------------------------------------------------------

LABEL_FILE = "metrical.npy"  # int8 [n_notes, 2] (downbeat, phase), row-aligned with notes.npy


def _label_windows(args):
    split_dir, vocab_path, lo, hi = args
    from .preprocessing.store import SegmentStore
    from .preprocessing.vocab import Vocab
    store, vocab = SegmentStore(split_dir, load_meta=False), Vocab.load(vocab_path)
    out = []
    for i in range(lo, hi):
        notes, _, toks = store.get(i)
        d, ph = align(notes[:, 1], notes[:, 0], vocab.decode(toks.tolist()))
        out.append(np.stack([d, ph], axis=1))
    return lo, out


def build_split_labels(split_dir, vocab_path, workers: int = 1, chunk: int = 500) -> np.ndarray:
    """Align every window of a split; writes ``split_dir/metrical.npy`` and returns it."""
    from pathlib import Path

    from .preprocessing.store import SegmentStore
    store = SegmentStore(split_dir, load_meta=False)
    idx = store.index
    labels = np.full((len(store.notes), 2), IGNORE, np.int8)
    jobs = [(str(split_dir), str(vocab_path), lo, min(lo + chunk, len(idx))) for lo in range(0, len(idx), chunk)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_label_windows, jobs))
    else:
        results = [_label_windows(j) for j in jobs]
    for lo, out in results:
        for k, lab in enumerate(out):
            a, b = idx[lo + k, 0], idx[lo + k, 1]
            labels[a:b] = lab
    np.save(Path(split_dir) / LABEL_FILE, labels)
    return labels


def load_split_labels(split_dir, n_notes: int) -> np.ndarray | None:
    """``split_dir/metrical.npy`` (memory-mapped) if it exists and matches the split, else None."""
    from pathlib import Path
    p = Path(split_dir) / LABEL_FILE
    if not p.exists():
        return None
    lab = np.load(p, mmap_mode="r")
    return lab if lab.shape == (n_notes, 2) else None


# ---------------------------------------------------------------------------------------------
# the auxiliary loss and its validation scores
# ---------------------------------------------------------------------------------------------

def aux_loss(aux, downbeat, phase):
    """Auxiliary metrical loss: binary cross-entropy of the downbeat logit plus cross-entropy of
    the phase logits, over notes with a label (-1 = ignored). aux [B,N,1+N_PHASE], labels [B,N]."""
    import torch
    import torch.nn.functional as F
    zero = aux.sum() * 0.0  # keeps the graph when a batch has no labels at all
    d, p = downbeat >= 0, phase >= 0
    l_d = F.binary_cross_entropy_with_logits(aux[..., 0][d].float(), downbeat[d].float()) if d.any() else zero
    l_p = F.cross_entropy(aux[..., 1:][p].float(), phase[p]) if p.any() else zero
    return l_d + l_p


def aux_counts(aux, downbeat, phase) -> dict:
    """Counts for downbeat precision / recall and phase accuracy (summed over batches by the caller)."""
    d, p = downbeat >= 0, phase >= 0
    pred_d = aux[..., 0] > 0
    true_d = downbeat == 1
    return {"db_tp": int((pred_d & true_d & d).sum()), "db_pred": int((pred_d & d).sum()),
            "db_true": int((true_d & d).sum()),
            "ph_ok": int(((aux[..., 1:].argmax(-1) == phase) & p).sum()), "ph_n": int(p.sum())}


def aux_scores(c: dict) -> dict:
    """downbeat_f1, phase_acc from summed ``aux_counts``."""
    prec = c["db_tp"] / c["db_pred"] if c["db_pred"] else 0.0
    rec = c["db_tp"] / c["db_true"] if c["db_true"] else 0.0
    return {"downbeat_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "phase_acc": c["ph_ok"] / c["ph_n"] if c["ph_n"] else 0.0}
