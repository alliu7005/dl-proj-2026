"""Per-note score labels: the note labeller's targets (v2), and rendering labelled notes back into
the target token grammar.

The v2 system writes the score note by note instead of token by token: every PERFORMED note gets
the labels that place it in the score (bar, position in the bar, written duration, staff, grace /
arpeggio / ornament flags), or "delete" (a transcription error, or an extra note of a trill or
tremolo the score writes as one). Rendering turns labelled notes back into tokens, so the same
metrics (src/eval/metrics.py) score both systems.

Labels come from aligning a window's performed notes to its target tokens (``align_notes``): the
note alignment of src/metrical.py (fit_alignment) pairs the notes that start a sound; grace notes,
which it leaves out, are then paired by pitch and time through the same time map. Notes the score
writes as a tie continuation are not performed: a note's label holds the whole tied length, and the
renderer splits it at bar lines into the note and its tied continuations (``render``), the way
scores write a note held across a bar line. Score notes no performed note pairs with (the
alignment missed them, or the pianist did not play them) cannot be produced: ``ceiling`` measures
what that and the tie rule cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .metrical import _map_through, fit_alignment
from .preprocessing.detokenize import parse_tokens
from .preprocessing.score import Measure, NoteEvent, tokenize_bars

GRACE_TOL = 0.25  # seconds: a grace note's performed onset vs its main note's predicted time


def bar_quarters(ts: str) -> Fraction:
    try:
        n, d = ts.split("/")
        return Fraction(int(n) * 4, int(d))
    except (ValueError, ZeroDivisionError):
        return Fraction(4)


@dataclass
class ScoreNote:
    bar: int            # index of the bar in the window's tokens (0 = the first, maybe partial, bar)
    pos: Fraction       # quarters from the bar line
    pitch: int
    dur: Fraction       # written duration (0 for grace notes)
    staff: int
    grace: bool = False
    tie: bool = False   # a tie continuation (not performed)
    arp: bool = False
    orns: tuple = ()
    total: Fraction = Fraction(0)  # written length including its tied continuations (set by tie_totals)


@dataclass
class Window:
    """The score frame of a window's tokens: per bar its time / key signature, whether the first
    bar starts at a bar line, and the position the first bar starts at (and, for a frame cut from a
    piece's estimated grid, which of the piece's bars it starts with)."""
    bars: list = field(default_factory=list)   # [(ts, key)]
    opens_with_bar: bool = True
    start_pos: Fraction | None = Fraction(0)   # None: the renderer takes the first note's position
    first_bar: int = 0                         # window_frame: the grid's bar index of bars[0]


def score_notes(tokens: list[str]) -> tuple[list[ScoreNote], Window]:
    """Every written note of a window's tokens, in token order, and the window's bars."""
    bars = parse_tokens(tokens)
    win = Window(bars=[(b.ts, b.key) for b in bars], opens_with_bar=bool(tokens) and tokens[0] == "bar")
    notes = []
    for j, b in enumerate(bars):
        for n in b.notes:
            notes.append(ScoreNote(j, n.pos, n.pitch, n.dur, n.staff, n.grace, n.tie, n.arp, tuple(sorted(n.orns))))
    if not win.opens_with_bar and notes:
        win.start_pos = min(n.pos for n in notes if n.bar == 0) if any(n.bar == 0 for n in notes) else Fraction(0)
    tie_totals(notes, win)
    return notes, win


def abs_quarters(win: Window, bar: int, pos: Fraction) -> Fraction:
    """Quarters from the window's first bar line (nominal bar lengths, as the metrics count them)."""
    return sum((bar_quarters(win.bars[k][0]) for k in range(bar)), Fraction(0)) + pos


def tie_totals(notes: list[ScoreNote], win: Window) -> None:
    """Set each sounding note's ``total``: its duration plus every tie continuation that follows it
    (same pitch and staff, starting exactly where the previous part ends)."""
    starts = {}
    for k, n in enumerate(notes):
        if n.tie:
            starts.setdefault((n.pitch, n.staff, abs_quarters(win, n.bar, n.pos)), []).append(k)
    for n in notes:
        if n.tie or n.grace:
            n.total = n.dur
            continue
        total, end = n.dur, abs_quarters(win, n.bar, n.pos) + n.dur
        while (n.pitch, n.staff, end) in starts and starts[(n.pitch, n.staff, end)]:
            c = notes[starts[(n.pitch, n.staff, end)].pop(0)]
            total += c.dur
            end += c.dur
        n.total = total


def align_notes(onsets: np.ndarray, pitches: np.ndarray, tokens: list[str]):
    """(score notes, window, match {performed index: score note index}). Performed notes with
    onsets < 0 (carried over from before the window) are never matched."""
    notes, win = score_notes(tokens)
    match = {}
    fit = fit_alignment(np.asarray(onsets, np.float64), np.asarray(pitches), tokens)
    if fit is None:
        return notes, win, match
    score, sq, live, pairs, pt = fit
    # fit_alignment's score list = the sounding, non-grace notes in token order
    sounding = [k for k, n in enumerate(notes) if not n.tie and not n.grace]
    assert len(sounding) == len(score), (len(sounding), len(score))
    for li, si in pairs.items():
        match[int(live[li])] = sounding[si]
    # grace notes: same pitch, performed near (before) their main note's predicted time
    graces = [k for k, n in enumerate(notes) if n.grace]
    if graces and len(pairs) >= 2:
        gq = np.array([float(abs_quarters(win, notes[k].bar, notes[k].pos)) for k in graces])
        gt = _map_through(pairs, np.concatenate([sq, gq]), pt)[len(sq):]
        free = [i for i in range(len(onsets)) if onsets[i] >= 0 and i not in match]
        used = set()
        for k, t in sorted(zip(graces, gt), key=lambda x: x[1]):
            cands = [i for i in free if i not in used and int(pitches[i]) == notes[k].pitch
                     and abs(float(onsets[i]) - t) <= GRACE_TOL]
            if cands:
                i = min(cands, key=lambda i: abs(float(onsets[i]) - t))
                used.add(i)
                match[i] = k
    return notes, win, match


@dataclass
class Placed:
    """A note as the labeller writes it: where (bar of the window, position), what (pitch, written
    length including ties, staff), and its flags."""
    bar: int
    pos: Fraction
    pitch: int
    total: Fraction
    staff: int
    grace: bool = False
    arp: bool = False
    orns: tuple = ()


def render(win: Window, placed: list[Placed]) -> list[str]:
    """Labelled notes -> tokens. A note longer than the room left in its bar is cut at the bar line
    and continued in the next bar(s) as tied notes (``tie``), as scores write held notes, up to the
    window's last bar; a note placed past the window's bars gets bars of its last time / key
    signature."""
    n_bars = max([len(win.bars)] + [p.bar + 1 for p in placed])
    bars = list(win.bars) + [win.bars[-1] if win.bars else ("4/4", 0)] * (n_bars - len(win.bars))
    measures = [Measure(length=bar_quarters(ts), time_signature=ts, key=key) for ts, key in bars]
    seq = 0
    for p in placed:
        if p.grace:
            measures[p.bar].events.append(NoteEvent(p.pos, p.staff, p.pitch, Fraction(0), grace=True, seq=seq))
            seq += 1
            continue
        bar, pos, left, first = p.bar, p.pos, p.total, True
        while left > 0 and bar < len(measures):  # continuations stop at the window's last bar
            room = measures[bar].length - pos
            part = min(left, room) if room > 0 else left
            measures[bar].events.append(NoteEvent(pos, p.staff, p.pitch, part, tie=not first,
                                                  arpeggio=p.arp and first, ornaments=p.orns if first else ()))
            left -= part
            bar, pos, first = bar + 1, Fraction(0), False
    sp = win.start_pos
    if not win.opens_with_bar and sp is None:  # an estimated frame: the first labelled note's position
        first = [p.pos for p in placed if p.bar == 0]
        sp = min(first) if first else Fraction(0)
    start = float(sp) if not win.opens_with_bar and sp and sp > 0 else None
    return tokenize_bars(measures, [[j] for j in range(len(measures))], start_pos=start)


def oracle_placed(notes: list[ScoreNote], match: dict) -> list[Placed]:
    """The labels of the matched performed notes, exactly as the score has them."""
    return [Placed(notes[k].bar, notes[k].pos, notes[k].pitch, notes[k].total, notes[k].staff,
                   notes[k].grace, notes[k].arp, notes[k].orns) for _, k in sorted(match.items())]


# ---------------------------------------------------------------------------------------------
# label classes (the labeller's heads) and their storage per split
# ---------------------------------------------------------------------------------------------

ORN_NAMES = ("fermata", "mordent", "tremolo", "trill", "turn")
FLAG_GRACE, FLAG_ARP = 1, 2                     # flags bitmask; ornament k is bit 2 + k
N_FLAGS = 2 + len(ORN_NAMES)
N_POS = 16                                      # pos_0 .. pos_15: whole quarters from the bar line
LABEL_COLS = ("keep", "bar", "pos_num", "pos_den", "dur_num", "dur_den", "staff", "flags")
LABEL_FILE = "notelabels.npz"
IGNORE = -1


def flags_of(grace: bool, arp: bool, orns: tuple) -> int:
    f = (FLAG_GRACE if grace else 0) | (FLAG_ARP if arp else 0)
    for o in orns:
        if o in ORN_NAMES:
            f |= 1 << (2 + ORN_NAMES.index(o))
    return f


def unflag(f: int) -> tuple[bool, bool, tuple]:
    return bool(f & FLAG_GRACE), bool(f & FLAG_ARP), tuple(o for k, o in enumerate(ORN_NAMES) if f >> (2 + k) & 1)


class LabelSpace:
    """The classes of the position and duration heads, from the target vocabulary: a position is
    ``pos_k`` (k whole quarters) + optional ``sub_x`` (the target grammar's own split), a duration
    is one of the ``d_*`` values (a longer tied total takes the nearest)."""

    def __init__(self, itos: list[str]):
        self.subs = [Fraction(t[4:]) for t in itos if t.startswith("sub_")]
        self.sub_index = {f: k + 1 for k, f in enumerate(self.subs)}   # 0 = on the whole quarter
        self.durs = sorted({Fraction(t[2:]) for t in itos if t.startswith("d_")})
        self.dur_index = {f: k for k, f in enumerate(self.durs)}
        self._dur_values = np.array([float(d) for d in self.durs])

    @property
    def n_sub(self) -> int:
        return len(self.subs) + 1

    @property
    def n_dur(self) -> int:
        return len(self.durs)

    def pos_classes(self, pos: Fraction) -> tuple[int, int]:
        whole = pos.numerator // pos.denominator
        rest = pos - whole
        if whole >= N_POS or (rest and rest not in self.sub_index):
            return IGNORE, IGNORE
        return int(whole), self.sub_index.get(rest, 0)

    def pos_value(self, k: int, sub: int) -> Fraction:
        return Fraction(int(k)) + (self.subs[sub - 1] if sub > 0 else Fraction(0))

    def dur_class(self, d: Fraction) -> int:
        if d in self.dur_index:
            return self.dur_index[d]
        return int(np.argmin(np.abs(self._dur_values - float(d))))

    def dur_value(self, k: int) -> Fraction:
        return self.durs[int(k)]


def window_labels(notes: np.ndarray, tokens: list[str]):
    """One window: (labels int32 [n, len(LABEL_COLS)] row-aligned with its stored notes, grid
    (t [K] seconds, q [K] quarters from the window's first bar line, bar [K]) of the score's beats
    timed through the note alignment, or empty arrays when too few notes align). ``notes`` =
    stored rows (pitch, onset, duration, velocity)."""
    from .beats import beat_layout, window_grid
    lab = np.full((len(notes), len(LABEL_COLS)), IGNORE, np.int32)
    on = notes[:, 1].astype(np.float64)
    lab[on >= 0, 0] = 0                                    # performed in the window: delete unless paired
    snotes, win, match = align_notes(on, notes[:, 0], tokens)
    for i, k in match.items():
        s = snotes[k]
        total = Fraction(0) if s.grace else s.total
        lab[i] = (1, s.bar, s.pos.numerator, s.pos.denominator, total.numerator, total.denominator, s.staff,
                  flags_of(s.grace, s.arp, s.orns))
    g = window_grid(on, notes[:, 0], tokens)
    if g is None or len(g["t"]) < 2:
        return lab, (np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.int16))
    q = np.array([float(abs_quarters(win, int(b), Fraction(0))) + k * beat_layout(win.bars[int(b)][0])[1]
                  for b, k in zip(g["bar"], g["beat"])])
    return lab, (g["t"].astype(np.float32), q.astype(np.float32), g["bar"].astype(np.int16))


def _label_chunk(args):
    split_dir, vocab_path, lo, hi = args
    from .preprocessing.store import SegmentStore
    from .preprocessing.vocab import Vocab
    store, vocab = SegmentStore(split_dir, load_meta=False), Vocab.load(vocab_path)
    out = []
    for i in range(lo, hi):
        notes, _, toks = store.get(i)
        out.append(window_labels(notes, vocab.decode(toks.tolist())))
    return lo, out


def build_split(split_dir, vocab_path, workers: int = 1, chunk: int = 400) -> dict:
    """Labels and grids of every window of a split -> ``split_dir/notelabels.npz``; returns it."""
    from pathlib import Path

    from .preprocessing.store import SegmentStore
    store = SegmentStore(split_dir, load_meta=False)
    idx = store.index
    jobs = [(str(split_dir), str(vocab_path), lo, min(lo + chunk, len(idx))) for lo in range(0, len(idx), chunk)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_label_chunk, jobs))
    else:
        results = [_label_chunk(j) for j in jobs]
    labels = np.full((len(store.notes), len(LABEL_COLS)), IGNORE, np.int32)
    gt, gq, gb, gidx = [], [], [], np.zeros((len(idx), 2), np.int64)
    n = 0
    for lo, out in sorted(results, key=lambda r: r[0]):
        for k, (lab, (t, q, b)) in enumerate(out):
            i = lo + k
            labels[idx[i, 0]:idx[i, 1]] = lab
            gidx[i] = (n, n + len(t))
            gt.append(t)
            gq.append(q)
            gb.append(b)
            n += len(t)
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)  # noqa: E731
    out = {"labels": labels, "grid_t": cat(gt, np.float32), "grid_q": cat(gq, np.float32),
           "grid_bar": cat(gb, np.int16), "grid_index": gidx}
    np.savez(Path(split_dir) / LABEL_FILE, **out)
    return out


def load_split(split_dir, n_notes: int, n_windows: int) -> dict | None:
    """``split_dir/notelabels.npz`` if it exists and matches the split, else None."""
    from pathlib import Path
    p = Path(split_dir) / LABEL_FILE
    if not p.exists():
        return None
    d = dict(np.load(p))
    ok = d["labels"].shape == (n_notes, len(LABEL_COLS)) and d["grid_index"].shape == (n_windows, 2)
    return d if ok else None


# ---------------------------------------------------------------------------------------------
# where the grid puts each note: the labeller's grid features
# ---------------------------------------------------------------------------------------------

FOURIER_PERIODS = (0.25, 1 / 3, 0.5, 1.0, 2.0)   # quarters
N_GRID_FEATURES = 2 * len(FOURIER_PERIODS) + 2 + 5


def grid_position(onsets: np.ndarray, grid_t: np.ndarray, grid_q: np.ndarray):
    """Each note's position in quarters from the window's first bar line, read off the grid:
    interpolated between the beats around it, extended with the end slopes outside them. Returns
    (q [N], seconds per quarter at each note [N]); None without a usable grid."""
    if len(grid_t) < 2:
        return None
    order = np.argsort(grid_t, kind="stable")
    t, q = np.asarray(grid_t, np.float64)[order], np.asarray(grid_q, np.float64)[order]
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    t, q = t[keep], q[keep]
    if len(t) < 2:
        return None
    on = np.asarray(onsets, np.float64)
    qn = np.interp(on, t, q)
    s0 = (q[1] - q[0]) / (t[1] - t[0])
    s1 = (q[-1] - q[-2]) / (t[-1] - t[-2])
    qn = np.where(on < t[0], q[0] + (on - t[0]) * s0, qn)
    qn = np.where(on > t[-1], q[-1] + (on - t[-1]) * s1, qn)
    slope = np.gradient(q, t) if len(t) > 2 else np.full(len(t), s0)
    spq = 1.0 / np.clip(np.interp(on, t, slope), 1e-3, None)
    return qn, spq


def bar_of(q: np.ndarray, win: Window) -> tuple[np.ndarray, np.ndarray]:
    """(bar index, quarters from that bar line) of positions in quarters from the window's first
    bar line, with the window's bar lengths (the last one repeated past its end; a position
    before the first bar line counts in bar 0)."""
    lengths = [float(bar_quarters(ts)) for ts, _ in win.bars] or [4.0]
    starts = np.concatenate([[0.0], np.cumsum(lengths)])
    q = np.asarray(q, np.float64)
    b = np.clip(np.searchsorted(starts, q, side="right") - 1, 0, None)
    last = len(lengths) - 1
    past = b > last
    extra = np.floor((q - starts[last + 1]) / lengths[last]).astype(int) + 1
    bar = np.where(past, last + extra, b).astype(np.int64)
    start = np.where(past, starts[last + 1] + (extra - 1) * lengths[last], starts[np.minimum(b, last)])
    return bar, q - start


def grid_features(onsets: np.ndarray, grid_t: np.ndarray, grid_q: np.ndarray, win: Window):
    """Per note: (features float32 [N, N_GRID_FEATURES], bar estimate [N] int, position estimate
    [N] quarters from that bar line). Features: sin / cos of the grid position at FOURIER_PERIODS,
    sin / cos of the position in the bar, the position in the bar / 4, the bar length / 4, log2
    seconds per quarter, the grid's beat length / 4 (quarters per beat), and "has a grid"; all 0
    without a grid (bar and position estimates 0)."""
    from .beats import beat_layout
    n = len(onsets)
    feats = np.zeros((n, N_GRID_FEATURES), np.float32)
    gp = grid_position(onsets, grid_t, grid_q)
    if gp is None or n == 0:
        return feats, np.zeros(n, np.int64), np.zeros(n)
    q, spq = gp
    bar, pos = bar_of(q, win)
    ts_of = [win.bars[min(int(b), len(win.bars) - 1)][0] if win.bars else "4/4" for b in bar]
    lengths = np.array([float(bar_quarters(ts)) for ts in ts_of])
    qpb = np.array([beat_layout(ts)[1] for ts in ts_of])
    cols = []
    for p in FOURIER_PERIODS:
        cols += [np.sin(2 * np.pi * q / p), np.cos(2 * np.pi * q / p)]
    cols += [np.sin(2 * np.pi * pos / lengths), np.cos(2 * np.pi * pos / lengths)]
    cols += [pos / 4.0, lengths / 4.0, np.clip(np.log2(spq / 0.5), -3, 3), qpb / 4.0, np.ones(n)]
    feats[:] = np.stack(cols, axis=1)
    return feats, bar, pos


EXTRA_SNAP = (1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4)  # where in its beat a spurious beat's q is put


class GridNoise:
    """The beat tracker's timing on a window's TRUE grid, for training the labeller (``p`` of the
    windows): the grid the real system gives it is anchored on tracked beats, not on the beats as
    the notes were played (the true grid, timed through the note alignment). Same bars, same
    positions, so the labels stay valid; only the grid's anchors move: the fitted Beat This! errors
    of src/beats.py BeatNoise (``configs/beat_noise.json``), one performance profile per window:
      tempo octave   p_double: k - 1 more anchors evenly into every beat (their q exact, as the meter
                     model's 1/k steps give them); p_half: every k-th anchor only
      miss           anchors dropped (the grid interpolates across them)
      extra          spurious anchors between two beats at the fitted phases, their q snapped to the
                     nearest EXTRA_SNAP of the beat (the meter model's step): off by the tracker's
                     phase error, so the grid bends locally
      offset/jitter  every anchor's time moved by offset + N(0, jitter)
    The bar and time-signature errors of a meter model are not simulated (the labels would no
    longer hold): those are measured by the meter grids themselves."""

    def __init__(self, beat_noise, p: float = 0.8):
        self.bn, self.p = beat_noise, float(p)

    @classmethod
    def from_json(cls, path, p: float = 0.8) -> "GridNoise":
        from .beats import BeatNoise
        return cls(BeatNoise.from_json(path), p)

    def __call__(self, t: np.ndarray, q: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        t, q = np.asarray(t, np.float64).copy(), np.asarray(q, np.float64).copy()
        if len(t) < 3 or self.p <= 0 or rng.random() >= self.p:
            return t, q
        order = np.argsort(t, kind="stable")
        t, q = t[order], q[order]
        bn = self.bn
        r = bn.rates(rng)
        u = rng.random()
        if u < bn.p_double:
            k = bn._factor(bn.double_k, rng)
            frac = np.arange(1, k) / k
            t = np.concatenate([t, (t[:-1, None] + frac * np.diff(t)[:, None]).ravel()])
            q = np.concatenate([q, (q[:-1, None] + frac * np.diff(q)[:, None]).ravel()])
        elif u < bn.p_double + bn.p_half:
            k = bn._factor(bn.half_k, rng)
            keep = (np.arange(len(t)) - int(rng.integers(k))) % k == 0
            if keep.sum() >= 2:
                t, q = t[keep], q[keep]
        order = np.argsort(t, kind="stable")
        t, q = t[order], q[order]
        if r["miss"] > 0 and len(t) > 2:
            keep = rng.random(len(t)) >= r["miss"]
            keep[[0, -1]] = True  # the window stays covered
            t, q = t[keep], q[keep]
        if r["extra"] > 0 and len(t) >= 2:
            e = min(r["extra"], 0.9)
            ratio = e / (1 - e)   # spurious beats per real one; above 1 (extra > 0.5: PM2S's profiles reach
            # 0.73) no binomial can draw it: a Poisson with the same mean, as src/beats.py BeatNoise does
            n = int(rng.binomial(len(t), ratio)) if ratio <= 1 else int(rng.poisson(len(t) * ratio))
            if n:
                j = rng.integers(len(t) - 1, size=n)
                if bn.extra_phase:
                    pp = np.asarray(bn.extra_phase, np.float64)
                    b = rng.choice(len(pp), size=n, p=pp / pp.sum())
                    phase = (b + rng.random(n)) / len(pp)
                else:
                    phase = rng.random(n)
                snap = np.asarray(EXTRA_SNAP)[np.abs(phase[:, None] - np.asarray(EXTRA_SNAP)[None]).argmin(1)]
                t = np.concatenate([t, t[j] + phase * (t[j + 1] - t[j])])
                q = np.concatenate([q, q[j] + snap * (q[j + 1] - q[j])])
        t = t + r["offset"] + (rng.normal(0, r["jitter"], len(t)) if r["jitter"] > 0 else 0.0)
        order = np.lexsort((q, t))
        t, q = t[order], q[order]
        # a grid's anchors increase in time and position: drop any the jitter put out of order
        keep = np.ones(len(t), bool)
        last_t, last_q = t[0], q[0]
        for i in range(1, len(t)):
            if t[i] - last_t > 0.03 and q[i] > last_q + 1e-9:
                last_t, last_q = t[i], q[i]
            else:
                keep[i] = False
        return t[keep], q[keep]


# ---------------------------------------------------------------------------------------------
# training targets of one item (src/dataset.py, notelabels=True)
# ---------------------------------------------------------------------------------------------

TARGET_COLS = ("keep", "delta", "pos", "sub", "dur", "staff")  # delta: bar correction + 1 (0, 1, 2)


def transfer_rows(rows: np.ndarray, ref: np.ndarray, notes: np.ndarray, tol: float = 0.08) -> np.ndarray:
    """Carry label rows from ``ref`` notes to ``notes`` after synthetic noise moved, dropped and
    added notes: each note takes the row of the nearest-onset reference note of its pitch within
    ``tol`` seconds (each used once); a note the noise added is a transcription error to delete
    (keep 0, nothing else known)."""
    from .metrical import _match
    out = np.full((len(notes), rows.shape[1]), IGNORE, np.int32)
    pairs = _match(notes[:, 1].astype(np.float64), notes[:, 0].astype(np.int64),
                   ref[:, 1].astype(np.float64), ref[:, 0].astype(np.int64), tol)
    for k, r in pairs.items():
        out[k] = rows[r]
    added = np.array([k not in pairs for k in range(len(notes))], bool) & (notes[:, 1] >= 0)
    out[added, 0] = 0
    return out


def targets(rows: np.ndarray, bar_est: np.ndarray, space: LabelSpace) -> tuple[np.ndarray, np.ndarray]:
    """Label rows (LABEL_COLS) -> (targets int64 [N, len(TARGET_COLS)], flags int64 [N, N_FLAGS]),
    IGNORE where a head has nothing to learn: everything but keep for a note to delete or of
    unknown keep, the bar correction when the grid's bar is more than one off, the position when
    it is not a pos_k + sub_x of the vocabulary."""
    n = len(rows)
    tg = np.full((n, len(TARGET_COLS)), IGNORE, np.int64)
    fl = np.full((n, N_FLAGS), IGNORE, np.int64)
    tg[:, 0] = rows[:, 0]
    for i in np.nonzero(rows[:, 0] == 1)[0]:
        r = rows[i]
        d = int(r[1]) - int(bar_est[i])
        if abs(d) <= 1:
            tg[i, 1] = d + 1
        k, sub = space.pos_classes(Fraction(int(r[2]), int(r[3])))
        tg[i, 2], tg[i, 3] = k, sub
        tg[i, 4] = space.dur_class(Fraction(int(r[4]), int(r[5])))
        tg[i, 5] = r[6]
        fl[i] = [(int(r[7]) >> b) & 1 for b in range(N_FLAGS)]
    return tg, fl


# ---------------------------------------------------------------------------------------------
# an estimated grid (src/meter.py) in a window's frame, and the key signature from the notes
# ---------------------------------------------------------------------------------------------

# Krumhansl-Kessler key profiles (C major / C minor), the classic tonal-hierarchy weights
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}  # tonic pc -> sharps


def estimate_key(notes: np.ndarray) -> int:
    """Key signature (sharps > 0, flats < 0) of a set of notes: the major or minor key whose
    Krumhansl-Kessler profile correlates best with the duration-weighted pitch-class profile (a
    minor key takes its relative major's signature). ``notes``: pitch, onset, offset or
    duration, ... (only pitch and the third column's length are used)."""
    if len(notes) == 0:
        return 0
    w = np.clip(np.abs(notes[:, 2] - notes[:, 1]) if (notes[:, 2] > notes[:, 1]).mean() > 0.5 else notes[:, 2], 0.05, 4.0)
    prof = np.bincount(notes[:, 0].astype(int) % 12, weights=w, minlength=12)
    best, key = -np.inf, 0
    for tonic in range(12):
        for profile, minor in ((_KK_MAJOR, False), (_KK_MINOR, True)):
            r = np.corrcoef(prof, np.roll(profile, tonic))[0, 1]
            if r > best:
                best, key = r, _FIFTHS[(tonic + 3) % 12] if minor else _FIFTHS[tonic]
    return int(np.clip(key, -7, 7))


def window_frame(grid: dict, perf_t0: float, onsets: np.ndarray, key: int):
    """A piece's estimated grid (src/meter.py grid_from_states, absolute seconds) in one window's
    frame: (grid_t [K] window seconds, grid_q [K] quarters from the window's first bar line,
    Window). The window's bars run from the bar the grid puts its first note in to the bar it
    puts its last note in (``onsets``: its own notes, window seconds; carried-over ones, < 0,
    aside), with the grid's time signatures and ``key`` (a note labelled into a later bar gets bars
    of the last one, as in ``render``); it opens with a bar line when its first note sits on it
    (within 1/24 quarter), else its start position is left to the renderer (the first labelled
    note's)."""
    t = np.asarray(grid["t"], np.float64) - perf_t0
    q = np.asarray(grid["q"], np.float64)
    if len(t) < 2:
        return t, q, Window(bars=[("4/4", key)])
    live = np.asarray(onsets, np.float64)
    live = live[live >= 0]
    first = float(live.min()) if len(live) else 0.0
    last = float(live.max()) if len(live) else 0.0
    qf, ql = grid_position(np.array([first, last]), t, q)[0]
    bq = np.asarray(grid["bar_q"], np.float64)
    b0 = max(0, int(np.searchsorted(bq, qf + 1e-6, side="right")) - 1)
    b1 = max(b0, int(np.searchsorted(bq, ql + 1e-6, side="right")) - 1)
    bars = [(ts, key) for ts in grid["bars"][b0:b1 + 1]] or [(grid["bars"][-1], key)]
    win = Window(bars=bars, opens_with_bar=(qf - bq[b0]) < 1 / 24, start_pos=None, first_bar=b0)
    return t, q - bq[b0], win
