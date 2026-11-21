"""The meter model (v2, Phase 2): from tracked beats and the performed notes to the score's metrical
grid of a whole piece (notated beats with bar lines and a time signature per bar), the grid the
note labeller (src/labeller.py) reads every note's position from.

A tracker's beats are close to the notated beats but not the same: Beat This! is at double tempo
on 18% of pieces, sometimes at the eighth notes of a compound meter, misses 12% of beats, adds 13%,
and marks downbeats almost independently per beat (src/beats.py). So the meter model works on the
TRACKED beats and predicts, for each:
  bar      does a bar start on this beat
  step     how many quarter notes it lasts, to the next tracked beat (Q_STEPS): absorbs a doubled
           or tripled tempo, a missed or an extra beat
  ts       the time signature of its bar (TS_CLASSES)
from per-beat features of the notes around it (``beat_features``: where notes fall inside the
beat, how loud / low / long the notes on the beat are, the change of harmony across it, the
tracker's own downbeat flag, the beat's length). A transformer over the piece's beat sequence
makes the predictions; ``decode`` then picks the most probable CONSISTENT grid (a Viterbi search
over time signature x position in the bar: bars add up to their time signature, which changes only
at a bar line), and ``grid_from_states`` turns it into (beat time, quarters from the first bar line)
anchors plus the bars' time signatures.

Training targets (``targets``) come from ASAP's annotations (``annotation_grid``): the annotated
beats with their quarter-note positions (beat_layout of each bar's time signature) and downbeats,
mapped onto the tracked beats by time.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np

from .beats import MATCH_TOL, beat_layout

TS_CLASSES = ("4/4", "2/4", "3/4", "6/8", "3/8", "5/8", "9/8", "2/2", "12/16", "12/8", "6/4", "3/2", "4/8",
              "5/4", "4/2")
Q_STEPS = tuple(Fraction(x) for x in ("1/4", "1/3", "1/2", "2/3", "3/4", "1", "4/3", "3/2", "2", "3", "4"))
STEP_TOL = 0.2      # a step within 20% of a Q_STEPS value takes it; further off -> no step target
GRID = 12           # positions in the bar are multiples of 1/12 quarter (16ths and triplets)
IGNORE = -1


def bar_quarters(ts: str) -> Fraction:
    n, d = ts.split("/")
    return Fraction(int(n) * 4, int(d))


# ---------------------------------------------------------------------------------------------
# ground truth from the annotations
# ---------------------------------------------------------------------------------------------

def annotation_grid(a: dict) -> dict:
    """ASAP annotations of one performance -> {"t": beat times, "q": quarters from the first beat,
    "down": bar starts, "ts": TS_CLASSES index per beat (-1: a meter outside the classes),
    "key": key signature per beat}. Each beat lasts its bar's beat length (beat_layout)."""
    t = np.asarray(a["performance_beats"], np.float64)
    downs = np.asarray(a["performance_downbeats"], np.float64)
    down = np.isin(np.round(t, 6), np.round(downs, 6))
    tss = sorted((float(k), v[0]) for k, v in a.get("perf_time_signatures", {}).items())
    keys = sorted((float(k), int(v[0])) for k, v in a.get("perf_key_signatures", {}).items())

    def active(changes, x, default):
        cur = changes[0][1] if changes else default
        for when, v in changes:
            if when <= x + 1e-6:
                cur = v
        return cur
    ts = [active(tss, x, "4/4") for x in t]
    qpb = np.array([beat_layout(s)[1] for s in ts])
    q = np.concatenate([[0.0], np.cumsum(qpb[:-1])])
    return {"t": t, "q": q, "down": down, "ts": np.array([TS_CLASSES.index(s) if s in TS_CLASSES else IGNORE
                                                          for s in ts]),
            "key": np.array([active(keys, x, 0) for x in t]), "ts_names": ts}


def quarters_at(t: np.ndarray, gt: dict) -> np.ndarray:
    """Quarter-note position of any times on the annotated grid (linear between beats, extended
    with the end tempos)."""
    bt, bq = gt["t"], gt["q"]
    out = np.interp(t, bt, bq)
    s0 = (bq[1] - bq[0]) / max(bt[1] - bt[0], 1e-6)
    s1 = (bq[-1] - bq[-2]) / max(bt[-1] - bt[-2], 1e-6)
    out = np.where(t < bt[0], bq[0] + (t - bt[0]) * s0, out)
    return np.where(t > bt[-1], bq[-1] + (t - bt[-1]) * s1, out)


def targets(beat_t: np.ndarray, gt: dict) -> dict:
    """Per tracked beat: bar (1 / 0; -1 outside the annotated span), step (Q_STEPS index; -1 when
    the true length is no Q_STEPS value within STEP_TOL), ts (TS_CLASSES index of the nearest
    annotated beat's bar; -1 outside)."""
    k = len(beat_t)
    inside = (beat_t >= gt["t"][0] - MATCH_TOL) & (beat_t <= gt["t"][-1] + MATCH_TOL)
    bar = np.where(inside, 0, IGNORE)
    for d in gt["t"][gt["down"]]:
        j = int(np.argmin(np.abs(beat_t - d))) if k else -1
        if j >= 0 and abs(beat_t[j] - d) <= MATCH_TOL:
            bar[j] = 1
    q = quarters_at(beat_t, gt)
    steps = np.diff(q) if k > 1 else np.zeros(0)
    steps = np.concatenate([steps, steps[-1:]]) if k > 1 else np.zeros(k)
    vals = np.array([float(s) for s in Q_STEPS])
    rel = np.abs(steps[:, None] - vals[None]) / vals[None]
    step = np.where(rel.min(1) <= STEP_TOL, rel.argmin(1), IGNORE)
    step = np.where(inside, step, IGNORE)
    near = np.clip(np.searchsorted(gt["t"], beat_t), 0, len(gt["t"]) - 1)
    prev = np.clip(near - 1, 0, len(gt["t"]) - 1)
    near = np.where(np.abs(gt["t"][prev] - beat_t) < np.abs(gt["t"][near] - beat_t), prev, near)
    ts = np.where(inside, gt["ts"][near], IGNORE)
    return {"bar": bar.astype(np.int64), "step": step.astype(np.int64), "ts": ts.astype(np.int64)}


# ---------------------------------------------------------------------------------------------
# per-beat features from the notes
# ---------------------------------------------------------------------------------------------

N_PHASE_BINS = 12
AT_BEAT = 0.05      # seconds: a note "on" the beat
N_BEAT_FEATURES = 5 + N_PHASE_BINS + 8 + 2


def beat_features(notes: np.ndarray, beat_t: np.ndarray, beat_d: np.ndarray) -> np.ndarray:
    """[K, N_BEAT_FEATURES] for K tracked beats. ``notes``: [N, 4] pitch, onset, offset, velocity
    (load_performance's columns), ``beat_d``: the tracker's downbeat flags. Per beat (its interval
    = up to the next beat; the last one repeats the previous length):
      beat length (log2 s / 0.5), log ratio to the previous and next beat, tracker downbeat, 1
      where in the beat the notes start (N_PHASE_BINS fractions) and how many (log)
      the notes on the beat: how many (log), loudest velocity, velocity vs the beat's mean, lowest
        pitch, any below middle-C's octave (bass), their longest duration (log2 s)
      change of harmony across the beat (1 - cosine of the pitch-class profiles of the beat and
        the previous one), lowest pitch of the beat's notes"""
    k = len(beat_t)
    f = np.zeros((k, N_BEAT_FEATURES), np.float32)
    if k == 0:
        return f
    t = np.asarray(beat_t, np.float64)
    ln = np.diff(t)
    ln = np.concatenate([ln, ln[-1:]]) if k > 1 else np.array([0.5])
    ln = np.clip(ln, 0.05, 5.0)
    on, pitch, vel = notes[:, 1].astype(np.float64), notes[:, 0].astype(np.int64), notes[:, 3].astype(np.float64)
    dur = np.clip(notes[:, 2] - notes[:, 1], 0.01, 20.0)
    order = np.argsort(on, kind="stable")
    on, pitch, vel, dur = on[order], pitch[order], vel[order], dur[order]
    f[:, 0] = np.log2(ln / 0.5)
    f[:, 1] = np.concatenate([[0.0], np.log2(ln[1:] / ln[:-1])])
    f[:, 2] = np.concatenate([np.log2(ln[1:] / ln[:-1]), [0.0]]) if k > 1 else 0.0
    f[:, 3] = np.asarray(beat_d, np.float32)
    f[:, 4] = 1.0
    prev_chroma = np.zeros(12)
    for j in range(k):
        lo, hi = np.searchsorted(on, t[j] - AT_BEAT), np.searchsorted(on, t[j] + ln[j] - AT_BEAT)
        if hi > lo:
            ph = np.clip((on[lo:hi] - t[j]) / ln[j], 0, 0.999)
            h = np.bincount((ph * N_PHASE_BINS).astype(int), minlength=N_PHASE_BINS)[:N_PHASE_BINS]
            f[j, 5:5 + N_PHASE_BINS] = h / max(hi - lo, 1)
        c = 5 + N_PHASE_BINS
        f[j, c] = math.log1p(hi - lo)
        a0, a1 = np.searchsorted(on, t[j] - AT_BEAT), np.searchsorted(on, t[j] + AT_BEAT)
        if a1 > a0:
            f[j, c + 1] = math.log1p(a1 - a0)
            f[j, c + 2] = vel[a0:a1].max() / 127.0
            f[j, c + 3] = (vel[a0:a1].mean() - (vel[lo:hi].mean() if hi > lo else vel[a0:a1].mean())) / 32.0
            f[j, c + 4] = pitch[a0:a1].min() / 128.0
            f[j, c + 5] = float(pitch[a0:a1].min() < 48)
            f[j, c + 6] = np.log2(dur[a0:a1].max() / 0.5) / 4.0
        chroma = np.bincount(pitch[lo:hi] % 12, minlength=12).astype(float) if hi > lo else np.zeros(12)
        na, nb = np.linalg.norm(chroma), np.linalg.norm(prev_chroma)
        f[j, c + 7] = 1.0 - (chroma @ prev_chroma) / (na * nb) if na > 0 and nb > 0 else 0.0
        f[j, c + 8] = pitch[lo:hi].min() / 128.0 if hi > lo else 0.0
        prev_chroma = chroma
    return f


# ---------------------------------------------------------------------------------------------
# consistent decoding: Viterbi over (time signature, position in the bar)
# ---------------------------------------------------------------------------------------------

@dataclass
class DecodeConfig:
    ts_change: float = math.log(1e-3)     # log-prob of a new time signature at a bar line
    mid_step_bar: float = math.log(1e-2)  # log-prob of a bar line falling between two tracked beats
    ts_weight: float = 1.0                # weight of the time-signature head's log-probs

    def to_dict(self):
        return asdict(self)


def _states(ts_idx: tuple[int, ...]):
    """All (ts, position in GRID units) states, and each ts's bar length in GRID units."""
    lens = [int(bar_quarters(TS_CLASSES[c]) * GRID) for c in ts_idx]
    states = [(c, p) for c, L in zip(ts_idx, lens) for p in range(L)]
    return states, dict(zip(ts_idx, lens))


def decode(logp_bar: np.ndarray, logp_step: np.ndarray, logp_ts: np.ndarray, cfg: DecodeConfig = DecodeConfig(),
           ts_idx: tuple[int, ...] | None = None) -> list[tuple[int, int]]:
    """Most probable consistent (time signature, position in the bar) per tracked beat.
    logp_bar [K, 2] (no / yes), logp_step [K, len(Q_STEPS)], logp_ts [K, len(TS_CLASSES)].
    Beat k+1's position = beat k's + its step; reaching the bar length exactly starts a bar on beat
    k+1 (where the time signature may change), passing it puts the bar line between the two
    (penalised: a tracker that missed the downbeat's beat)."""
    ts_idx = tuple(range(len(TS_CLASSES))) if ts_idx is None else ts_idx
    states, lens = _states(ts_idx)
    n, k = len(states), len(logp_bar)
    index = {s: i for i, s in enumerate(states)}
    steps = [int(s * GRID) for s in Q_STEPS]
    nxt = np.zeros((n, len(steps)), np.int64)
    pen = np.zeros((n, len(steps)))
    for i, (c, p) in enumerate(states):
        for j, s in enumerate(steps):
            q = p + s
            if q >= lens[c]:
                if q > lens[c]:
                    pen[i, j] = cfg.mid_step_bar
                q = (q - lens[c]) % lens[c]
            nxt[i, j] = index[(c, q)]
    st_c = np.array([c for c, _ in states])
    st_p = np.array([p for _, p in states])
    at0 = st_p == 0

    def emit(j):
        return np.where(at0, logp_bar[j, 1], logp_bar[j, 0]) + cfg.ts_weight * logp_ts[j, st_c]
    score = emit(0)
    back = np.zeros((k, n), np.int64)
    for j in range(1, k):
        cand = score[:, None] + logp_step[j - 1][None, :] + pen  # [n, steps]
        new = np.full(n, -np.inf)
        arg = np.zeros(n, np.int64)
        flat_to, flat_val = nxt.ravel(), cand.ravel()
        order = np.argsort(flat_val, kind="stable")             # ascending: the last write is the max
        new[flat_to[order]] = flat_val[order]
        arg[flat_to[order]] = (np.arange(n * len(steps))[order]) // len(steps)
        # a new time signature at a bar line: from the best bar start of any other time signature
        starts = np.nonzero(at0)[0]
        best = starts[np.argmax(new[starts])]
        alt = new[best] + cfg.ts_change
        better = alt > new[starts]
        new[starts[better]] = alt
        arg[starts[better]] = arg[best]
        back[j] = arg
        score = new + emit(j)
    path = [int(np.argmax(score))]
    for j in range(k - 1, 0, -1):
        path.append(int(back[j, path[-1]]))
    path.reverse()
    return [states[i] for i in path]


def grid_from_states(beat_t: np.ndarray, states: list[tuple[int, int]]) -> dict:
    """Decoded states -> the grid: {"t": beat times, "q": quarters from the first bar line (the bar
    the first beat is in; a pickup's first beat has q > 0), "bars": time signature of each bar,
    "bar_q": each bar line's q, "bar_t": its time (interpolated where it falls between beats)}.
    Consecutive beats of one time signature are ``(p' - p) mod bar length`` apart (a whole bar when
    equal); a new time signature starts at its bar line."""
    t = np.asarray(beat_t, np.float64)
    if not states:
        return {"t": t, "q": np.zeros(0), "bars": [], "bar_q": np.zeros(0), "bar_t": np.zeros(0)}
    lens = {c: int(bar_quarters(TS_CLASSES[c]) * GRID) for c, _ in states}
    c0, p0 = states[0]
    acc = [p0]                                   # GRID units from the first bar line
    bars, bar_q = [TS_CLASSES[c0]], [0]
    for (pc, pp), (c, p) in zip(states[:-1], states[1:]):
        L = lens[pc]
        if c == pc:
            d = (p - pp) % L or L
        else:
            d = L - pp + p
        acc.append(acc[-1] + d)
        line = acc[-1] - p
        if line > bar_q[-1]:
            bar_q.append(line)
            bars.append(TS_CLASSES[c])
    q = np.array(acc, np.float64) / GRID
    bq = np.array(bar_q, np.float64) / GRID
    bar_t = np.interp(bq, q, t) if len(t) > 1 else np.zeros(len(bq))
    if len(t) > 1:  # bar lines before the first beat: extend with the first tempo
        s0 = (t[1] - t[0]) / max(q[1] - q[0], 1e-6)
        bar_t = np.where(bq < q[0], t[0] + (bq - q[0]) * s0, bar_t)
    return {"t": t, "q": q, "bars": bars, "bar_q": bq, "bar_t": bar_t}


# ---------------------------------------------------------------------------------------------
# training samples, the model, its loss, prediction and the grid's scores
# ---------------------------------------------------------------------------------------------

def sample(perf: dict, rng: np.random.Generator, noise=None, chunk: int = 0, note_noise=None):
    """One training / validation sample of a performance: its tracked beats (the real tracker's if
    it has them, else the annotated beats corrupted by ``noise``, a src/beats.py BeatNoise; exact
    without), their features and targets, cut to a random ``chunk`` of beats (0 = all).
    ``note_noise``: a src/preprocessing/noise.py NoiseModel for the notes (clean MIDI -> like a
    transcription). Returns (features [K, F], targets {bar, step, ts}, beat times [K])."""
    gt, notes = perf["gt"], perf["notes"]
    if perf.get("beats") is not None:
        bt, bd = perf["beats"]
    elif noise is not None:
        bt, bd = noise(gt["t"], gt["down"], rng)
    else:
        bt, bd = gt["t"], gt["down"]
    bt, bd = np.asarray(bt, np.float64), np.asarray(bd, bool)
    if note_noise is not None and len(notes):
        store = np.stack([notes[:, 0], notes[:, 1], notes[:, 2] - notes[:, 1], notes[:, 3]], axis=1).astype(np.float32)
        noisy, _ = note_noise(store, np.zeros((0, 2), np.float32), rng)
        notes = np.stack([noisy[:, 0], noisy[:, 1], noisy[:, 1] + noisy[:, 2], noisy[:, 3]], axis=1)
    if chunk and len(bt) > chunk:
        a = int(rng.integers(0, len(bt) - chunk + 1))
        bt, bd = bt[a:a + chunk], bd[a:a + chunk]
    return beat_features(notes, bt, bd), targets(bt, gt), bt


@dataclass
class MeterConfig:
    n_features: int = N_BEAT_FEATURES
    d_model: int = 256
    n_heads: int = 8
    layers: int = 6
    ff: int = 1024
    dropout: float = 0.1
    n_steps: int = len(Q_STEPS)
    n_ts: int = len(TS_CLASSES)

    def to_dict(self):
        return asdict(self)


def build_meter_model(cfg: dict):
    """The beat-sequence transformer (Scoreformer encoder layers, rotary positions over the beat
    index) with the bar / step / ts heads."""
    import torch
    import torch.nn as nn

    from .scoreformer import EncoderLayer, ScoreformerConfig, rope_angles

    c = MeterConfig(**cfg)

    class MeterModel(nn.Module):
        arch = "meter"

        def __init__(self):
            super().__init__()
            self.cfg = c
            enc = ScoreformerConfig(vocab_size=1, d_model=c.d_model, n_heads=c.n_heads, enc_layers=c.layers,
                                    dec_layers=0, ff=c.ff, dropout=c.dropout)
            self.inp = nn.Sequential(nn.Linear(c.n_features, c.d_model), nn.LayerNorm(c.d_model), nn.Dropout(c.dropout))
            self.layers = nn.ModuleList(EncoderLayer(enc) for _ in range(c.layers))
            self.norm = nn.LayerNorm(c.d_model)
            self.head = nn.Linear(c.d_model, 2 + c.n_steps + c.n_ts)
            dh = c.d_model // c.n_heads
            self.register_buffer("freq", 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh)), persistent=False)

        def forward(self, x, mask):
            pos = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
            cos, sin = rope_angles(pos, self.freq)
            h = self.inp(x)
            m = mask[:, None, None, :]
            for layer in self.layers:
                h = layer(h, cos, sin, m)
            o = self.head(self.norm(h))
            return {"bar": o[..., :2], "step": o[..., 2:2 + c.n_steps], "ts": o[..., 2 + c.n_steps:]}

    return MeterModel()


def meter_loss(out: dict, tg: dict):
    """Cross-entropy of the three heads over the beats each has a target for."""
    import torch.nn.functional as F
    total, parts = 0.0, {}
    for k in ("bar", "step", "ts"):
        y = tg[k]
        m = y >= 0
        loss = F.cross_entropy(out[k][m].float(), y[m]) if m.any() else out[k].sum() * 0.0
        parts[k] = float(loss.detach())
        total = total + loss
    return total, parts


def predict_grid(model, notes: np.ndarray, beat_t: np.ndarray, beat_d: np.ndarray, device="cpu",
                 cfg: DecodeConfig = DecodeConfig()) -> dict:
    """A whole performance: the model's per-beat log-probabilities, decoded into a consistent grid
    (``decode``, ``grid_from_states``)."""
    import torch
    feats = beat_features(notes, beat_t, beat_d)
    if len(feats) < 2:
        return grid_from_states(beat_t, [(0, 0)] * len(beat_t))
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats)[None].to(device)
        out = model(x, torch.ones(1, len(feats), dtype=torch.bool, device=device))
        lp = {k: torch.log_softmax(v[0].float(), -1).cpu().numpy() for k, v in out.items()}
    return grid_from_states(beat_t, decode(lp["bar"], lp["step"], lp["ts"], cfg))


def grid_beats(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    """The grid's notated beats: (times, bar start?) of every beat of every bar (beat_layout of
    its time signature), timed through the grid's (t, q) anchors."""
    q, t = grid["q"], grid["t"]
    if len(q) < 2:
        return np.zeros(0), np.zeros(0, bool)
    bq = list(grid["bar_q"]) + [np.inf]  # the last bar runs to the last beat (the q range below)
    qs, down = [], []
    for k, ts in enumerate(grid["bars"]):
        n, qpb = beat_layout(ts)
        for j in range(n):
            x = bq[k] + j * qpb
            if x < bq[k + 1] - 1e-9 and q[0] - 1e-9 <= x <= q[-1] + 1e-9:
                qs.append(x)
                down.append(j == 0)
    return np.interp(np.array(qs), q, t), np.array(down, bool)


def grid_scores(grid: dict, gt: dict) -> dict:
    """How close a decoded grid is to the annotated one: beat and downbeat F-measure of its
    notated beats (``grid_beats``) against the annotated beats and bar starts, and the share of
    annotated bars whose time signature it has right (the bar its downbeat falls in)."""
    from .beats import f_measure
    bt, bd = grid_beats(grid)
    lo, hi = gt["t"][0] - MATCH_TOL, gt["t"][-1] + MATCH_TOL
    keep = (bt >= lo) & (bt <= hi)
    bt, bd = bt[keep], bd[keep]
    downs = gt["t"][gt["down"]]
    ts_ok, n = 0, 0
    bar_t = np.asarray(grid["bar_t"])
    for k in np.nonzero(gt["down"])[0]:
        j = int(np.searchsorted(bar_t, gt["t"][k] + MATCH_TOL)) - 1
        if 0 <= j < len(grid["bars"]):
            ts_ok += grid["bars"][j] == gt["ts_names"][k]
        n += 1
    return {"beat_f": f_measure(gt["t"], bt), "downbeat_f": f_measure(downs, bt[bd]),
            "ts_acc": ts_ok / max(n, 1)}
