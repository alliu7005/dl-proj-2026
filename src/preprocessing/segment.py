"""Cut an aligned (performance, score) pair into measure-aligned training segments."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .performance import ONSET, OFFSET, TimeMap
from .score import Measure, bar_length, tokenize_bars


@dataclass
class SegmentConfig:
    # "windows": cut the performance at arbitrary note gaps; the target says where in the bar the
    #            window starts and marks every bar line (no bar annotations needed at inference).
    # "bars":    cut at annotated bar lines (needs a downbeat tracker at inference).
    mode: str = "windows"
    window_notes: tuple = (48, 192)  # train: window size (notes) drawn uniformly per window
    eval_window_notes: int = 128     # val/test: fixed size
    window_overlap: float = 0.5      # train overlap between consecutive windows (eval: 0)
    cut_search: float = 0.25         # cut at the largest onset gap within +-cut_search * size
    prompt: bool = True              # store the score just before each window as decoder context
    bars_per_segment: int = 4
    stride: int = 2                 # bars between segment starts (== bars_per_segment -> no overlap)
    onset_tolerance: float = 0.125  # quarter notes; notes played this early still belong to the next bar
    pedal_targets: bool = True      # add ped_on / ped_off tokens derived from performance CC64
    pedal_grid: int = 4             # pedal positions quantized to 1/pedal_grid quarter note
    max_input_notes: int = 768
    max_target_tokens: int = 1536
    max_bar_quarters: int = 16      # drop segments containing longer bars (unbarred cadenzas)
    min_pitch_f1: float = 0.6       # drop segments whose performed pitches don't match the score's
                                    # (local beat-annotation errors); 0 disables
    carry_notes: bool = True        # also input notes struck before the window but still held into it
                                    # (stored with negative onset), so tied-over targets have evidence
    carry_min_quarters: float = 0.25  # ...held at least this far past the window start (filters
                                      # ordinary legato overlap at the bar line)
    carry_pedal_sustained: bool = True  # also store notes whose key was released before that point
                                        # but which the sustain pedal keeps sounding. They are only
                                        # *candidates*: the Dataset drops a carried note unless its
                                        # key (after noise, e.g. Kong-style pedal lengthening) is still
                                        # down at ``carry_min_sec`` (see dataset.drop_released_carried)


def tied_into_bar(measures: list[Measure], bar: list[int]) -> Counter:
    """Pitches whose note at the very start of ``bar`` continues a tie from the previous bar."""
    return Counter(e.pitch for e in measures[bar[0]].events if e.tie and e.offset == 0 and not e.grace)


def pitch_f1(perf_pitches, tokens: list[str]) -> float:
    """F1 between the multiset of performed pitches and onset pitches in the target tokens."""
    score = Counter(int(t[2:]) for i, t in enumerate(tokens)
                    if t.startswith("p_") and not (i + 2 < len(tokens) and tokens[i + 2] == "tie"))
    perf = Counter(int(p) for p in perf_pitches)
    total = sum(score.values()) + sum(perf.values())
    return 2 * sum((score & perf).values()) / total if total else 1.0


def parse_downbeat_map(mp) -> list[list[int]] | None:
    """ASAP ``downbeats_score_map`` -> list of bars, each a list of measure indices.
    Entries are ints or strings like '8-0' (bar split across a repeat). NaN -> None."""
    if not isinstance(mp, list):
        return None
    return [[int(x) for x in str(v).split("-")] for v in mp]


def bar_timeline(bars: list[list[int]], measures: list[Measure], downbeats) -> tuple:
    """Score-time (s) start/end and quarter-length for each bar; prepends a pickup bar if present."""
    db = np.asarray(downbeats, dtype=np.float64)
    lengths = [bar_length(measures, b) for b in bars]
    starts = list(db)
    ends = list(db[1:])
    rate = (db[-1] - db[-2]) / float(lengths[-2]) if len(db) >= 2 and lengths[-2] > 0 else 0.5
    ends.append(db[-1] + float(lengths[-1]) * rate)

    first = bars[0][0]
    if 0 < first <= 2 and all(i < len(measures) for i in range(first)):  # anacrusis
        pick = list(range(first))
        pl = bar_length(measures, pick)
        r0 = (ends[0] - starts[0]) / float(lengths[0]) if lengths[0] > 0 else rate
        bars = [pick] + bars
        lengths = [pl] + lengths
        starts = [starts[0] - float(pl) * r0] + starts
        ends = [starts[1]] + ends
    return bars, np.asarray(starts), np.asarray(ends), lengths


def _pedal_tokens(pedal, tm: TimeMap, starts, ends, lengths, j0, j1, lo, hi, cfg) -> list[list]:
    out = [[] for _ in range(j1 - j0)]
    for on, off in pedal:
        for t, tok in ((on, "ped_on"), (off, "ped_off")):
            s = float(tm(t))
            if tok == "ped_on" and s < lo <= float(tm(off)):
                s = max(starts[j0], lo)  # pedal already held when the segment starts
            if not lo <= s < hi:
                continue
            k = int(np.searchsorted(starts, s, side="right")) - 1
            k = min(max(k, j0), j1 - 1)
            span = ends[k] - starts[k]
            pos = (s - starts[k]) / span * float(lengths[k]) if span > 0 else 0.0
            pos = Fraction(round(pos * cfg.pedal_grid), cfg.pedal_grid)
            pos = min(max(pos, Fraction(0)), max(lengths[k] - Fraction(1, cfg.pedal_grid), Fraction(0)))
            out[k - j0].append((pos, tok))
    return out


def segment_pair(notes, pedal, ann: dict, measures: list[Measure], cfg: SegmentConfig,
                 meta: dict) -> tuple[list[dict], dict]:
    """Return (segments, stats) for one performance."""
    stats = {"too_long": 0, "empty": 0, "long_bar": 0, "low_pitch_f1": 0, "carry_candidates": 0,
             "carried_notes": 0, "ties_at_window_start": 0, "ties_with_carried_note": 0}
    bars = parse_downbeat_map(ann["downbeats_score_map"])
    if bars is None or len(bars) != len(ann["midi_score_downbeats"]) or len(bars) < 2:
        return [], {"skipped": "bad_downbeat_map"}
    if max(i for b in bars for i in b) >= len(measures):
        return [], {"skipped": "map_out_of_range"}
    tm = TimeMap(ann["performance_beats"], ann["midi_score_beats"])
    inv = tm.inverse()
    bars, starts, ends, lengths = bar_timeline(bars, measures, ann["midi_score_downbeats"])

    score_on = tm(notes[:, ONSET])
    score_off = tm(notes[:, OFFSET])
    # when the note stops sounding: key release, or pedal release if the pedal is down at key release
    sound_end = notes[:, OFFSET].astype(np.float64)
    if cfg.carry_pedal_sustained and len(pedal):
        k = np.searchsorted(pedal[:, 0], sound_end) - 1
        under = (k >= 0) & (sound_end < pedal[np.clip(k, 0, None), 1])
        sound_end[under] = pedal[k[under], 1]
    score_sound_end = tm(sound_end)
    spq =np.median((ends - starts) / np.maximum([float(x) for x in lengths], 1e-6))
    tol = cfg.onset_tolerance * spq
    nb, W = len(bars), cfg.bars_per_segment
    seg_starts = list(range(0, max(nb - W, 0) + 1, cfg.stride))
    if seg_starts[-1] + W < nb:
        seg_starts.append(nb - W)

    segs = []
    for j0 in seg_starts:
        j1 = min(j0 + W, nb)
        lo = -math.inf if j0 == 0 else starts[j0] - tol
        hi = math.inf if j1 == nb else ends[j1 - 1] - tol
        if max(lengths[j0:j1]) > cfg.max_bar_quarters:
            stats["long_bar"] += 1
            continue
        idx = np.nonzero((score_on >= lo) & (score_on < hi))[0]
        if len(idx) == 0:
            stats["empty"] += 1
            continue
        seg_notes = notes[idx]
        t0 = min(float(inv(starts[j0] - tol)), float(seg_notes[:, ONSET].min()))
        t1 = max(float(inv(ends[j1 - 1] - tol)), float(seg_notes[:, ONSET].max()))
        # notes struck before the window and still held into it -> negative onset (always < 0:
        # they precede both the window start and every in-window onset, and TimeMap is monotonic)
        carried, carry_min_sec = np.zeros(0, dtype=np.int64), 0.0
        if cfg.carry_notes and j0 > 0:
            carry_at = starts[j0] + cfg.carry_min_quarters * spq
            carry_min_sec = float(inv(carry_at)) - t0  # key must be down until here (segment seconds)
            carried = np.nonzero((score_on < lo) & (score_sound_end >= carry_at))[0]
        all_notes = np.concatenate([notes[carried], seg_notes])
        out = np.stack([all_notes[:, 0], all_notes[:, ONSET] - t0,
                        all_notes[:, OFFSET] - all_notes[:, ONSET], all_notes[:, 3]], axis=1)
        ped = pedal[(pedal[:, 1] > t0) & (pedal[:, 0] < t1)] - t0 if len(pedal) else pedal
        ped_tok = _pedal_tokens(pedal, tm, starts, ends, lengths, j0, j1, lo, hi, cfg) if cfg.pedal_targets else None
        tokens = tokenize_bars(measures, bars[j0:j1], ped_tok)
        if len(out) > cfg.max_input_notes or len(tokens) + 2 > cfg.max_target_tokens:
            stats["too_long"] += 1
            continue
        f1 = pitch_f1(out[len(carried):, 0], tokens)  # carried notes pair with tie tokens, not onsets
        if f1 < cfg.min_pitch_f1:
            stats["low_pitch_f1"] += 1
            continue
        if j0 > 0:  # how often a tie into the window's first beat has a (clean, key-held) carried note
            ties = tied_into_bar(measures, bars[j0])
            visible = carried[notes[carried, OFFSET] - t0 >= carry_min_sec]
            stats["carry_candidates"] += len(carried)
            stats["carried_notes"] += len(visible)
            stats["ties_at_window_start"] += sum(ties.values())
            stats["ties_with_carried_note"] += sum((ties & Counter(int(p) for p in notes[visible, 0])).values())
        segs.append({
            **meta,
            "bar_start": j0,
            "n_bars": j1 - j0,
            "window_sec": np.float32(t1 - t0),
            "perf_t0": np.float32(t0),            # absolute performance time of onset 0
            "carry_min_sec": np.float32(carry_min_sec),  # carried notes need key down until here
            "notes": out.astype(np.float32),     # pitch, onset (s, rel.), duration (s), velocity
            "pedal": np.asarray(ped, dtype=np.float32).reshape(-1, 2),  # (on, off) rel. seconds
            "tokens": tokens,
        })
    stats["segments"] = len(segs)
    return segs, stats


# ----------------------------------------------------------------------------------------------
# "windows" mode: cut anywhere, let the model infer the bar lines
# ----------------------------------------------------------------------------------------------

def prompt_from_output(history: list[str]) -> list[str]:
    """Inference-time prompt for the next window, from all tokens decoded so far (the windows'
    outputs concatenated): the current bar from its last ``bar`` line, pedal removed, with the
    current time/key signature after the bar line. Matches the prompts stored at training time."""
    if "bar" not in history:
        return []
    k = len(history) - 1 - history[::-1].index("bar")
    ts = next((t for t in reversed(history) if t.startswith("ts_")), None)
    key = next((t for t in reversed(history) if t.startswith("key_")), None)
    tail = [t for t in history[k + 1:] if not t.startswith(("ped_", "ts_", "key_"))]
    return ["bar"] + [t for t in (ts, key) if t] + tail

def cut_windows(onsets: np.ndarray, size_fn, overlap: float, search: float) -> list[tuple[int, int]]:
    """Split notes (sorted by onset) into windows [a, b) of about ``size_fn()`` notes each.

    Every cut sits at the largest onset gap within +-``search`` * size of its target, so chords
    and ornaments are not split and the cut falls clearly between two score positions."""
    n = len(onsets)
    gap_before = np.concatenate([[0.0], np.diff(onsets)])  # gap_before[i]: cut between i-1 and i

    def best_cut(target: float, size: int, lo_lim: int, hi_lim: int) -> int:
        lo = max(lo_lim, int(round(target - search * size)))
        hi = min(hi_lim, int(round(target + search * size)))
        if hi < lo:
            return int(min(max(round(target), lo_lim), hi_lim))
        return lo + int(np.argmax(gap_before[lo:hi + 1]))

    out, a = [], 0
    while a < n:
        size = max(int(size_fn()), 2)
        if n - a <= size * (1 + search):
            out.append((a, n))
            break
        b = best_cut(a + size, size, a + 1, n - 1)
        out.append((a, b))
        nxt = b if overlap <= 0 else best_cut(a + size * (1 - overlap), size, a + 1, b - 1)
        a = max(nxt, a + 1)
    return out


def _locate(s: float, starts, ends, lengths) -> tuple[int, float]:
    """Score time (s) -> (bar index, position in quarters within that bar)."""
    nb = len(starts)
    if not np.isfinite(s):
        return (0, 0.0) if s < 0 else (nb - 1, float(lengths[-1]))
    k = int(np.clip(np.searchsorted(starts, s, side="right") - 1, 0, nb - 1))
    span = ends[k] - starts[k]
    pos = float(np.clip((s - starts[k]) / span * float(lengths[k]) if span > 0 else 0.0, 0, float(lengths[k])))
    if pos >= float(lengths[k]) - 1e-9 and k + 1 < nb:
        k, pos = k + 1, 0.0
    return k, pos


def _window_pedal_tokens(pedal, tm, starts, ends, lengths, k_lo, k_hi, s_lo, s_hi, start_pos, grid):
    out = [[] for _ in range(k_hi - k_lo + 1)]
    step = Fraction(1, grid)
    first = Fraction(math.ceil(start_pos * grid - 1e-6), grid)

    def place(s):
        k, pos = _locate(s, starts, ends, lengths)
        k = min(max(k, k_lo), k_hi)
        last = max(lengths[k] - step, Fraction(0))
        p = min(max(Fraction(round(pos * grid), grid), Fraction(0)), last)
        return k, (max(p, min(first, last)) if k == k_lo else p)

    for on, off in pedal:
        son, soff = float(tm(on)), float(tm(off))
        held = son < s_lo <= soff  # pedal already down when the window starts
        on_at = place(s_lo) if held else (place(son) if s_lo <= son < s_hi else None)
        off_at = place(soff) if s_lo <= soff < s_hi else None
        if held and on_at == off_at:
            continue  # released right at the window start: nothing to notate
        for at, tok in ((on_at, "ped_on"), (off_at, "ped_off")):
            if at is not None:
                out[at[0] - k_lo].append((at[1], tok))
    return out


def segment_windows(notes, pedal, ann: dict, measures: list[Measure], cfg: SegmentConfig, meta: dict,
                    train: bool, seed: int = 0, windows: list | None = None) -> tuple[list[dict], dict]:
    """Cut one performance into note windows at arbitrary points (``mode: windows``).

    Each segment: input = performed notes whose onset falls in the window (+ carried-over notes),
    target = the score between the window's two cut points (first bar may start mid-bar, last bar
    may end mid-bar), prompt = the score just before the window (the rest of the bar the window
    starts in, or the whole previous bar if it starts on a bar line).

    ``windows`` = [(s_lo, s_hi, win_id)] in score seconds (None = piece start/end) reuses given
    cut points, e.g. to segment a real transcription exactly like the clean performance."""
    stats = {"too_long": 0, "empty": 0, "long_bar": 0, "low_pitch_f1": 0, "carry_candidates": 0,
             "carried_notes": 0, "ties_at_window_start": 0, "ties_with_carried_note": 0}
    bars = parse_downbeat_map(ann["downbeats_score_map"])
    if bars is None or len(bars) != len(ann["midi_score_downbeats"]) or len(bars) < 2:
        return [], {"skipped": "bad_downbeat_map"}
    if max(i for b in bars for i in b) >= len(measures):
        return [], {"skipped": "map_out_of_range"}
    tm = TimeMap(ann["performance_beats"], ann["midi_score_beats"])
    inv = tm.inverse()
    bars, starts, ends, lengths = bar_timeline(bars, measures, ann["midi_score_downbeats"])
    spq = np.median((ends - starts) / np.maximum([float(x) for x in lengths], 1e-6))

    score_on = tm(notes[:, ONSET])
    sound_end = notes[:, OFFSET].astype(np.float64)
    if cfg.carry_pedal_sustained and len(pedal):
        kk = np.searchsorted(pedal[:, 0], sound_end) - 1
        under = (kk >= 0) & (sound_end < pedal[np.clip(kk, 0, None), 1])
        sound_end[under] = pedal[kk[under], 1]
    score_sound_end = tm(sound_end)

    if windows is None:
        rng = np.random.default_rng(seed)
        lo_n, hi_n = cfg.window_notes if train else (cfg.eval_window_notes,) * 2
        cuts = cut_windows(notes[:, ONSET], lambda: rng.integers(lo_n, hi_n + 1),
                           cfg.window_overlap if train else 0.0, cfg.cut_search)

        def mid(i):
            return float(score_on[i - 1] + score_on[i]) / 2
        windows = [(None if a == 0 else mid(a), None if b == len(notes) else mid(b), w)
                   for w, (a, b) in enumerate(cuts)]

    segs = []
    for s_lo_, s_hi_, win in windows:
        s_lo = -math.inf if s_lo_ is None else s_lo_
        s_hi = math.inf if s_hi_ is None else s_hi_
        idx = np.nonzero((score_on >= s_lo) & (score_on < s_hi))[0]
        if len(idx) == 0:
            stats["empty"] += 1
            continue
        k_lo, start_pos = _locate(s_lo, starts, ends, lengths)
        k_hi, end_pos = _locate(s_hi, starts, ends, lengths)
        if end_pos <= 1e-9 and k_hi > k_lo:  # window ends exactly on a bar line
            k_hi, end_pos = k_hi - 1, float(lengths[k_hi - 1])
        end_cut = None if s_hi_ is None else end_pos
        if max(lengths[k_lo:k_hi + 1]) > cfg.max_bar_quarters:
            stats["long_bar"] += 1
            continue

        seg_notes = notes[idx]
        first_on = float(seg_notes[:, ONSET].min())
        t0 = min(float(inv(s_lo)) if s_lo_ is not None else float(inv(starts[0])), first_on)
        t1 = float(inv(s_hi)) if s_hi_ is not None else float(seg_notes[:, OFFSET].max())

        carried, carry_min_sec = np.zeros(0, dtype=np.int64), 0.0
        if cfg.carry_notes and s_lo_ is not None:
            carry_at = s_lo + cfg.carry_min_quarters * spq
            carry_min_sec = float(inv(carry_at)) - t0
            carried = np.nonzero((score_on < s_lo) & (score_sound_end >= carry_at))[0]
        all_notes = np.concatenate([notes[carried], seg_notes])
        out = np.stack([all_notes[:, 0], all_notes[:, ONSET] - t0,
                        all_notes[:, OFFSET] - all_notes[:, ONSET], all_notes[:, 3]], axis=1)
        out[: len(carried), 1] = np.minimum(out[: len(carried), 1], -1e-3)
        ped = pedal[(pedal[:, 1] > t0) & (pedal[:, 0] < t1)] - t0 if len(pedal) else pedal

        ped_tok = _window_pedal_tokens(pedal, tm, starts, ends, lengths, k_lo, k_hi, s_lo, s_hi,
                                       start_pos, cfg.pedal_grid) if cfg.pedal_targets else None
        tokens = tokenize_bars(measures, bars[k_lo:k_hi + 1], ped_tok, start_pos=start_pos, end_pos=end_cut)
        prompt = []
        if cfg.prompt and s_lo_ is not None:
            if start_pos > 1e-9:
                prompt = tokenize_bars(measures, [bars[k_lo]], end_pos=start_pos)
            elif k_lo > 0:
                prompt = tokenize_bars(measures, [bars[k_lo - 1]])
        if len(out) > cfg.max_input_notes or len(tokens) + len(prompt) + 3 > cfg.max_target_tokens:
            stats["too_long"] += 1
            continue
        f1 = pitch_f1(seg_notes[:, 0], tokens)
        if f1 < cfg.min_pitch_f1:
            stats["low_pitch_f1"] += 1
            continue
        if s_lo_ is not None:  # ties in the window's first quarter note vs (clean) carried notes
            first_bar = measures[bars[k_lo][0]]
            ties = Counter(e.pitch for e in first_bar.events if e.tie and not e.grace
                           and start_pos - 1e-9 <= float(e.offset) < start_pos + 1)
            visible = carried[notes[carried, OFFSET] - t0 >= carry_min_sec]
            stats["carry_candidates"] += len(carried)
            stats["carried_notes"] += len(visible)
            stats["ties_at_window_start"] += sum(ties.values())
            stats["ties_with_carried_note"] += sum((ties & Counter(int(p) for p in notes[visible, 0])).values())
        segs.append({
            **meta,
            "win": int(win),
            "bar_start": int(k_lo),
            "n_bars": int(k_hi - k_lo + 1),
            "start_pos": float(start_pos),
            "s_lo": s_lo_, "s_hi": s_hi_,          # score-time cut points (None = piece start/end)
            "window_sec": np.float32(t1 - t0),
            "perf_t0": np.float32(t0),
            "carry_min_sec": np.float32(carry_min_sec),
            "notes": out.astype(np.float32),
            "pedal": np.asarray(ped, dtype=np.float32).reshape(-1, 2),
            "tokens": tokens,
            "prompt": prompt,
        })
    stats["segments"] = len(segs)
    return segs, stats
