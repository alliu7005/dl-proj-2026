"""Gangqinpu (Zeng, Mathur & Wang 2026; arXiv 2610.34694) for the meter model: whole performances
with a ground-truth grid, like PERiScoPe's (scripts/build_meter_data.py).

The Hugging Face release (``Wayne42/gangqinpu``, gated: accept the CC BY-NC-SA 4.0 licence on the
dataset page, then ``hf auth login``) is Parquet: ``data/{train,validation,test}-*.parquet``, one
row per score-performance pair, 18,786 pairs of web piano recordings (pop, classical, jazz, folk,
...). Every performance is a Transkun V2 transcription (ours too), aligned note by note to its
score. The columns used here:

  sample_id, split, group_id, score_group_id, genre, audio_url, source
  performance_notes   [{time, duration, pitch, velocity}] seconds of the audio; includes the
                      notes the alignment's refinement interpolated where nothing was played
  alignment           {performance_note_indices, interpolated_mask, ...}: per score note, its
                      performance note (-1: none) and whether that note was interpolated
  beats               [{measure, beat_in_measure, performance_time_sec (alias time_sec),
                      is_downbeat, is_pickup, time_valid, time_signature_inferred, method,
                      confidence, label, time_signature, beats_per_bar, ...}]: the score's beats
                      timed through the alignment; ``beats_per_bar`` counts compound meters in
                      dotted beats (6/8 -> 2), as ASAP's annotations and src/beats.beat_layout do

``performance(row)`` turns one row into the meter model's performances: the transcribed notes (the
interpolated ones dropped) and the grid of every run of consecutive trustworthy bars
(src/meter.py aligned_grids). A beat is trusted when its time is valid, it was timed from matched
notes or between them (not extrapolated or inside an alignment hole), its confidence is at least
``min_confidence`` and it is no pickup; a bar is kept when all its beats are, their positions fit
its time signature, the times increase and the tempo is plausible. Bars whose time signature was
inferred (not written in the score) are dropped unless ``keep_inferred``. Each dropped bar splits
the performance, as a dropped window splits a PERiScoPe one.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..beats import beat_layout
from ..meter import bar_quarters

SPLITS = {"train": "train", "val": "validation", "test": "test"}   # ours -> the release's
TRUSTED_METHODS = ("matched_anchor", "interpolated_between_anchors")
COLUMNS = ["sample_id", "split", "group_id", "score_group_id", "genre", "audio_url", "source",
           "performance_notes", "alignment", "beats"]


@dataclass
class GridFilter:
    min_confidence: float = 0.5      # the release's example treats >= 0.5 as reliable
    keep_inferred: bool = False      # bars whose time signature the release inferred
    min_qpm: float = 15.0            # plausible written tempo between consecutive beats
    max_qpm: float = 480.0
    min_beats: int = 16              # shortest run of bars kept (src/meter.py aligned_grids)


def performance_notes(row: dict) -> np.ndarray:
    """[N, 4] float32 (pitch, onset, offset, velocity), seconds, sorted by onset, as
    src/preprocessing/performance.load_performance gives them: the transcribed notes only (the
    ones the refinement interpolated are dropped)."""
    notes = row.get("performance_notes") or []
    if not notes:
        return np.zeros((0, 4), np.float32)
    arr = np.array([[n["pitch"], n["time"], n["time"] + max(float(n.get("duration") or 0.0), 1e-3),
                     n.get("velocity") or 64] for n in notes], np.float64)
    al = row.get("alignment") or {}
    idx, interp = al.get("performance_note_indices"), al.get("interpolated_mask")
    if idx is not None and interp is not None and len(idx) == len(interp):
        idx, interp = np.asarray(idx, np.int64), np.asarray(interp, bool)
        drop = idx[interp & (idx >= 0) & (idx < len(arr))]
        keep = np.ones(len(arr), bool)
        keep[drop] = False
        arr = arr[keep]
    arr = arr[np.argsort(arr[:, 1], kind="stable")]
    return arr.astype(np.float32)


def _beat_time(b: dict):
    t = b.get("performance_time_sec")
    return b.get("time_sec") if t is None else t


def trusted_beats(beats: list, flt: GridFilter = GridFilter()) -> tuple[list, Counter]:
    """(times, bars, beat indices from 0, time signatures) of the beats in trustworthy bars, and
    why the others were dropped (counts of beats)."""
    why = Counter()
    if not beats:
        return [np.zeros(0), np.zeros(0, np.int64), np.zeros(0, np.int64), []], why
    # beat_in_measure: 0- or 1-based? the downbeats say (they are beat 0)
    downs = [b.get("beat_in_measure") for b in beats if b.get("is_downbeat") and b.get("beat_in_measure") is not None]
    base = Counter(downs).most_common(1)[0][0] if downs else 0
    # bars in the order the beats come (played order): a new bar wherever the measure changes. Its
    # number for aligned_grids: the next one after a step to the next measure or back (a repeat),
    # leaving a gap after a skip forward, so the run is split there (what lies between is unknown)
    groups, prev_m, bar_no = [], None, -1
    for b in beats:
        if b.get("measure") is None or b.get("beat_in_measure") is None:
            why["no_position"] += 1
            continue
        m = int(b["measure"])
        if m != prev_m:
            bar_no = 0 if prev_m is None else bar_no + (m - prev_m if m > prev_m + 1 else 1)
            groups.append((bar_no, []))
            prev_m = m
        groups[-1][1].append(b)
    times, bars, idx, tss = [], [], [], []
    last_t = -np.inf
    for m, bb in groups:
        bb = sorted(bb, key=lambda b: b["beat_in_measure"])
        ts = bb[0].get("time_signature")
        reason, pos = None, []
        if any(b.get("is_pickup") for b in bb):
            reason = "pickup"
        elif not isinstance(ts, str) or "/" not in ts or any(b.get("time_signature") != ts for b in bb):
            reason = "time_signature"
        else:
            # the release's beats onto ours: it counts some meters in other beats than ASAP and
            # beat_layout do (6/4 in 6 quarters, ours 2 dotted halves; 12/4, 15/16), and a few beats
            # sit between beats (4.5). Each beat's place in the bar in quarters; only those on one of
            # our beats are kept, as that beat (6/4: its quarters 1 and 4 -> our beats 1 and 2)
            n_ours, qpb = beat_layout(ts)
            try:
                bar_q = float(bar_quarters(ts))
            except (ValueError, ZeroDivisionError):
                bar_q = n_ours * qpb
            theirs = bb[0].get("beats_per_bar") or n_ours
            their_qpb = bar_q / theirs
            kept = []
            for b in bb:
                k = (float(b["beat_in_measure"]) - base) * their_qpb / qpb
                if abs(k - round(k)) < 1e-6:
                    kept.append((b, int(round(k))))
                else:
                    why["between_our_beats"] += 1
            bb, pos = [b for b, _ in kept], [p for _, p in kept]
            if not bb:
                continue
        if reason:
            pass
        elif not flt.keep_inferred and any(b.get("time_signature_inferred") for b in bb):
            reason = "inferred_time_signature"
        elif any(not b.get("time_valid", True) or _beat_time(b) is None for b in bb):
            reason = "time_invalid"
        elif any(b.get("method") not in TRUSTED_METHODS for b in bb):
            reason = "untimed_method"
        elif any((b.get("confidence") if b.get("confidence") is not None else 0.0) < flt.min_confidence
                 or b.get("label") == "bR" for b in bb):
            reason = "low_confidence"
        else:
            t = np.array([float(_beat_time(b)) for b in bb])
            if len(set(pos)) != len(pos) or min(pos) < 0 or max(pos) >= n_ours:
                reason = "beats_vs_time_signature"
            elif t[0] <= last_t or np.any(np.diff(t) <= 0):
                reason = "time_order"
            else:
                gaps = np.diff(t) / np.diff(pos)            # seconds per beat between kept beats
                qpm = 60.0 * qpb / gaps if len(gaps) else np.array([])
                if len(qpm) and (qpm.min() < flt.min_qpm or qpm.max() > flt.max_qpm):
                    reason = "tempo"
        if reason:
            why[reason] += len(bb)
            continue
        for b, p in zip(bb, pos):
            times.append(float(_beat_time(b)))
            bars.append(m)
            idx.append(p)
            tss.append(ts)
        last_t = times[-1]
        why["kept"] += len(bb)
    return [np.array(times, np.float64), np.array(bars, np.int64), np.array(idx, np.int64), tss], why


def performance(row: dict, flt: GridFilter = GridFilter(), margin: float = 0.5) -> tuple[list, Counter]:
    """One release row -> the meter model's performances (one per run of consecutive trusted
    bars, as scripts/build_meter_data.py from_alignment makes them for PERiScoPe) and the beat
    counts of what was kept or dropped."""
    from ..meter import aligned_grids

    (t, bars, idx, tss), why = trusted_beats(row.get("beats") or [], flt)
    notes = performance_notes(row)
    segs = aligned_grids(t, bars, idx, tss, flt.min_beats) if len(t) else []
    out = []
    for k, gt in enumerate(segs):
        lo, hi = gt["t"][0] - margin, gt["t"][-1] + margin
        seg_notes = notes[(notes[:, 1] >= lo) & (notes[:, 1] <= hi)]
        if len(seg_notes) < 8:
            why["segment_few_notes"] += len(gt["t"])
            continue
        out.append({"performance": f"gangqinpu/{row['sample_id']}", "segment": k, "segments": len(segs),
                    "span": (float(lo), float(hi)), "piece": f"gangqinpu/{row.get('score_group_id') or row['sample_id']}",
                    "group": row.get("group_id"), "genre": row.get("genre"), "dataset": "gangqinpu",
                    "notes": seg_notes, "gt": gt, "beats": None, "real_notes": True})
    return out, why


def split_files(root: Path, split: str) -> list:
    """The release's Parquet files of one of our splits (train / val / test); [] if not downloaded."""
    return sorted((Path(root) / "data").glob(f"{SPLITS[split]}-*.parquet"))


def iter_rows(root: Path, split: str, columns=COLUMNS, batch_size: int = 64):
    """The release's rows of one of our splits (train / val / test), Gangqinpu's own only (not its
    ASAP reference rows), one audio recording once (370 rows share an audio URL)."""
    import pyarrow.parquet as pq

    files = split_files(root, split)
    if not files:
        raise FileNotFoundError(f"no {Path(root) / 'data' / (SPLITS[split] + '-*.parquet')}: download the release "
                                "(pace/v2_gangqinpu.sh, or huggingface-cli download Wayne42/gangqinpu --repo-type dataset)")
    seen = set()
    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in columns if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            for row in batch.to_pylist():
                if row.get("source") not in (None, "gangqinpu"):
                    continue
                url = row.get("audio_url")
                if url:
                    if url in seen:
                        continue
                    seen.add(url)
                yield row


def build_split(root: Path, split: str, flt: GridFilter = GridFilter(), limit: int = 0) -> tuple[list, dict]:
    """Every performance of one split (``limit``: the first N rows only) and a report: rows,
    performances, segments, beats kept / dropped by reason, genres."""
    perfs, why, genres, rows, empty = [], Counter(), Counter(), 0, 0
    for row in iter_rows(root, split):
        rows += 1
        out, w = performance(row, flt)
        why.update(w)
        if out:
            perfs.extend(out)
            genres[row.get("genre") or "unknown"] += 1
        else:
            empty += 1
        if limit and rows >= limit:
            break
    report = {"rows": rows, "performances": rows - empty, "segments": len(perfs),
              "beats": dict(sorted(why.items())), "genres": dict(genres.most_common())}
    return perfs, report
