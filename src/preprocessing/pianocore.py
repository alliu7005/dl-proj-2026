"""PianoCoRe (Borovik 2026) access: remote parquet shards, ASAP piece mapping, aligned note arrays.

The Hugging Face release (``SyMuPe/PianoCoRe``) is 18 parquet shards (~8.6 GB) whose rows embed the
MIDI / MusicXML / alignment files as bytes columns. Parquet is columnar, so metadata columns and
individual row groups can be read over HTTP without downloading whole shards.

Refined (tier A/A*) rows provide:
  * ``refined_score_midi_bytes``        single-track score MIDI (ticks, 480/quarter)
  * ``refined_performance_midi_bytes``  performance MIDI with one note per score note (+ CC64 pedal)
  * ``refined_alignment_bytes``         npz: ``perf_idx`` [S] (score note i -> performance note
                                        perf_idx[i]) and ``interpolated`` [S] (True = synthetic note
                                        inserted where no performed note matched)
"""
from __future__ import annotations

import io
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .performance import pedal_intervals

HF_REPO = "datasets/SyMuPe/PianoCoRe"
META_COLS = [
    "id", "composer", "composition", "movement", "performance_id", "split", "tier_b", "tier_a",
    "tier_a_star", "score_dataset", "score_id", "score_xml_path", "performance_dataset",
    "performance_midi_path", "performance_note_count", "performance_duration", "performer",
    "is_transcription", "capture_model", "raw_recall", "raw_precision", "is_duplicate",
    "quality_label", "is_refined", "refined_performance_note_count",
    "refined_performance_interpolated_note_count", "refined_recall",
]
ALIGN_COLS = ["refined_performance_midi_bytes", "refined_score_midi_bytes", "refined_alignment_bytes"]


def hf_fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def shard_paths(fs) -> list[str]:
    return sorted(fs.glob(f"{HF_REPO}/data/*.parquet"))


def open_shard(fs, path: str):
    import pyarrow.parquet as pq
    return pq.ParquetFile(fs.open(path, "rb", block_size=1 << 20))


def work_key(composer, composition, movement) -> str:
    s = lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)  # noqa: E731
    return f"{s(composer)}|{s(composition)}|{s(movement)}"


def map_works_to_asap(meta, asap_rows: list[dict]) -> dict[str, list[str]]:
    """PianoCoRe work key -> ASAP folders, via the ASAP performances PianoCoRe contains.

    PianoCoRe's ASAP rows have ``performance_id = "ASAP_<stem>"`` where <stem> is the ASAP
    performance file name. Stems can repeat across folders (one MAESTRO recital -> several pieces),
    so each work votes over candidate folders and keeps every folder with >= half the top vote.
    """
    folders_by_stem = defaultdict(set)
    for r in asap_rows:
        folders_by_stem[Path(r["midi_performance"]).stem].add(r["folder"])
    asap = meta[meta["performance_dataset"] == "ASAP"]
    out = {}
    for key, ids in asap.groupby("work_key")["performance_id"]:
        votes = Counter(f for pid in ids for f in folders_by_stem.get(pid.removeprefix("ASAP_"), ()))
        if votes:
            top = votes.most_common(1)[0][1]
            out[key] = sorted(f for f, v in votes.items() if v * 2 >= top)
    return out


def load_aligned(perf_bytes: bytes, score_bytes: bytes, align_bytes: bytes,
                 pedal_threshold: int = 64) -> dict[str, np.ndarray]:
    """One aligned (score note, performance note) pair per score note, in score order.

    Returns arrays of length S: ``beat`` / ``dur_beats`` (score onset / duration in quarters),
    ``pitch``, ``onset`` / ``offset`` (performance seconds), ``velocity``, ``interpolated`` (bool),
    plus ``pedal`` [K, 2] sustain intervals in seconds.
    """
    import symusic

    perf = symusic.Score.from_midi(perf_bytes).to("second")
    score = symusic.Score.from_midi(score_bytes)
    if len(perf.tracks) != 1 or len(score.tracks) != 1:
        raise ValueError(f"expected single-track refined MIDI, got {len(score.tracks)}/{len(perf.tracks)}")
    al = np.load(io.BytesIO(align_bytes))
    idx, interp = al["perf_idx"].astype(np.int64), al["interpolated"].astype(bool)
    sn, pn = score.tracks[0].notes.numpy(), perf.tracks[0].notes.numpy()
    if len(idx) != len(sn["time"]):
        raise ValueError(f"alignment length {len(idx)} != score notes {len(sn['time'])}")
    tpq = score.ticks_per_quarter
    onset = pn["time"][idx].astype(np.float64)
    offset = onset + pn["duration"][idx]
    ctl = perf.tracks[0].controls.numpy()
    sus = ctl["number"] == 64
    ccs = sorted(zip(ctl["time"][sus].tolist(), ctl["value"][sus].tolist()))
    return {
        "beat": sn["time"] / tpq, "dur_beats": sn["duration"] / tpq, "pitch": sn["pitch"],
        "perf_pitch": pn["pitch"][idx], "onset": onset, "offset": offset,
        "velocity": pn["velocity"][idx].astype(np.float64), "interpolated": interp,
        "pedal": pedal_intervals(ccs, pedal_threshold, float(offset.max()) if len(offset) else 0.0),
    }


# ----------------------------------------------------------------------------------------------
# Building training pairs: raw transcription (input) + unfolded MusicXML (target) + time map
# ----------------------------------------------------------------------------------------------
FETCH_COLS = ["performance_midi_bytes"] + ALIGN_COLS  # per performance; score_xml_bytes once per score


def load_unfolded_score(xml_bytes: bytes, suffix: str):
    """MusicXML/MXL bytes -> ``score.load_score`` measures with repeats written out, because
    PianoCoRe's score MIDI (made with partitura) plays repeats. Falls back to the score as written
    if music21 can't expand it; ``score_midi_shift`` then reports a poor match."""
    import tempfile

    import music21 as m21

    from .score import load_score

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / f"score{suffix}"
        f.write_bytes(xml_bytes)
        s = m21.converter.parse(str(f))
    try:
        s = s.expandRepeats()
    except Exception:  # noqa: BLE001
        pass
    return load_score(s)


def score_midi_shift(score_midi_bytes: bytes, measures, max_shift_q: float = 8.0,
                     grid: int = 48) -> tuple[float, float]:
    """Constant offset (quarters) from score-MIDI time to MusicXML time (partitura places a pickup
    at the end of a full first bar), and the onset+pitch F1 between the two at that offset."""
    import symusic

    sc = symusic.Score.from_midi(score_midi_bytes)
    sn = sc.tracks[0].notes.numpy()
    off = np.cumsum([0.0] + [float(m.length) for m in measures])
    xml = Counter((round((off[k] + float(e.offset)) * grid), e.pitch)
                  for k, m in enumerate(measures) for e in m.events if not e.grace and not e.tie)
    on = np.round(sn["time"] / sc.ticks_per_quarter * grid).astype(np.int64)
    best_hits, best_shift = 0, 0
    for shift in range(-int(max_shift_q * grid), int(max_shift_q * grid) + 1, grid // 8):
        hits = sum((Counter(zip((on - shift).tolist(), sn["pitch"].tolist())) & xml).values())
        if hits > best_hits:
            best_hits, best_shift = hits, shift
    n = sum(xml.values()) + len(on)
    return best_shift / grid, (2 * best_hits / n if n else 0.0)


def alignment_pairs(perf_bytes: bytes, score_bytes: bytes, align_bytes: bytes, shift_q: float):
    """(performance onset s, MusicXML onset in quarters) for every aligned, non-interpolated note."""
    d = load_aligned(perf_bytes, score_bytes, align_bytes)
    keep = ~d["interpolated"]
    return d["onset"][keep], d["beat"][keep] - shift_q


def match_raw_to_refined(raw: np.ndarray, d: dict, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair refined (aligned) notes with the raw transcription's notes they were copied from.

    PianoCoRe's refined performance is an EDITED copy of the raw transcription: lead-in trimmed,
    pauses shortened, sections the score doesn't have cut out (e.g. a played exposition repeat), so
    its clock is not a constant shift of the raw one. Notes copied unchanged keep their exact pitch,
    velocity and duration; a refined note whose (pitch, velocity, duration) occurs exactly once in
    the raw file is an anchor (~65% of notes in practice). Returns (refined index, raw index) pairs."""
    sig = defaultdict(list)
    for j, (p, on, off, v) in enumerate(raw):
        sig[(int(p), int(v), round(float(off - on), 3))].append(j)
    ref_idx, raw_idx = [], []
    for k in np.nonzero(keep)[0]:
        c = sig.get((int(d["perf_pitch"][k]), int(d["velocity"][k]), round(float(d["offset"][k] - d["onset"][k]), 3)), [])
        if len(c) == 1:
            ref_idx.append(k)
            raw_idx.append(c[0])
    return np.asarray(ref_idx, dtype=np.int64), np.asarray(raw_idx, dtype=np.int64)


def pianocore_performance(row: dict, measures, shift_q: float, min_anchor_frac: float = 0.3,
                          max_anchor_gap_s: float = 3.0):
    """One PianoCoRe row -> (notes, pedal, timeline, info) for ``segment.segment_windows``, or an
    error string.

    Input notes are the RAW transcription (real transcriber errors kept). The time map is built
    from raw notes matched exactly to aligned refined notes (``match_raw_to_refined``), each
    giving (raw onset, score quarter); interpolated notes are excluded. Raw notes more than
    ``max_anchor_gap_s`` from any anchor are dropped: they are passages the alignment cut out
    (unscored repeats, lead-in/out), so no target exists for them."""
    from .performance import load_performance
    from .segment import timeline_from_alignment

    d = load_aligned(row["refined_performance_midi_bytes"], row["refined_score_midi_bytes"],
                     row["refined_alignment_bytes"])
    keep = ~d["interpolated"]
    notes, pedal = load_performance(row["performance_midi_bytes"])
    ref_i, raw_i = match_raw_to_refined(notes, d, keep)
    frac = len(ref_i) / max(int(keep.sum()), 1)
    if frac < min_anchor_frac:
        return f"raw_vs_refined_anchors_{frac:.2f}"
    anchor_t = np.sort(notes[raw_i, 1])
    j = np.searchsorted(anchor_t, notes[:, 1])
    near = np.minimum(np.abs(notes[:, 1] - anchor_t[np.clip(j - 1, 0, len(anchor_t) - 1)]),
                      np.abs(notes[:, 1] - anchor_t[np.clip(j, 0, len(anchor_t) - 1)]))
    kept = near <= max_anchor_gap_s
    tl = timeline_from_alignment(notes[raw_i, 1], d["beat"][ref_i] - shift_q, measures)
    if isinstance(tl, str):
        return tl
    dropped = int((~kept).sum())
    notes = notes[kept]
    if len(pedal):  # keep pedal presses overlapping the kept notes' span
        pedal = pedal[(pedal[:, 1] > notes[:, 1].min() - 1) & (pedal[:, 0] < notes[:, 2].max() + 1)]
    return notes, pedal, tl, {"anchor_frac": round(frac, 3), "dropped_unaligned_notes": dropped}
