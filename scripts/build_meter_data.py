"""Whole performances for the meter model (v2, Phase 2): notes + ground-truth grid (+ tracked beats).

    python scripts/build_meter_data.py                      # -> data/processed/meter/<split>.pkl
    python scripts/build_meter_data.py --workers 8 --check  # + the alignment grids vs ASAP's annotations

ASAP (train / val / test): every aligned performance of the split's pieces (data/processed/
splits.json): its performance MIDI's notes and its annotated grid (src/meter.py annotation_grid:
beats with quarter positions, bar starts, time and key signatures). For the real-transcription
splits (<split>_transcribed, when their report.json exists): the Transkun transcription shifted
onto the ASAP timeline and cut to the performance (as scripts/system_inputs.py does), and the
beats Beat This! tracked (beats_tracked.npz).

PERiScoPe (train_periscope / val_periscope / test_periscope, from the PERiScoPe windows of the
processed train / val / test_periscope splits): no beat annotations, but every window is aligned
to its score, so the score's beats are timed through each window's alignment (src/beats.py
window_grid), joined by global bar number (merge_performance_bars) and turned into the same kind
of grid (src/meter.py aligned_grids: one per run of consecutive bars, a dropped window splits a
performance into segments); the windows' notes (real Transkun output) are joined back into the
performance (stitch_notes). The meter model trains on them with the tracker simulated.

Gangqinpu (--gangqinpu DIR: the downloaded Hugging Face release; train_gangqinpu / val_gangqinpu /
test_gangqinpu): 18,786 web piano recordings (pop, classical, jazz, folk, ...) transcribed by
Transkun V2 and aligned to their scores; the score's beats timed through that alignment come with
the release, so each performance's trusted bars become the same kind of grid
(src/preprocessing/gangqinpu.py: the filters, and why beats were dropped in
meter/gangqinpu_report.json). The notes are real transcriptions; the tracker is simulated.
--gangqinpu-only builds just these three files (the ASAP and PERiScoPe ones stay as they are).

--check times the ASAP test performances the same way (from their windows' alignments) and
compares the result with their annotations: how far the PERiScoPe ground truth can be trusted
(written to data/processed/meter/check.json).
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.system_inputs import MARGIN, crop  # noqa: E402
from src.beats import MATCH_TOL, beats_path, f_measure, load_beats, merge_performance_bars, window_grid  # noqa: E402
from src.meter import aligned_grids, annotation_grid, stitch_notes  # noqa: E402
from src.preprocessing.performance import load_performance  # noqa: E402
from src.preprocessing import gangqinpu  # noqa: E402
from src.preprocessing.gangqinpu import GridFilter  # noqa: E402

PERISCOPE = {"train_periscope": "train", "val_periscope": "val", "test_periscope": "test_periscope"}


def build(asap: Path, data: Path, transcribed: Path, maestro: Path) -> dict:
    ann = json.load(open(asap / "asap_annotations.json"))
    meta = list(csv.DictReader(open(asap / "metadata.csv", encoding="utf-8")))
    sp = json.load(open(data / "splits.json"))
    pieces = sp.get("asap", sp)
    out = {}
    for split in ("train", "val", "test"):
        perfs = []
        for row in meta:
            perf = row["midi_performance"]
            a = ann.get(perf)
            if pieces.get(row["folder"]) != split or a is None or not a.get("score_and_performance_aligned", True):
                continue
            notes, _ = load_performance(asap / perf)
            if len(notes) < 8 or len(a["performance_beats"]) < 4:
                continue
            perfs.append({"performance": perf, "piece": row["folder"], "dataset": "asap", "notes": notes,
                          "gt": annotation_grid(a), "beats": None})
        out[split] = perfs
        rep = data / f"{split}_transcribed" / "report.json"
        bp = beats_path(data / f"{split}_transcribed", "tracked")
        if rep.exists() and bp.exists() and (maestro / "maestro-v3.0.0.csv").exists():
            rows = {r["midi_filename"]: r for r in csv.DictReader(open(maestro / "maestro-v3.0.0.csv", encoding="utf-8"))}
            by_perf = {r["midi_performance"]: r for r in meta}
            tracked = load_beats(bp)
            real = []
            for r in json.loads(rep.read_text()):
                perf = r.get("performance")
                if "offset_s" not in r or perf not in tracked or perf not in ann:
                    continue
                mm = by_perf[perf]["maestro_midi_performance"].replace("{maestro}/", "")
                src = transcribed / Path(rows[mm]["audio_filename"]).with_suffix(".mid")
                if not src.exists():
                    continue
                clean, _ = load_performance(asap / perf)
                notes, pedal = load_performance(src)
                notes[:, 1:3] -= float(r["offset_s"])
                lo, hi = float(clean[:, 1].min()) - MARGIN, float(clean[:, 2].max()) + MARGIN
                notes, _ = crop(notes, pedal, lo, hi)
                bt, bd = tracked[perf]
                keep = (bt >= lo) & (bt <= hi)
                real.append({"performance": perf, "piece": by_perf[perf]["folder"], "dataset": "asap",
                             "notes": notes, "gt": annotation_grid(ann[perf]), "beats": (bt[keep], bd[keep])})
            out[f"{split}_transcribed"] = real
    return out


def _windows(job):
    """(performance, piece, source, bar_start, perf_t0, stored notes, window grid) per window."""
    split_dir, vocab_path, idx = job
    from src.preprocessing.store import SegmentStore
    from src.preprocessing.vocab import Vocab
    store, vocab = SegmentStore(split_dir), Vocab.load(vocab_path)
    out = []
    for i in idx:
        notes, _, toks = store.get(i)
        m = store.meta[i]
        out.append((m["performance"], m.get("composition") or m.get("piece"), m.get("source", "clean"),
                    int(m["bar_start"]), float(m["perf_t0"]), notes,
                    window_grid(notes[:, 1], notes[:, 0], vocab.decode(toks.tolist()))))
    return out


def from_alignment(split_dir: Path, vocab_path: Path, dataset: str, workers: int, min_beats: int = 16,
                   only: set | None = None) -> list:
    """Every ``dataset`` performance of a processed split as meter-model performances whose ground
    truth is the score timed through its windows' alignments: one entry per segment (a run of
    consecutive bars, ``aligned_grids``), with the stitched notes of its span (+ MARGIN).
    ``only``: just these performances."""
    from src.preprocessing.store import SegmentStore
    store = SegmentStore(split_dir)
    idx = [i for i, m in enumerate(store.meta) if m.get("dataset") == dataset
           and (only is None or m["performance"] in only)]
    jobs = [(str(split_dir), str(vocab_path), idx[k:k + 300]) for k in range(0, len(idx), 300)]
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_windows, jobs))
    else:
        results = [_windows(j) for j in jobs]
    by = defaultdict(list)
    for res in results:
        for w in res:
            by[w[0]].append(w)
    perfs = []
    for perf in sorted(by):
        ws = by[perf]
        t, _, bars, beats, ts = merge_performance_bars([(b0, t0, g) for _, _, _, b0, t0, _, g in ws])
        notes = stitch_notes([(t0, n) for _, _, _, _, t0, n, _ in ws])
        segs = aligned_grids(t, bars, beats, ts, min_beats)
        for k, gt in enumerate(segs):
            lo, hi = gt["t"][0] - MARGIN, gt["t"][-1] + MARGIN
            seg_notes = notes[(notes[:, 1] >= lo) & (notes[:, 1] <= hi)]
            if len(seg_notes) < 8:
                continue
            perfs.append({"performance": perf, "segment": k, "segments": len(segs), "span": (float(lo), float(hi)),
                          "piece": ws[0][1], "dataset": dataset, "notes": seg_notes, "gt": gt, "beats": None,
                          "real_notes": ws[0][2] != "clean"})
    return perfs


def check(aligned: list, annotated: list) -> dict:
    """The alignment-derived ground truth of ASAP performances against their annotations: beat /
    downbeat F-measure of the aligned beats, and the share of aligned beats whose time signature
    is the annotated one (at the nearest annotated beat within MATCH_TOL)."""
    ann = {p["performance"]: p["gt"] for p in annotated}
    fb, fd, ts_ok, ts_n = [], [], 0, 0
    for p in aligned:
        gt = ann.get(p["performance"])
        if gt is None:
            continue
        a = p["gt"]
        lo, hi = a["t"][0] - MATCH_TOL, a["t"][-1] + MATCH_TOL
        inside = (gt["t"] >= lo) & (gt["t"] <= hi)
        ref_t, ref_d = gt["t"][inside], gt["t"][inside & gt["down"]]
        fb.append(f_measure(ref_t, a["t"]))
        fd.append(f_measure(ref_d, a["t"][a["down"]]))
        j = np.clip(np.searchsorted(gt["t"], a["t"]), 1, len(gt["t"]) - 1)
        j = np.where(np.abs(gt["t"][j - 1] - a["t"]) < np.abs(gt["t"][j] - a["t"]), j - 1, j)
        near = np.abs(gt["t"][j] - a["t"]) <= MATCH_TOL
        ts_ok += int(sum(a["ts_names"][i] == gt["ts_names"][j[i]] for i in np.nonzero(near)[0]))
        ts_n += int(near.sum())
    return {"segments": len(fb), "beat_f": float(np.mean(fb)) if fb else None,
            "downbeat_f": float(np.mean(fd)) if fd else None, "ts_agreement": ts_ok / max(ts_n, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--asap", default="data/raw/asap_bd")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--transcribed", default="data/transcribed_transkun")
    ap.add_argument("--maestro", default="data/raw/maestro")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-periscope", action="store_true", help="ASAP (and its transcriptions) only")
    ap.add_argument("--check", action="store_true",
                    help="also time the ASAP test performances from their alignments and compare with the annotations")
    ap.add_argument("--gangqinpu", default=None, help="the Gangqinpu release (its data/*.parquet): build its splits too")
    ap.add_argument("--gangqinpu-only", action="store_true", help="build only the Gangqinpu splits")
    ap.add_argument("--gangqinpu-limit", type=int, default=0, help="rows per split (0 = all; for a quick look)")
    ap.add_argument("--gangqinpu-min-confidence", type=float, default=GridFilter.min_confidence)
    ap.add_argument("--gangqinpu-keep-inferred", action="store_true",
                    help="keep bars whose time signature the release inferred (not written in the score)")
    args = ap.parse_args(argv)
    data = Path(args.data)
    if args.gangqinpu_only and not args.gangqinpu:
        raise SystemExit("--gangqinpu-only needs --gangqinpu <release folder>")
    out = {} if args.gangqinpu_only else build(Path(args.asap), data, Path(args.transcribed), Path(args.maestro))
    vocab = data / "vocab.json"
    if args.gangqinpu:
        flt = GridFilter(min_confidence=args.gangqinpu_min_confidence, keep_inferred=args.gangqinpu_keep_inferred)
        reports = {}
        for split in gangqinpu.SPLITS:
            if not gangqinpu.split_files(Path(args.gangqinpu), split):
                print(f"gangqinpu {split}: not downloaded, skipped", flush=True)
                continue
            perfs, reports[split] = gangqinpu.build_split(Path(args.gangqinpu), split, flt, args.gangqinpu_limit)
            out[f"{split}_gangqinpu"] = perfs
            print(f"gangqinpu {split}: {json.dumps(reports[split])}", flush=True)
        (data / "meter").mkdir(parents=True, exist_ok=True)
        (data / "meter" / "gangqinpu_report.json").write_text(
            json.dumps({"filter": vars(flt), "splits": reports}, indent=1))
    if not args.no_periscope and not args.gangqinpu_only:
        for name, split in PERISCOPE.items():
            if (data / split / "index.npy").exists():
                perfs = from_alignment(data / split, vocab, "pianocore", args.workers)
                if perfs:
                    out[name] = perfs
    (data / "meter").mkdir(parents=True, exist_ok=True)
    for split, perfs in out.items():
        with open(data / "meter" / f"{split}.pkl", "wb") as f:
            pickle.dump(perfs, f, protocol=pickle.HIGHEST_PROTOCOL)
        beats = sum(len(p["gt"]["t"]) for p in perfs)
        n_perf = len({p["performance"] for p in perfs})
        n_piece = len({p.get("piece") for p in perfs})
        print(f"{split}: {n_perf} performances ({len(perfs)} segments, {n_piece} pieces), {beats} ground-truth "
              f"beats -> {data / 'meter' / (split + '.pkl')}")
    if args.check and not args.gangqinpu_only and (data / "test" / "index.npy").exists():
        res = check(from_alignment(data / "test", vocab, "asap", args.workers), out.get("test", []))
        (data / "meter" / "check.json").write_text(json.dumps(res, indent=1))
        print(f"alignment ground truth vs ASAP annotations (test, {res['segments']} segments): beat F "
              f"{res['beat_f']:.3f}, downbeat F {res['downbeat_f']:.3f}, time signature {res['ts_agreement']:.3f}")


if __name__ == "__main__":
    main()
