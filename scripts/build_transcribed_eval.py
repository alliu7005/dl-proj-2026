"""Build an evaluation split whose inputs are REAL Kong transcriptions instead of clean MIDI.

For every ASAP performance in ``--asap-split`` that has MAESTRO audio and a transcription in
data/transcribed/, the transcription is shifted onto the ASAP performance timeline (ASAP files can
be cut/offset relative to MAESTRO), segmented with exactly the same bars as the clean split, and
written to data/processed/<asap-split>_transcribed/ with the same vocab. Segment i there pairs with
the clean segment that has the same (performance, bar_start), so clean-vs-real inputs can be
compared on identical targets.

    python scripts/build_transcribed_eval.py --asap-split test

The ``maestro_split`` meta field records whether Kong's model was trained on that audio
(``train``); report those separately from ``validation``/``test`` recordings.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.performance import load_performance  # noqa: E402
from src.preprocessing.score import load_score  # noqa: E402
from src.preprocessing.segment import SegmentConfig, segment_pair  # noqa: E402
from src.preprocessing.split import composition_id  # noqa: E402
from src.preprocessing.store import SegmentStore, write_split  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402


def estimate_offset(ref: np.ndarray, other: np.ndarray, tol: float = 0.05, n_ref: int = 200) -> tuple[float, float]:
    """Offset d such that other_onset - d ~= ref_onset. Returns (d, fraction of ref notes matched)."""
    a = ref[:n_ref]
    by_pitch = {p: np.sort(other[other[:, 0] == p, 1]) for p in np.unique(a[:, 0])}
    cands = np.concatenate([by_pitch[a[k, 0]] - a[k, 1] for k in range(min(5, len(a)))])

    def score(d):
        hits = 0
        for p, t in zip(a[:, 0], a[:, 1]):
            arr = by_pitch[p]
            j = np.searchsorted(arr, t + d)
            hits += any(abs(arr[jj] - (t + d)) <= tol for jj in (j - 1, j) if 0 <= jj < len(arr))
        return hits / len(a)

    best = max(cands, key=score, default=0.0)
    m = [o - t for p, t in zip(a[:, 0], a[:, 1]) for o in by_pitch[p] if abs(o - t - best) <= tol]
    d = float(np.median(m)) if m else float(best)
    return d, score(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--asap-split", default="test")
    ap.add_argument("--maestro", default="data/raw/maestro")
    ap.add_argument("--transcribed", default="data/transcribed")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cfg = yaml.safe_load(open(args.config))
    root, out = Path(cfg["asap_root"]), Path(cfg["out_dir"])
    seg_cfg = SegmentConfig(**cfg["segment"])
    # same windows as the clean eval split; keep every window (transcriptions are noisy by design)
    seg_cfg = replace(seg_cfg, stride=seg_cfg.bars_per_segment, min_pitch_f1=0.0,
                      max_input_notes=seg_cfg.max_input_notes * 2)
    vocab = Vocab.load(out / "vocab.json")
    clean = SegmentStore(out / args.asap_split)
    clean_keys = {(m["performance"], m["bar_start"]): i for i, m in enumerate(clean.meta)}

    ann = json.load(open(root / "asap_annotations.json"))
    maestro = {r["midi_filename"]: r for r in csv.DictReader(open(Path(args.maestro) / "maestro-v3.0.0.csv", encoding="utf-8"))}
    pieces = json.load(open(cfg["fixed_split"]))["pieces"] if cfg.get("fixed_split") else \
        json.load(open(out / "splits.json"))
    scores, segs, report = {}, [], []
    for row in csv.DictReader(open(root / "metadata.csv")):
        mm = row["maestro_midi_performance"].replace("{maestro}/", "")
        if pieces.get(row["folder"]) != args.asap_split or mm not in maestro:
            continue
        pred = Path(args.transcribed) / Path(maestro[mm]["audio_filename"]).with_suffix(".mid")
        perf = row["midi_performance"]
        if not pred.exists() or not any(k[0] == perf for k in clean_keys):
            report.append({"performance": perf, "skipped": "no transcription" if not pred.exists() else "not in clean split"})
            continue
        clean_notes, _ = load_performance(root / perf)
        notes, pedal = load_performance(pred)
        d, frac = estimate_offset(clean_notes, notes)
        notes[:, 1:3] -= d
        pedal = pedal - d if len(pedal) else pedal
        if row["folder"] not in scores:
            scores[row["folder"]] = load_score(root / row["xml_score"])
        meta = {"dataset": "asap", "piece": row["folder"], "composition": composition_id("asap", row["folder"]),
                "performance": perf, "split": f"{args.asap_split}_transcribed", "source": "kong2021",
                "maestro_split": maestro[mm]["split"]}
        s, st = segment_pair(notes, pedal, ann[perf], scores[row["folder"]], seg_cfg, meta)
        s = [x for x in s if (perf, x["bar_start"]) in clean_keys]
        for x in s:  # targets are ground truth: same tokens as the clean segment (incl. pedal from clean CC64)
            x["tokens"] = vocab.decode(clean.get(clean_keys[(perf, x["bar_start"])])[2].tolist())
        segs += s
        report.append({"performance": perf, "offset_s": round(d, 3), "offset_match": round(frac, 3),
                       "maestro_split": maestro[mm]["split"], "segments": len(s)})
        print(f"{perf}: offset {d:+.3f}s (match {frac:.2f}), MAESTRO {maestro[mm]['split']}, {len(s)} segments")

    sizes = (len(segs), sum(len(s["notes"]) for s in segs), sum(len(s["pedal"]) for s in segs),
             sum(len(s["tokens"]) for s in segs))
    dst = out / f"{args.asap_split}_transcribed"
    write_split(dst, iter(segs), sizes, vocab)
    json.dump(report, open(dst / "report.json", "w"), indent=1)
    print(f"{len(segs)} segments from {sum('segments' in r for r in report)} performances "
          f"(MAESTRO split of the audio: {dict(Counter(s['maestro_split'] for s in segs))}) -> {dst}")


if __name__ == "__main__":
    main()
