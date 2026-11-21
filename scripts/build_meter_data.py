"""Whole performances for the meter model (v2, Phase 2): notes + annotated grid (+ tracked beats).

    python scripts/build_meter_data.py            # -> data/processed/meter/<split>.pkl

For every aligned ASAP performance of the train / val / test pieces (data/processed/splits.json):
its performance MIDI's notes and its annotated grid (src/meter.py annotation_grid: beats with
quarter positions, bar starts, time and key signatures). For the real-transcription splits
(<split>_transcribed, when their report.json exists): the Transkun transcription shifted onto the
ASAP timeline and cut to the performance (as scripts/system_inputs.py does), and the beats Beat
This! tracked (beats_tracked.npz). The meter model trains on the clean ones with the tracker's
errors simulated (src/beats.py BeatNoise) and on the transcribed ones with the real tracker.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.system_inputs import MARGIN, crop  # noqa: E402
from src.beats import beats_path, load_beats  # noqa: E402
from src.meter import annotation_grid  # noqa: E402
from src.preprocessing.performance import load_performance  # noqa: E402


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
            perfs.append({"performance": perf, "notes": notes, "gt": annotation_grid(a), "beats": None})
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
                real.append({"performance": perf, "notes": notes, "gt": annotation_grid(ann[perf]),
                             "beats": (bt[keep], bd[keep])})
            out[f"{split}_transcribed"] = real
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--asap", default="data/raw/asap_bd")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--transcribed", default="data/transcribed_transkun")
    ap.add_argument("--maestro", default="data/raw/maestro")
    args = ap.parse_args(argv)
    data = Path(args.data)
    out = build(Path(args.asap), data, Path(args.transcribed), Path(args.maestro))
    (data / "meter").mkdir(parents=True, exist_ok=True)
    for split, perfs in out.items():
        with open(data / "meter" / f"{split}.pkl", "wb") as f:
            pickle.dump(perfs, f, protocol=pickle.HIGHEST_PROTOCOL)
        beats = sum(len(p["gt"]["t"]) for p in perfs)
        print(f"{split}: {len(perfs)} performances, {beats} annotated beats -> {data / 'meter' / (split + '.pkl')}")


if __name__ == "__main__":
    main()
