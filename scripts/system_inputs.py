"""Whole-piece inputs for the system-level comparison (ours vs Beyer & Dai): one performance MIDI
per test performance and condition, with the ASAP reference score it is scored against.

    python scripts/system_inputs.py            # -> outputs/system_eval/manifest.json (+ inputs/real/*.mid)

Every system then turns each MIDI into one MusicXML score of the whole piece (scripts/system_scores.py
for ours, scripts/beyer_dai.py predict for Beyer & Dai), and scripts/beyer_dai.py score rates them all
with the same metrics (MUSTER and score similarity, as in Beyer & Dai's evaluation). Unlike the
window metrics of scripts/evaluate_pieces.py, nothing here depends on how a system cuts a piece up.

Conditions:
  clean  the ASAP test performances' own MIDI (Beyer & Dai's test set: 14 pieces, 59 performances)
  real   the same performances as Transkun V2 transcribed them from the MAESTRO audio (the ones in
         data/processed/test_transcribed): shifted onto the ASAP timeline by the offset
         scripts/build_transcribed.py found, and cut to the ASAP performance's span, since a MAESTRO
         recording can hold more than one ASAP piece (a prelude and its fugue). Times stay on the
         ASAP timeline, so the tracked beats (beats_tracked.npz) line up with the notes.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.performance import load_performance  # noqa: E402
from src.preprocessing.store import SegmentStore  # noqa: E402

MARGIN = 0.5  # seconds kept before the first / after the last note of the ASAP performance


def slug(performance: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", performance.rsplit(".", 1)[0])


def crop(notes: np.ndarray, pedal: np.ndarray, lo: float, hi: float):
    """The notes starting inside [lo, hi] and the pedal presses overlapping it (clipped)."""
    keep = (notes[:, 1] >= lo) & (notes[:, 1] <= hi)
    ped = pedal[(pedal[:, 1] > lo) & (pedal[:, 0] < hi)] if len(pedal) else pedal.reshape(-1, 2)
    if len(ped):
        ped = np.stack([np.maximum(ped[:, 0], lo), np.minimum(ped[:, 1], hi)], axis=1)
    return notes[keep], ped


def write_midi(notes: np.ndarray, pedal: np.ndarray, path: Path) -> None:
    """[N,4] notes as load_performance gives them (pitch, onset, offset, velocity; seconds) + [K,2]
    pedal intervals -> a piano MIDI file."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="Piano")
    for p, on, off, vel in notes:
        inst.notes.append(pretty_midi.Note(velocity=int(np.clip(round(vel), 1, 127)), pitch=int(p),
                                           start=float(on), end=float(max(off, on + 1e-3))))
    for a, b in pedal:
        inst.control_changes += [pretty_midi.ControlChange(64, 127, float(a)), pretty_midi.ControlChange(64, 0, float(b))]
    inst.control_changes.sort(key=lambda c: c.time)
    pm.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def build(data: Path, asap: Path, transcribed: Path, maestro: Path, out: Path) -> list[dict]:
    meta = {r["midi_performance"]: r for r in csv.DictReader(open(asap / "metadata.csv", encoding="utf-8"))}
    entries = []
    perfs = sorted({m["performance"] for m in SegmentStore(data / "test").meta})
    for perf in perfs:
        entries.append({"condition": "clean", "performance": perf, "slug": slug(perf),
                        "midi": str(asap / perf), "reference": str(asap / meta[perf]["xml_score"])})
    rep_path = data / "test_transcribed" / "report.json"
    if rep_path.exists():
        rows = {r["midi_filename"]: r for r in csv.DictReader(open(maestro / "maestro-v3.0.0.csv", encoding="utf-8"))}
        for r in json.loads(rep_path.read_text()):
            if "offset_s" not in r:
                continue
            perf = r["performance"]
            mm = meta[perf]["maestro_midi_performance"].replace("{maestro}/", "")
            src = transcribed / Path(rows[mm]["audio_filename"]).with_suffix(".mid")
            path = out / "inputs" / "real" / f"{slug(perf)}.mid"
            n = real_input(asap / perf, src, float(r["offset_s"]), path)
            entries.append({"condition": "real", "performance": perf, "slug": slug(perf), "midi": str(path),
                            "reference": str(asap / meta[perf]["xml_score"]), "notes": n,
                            "maestro_split": r.get("maestro_split")})
    return entries


def real_input(clean_midi: Path, transcription: Path, offset_s: float, out: Path) -> int:
    """Write the transcription of one ASAP performance: shifted onto the ASAP timeline by
    ``offset_s`` (scripts/build_transcribed.py), cut to the performance's span plus MARGIN.
    Returns the number of notes kept."""
    clean, _ = load_performance(clean_midi)
    notes, pedal = load_performance(transcription)
    notes[:, 1:3] -= offset_s  # onsets and offsets (load_performance's columns 1, 2)
    pedal = pedal - offset_s if len(pedal) else pedal
    lo = float(clean[:, 1].min()) - MARGIN
    hi = float(clean[:, 2].max()) + MARGIN
    notes, pedal = crop(notes, pedal, lo, hi)
    write_midi(notes, pedal, out)
    return int(len(notes))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--transcribed", default="data/transcribed_transkun")
    ap.add_argument("--maestro", default="data/raw/maestro")
    ap.add_argument("--out", default="outputs/system_eval")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    entries = build(Path(cfg["out_dir"]), Path(cfg["asap_root"]), Path(args.transcribed), Path(args.maestro), out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(entries, indent=1))
    n = {c: sum(e["condition"] == c for e in entries) for c in ("clean", "real")}
    print(f"wrote {out / 'manifest.json'}: {n['clean']} clean, {n['real']} real performances")


if __name__ == "__main__":
    main()
