"""Does Beat This! track a render like the recording? (src/render.py)

    python scripts/compare_render.py --n 30            # SoundFonts from $SOUNDFONTS, each on its own
        -> outputs/render_check/render_check.json + render_check.md

ASAP performances whose MAESTRO recording is here: the recording, and FluidSynth renders of the same
performance's MIDI (MAESTRO's Disklavier file: the same playing, the same clock), both tracked by the
Beat This! fold model that never heard the performance, both compared with ASAP's annotated beats
(src/beats.py compare). One performance per piece; never a test piece. A SoundFont whose renders
track like the recordings (beat F, downbeat F, tempo-octave errors, missed and extra beats) can stand
in for recordings in the meter model's data (scripts/track_meter_data.py beat_this_render).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.track_beats import Tracker, checkpoint_for, load_folds  # noqa: E402
from src.beats import compare  # noqa: E402
from src.preprocessing.performance import estimate_offset, load_performance  # noqa: E402


def pick(asap: Path, maestro: Path, folds: dict, splits: dict, ann: dict, n: int) -> list:
    """Up to ``n`` (metadata row, recording, MIDI): one per piece, with a recording here, annotated,
    a held-out fold model, not a test piece."""
    out, seen = [], set()
    for r in csv.DictReader(open(asap / "metadata.csv", encoding="utf-8")):
        wav = Path(r["maestro_audio_performance"].replace("{maestro}", str(maestro)))
        midi = Path(r["maestro_midi_performance"].replace("{maestro}", str(maestro)))
        perf = r["midi_performance"]
        if (r["folder"] in seen or splits.get(r["folder"]) == "test" or perf not in ann or not wav.exists()
                or not midi.exists() or checkpoint_for(perf, folds) == "final0"):
            continue
        seen.add(r["folder"])
        out.append((r, wav, midi))
        if len(out) >= n:
            break
    return out


def summary(rows: list) -> dict:
    def mean(key, normal=False):
        v = [r[key] for r in rows if key in r and (not normal or r.get("octave") == "normal")]
        return round(float(np.mean(v)), 4) if v else None
    return {"performances": len(rows), "beat_f": mean("beat_f"), "downbeat_f": mean("downbeat_f"),
            "wrong_octave": sum(r.get("octave") != "normal" for r in rows),
            "miss_at_tempo": mean("miss", True), "extra_at_tempo": mean("extra", True)}


def main(argv=None, tracker=None, renderer_for=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--asap", default="data/raw/asap_bd")
    ap.add_argument("--maestro", default="data/raw/maestro")
    ap.add_argument("--splits", default="data/processed/splits.json")
    ap.add_argument("--folds", default="data/raw/beat_this/asap_8-folds.split")
    ap.add_argument("--soundfonts", default=None, help="each compared on its own (default: $SOUNDFONTS)")
    ap.add_argument("--fluidsynth", default=None)
    ap.add_argument("--out", default="outputs/render_check")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    from src.render import Renderer, find_soundfonts
    fonts = find_soundfonts(args.soundfonts) if renderer_for is None else [Path("stand-in")]
    if renderer_for is None:
        renderer_for = lambda font: Renderer([font], args.fluidsynth)  # noqa: E731
    if tracker is None:
        device = args.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tr = Tracker(device)
        tracker = lambda wav, ck: tr(wav, ck)  # noqa: E731
    asap, maestro = Path(args.asap), Path(args.maestro)
    folds = load_folds(Path(args.folds))
    sp = json.loads(Path(args.splits).read_text()) if Path(args.splits).exists() else {}
    splits = sp.get("asap", sp)
    ann = json.loads((asap / "asap_annotations.json").read_text())
    chosen = pick(asap, maestro, folds, splits, ann, args.n)
    if not chosen:
        raise SystemExit(f"no ASAP performance with its MAESTRO recording under {maestro}")
    rows = {"recording": []} | {font.name: [] for font in fonts}
    renderers = {font.name: renderer_for(font) for font in fonts}
    for r, wav, midi in chosen:
        perf = r["midi_performance"]
        a = ann[perf]
        gt_t = np.asarray(a["performance_beats"], np.float64)
        gt_d = np.isin(np.round(gt_t, 4), np.round(np.asarray(a["performance_downbeats"], np.float64), 4))
        d, match = estimate_offset(load_performance(asap / perf)[0], load_performance(midi)[0])
        ck = checkpoint_for(perf, folds)
        line = [f"{perf} ({ck}, offset match {match:.2f})"]
        t, dn = tracker(wav, ck)
        rows["recording"].append({"performance": perf, **compare(gt_t, gt_d, np.asarray(t) - d, dn)})
        for name, render in renderers.items():
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "render.wav"
                render(midi, out, perf)
                t, dn = tracker(out, ck)
            rows[name].append({"performance": perf, **compare(gt_t, gt_d, np.asarray(t) - d, dn)})
        line += [f"{k} {v[-1]['beat_f']:.3f}/{v[-1]['downbeat_f']:.3f}" for k, v in rows.items()]
        print(" | ".join(line), flush=True)
    res = {k: summary(v) for k, v in rows.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "render_check.json").write_text(json.dumps({"summary": res, "performances": rows}, indent=1, default=float))
    md = ["| audio | performances | beat F | downbeat F | wrong tempo octave | missed / extra beats at the right tempo |",
          "|---|---|---|---|---|---|"]
    for k, s in res.items():
        md.append(f"| {k} | {s['performances']} | {s['beat_f']} | {s['downbeat_f']} | {s['wrong_octave']} | "
                  f"{s['miss_at_tempo']} / {s['extra_at_tempo']} |")
    (out / "render_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    return res


if __name__ == "__main__":
    main()
