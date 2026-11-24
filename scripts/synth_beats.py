"""Beat This! on the clean test MIDI rendered to audio: the beats a MIDI input gets (src/synth.py).

    python scripts/synth_beats.py           # -> outputs/system_eval/beats_synth.npz (+ .json)

For the system-level comparison (scripts/system_scores.py): a clean performance MIDI has no
recording, so a system that needs beats (v2; a seq2seq beat model with --midi-beats synth) gets
the beats Beat This! hears in the MIDI rendered by src/synth.py, as the demo does for an uploaded
MIDI file. Each ASAP piece is tracked by the cross-validation model that never trained on it
(scripts/track_beats.py checkpoint_for), and scored against the ASAP annotations
(data/processed/test/beats_gt.npz) next to the same performances' real-audio beats
(test_transcribed/beats_tracked.json), so the synthesized audio's cost is on record.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.track_beats import Tracker, checkpoint_for, load_folds  # noqa: E402
from src.beats import beats_path, compare, load_beats, save_beats  # noqa: E402
from src.preprocessing.performance import load_performance  # noqa: E402
from src.synth import render, write_wav  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/system_eval/manifest.json")
    ap.add_argument("--out", default="outputs/system_eval/beats_synth.npz")
    ap.add_argument("--folds", default="data/raw/beat_this/asap_8-folds.split")
    ap.add_argument("--gt", default="data/processed/test", help="split folder with the annotated beats (beats_gt.npz)")
    ap.add_argument("--real", default="data/processed/test_transcribed",
                    help="split folder whose beats_tracked.json has the real-audio beat F per performance")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    entries = [e for e in json.loads(Path(args.manifest).read_text()) if e["condition"] == "clean"]
    folds = load_folds(Path(args.folds))
    gt = load_beats(beats_path(Path(args.gt), "gt"))
    real_rep = Path(args.real) / "beats_tracked.json"
    real = {r["performance"]: r for r in json.loads(real_rep.read_text())["performances"]} if real_rep.exists() else {}
    tracker = Tracker(device)
    beats, rows = {}, []
    with tempfile.TemporaryDirectory() as tmp:
        for k, e in enumerate(entries):
            notes, _ = load_performance(e["midi"])
            wav = write_wav(Path(tmp) / "performance.wav", render(notes[:, :4]))
            ck = checkpoint_for(e["performance"], folds)
            t, d = tracker(wav, ck)
            beats[e["performance"]] = (t, d)
            row = {"performance": e["performance"], "checkpoint": ck, "beats": int(len(t)), "downbeats": int(d.sum())}
            if e["performance"] in gt:
                c = compare(*gt[e["performance"]], t, d)
                row.update(beat_f=c["beat_f"], downbeat_f=c["downbeat_f"], octave=c["octave"])
            if "beat_f" in real.get(e["performance"], {}):
                row.update(real_audio_beat_f=real[e["performance"]]["beat_f"],
                           real_audio_downbeat_f=real[e["performance"]]["downbeat_f"])
            rows.append(row)
            print(f"{k + 1}/{len(entries)} {e['performance']}: {ck}, {len(t)} beats"
                  + (f", beat F {row['beat_f']:.3f}, downbeat F {row['downbeat_f']:.3f}" if "beat_f" in row else "")
                  + (f" (real audio {row['real_audio_beat_f']:.3f} / {row['real_audio_downbeat_f']:.3f})"
                     if "real_audio_beat_f" in row else ""), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_beats(out, beats)

    def mean(key, rs):
        v = [r[key] for r in rs if key in r]
        return float(np.mean(v)) if v else None
    both = [r for r in rows if "beat_f" in r and "real_audio_beat_f" in r]
    summary = {"performances": len(rows), "beat_f": mean("beat_f", rows), "downbeat_f": mean("downbeat_f", rows),
               "with_real_audio": len(both), "beat_f_same": mean("beat_f", both),
               "downbeat_f_same": mean("downbeat_f", both), "real_audio_beat_f": mean("real_audio_beat_f", both),
               "real_audio_downbeat_f": mean("real_audio_downbeat_f", both)}
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "performances": rows}, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
