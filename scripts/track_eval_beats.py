"""The whole-piece evaluation's recordings tracked by our Beat This! (a model set, src/bt_models.py),
for scripts/system_scores.py --beats-tag <tag>.

    python scripts/track_eval_beats.py --models outputs/beat_this/bt_ft --tag ft
        -> data/processed/test_transcribed/beats_tracked_ft.npz      (ASAP's real recordings)
           outputs/system_eval_gangqinpu/beats_gangqinpu_ft.npz      (Gangqinpu's test recordings)

Each recording is tracked by the model that never heard its piece: for test pieces that is the
model fine-tuned on every fold, since the export never takes a test piece; a test piece found among
the fine-tuning's pieces stops it (a leak). As scripts/track_beats.py and
scripts/gangqinpu_inputs.py did with the released Beat This!, on the same clocks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_gangqinpu_audio import audio_path  # noqa: E402
from scripts.track_beats import audio_files  # noqa: E402
from src.beats import beats_path, compare, load_beats, save_beats  # noqa: E402
from src.bt_models import ModelSet, piece_of  # noqa: E402


def main(argv=None, track=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="a model set folder (models.json)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", default="test_transcribed", help="ASAP's real-recording splits")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--asap", default="data/raw/asap_bd")
    ap.add_argument("--maestro", default="data/raw/maestro")
    ap.add_argument("--gangqinpu", default="outputs/system_eval_gangqinpu/manifest.json")
    ap.add_argument("--audio", default="data/raw/gangqinpu/audio")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    ms = ModelSet(args.models)
    if track is None:
        from scripts.track_beats import Tracker
        device = args.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tr = Tracker(device)
        track = lambda wav, ck: tr(wav, ck)  # noqa: E731

    def checkpoint(piece: str, what: str) -> str:
        if piece in ms.pieces:
            raise SystemExit(f"{what}: its piece {piece} was in the fine-tuning (fold {ms.pieces[piece]}): a leak")
        return str(ms.full)
    report = {}
    splits = [s for s in args.splits.split(",") if s]
    audio = audio_files(Path(args.asap), Path(args.maestro)) if splits else {}   # ASAP's recordings: only for its splits
    for split in splits:
        d = Path(args.data) / split
        rep = d / "report.json"
        if not rep.exists():
            print(f"skip {split}: no {rep}")
            continue
        gt = load_beats(beats_path(d, "gt"))
        beats, scores = {}, []
        for r in json.loads(rep.read_text()):
            perf = r.get("performance")
            wav = audio.get(perf)
            if "offset_s" not in r or wav is None or not Path(wav).exists():
                continue
            t, dn = track(wav, checkpoint(piece_of(perf), perf))
            beats[perf] = (np.asarray(t, np.float64) - float(r["offset_s"]), np.asarray(dn, bool))
            if perf in gt:
                scores.append(compare(*gt[perf], *beats[perf]))
        save_beats(beats_path(d, f"tracked_{args.tag}"), beats)
        report[split] = {"performances": len(beats),
                         "beat_f": float(np.mean([c["beat_f"] for c in scores])) if scores else None,
                         "downbeat_f": float(np.mean([c["downbeat_f"] for c in scores])) if scores else None}
        print(f"{split}: {json.dumps(report[split])}", flush=True)
    man = Path(args.gangqinpu)
    if man.exists():
        beats = {}
        for e in json.loads(man.read_text()):
            perf = e["performance"]
            if e.get("condition") != "gangqinpu" or perf in beats:
                continue
            wav = audio_path(Path(args.audio), "test", perf.split("/", 1)[1])
            if not wav.exists():
                continue
            piece = f"gangqinpu/{e['group']}" if e.get("group") else perf
            t, dn = track(wav, checkpoint(piece, perf))
            beats[perf] = (np.asarray(t, np.float64), np.asarray(dn, bool))
        save_beats(man.parent / f"beats_gangqinpu_{args.tag}.npz", beats)
        report["gangqinpu"] = {"performances": len(beats)}
        print(f"gangqinpu: {len(beats)} recordings", flush=True)
    out = Path(args.models) / f"eval_beats_{args.tag}.json"
    out.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
