"""Piano recording (or performance MIDI) -> MusicXML, from the command line. Same pipelines as the
web demo (demo/app.py): Transkun V2, then v2 (the meter model's grid of the piece, then the note
labeller) or the seq2seq score model decoding window by window.

    python scripts/audio_to_score.py recording.wav --v2                  # -> outputs/demo/recording/
    python scripts/audio_to_score.py recording.wav --v2 --labeller outputs/runs/nl_base/best.pt \\
                                     --meter outputs/runs/meter_v2_nonote/best.pt
    python scripts/audio_to_score.py recording.mp3 --ckpt outputs/runs/sf_base_reg_cab/best.pt --out my_dir
    python scripts/audio_to_score.py performance.mid --mode chained      # skip transcription; unguarded context

Writes transcription.mid, tokens.txt, score.musicxml, score.mid and result.json (timings, counts).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import GUARD_MARGIN, MODES, ScorePipeline, V2Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="audio file (wav, mp3, flac, ...) or .mid")
    ap.add_argument("--v2", action="store_true", help="v2: meter model + note labeller (else the seq2seq model)")
    ap.add_argument("--labeller", default="outputs/runs/nl_base_gn/best.pt", help="v2: the note labeller")
    ap.add_argument("--meter", default="outputs/runs/meter_v3/best.pt", help="v2: the meter model")
    ap.add_argument("--ckpt", default="outputs/runs/sf_base_reg_cab/best.pt", help="the seq2seq model")
    ap.add_argument("--vocab", default=None, help="default: next to the checkpoint, else data/processed")
    ap.add_argument("--out", default=None, help="default: outputs/demo/<input name>")
    ap.add_argument("--mode", default=MODES[0], choices=list(MODES),
                    help="seq2seq: meter (default): guarded + consistent time signatures; guarded: previous output "
                         "as context, degenerate windows redone without it; chained: previous output as context, "
                         "unguarded; none: no context")
    ap.add_argument("--guard-margin", type=float, default=GUARD_MARGIN)
    ap.add_argument("--delete-above", type=float, default=0.8,
                    help="v2: leave a note out only when the labeller's p(delete) is above this (0.5: as "
                         "evaluated, 1: keep every note; default 0.8, as the demo)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.v2:
        pipe = V2Pipeline(args.labeller, args.meter, args.vocab, args.device)
    else:
        pipe = ScorePipeline(args.ckpt, args.vocab, args.device)
    print(f"model: {pipe.info}", flush=True)
    out = Path(args.out or Path("outputs/demo") / Path(args.input).stem)

    def progress(stage, frac, msg):
        print(f"  [{stage}] {msg}", end="\r" if stage == "score" else "\n", flush=True)

    if args.v2:
        res = pipe.run(args.input, out, progress=progress, delete_above=args.delete_above)
    else:
        res = pipe.run(args.input, out, mode=args.mode, progress=progress, guard_margin=args.guard_margin)
    print()
    for k, v in res.stats.items():
        print(f"  {k:16s} {v}")
    for e in res.errors:
        print(f"  warning: {e}")
    for k, v in res.files.items():
        print(f"  {k:20s} {v}")


if __name__ == "__main__":
    main()
