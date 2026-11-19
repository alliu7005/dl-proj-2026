"""Piano recording (or performance MIDI) -> MusicXML, from the command line. Same pipeline as the
web demo (demo/app.py): Transkun V2, then the score model decoding window by window.

    python scripts/audio_to_score.py recording.wav                       # -> outputs/demo/recording/
    python scripts/audio_to_score.py recording.mp3 --ckpt outputs/runs/sf_base_reg_ca/best.pt --out my_dir
    python scripts/audio_to_score.py performance.mid --mode chained      # skip transcription; unguarded context

Writes transcription.mid, tokens.txt, score.musicxml, score.mid and result.json (timings, counts).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import GUARD_MARGIN, MODES, ScorePipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="audio file (wav, mp3, flac, ...) or .mid")
    ap.add_argument("--ckpt", default="outputs/runs/sf_base_reg_ca/best.pt")
    ap.add_argument("--vocab", default=None, help="default: next to the checkpoint, else data/processed")
    ap.add_argument("--out", default=None, help="default: outputs/demo/<input name>")
    ap.add_argument("--mode", default=MODES[0], choices=list(MODES),
                    help="meter (default): guarded + consistent time signatures; guarded: previous output as "
                         "context, degenerate windows redone without it; chained: previous output as context, "
                         "unguarded; none: no context")
    ap.add_argument("--guard-margin", type=float, default=GUARD_MARGIN)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    pipe = ScorePipeline(args.ckpt, args.vocab, args.device)
    print(f"model: {pipe.info}", flush=True)
    out = Path(args.out or Path("outputs/demo") / Path(args.input).stem)

    def progress(stage, frac, msg):
        print(f"  [{stage}] {msg}", end="\r" if stage == "score" else "\n", flush=True)

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
