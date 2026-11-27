"""Piano recording (or performance MIDI) -> MusicXML, from the command line. Same pipelines as the
web demo (demo/app.py): Transkun V2, then v2 (the meter model's grid of the piece, then the note
labeller) or the seq2seq score model decoding window by window.

    python scripts/audio_to_score.py recording.wav --v2                  # -> outputs/demo/recording/
    python scripts/audio_to_score.py recording.wav --v2 --labeller outputs/runs/nl_base/best.pt \\
                                     --meter outputs/runs/meter_v2_nonote/best.pt
    python scripts/audio_to_score.py recording.mp3 --ckpt outputs/runs/sf_base_reg_cab/best.pt --out my_dir
    python scripts/audio_to_score.py performance.mid --mode chained      # skip transcription; unguarded context
    python scripts/audio_to_score.py performance.mid --v2 --beats pm2s   # beats from the notes (PM2S), not
                                                                         # Beat This! on the MIDI rendered to audio

Writes transcription.mid, tokens.txt, score.musicxml, score.mid and result.json (timings, counts).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import BEAT_SOURCES, GUARD_MARGIN, MODES, ScorePipeline, V2Pipeline  # noqa: E402
from src.release import DELETE_ABOVE, GLISSANDO, GRID_TEMPO, READABLE, TEMPO_MARKS, TIDY, TWO_VOICES, default_path, rule_settings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="audio file (wav, mp3, flac, ...) or .mid")
    ap.add_argument("--v2", action="store_true", help="v2: meter model + note labeller (else the seq2seq model)")
    ap.add_argument("--labeller", default=default_path("v2_labeller"),
                    help="v2: the note labeller (default: models/, else outputs/runs/)")
    ap.add_argument("--meter", default=default_path("v2_meter"), help="v2: the meter model")
    ap.add_argument("--ckpt", default=default_path("seq2seq"), help="the seq2seq model")
    ap.add_argument("--vocab", default=None, help="default: next to the checkpoint, else data/processed")
    ap.add_argument("--out", default=None, help="default: outputs/demo/<input name>")
    ap.add_argument("--mode", default=MODES[0], choices=list(MODES),
                    help="seq2seq: meter (default): guarded + consistent time signatures; guarded: previous output "
                         "as context, degenerate windows redone without it; chained: previous output as context, "
                         "unguarded; none: no context")
    ap.add_argument("--guard-margin", type=float, default=GUARD_MARGIN)
    ap.add_argument("--delete-above", type=float, default=DELETE_ABOVE,
                    help="v2: leave a note out only when the labeller's p(delete) is above this (0.5: as "
                         f"trained, 1: keep every note; default {DELETE_ABOVE}, as the demo)")
    ap.add_argument("--beats", default="beat_this", choices=list(BEAT_SOURCES),
                    help="v2: where the beats come from: Beat This! (the recording; a MIDI file rendered to audio) "
                         "or PM2S on the notes (src/pm2s_beats.py)")
    ap.add_argument("--no-tidy", action="store_true",
                    help="v2: keep written lengths that run just past the next onset (default: trim them)")
    ap.add_argument("--no-glissando", action="store_true", help="v2: write each note of a glissando, not its first and last joined by a gliss. line")
    ap.add_argument("--no-timing-check", action="store_true", help="v2: keep the labeller's triplets even where the timing says straight eighths (src/timing.py)")
    ap.add_argument("--no-beat-decode", action="store_true", help="v2: take each note's likeliest split of the quarter, not one decoded a beat at a time with the timing (src/beat_decode.py)")
    ap.add_argument("--no-timing-beats", action="store_true", help="v2: keep the labeller's triplets in beats whose timing fits straight sixteenths better (src/timing.py check_beats)")
    ap.add_argument("--no-readable", action="store_true", help="v2: keep rests of a 16th or less, overlaps of a 16th or less, and stems as written (src/readable.py)")
    ap.add_argument("--no-fix-jumps", action="store_true", help="v2: keep the meter model's stretched bars (a jump of beats in one beat's time) as decoded")
    ap.add_argument("--no-two-voices", action="store_true", help="v2: each note into the first voice free when it starts, however many voices that takes (not two to a staff, ordered by pitch)")
    ap.add_argument("--no-tempo-marks", action="store_true", help="v2: print no tempo marks in the MusicXML (src/tempo.py)")
    ap.add_argument("--constant-tempo", action="store_true", help="v2: play the score MIDI at one tempo for the whole piece, not the grid's tempo bar by bar")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.v2:
        rules = rule_settings()   # the benchmark's verdict on the correction rules (src/release.py)
        pipe = V2Pipeline(args.labeller, args.meter, args.vocab, args.device, tidy=TIDY and not args.no_tidy,
                          glissando=GLISSANDO and not args.no_glissando,
                          timing_check=rules["timing_check"] and not args.no_timing_check,
                          readable=READABLE and not args.no_readable, fix_jumps=rules["fix_jumps"] and not args.no_fix_jumps,
                          two_voices=TWO_VOICES and not args.no_two_voices,
                          timing_beats=rules["timing_beats"] and not args.no_timing_beats,
                          beat_decode=rules["beat_decode"] and not args.no_beat_decode,
                          tempo_marks=TEMPO_MARKS and not args.no_tempo_marks,
                          grid_tempo=GRID_TEMPO and not args.constant_tempo)
    else:
        pipe = ScorePipeline(args.ckpt, args.vocab, args.device)
    print(f"model: {pipe.info}", flush=True)
    out = Path(args.out or Path("outputs/demo") / Path(args.input).stem)

    def progress(stage, frac, msg):
        print(f"  [{stage}] {msg}", end="\r" if stage == "score" else "\n", flush=True)

    if args.v2:
        res = pipe.run(args.input, out, progress=progress, delete_above=args.delete_above, beat_source=args.beats)
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
