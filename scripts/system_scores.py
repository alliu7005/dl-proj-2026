"""Our system on the whole-piece inputs of scripts/system_inputs.py: one MusicXML score per performance.

    python scripts/system_scores.py --run sf_base_reg_cab                 # meter mode (the demo default)
    python scripts/system_scores.py --run sf_base_reg_cab --mode tsearch_beat
    -> outputs/system_eval/<run>_<mode>/<condition>/<slug>.musicxml   (+ summary.json)

Runs the demo pipeline (src/pipeline.py) exactly as the demo does on a MIDI file: windows cut from
the performance itself (no score information), decoded window by window. A model trained with
beats gets, on the real transcriptions, the beats Beat This! tracked in the audio
(data/processed/test_transcribed/beats_tracked.npz, same ASAP timeline as the cropped MIDI): what
the demo gives it for audio. On clean MIDI there is no audio, so no beats, like the demo's MIDI
input. scripts/beyer_dai.py score then rates these scores against the ASAP references.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.beats import beats_path, load_beats  # noqa: E402
from src.pipeline import MODES, ScorePipeline  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sf_base_reg_cab")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--vocab", default="data/processed/vocab.json")
    ap.add_argument("--mode", default=MODES[0], choices=list(MODES))
    ap.add_argument("--manifest", default="outputs/system_eval/manifest.json")
    ap.add_argument("--conditions", default="clean,real")
    ap.add_argument("--out", default="outputs/system_eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--tracked", default="data/processed/test_transcribed",
                    help="split folder whose beats_tracked.npz holds the real transcriptions' beats")
    args = ap.parse_args(argv)
    entries = [e for e in json.loads(Path(args.manifest).read_text()) if e["condition"] in args.conditions.split(",")]
    pipe = ScorePipeline(Path(args.runs_dir) / args.run / args.ckpt, vocab=args.vocab, device=args.device)
    tracked = {}
    bp = beats_path(Path(args.tracked), "tracked")
    if pipe.info["beats"] and bp.exists():
        tracked = load_beats(bp)
    system = f"{args.run}_{args.mode}"
    out = Path(args.out) / system
    done = []
    for k, e in enumerate(entries):
        dst = out / e["condition"] / f"{e['slug']}.musicxml"
        beats = tracked.get(e["performance"]) if e["condition"] == "real" else None
        t = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            res = pipe.run(e["midi"], tmp, mode=args.mode, beats=beats)
            ok = "musicxml" in res.files
            if ok:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(res.files["musicxml"], dst)
        done.append({"condition": e["condition"], "performance": e["performance"], "musicxml": str(dst) if ok else None,
                     "errors": res.errors, "beats": res.stats.get("beats"), "seconds": round(time.perf_counter() - t, 1),
                     **{k_: res.stats.get(k_) for k_ in ("windows", "guard_fallbacks", "meter_fixes", "time_signature")}})
        print(f"{k + 1}/{len(entries)} {e['condition']} {e['performance']}: {'ok' if ok else res.errors} "
              f"({done[-1]['seconds']} s)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"run": args.run, "mode": args.mode, "pieces": done}, indent=1))
    print(f"wrote {out}: {sum(d['musicxml'] is not None for d in done)}/{len(done)} scores")


if __name__ == "__main__":
    main()
