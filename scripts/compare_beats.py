"""PM2S (beats from the notes, src/pm2s_beats.py) against Beat This! on the whole-piece test inputs,
scored against the ASAP beat annotations (src/beats.compare, as every beat result here).

    python scripts/compare_beats.py          # -> outputs/system_eval/beats_compare.{json,md} + beats_pm2s.npz

For each input of outputs/system_eval/manifest.json (scripts/system_inputs.py):
  clean  the ASAP test MIDI: PM2S on its notes, against Beat This! on the MIDI rendered to audio
         (outputs/system_eval/beats_synth.npz, scripts/synth_beats.py: the held-out fold models),
         and Beat This! on the performance's real recording where it has one
  real   the Transkun transcriptions: PM2S on the transcription, against Beat This! on the recording
         (data/processed/test_transcribed/beats_tracked.npz)
Each source's beat / downbeat F-measure, the share of performances at the wrong tempo octave, and
paired: on how many performances PM2S is better. Every source is also scored on the performances
they all have (``common``). PM2S's released weights were trained on all of ASAP (these
performances included), the fold models never heard their piece: optimistic for PM2S until it is
retrained without them (--weights / PM2S_WEIGHTS takes such a checkpoint).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.beats import beats_path, compare, load_beats, save_beats  # noqa: E402
from src.pm2s_beats import PM2SBeats  # noqa: E402
from src.preprocessing.performance import load_performance  # noqa: E402

KEYS = ("beat_f", "downbeat_f")


def score(gt: dict, beats: dict, perfs: list) -> dict:
    """performance -> compare() of its beats (the performances that have both)."""
    return {p: compare(*gt[p], *beats[p]) for p in perfs if p in gt and p in beats}


def summary(scored: dict, perfs: list) -> dict:
    rows = [scored[p] for p in perfs if p in scored]
    if not rows:
        return {"n": 0}
    out = {"n": len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in KEYS}}
    out["wrong_octave"] = float(np.mean([r.get("octave") in ("double", "half") for r in rows]))
    return out


def table(results: dict) -> str:
    lines = ["| condition | beats | performances | beat F | downbeat F | wrong tempo octave | PM2S better (beat / downbeat) |",
             "|---|---|---|---|---|---|---|"]
    for cond, res in results.items():
        for name, s in res["common"].items():
            if not s.get("n"):
                continue
            better = res["pm2s_better"].get(name)
            b = f"{better['beat']} / {better['downbeat']} of {better['n']}" if better else ""
            lines.append(f"| {cond} | {name} | {s['n']} | {s['beat_f']:.3f} | {s['downbeat_f']:.3f} | "
                         f"{s['wrong_octave']:.2f} | {b} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/system_eval/manifest.json")
    ap.add_argument("--gt", default="data/processed/test", help="split folder with the annotated beats (beats_gt.npz)")
    ap.add_argument("--tracked", default="data/processed/test_transcribed",
                    help="split folder whose beats_tracked.npz holds Beat This!'s beats of the recordings")
    ap.add_argument("--synth-beats", default="outputs/system_eval/beats_synth.npz")
    ap.add_argument("--weights", default=None, help="PM2S checkpoint (default: the released one, cached)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs/system_eval")
    args = ap.parse_args(argv)
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    entries = json.loads(Path(args.manifest).read_text())
    gt = load_beats(beats_path(Path(args.gt), "gt"))
    if not gt:
        raise SystemExit(f"no {beats_path(Path(args.gt), 'gt')}: scripts/build_beats.py makes it")
    recording = load_beats(beats_path(Path(args.tracked), "tracked"))
    synth = load_beats(args.synth_beats)
    tracker = PM2SBeats(args.weights, device=device)
    pm2s, results, t0 = {}, {}, time.perf_counter()
    for cond in ("clean", "real"):
        es = [e for e in entries if e["condition"] == cond]
        mine = {}
        for e in es:
            notes, _ = load_performance(e["midi"])
            mine[e["performance"]] = tracker(notes)
        pm2s[cond] = mine
        perfs = [e["performance"] for e in es]
        sources = {"PM2S (notes)": score(gt, mine, perfs)}
        if cond == "clean":
            sources["Beat This! (rendered MIDI)"] = score(gt, synth, perfs)
        sources["Beat This! (recording)"] = score(gt, recording, perfs)
        common = [p for p in perfs if all(p in s for s in sources.values() if s)]
        res = {"all": {k: summary(v, perfs) for k, v in sources.items()},
               "common": {k: summary(v, common) for k, v in sources.items()}, "pm2s_better": {},
               "performances": {p: {k: {m: v[p][m] for m in KEYS + ("octave",) if m in v[p]}
                                    for k, v in sources.items() if p in v} for p in perfs}}
        ours = sources["PM2S (notes)"]
        for name, other in sources.items():
            if name == "PM2S (notes)" or not other:
                continue
            both = [p for p in perfs if p in ours and p in other]
            res["pm2s_better"][name] = {"n": len(both),
                                        "beat": int(sum(ours[p]["beat_f"] > other[p]["beat_f"] for p in both)),
                                        "downbeat": int(sum(ours[p]["downbeat_f"] > other[p]["downbeat_f"] for p in both))}
        results[cond] = res
        print(f"{cond}: " + "; ".join(f"{k} {s['beat_f']:.3f} / {s['downbeat_f']:.3f} (n {s['n']})"
                                       for k, s in res["common"].items() if s.get("n")), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_beats(out / "beats_pm2s.npz", pm2s["clean"])
    save_beats(out / "beats_pm2s_real.npz", pm2s["real"])
    (out / "beats_compare.json").write_text(json.dumps({"weights": tracker.weights, "device": device,
                                                        "seconds": round(time.perf_counter() - t0, 1),
                                                        "results": results}, indent=1))
    md = ("PM2S (beats from the notes) against Beat This!, ASAP test inputs, scored against the annotations "
          "on the performances every source has. PM2S's released weights were trained on all of ASAP "
          "(these performances included); Beat This!'s fold models never heard their piece.\n\n"
          + table(results) + "\n")
    (out / "beats_compare.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
