"""Whole-piece inputs and references from Gangqinpu's test split: the system-level comparison of
scripts/system_inputs.py on music outside classical piano (pop, folk, jazz, classical arrangements).

    python scripts/gangqinpu_inputs.py                     # -> outputs/system_eval_gangqinpu/manifest.json
    python scripts/gangqinpu_inputs.py --n 30 --no-transcribe

Picks ``--n`` test performances that have a downloaded recording (scripts/fetch_gangqinpu_audio.py),
one per score (the release's ``score_group_id``), spread over genres in their proportions and
otherwise at random (seeded). For each:
  reference       the written score (src/preprocessing/gangqinpu.py score_measures, every measure,
                  graces and ties; no pedal) as MusicXML (src/preprocessing/detokenize.py)
  gangqinpu       the recording's Transkun V2 notes the release aligned to the score, as MIDI with its
                  pedal: real timing and lengths, the transcriber's missed notes, none of its extra ones
                  (the alignment dropped them), like ASAP's clean MIDI is to its transcription
  gangqinpu_real  the recording transcribed again by Transkun V2 here (scripts/transcribe.py), cut to
                  the aligned notes' span plus 0.5 s: what the demo gets from an upload of the
                  recording. Left out with --no-transcribe, or if Transkun is not installed
and Beat This! on the recording (the final0 model; it never trained on Gangqinpu):
outputs/system_eval_gangqinpu/beats_gangqinpu.npz, keyed by performance, the beats every system
gets on both conditions (the inputs keep the recording's clock). Then:

    python scripts/system_scores.py --v2 nl_base_gn+meter_v3_tp --tidy --delete-above 0.95 \\
        --manifest outputs/system_eval_gangqinpu/manifest.json --out outputs/system_eval_gangqinpu \\
        --conditions gangqinpu,gangqinpu_real
    python scripts/beyer_dai.py score --systems all --manifest outputs/system_eval_gangqinpu/manifest.json \\
        --out outputs/system_eval_gangqinpu --conditions gangqinpu,gangqinpu_real ...
    python scripts/compare_systems.py --results outputs/system_eval_gangqinpu/results.json \\
        --out outputs/system_eval_gangqinpu/paired.md

The test split is the release's: made jointly with ASAP by score similarity, nothing in it shares a
score with any training data of ours.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_gangqinpu_audio import audio_path  # noqa: E402
from scripts.system_inputs import MARGIN, crop, write_midi  # noqa: E402
from src.beats import save_beats  # noqa: E402
from src.preprocessing import gangqinpu as G  # noqa: E402

CONDITIONS = ("gangqinpu", "gangqinpu_real")


def slug(performance: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", performance)


def choose(rows: list, n: int, seed: int = 0) -> list:
    """``n`` rows, one per score, spread over genres in proportion (largest remainders), at random
    within each genre (``seed``)."""
    rng = np.random.default_rng(seed)
    by_score = {}
    for r in sorted(rows, key=lambda r: str(r["sample_id"])):
        by_score.setdefault(r.get("score_group_id") or r["sample_id"], r)
    by_genre = defaultdict(list)
    for r in by_score.values():
        by_genre[r.get("genre") or "unknown"].append(r)
    total = sum(len(v) for v in by_genre.values())
    n = min(n, total)
    quota = {g: n * len(v) / total for g, v in by_genre.items()}
    take = {g: int(q) for g, q in quota.items()}
    for g in sorted(quota, key=lambda g: (-(quota[g] - take[g]), g))[: n - sum(take.values())]:
        take[g] += 1
    out = []
    for g in sorted(by_genre):
        v = by_genre[g]
        out += [v[i] for i in sorted(rng.choice(len(v), take[g], replace=False))]
    return sorted(out, key=lambda r: str(r["sample_id"]))


def main(argv=None, beat_this=None, transcribe=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default="data/raw/gangqinpu")
    ap.add_argument("--audio", default="data/raw/gangqinpu/audio")
    ap.add_argument("--n", type=int, default=60, help="test performances (one per score)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-transcribe", action="store_true", help="no gangqinpu_real (no Transkun run)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs/system_eval_gangqinpu")
    args = ap.parse_args(argv)
    out = Path(args.out)
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [r for r in G.iter_rows(Path(args.release), "test", columns=G.SCORE_COLUMNS)
            if audio_path(Path(args.audio), "test", str(r["sample_id"])).exists()]
    if not rows:
        raise SystemExit(f"no test recordings under {Path(args.audio) / 'test'}: scripts/fetch_gangqinpu_audio.py")
    if beat_this is None:
        from scripts.track_beats import Tracker
        bt = Tracker(device)
        beat_this = lambda path: bt(path, "final0")  # noqa: E731
    if transcribe is None and not args.no_transcribe:
        try:
            from scripts.transcribe import Transcriber
            tk = Transcriber(device)
            transcribe = lambda wav, mid: tk(str(wav), str(mid))  # noqa: E731
        except Exception as e:  # noqa: BLE001  (Transkun not installed here: the aligned notes only)
            print(f"Transkun unavailable ({type(e).__name__}: {e}): no gangqinpu_real", flush=True)
    from src.preprocessing.detokenize import tokens_to_musicxml
    from src.preprocessing.performance import load_performance

    entries, beats, skipped = [], {}, []
    for k, row in enumerate(choose(rows, args.n, args.seed)):
        perf = f"gangqinpu/{row['sample_id']}"
        s = slug(perf)
        res = G.labeller_performance(row)
        if isinstance(res, str):
            skipped.append({"performance": perf, "why": res})
            continue
        notes, pedal, _, measures, info = res
        ref = out / "references" / f"{s}.musicxml"
        ref.parent.mkdir(parents=True, exist_ok=True)
        try:
            tokens_to_musicxml(G.score_tokens(measures), str(ref))
        except Exception as e:  # noqa: BLE001
            skipped.append({"performance": perf, "why": f"reference: {type(e).__name__}: {e}"[:200]})
            continue
        wav = audio_path(Path(args.audio), "test", str(row["sample_id"]))
        try:
            beats[perf] = beat_this(wav)
        except Exception as e:  # noqa: BLE001  (an unreadable download)
            skipped.append({"performance": perf, "why": f"Beat This!: {type(e).__name__}: {e}"[:200]})
            continue
        common = {"performance": perf, "slug": s, "reference": str(ref), "genre": row.get("genre") or "unknown",
                  "group": row.get("score_group_id")}
        mid = out / "inputs" / "gangqinpu" / f"{s}.mid"
        write_midi(notes, pedal, mid)
        entries.append({"condition": "gangqinpu", "midi": str(mid), "notes": int(len(notes)), **common})
        if transcribe is not None:
            real = out / "inputs" / "gangqinpu_real" / f"{s}.mid"
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    raw = Path(tmp) / "raw.mid"
                    transcribe(wav, raw)
                    rn, rp = load_performance(raw)
                rn, rp = crop(rn, rp, float(notes[:, 1].min()) - MARGIN, float(notes[:, 2].max()) + MARGIN)
                write_midi(rn, rp, real)
                entries.append({"condition": "gangqinpu_real", "midi": str(real), "notes": int(len(rn)), **common})
            except Exception as e:  # noqa: BLE001
                skipped.append({"performance": perf, "why": f"Transkun: {type(e).__name__}: {e}"[:200]})
        print(f"{k + 1}: {perf} ({common['genre']}): {len(notes)} aligned notes, {len(measures)} bars", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    save_beats(out / "beats_gangqinpu.npz", beats)
    (out / "manifest.json").write_text(json.dumps(entries, indent=1))
    (out / "inputs_report.json").write_text(json.dumps({"chosen": args.n, "recordings": len(rows),
                                                        "skipped": skipped, "seed": args.seed}, indent=1))
    n = {c: sum(e["condition"] == c for e in entries) for c in CONDITIONS}
    print(f"wrote {out / 'manifest.json'}: {n} | skipped {len(skipped)}: {skipped[:3]}")


if __name__ == "__main__":
    main()
