"""PianoCoRe performances re-transcribed with Transkun V2 from their recordings
(scripts/fetch_pianocore_audio.py), so their notes carry the transcriber errors the system meets at
test time, not another transcriber's.

    python scripts/retranscribe_pianocore.py --dir data/raw/pianocore/atepp --out data/raw/pianocore/atepp_tk

PianoCoRe aligned its scores to the ORIGINAL transcription (exact note copies, src/preprocessing/
pianocore.py match_raw_to_refined), so that one keeps giving the time map to the score; ours is put
on its clock by the constant offset between the two (same recording: estimate_offset), and becomes
the model input (``transcription_midi_bytes``, read by pianocore_performance). A recording whose
transcription does not match the original (another upload, a sped-up video, a different
performance) is left out (--min-match). Each kept row also records its recording and where the
original clock is in it (``audio_path``, ``audio_offset_s``: recording time = clock + offset), for
the Beat This! export's real recordings. Output: <out>/rows-*.parquet, the scores, selection.parquet,
retranscribe_report.json; Transkun's MIDI kept in <out>/transcriptions/ (a run again goes on).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def midi_bytes(notes: np.ndarray, pedal: np.ndarray) -> bytes:
    from src.render import write_midi
    with tempfile.TemporaryDirectory() as t:
        f = write_midi(notes, Path(t) / "x.mid", pedal if len(pedal) else None)
        return f.read_bytes()


def main(argv=None, transcribe=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="fetched rows and their audio/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-match", type=float, default=0.5,
                    help="share of the original transcription's first notes ours must match")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.preprocessing.performance import estimate_offset, load_performance
    src, out = Path(args.dir), Path(args.out)
    (out / "transcriptions").mkdir(parents=True, exist_ok=True)
    if transcribe is None:
        from scripts.transcribe import Transcriber
        tr = Transcriber(args.device)
        transcribe = lambda audio, midi: tr(str(audio), str(midi))  # noqa: E731
    report, kept_ids, t0 = {}, [], time.perf_counter()
    for f in sorted(src.glob("rows-*.parquet")):
        target = out / f.name
        rows = pq.read_table(f).to_pylist()
        new = []
        for r in rows:
            pid = str(r["id"])
            wav = src / "audio" / f"{pid}.mp3"
            if not wav.exists():
                report[pid] = "no recording"
                continue
            mid = out / "transcriptions" / f"{pid}.mid"
            if not mid.exists():
                try:
                    transcribe(wav, str(mid) + ".part")
                    Path(str(mid) + ".part").replace(mid)
                except Exception as e:  # noqa: BLE001
                    report[pid] = f"transkun failed: {str(e)[:150]}"
                    continue
            ours, pedal = load_performance(mid)
            orig, _ = load_performance(r["performance_midi_bytes"])
            if len(ours) < 8 or len(orig) < 8:
                report[pid] = "too few notes"
                continue
            d, match = estimate_offset(orig, ours, n_ref=400, n_cand=40)
            if match < args.min_match:
                report[pid] = f"no match ({match:.2f})"
                continue
            ours = ours.copy()
            ours[:, 1:3] -= d                       # onto the original transcription's clock
            pedal = np.asarray(pedal, np.float64) - d if len(pedal) else pedal
            new.append({**r, "transcription_midi_bytes": midi_bytes(ours, pedal), "audio_path": str(wav),
                        "audio_offset_s": float(d), "transcription_match": round(float(match), 3)})
            report[pid] = f"kept ({match:.2f})"
            kept_ids.append(pid)
        if new:
            tbl = pa.Table.from_pylist(new)
            pq.write_table(tbl, target)
        elif target.exists():
            target.unlink()
        print(f"{f.name}: {len(new)} of {len(rows)} kept ({time.perf_counter() - t0:.0f} s)", flush=True)
    for s in src.glob("scores-*.parquet"):              # the scores, as fetched
        if not (out / s.name).exists():
            shutil.copy2(s, out / s.name)
    if (src / "selection.parquet").exists():
        sel = pd.read_parquet(src / "selection.parquet")
        sel[sel["id"].astype(str).isin(kept_ids)].to_parquet(out / "selection.parquet")
    counts = pd.Series([v.split(" (")[0] for v in report.values()]).value_counts().to_dict()
    (out / "retranscribe_report.json").write_text(json.dumps({"counts": counts, "performances": report}, indent=1))
    print(f"{out}: {counts}", flush=True)
    return report


if __name__ == "__main__":
    main()
