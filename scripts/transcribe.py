"""Transcribe downloaded MAESTRO audio with Kong et al. (2021) (``piano_transcription_inference``).

Runs in the separate CUDA env (Python 3.12):
    uv venv --python 3.12 .venv-transcribe
    uv pip install --python .venv-transcribe/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv-transcribe/Scripts/python.exe piano_transcription_inference pretty_midi

    .venv-transcribe/Scripts/python scripts/transcribe.py            # every downloaded .wav

Writes data/transcribed/<year>/<name>.mid and data/transcribed/pairs.csv (pred, gt, maestro_split)
for scripts/fit_noise.py. Already-transcribed files are skipped.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro", default="data/raw/maestro")
    ap.add_argument("--out", default="data/transcribed")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    import librosa
    from piano_transcription_inference import PianoTranscription, sample_rate

    root, out = Path(args.maestro), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {r["audio_filename"]: r for r in csv.DictReader(open(root / "maestro-v3.0.0.csv", encoding="utf-8"))}
    wavs = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.wav"))
    device = args.device if torch.cuda.is_available() else "cpu"
    model = PianoTranscription(device=device)

    pairs = []
    for i, rel in enumerate(wavs):
        r = rows[rel]
        pred = out / Path(rel).with_suffix(".mid")
        if not pred.exists():
            pred.parent.mkdir(parents=True, exist_ok=True)
            t = time.time()
            # librosa/soundfile reads WAV directly (the package's own loader needs ffmpeg)
            audio, _ = librosa.load(str(root / rel), sr=sample_rate, mono=True)
            model.transcribe(audio, str(pred))
            print(f"[{i + 1}/{len(wavs)}] {rel}  {float(r['duration']):.0f}s audio in {time.time() - t:.0f}s",
                  flush=True)
        pairs.append((pred.as_posix(), (root / r["midi_filename"]).as_posix(), r["split"]))
    with open(out / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred", "gt", "maestro_split"])
        w.writerows(pairs)
    print("wrote", out / "pairs.csv")


if __name__ == "__main__":
    main()
