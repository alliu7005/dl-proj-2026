"""Transcribe downloaded MAESTRO audio with Transkun V2 (Yan & Duan, ISMIR 2024).

Transkun V2 is the transcriber PianoCoRe's PERiScoPe performances were made with, so the model sees
the same kind of transcription errors in training (PERiScoPe) and at inference. Its shipped
checkpoint does not extend notes to the sustain-pedal release; sustain pedal is output as CC64.

    python scripts/transcribe.py                     # every downloaded .wav (CUDA if available)
    python scripts/transcribe.py --audio x.wav --out-midi x.mid   # a single file

Writes data/transcribed/<year>/<name>.mid and data/transcribed/pairs.csv (pred, gt, maestro_split)
for scripts/fit_noise.py. Already-transcribed files are skipped.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


class Transcriber:
    """Transkun V2 with the checkpoint shipped in the pip package (``pretrained/2.0.pt``)."""

    def __init__(self, device: str = "cuda"):
        import importlib.resources as ir

        import moduleconf
        import torch

        pre = ir.files("transkun") / "pretrained"
        conf = moduleconf.parseFromFile(str(pre / "2.0.conf"))
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        model = conf["Model"].module.TransKun(conf=conf["Model"].config).to(self.device)
        ckpt = torch.load(str(pre / "2.0.pt"), map_location=self.device)
        model.load_state_dict(ckpt.get("best_state_dict", ckpt.get("state_dict")), strict=False)
        self.model = model.eval()
        self.torch = torch

    def __call__(self, audio_path: str, out_path: str):
        import librosa
        from transkun.Data import writeMidi

        # [channels, T] at the model rate; Transkun takes [T, channels]
        y, _ = librosa.load(str(audio_path), sr=self.model.fs, mono=False)
        x = self.torch.from_numpy(np.atleast_2d(y).T.astype(np.float32))
        # default 16 s chunks (8 s hop); a very dense chunk can exceed a 6 GB GPU, so retry with
        # 8 s chunks, then on the CPU
        attempts = [(self.device, {}), (self.device, {"segmentSizeInSecond": 8, "stepInSecond": 4})]
        if self.device != "cpu":
            attempts.append(("cpu", {}))
        for k, (dev, kw) in enumerate(attempts):
            try:
                self.model.to(dev)
                with self.torch.no_grad():
                    notes = self.model.transcribe(x.to(dev), **kw)
                break
            except (RuntimeError, self.torch.OutOfMemoryError) as e:
                if "out of memory" not in str(e) or k == len(attempts) - 1:
                    raise
                print(f"  out of memory ({dev}, {kw or 'default chunks'}); retrying", flush=True)
            finally:
                if self.device == "cuda":
                    self.torch.cuda.empty_cache()
        self.model.to(self.device)
        writeMidi(notes).write(str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro", default="data/raw/maestro")
    ap.add_argument("--out", default="data/transcribed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audio", help="transcribe just this file (with --out-midi)")
    ap.add_argument("--out-midi")
    args = ap.parse_args()

    if args.audio:
        Transcriber(args.device)(args.audio, args.out_midi or str(Path(args.audio).with_suffix(".mid")))
        return

    root, out = Path(args.maestro), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {r["audio_filename"]: r for r in csv.DictReader(open(root / "maestro-v3.0.0.csv", encoding="utf-8"))}
    wavs = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.wav"))
    print(f"Transkun V2: {len(wavs)} recordings")

    pairs = []
    for i, rel in enumerate(wavs):
        r = rows[rel]
        pred = out / Path(rel).with_suffix(".mid")
        if not pred.exists():
            pred.parent.mkdir(parents=True, exist_ok=True)
            t = time.time()
            # one subprocess per recording: a CUDA out-of-memory error can leave the GPU context
            # unusable, so a crash only costs that file, which is then redone on the CPU
            for dev in dict.fromkeys([args.device, "cpu"]):
                res = subprocess.run([sys.executable, __file__, "--audio", str(root / rel),
                                      "--out-midi", str(pred), "--device", dev],
                                     capture_output=True, text=True)
                if res.returncode == 0 and pred.exists():
                    break
                print(f"  failed on {dev}: {res.stderr.strip().splitlines()[-1] if res.stderr.strip() else '?'}",
                      flush=True)
            else:
                raise RuntimeError(f"could not transcribe {rel}")
            print(f"[{i + 1}/{len(wavs)}] {rel}  {float(r['duration']):.0f}s audio in {time.time() - t:.0f}s"
                  f" ({dev})", flush=True)
        pairs.append((pred.as_posix(), (root / r["midi_filename"]).as_posix(), r["split"]))
    with open(out / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred", "gt", "maestro_split"])
        w.writerows(pairs)
    print("wrote", out / "pairs.csv")


if __name__ == "__main__":
    main()
