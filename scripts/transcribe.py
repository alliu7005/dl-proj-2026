"""Transcribe downloaded MAESTRO audio with Kong et al. (2021) or Transkun V2 (Yan & Duan, ISMIR 2024).

Two transcribers, each writing to its own folder so their outputs never mix:

* ``--model kong`` (default; -> data/transcribed/): Kong et al.'s ``piano_transcription_inference``,
  the transcriber named in the proposal. All current noise statistics (configs/noise_stats.json),
  the *_transcribed splits and the baselines were built from these. Same model and post-processing
  as the package's ``PianoTranscription.transcribe`` (identical output up to ~1 ms GPU rounding),
  ~20x faster: batched GPU inference (the package hard-codes batch_size=1), audio decoding
  prefetched in threads, CPU-heavy note decoding in worker processes.
      env: Python <= 3.12, ``pip install piano_transcription_inference audioread librosa``

* ``--model transkun`` (-> data/transcribed_transkun/): Transkun V2, the transcriber PianoCoRe's
  PERiScoPe performances were made with, so training on PERiScoPe and testing on these use the
  same kind of transcription errors. Its shipped checkpoint does not extend notes to the
  sustain-pedal release; sustain pedal is output as CC64. One subprocess per recording, so a CUDA
  out-of-memory error only costs that file (retried with smaller chunks, then on the CPU).
      env: ``pip install --no-deps transkun==2.0.1`` + moduleconf, mir-eval, pydub, torch-optimizer, soxr

    python scripts/transcribe.py                          # Kong, every downloaded .wav
    python scripts/transcribe.py --model transkun
    python scripts/transcribe.py --model transkun --audio x.wav --out-midi x.mid   # one file

Writes <out>/<year>/<name>.mid and <out>/pairs.csv (pred, gt, maestro_split) for
scripts/fit_noise.py. Already-transcribed files are skipped.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

DEFAULT_OUT = {"kong": "data/transcribed", "transkun": "data/transcribed_transkun"}


# --------------------------------------------------------------------------- Kong et al. (2021)

def _load(path: str, sr: int) -> np.ndarray:
    import librosa  # librosa/soundfile read WAV directly (the package's loader needs ffmpeg)
    return librosa.load(path, sr=sr, mono=True)[0]


def _postprocess(output_dict: dict, params: dict, midi_path: str) -> int:
    """Runs in a worker process: model outputs -> note/pedal events -> MIDI file."""
    from piano_transcription_inference.utilities import RegressionPostProcessor, write_events_to_midi
    pp = RegressionPostProcessor(**params)
    notes, pedals = pp.output_dict_to_midi_events(output_dict)
    Path(midi_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = midi_path + ".part"
    write_events_to_midi(start_time=0, note_events=notes, pedal_events=pedals, midi_path=tmp)
    Path(tmp).replace(midi_path)
    return len(notes)


def transcribe_kong(root: Path, out: Path, todo: list[str], device: str, batch_size: int, workers: int):
    import torch
    from piano_transcription_inference import PianoTranscription, sample_rate

    device = device if torch.cuda.is_available() else "cpu"
    pt = PianoTranscription(device=device)
    model = pt.model.eval()
    seg = pt.segment_samples
    params = dict(frames_per_second=pt.frames_per_second, classes_num=pt.classes_num,
                  onset_threshold=pt.onset_threshold, offset_threshold=pt.offset_threshod,
                  frame_threshold=pt.frame_threshold, pedal_offset_threshold=pt.pedal_offset_threshold)
    print(f"Kong: {len(todo)} to transcribe on {device} (batch {batch_size}, {workers} workers)", flush=True)
    t_start, audio_sec = time.time(), 0.0
    with ThreadPoolExecutor(2) as loader, ProcessPoolExecutor(workers) as post:
        futures = [loader.submit(_load, str(root / w), sample_rate) for w in todo[:2]]
        pending = []
        for i, rel in enumerate(todo):
            audio = futures[i].result()
            if i + 2 < len(todo):  # keep two recordings decoding ahead of the GPU
                futures.append(loader.submit(_load, str(root / todo[i + 2]), sample_rate))
            n = len(audio)
            audio = np.concatenate([audio, np.zeros(int(np.ceil(n / seg)) * seg - n, audio.dtype)])
            segments = pt.enframe(audio[None, :], seg)
            outs: dict[str, list] = {}
            with torch.inference_mode():
                for b in range(0, len(segments), batch_size):
                    x = torch.from_numpy(np.asarray(segments[b:b + batch_size], np.float32)).to(device)
                    for k, v in model(x).items():
                        outs.setdefault(k, []).append(v.float().cpu().numpy())
            output_dict = {k: pt.deframe(np.concatenate(v))[0:n] for k, v in outs.items()}
            pending.append(post.submit(_postprocess, output_dict, params, str(out / Path(rel).with_suffix(".mid"))))
            audio_sec += n / sample_rate
            print(f"[{i + 1}/{len(todo)}] {rel}  ({audio_sec / (time.time() - t_start):.1f}x real time so far)",
                  flush=True)
        for f in pending:
            f.result()
    if todo:
        print(f"done: {audio_sec / 3600:.2f} h of audio in {(time.time() - t_start) / 60:.1f} min")


# --------------------------------------------------------------------------- Transkun V2

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


def transcribe_transkun(root: Path, out: Path, todo: list[str], rows: dict, device: str):
    print(f"Transkun V2: {len(todo)} to transcribe", flush=True)
    for i, rel in enumerate(todo):
        pred = out / Path(rel).with_suffix(".mid")
        pred.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        # one subprocess per recording: a CUDA out-of-memory error can leave the GPU context
        # unusable, so a crash only costs that file, which is then redone on the CPU
        for dev in dict.fromkeys([device, "cpu"]):
            res = subprocess.run([sys.executable, __file__, "--model", "transkun", "--audio", str(root / rel),
                                  "--out-midi", str(pred), "--device", dev], capture_output=True, text=True)
            if res.returncode == 0 and pred.exists():
                break
            print(f"  failed on {dev}: {res.stderr.strip().splitlines()[-1] if res.stderr.strip() else '?'}",
                  flush=True)
        else:
            raise RuntimeError(f"could not transcribe {rel}")
        print(f"[{i + 1}/{len(todo)}] {rel}  {float(rows[rel]['duration']):.0f}s audio in {time.time() - t:.0f}s"
              f" ({dev})", flush=True)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["kong", "transkun"], default="kong")
    ap.add_argument("--maestro", default="data/raw/maestro")
    ap.add_argument("--out", default=None, help="default: data/transcribed (kong), data/transcribed_transkun")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16, help="kong: 10 s segments per GPU batch")
    ap.add_argument("--workers", type=int, default=6, help="kong: CPU processes for post-processing")
    ap.add_argument("--only", nargs="*", help="only these audio files (relative to --maestro)")
    ap.add_argument("--audio", help="transkun: transcribe just this file (with --out-midi)")
    ap.add_argument("--out-midi")
    args = ap.parse_args()

    if args.audio:
        if args.model != "transkun":
            raise SystemExit("--audio is only supported with --model transkun")
        Transcriber(args.device)(args.audio, args.out_midi or str(Path(args.audio).with_suffix(".mid")))
        return

    root, out = Path(args.maestro), Path(args.out or DEFAULT_OUT[args.model])
    out.mkdir(parents=True, exist_ok=True)
    rows = {r["audio_filename"]: r for r in csv.DictReader(open(root / "maestro-v3.0.0.csv", encoding="utf-8"))}
    wavs = args.only or sorted(p.relative_to(root).as_posix() for p in root.rglob("*.wav"))
    todo = [w for w in wavs if not (out / Path(w).with_suffix(".mid")).exists()]
    if args.model == "kong":
        transcribe_kong(root, out, todo, args.device, args.batch_size, args.workers)
    else:
        transcribe_transkun(root, out, todo, rows, args.device)

    pairs = [((out / Path(w).with_suffix(".mid")).as_posix(), (root / rows[w]["midi_filename"]).as_posix(),
              rows[w]["split"]) for w in sorted(p.relative_to(root).as_posix() for p in root.rglob("*.wav"))
             if (out / Path(w).with_suffix(".mid")).exists()]
    with open(out / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred", "gt", "maestro_split"])
        w.writerows(pairs)
    print(f"wrote {out / 'pairs.csv'} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
