"""Transcribe downloaded MAESTRO audio with Kong et al. (2021) (``piano_transcription_inference``).

Same model and post-processing as the package's ``PianoTranscription.transcribe`` (identical
output), but much faster:
  * batched GPU inference (the package hard-codes batch_size=1),
  * audio decoding/resampling prefetched in background threads,
  * the CPU-heavy note/pedal decoding + MIDI writing runs in worker processes,
    so the GPU never waits for it.

Runs in the separate CUDA env (Python 3.12):
    uv venv --python 3.12 .venv-transcribe
    uv pip install --python .venv-transcribe/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv-transcribe/Scripts/python.exe piano_transcription_inference audioread librosa pretty_midi

    .venv-transcribe/Scripts/python scripts/transcribe.py            # every downloaded .wav

Writes data/transcribed/<year>/<name>.mid and data/transcribed/pairs.csv (pred, gt, maestro_split)
for scripts/fit_noise.py. Already-transcribed files are skipped.
"""
from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro", default="data/raw/maestro")
    ap.add_argument("--out", default="data/transcribed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16, help="10 s segments per GPU batch")
    ap.add_argument("--workers", type=int, default=6, help="CPU processes for post-processing")
    ap.add_argument("--only", nargs="*", help="only these audio files (relative to --maestro)")
    args = ap.parse_args()

    import torch
    from piano_transcription_inference import PianoTranscription, sample_rate

    root, out = Path(args.maestro), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {r["audio_filename"]: r for r in csv.DictReader(open(root / "maestro-v3.0.0.csv", encoding="utf-8"))}
    wavs = args.only or sorted(p.relative_to(root).as_posix() for p in root.rglob("*.wav"))
    todo = [w for w in wavs if not (out / Path(w).with_suffix(".mid")).exists()]
    device = args.device if torch.cuda.is_available() else "cpu"
    pt = PianoTranscription(device=device)
    model = pt.model.eval()
    seg = pt.segment_samples
    params = dict(frames_per_second=pt.frames_per_second, classes_num=pt.classes_num,
                  onset_threshold=pt.onset_threshold, offset_threshold=pt.offset_threshod,
                  frame_threshold=pt.frame_threshold, pedal_offset_threshold=pt.pedal_offset_threshold)
    print(f"{len(wavs)} recordings, {len(todo)} to transcribe on {device} "
          f"(batch {args.batch_size}, {args.workers} post-processing workers)", flush=True)

    t_start, audio_sec = time.time(), 0.0
    with ThreadPoolExecutor(2) as loader, ProcessPoolExecutor(args.workers) as post:
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
                for b in range(0, len(segments), args.batch_size):
                    x = torch.from_numpy(np.asarray(segments[b:b + args.batch_size], np.float32)).to(device)
                    for k, v in model(x).items():
                        outs.setdefault(k, []).append(v.float().cpu().numpy())
            output_dict = {k: pt.deframe(np.concatenate(v))[0:n] for k, v in outs.items()}
            pred = out / Path(rel).with_suffix(".mid")
            pending.append(post.submit(_postprocess, output_dict, params, str(pred)))
            audio_sec += n / sample_rate
            el = time.time() - t_start
            print(f"[{i + 1}/{len(todo)}] {rel}  ({audio_sec / el:.1f}x real time so far)", flush=True)
        for f in pending:
            f.result()

    pairs = [((out / Path(w).with_suffix(".mid")).as_posix(), (root / rows[w]["midi_filename"]).as_posix(),
              rows[w]["split"]) for w in sorted(p.relative_to(root).as_posix() for p in root.rglob("*.wav"))
             if (out / Path(w).with_suffix(".mid")).exists()]
    with open(out / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred", "gt", "maestro_split"])
        w.writerows(pairs)
    if todo:
        print(f"done: {audio_sec / 3600:.2f} h of audio in {(time.time() - t_start) / 60:.1f} min; "
              f"wrote {out / 'pairs.csv'} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
