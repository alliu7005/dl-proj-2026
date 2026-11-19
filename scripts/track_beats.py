"""Beat This! on the MAESTRO audio of the real-transcription splits -> <split>/beats_tracked.npz.

    pip install beat-this rotary-embedding-torch einops       # once (environment.yml has them)
    python scripts/track_beats.py                             # test / train / val _transcribed
    python scripts/track_beats.py --splits test_transcribed --device cpu

Beat This! (Foscarin, Schlüter & Widmer, ISMIR 2024; MIT licence, weights downloaded on first use)
was trained on ASAP among 17 other datasets, so its released model has heard most ASAP recordings.
Each performance is therefore tracked by the cross-validation model that did NOT train on it: the
authors' 8-fold split (beat_this_annotations, asap/8-folds.split, downloaded once to --folds) puts
every ASAP track in one fold k, and checkpoint ``fold<k>`` was trained without that fold. ASAP
tracks not in the split (the paper left out recordings with rubato annotations) never were in
training, and get ``final0``. The beats are shifted onto the ASAP performance timeline by the
same offset scripts/build_transcribed.py applied to the transcription (its report.json), so they
line up with the transcribed notes. Writes <split>/beats_tracked.json: the checkpoint per
performance and the beat / downbeat F-measure against the ground truth (beats_gt.npz) if present.

``Tracker`` is also what the demo uses (src/pipeline.py) on uploaded audio, with ``final0``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.beats import beats_path, f_measure, load_beats, save_beats  # noqa: E402

D = Path("data/processed")
FOLDS_URL = "https://raw.githubusercontent.com/CPJKU/beat_this_annotations/main/asap/8-folds.split"


class Tracker:
    """Beat This! beat and downbeat tracking: ``tracker(audio_path, checkpoint)`` -> (times, down)
    in seconds of the audio. One model per checkpoint, loaded on first use."""

    def __init__(self, device: str = "cpu"):
        try:
            from beat_this.inference import File2Beats  # noqa: F401
        except ImportError as e:
            raise ImportError("Beat This! is not installed: pip install beat-this rotary-embedding-torch einops") from e
        self.device = device
        self._models = {}

    def __call__(self, audio, checkpoint: str = "final0"):
        from beat_this.inference import File2Beats
        if checkpoint not in self._models:
            self._models[checkpoint] = File2Beats(checkpoint_path=checkpoint, device=self.device, dbn=False)
        beats, downbeats = self._models[checkpoint](str(audio))
        beats, downbeats = np.asarray(beats, np.float64), np.asarray(downbeats, np.float64)
        if len(downbeats) > len(beats):  # the README and File2File name the two outputs differently
            beats, downbeats = downbeats, beats
        beats = np.unique(beats)
        down = np.isin(np.round(beats, 4), np.round(downbeats, 4))  # downbeats are snapped to beats
        return beats, down


def asap_stem(performance: str) -> str:
    """ASAP path -> the track name Beat This!'s splits use: Bach/Fugue/bwv_846/Shi05M.mid -> Bach_Fugue_bwv_846_Shi05M."""
    return performance[:-4].replace("/", "_") if performance.endswith(".mid") else performance.replace("/", "_")


def load_folds(path: Path) -> dict:
    """ASAP track stem -> fold (0-7), from Beat This!'s 8-folds.split (downloaded if missing)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FOLDS_URL, timeout=60) as r:
            path.write_bytes(r.read())
    folds = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[1].strip().isdigit():
            folds[parts[0].strip()] = int(parts[1])
    if not folds:
        raise ValueError(f"{path}: no '<track>\\t<fold>' lines")
    return folds


def checkpoint_for(performance: str, folds: dict) -> str:
    """The Beat This! model that never trained on this ASAP performance."""
    k = folds.get(asap_stem(performance))
    return "final0" if k is None else f"fold{k}"


def audio_files(asap_root: Path, maestro: Path) -> dict:
    """ASAP performance -> its MAESTRO audio file (ASAP metadata.csv + the MAESTRO csv)."""
    rows = {r["midi_filename"]: r for r in csv.DictReader(open(maestro / "maestro-v3.0.0.csv", encoding="utf-8"))}
    out = {}
    for row in csv.DictReader(open(asap_root / "metadata.csv", encoding="utf-8")):
        mm = row.get("maestro_midi_performance", "").replace("{maestro}/", "")
        if mm in rows:
            out[row["midi_performance"]] = maestro / rows[mm]["audio_filename"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="test_transcribed,train_transcribed,val_transcribed")
    ap.add_argument("--asap", default="data/raw/asap_bd")
    ap.add_argument("--maestro", default="data/raw/maestro")
    ap.add_argument("--folds", default="data/raw/beat_this/asap_8-folds.split")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = load_folds(Path(args.folds))
    audio = audio_files(Path(args.asap), Path(args.maestro))
    tracker = Tracker(device)
    print(f"Beat This! on {device}; {len(folds)} ASAP tracks in its 8-fold split", flush=True)
    for split in args.splits.split(","):
        split_dir = D / split
        rep_path = split_dir / "report.json"
        if not rep_path.exists():
            print(f"skip {split}: {rep_path} missing (scripts/build_transcribed.py makes it)")
            continue
        report = [r for r in json.loads(rep_path.read_text()) if "segments" in r]
        gt = load_beats(beats_path(split_dir, "gt"))
        beats, rows = {}, []
        for k, r in enumerate(report):
            perf = r["performance"]
            wav = audio.get(perf)
            if wav is None or not wav.exists():
                rows.append({"performance": perf, "skipped": "no audio"})
                continue
            ck = checkpoint_for(perf, folds)
            t, d = tracker(wav, ck)
            t = t - float(r["offset_s"])  # onto the ASAP timeline, like the transcription
            beats[perf] = (t, d)
            row = {"performance": perf, "checkpoint": ck, "beats": int(len(t)), "downbeats": int(d.sum())}
            if perf in gt:
                gt_t, gt_d = gt[perf]
                lo, hi = max(gt_t[0], t[0]), min(gt_t[-1], t[-1])
                g_in, t_in = (gt_t >= lo) & (gt_t <= hi), (t >= lo) & (t <= hi)
                row.update(beat_f=f_measure(gt_t[g_in], t[t_in]), downbeat_f=f_measure(gt_t[g_in & gt_d], t[t_in & d]))
            rows.append(row)
            print(f"  {split} {k + 1}/{len(report)} {perf}: {ck}, {len(t)} beats"
                  + (f", beat F {row['beat_f']:.3f}, downbeat F {row['downbeat_f']:.3f}" if "beat_f" in row else ""),
                  flush=True)
        save_beats(beats_path(split_dir, "tracked"), beats)
        scored = [x for x in rows if "beat_f" in x]
        summary = {"split": split, "performances": len(beats), "skipped": sum("skipped" in x for x in rows),
                   "held_out_checkpoints": sum(x.get("checkpoint", "").startswith("fold") for x in rows),
                   "beat_f_mean": float(np.mean([x["beat_f"] for x in scored])) if scored else None,
                   "downbeat_f_mean": float(np.mean([x["downbeat_f"] for x in scored])) if scored else None}
        (split_dir / "beats_tracked.json").write_text(json.dumps({"summary": summary, "performances": rows}, indent=1))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
