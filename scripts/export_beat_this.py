"""Our piano data in Beat This!'s training format, for fine-tuning it (scripts/train_beat_this.py).

    python scripts/export_beat_this.py --workers 16      # -> data/beat_this_ft/{annotations,audio}

Four datasets, from the meter model's training and validation performances (data/processed/meter;
never a test one), each beat and downbeat from the annotations (ASAP) or the score alignment
(PERiScoPe, Gangqinpu):
  asap_render        ASAP's performance MIDI (pedal included), rendered (src/render.py)
  asap_real          ASAP's performances whose MAESTRO recording is here, cut to the annotated piece
                     (a MAESTRO recording can hold several pieces; its clock found by matching its
                     MIDI to ASAP's, src/preprocessing/performance.py estimate_offset)
  periscope_render   PianoCoRe's segments (PERiScoPe, and ATEPP / GiantMIDI-Piano re-transcribed),
                     their transcribed notes rendered
  pianocore_real     PianoCoRe performances whose recording we downloaded (scripts/fetch_pianocore_audio.py,
                     scripts/retranscribe_pianocore.py: ATEPP, GiantMIDI-Piano), cut to each aligned
                     segment: real classical recordings
  gangqinpu_render   Gangqinpu's segments, their transcribed notes rendered
  gangqinpu_real     Gangqinpu's downloaded recordings, cut to each aligned segment
Only the annotated / aligned stretch of a recording is used: outside it there are beats nobody
annotated, which would teach the tracker that there are none. Real recordings and renders both go in:
the recordings are what it tracks at test time, the renders what it tracks for the meter model's
training data (and MIDI uploads). --hours caps a dataset (a fixed draw): by default every recording,
150 h of each kind of render, so the renders do not drown the recordings.

The layout is Beat This!'s (its BeatDataModule reads it):
  annotations/<dataset>/annotations/beats/<item>.beats   time, beat in the bar (1: downbeat)
  annotations/<dataset>/info.json                        {"has_downbeats": true}
  annotations/<dataset>/single.split                     item, train / val (the validation pieces)
  annotations/<dataset>/8-folds.split                    item, fold (training pieces only; K folds,
                                                         the name Beat This! reads for --fold)
  annotations/<dataset>/annotations/meter/<item>.meter   time, time-signature class
                                                         (src/bt_multitask.py: the multi-task variant's targets)
  audio/spectrograms/<dataset>.npz                       <item>/track.npy: float16 log-mel spectrogram
                                                         (Beat This!'s: 22.05 kHz, 50 frames/s, 128 bands)
and pieces.json: every exported training piece with its fold (src/bt_models.py fold_of), for the
models' models.json.

The mix: four cells, classical / pop x recordings / renders (CELLS); by default each kind of
recording counts twice as much as its renders (--shares): the recordings are what the tracker meets
at test time, the renders add pieces with no recording (PERiScoPe, Aria-MIDI) but should not make it
better on renders than on recordings, since the meter model's training beats of those pieces come
from renders (scripts/compare_render.py --models checks it after fine-tuning). Beat This! samples
each item in proportion to its length, so a cell's share of training
is its hours times how often its items are listed; the split files list a smaller cell's training
items several times (at most --max-repeat, so a few hours of recordings are not learnt by heart),
the largest cell once. The folds are by piece in every dataset, so each fold-out model trains on
every cell (export_report.json: each fold model's hours per cell; a warning if one lacks a cell).
Renders get a random transposition (+-3) and tempo (x0.9-1.1) per item with
--augment (Beat This!'s own pitch / tempo variants are not precomputed here). Run again, it adds
only what is missing.
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import re
import sys
import tempfile
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_gangqinpu_audio import audio_path  # noqa: E402
from src.bt_models import DEFAULT_K, fold_of, piece_of  # noqa: E402
from src.meter import perf_key  # noqa: E402

SR = 22050
PAD = 1.0
DATASETS = ("asap_render", "asap_real", "periscope_render", "pianocore_real", "gangqinpu_render", "gangqinpu_real")
SOURCES = {"asap_render": "", "asap_real": "", "periscope_render": "_periscope", "pianocore_real": "_periscope",
           "gangqinpu_render": "_gangqinpu", "gangqinpu_real": "_gangqinpu"}   # meter/<split><suffix>.pkl ("": ASAP's)
CELLS = {"asap_render": "classical_render", "asap_real": "classical_real", "periscope_render": "classical_render",
         "pianocore_real": "classical_real",
         "gangqinpu_render": "pop_render", "gangqinpu_real": "pop_real"}
DEFAULT_SHARES = "classical_real=1,classical_render=0.5,pop_real=1,pop_render=0.5"
DEFAULT_HOURS = ("asap_render=all,asap_real=all,periscope_render=150,pianocore_real=all,gangqinpu_render=150,"
                 "gangqinpu_real=all")


def stem_of(key: str) -> str:
    """A performance key as an item name (no '/', which Beat This! uses between dataset and item)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_")


def beat_numbers(down: np.ndarray) -> np.ndarray:
    """Beat This!'s second column: 1 on a downbeat, then 2, 3, ... through the bar (2 before the
    first downbeat)."""
    out, cur = np.zeros(len(down), int), 1
    for i, d in enumerate(np.asarray(down, bool)):
        cur = 1 if d else cur + 1
        out[i] = cur
    first = np.argmax(down) if np.any(down) else len(down)
    out[:first] = 2
    return out


def parse_hours(spec: str) -> dict:
    out = {}
    for part in spec.split(","):
        name, _, v = part.partition("=")
        if name not in DATASETS:
            raise SystemExit(f"--hours: unknown dataset {name!r} ({', '.join(DATASETS)})")
        out[name] = None if v in ("all", "") else float(v)
    return out


def duration(p: dict) -> float:
    t = np.asarray(p["gt"]["t"], np.float64)
    return float(t[-1] - t[0]) if len(t) > 1 else 0.0


def choose(perfs: list, hours: float | None) -> list:
    """A fixed draw (by key) up to ``hours`` of annotated music; all of them for None."""
    perfs = sorted(perfs, key=lambda p: zlib.crc32(perf_key(p).encode()))
    if hours is None:
        return perfs
    out, total = [], 0.0
    for p in perfs:
        if total >= hours * 3600:
            break
        out.append(p)
        total += duration(p)
    return out


def pianocore_recordings(dirs: list) -> dict:
    """PianoCoRe performance id -> (its recording, where the performance's clock is in it), from
    scripts/retranscribe_pianocore.py's rows (recording time = clock + offset)."""
    import pyarrow.parquet as pq
    out = {}
    for d in dirs:
        for f in sorted(Path(d).glob("rows-*.parquet")):
            t = pq.read_table(f, columns=["id", "audio_path", "audio_offset_s"]).to_pylist()
            for r in t:
                if r["audio_path"] and Path(r["audio_path"]).exists():
                    out[str(r["id"])] = (Path(r["audio_path"]), float(r["audio_offset_s"]))
    return out


def maestro_recordings(asap: Path, maestro: Path) -> dict:
    """ASAP performance -> (its MAESTRO recording, MAESTRO's MIDI of it), for the ones here."""
    import csv
    out = {}
    meta = asap / "metadata.csv"
    if not meta.exists():
        return out
    for r in csv.DictReader(open(meta, encoding="utf-8")):
        wav = Path(r.get("maestro_audio_performance", "").replace("{maestro}", str(maestro)))
        midi = Path(r.get("maestro_midi_performance", "").replace("{maestro}", str(maestro)))
        if r.get("maestro_audio_performance") and wav.exists() and midi.exists():
            out[r["midi_performance"]] = (wav, midi)
    return out


def real_asap_item(p: dict, asap: Path, rec: tuple):
    """An ASAP performance's recording cut to its annotated beats (PAD either side) and the beats on
    that cut's clock, or None when the recording's MIDI does not match ASAP's."""
    from src.preprocessing.performance import estimate_offset, load_performance
    wav, midi = rec
    d, match = estimate_offset(load_performance(asap / p["performance"])[0], load_performance(midi)[0])
    if match < 0.8:
        return None
    t = np.asarray(p["gt"]["t"], np.float64)
    lo = max(0.0, t[0] - PAD + d)
    x = read_audio(wav, lo, t[-1] + PAD + d)
    return x, t + d - lo, np.asarray(p["gt"]["down"], bool)


def meter_classes(gt: dict, n: int):
    """The time-signature classes (src/bt_multitask.py) of an item's beats: the last ``n`` of its
    ground truth (an item keeps all of its beats, or a render drops a few at the start); None
    without them."""
    if n <= 0 or "ts" not in gt:
        return None
    return np.asarray(gt["ts"], np.int64)[-n:]


def mix_repeats(hours: dict, shares: dict, max_repeat: int) -> dict:
    """How often each cell's training items are listed: the cell with the most hours per share
    once, the others as near their share as whole repeats (at most ``max_repeat``) bring them.
    ``hours``: cell -> training hours (cells without any are left out)."""
    have = {c: h for c, h in hours.items() if h > 0 and shares.get(c, 0) > 0}
    if not have:
        return {}
    target = max(h / shares[c] for c, h in have.items())
    return {c: int(np.clip(round(target * shares[c] / h), 1, max_repeat)) for c, h in have.items()}


def augment_params(key: str) -> tuple[int, float]:
    """(semitones, tempo factor) for an item, the same every run."""
    r = np.random.default_rng(zlib.crc32(key.encode()))
    return int(r.integers(-3, 4)), float(np.exp(r.uniform(np.log(0.9), np.log(1.1))))


def render_item(p: dict, asap_root: Path, augment: bool):
    """(notes, pedal) to render, the beats on its clock, or None. Times start PAD before the first note."""
    from src.preprocessing.performance import load_performance
    midi = asap_root / p["performance"] if p.get("dataset") == "asap" else None
    if midi is not None and midi.exists() and p.get("segments", 1) <= 1:
        notes, pedal = load_performance(midi)
    else:
        notes, pedal = np.asarray(p["notes"], np.float64), None
    notes = np.array(notes, np.float64, copy=True)
    if len(notes) < 2:
        return None
    t0 = max(0.0, float(notes[:, 1].min()) - PAD)
    notes[:, 1:3] -= t0
    pedal = None if pedal is None or not len(pedal) else np.asarray(pedal, np.float64) - t0
    beats = np.asarray(p["gt"]["t"], np.float64) - t0
    if augment:
        k, s = augment_params(perf_key(p))
        lo, hi = 21 - notes[:, 0].min(), 108 - notes[:, 0].max()
        notes[:, 0] += int(np.clip(k, lo, hi))
        notes[:, 1:3] *= s
        beats = beats * s
        pedal = None if pedal is None else pedal * s
    keep = beats >= 0
    return (notes, pedal), beats[keep], np.asarray(p["gt"]["down"], bool)[keep]


class Spect:
    """Beat This!'s log-mel spectrogram of a mono 22.05 kHz signal: [frames, 128] float16."""

    def __init__(self, device: str = "cpu"):
        import torch
        from beat_this.preprocessing import LogMelSpect
        self.torch, self.spect = torch, LogMelSpect(device=device)
        self.device = device

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            x = self.torch.tensor(np.asarray(signal, np.float32), device=self.device)
            return self.spect(x).cpu().numpy().astype(np.float16)


def read_audio(path, start: float | None = None, end: float | None = None) -> np.ndarray:
    """Mono 22.05 kHz, from ``start`` to ``end`` seconds."""
    import soxr
    from beat_this.preprocessing import load_audio
    x, sr = load_audio(str(path), dtype="float32")
    x = x.mean(1) if x.ndim == 2 else x
    if sr != SR:
        x = soxr.resample(x, in_rate=sr, out_rate=SR)
    a = 0 if start is None else max(0, int(round(start * SR)))
    b = len(x) if end is None else min(len(x), int(round(end * SR)))
    return x[a:b]


class Store:
    """One dataset's output: spectrograms appended to <dataset>.npz (uncompressed, as Beat This!
    memory-maps it), beat files, the split files written at the end."""

    def __init__(self, out: Path, name: str):
        self.name = name
        self.ann = out / "annotations" / name
        (self.ann / "annotations" / "beats").mkdir(parents=True, exist_ok=True)
        (self.ann / "info.json").write_text(json.dumps({"has_downbeats": True}))
        self.npz = out / "audio" / "spectrograms" / f"{name}.npz"
        self.npz.parent.mkdir(parents=True, exist_ok=True)
        self.have, self.seconds = set(), {}
        if self.npz.exists():
            with zipfile.ZipFile(self.npz) as z:
                for i in z.infolist():
                    if i.filename.endswith("/track.npy"):
                        stem = i.filename.split("/")[0]
                        self.have.add(stem)
                        self.seconds[stem] = max(0, i.file_size - 128) / (128 * 2) / 50   # float16 x 128 bands, 50 frames/s
        self.zip = zipfile.ZipFile(self.npz, "a", compression=zipfile.ZIP_STORED, allowZip64=True)
        self.split, self.folds = {}, {}

    def done(self, stem: str) -> bool:
        return stem in self.have and (self.ann / "annotations" / "beats" / f"{stem}.beats").exists()

    def add(self, stem: str, spect: np.ndarray, beats: np.ndarray, down: np.ndarray,
            ts: np.ndarray | None = None):
        buf = io.BytesIO()
        np.save(buf, spect)
        self.zip.writestr(f"{stem}/track.npy", buf.getvalue())
        self.seconds[stem] = len(spect) / 50
        np.savetxt(self.ann / "annotations" / "beats" / f"{stem}.beats",
                   np.stack([beats, beat_numbers(down)], axis=1), fmt=["%.4f", "%d"], delimiter="\t")
        if ts is not None:   # the multi-task variant's targets
            from src.bt_multitask import write_meter
            write_meter(self.ann / "annotations" / "meter" / f"{stem}.meter", beats, ts)
        self.have.add(stem)

    def finish(self):
        """Only the items written are kept (Beat This! fails on a listed item with no beat file)."""
        self.zip.close()
        self.split = {s: v for s, v in self.split.items() if self.done(s)}
        self.folds = {s: f for s, f in self.folds.items() if self.done(s)}

    def train_hours(self, fold_out: int | None = None) -> float:
        """Hours of training items (those of every fold but ``fold_out``)."""
        return sum(self.seconds.get(s, 0.0) for s, f in self.folds.items() if f != fold_out) / 3600

    def write_splits(self, repeat: int = 1):
        """single.split (train / val) and 8-folds.split (fold), each training item listed ``repeat``
        times (the mix); validation items once."""
        rows = [(s, v) for s, v in sorted(self.split.items()) for _ in range(repeat if v == "train" else 1)]
        (self.ann / "single.split").write_text("".join(f"{s}\t{v}\n" for s, v in rows))
        rows = [(s, f) for s, f in sorted(self.folds.items()) for _ in range(repeat)]
        (self.ann / "8-folds.split").write_text("".join(f"{s}\t{f}\n" for s, f in rows))


def main(argv=None, renderer=None, spect=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--asap", default="data/raw/asap_bd")
    ap.add_argument("--audio", default="data/raw/gangqinpu/audio")
    ap.add_argument("--maestro", default="data/raw/maestro", help="asap_real: MAESTRO's recordings")
    ap.add_argument("--pianocore-real", default="data/raw/pianocore/atepp_tk",
                    help="pianocore_real: re-transcribed PianoCoRe folders with recordings (comma-separated)")
    ap.add_argument("--out", default="data/beat_this_ft")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--hours", default=DEFAULT_HOURS, help="hours of annotated music per dataset (all: no cap)")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="folds for cross-fitting")
    ap.add_argument("--augment", action="store_true", help="renders: a random transposition and tempo per item")
    ap.add_argument("--shares", default=DEFAULT_SHARES, help="each cell's share of the training (classical / pop x real / render)")
    ap.add_argument("--max-repeat", type=int, default=4, help="the most a cell's training items are listed")
    ap.add_argument("--workers", type=int, default=8, help="renders at once")
    ap.add_argument("--soundfonts", default=None)
    ap.add_argument("--fluidsynth", default=None)
    ap.add_argument("--device", default="cpu", help="for the spectrograms")
    args = ap.parse_args(argv)
    names = [d for d in args.datasets.split(",") if d]
    if any(d not in DATASETS for d in names):
        raise SystemExit(f"--datasets: from {DATASETS}")
    hours = {d: None for d in DATASETS} | parse_hours(args.hours)
    data, out, asap = Path(args.data), Path(args.out), Path(args.asap)
    if renderer is None and any(d.endswith("_render") for d in names):
        from src.render import Renderer
        renderer = Renderer(args.soundfonts, args.fluidsynth)
    spect = spect or Spect(args.device)
    pieces, report, t_start = {}, {}, time.perf_counter()
    pf = out / "pieces.json"
    if pf.exists():
        old = json.loads(pf.read_text())
        if int(old.get("k", args.k)) != args.k:
            raise SystemExit(f"{pf}: exported with k={old['k']}, not {args.k} (a new --out, or the same k)")
        pieces = old.get("pieces", {})
    shares = {}
    for part in args.shares.split(","):
        c, _, v = part.partition("=")
        if c not in set(CELLS.values()):
            raise SystemExit(f"--shares: unknown cell {c!r} ({sorted(set(CELLS.values()))})")
        shares[c] = float(v)
    stores = {}
    for name in names:
        store = Store(out, name)
        stores[name] = store
        rep = {"items": 0, "hours": 0.0, "skipped": 0}
        for split in ("train", "val"):
            f = data / "meter" / f"{split}{SOURCES[name]}.pkl"
            if not f.exists():
                continue
            perfs = pickle.load(open(f, "rb"))
            if name.startswith("asap_"):
                perfs = [p for p in perfs if p.get("dataset") == "asap"]
            if name == "asap_real":
                recs = maestro_recordings(asap, Path(args.maestro))
                perfs = [p for p in perfs if p["performance"] in recs]
            if name == "pianocore_real":
                precs = pianocore_recordings([d for d in args.pianocore_real.split(",") if d])
                perfs = [p for p in perfs if str(p["performance"]) in precs]
            if name == "gangqinpu_real":
                perfs = [p for p in perfs if audio_path(Path(args.audio), split, p["performance"].split("/", 1)[1]).exists()]
            perfs = choose(perfs, hours[name] if split == "train" else (None if hours[name] is None else hours[name] / 10))
            todo = []
            for p in perfs:
                stem = stem_of(perf_key(p))
                store.split[stem] = split
                if split == "train":
                    fold = fold_of(piece_of(p), args.k)
                    store.folds[stem] = fold
                    pieces[piece_of(p)] = fold
                if not store.done(stem):
                    todo.append((stem, p))
            print(f"{name} {split}: {len(perfs)} items, {len(todo)} to export", flush=True)
            with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(max(1, args.workers)) as pool:
                def make(job):
                    stem, p = job
                    if name == "asap_real":
                        item = real_asap_item(p, asap, recs[p["performance"]])
                        return (stem, None, None, None) if item is None else (stem, *item)
                    if name == "pianocore_real":
                        wav, off = precs[str(p["performance"])]
                        lo, hi = p["span"]
                        start = max(0.0, lo + off)          # the cut cannot start before the recording does
                        x = read_audio(wav, start, hi + off)
                        t = np.asarray(p["gt"]["t"], np.float64) + off - start
                        return stem, x, t, np.asarray(p["gt"]["down"], bool)
                    if name == "gangqinpu_real":
                        lo, hi = p["span"]
                        sid = p["performance"].split("/", 1)[1]
                        start = max(0.0, lo)
                        x = read_audio(audio_path(Path(args.audio), split, sid), start, hi)
                        t = np.asarray(p["gt"]["t"], np.float64) - start
                        return stem, x, t, np.asarray(p["gt"]["down"], bool)
                    item = render_item(p, asap, args.augment)
                    if item is None:
                        return stem, None, None, None
                    (notes, pedal), t, d = item
                    wav = Path(tmp) / f"{stem}.wav"
                    renderer((notes, pedal) if pedal is not None else notes, wav, perf_key(p))
                    x = read_audio(wav)
                    wav.unlink(missing_ok=True)
                    return stem, x, t, d
                from collections import deque
                by_stem = dict(todo)
                jobs, pending = iter(todo), deque()     # a few ahead, so finished audio does not pile up

                def ahead():
                    j = next(jobs, None)
                    if j is not None:
                        pending.append((j[0], pool.submit(make, j)))
                for _ in range(2 * max(1, args.workers)):
                    ahead()
                k = -1
                while pending:
                    k += 1
                    first, fut = pending.popleft()
                    ahead()
                    try:
                        stem, x, t, d = fut.result()
                        ts = meter_classes(by_stem[stem]["gt"], 0 if t is None else len(t))
                    except Exception as e:  # noqa: BLE001
                        print(f"  {first}: failed: {e}", flush=True)
                        rep["skipped"] += 1
                        continue
                    if x is None or len(t) < 4 or len(x) < SR:
                        rep["skipped"] += 1
                        continue
                    store.add(stem, spect(x), t, d, ts)
                    rep["items"] += 1
                    rep["hours"] += len(x) / SR / 3600
                    if (k + 1) % 200 == 0:
                        print(f"  {name} {split}: {k + 1} of {len(todo)} ({time.perf_counter() - t_start:.0f} s)", flush=True)
        store.finish()
        rep["hours"] = round(rep["hours"], 2)
        report[name] = rep
        print(f"{name}: {json.dumps(rep)}", flush=True)
    # the mix: every dataset of the export (also ones an earlier run exported with other --datasets)
    for name in DATASETS:
        if name not in stores and (out / "annotations" / name / "8-folds.split").exists():
            st = Store(out, name)
            for line in (out / "annotations" / name / "single.split").read_text().splitlines():
                stem, v = line.split("\t")
                st.split[stem] = v
            for line in (out / "annotations" / name / "8-folds.split").read_text().splitlines():
                stem, f = line.split("\t")
                st.folds[stem] = int(f)
            st.finish()
            stores[name] = st
    cell_hours = {}
    for name, st in stores.items():
        cell_hours[CELLS[name]] = cell_hours.get(CELLS[name], 0.0) + st.train_hours()
    repeats = mix_repeats(cell_hours, shares, args.max_repeat)
    for name, st in stores.items():
        st.write_splits(repeats.get(CELLS[name], 1))
    eff = {c: h * repeats.get(c, 1) for c, h in cell_hours.items()}
    mix = {c: {"hours": round(h, 2), "repeat": repeats.get(c, 1), "share": round(eff[c] / max(sum(eff.values()), 1e-9), 3)}
           for c, h in sorted(cell_hours.items())}
    folds = {}
    for k in range(args.k):   # what each fold-out model trains on
        hrs = {}
        for name, st in stores.items():
            hrs[CELLS[name]] = hrs.get(CELLS[name], 0.0) + st.train_hours(fold_out=k)
        folds[f"without fold {k}"] = {c: round(h, 2) for c, h in sorted(hrs.items())}
        lacking = [c for c, h in hrs.items() if h <= 0 and cell_hours.get(c, 0) > 0]
        if lacking:
            print(f"WARNING: the model without fold {k} has no {lacking} (all of it in fold {k})", flush=True)
    print(f"mix: {json.dumps(mix)}", flush=True)
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps({"k": args.k, "pieces": pieces}, indent=1))
    (out / "export_report.json").write_text(json.dumps({"k": args.k, "datasets": report, "mix": mix, "folds": folds,
                                                        "seconds": round(time.perf_counter() - t_start, 1)}, indent=1))
    return report


if __name__ == "__main__":
    main()
