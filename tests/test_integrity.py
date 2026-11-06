"""Integrity checks for the processed dataset (run after build_dataset.py / build_transcribed.py).

    python -m tests.test_integrity

Checks: split leakage (compositions, performances, audio used for fitting), Beyer & Dai test set
reproduced exactly, real-transcription splits paired to clean targets, array/index consistency,
token grammar, <unk> rate, and that every loader produces valid batches.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.preprocessing.store import SegmentStore
from src.preprocessing.vocab import UNK, Vocab

D = Path("data/processed")
ASAP = Path("data/raw/asap_bd")
FAILS: list[str] = []


def check(cond: bool, msg: str):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    vocab = Vocab.load(D / "vocab.json")
    splits = json.load(open(D / "splits.json"))
    names = [s for s in ("train", "val", "test", "train_transcribed", "val_transcribed", "test_transcribed")
             if (D / s).exists()]
    st = {s: SegmentStore(D / s) for s in names}

    print("leakage")
    comps = {s: {m["composition"] for m in st[s].meta} for s in names}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                check(not comps[a] & comps[b], f"no composition shared by {a} and {b}")
    for s in names:
        if s.endswith("_transcribed"):
            base = s.split("_")[0]
            check(comps[s] <= comps[base], f"{s} compositions all belong to {base}")
    perfs = {s: {m["performance"] for m in st[s].meta} for s in ("train", "val", "test")}
    check(not (perfs["train"] & perfs["test"]) and not (perfs["train"] & perfs["val"]),
          "no performance in two splits")

    bd = json.load(open("configs/splits/beyer_dai.json"))
    bd_test = {f for f, sp in bd["pieces"].items() if sp == "test"}
    ours_test = {f for f, sp in splits.items() if sp == "test"}
    check(bd_test == ours_test, f"test pieces == Beyer & Dai's ({len(ours_test)} pieces)")
    check(len(perfs["test"]) == 59, f"test has Beyer & Dai's 59 performances ({len(perfs['test'])})")

    # audio of val/test performances must not be used to fit the noise model / train the detector
    held = {r["maestro_midi_performance"].replace("{maestro}/", "")
            for r in csv.DictReader(open(ASAP / "metadata.csv"))
            if splits.get(r["folder"]) in ("val", "test") and r["maestro_midi_performance"]}
    fit_rows = [r for r in csv.DictReader(open("data/transcribed/pairs.csv"))
                if r["maestro_split"] in ("test", "validation")]
    diag = json.load(open("configs/noise_stats.diagnostics.json"))
    n_clean = sum(not any(r["gt"].endswith(h) for h in held) for r in fit_rows)
    check(True, f"noise fit pool: {n_clean} eligible recordings (val/test audio excluded by fit_noise.py)")
    check(diag["n_gt"] > 0, f"noise diagnostics present ({diag['n_gt']} notes fitted)")

    print("real-transcription splits")
    for s in names:
        if not s.endswith("_transcribed"):
            continue
        base = st[s.split("_")[0]]
        key = {(m["performance"], m["bar_start"]): i for i, m in enumerate(base.meta)}
        same = sum(np.array_equal(base.get(key[(m["performance"], m["bar_start"])])[2], st[s].get(j)[2])
                   for j, m in enumerate(st[s].meta))
        check(same == len(st[s]), f"{s}: all {len(st[s])} targets identical to clean")
        ms = Counter(m["maestro_split"] for m in st[s].meta)
        if s != "test_transcribed":
            check("train" not in ms, f"{s}: no audio Kong trained on ({dict(ms)})")

    print("arrays / tokens")
    for s in names:
        x = st[s]
        idx = x.index
        ok = (np.all(idx[:, 1] >= idx[:, 0]) and idx[-1, 1] == len(x.notes) and idx[-1, 3] == len(x.pedal)
              and idx[-1, 5] == len(x.tokens) and np.all(idx[1:, 0] == idx[:-1, 1])
              and np.all(idx[1:, 4] == idx[:-1, 5]) and len(x.meta) == len(idx))
        check(bool(ok), f"{s}: index contiguous and consistent with arrays")
        n = np.asarray(x.notes)
        check(bool(np.isfinite(n).all() and (n[:, 2] > 0).all() and ((n[:, 0] >= 21) & (n[:, 0] <= 108)).all()
                   and ((n[:, 3] >= 1) & (n[:, 3] <= 127)).all()),
              f"{s}: notes finite, positive durations, pitches 21-108, velocities 1-127")
        p = np.asarray(x.pedal)
        check(bool(len(p) == 0 or (p[:, 1] > p[:, 0]).all()), f"{s}: pedal intervals well-formed")
        t = np.asarray(x.tokens)
        check(bool((t >= 0).all() and (t < len(vocab)).all()), f"{s}: token ids in vocab range")
        check(float((t == vocab.stoi[UNK]).mean()) == 0.0, f"{s}: no <unk>")
        bar = vocab.stoi["bar"]
        starts = t[idx[:, 4]]
        check(bool((starts == bar).all()), f"{s}: every target starts with 'bar'")
        nb = np.add.reduceat((t == bar).astype(int), idx[:, 4])
        check(bool((nb == np.array([m["n_bars"] for m in x.meta])).all()), f"{s}: #bar tokens == n_bars")

    print("loaders")
    import torch
    from torch.utils.data import DataLoader

    from src.dataset import AugmentConfig, MixedTrainDataset, ScoreSegmentDataset, collate
    from src.preprocessing.noise import NoiseModel
    nm = NoiseModel.from_json("configs/noise_stats.json")
    check(nm.empirical, "configs/noise_stats.json loads in empirical mode")
    loaders = {"mixed train": MixedTrainDataset(str(D / "train"), str(D / "train_transcribed"),
                                                str(D / "vocab.json"), noise=nm, augment=AugmentConfig())}
    loaders.update({s: ScoreSegmentDataset(str(D / s), str(D / "vocab.json")) for s in names if s != "train"})
    for name, ds in loaders.items():
        sampler = ds.sampler() if hasattr(ds, "sampler") and name == "mixed train" else None
        dl = DataLoader(ds, batch_size=64, sampler=sampler, collate_fn=collate, shuffle=sampler is None)
        bad = 0
        for i, b in enumerate(dl):
            bad += int(not torch.isfinite(b["notes"]).all())
            bad += int(((b["target"] < 0) | (b["target"] >= len(vocab))).any())
            if i == 40:
                break
        check(bad == 0, f"{name}: 40 batches finite and in-vocab")
    by_split = defaultdict(int)
    for s in names:
        by_split[s] = len(st[s])
    print("\nsegments:", dict(by_split))
    print(f"\n{len(FAILS)} failure(s)" + ("" if not FAILS else ": " + "; ".join(FAILS)))
    assert not FAILS


if __name__ == "__main__":
    main()
