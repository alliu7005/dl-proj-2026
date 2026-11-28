"""Round 14: the labeller's masked-note pretraining (scripts/pretrain_labeller.py) and its --init,
meter pretraining from several stores, the beat-error analysis (scripts/beat_errors.py), the
recipes, the report and pace/v2_round14.sh.

    python -m tests.test_round14
"""
from __future__ import annotations

import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _vocab() -> Path | None:
    for p in (ROOT / "data" / "processed" / "vocab.json", ROOT / "models" / "v2_labeller" / "vocab.json"):
        if p.exists():
            return p
    return None


def test_labeller_pretraining():
    """prep cuts MIDI files into the labeller's windows; train runs; its checkpoint's encoder loads
    into a labeller of the same config with only the heads missing."""
    import torch

    from scripts.pretrain_labeller import labeller_cfg, main
    from src.dataset import N_FEATURES_CONTEXT
    from src.labeller import build_labeller
    from src.notelabels import N_GRID_FEATURES
    from tests.test_pipeline import make_performance
    voc = _vocab()
    if voc is None:
        print("  (skipped: no vocab.json)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "data").mkdir()
        shutil.copy(voc, tmp / "data" / "vocab.json")
        (tmp / "midi").mkdir()
        for k in range(3):
            make_performance(tmp / "midi" / f"p{k}.mid", n=300, seed=k)
        store = tmp / "store.npz"
        assert main(["prep", "--midi", str(tmp / "midi"), "--data", str(tmp / "data"), "--out", str(store)]) == 0
        z = np.load(store)
        assert len(z["offsets"]) > 3 and z["feats"].shape[1] == N_FEATURES_CONTEXT
        assert main(["train", "--store", str(store), "--data", str(tmp / "data"), "--runs-dir", str(tmp / "runs"),
                     "--name", "pt", "--size", "small", "--steps", "3", "--eval-every", "3", "--batch", "4",
                     "--device", "cpu"]) == 0
        ck = torch.load(tmp / "runs" / "pt" / "best.pt", weights_only=False)
        cfg = labeller_cfg(tmp / "data", "small")
        assert ck["config"] == cfg and cfg["n_features"] == N_FEATURES_CONTEXT + N_GRID_FEATURES
        assert cfg["context_before"] == 64 and cfg["context_after"] == 32
        model = build_labeller(cfg)
        enc = {k: v for k, v in ck["model"].items() if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(enc, strict=False)
        assert not unexpected and missing and all(k.startswith("head.") for k in missing)
        assert "val_pitch_acc" in ck["pretrain"] and set(ck["heads"]) >= {"pitch.weight", "dur.weight"}
    src = (ROOT / "scripts" / "train_labeller.py").read_text()
    assert "--init" in src and 'not k.startswith("head.")' in src


def test_batch_masks_only_own_notes():
    from scripts.pretrain_labeller import MASK_PITCH, batch
    from src.dataset import ROLE_BEFORE, ROLE_WINDOW
    rng = np.random.default_rng(0)
    n = 60
    f = np.zeros((n, 10), np.float16)
    f[:, 0] = np.arange(40, 40 + n)
    f[:, 3] = np.linspace(-2, 2, n)
    roles = np.array([ROLE_BEFORE] * 20 + [ROLE_WINDOW] * 40, np.int8)
    X, M, H, P, D = batch((f, roles, np.array([0, n])), [0], 27, 0.5, rng, "cpu")
    h = H[0].numpy()
    assert h.any() and not h[:20].any()                                 # context notes never hidden
    assert (X[0, h, 0].numpy() == MASK_PITCH).all() and (X[0, ~h & M[0].numpy(), 0].numpy() != MASK_PITCH).all()
    assert (P[0, h].numpy() == np.arange(40, 40 + n)[h]).all()          # the targets keep the true pitch
    assert X.shape == (1, n, 27) and (X[0, :, 10:] == 0).all()           # no grid
    assert D[0].max() <= 15


def test_beat_errors():
    from scripts.beat_errors import errors
    t = 1.0 + 0.5 * np.arange(33)
    gt = {"t": t, "down": np.arange(33) % 4 == 0}
    e = errors(t.copy(), np.arange(33) % 4 == 0, [gt])
    assert e["beat_f"] == 1 and e["missed"] == 0 and e["extra"] == 0 and e["level_class"] == "right level"
    e = errors(t[::2].copy(), np.zeros(17, bool), [gt])                   # every other beat
    assert e["level_class"] == "half tempo (x2)" and abs(e["missed"] - 0.5) < 0.05 and e["extra"] == 0
    e = errors(t[:-1] + 0.25, np.zeros(32, bool), [gt])                   # half a beat off
    assert e["offbeat"] > 0.9 and e["beat_f"] < 0.1
    e = errors(np.sort(np.concatenate([t, t[:-1] + 0.25])), np.zeros(65, bool), [gt])   # double tempo
    assert e["level_class"] == "double tempo (x0.5)" and e["missed"] == 0 and e["extra"] > 0.45


def test_meter_pretraining_joins_stores():
    from scripts.pretrain_meter import main
    from src.meter import N_BEAT_FEATURES
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for i in range(2):
            perfs = [{"file": f"{i}_{k}.mid", "x": rng.normal(size=(60, N_BEAT_FEATURES)).astype(np.float16)}
                     for k in range(15)]
            pickle.dump(perfs, open(tmp / f"s{i}.pkl", "wb"))
        assert main(["train", "--data", f"{tmp / 's0.pkl'},{tmp / 's1.pkl'}", "--runs-dir", str(tmp / "runs"),
                     "--name", "pt", "--steps", "2", "--eval-every", "2", "--batch", "4", "--chunk", "32",
                     "--d-model", "64", "--layers", "1", "--device", "cpu"]) == 0
        import torch
        assert torch.load(tmp / "runs" / "pt" / "best.pt", weights_only=False)["pretrain"]["performances"] == 30


def test_recipes():
    def flags(fn, name):
        r = subprocess.run(["bash", "-c", f'eval "$(sed -n "/^{fn}()/,/^}}/p" pace/v2_jobs.sh)"; {fn} {name}'],
                           cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip()
    base = flags("meter_flags", "meter_v4_gq")
    assert flags("meter_flags", "meter_v4_gq_ptr") == "--init outputs/runs/meter_pt/best.pt " + base
    assert flags("meter_flags", "meter_v4_gq_ptr_s2").endswith("--seed 2")
    assert flags("meter_flags", "meter_v4_gq_pt150_p12k") == flags("meter_flags", "meter_v4_gq_pt_p12k").replace("meter_pt/", "meter_pt150/")
    assert flags("nl_flags", "nl_gq_ptl_p20k") == flags("nl_flags", "nl_gq_p20k") + " --init outputs/runs/nl_pt/best.pt"


def test_report_runs_empty():
    from scripts.round14_report import main
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        assert main(["--runs-dir", str(tmp), "--asap", str(tmp / "x.json"), "--gq", str(tmp / "y.json"),
                     "--out", str(tmp / "out")]) == 0
        assert "Round 14" in (tmp / "out" / "report.md").read_text()


def test_scoring_reuses_and_merges():
    """beyer_dai.py score rates only predictions it has not rated (or that changed), and merges with a
    results.json another job wrote meanwhile instead of dropping its systems."""
    import argparse
    import json
    import os

    import scripts.beyer_dai as B
    calls = []

    def fake_rate(job):
        calls.append(job[1])
        return {"muster": {"OnsetER": 1.0, "MeanER": 2.0} if job[1] else None, "sim": {"NoteDuration": 0.1} if job[1] else None,
                "error": None if job[1] else "no prediction"}
    real = B.rate
    B.rate = fake_rate
    try:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            out = tmp / "eval"
            entries = []
            for k in range(2):
                ref = tmp / f"ref{k}.musicxml"
                ref.write_text("x")
                entries.append({"condition": "clean", "performance": f"p{k}", "slug": f"p{k}", "reference": str(ref)})
            (tmp / "manifest.json").write_text(json.dumps(entries))
            for sysname in ("sysA", "sysB"):
                (out / sysname / "clean").mkdir(parents=True)
                (out / sysname / "summary.json").write_text("{}")
                for e in entries:
                    (out / sysname / "clean" / f"{e['slug']}.musicxml").write_text("x")
            args = argparse.Namespace(manifest=str(tmp / "manifest.json"), conditions="clean", out=str(out), systems="all",
                                      repo="unused", jobs=1, rescore=False)
            B.score(args)
            assert len(calls) == 6                                           # 2 systems + the reference, 2 pieces
            calls.clear()
            B.score(args)
            assert calls == []                                               # nothing new
            f = out / "sysA" / "clean" / "p0.musicxml"
            os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 5))
            B.score(args)
            assert calls == [str(f)]                                         # only the changed one
            # another job's system lands in results.json meanwhile: kept when this one writes
            d = json.loads((out / "results.json").read_text())
            d["pieces"].append({"system": "sysC", "condition": "clean", "performance": "p0", "muster": {"OnsetER": 1.0, "MeanER": 3.0},
                                "sim": {"NoteDuration": 0.2}, "error": None, "pred_mtime": 1.0})
            (out / "results.json").write_text(json.dumps(d))
            calls.clear()
            B.score(args)
            res = json.loads((out / "results.json").read_text())
            assert "sysC" in res["results"]["clean"] and {e["system"] for e in res["pieces"]} >= {"sysA", "sysB", "sysC"}
            B.score(argparse.Namespace(**{**vars(args), "rescore": True}))
            assert len(calls) == 6
    finally:
        B.rate = real


def test_round14_batch():
    from tests.batch_env import batch_env
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_round14.sh", tmp / "pace")
        for f in ("outputs/runs/meter_pt/best.pt", "data/raw/aria_midi/unique/fetch.json", "outputs/runs/nl_gq/config.json",
                  "data/processed/vocab.json", "outputs/system_eval_gangqinpu/manifest.json",
                  "outputs/system_eval/manifest.json", "bd/MIDI2ScoreTF.ckpt"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "squeue").write_text('#!/bin/bash\ncase "$*" in *g1final*) echo 950 ;; esac\n')
        for f in (tmp / "bin").iterdir():
            f.chmod(0o755)
        r = subprocess.run(["bash", "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_round14.sh'], cwd=tmp,
                           env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd")),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        by = {ln.split("\t")[1].split("-J ")[1].split()[0]: ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()}
        assert list(by) == ["r4tests", "m4train", "m4ens", "m4asap", "m4gq", "d4fetch", "d4prep0", "d4prep1", "d4prep2",
                            "d4prep3", "d4train", "d4ft", "b4errors", "l4prep", "l4train", "l4ft", "l4asap", "l4gq", "r4final"]
        t0 = by["r4tests"][0]
        for j in ("m4train", "d4fetch", "b4errors", "l4prep"):      # the four start together, after the tests
            assert f"afterok:{t0}" in by[j][1], j
        assert "afterany:950" in by["b4errors"][1] and "beat_errors.py" in by["b4errors"][1]
        assert "meters meter_v4_gq_ptr meter_v4_gq_ptr_s1 meter_v4_gq_ptr_s2" in by["m4train"][1]
        assert "ensemble meter_v4_gq_ptr_ens3t tta" in by["m4ens"][1]
        assert "nl_gq+meter_v4_gq_ptr_ens3t:tidy,d95,missing" in by["m4gq"][1]
        assert "nl_gq+meter_v4_gq_ens3t:" not in by["m4gq"][1]               # round 11 runs that one
        shards = ":".join(by[f"d4prep{i}"][0] for i in range(4))
        assert f"afterok:{shards}" in by["d4train"][1] and "pretrain_dd3.pkl" in by["d4train"][1]
        assert "--shard 2/4" in by["d4prep2"][1] and "--subset deduped --max-files 150000" in by["d4fetch"][1]
        assert "labellers nl_gq_ptl_p20k" in by["l4ft"][1]
        ends = ":".join(by[k][0] for k in ("m4asap", "m4gq", "d4ft", "l4asap", "l4gq"))
        assert f"afterany:{ends}" in by["r4final"][1] and "round14_report.py" in by["r4final"][1]


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
