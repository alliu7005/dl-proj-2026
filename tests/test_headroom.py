"""Tests for scripts/headroom.py: the top-k oracle on the labeller's logits, the meter's
alternative grids and their per-stretch scores, the meter test end to end on a small built data set.

    python -m tests.test_headroom        # or: pytest tests/test_headroom.py
"""
from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.headroom import alternatives, oracle_logits, segment_scores  # noqa: E402


def test_oracle_logits():
    import torch
    out = {"dur": torch.tensor([[[3.0, 2.0, 1.0, 0.0, -1.0], [0.0, 5.0, 4.0, 3.0, 2.0]]])}
    labels = torch.full((1, 2, 6), -1, dtype=torch.long)
    labels[0, 0, 4] = 2          # third likeliest: in the top 4 (and top 3), not the top 2
    labels[0, 1, 4] = 0          # the least likely of note 2: in no top 4
    o2 = oracle_logits(out, labels, ("dur",), 2)
    o4 = oracle_logits(out, labels, ("dur",), 4)
    assert o2["dur"][0, 0].argmax() == 0 and o4["dur"][0, 0].argmax() == 2
    assert o4["dur"][0, 1].argmax() == 1                      # not among the top 4: unchanged
    assert torch.equal(out["dur"], torch.tensor([[[3.0, 2.0, 1.0, 0.0, -1.0], [0.0, 5.0, 4.0, 3.0, 2.0]]]))
    labels[0, 0, 4] = -1                                      # no target: untouched
    assert torch.equal(oracle_logits(out, labels, ("dur",), 4)["dur"][0, 0], out["dur"][0, 0])


def _perf(n_bars=16, seed=0):
    """A 4/4 performance with an exact grid: a chord on each beat, louder on downbeats."""
    t = np.arange(4 * n_bars) * 0.5 + 0.3
    rows = [(48 if k % 4 == 0 else 55, x, x + 0.4, 90 if k % 4 == 0 else 60) for k, x in enumerate(t)]
    rows += [(64, x + 0.25, x + 0.45, 50) for x in t]
    gt = {"t": t, "q": np.arange(len(t), dtype=float), "down": np.arange(len(t)) % 4 == 0,
          "ts": np.zeros(len(t), int), "key": np.zeros(len(t), int), "ts_names": ["4/4"] * len(t)}
    return {"performance": f"p{seed}", "piece": f"p{seed}", "dataset": "asap",
            "notes": np.array(sorted(rows, key=lambda r: r[1]), np.float32), "gt": gt, "beats": None}


def test_alternatives_and_segments():
    from src.meter import DecodeConfig, grid_scores, oracle_logp
    p = _perf()
    lp = oracle_logp(p["gt"]["t"], p["gt"])
    alts = alternatives(p["gt"]["t"], lp, DecodeConfig())
    assert {"decoded", "shift1", "shift2", "shift3", "one_meter"} <= set(alts) and any(k.startswith("ts_") for k in alts)
    s = grid_scores(alts["decoded"], p["gt"])
    assert s["downbeat_f"] > 0.99 and s["ts_acc"] > 0.99
    assert grid_scores(alts["shift1"], p["gt"])["downbeat_f"] < 0.5      # the bar lines a beat off
    seg = segment_scores(alts["decoded"], p["gt"], 4)
    assert len(seg) == 4 and all(f > 0.99 and a > 0.99 for f, a in seg)
    bad = segment_scores(alts["shift2"], p["gt"], 4)
    assert all(f < 0.5 for f, _ in bad)


def test_meter_end_to_end():
    import torch

    from scripts.headroom import main
    from src.meter import build_meter_model
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "data" / "meter").mkdir(parents=True)
        pickle.dump([_perf(seed=i) for i in range(3)], open(d / "data" / "meter" / "test.pkl", "wb"))
        (d / "runs" / "m").mkdir(parents=True)
        torch.manual_seed(0)
        cfg = dict(d_model=32, n_heads=4, layers=1, ff=64)
        torch.save({"model": build_meter_model(cfg).state_dict(), "config": cfg}, d / "runs" / "m" / "best.pt")
        assert main(["meter", "--run", "m", "--conditions", "clean", "--data", str(d / "data"), "--runs-dir",
                     str(d / "runs"), "--out", str(d / "out"), "--device", "cpu", "--segment", "4"]) == 0
        import json
        r = json.loads((d / "out" / "meter.json").read_text())["clean"]
        assert r["performances"] == 3 and r["piece"]["oracle"] >= r["piece"]["decoded"]
        assert r["segment"]["oracle"] >= r["segment"]["decoded"]
        assert "Meter: the best of a few alternative grids" in (d / "out" / "report.md").read_text()


def test_batch_graph():
    import shutil
    import subprocess

    from tests.batch_env import batch_env
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_headroom.sh", tmp / "pace")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "squeue").write_text("#!/bin/bash\necho 900\n")
        for f in (tmp / "bin").iterdir():
            f.chmod(0o755)
        r = subprocess.run(["bash", "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_headroom.sh'], cwd=tmp,
                           env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me"), capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        by = {ln.split("\t")[1].split("-J ")[1].split()[0]: ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()}
        assert list(by) == ["h1tests", "h1lab", "h1met", "h1report"]
        t0 = by["h1tests"][0]
        assert f"afterok:{t0},afterany:900" in by["h1lab"][1] and "headroom.py labeller --run nl_gq" in by["h1lab"][1]
        assert f"afterok:{t0}" in by["h1met"][1] and "headroom.py meter --run meter_v4_gq --max-perfs 400" in by["h1met"][1]
        assert f"afterany:{by['h1lab'][0]}:{by['h1met'][0]}" in by["h1report"][1]


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
