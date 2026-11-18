"""Tests for the training schedule: learning rate, early stopping, cooldown, resume.

    python -m tests.test_training_stop          # ~3 min on a CPU; needs data/processed (any build)

Unit tests check ``train_model.learning_rate``. The rest run scripts/train_model.py for a few
dozen steps on the CPU with the learning rate set to 0, so the validation loss never improves and
early stopping is guaranteed to fire at a known step:

  frozen run      best at the first eval -> early stop after ``patience`` steps -> cooldown back to
                  the best checkpoint for ``cooldown`` steps -> done, at exactly the expected steps
  resume (done)   resubmitting a finished run trains nothing
  resume (cool)   a checkpoint saved mid-cooldown finishes the cooldown, then stops
  --cooldown 0    an early stop ends training at once
  --patience 0    no early stopping: runs to --max-steps
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_model import learning_rate  # noqa: E402

RUNS = ROOT / "outputs" / "runs"
# 2048-token batches -> ~9.5k steps per epoch on the ASAP-only build (more on bigger builds), so
# patience / cooldown are given in tiny fractions of an epoch; the tests read the actual step
# counts from the log instead of assuming them.
BASE = ["--size", "small", "--lr", "0", "--warmup", "2", "--eval-every", "3", "--eval-loss-windows", "8",
        "--eval-decode-windows", "1", "--workers", "0", "--batch-tokens", "2048", "--max-steps", "200"]


def test_learning_rate():
    kw = dict(lr=1e-3, warmup=10, total_steps=110, min_lr_frac=0.05)
    assert math.isclose(learning_rate(0, **kw), 1e-4)                       # warm-up: (0 + 1) / 10
    assert math.isclose(learning_rate(9, **kw), 1e-3)                       # end of warm-up
    assert math.isclose(learning_rate(10, **kw), 1e-3)                      # cosine starts at the peak
    assert math.isclose(learning_rate(60, **kw), (1e-3 + 5e-5) / 2)         # half way: mean of peak and min
    assert math.isclose(learning_rate(110, **kw), 5e-5)                     # minimum at total_steps
    assert math.isclose(learning_rate(500, **kw), 5e-5)                     # and after it
    cool = {"start": 100, "steps": 20, "lr0": 6e-4}
    c = dict(kw, phase="cooldown", cool=cool)
    assert math.isclose(learning_rate(100, **c), 6e-4)                      # starts at the best step's rate
    assert math.isclose(learning_rate(110, **c), (6e-4 + 5e-5) / 2)         # linear
    assert math.isclose(learning_rate(120, **c), 5e-5)                      # ends at the minimum
    assert math.isclose(learning_rate(150, **c), 5e-5)
    vals = [learning_rate(s, **c) for s in range(100, 121)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))                      # never increases


def train(name, *extra, resume=False):
    cmd = [sys.executable, "scripts/train_model.py", "--name", name, *BASE, *extra] + (["--resume"] if resume else [])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


def steps_per_epoch(name):
    import json
    return json.loads((RUNS / name / "config.json").read_text())["steps_per_epoch"]


def done_step(out):
    return int(re.search(r"done: (\d+) steps", out).group(1))


def ckpt(name, which="last.pt"):
    return torch.load(RUNS / name / which, map_location="cpu", weights_only=False)


def fresh(name):
    shutil.rmtree(RUNS / name, ignore_errors=True)
    return name


def test_early_stop_cooldown_and_resume():
    name = fresh("_test_stop")
    try:
        out = train(name, "--patience", "0.001", "--cooldown", "0.001")
        spe = steps_per_epoch(name)
        pat, cool = max(1, int(0.001 * spe)), max(1, int(0.001 * spe))
        best = 3                                     # lr 0: the first eval is the best, nothing improves
        stop = next(s for s in range(3, 10_000, 3) if s - best >= pat)      # first eval past the patience
        assert f"early stop at step {stop} " in out, out
        assert f"cooldown: back to the best checkpoint (step {best})" in out, out
        assert f"cooldown finished at step {stop + cool}" in out, out
        assert done_step(out) == stop + cool and "(early stop + cooldown)" in out, out
        ck = ckpt(name)
        assert ck["phase"] == "done" and ck["step"] == stop + cool and ck["best_step"] == best
        assert ck["cool"] == {"start": stop, "steps": cool, "lr0": 0.0}

        out = train(name, "--patience", "0.001", "--cooldown", "0.001", resume=True)
        assert "already finished" in out and done_step(out) == stop + cool, out     # nothing trained

        # a job killed mid-cooldown: rewind the checkpoint into the cooldown, then resume
        ck["phase"], ck["step"] = "cooldown", stop + 1
        torch.save(ck, RUNS / name / "last.pt")
        out = train(name, "--patience", "0.001", "--cooldown", "0.001", resume=True)
        assert "phase cooldown" in out and f"cooldown finished at step {stop + cool}" in out, out
        assert done_step(out) == stop + cool and ckpt(name)["phase"] == "done", out
    finally:
        shutil.rmtree(RUNS / name, ignore_errors=True)


def test_no_cooldown_and_no_patience():
    name = fresh("_test_stop0")
    try:
        out = train(name, "--patience", "0.001", "--cooldown", "0")
        spe = steps_per_epoch(name)
        stop = next(s for s in range(3, 10_000, 3) if s - 3 >= max(1, int(0.001 * spe)))
        assert f"early stop at step {stop} " in out and "cooldown:" not in out, out
        assert done_step(out) == stop and ckpt(name)["phase"] == "done", out
    finally:
        shutil.rmtree(RUNS / name, ignore_errors=True)
    name = fresh("_test_nopat")
    try:
        out = train(name, "--patience", "0", "--max-steps", "12")
        assert "early stop" not in out and done_step(out) == 12, out
        assert ckpt(name)["phase"] == "done"
    finally:
        shutil.rmtree(RUNS / name, ignore_errors=True)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
