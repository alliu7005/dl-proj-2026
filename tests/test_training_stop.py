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
  context + aux   context notes and the auxiliary metrical head are recorded and kept on resume
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


def status(name):
    import json
    return json.loads((RUNS / name / "status.json").read_text())


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
        st = status(name)  # the same state, readable without torch (pace/reevaluate.sh)
        assert st["phase"] == "done" and st["step"] == stop + cool and st["best_step"] == best, st
        assert st["stopped_early"] is True and st["best"] == ck["best"], st

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


def test_status_while_training():
    """status.json exists from the start and says "train" until the run is finished: a job killed
    mid-run (time limit) leaves a best.pt, but src/eval/plan.py must not treat the run as finished.
    The kill also lands at a random point, possibly mid-save: checkpoints are written atomically,
    so the resubmitted run still loads (a truncated last.pt used to make the resume crash)."""
    import time
    from src.eval.plan import training_state
    name = fresh("_test_status")
    cmd = [sys.executable, "scripts/train_model.py", "--name", name, *BASE, "--patience", "0",
           "--max-steps", "100000"]
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while not (RUNS / name / "last.pt").exists():  # the first eval has saved its checkpoints
            assert p.poll() is None and time.time() - t0 < 600, "training exited or never evaluated"
            time.sleep(0.2)
        st = status(name)
        assert st["phase"] == "train" and st["total_steps"] == 100000, st
        assert training_state(RUNS / name) == "train"
    finally:
        p.kill()
        p.wait()
    try:
        assert training_state(RUNS / name) == "train"  # killed: still unfinished
        assert ckpt(name)["step"] >= 3 and ckpt(name, "best.pt")["step"] >= 3  # complete, loadable
        out = train(name, "--patience", "0", "--max-steps", "6", resume=True)
        assert done_step(out) >= 6 and training_state(RUNS / name) == "done", out
    finally:
        shutil.rmtree(RUNS / name, ignore_errors=True)


def test_context_and_aux_training():
    """A short real run with context notes and the auxiliary metrical head: the checkpoint records
    both, validation logs the head's scores, and a resume without the flags keeps the run's inputs."""
    import csv
    import json
    name = fresh("_test_ctxaux")
    try:
        extra = ["--lr", "1e-3", "--patience", "0", "--max-steps", "6", "--context-before", "16",
                 "--context-after", "8", "--aux-metrical"]
        train(name, *extra)
        cfg = json.loads((RUNS / name / "config.json").read_text())["model"]
        assert cfg["n_features"] == 10 and cfg["context_before"] == 16 and cfg["context_after"] == 8
        assert cfg["aux_metrical"] is True
        rows = list(csv.DictReader(open(RUNS / name / "log.csv")))
        assert "val_asap_downbeat_f1" in rows[-1] and "val_asap_phase_acc" in rows[-1]
        ck = ckpt(name)
        assert any(k.startswith("aux.") for k in ck["model"])
        # resume mid-run with no context / aux flags at all: the run keeps what it was started with
        # (its data would not even fit the model otherwise: 10 input features, auxiliary labels)
        ck["phase"], ck["step"] = "train", 3
        torch.save(ck, RUNS / name / "last.pt")
        out = train(name, "--lr", "1e-3", "--patience", "0", "--max-steps", "6", resume=True)
        assert done_step(out) == 6 and "early stop" not in out, out
        cfg2 = ckpt(name)["config"]
        assert cfg2["context_before"] == 16 and cfg2["aux_metrical"] is True
        out = train(name, "--lr", "1e-3", "--patience", "0", "--max-steps", "6", resume=True)
        assert "already finished" in out and "early stop" not in out, out  # a finished run is not "early"
    finally:
        shutil.rmtree(RUNS / name, ignore_errors=True)


def test_batches_resume_mid_epoch():
    """TokenBudgetBatches: an epoch's batches depend on (seed, epoch) only, not on the global torch
    RNG (a resumed run re-seeds it); skipping k batches gives exactly the rest of the epoch; the
    epoch length counts whichever budget binds (notes here)."""
    import numpy as np
    from scripts.train import TokenBudgetBatches
    w = torch.ones(500, dtype=torch.double)
    tok, notes = np.full(500, 20), np.full(500, 100)

    def batches(epoch, global_seed, skip=0):
        torch.manual_seed(global_seed)
        b = TokenBudgetBatches(torch.utils.data.WeightedRandomSampler(w, 500, replacement=True), tok, notes,
                               max_tokens=400, max_notes=600, seed=3)
        b.set_epoch(epoch)
        b.skip = skip
        return list(b), len(b)
    e1, n = batches(1, 0)
    assert batches(1, 99)[0] == e1 and batches(2, 0)[0] != e1
    assert batches(1, 7, skip=10)[0] == e1[10:]
    assert len(batches(1, 0, skip=0)[0]) == len(e1)          # skip is used once
    assert abs(n - len(e1)) <= 0.1 * len(e1), (n, len(e1))   # 6 notes-bound per batch, not 20 token-bound


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
