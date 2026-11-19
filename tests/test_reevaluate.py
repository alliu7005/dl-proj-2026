"""Tests for pace/reevaluate.sh: which runs it submits, and with what arguments.

    python -m tests.test_reevaluate        # or: pytest tests/test_reevaluate.py

Runs the real script with bash in a scratch copy of the repo layout, with stand-ins for squeue
(a list of queued jobs), sbatch (records each submission and its NAME / ARGS) and python3 (this
interpreter, so src.eval.plan runs exactly as on the login node). Needs bash; skipped without it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.plan import DEFAULT_MODES, margins_of, mode_names, stamp  # noqa: E402

ALL = DEFAULT_MODES.split(",")
NAMES = mode_names(ALL, 0.0, [])
MARGINS = margins_of(ALL, 0.0, [])


def complete(conds):
    out = {c: {n: stamp({"note_f1": 0.5}, MARGINS.get(n)) for n in NAMES} for c in conds}
    if "real" in conds:
        out["real_unseen_audio"] = out["real"]
    return out


def setup(tmp: Path):
    (tmp / "pace").mkdir()
    shutil.copy(ROOT / "pace" / "reevaluate.sh", tmp / "pace")
    (tmp / "pace" / "evaluate.sbatch").write_text("#!/bin/bash\n")
    shutil.copytree(ROOT / "src" / "eval", tmp / "src" / "eval", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp / "src" / "__init__.py").write_text("")
    for split in ("test", "test_transcribed", "test_periscope"):
        (tmp / "data" / "processed" / split).mkdir(parents=True)
        (tmp / "data" / "processed" / split / "index.npy").write_bytes(b"")
    runs = tmp / "outputs" / "runs"

    def run(name, results=None, phase=None, params=10_000_000):
        d = runs / name
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"")
        (d / "config.json").write_text(json.dumps({"params": params}))
        if results is not None:
            (d / "eval_pieces_best.json").write_text(json.dumps(results))
        if phase is not None:
            (d / "status.json").write_text(json.dumps({"phase": phase}))

    every = ("clean", "synthetic", "real", "periscope")
    run("done_all", complete(every))                          # nothing missing
    run("no_real", complete(("clean", "synthetic", "periscope")))
    run("big_no_real", complete(("clean", "synthetic", "periscope")), params=45_000_000)
    run("training", None, phase="cooldown")                   # best.pt exists, training not finished
    run("finished", complete(every), phase="done")
    run("queued")                                             # an evaluation already in the queue
    run("smoke_x")                                            # smoke tests are never evaluated
    (runs / "no_ckpt").mkdir(parents=True)

    bin_ = tmp / "bin"
    bin_.mkdir()
    (bin_ / "squeue").write_text('#!/bin/bash\necho "777 eval_queued"\n')
    # one line per submission: NAME, ARGS and sbatch's own arguments, tab-separated
    (bin_ / "sbatch").write_text('#!/bin/bash\nprintf "%s\\t%s\\t%s\\n" "$NAME" "$ARGS" "$*" >> "$SUBMITS"\necho 1000\n')
    py = Path(sys.executable).as_posix()
    (bin_ / "python3").write_text(f'#!/bin/bash\nexec "{py}" "$@"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    return bin_


def reevaluate(tmp, bin_, *runs, **env):
    submits = tmp / f"submits_{len(list(tmp.glob('submits_*')))}.txt"
    e = dict(os.environ, USER="me", SUBMITS=str(submits), **env)
    r = subprocess.run([shutil.which("bash"), "-c", f'export PATH="$PWD/bin:$PATH"; bash pace/reevaluate.sh "$@"',
                        "x", *runs], cwd=tmp, env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = submits.read_text().splitlines() if submits.exists() else []
    return r.stdout, {ln.split("\t")[0]: ln.split("\t") for ln in lines}  # name: [name, ARGS, sbatch args]


def test_reevaluate():
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        bin_ = setup(tmp)
        out, sub = reevaluate(tmp, bin_, ARGS="--conditions real")
        assert set(sub) == {"no_real", "big_no_real"}, out
        assert sub["no_real"][1] == "--conditions real --only-missing", sub
        assert "-t 2:30:00" in sub["no_real"][2] and "-t 4:00:00" in sub["big_no_real"][2], sub
        assert "-J eval_no_real" in sub["no_real"][2] and "--constraint=gpu-h100|gpu-h200|gpu-rtxpro-blackwell" \
            in sub["no_real"][2], sub
        assert "real: " + ",".join(NAMES) in out, out                 # what each job will compute
        assert "skip done_all: nothing missing" in out and "skip finished: nothing missing" in out, out
        assert "skip training: training not finished (phase cooldown)" in out, out
        assert "skip queued: already being evaluated (job 777)" in out and "smoke_x" not in out, out
        assert "2 job(s) submitted" in out, out

        # extra margins: every finished run gets the three new modes, and only those
        out, sub = reevaluate(tmp, bin_, "done_all", ARGS="--conditions clean --extra-margins -0.05")
        assert sub["done_all"][1] == "--conditions clean --extra-margins -0.05 --only-missing", sub
        assert "clean: guarded@-0.05,meter@-0.05,piece@-0.05\n" in out, out
        # FORCE recomputes existing results (without --only-missing), but never a run in training
        out, sub = reevaluate(tmp, bin_, "done_all", "training", FORCE="1")
        assert set(sub) == {"done_all"} and sub["done_all"][1] == "", sub
        # AFTER: one Slurm dependency for every submitted job (pace/beats.sh: after the trainings start)
        out, sub = reevaluate(tmp, bin_, "no_real", ARGS="--conditions real", AFTER="after:11:12")
        assert "--dependency=after:11:12" in sub["no_real"][2], sub
        out, sub = reevaluate(tmp, bin_, "no_real", ARGS="--conditions real")
        assert "--dependency" not in sub["no_real"][2], sub
        out, _ = reevaluate(tmp, bin_, "no_ckpt")
        assert "skip no_ckpt: outputs/runs/no_ckpt/best.pt not found" in out, out
        # a bad argument stops the script before anything is submitted
        e = dict(os.environ, USER="me", SUBMITS=str(tmp / "bad.txt"), ARGS="--modes guardd")
        r = subprocess.run([shutil.which("bash"), "-c", f'export PATH="$PWD/bin:$PATH"; bash pace/reevaluate.sh'],
                           cwd=tmp, env=e, capture_output=True, text=True)
        assert r.returncode != 0 and "unknown mode" in r.stdout and not (tmp / "bad.txt").exists(), r.stdout


def test_beats_batch():
    """pace/beats.sh: the job graph (names, arguments, dependencies) and the piece-mode evaluations."""
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        bin_ = setup(tmp)
        shutil.copy(ROOT / "pace" / "beats.sh", tmp / "pace")
        # sbatch: a new job id per call, one line per submission (NAME, ARGS, its arguments)
        (bin_ / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                     'echo $n > "$SUBMITS.n"\n'
                                     'printf "%s\\t%s\\t%s\\t%s\\n" "$n" "$NAME" "$ARGS" "$*" >> "$SUBMITS"\n'
                                     'echo $n\n')
        (bin_ / "sbatch").chmod(0o755)
        e = dict(os.environ, USER="me", SUBMITS=str(tmp / "jobs.txt"))
        r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/beats.sh'],
                           cwd=tmp, env=e, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        jobs = [ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()]
        by_id = {j[0]: j for j in jobs}
        prep, track = jobs[0], jobs[1]
        assert "-J beatprep" in prep[3] and "build_beats.py --check" in prep[3] and "tests.test_beats" in prep[3]
        assert f"--dependency=afterok:{prep[0]}" in track[3] and "track_beats.sbatch" in track[3]
        train = {j[1]: j for j in jobs if "train.sbatch" in j[3]}
        assert set(train) == {"sf_base_reg_ca_r", "sf_base_reg_cab", "s2s_m_cab"}, train.keys()
        assert f"afterok:{prep[0]}" in train["sf_base_reg_ca_r"][3] and "--beats" not in train["sf_base_reg_ca_r"][2]
        for name in ("sf_base_reg_cab", "s2s_m_cab"):
            assert f"afterok:{track[0]}" in train[name][3] and train[name][2].endswith("--beats"), train[name]
            assert "--context-before 64 --context-after 32 --aux-metrical" in train[name][2]
        # the control differs from the beat run only by --beats
        assert train["sf_base_reg_cab"][2] == train["sf_base_reg_ca_r"][2] + " --beats"
        ids = ":".join(train[n][0] for n in ("sf_base_reg_ca_r", "sf_base_reg_cab", "s2s_m_cab"))
        evals = [j for j in jobs if "evaluate.sbatch" in j[3]]
        assert evals and all(f"--dependency=after:{ids}" in j[3] for j in evals), evals
        assert all(j[2] == "--modes piece --only-missing" for j in evals) and len(by_id) == len(jobs)
        # done_all / finished already have piece results everywhere (built from the current modes);
        # no_real / big_no_real lack it on real; training / queued / smoke runs are never submitted
        assert {j[1] for j in evals} == {"no_real", "big_no_real"}, [j[1] for j in evals]
        assert "skip done_all: nothing missing" in r.stdout and "skip training: training not finished" in r.stdout


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
