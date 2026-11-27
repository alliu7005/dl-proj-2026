"""Tests for meter ensembles (src/meter.py MeterEnsemble, view_of, mean_logp, load_meter;
scripts/make_ensemble.py).

    python -m tests.test_ensemble        # or: pytest tests/test_ensemble.py

Covers: the identity view is the input itself, transposition stays on the keyboard, a stretch
scales every time; the mixture of log-probabilities is normalised and is the mean probability; a
one-member, one-view ensemble predicts exactly what its model does; two members average their
predictions, views average the model's predictions on each view; an ensemble run written by
make_ensemble.py loads by its folder like a trained run (and decodes into a grid), and the script
refuses what it cannot make (a missing or unfinished member, a duplicate, a nested ensemble, a
trained run's folder); members must read the same evidence.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.meter import (ENSEMBLE_FILE, IDENTITY_VIEW, N_ALL_FEATURES, TTA_VIEWS, MeterEnsemble,  # noqa: E402
                       build_meter_model, grid_from_logp, load_meter, mean_logp, predict_logp, view_of)

TINY = dict(d_model=32, n_heads=4, layers=1, ff=64, dropout=0.0)


def piece(n_beats: int = 24, seed: int = 0):
    """A small performance: beats every 0.5 s, a chord on each beat and a note between."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_beats) * 0.5 + 0.2
    rows = []
    for k, b in enumerate(t):
        for p in (48 + (k % 4), 60, 64):
            rows.append((p, b, b + 0.4, 70 + 10 * (k % 4 == 0)))
        rows.append((72 + int(rng.integers(0, 5)), b + 0.25, b + 0.45, 50))
    notes = np.array(rows, np.float32)
    down = np.zeros(n_beats, bool)
    down[::4] = True
    return notes, t, down


def model(seed: int, **kw):
    import torch
    torch.manual_seed(seed)
    return build_meter_model({**TINY, **kw}).eval()


def close(a: dict, b: dict, tol=1e-5):
    return set(a) == set(b) and all(np.allclose(a[k], b[k], atol=tol) for k in a)


def test_views():
    notes, t, _ = piece()
    n, bt, a = view_of(notes, t, None, IDENTITY_VIEW)
    assert n is notes and bt is t and a is None                       # the input itself
    n, bt, a = view_of(notes, t, (t, np.zeros(len(t), bool)), (3, 1.1))
    assert np.allclose(n[:, 0], notes[:, 0] + 3) and np.allclose(n[:, 1:3], notes[:, 1:3] * 1.1, atol=1e-5)
    assert np.allclose(bt, t * 1.1) and np.allclose(a[0], t * 1.1) and np.allclose(n[:, 3], notes[:, 3])
    high = notes.copy()
    high[0, 0] = 107                                                  # near the top of the keyboard
    n, _, _ = view_of(high, t, None, (3, 1.0))
    assert n[:, 0].max() == 108 and np.allclose(n[:, 0] - high[:, 0], 1)   # clipped to a 1-semitone shift
    assert len(view_of(np.zeros((0, 4), np.float32), t, None, (3, 0.9))[0]) == 0


def test_mean_logp():
    rng = np.random.default_rng(0)

    def lp():
        x = rng.normal(size=(5, 3))
        return {"bar": x - np.log(np.exp(x).sum(1, keepdims=True))}
    a, b = lp(), lp()
    m = mean_logp([a, b])
    assert np.allclose(np.exp(m["bar"]), (np.exp(a["bar"]) + np.exp(b["bar"])) / 2)
    assert np.allclose(np.exp(m["bar"]).sum(1), 1)
    assert np.allclose(mean_logp([a, a])["bar"], a["bar"])
    ta, tb = dict(a, tempo=np.array([4.0, 0.3])), dict(b, tempo=np.array([5.0, 0.4]))
    mt = mean_logp([ta, tb])["tempo"]
    assert np.isclose(mt[0], 4.5) and np.isclose(mt[1], np.sqrt((0.09 + 0.16) / 2 + 0.25))
    assert "tempo" not in mean_logp([ta, b])                         # only when every member has one


def test_ensemble_predictions():
    notes, t, down = piece()
    m0, m1 = model(0), model(1)
    one = predict_logp(m0, notes, t, down)
    assert close(predict_logp(MeterEnsemble([m0]), notes, t, down), one)       # one member, one view = the model
    two = predict_logp(MeterEnsemble([m0, m1]), notes, t, down)
    assert close(two, mean_logp([one, predict_logp(m1, notes, t, down)]))
    assert not close(two, one)
    tta = predict_logp(MeterEnsemble([m0], TTA_VIEWS), notes, t, down)
    by_view = []
    for v in TTA_VIEWS:
        vn, vt, _ = view_of(notes, t, None, v)
        by_view.append(predict_logp(m0, vn, vt, down))
    assert close(tta, mean_logp(by_view))
    assert all(tta[h].shape == one[h].shape for h in one)
    assert predict_logp(MeterEnsemble([m0, m1]), notes, t[:1], down[:1]) is None
    g = grid_from_logp(t, two)                                       # the decoder takes it as a model's
    assert len(g["bars"]) >= 1 and len(g["t"]) == len(t)
    try:
        MeterEnsemble([m0, model(2, n_features=N_ALL_FEATURES)])
        raise AssertionError("members reading different evidence were accepted")
    except ValueError:
        pass


def test_ensemble_run():
    import torch

    from scripts.make_ensemble import main as make
    notes, t, down = piece(seed=1)
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        ms = {}
        for i, name in enumerate(("ma", "mb")):
            ms[name] = model(i)
            (runs / name).mkdir()
            torch.save({"model": ms[name].state_dict(), "config": {**TINY}, "arch": "meter"}, runs / name / "best.pt")
            (runs / name / "status.json").write_text(json.dumps({"phase": "done"}))
        assert make(["--name", "ens", "--members", "ma", "mb", "--runs-dir", d]) == 0
        spec = json.loads((runs / "ens" / ENSEMBLE_FILE).read_text())
        assert spec["members"] == ["ma/best.pt", "mb/best.pt"] and spec["views"] == [[0, 1.0]]
        assert json.loads((runs / "ens" / "status.json").read_text())["phase"] == "done"
        assert not (runs / "ens" / "best.pt").exists()
        ens = load_meter(runs / "ens" / "best.pt")
        assert isinstance(ens, MeterEnsemble) and len(ens.members) == 2 and ens.names == spec["members"]
        want = mean_logp([predict_logp(ms["ma"], notes, t, down), predict_logp(ms["mb"], notes, t, down)])
        assert close(predict_logp(ens, notes, t, down), want)
        assert sum(p.numel() for p in ens.parameters()) == 2 * sum(p.numel() for p in ms["ma"].parameters())
        single = load_meter(runs / "ma" / "best.pt")                # a trained run loads as before
        assert not isinstance(single, MeterEnsemble) and close(predict_logp(single, notes, t, down),
                                                               predict_logp(ms["ma"], notes, t, down))
        assert make(["--name", "ens", "--members", "ma", "mb", "--runs-dir", d]) == 0   # the same again: kept
        assert make(["--name", "tta", "--members", "ma", "--views", "tta", "--runs-dir", d]) == 0
        assert len(load_meter(runs / "tta" / "best.pt").views) == len(TTA_VIEWS)
        (runs / "mc").mkdir()
        torch.save({"model": ms["ma"].state_dict(), "config": {**TINY}}, runs / "mc" / "best.pt")
        (runs / "mc" / "status.json").write_text(json.dumps({"phase": "train"}))
        for bad in (["--name", "ens", "--members", "ma"],                  # a different ensemble of that name
                    ["--name", "x", "--members", "ma", "nope"],            # a missing member
                    ["--name", "x", "--members", "ma", "ma"],              # listed twice
                    ["--name", "x", "--members", "ma", "ens"],             # an ensemble as a member
                    ["--name", "x", "--members", "ma", "mc"],              # not finished
                    ["--name", "ma", "--members", "mb", "--views", "tta"],  # a trained run's folder
                    ["--name", "x", "--members", "ma"]):                   # one member, no views
            try:
                make([*bad, "--runs-dir", d])
                raise AssertionError(f"accepted {bad}")
            except SystemExit as e:
                assert e.code not in (0, None), bad
        assert make(["--name", "ens", "--members", "mb", "ma", "--runs-dir", d, "--force"]) == 0
        assert load_meter(runs / "ens" / "best.pt").names == ["mb/best.pt", "ma/best.pt"]


def test_batch_graph():
    """pace/v2_ensemble.sh with stand-ins for sbatch / squeue: it stops when an earlier round's file
    is missing; with everything there, 7 jobs in order with the right dependencies, nothing
    published, the seeds trained by the job library, the three ensembles made from the seeds,
    whole pieces on both test sets, the report last; STAGE=eval: the last four only."""
    import shutil
    import subprocess

    from tests.batch_env import batch_env
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_ensemble.sh", tmp / "pace")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "squeue").write_text('#!/bin/bash\n')
        for f in (tmp / "bin").iterdir():
            f.chmod(0o755)
        bd = tmp / "bd"

        def run(env=None):
            if (tmp / "jobs.txt").exists():
                (tmp / "jobs.txt").unlink()
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_ensemble.sh'],
                               cwd=tmp, env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(bd), **(env or {})),
                               capture_output=True, text=True)
            jobs = ([(ln.split("\t")[0], ln.split("\t")[1]) for ln in (tmp / "jobs.txt").read_text().splitlines()]
                    if (tmp / "jobs.txt").exists() else [])
            return r, jobs
        r, jobs = run()
        assert r.returncode != 0 and not jobs and "missing: data/processed/vocab.json" in r.stdout, r.stdout + r.stderr
        for f in ("data/processed/vocab.json", "data/processed/meter/train_gangqinpu.pkl",
                  "data/processed/meter/val_gangqinpu.pkl", "data/processed/meter/test_gangqinpu.pkl",
                  "data/processed/meter/tracked_train_gangqinpu_beat_this.npz",
                  "data/processed/meter/tracked_test_gangqinpu_beat_this.npz", "configs/beat_noise_bt_gangqinpu.json",
                  "outputs/system_eval/manifest.json", "outputs/system_eval_gangqinpu/manifest.json",
                  "bd/MIDI2ScoreTF.ckpt"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        r, jobs = run()
        assert r.returncode == 0, r.stdout + r.stderr
        by = {cmd.split("-J ")[1].split()[0]: (jid, cmd) for jid, cmd in jobs}
        assert list(by) == ["e1tests", "e1prep", "e1train", "e1ens", "e1asap", "e1gq", "e1final"], list(by)
        tests, prep, train, ens, asap, gq, final = (by[k] for k in by)
        assert "tests.test_ensemble" in tests[1] and "--dependency" not in tests[1]
        assert f"afterok:{tests[0]}" in prep[1] and "install_run.py --run nl_gq" in prep[1]
        assert "ensemble meter_v4_gq_tta tta meter_v4_gq" in prep[1] and "gangqinpu_inputs.py" in prep[1]
        assert f"afterok:{tests[0]}" in train[1] and "meters meter_v4_gq_s1 meter_v4_gq_s2" in train[1]
        assert f"afterok:{train[0]}:{prep[0]}" in ens[1]
        assert "ensemble meter_v4_gq_ens3 none meter_v4_gq meter_v4_gq_s1 meter_v4_gq_s2" in ens[1]
        assert "ensemble meter_v4_gq_ens3t tta meter_v4_gq meter_v4_gq_s1 meter_v4_gq_s2" in ens[1]
        for j in (asap, gq):
            assert f"afterany:{ens[0]},afterok:{prep[0]}" in j[1] and "v2_jobs.sh systems" in j[1]
            assert "nl_gq+meter_v4_gq:tidy,d80,missing" in j[1]
        assert "--manifest outputs/system_eval_gangqinpu/manifest.json" in gq[1]
        assert "--manifest" not in asap[1].split("v2_jobs.sh")[1]
        assert f"afterany:{asap[0]}:{gq[0]}" in final[1] and "ensemble_report.py" in final[1]
        for _, cmd in jobs[1:6]:   # every job that could train or evaluate a run publishes nothing
            assert "PUBLISH=0" in cmd, cmd
        r, jobs = run({"STAGE": "eval"})
        names = [cmd.split("-J ")[1].split()[0] for _, cmd in jobs]
        assert r.returncode == 0 and names == ["e1ens", "e1asap", "e1gq", "e1final"], (names, r.stdout + r.stderr)
        assert "--dependency" not in jobs[0][1]


def test_job_library():
    """pace/v2_jobs.sh knows the new seeds (meter_v4_gq's flags with --seed) and their decoder
    tuning, and its ensemble task is listed."""
    import subprocess
    src = (ROOT / "pace" / "v2_jobs.sh").read_text()
    for m, seed in (("meter_v4_gq_s1", 1), ("meter_v4_gq_s2", 2)):
        out = subprocess.run(["bash", "-c", f'eval "$(sed -n "/^meter_flags()/,/^}}/p" pace/v2_jobs.sh)"; meter_flags {m}'],
                             cwd=ROOT, capture_output=True, text=True)
        base = subprocess.run(["bash", "-c", 'eval "$(sed -n "/^meter_flags()/,/^}/p" pace/v2_jobs.sh)"; meter_flags meter_v4_gq'],
                              cwd=ROOT, capture_output=True, text=True)
        assert out.returncode == 0 and out.stdout.strip() == f"{base.stdout.strip()} --seed {seed}", (out.stdout, base.stdout)
        ev = subprocess.run(["bash", "-c", f'eval "$(sed -n "/^meter_eval_flags()/,/^}}/p" pace/v2_jobs.sh)"; meter_eval_flags {m}'],
                            cwd=ROOT, capture_output=True, text=True)
        assert ev.stdout.strip() == "--tune --tempo-prior", ev.stdout
    assert "  ensemble)" in src and "localtempo, ensemble}" in src


def test_install_run():
    """scripts/install_run.py with a stand-in for the download: the published folder becomes a
    finished run (model.pt -> best.pt, its other files copied, status done) that loads; an existing
    run is kept; --unless-same installs nothing when the local weights are the published ones and
    installs when they differ; a run the repo lacks fails."""
    import shutil

    import torch

    from scripts.install_run import main as inst
    with tempfile.TemporaryDirectory() as d:
        pub = Path(d) / "pub"
        (pub / "runs" / "m").mkdir(parents=True)
        m = model(0)
        torch.save({"model": m.state_dict(), "config": {**TINY}, "arch": "meter"}, pub / "runs" / "m" / "model.pt")
        (pub / "runs" / "m" / "decode.json").write_text('{"decode": {}}')

        def fetch(repo, out, revision, allow):
            name = allow[0].split("/")[1]
            if not (pub / "runs" / name).exists():
                raise FileNotFoundError(name)
            shutil.copytree(pub / "runs" / name, Path(out) / "runs" / name)
        runs = Path(d) / "runs"
        assert inst(["--run", "m", "--runs-dir", str(runs)], fetch=fetch) == 0
        assert (runs / "m" / "best.pt").exists() and (runs / "m" / "decode.json").exists()
        assert not (runs / "m" / "model.pt").exists()
        assert json.loads((runs / "m" / "status.json").read_text())["phase"] == "done"
        assert not isinstance(load_meter(runs / "m" / "best.pt"), MeterEnsemble)
        assert inst(["--run", "m", "--runs-dir", str(runs)], fetch=fetch) == 0           # there: kept
        assert inst(["--run", "m", "--as", "m_hf", "--unless-same", "m", "--runs-dir", str(runs)], fetch=fetch) == 0
        assert not (runs / "m_hf").exists()                                               # the same weights
        torch.save({"model": model(5).state_dict(), "config": {**TINY}}, runs / "m" / "best.pt")
        assert inst(["--run", "m", "--as", "m_hf", "--unless-same", "m", "--runs-dir", str(runs)], fetch=fetch) == 0
        assert (runs / "m_hf" / "best.pt").exists()                                       # different: a new member
        assert inst(["--run", "nope", "--runs-dir", str(runs)], fetch=fetch) == 1


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
