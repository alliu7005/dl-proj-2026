"""PM2S retrained without our test pieces: scripts/train_pm2s.py (targets, augmentation, training
end to end, resuming, the stable settings), the weights' tag that keeps its outputs apart from the
released weights' (src/pm2s_beats.py weights_tag), scripts/pick_run.py, pace/v2_pm2s.sh and
pace/v2_pm2s2.sh.

    python -m tests.test_pm2s_train
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_pm2s  # noqa: E402
from src.pm2s_beats import TOLERANCE  # noqa: E402


@contextmanager
def pm2s_env(weights=None, tag=None):
    old = {k: os.environ.get(k) for k in ("PM2S_WEIGHTS", "PM2S_TAG")}
    for k, v in (("PM2S_WEIGHTS", weights), ("PM2S_TAG", tag)):
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def make_perf(name: str, n_beats: int = 40, ibi: float = 0.5, seed: int = 0) -> dict:
    """A steady 4/4 performance: a chord on every beat, a note between beats, beats annotated."""
    rng = np.random.default_rng(seed)
    t = 1.0 + ibi * np.arange(n_beats)
    notes = []
    for i, b in enumerate(t):
        for p in (48, 55, 64):
            notes.append([p + (i % 3), b + rng.normal(0, 0.005), b + ibi * 0.9, 70])
        notes.append([72, b + ibi / 2, b + ibi * 0.9, 50])
    notes.append([60, 0.2, 0.4, 40])                       # before the first beat: left out
    notes = np.array(sorted(notes, key=lambda r: r[1]), np.float32)
    down = np.arange(n_beats) % 4 == 0
    return {"performance": name, "piece": name, "dataset": "x", "notes": notes,
            "gt": {"t": t, "down": down}, "beats": None}


def write_meter(data: Path):
    m = data / "meter"
    m.mkdir(parents=True)
    for split, n in (("train", 3), ("val", 2), ("train_gangqinpu", 3), ("val_gangqinpu", 2)):
        perfs = [make_perf(f"{split}/{i}", ibi=0.45 + 0.05 * i, seed=i) for i in range(n)]
        pickle.dump(perfs, open(m / f"{split}.pkl", "wb"))


def test_targets():
    beats = np.array([1.0, 1.5, 2.0])
    onsets = np.array([0.98, 1.05, 1.25, 1.5, 1.6, 2.0, 2.3])
    yb, yd, yt = train_pm2s.targets(onsets, beats, np.array([1.0]))
    assert yb.tolist() == [1, 1, 0, 1, 0, 1, 0]                       # every note within 70 ms of a beat
    assert yd.tolist() == [1, 1, 0, 0, 0, 0, 0]
    assert yt.tolist() == [0, 50, 50, 50, 50, 0, 0]                   # 0.5 s = 50 classes; none outside
    assert abs(TOLERANCE - 0.07) < 1e-9
    _, _, far = train_pm2s.targets(np.array([1.2]), np.array([1.0, 6.0]), np.zeros(0))
    assert far.tolist() == [0]                                        # beyond 4 s: no tempo target


def test_example_augment_segment():
    ex = train_pm2s.example(make_perf("a"))
    assert ex["notes"][:, 1].min() >= ex["t"][0] - TOLERANCE           # the note before the first beat is out
    assert np.all(np.diff(ex["notes"][:, 1]) >= 0)
    rng = np.random.default_rng(0)
    for _ in range(30):
        a = train_pm2s.augment(ex, rng)
        assert a["notes"][:, 0].min() >= 0 and a["notes"][:, 0].max() <= 127
        assert np.all(np.diff(a["notes"][:, 1]) >= 0)
        r = a["t"][1] - a["t"][0]
        assert 0.5 / 1.2 - 1e-6 <= r <= 0.5 / 0.8 + 1e-6               # tempo x U(0.8, 1.2)
        seg, yb, yd, yt = train_pm2s.segment(a, 64, rng)
        assert len(seg) == 64 and len(yb) == len(yd) == len(yt) == 64 and yb.sum() > 0
    assert train_pm2s.example({**make_perf("b"), "notes": np.zeros((3, 4), np.float32)}) is None


def test_train_resume_and_load():
    """A few steps on CPU: best.pt is a state dict PM2SBeats loads; --resume goes on from last.pt."""
    from src.pm2s_beats import PM2SBeats
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        write_meter(tmp / "data")
        base = ["--data", str(tmp / "data"), "--runs-dir", str(tmp / "runs"), "--name", "p", "--device", "cpu",
                "--batch-size", "2", "--length", "64", "--eval-every", "2", "--normal-priority"]
        train_pm2s.main(base + ["--steps", "2"])
        run = tmp / "runs" / "p"
        st = json.loads((run / "status.json").read_text())
        assert st["phase"] == "done" and st["step"] == 2
        cfg = json.loads((run / "config.json").read_text())
        assert cfg["data"]["train_by_source"] == {"asap": 3, "gangqinpu": 3}
        assert abs(cfg["data"]["sample_share"]["gangqinpu"] - 0.5) < 1e-9
        assert set(cfg["data"]["val_by_source"]) == {"asap", "gangqinpu"}
        log = (run / "log.csv").read_text().splitlines()
        assert log[0].startswith("step,") and "gangqinpu_downbeat_f" in log[0] and len(log) == 2
        train_pm2s.main(base + ["--steps", "4", "--resume"])
        st = json.loads((run / "status.json").read_text())
        assert st["phase"] == "done" and st["step"] == 4
        assert len((run / "log.csv").read_text().splitlines()) == 3      # appended, one header
        tracker = PM2SBeats(run / "best.pt")
        bt, bd = tracker(make_perf("c")["notes"])
        assert bt.ndim == 1 and bd.dtype == bool and len(bt) == len(bd)
        train_pm2s.main(base + ["--steps", "8", "--resume", "--max-hours", "1e-9"])   # stops at its first evaluation
        st = json.loads((run / "status.json").read_text())
        assert st["phase"] == "stopped" and st["step"] == 6


def test_weights_tag():
    """Other PM2S weights: every output made with them gets their tag, the released ones' names stay."""
    from scripts.system_scores import system_name
    from scripts.track_meter_data import tracked_path
    from src.pm2s_beats import default_weights_path, tagged, weights_tag
    with pm2s_env():
        assert weights_tag() == "" and tagged("beats_compare") == "beats_compare"
        assert system_name(None, "", "", "nl+m", "pm2s", 0.95, True, "pm2s") == "v2_nl_m_d95_tidy_pm_rpm"
        assert tracked_path(Path("d"), "train_gangqinpu", "pm2s").name == "tracked_train_gangqinpu_pm2s.npz"
    with pm2s_env("w/best.pt", "ho"):
        assert weights_tag() == "ho" and default_weights_path() == Path("w/best.pt")
        assert system_name(None, "", "", "nl+m", "pm2s", 0.95, True, "pm2s") == "v2_nl_m_d95_tidy_pm_rpm_ho"
        assert system_name(None, "", "", "nl+m", "synth", 0.95, True, "audio") == "v2_nl_m_d95_tidy"   # no PM2S: no tag
        assert tracked_path(Path("d"), "train_gangqinpu", "pm2s").name == "tracked_train_gangqinpu_pm2s_ho.npz"
        assert tracked_path(Path("d"), "test_gangqinpu", "beat_this").name == "tracked_test_gangqinpu_beat_this.npz"
    for weights, tag in (("w/best.pt", None), (None, "ho"), ("w/best.pt", "h-o")):
        with pm2s_env(weights, tag):
            try:
                weights_tag()
            except SystemExit:
                pass
            else:
                raise AssertionError((weights, tag))


def test_batch_script():
    """pace/v2_pm2s.sh: tests, then PM2S trained; its beats stored and the existing systems with them
    after training; the meter model for it after the stored beats (two chained jobs); its evaluation
    after both; the final scoring after everything. Every job that runs PM2S gets its weights + tag."""
    from tests.batch_env import batch_env
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_pm2s.sh", tmp / "pace")
        for f in ("data/processed/meter/train_gangqinpu.pkl", "data/processed/meter/val_gangqinpu.pkl",
                  "outputs/runs/meter_v3_tp/decode.json", "outputs/runs/meter_v4_gq_pm/best.pt",
                  "outputs/runs/nl_base_gn/best.pt", "outputs/system_eval/manifest.json",
                  "data/raw/gangqinpu/data/test-0.parquet", "bd/MIDI2ScoreTF.ckpt"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "sbatch").chmod(0o755)

        def run_batch(env=None, ok=True):
            if (tmp / "jobs.txt").exists():
                (tmp / "jobs.txt").unlink()
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_pm2s.sh'],
                               cwd=tmp, env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd"),
                                                      **(env or {})), capture_output=True, text=True)
            assert (r.returncode == 0) == ok, r.stdout + r.stderr
            if not ok:
                return r.stdout + r.stderr
            return [ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()]
        jobs = run_batch()
        by = {cmd.split("-J ")[1].split()[0]: (jid, cmd) for jid, cmd in jobs}
        assert list(by) == ["phtests", "phtrain", "phtrack", "phsys", "phmeter", "phmeter2", "pheval", "phfinal"], list(by)
        assert len(by) <= 8                                                   # PACE's per-user job limit
        tests, train = by["phtests"][0], by["phtrain"][0]
        assert "--dependency" not in by["phtests"][1] and "tests.test_pm2s_train" in by["phtests"][1]
        tr = by["phtrain"][1]
        assert f"afterok:{tests}" in tr and "train_pm2s.py --name pm2s_ho" in tr and "--gres=gpu:1" in tr
        assert "--resume" not in tr and '"phase": "done"' in tr               # fails unless trained to the end
        assert "PM2S_WEIGHTS" not in tr
        for name in ("phtrack", "phsys", "phmeter", "phmeter2", "pheval"):     # every job that runs PM2S
            assert 'PM2S_WEIGHTS="$SLURM_SUBMIT_DIR/outputs/runs/pm2s_ho/best.pt" PM2S_TAG=ho' in by[name][1], name
        track = by["phtrack"][1]
        assert f"afterok:{train}" in track and "track_meter_data.py --release data/raw/gangqinpu --trackers pm2s" in track
        assert track.index("track_meter_data.py") < track.index("fit_pm2s_noise.py") < track.index("compare_beats.py")
        sysj = by["phsys"][1]
        assert f"afterok:{train}" in sysj
        assert "systems nl_base_gn+meter_v3_tp:tidy,d95,pm,rpm nl_base_gn+meter_v4_gq_pm:tidy,d95,pm,rpm" in sysj
        assert 'PM2S_RUNS="meter_v4_gq_pm" bash pace/v2_jobs.sh trackers meter_v4_gq_pm' in sysj
        assert f"afterok:{by['phtrack'][0]}" in by["phmeter"][1] and "meters meter_v4_gq_pmho" in by["phmeter"][1]
        assert f"afterany:{by['phmeter'][0]}" in by["phmeter2"][1] and "meters meter_v4_gq_pmho" in by["phmeter2"][1]
        ev = by["pheval"][1]
        assert f"afterany:{by['phmeter2'][0]}" in ev
        assert 'PM2S_RUNS="meter_v4_gq_pmho" bash pace/v2_jobs.sh trackers meter_v4_gq_pmho' in ev
        assert "systems nl_base_gn+meter_v4_gq_pmho:tidy,d95,pm,rpm" in ev and "evals nl_base_gn" in ev
        fin = by["phfinal"][1]
        assert f"afterany:{by['phsys'][0]}:{by['pheval'][0]}" in fin and "compare_systems.py" in fin
        assert "--gres" not in fin
        # a stopped PM2S run resumes
        (tmp / "outputs/runs/pm2s_ho").mkdir(parents=True)
        (tmp / "outputs/runs/pm2s_ho/last.pt").write_text("x")
        by = {cmd.split("-J ")[1].split()[0]: cmd for _, cmd in run_batch()}
        assert "train_pm2s.py --name pm2s_ho --normal-priority --max-hours 7.5 --resume" in by["phtrain"]
        # STAGE=meter: one more (resuming) meter job, then the evaluation and the scoring
        jobs = run_batch({"STAGE": "meter"})
        by = {cmd.split("-J ")[1].split()[0]: (jid, cmd) for jid, cmd in jobs}
        assert list(by) == ["phmeter3", "pheval", "phfinal"], list(by)
        assert "--dependency" not in by["phmeter3"][1] and f"afterany:{by['phmeter3'][0]}" in by["pheval"][1]
        assert f"afterany:{by['pheval'][0]}" in by["phfinal"][1]
        assert list(dict((c.split("-J ")[1].split()[0], 1) for _, c in run_batch({"STAGE": "eval"}))) == ["pheval", "phfinal"]
        assert "STAGE must be" in run_batch({"STAGE": "x"}, ok=False)
        (tmp / "outputs/runs/meter_v4_gq_pm/best.pt").unlink()
        assert "no outputs/runs/meter_v4_gq_pm/best.pt" in run_batch(ok=False)


def test_job_flags():
    """pace/v2_jobs.sh: meter_v4_gq_pmho trains on the held-out PM2S's errors and stored beats, its
    decoder tuned on them, and the trackers task tests it with PM2S."""
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    script = ('source <(sed -n "/^meter_flags()/,/^}/p;/^meter_eval_flags()/,/^}/p" pace/v2_jobs.sh); '
              'meter_flags meter_v4_gq_pmho; meter_eval_flags meter_v4_gq_pmho; '
              'eval "$(grep "^PM2S_RUNS=" pace/v2_jobs.sh)"; echo "$PM2S_RUNS"')
    from tests.batch_env import batch_env
    r = subprocess.run([shutil.which("bash"), "-c", script], cwd=ROOT, env=batch_env(), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    flags, ev, runs = r.stdout.strip().splitlines()
    assert "--beat-noise configs/beat_noise_pm2s_ho.json" in flags and "--tracked pm2s" in flags
    assert "--real-beats simulated" in flags and "--gangqinpu-share 0.5" in flags
    assert ev == "--tune --tempo-prior --tune-noise configs/beat_noise_pm2s_ho.json"
    assert runs.split() == ["meter_v4_gq_pm", "meter_v4_gq_pmho", "meter_v4_gq_pmhs"]
    r = subprocess.run([shutil.which("bash"), "-c", script.replace("pmho", "pmhs")], cwd=ROOT, env=batch_env(),
                       capture_output=True, text=True)
    flags, ev, _ = r.stdout.strip().splitlines()
    assert "--beat-noise configs/beat_noise_pm2s_hos.json" in flags and "--tracked pm2s" in flags
    assert ev == "--tune --tempo-prior --tune-noise configs/beat_noise_pm2s_hos.json"


def test_stable_settings():
    """The stable settings: warmup then cosine to --min-lr-frac; PM2S's step schedule unchanged by
    default; the gradient norm logged; clipping applied; bad --betas refused."""
    import argparse

    import torch
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    cos = train_pm2s.make_schedule(opt, argparse.Namespace(schedule="cosine", warmup=10, steps=110,
                                                           min_lr_frac=0.05, lr_step=40))
    lrs = []
    for _ in range(110):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        cos.step()
    assert abs(lrs[0] - 0.1) < 1e-9 and abs(lrs[9] - 1.0) < 1e-9          # warmup
    assert abs(lrs[60] - 0.525) < 0.02 and lrs[-1] < 0.06 and all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:]))
    opt2 = torch.optim.AdamW(model.parameters(), lr=1.0)
    st = train_pm2s.make_schedule(opt2, argparse.Namespace(schedule="step", warmup=0, steps=100, min_lr_frac=0.05,
                                                            lr_step=40))
    assert isinstance(st, torch.optim.lr_scheduler.StepLR)                 # PM2S's, as pm2s_ho trained
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        write_meter(tmp / "data")
        base = ["--data", str(tmp / "data"), "--runs-dir", str(tmp / "runs"), "--name", "s", "--device", "cpu",
                "--batch-size", "2", "--length", "64", "--eval-every", "2", "--normal-priority", "--steps", "4",
                "--lr", "3e-4", "--betas", "0.9,0.999", "--eps", "1e-8", "--clip", "1.0", "--schedule", "cosine",
                "--warmup", "2"]
        train_pm2s.main(base)
        run = tmp / "runs" / "s"
        log = (run / "log.csv").read_text().splitlines()
        head = log[0].split(",")
        assert "grad_norm" in head and float(log[-1].split(",")[head.index("grad_norm")]) > 0
        lr_last = float(log[-1].split(",")[head.index("lr")])
        assert lr_last < 3e-4                                               # decayed
        cfg = json.loads((run / "config.json").read_text())
        assert cfg["args"]["betas"] == "0.9,0.999" and cfg["args"]["clip"] == 1.0
        try:
            train_pm2s.main(base[:-12] + ["--name", "bad", "--betas", "0.9"])
            raise AssertionError("--betas 0.9 was accepted")
        except SystemExit as e:
            assert "--betas" in str(e)


def test_pick_run():
    """scripts/pick_run.py: the run with the higher best score is linked under the new name (with
    chosen.json); runs without best.pt are left out; a folder that is not a pick is not replaced."""
    from scripts import pick_run
    with tempfile.TemporaryDirectory() as t:
        runs = Path(t)
        for name, best in (("a", 0.61), ("b", 0.67), ("c", 0.9)):
            (runs / name).mkdir()
            (runs / name / "status.json").write_text(json.dumps({"best": best, "best_step": 10, "phase": "done"}))
            (runs / name / "config.json").write_text(json.dumps({"name": name}))
            if name != "c":                                                 # c has no best.pt: left out
                (runs / name / "best.pt").write_text(name)
        pick_run.main(["--runs", "a", "b", "c", "--as", "hos", "--runs-dir", str(runs)])
        chosen = json.loads((runs / "hos" / "chosen.json").read_text())
        assert chosen["chosen"] == "b" and set(chosen["candidates"]) == {"a", "b"}
        assert (runs / "hos" / "best.pt").read_text() == "b"
        pick_run.main(["--runs", "a", "b", "--as", "hos", "--runs-dir", str(runs), "--lower-better"])
        assert (runs / "hos" / "best.pt").read_text() == "a"                # replaced: it was a pick
        (runs / "real").mkdir()
        try:
            pick_run.main(["--runs", "a", "--as", "real", "--runs-dir", str(runs)])
            raise AssertionError("a run folder was replaced")
        except SystemExit as e:
            assert "not a pick" in str(e)
        try:
            pick_run.main(["--runs", "c", "--as", "x", "--runs-dir", str(runs)])
            raise AssertionError("picked from nothing")
        except SystemExit as e:
            assert "no run" in str(e)


def test_batch_script2():
    """pace/v2_pm2s2.sh: tests, two stable trainings side by side, the pick + tracking after both
    (finished or not), the systems and the meter model after the tracking, its evaluation after it,
    the scoring after everything; every PM2S job gets the picked weights and the tag hos."""
    from tests.batch_env import batch_env
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_pm2s2.sh", tmp / "pace")
        for f in ("data/processed/meter/train_gangqinpu.pkl", "data/processed/meter/val_gangqinpu.pkl",
                  "outputs/runs/meter_v3_tp/decode.json", "outputs/runs/meter_v4_gq/best.pt",
                  "outputs/runs/meter_v4_gq_pm/best.pt", "outputs/runs/nl_gq/best.pt", "outputs/runs/nl_base_gn/best.pt",
                  "outputs/system_eval/manifest.json", "outputs/system_eval_gangqinpu/manifest.json",
                  "outputs/system_eval_gangqinpu/beats_gangqinpu.npz", "data/raw/gangqinpu/data/test-0.parquet",
                  "bd/MIDI2ScoreTF.ckpt"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "sbatch").chmod(0o755)

        def run_batch(env=None):
            if (tmp / "jobs.txt").exists():
                (tmp / "jobs.txt").unlink()
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_pm2s2.sh'],
                               cwd=tmp, env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd"),
                                                      **(env or {})), capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            return {cmd.split("-J ")[1].split()[0]: (jid, cmd)
                    for jid, cmd in (ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines())}
        by = run_batch()
        assert list(by) == ["p2tests", "p2train_a", "p2train_b", "p2track", "p2sys", "p2meter", "p2eval", "p2final"], list(by)
        tests = by["p2tests"][0]
        for job, run, lr in (("p2train_a", "pm2s_ho2", "3e-4"), ("p2train_b", "pm2s_ho3", "1e-4")):
            cmd = by[job][1]
            assert f"afterok:{tests}" in cmd and f"train_pm2s.py --name {run} --lr {lr}" in cmd and "--gres=gpu:1" in cmd
            assert "--betas 0.9,0.999 --eps 1e-8 --clip 1.0 --schedule cosine --warmup 1000" in cmd and "PM2S_WEIGHTS" not in cmd
        tr = by["p2track"][1]
        assert f"afterany:{by['p2train_a'][0]}:{by['p2train_b'][0]}" in tr
        assert tr.index("pick_run.py --runs pm2s_ho2 pm2s_ho3 --as pm2s_hos") < tr.index("track_meter_data.py") \
            < tr.index("fit_pm2s_noise.py") < tr.index("compare_beats.py")
        for job in ("p2track", "p2sys", "p2meter", "p2eval"):
            assert 'PM2S_WEIGHTS="$SLURM_SUBMIT_DIR/outputs/runs/pm2s_hos/best.pt" PM2S_TAG=hos' in by[job][1], job
        sysj = by["p2sys"][1]
        assert f"afterok:{by['p2track'][0]}" in sysj and "systems nl_base_gn+meter_v3_tp:tidy,d95,pm,rpm" in sysj
        assert "systems nl_gq+meter_v4_gq:tidy,d95,rsb" in sysj and "--manifest outputs/system_eval_gangqinpu/manifest.json" in sysj
        assert sysj.count("systems nl_gq+meter_v4_gq:tidy,d95,tc") == 2                       # the timing check, both sets
        assert f"afterok:{by['p2track'][0]}" in by["p2meter"][1] and "meters meter_v4_gq_pmhs" in by["p2meter"][1]
        ev = by["p2eval"][1]
        assert f"afterany:{by['p2meter'][0]}" in ev and ev.count("systems nl_gq+meter_v4_gq_pmhs:tidy,d95,pm,rpm") == 2
        assert 'PM2S_RUNS="meter_v4_gq_pmhs" bash pace/v2_jobs.sh trackers meter_v4_gq_pmhs' in ev
        fin = by["p2final"][1]
        assert f"afterany:{by['p2sys'][0]}:{by['p2eval'][0]}" in fin and fin.count("beyer_dai.py score") == 2
        assert "--gres" not in fin
        by = run_batch({"STAGE": "eval"})
        assert list(by) == ["p2eval", "p2final"]
        by = run_batch({"STAGE": "meter"})
        assert list(by) == ["p2meter2", "p2eval", "p2final"]


def test_rendered_midi_beats():
    """system_scores.py --real-beats synth (_rsb): on a real-like condition (Gangqinpu's), v2 tracks
    the input MIDI rendered to audio itself (no stored beats); the batch jobs pass rsb."""
    from scripts import system_scores
    assert system_scores.system_name(None, "", "", "nl+m", "synth", 0.95, True, "synth") == "v2_nl_m_d95_tidy_rsb"
    assert "rsb) out=\"$out --real-beats synth\" ;;" in (ROOT / "pace" / "v2_jobs.sh").read_text()
    calls = []

    class Pipe:
        info = {"system": "v2"}

        def __init__(self, *a, **k):
            pass

        def run(self, midi, out, beats=None, **kw):
            calls.append((beats, kw))
            raise RuntimeError("stop here")
    old = system_scores.V2Pipeline
    system_scores.V2Pipeline = Pipe
    try:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "m.json").write_text(json.dumps([{"condition": "gangqinpu", "performance": "gangqinpu/1",
                                                     "slug": "gangqinpu_1", "midi": "x.mid", "reference": "r"}]))
            system_scores.main(["--v2", "nl+m", "--manifest", str(tmp / "m.json"), "--out", str(tmp / "o"),
                                "--conditions", "gangqinpu", "--real-beats", "synth", "--runs-dir", str(tmp)])
    finally:
        system_scores.V2Pipeline = old
    assert calls == [(None, {"beat_checkpoint": "final0"})], calls


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
