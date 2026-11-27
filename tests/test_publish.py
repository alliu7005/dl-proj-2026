"""Tests for publishing finished runs to Hugging Face and the demo pulling them
(scripts/publish_model.py, scripts/download_models.py update, demo/app.py keep_updated, the
publish hook in pace/v2_jobs.sh).

    python -m tests.test_publish        # or: pytest tests/test_publish.py

The Hub is a stand-in (a folder as the repo, with commits and tags), so nothing leaves the machine.
Covers: the first run into an empty repo becomes the default; a worse run is archived only; the
same run again changes nothing; a better one replaces the default (its history kept); --force; a
meter model trained for PM2S, an unevaluated run, no login refused; the demo's update downloads only
when the repo's default changed (and never the archive), survives being offline, and swaps the new
models in; the batch jobs publish what they finish, PUBLISH=0 nothing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
V = ROOT / "data" / "processed" / "vocab.json"


class FakeHub:
    """huggingface_hub with a local folder as the model repo."""

    def __init__(self, root: Path, logged_in: bool = True):
        self.root, self.logged_in = root, logged_in
        self.commits, self.tags, self.downloads, self.offline = [], [], [], False
        hub = self

        class HfApi:
            def whoami(self):
                if not hub.logged_in:
                    raise RuntimeError("401 not logged in")
                return {"name": "me"}

            def create_repo(self, repo, repo_type, exist_ok):
                (hub.root / repo).mkdir(parents=True, exist_ok=True)

            def create_commit(self, repo_id, repo_type, operations, commit_message):
                for op in operations:
                    dst = hub.root / repo_id / op.path_in_repo
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(op.path_or_fileobj, dst)
                hub.commits.append((commit_message, sorted(op.path_in_repo for op in operations)))
                return f"commit {len(hub.commits)}"

            def create_tag(self, repo, tag, repo_type):
                hub.tags.append(tag)

        class CommitOperationAdd:
            def __init__(self, path_in_repo, path_or_fileobj):
                self.path_in_repo, self.path_or_fileobj = path_in_repo, path_or_fileobj

        self.HfApi, self.CommitOperationAdd = HfApi, CommitOperationAdd

    def hf_hub_download(self, repo_id, filename, repo_type):
        if self.offline:
            raise ConnectionError("offline")
        p = self.root / repo_id / filename
        if not p.exists():
            raise FileNotFoundError(f"404 EntryNotFound {filename}")
        return str(p)

    def snapshot_download(self, repo_id, revision, local_dir, repo_type, allow_patterns=None):
        import fnmatch
        src = self.root / repo_id
        for f in src.rglob("*"):
            rel = f.relative_to(src).as_posix()
            if f.is_file() and (allow_patterns is None or any(fnmatch.fnmatch(rel, p) for p in allow_patterns)):
                self.downloads.append(rel)
                (Path(local_dir) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, Path(local_dir) / rel)


def meter_run(d: Path, name: str, scores: dict, seed: int = 0, args: dict | None = None) -> Path:
    """A tiny meter run, finished and evaluated: (downbeat F + time-signature accuracy) / 2 = scores."""
    import torch

    from src.meter import MeterConfig, N_BEAT_FEATURES, build_meter_model
    torch.manual_seed(seed)
    run = d / name
    run.mkdir(parents=True)
    mc = MeterConfig(n_features=N_BEAT_FEATURES, d_model=32, layers=1, n_heads=4, ff=64).to_dict()
    torch.save({"model": build_meter_model(mc).state_dict(), "arch": "meter", "config": mc, "step": 10,
                "best_step": 8, "opt": {"x": 1}}, run / "best.pt")
    (run / "config.json").write_text(json.dumps({"arch": "meter", "args": {"tracked": "beat_this", "aux": "none",
                                                                           **(args or {})}}))
    (run / "decode.json").write_text(json.dumps({"decode": {"bar_weight": 2.0, "tempo_weight": 0.5}}))
    (run / "eval_meter.json").write_text(json.dumps(
        {c: {"grid_downbeat_f": v, "grid_ts_acc": v} for c, v in scores.items()}))
    return run


def test_publish():
    from scripts import publish_model as P
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        hub = FakeHub(tmp / "hub")
        runs = tmp / "runs"
        repo = ["--repo", "me/models", "--runs-dir", str(runs)]
        a = meter_run(runs, "meter_a", {"clean": 0.6, "real": 0.5, "gangqinpu": 0.7})
        assert P.main(["--run", str(a)] + repo, hub=hub) == 0
        m = json.loads((tmp / "hub" / "me/models" / "manifest.json").read_text())
        assert m["models"]["v2_meter"]["source_run"] == "meter_a" and m["release"].endswith("-v2_meter-meter_a")
        assert set(m["models"]["v2_meter"]["files"]) == {"model.pt", "config.json", "decode.json"}
        assert abs(m["models"]["v2_meter"]["scores"]["clean"] - 0.6) < 1e-9 and hub.tags == [m["release"]]
        files = hub.commits[0][1]
        assert "runs/meter_a/model.pt" in files and "v2_meter/model.pt" in files and "runs/meter_a/eval_meter.json" in files
        assert "README.md" in files and files.count("runs/meter_a/decode.json") == 1
        import torch
        ck = torch.load(tmp / "hub" / "me/models" / "v2_meter" / "model.pt", map_location="cpu", weights_only=False)
        assert "opt" not in ck                                                   # slimmed
        # worse: archived only
        b = meter_run(runs, "meter_b", {"clean": 0.5, "real": 0.5, "gangqinpu": 0.6, "periscope": 0.9}, seed=1)
        assert P.main(["--run", str(b)] + repo, hub=hub) == 0
        m = json.loads((tmp / "hub" / "me/models" / "manifest.json").read_text())
        assert m["models"]["v2_meter"]["source_run"] == "meter_a" and "meter_b" in m["archive"]
        assert not any(p.startswith("v2_meter/") for p in hub.commits[1][1]) and len(hub.tags) == 1
        # the same run again: nothing to do
        assert P.main(["--run", str(b)] + repo, hub=hub) == 0 and len(hub.commits) == 2
        assert P.main(["--run", str(a)] + repo, hub=hub) == 0 and len(hub.commits) == 2
        # better on the conditions both have: the new default, the old one in the history
        c = meter_run(runs, "meter_c", {"clean": 0.7, "real": 0.5, "gangqinpu": 0.8}, seed=2)
        assert P.main(["--run", str(c)] + repo, hub=hub) == 0
        m = json.loads((tmp / "hub" / "me/models" / "manifest.json").read_text())
        assert m["models"]["v2_meter"]["source_run"] == "meter_c" and m["history"][-1]["replaced"] == "meter_a"
        assert len(hub.tags) == 2
        # --force: the worse one becomes the default anyway
        assert P.main(["--run", str(b), "--force"] + repo, hub=hub) == 0
        assert json.loads((tmp / "hub" / "me/models" / "manifest.json").read_text())["models"]["v2_meter"]["source_run"] == "meter_b"
        # the default's own run with a file changed (re-tuned after publishing): the slot is updated
        n = len(hub.commits)
        (b / "decode.json").write_text(json.dumps({"decode": {"bar_weight": 4.0, "tempo_weight": 0.5}}))
        assert P.main(["--run", str(b)] + repo, hub=hub) == 0 and len(hub.commits) == n + 1
        assert "v2_meter/decode.json" in hub.commits[-1][1]
        m = json.loads((tmp / "hub" / "me/models" / "manifest.json").read_text())
        assert "files changed (decode.json)" in m["history"][-1]["why"]
        assert json.loads((tmp / "hub" / "me/models" / "v2_meter" / "decode.json").read_text())["decode"]["bar_weight"] == 4.0
        assert P.main(["--run", str(b)] + repo, hub=hub) == 0 and len(hub.commits) == n + 1   # and then nothing
        # refused: trained for PM2S, not tuned, not evaluated, not logged in
        pm = meter_run(runs, "meter_pm", {"clean": 0.9, "real": 0.9}, args={"tracked": "pm2s"})
        assert P.main(["--run", str(pm)] + repo, hub=hub) == 1
        nd = meter_run(runs, "meter_nd", {"clean": 0.9, "real": 0.9})
        (nd / "decode.json").unlink()
        assert P.main(["--run", str(nd)] + repo, hub=hub) == 1
        ne = meter_run(runs, "meter_ne", {"clean": 0.9, "real": 0.9})
        (ne / "eval_meter.json").unlink()
        assert P.main(["--run", str(ne)] + repo, hub=hub) == 1
        n = len(hub.commits)
        assert P.main(["--run", str(c)] + repo, hub=FakeHub(tmp / "hub", logged_in=False)) == 2
        assert P.main(["--run", str(c), "--dry-run"] + repo, hub=hub) == 0 and len(hub.commits) == n
        # the gate itself
        assert P.gate({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5})[0]                 # not lower: promoted
        assert not P.gate({"a": 0.9}, {"a": 0.1})[0]                                 # one shared condition: too few
        # the release's meter_v3 is scored by the run whose decoder it has (meter_v3_tp)
        tp = meter_run(runs, "meter_v3_tp", {"clean": 0.4, "real": 0.4})
        cur = {"source_run": "meter_v3", "sources": {"decode.json": str(tp / "decode.json")}}
        assert P.current_scores(cur, "meter", runs) == {"clean": 0.4, "real": 0.4}


def test_labeller_scores():
    """A labeller's gate: onset F1 with the true grid, per condition; NaN (no windows) left out."""
    from scripts import publish_model as P
    with tempfile.TemporaryDirectory() as t:
        run = Path(t) / "nl"
        run.mkdir()
        (run / "config.json").write_text(json.dumps({"arch": "labeller"}))
        (run / "eval_pieces_best.json").write_text(json.dumps({
            "clean": {"grid_true": {"onset_f1": 0.8}, "grid_meter_m": {"onset_f1": 0.5}},
            "real": {"grid_true": {"onset_f1": float("nan")}}, "gangqinpu": {"grid_true": {"onset_f1": 0.7}}}))
        assert P.kind_of(run) == "labeller" and P.scores(run, "labeller") == {"clean": 0.8, "gangqinpu": 0.7}
        assert P.incompatible(run, "labeller").endswith("best.pt not found")


def test_update_and_demo():
    """download_models.update: downloads the default (not the archive) when the repo's manifest
    differs from models/'s, nothing when it is the same, survives being offline; demo/app.py
    keep_updated swaps the new models in."""
    from scripts import download_models as D
    from scripts import publish_model as P
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        hub = FakeHub(tmp / "hub")
        runs = tmp / "runs"
        a = meter_run(runs, "meter_a", {"clean": 0.6, "real": 0.5})
        P.main(["--run", str(a), "--repo", "me/models"], hub=hub)
        b = meter_run(runs, "meter_b", {"clean": 0.1, "real": 0.1}, seed=1)
        P.main(["--run", str(b), "--repo", "me/models"], hub=hub)      # archived only
        out, logs = tmp / "models", []
        assert D.update("me/models", out, hub=hub, log=logs.append)
        assert (out / "v2_meter" / "model.pt").exists() and not (out / "runs").exists()   # not the archive
        assert not any(p.startswith("runs/") for p in hub.downloads) and not D.verify(out)
        n = len(hub.downloads)
        assert not D.update("me/models", out, hub=hub, log=logs.append) and len(hub.downloads) == n
        c = meter_run(runs, "meter_c", {"clean": 0.9, "real": 0.9}, seed=2)
        P.main(["--run", str(c), "--repo", "me/models"], hub=hub)
        assert D.update("me/models", out, hub=hub, log=logs.append)
        assert json.loads((out / "manifest.json").read_text())["models"]["v2_meter"]["source_run"] == "meter_c"
        hub.offline = True
        assert not D.update("me/models", out, hub=hub, log=logs.append) and "using what is here" in logs[-1]
        assert not D.update("", out, hub=hub)

        try:
            from demo.app import keep_updated
        except ModuleNotFoundError as e:   # the web server (fastapi / uvicorn) is not installed, e.g. on PACE
            print(f"  (demo/app.py keep_updated skipped: {e})")
            return
        pipes, calls = {"v2": "old"}, {"n": 0}

        class Stop(Exception):
            pass

        def sleep(_):
            calls["n"] += 1
            if calls["n"] > 3:
                raise Stop

        class P2:
            info = {"meter": "new"}
        checks = iter([False, True, True])
        try:
            keep_updated(pipes, 60, lambda: {"v2": P2()}, check=lambda log: next(checks), sleep=sleep, log=logs.append)
        except Stop:
            pass
        assert isinstance(pipes["v2"], P2) and "now serving" in logs[-1]


def test_jobs_publish():
    """pace/v2_jobs.sh publishes a meter model once it is trained and evaluated, a labeller after its
    evaluation; a failed upload does not fail the job; PUBLISH=0: nothing published."""
    from tests.batch_env import batch_env
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_jobs.sh", tmp / "pace")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "python").write_text(
            '#!/bin/bash\necho "$*" >> "$CALLS"\nname=""; prev=""\n'
            'for a in "$@"; do [ "$prev" = "--name" ] || [ "$prev" = "--run" ] && name=$a; prev=$a; done\n'
            'case "$1" in\n'
            '  scripts/train_meter.py|scripts/train_labeller.py) mkdir -p "outputs/runs/$name"; '
            'echo \'{"phase": "done"}\' > "outputs/runs/$name/status.json"; touch -d "2 days ago" "outputs/runs/$name/best.pt" ;;\n'
            '  scripts/eval_meter.py) echo "{}" > "outputs/runs/$name/eval_meter.json" ;;\n'
            '  scripts/publish_model.py) exit 2 ;;\n'
            'esac\nexit 0\n')
        (tmp / "bin" / "python").chmod(0o755)

        def jobs(args, env=None):
            calls = tmp / "calls.txt"
            if calls.exists():
                calls.unlink()
            r = subprocess.run([shutil.which("bash"), "-c", f'export PATH="$PWD/bin:$PATH"; bash pace/v2_jobs.sh {args}'],
                               cwd=tmp, env=batch_env(CALLS=str(calls), SLURM_JOB_ID="7", **(env or {})),
                               capture_output=True, text=True)
            return r.returncode, (calls.read_text().splitlines() if calls.exists() else []), r.stdout + r.stderr
        code, calls, out = jobs("meters meter_v4_gq")
        assert code == 0, out                                        # publishing failed (exit 2): the job is fine
        assert calls[-2].startswith("scripts/eval_meter.py --run meter_v4_gq") and \
            calls[-1] == "scripts/publish_model.py --run outputs/runs/meter_v4_gq", calls
        assert "not published" in out
        code, calls, out = jobs("labellers nl_gq")
        assert code == 0 and calls[-1] == "scripts/publish_model.py --run outputs/runs/nl_gq", calls
        shutil.rmtree(tmp / "outputs")
        code, calls, _ = jobs("meters meter_v4_gq", {"PUBLISH": "0"})
        assert code == 0 and not any("publish_model" in c for c in calls), calls


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
