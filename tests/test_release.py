"""Tests for the model release (src/release.py, scripts/export_models.py, scripts/download_models.py).

    python -m tests.test_release        # or: pytest tests/test_release.py

Exports tiny untrained checkpoints the way the training scripts save them, checks what the release
holds (no optimizer state, the right vocabulary, the tuned decoder, sha256 sums), loads it as the
demo does, then "downloads" it (a stand-in for huggingface_hub) and catches a damaged file.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing.vocab import Vocab  # noqa: E402

V = ROOT / "data" / "processed" / "vocab.json"


def _with_opt(path: Path):
    """What a training checkpoint also holds: the optimizer and resume state."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck.update(opt={"state": {0: torch.zeros(1000)}}, step=1234, best_step=1000, best=0.5, epoch=3, phase="done")
    torch.save(ck, path)


def test_default_path():
    """models/<file> when it is there, else the training run's checkpoint."""
    import src.release as rel
    old = rel.ROOT
    with tempfile.TemporaryDirectory() as t:
        try:
            rel.ROOT = Path(t)
            assert rel.default_path("v2_labeller") == str(Path("outputs/runs/nl_base_gn/best.pt"))
            f = Path(t) / "models" / "v2_labeller" / "model.pt"
            f.parent.mkdir(parents=True)
            f.write_text("x")
            assert rel.default_path("v2_labeller") == str(Path("models/v2_labeller/model.pt"))
            assert rel.default_path("seq2seq") == str(Path("outputs/runs/sf_base_reg_cab/best.pt"))
        finally:
            rel.ROOT = old
    assert set(rel.RELEASE) == {"v2_labeller", "v2_meter", "seq2seq"} and rel.DELETE_ABOVE == 0.95 and rel.TIDY


def test_export_and_download():
    """export_models.py on tiny checkpoints: each model without its optimizer state, its vocabulary
    (the run's own, else the first candidate that fits: a vocabulary one token short is passed
    over), the decoder given, the model card, sha256 sums; the release loads as the demo loads it.
    download_models.py fetches it (stand-in) and verifies it; a damaged file fails the check."""
    if not V.exists():
        print("  (skipped: no data/processed/vocab.json)")
        return
    from scripts import download_models, export_models
    from tests.test_pipeline import make_checkpoint
    from tests.test_v2_system import make_v2
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        lab, met = make_v2(tmp / "runs")
        seq_dir = tmp / "runs" / "s"
        seq_dir.mkdir()
        seq = make_checkpoint(seq_dir)
        (seq_dir / "vocab.json").unlink()                       # trained before runs kept their vocabulary
        shutil.copy(V, lab.parent / "vocab.json")                # kept by train_labeller.py (pin_vocab)
        for ck in (lab, met, seq):
            _with_opt(ck)
        short = Vocab.load(str(V))
        Vocab(short.itos[:-1], short.counts).save(str(tmp / "short.json"))   # another data build's tokens
        decode = tmp / "tp_decode.json"
        decode.write_text(json.dumps({"decode": {"bar_weight": 2.0, "tempo_weight": 1.0}}))
        card = tmp / "card.md"
        card.write_text("# models\n")
        out = tmp / "models"
        export_models.main(["--out", str(out), "--labeller", str(lab.parent), "--meter", str(met.parent),
                            "--decode", str(decode), "--seq2seq", str(seq_dir), "--vocab", str(tmp / "short.json"),
                            "--vocab", str(V), "--card", str(card)])
        assert (out / "README.md").read_text() == "# models\n"
        m = json.loads((out / "manifest.json").read_text())
        assert set(m["models"]) == {"v2_labeller", "v2_meter", "seq2seq"} and m["settings"] == {"delete_above": 0.95,
                                                                                               "tidy": True}
        sizes = {"v2_labeller": lab, "v2_meter": met, "seq2seq": seq}
        for name in m["models"]:
            ck = torch.load(out / name / "model.pt", map_location="cpu", weights_only=False)
            assert "opt" not in ck and "epoch" not in ck and ck["step"] == 1234 and ck["best_step"] == 1000, ck.keys()
            assert m["models"][name]["files"]["model.pt"]["bytes"] < sizes[name].stat().st_size, name
        assert m["models"]["seq2seq"]["sources"]["vocab.json"] == str(V)      # the short one does not fit
        assert m["models"]["v2_labeller"]["sources"]["vocab.json"] == str(lab.parent / "vocab.json")  # its own
        assert json.loads((out / "v2_meter" / "decode.json").read_text())["decode"]["tempo_weight"] == 1.0
        assert set(m["models"]["seq2seq"]["files"]) >= {"model.pt", "vocab.json"}
        assert not download_models.verify(out)

        fetched = []

        def snapshot_download(repo_id, revision, local_dir, repo_type):   # stand-in for the Hub
            fetched.append((repo_id, revision, repo_type))
            shutil.copytree(out, local_dir, dirs_exist_ok=True)
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = snapshot_download
        sys.modules["huggingface_hub"] = hub
        try:
            dst = tmp / "downloaded"
            download_models.main(["--repo", "someone/models", "--revision", "v2-round4", "--out", str(dst)])
            assert fetched == [("someone/models", "v2-round4", "model")]
            with open(dst / "v2_meter" / "model.pt", "ab") as f:
                f.write(b"x")
            assert any("sha256" in b for b in download_models.verify(dst))
            (dst / "seq2seq" / "vocab.json").unlink()
            assert any("missing" in b for b in download_models.verify(dst))
            try:
                download_models.main(["--repo", "", "--out", str(dst)])
                raise AssertionError("no repo: refused")
            except SystemExit as e:
                assert "--repo" in str(e)
        finally:
            del sys.modules["huggingface_hub"]


def test_export_refuses():
    """No checkpoint, no fitting vocabulary, no tuned decoder: a clear error, not a broken release."""
    if not V.exists():
        print("  (skipped: no data/processed/vocab.json)")
        return
    from scripts import export_models
    from tests.test_v2_system import make_v2
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        from tests.test_pipeline import make_checkpoint
        lab, met = make_v2(tmp / "runs")
        seq_dir = tmp / "runs" / "s"
        seq_dir.mkdir()
        make_checkpoint(seq_dir)
        (seq_dir / "vocab.json").unlink()
        short = Vocab.load(str(V))
        Vocab(short.itos[:-1], short.counts).save(str(tmp / "short.json"))
        ok = ["--labeller", str(lab.parent), "--meter", str(met.parent)]
        for args, msg in ((["--labeller", str(tmp / "none"), "--no-seq2seq"], "not found"),
                          (ok + ["--seq2seq", str(seq_dir), "--vocab", str(tmp / "short.json")], "fits"),
                          (ok + ["--vocab", str(V), "--decode", str(tmp / "none.json"), "--no-seq2seq"],
                           "tuned decoder")):
            try:
                export_models.main(["--out", str(tmp / "models"), "--no-check"] + args)
                raise AssertionError(f"{args}: not refused")
            except (FileNotFoundError, ValueError) as e:
                assert msg in str(e), (args, e)
            assert not (tmp / "models").exists()                  # nothing half-written in its place


def test_upload():
    """upload_models.py: refuses a release that does not match its manifest or has no model card;
    otherwise creates the repo (if needed), uploads the folder, tags it with the release's name once
    (a second upload of the same release leaves the tag)."""
    from scripts import upload_models
    calls = []

    class Api:
        tags = []

        def create_repo(self, repo_id, repo_type, private, exist_ok):
            calls.append(("create", repo_id, private, exist_ok))
            return f"https://huggingface.co/{repo_id}"

        def upload_folder(self, repo_id, folder_path, repo_type, commit_message, ignore_patterns):
            calls.append(("upload", repo_id, Path(folder_path).name, commit_message))
            return types.SimpleNamespace(commit_url="https://huggingface.co/x/commit/1")

        def list_repo_refs(self, repo_id, repo_type):
            return types.SimpleNamespace(tags=[types.SimpleNamespace(name=t) for t in Api.tags])

        def create_tag(self, repo_id, tag, repo_type):
            calls.append(("tag", repo_id, tag))
            Api.tags.append(tag)
    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = Api
    sys.modules["huggingface_hub"] = hub
    try:
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "models"
            (out / "v2_meter").mkdir(parents=True)
            (out / "v2_meter" / "model.pt").write_bytes(b"weights")
            from scripts.download_models import sha256
            manifest = {"release": "v2-test", "created": "2026-11-24",
                        "models": {"v2_meter": {"files": {"model.pt": {"sha256": sha256(out / "v2_meter" / "model.pt")}}}}}
            (out / "manifest.json").write_text(json.dumps(manifest))
            for setup, msg in ((lambda: None, "model card"),
                               (lambda: (out / "v2_meter" / "model.pt").write_bytes(b"other"), "manifest")):
                setup()
                try:
                    upload_models.main(["--repo", "someone/models", "--models", str(out)])
                    raise AssertionError("not refused")
                except SystemExit as e:
                    assert msg in str(e), e
                (out / "v2_meter" / "model.pt").write_bytes(b"weights")
                (out / "README.md").write_text("# card")
            assert not calls
            upload_models.main(["--repo", "someone/models", "--models", str(out), "--private"])
            upload_models.main(["--repo", "someone/models", "--models", str(out)])
            assert calls == [("create", "someone/models", True, True), ("upload", "someone/models", "models", "release v2-test"),
                             ("tag", "someone/models", "v2-test"),
                             ("create", "someone/models", False, True), ("upload", "someone/models", "models", "release v2-test")]
    finally:
        del sys.modules["huggingface_hub"]


def test_release_batch():
    """pace/v2_release.sh: the tests first; the release's whole-piece systems after them (cancelled
    if they fail), on any bf16 NVIDIA GPU; the export and the scoring after the systems (finished or
    not); refused without the system comparison set up."""
    import subprocess
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    from tests.batch_env import batch_env
    from tests.test_v2_system import _stub_bin
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_release.sh", tmp / "pace")
        _stub_bin(tmp)
        bd = tmp / "bd"
        bd.mkdir()

        def run():
            if (tmp / "jobs.txt").exists():
                (tmp / "jobs.txt").unlink()
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_release.sh'], cwd=tmp,
                               env=batch_env(SUBMITS=str(tmp / "jobs.txt"), BD=str(bd)), capture_output=True, text=True)
            rows = [ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()] if (tmp / "jobs.txt").exists() else []
            return r, rows
        r, rows = run()
        assert r.returncode == 1 and not rows and "bd_setup.sh" in r.stdout
        (tmp / "outputs" / "system_eval").mkdir(parents=True)
        (tmp / "outputs" / "system_eval" / "manifest.json").write_text("[]")
        (bd / "MIDI2ScoreTF.ckpt").write_text("x")
        r, rows = run()
        assert r.returncode == 0 and len(rows) == 3, (r.stdout + r.stderr, rows)
        by = {row[3].split("-J ")[1].split()[0]: row for row in rows}
        tests, sys_, fin = by["reltests"], by["relsys"], by["relfinal"]
        assert "--dependency" not in tests[3] and "tests.test_release" in tests[3]
        assert f"--dependency=afterok:{tests[0]}" in sys_[3] and "--kill-on-invalid-dep=yes" in sys_[3]
        assert "--constraint=cuda13" in sys_[3] and "--gres=gpu:1" in sys_[3]
        assert "v2_jobs.sh systems nl_base_gn+meter_v3_tp:tidy,d95 nl_base_gn+meter_v3_tp:tidy,d80 " \
               "nl_base_gn_s1+meter_v3_tp:tidy,d95" in sys_[3], sys_[3]
        assert f"--dependency=afterany:{sys_[0]}" in fin[3] and "--gres" not in fin[3]
        assert "scripts/export_models.py" in fin[3] and "beyer_dai.py score" in fin[3]
        assert fin[3].index("export_models.py") < fin[3].index("beyer_dai.py") < fin[3].index("compare_systems.py")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
