"""Tests for the rule verdict: scripts/rule_verdict.py's decisions, publish_model.py --settings, the
demo's models picking a settings-only change up (download_models.update), src/release.py
rule_settings, and pace/v2_decode2.sh.

    python -m tests.test_rule_verdict        # or: pytest tests/test_rule_verdict.py
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.rule_verdict import base_name, report, verdict, with_suffixes  # noqa: E402


def fake_results(effects: dict, conditions=("real", "gangqinpu_real"), pieces: int = 40, noise: float = 0.3):
    """{(system, condition): {performance: error}}: a base error per performance, each system's
    effect added (+ a little noise per performance), two performances per piece."""
    rng = random.Random(0)
    out = {}
    for c in conditions:
        perfs = [f"Comp/P{k}/perf{j}.mid" for k in range(pieces) for j in range(2)]
        base = {p: 10 + rng.random() * 4 for p in perfs}
        for system, eff in effects.items():
            e = eff.get(c, 0.0) if isinstance(eff, dict) else eff
            out[(system, c)] = {p: base[p] + e + rng.gauss(0, noise) for p in perfs}
    return out


def test_names():
    b = base_name("nl_gq_mc")
    assert b == "v2_nl_gq_mc_meter_v4_gq_d95_tidy"
    assert with_suffixes(b, "bd", "tc", "jf") == b + "_tc_jf_bd"
    assert with_suffixes(b, "bd", "rd", "tb", "jf", "tc") == b + "_tc_tb_rd_jf_bd"
    from scripts.system_scores import system_name   # the scoring script names them the same way
    assert system_name(None, "", "", "nl_gq_mc+meter_v4_gq", "synth", 0.95, True, "audio", timing_check=True,
                       timing_beats=True, readable=True, fix_jumps=True, beat_decode=True) == with_suffixes(
        b, "tc", "tb", "rd", "jf", "bd")


def test_verdict():
    b = base_name("L")
    dl = with_suffixes(b, "bd")
    res = fake_results({
        b: 0.0,
        dl: 0.0,                                              # the decoding: no difference -> stays on
        with_suffixes(b, "tc", "bd"): {"real": 0.0, "gangqinpu_real": -1.0},   # helps on pop, not on classical: on
        with_suffixes(b, "tb", "bd"): 0.0,                    # nothing: off
        with_suffixes(b, "jf", "bd"): {"real": -1.0, "gangqinpu_real": 1.0},   # helps one, hurts the other: off
        with_suffixes(b, "tc", "tb", "rd", "jf", "bd"): -0.5,
    })
    v = verdict(res, "L")
    assert v["rules"] == {"beat_decode": True, "timing_check": True, "timing_beats": False, "fix_jumps": False}
    assert v["evidence"]["timing_check"]["gangqinpu_real"]["verdict"] == -1
    assert v["evidence"]["fix_jumps"]["gangqinpu_real"]["verdict"] == 1
    assert v["deep_learning_config"] == dl and set(v["evidence"]["all_rules"]) == {"real", "gangqinpu_real"}
    md = report(v)
    assert "| timing_beats | **off** |" in md and "| timing_check | on |" in md
    # the decoding hurting somewhere: off
    res2 = fake_results({b: 0.0, dl: {"real": 1.0}})
    assert verdict(res2, "L")["rules"]["beat_decode"] is False
    # missing systems: undecided (None), which keeps the current setting
    v3 = verdict(fake_results({b: 0.0}), "L")
    assert v3["rules"] == {"beat_decode": None, "timing_check": None, "timing_beats": None, "fix_jumps": None}
    # a condition missing (its job failed): undecided, though the other condition is clear
    part = {k: d for k, d in fake_results({b: 0.0, dl: 0.0, with_suffixes(b, "tc", "bd"): -1.0}).items()
            if not (k[1] == "gangqinpu_real" and k[0] != b)}
    vp = verdict(part, "L")
    assert vp["rules"]["timing_check"] is None and vp["rules"]["beat_decode"] is None
    assert set(vp["evidence"]["timing_check"]) == {"real"}
    # one condition only, asked for
    v4 = verdict(fake_results({b: 0.0, dl: 0.0, with_suffixes(b, "tc", "bd"): -1.0}, conditions=("real",)), "L",
                 conditions=["real"])
    assert v4["rules"]["timing_check"] is True and v4["conditions"] == ["real"]


def test_main_writes(tmp=None):
    from scripts import rule_verdict as RV
    b = base_name("L")
    res = fake_results({b: 0.0, with_suffixes(b, "bd"): 0.0, with_suffixes(b, "tb", "bd"): -1.0})
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        files = []
        for path, cond in RV.CONDITIONS:
            pieces = [{"system": s, "condition": c, "performance": p, "muster": {"MeanER": e}}
                      for (s, c), d in res.items() if c == cond for p, e in d.items()]
            f = tmp / path
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"pieces": pieces}))
            files.append((str(f), cond))
        old = RV.CONDITIONS
        RV.CONDITIONS = tuple(files)
        try:
            RV.main(["--labeller", "L", "--out", str(tmp / "v.json")])
        finally:
            RV.CONDITIONS = old
        v = json.loads((tmp / "v.json").read_text())
        assert v["rules"]["timing_beats"] is True and v["rules"]["beat_decode"] is True
        assert (tmp / "v.md").exists()


def test_publish_and_follow():
    """publish_model.py --settings -> the manifest; download_models.update picks a settings-only change
    up without downloading; src/release.py rule_settings reads it."""
    from scripts import publish_model as P
    from scripts.download_models import update
    from src.release import DECIDED_RULES, rule_settings
    from tests.test_publish import FakeHub
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        hub = FakeHub(tmp / "hub")
        repo = tmp / "hub" / "me/models"
        repo.mkdir(parents=True)
        (repo / "manifest.json").write_text(json.dumps({"release": "r1", "models": {}, "settings": {"delete_above": 0.95}}))
        # the demo's copy, as downloaded
        local = tmp / "models"
        local.mkdir()
        (local / "manifest.json").write_text((repo / "manifest.json").read_text())
        assert rule_settings(local) == DECIDED_RULES
        v = {"date": "2026-12-01", "labeller": "nl_gq_mc", "rule": "x",
             "rules": {"beat_decode": True, "timing_check": False, "timing_beats": False, "fix_jumps": None},
             "evidence": {"timing_check": {}}}
        vf = tmp / "v.json"
        vf.write_text(json.dumps(v))
        assert P.main(["--settings", str(vf), "--repo", "me/models"], hub=hub) == 0
        m = json.loads((repo / "manifest.json").read_text())
        assert m["settings"]["rules"] == {"beat_decode": True, "timing_check": False, "timing_beats": False}
        assert m["settings"]["delete_above"] == 0.95 and m["settings"]["rule_verdict"]["labeller"] == "nl_gq_mc"
        assert hub.commits[-1][1] == ["manifest.json"]
        assert P.main(["--settings", str(vf), "--repo", "me/models"], hub=hub) == 0 and len(hub.commits) == 1   # again: nothing
        # the demo's next check: settings changed, nothing to download
        logs = []
        assert update("me/models", local, hub=hub, log=logs.append) is True
        assert not hub.downloads and "settings changed" in logs[-1]
        r = rule_settings(local)
        assert r == {"timing_check": False, "timing_beats": False, "fix_jumps": DECIDED_RULES["fix_jumps"],
                     "beat_decode": True}
        assert update("me/models", local, hub=hub, log=logs.append) is False      # and then up to date
        # not logged in: not published
        assert P.main(["--settings", str(vf), "--repo", "me/models"], hub=FakeHub(tmp / "hub", logged_in=False)) == 2
    # the demo and the CLI take the decided rules from rule_settings (their --no-* still turn off)
    for f in ("demo/app.py", "scripts/audio_to_score.py"):
        assert "rule_settings()" in (ROOT / f).read_text(encoding="utf-8"), f


def test_batch():
    """pace/v2_decode2.sh: 4 jobs; the systems wait for the tests and for pace/v2_decode.sh's tuning /
    evaluation jobs still queued; the verdict after the systems and its scoring job; PUBLISH=0."""
    import shutil
    import subprocess

    from tests.batch_env import batch_env
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    for publish, queued in (("1", True), ("0", False)):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "pace").mkdir()
            shutil.copy(ROOT / "pace" / "v2_decode2.sh", tmp / "pace")
            for f in ("outputs/runs/nl_gq/best.pt", "outputs/runs/meter_v4_gq/best.pt", "outputs/system_eval/manifest.json",
                      "outputs/system_eval_gangqinpu/manifest.json", "bd/MIDI2ScoreTF.ckpt"):
                (tmp / f).parent.mkdir(parents=True, exist_ok=True)
                (tmp / f).write_text("x")
            (tmp / "bin").mkdir()
            (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                                'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
            q = ('#!/bin/bash\nfor a in "$@"; do case "$prev" in -n) case "$a" in dctune) echo 7001;; dceval) echo 7002;; '
                 'dcfinal) echo 7005;; esac;; esac; prev=$a; done\n') if queued else "#!/bin/bash\nexit 0\n"
            (tmp / "bin" / "squeue").write_text(q)
            for b in ("sbatch", "squeue"):
                (tmp / "bin" / b).chmod(0o755)
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_decode2.sh'], cwd=tmp,
                               env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd"), PUBLISH=publish),
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            by = {cmd.split("-J ")[1].split()[0]: (jid, cmd)
                  for jid, cmd in (ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines())}
            assert list(by) == ["d2tests", "d2asap", "d2gq", "d2final"]
            assert "tests.test_rule_verdict" in by["d2tests"][1]
            for j in ("d2asap", "d2gq"):
                s = by[j][1]
                want = f"--dependency=afterok:{by['d2tests'][0]}" + (",afterany:7001:7002" if queued else " ")
                assert want in s, (j, s[:200])
                for spec in ("nl_gq+meter_v4_gq:tidy,d95,bd,missing", "nl_gq+meter_v4_gq:tidy,d95,tb,bd,missing",
                             "nl_gq+meter_v4_gq:tidy,d95,jf,bd,missing", "nl_gq_mc+meter_v4_gq:tidy,d95,missing",
                             "nl_gq_mc+meter_v4_gq:tidy,d95,bd,missing", "nl_gq_mc+meter_v4_gq:tidy,d95,tc,bd,missing",
                             "nl_gq_mc+meter_v4_gq:tidy,d95,tb,bd,missing", "nl_gq_mc+meter_v4_gq:tidy,d95,jf,bd,missing",
                             "nl_gq_mc+meter_v4_gq:tidy,d95,tc,tb,rd,jf,bd,missing"):
                    assert f" {spec}" in s, (j, spec)
            assert "--manifest outputs/system_eval_gangqinpu/manifest.json" in by["d2gq"][1]
            f = by["d2final"][1]
            assert f"afterany:{by['d2asap'][0]}:{by['d2gq'][0]}" + (":7005" if queued else "") in f
            assert f.count("beyer_dai.py score") == 2 and "rule_verdict.py --labeller auto" in f
            assert ("publish_model.py --settings outputs/rule_verdict.json" in f) == (publish == "1")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
