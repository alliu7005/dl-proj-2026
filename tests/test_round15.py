"""Round 15: the PM2S meters' recipes, the systems' names (seq2seq with PM2S's beats, v2 on another
seq2seq run's grid), the jobs' spec parsing, the report and pace/v2_round15.sh.

    python -m tests.test_round15
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _fn(fn: str, *args: str) -> str:
    r = subprocess.run(["bash", "-c", f'eval "$(sed -n "/^{fn}()/,/^}}/p" pace/v2_jobs.sh)"; {fn} "$@"', "_", *args],
                       cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip()


def test_recipes():
    base = _fn("meter_flags", "meter_v4_gq_pmhs")
    assert _fn("meter_flags", "meter_v4_gq_pmhs_ptr") == "--init outputs/runs/meter_pt/best.pt " + base
    assert _fn("meter_flags", "meter_v4_gq_pmhs_ptr_s1") == "--init outputs/runs/meter_pt/best.pt " + base + " --seed 1"
    for m in ("meter_v4_gq_pmhs_ptr", "meter_v4_gq_pmhs_ptr_s2"):
        assert _fn("meter_eval_flags", m) == _fn("meter_eval_flags", "meter_v4_gq_pmhs")
    assert "--grid seq2seq --grid-run sf_c128_gq30" in _fn("v2_flags", "tidy,s2gq")
    text = (ROOT / "pace" / "v2_jobs.sh").read_text()
    runs = text.split('PM2S_RUNS=${PM2S_RUNS-"')[1].split('"')[0].split()
    assert {"meter_v4_gq_pmhs_ptr_s2", "meter_v4_gq_pmhs_ptr_ens3", "meter_v4_gq_pmhs_ens4"} <= set(runs)


def test_names():
    from scripts.system_scores import system_name
    base = dict(delete_above=0.95, tidy=True)
    assert system_name(None, "meter", "synth", "nl_gq+meter_v4_gq", grid="seq2seq", **base) \
        == "v2_nl_gq_meter_v4_gq_d95_tidy_s2g"
    assert system_name(None, "meter", "synth", "nl_gq+meter_v4_gq", grid="seq2seq", grid_run="sf_c128_gq30", **base) \
        == "v2_nl_gq_meter_v4_gq_d95_tidy_s2g_sf_c128_gq30"
    old = {k: os.environ.pop(k, None) for k in ("PM2S_WEIGHTS", "PM2S_TAG")}
    try:
        assert system_name("sf_c128_gq30", "meter", "none", None, seq_beats="pm2s") == "sf_c128_gq30_meter_spm"
        os.environ.update(PM2S_WEIGHTS="w.pt", PM2S_TAG="hos")
        assert system_name("sf_c128_gq30", "meter", "none", None, seq_beats="pm2s") == "sf_c128_gq30_meter_spm_hos"
        assert system_name("sf_c128_gq30", "meter", "synth", None) == "sf_c128_gq30_meter_sb"
        from scripts.round15_report import cand
        assert system_name(None, "meter", "synth", "nl_gq+meter_v4_gq_pmhs", "pm2s", 0.95, True, "pm2s",
                           fix_jumps=True) == cand("meter_v4_gq_pmhs")
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_seq_beats_only_for_seq2seq():
    r = subprocess.run([sys.executable, "scripts/system_scores.py", "--v2", "a+b", "--seq-beats", "pm2s"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "--seq-beats is for seq2seq" in r.stderr


def test_report_runs_empty():
    from scripts.round15_report import main
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        assert main(["--runs-dir", str(tmp), "--asap", str(tmp / "x.json"), "--gq", str(tmp / "y.json"),
                     "--out", str(tmp / "out")]) == 0
        assert "Round 15" in (tmp / "out" / "report.md").read_text()


def test_round15_batch():
    from tests.batch_env import batch_env
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_round15.sh", tmp / "pace")
        for f in ("outputs/runs/pm2s_hos/best.pt", "outputs/runs/meter_v4_gq_pmhs/eval_meter.json",
                  "configs/beat_noise_pm2s_hos.json", "outputs/runs/meter_pt/best.pt", "outputs/runs/sf_c128_gq30/best.pt",
                  "outputs/runs/nl_gq/config.json", "data/processed/vocab.json", "outputs/system_eval_gangqinpu/manifest.json",
                  "outputs/system_eval/manifest.json", "bd/MIDI2ScoreTF.ckpt"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "sbatch").chmod(0o755)
        r = subprocess.run(["bash", "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_round15.sh'], cwd=tmp,
                           env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd")),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        by = {ln.split("\t")[1].split("-J ")[1].split()[0]: ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()}
        assert list(by) == ["r5tests", "c5asap", "c5gq", "m5train", "m5ens", "m5asap", "m5gq", "q5asap", "q5gq",
                            "g5asap", "g5gq", "r5final"]
        t0 = by["r5tests"][0]
        for j in ("c5asap", "c5gq", "m5train", "q5asap", "q5gq", "g5asap", "g5gq"):   # side by side after the tests
            assert f"afterok:{t0}" in by[j][1], j
        for j in ("c5asap", "c5gq", "m5train", "m5ens", "m5asap", "m5gq", "q5asap", "q5gq"):   # PM2S's retrained weights
            assert "PM2S_TAG=hos" in by[j][1] and "pm2s_hos/best.pt" in by[j][1], j
        assert "nl_gq+meter_v4_gq_pmhs:tidy,d95,jf,pm,rpm,missing" in by["c5gq"][1]
        assert "d80,jf,pm,rpm" in by["c5asap"][1] and "--conditions clean" in by["c5asap"][1]
        assert "ENS_NOISE=configs/beat_noise_pm2s_hos.json" in by["m5ens"][1] and "meter_v4_gq_pmhs_ens4 none meter_v4_gq_pmhs" in by["m5ens"][1]
        assert "nl_gq+meter_v4_gq_pmhs_ens4:tidy,d95,jf,pm,rpm,missing" in by["m5gq"][1]
        assert "sf_c128_gq30:meter:spm" in by["q5asap"][1] and "s2gq" in by["g5gq"][1]
        ends = ":".join(by[k][0] for k in ("c5asap", "c5gq", "m5asap", "m5gq", "q5asap", "q5gq", "g5asap", "g5gq"))
        assert f"afterany:{ends}" in by["r5final"][1] and "round15_report.py" in by["r5final"][1]


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
