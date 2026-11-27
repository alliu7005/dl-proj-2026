"""Tests for the tracker's level per stretch (src/beats.py level_runs, compare, fit_noise(levels=True),
BeatNoise level profiles, level_replay), the meter's kind passed to the noise (src/meter.py
noisy_beats), compound meters weighted in meter training (scripts/train_meter.py weights), the fits'
--levels, scripts/meter_ts_report.py, and pace/v2_compound.sh.

    python -m tests.test_beat_levels        # or: pytest tests/test_beat_levels.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.beats import (BeatNoise, compare, fit_noise, is_compound, level_replay, level_runs,  # noqa: E402
                       piece_compound)


def switching(n=100, a=30, b=60):
    """True beats every 0.5 s (6/8: two a bar); the tracker on them up to beat a, at two-thirds of
    them (three to a bar) up to beat b, then on the eighths (three to a beat)."""
    gt = 0.5 * np.arange(n)
    tr = np.concatenate([gt[:a], 0.5 * np.arange(a, b, 2 / 3), 0.5 * np.arange(b, n - 1 + 1e-9, 1 / 3)])
    return gt, np.arange(n) % 2 == 0, tr


def test_kinds():
    assert is_compound("6/8") and is_compound("12/8") and is_compound("9/8") and is_compound("6/4")
    assert not is_compound("3/4") and not is_compound("4/4") and not is_compound("3/8") and not is_compound("x")
    assert piece_compound({"ts_names": ["6/8"] * 8 + ["3/4"] * 2}) is True
    assert piece_compound({"ts_names": ["3/4"] * 8}) is False and piece_compound({}) is None


def test_level_runs():
    gt, _, tr = switching()
    runs = level_runs(gt, tr)
    assert [round(r[0], 3) for r in runs] == [1.0, 0.667, 0.333], runs
    assert abs(runs[0][2] - 30) < 1 and abs(runs[1][2] - 60) < 1 and abs(runs[-1][2] - 99) < 0.5
    # a missed beat or a spurious one is a slip, not a stretch
    one = np.delete(gt, 40)
    assert [r[0] for r in level_runs(gt, one)] == [1.0]
    assert [r[0] for r in level_runs(gt, np.sort(np.r_[gt, 20.25]))] == [1.0]
    assert level_runs(gt, gt[:4]) == []


def test_compare_and_fit():
    gt, down, tr = switching()
    c = compare(gt, down, tr, np.zeros(len(tr), bool), compound=True)
    assert c["compound"] is True and [round(lv, 3) for lv, _ in c["levels"]] == [1.0, 0.667, 0.333]
    assert abs(sum(f for _, f in c["levels"]) - 1) < 1e-6 and abs(c["levels"][0][1] - 30 / 99) < 0.02
    assert sum(v[1] for v in c["level_down"].values()) > 0
    normal = compare(gt, down, gt, down, compound=False)
    cs = [c, normal]
    old, new = fit_noise(cs), fit_noise(cs, levels=True)
    assert old.level_profiles is None and old.p_levels == 0.0          # the fit as before
    assert new.p_levels == 0.5 and len(new.level_profiles) == 1
    q = new.level_profiles[0]
    assert q["compound"] is True and q["bar"] == 2 and len(q["segments"]) == 3
    assert {k: getattr(old, k) for k in ("p_double", "p_half", "miss", "extra")} == \
        {k: getattr(new, k) for k in ("p_double", "p_half", "miss", "extra")}
    d = new.to_dict()
    assert BeatNoise.from_dict(json.loads(json.dumps(d))).level_profiles == new.level_profiles


def test_replay():
    gt = 0.5 * np.arange(61)
    cls = np.zeros(61, np.int64)
    t, c, wrong = level_replay(gt, cls, [[1.0, 0.5], [2 / 3, 0.5]])
    runs = level_runs(gt, t)
    assert [round(r[0], 3) for r in runs] == [1.0, 0.667] and abs(runs[0][2] - 30) < 1
    assert not wrong[t < 14.9].any() and wrong[t > 15.5].all()
    on = np.isclose(t[:, None], gt[None, :]).any(1)
    assert (c[~on] == 3).all()                                            # between true beats: EXTRA
    # BeatNoise: a compound piece gets a compound profile, a simple one a simple profile
    base = dict(miss=0.0, extra=0.0, jitter=0.0, offset=0.0, p_offbeat=0.0, profiles=None)
    prof = lambda segs, comp: {"bar": 2, "compound": comp, "segments": segs, "down_hit": 0.9,  # noqa: E731
                               "down_half": 0.1, "down_other": 0.1, "down_extra": 0.1}
    nz = BeatNoise(**base, p_levels=1.0, level_profiles=[prof([[2 / 3, 1.0]], True), prof([[2.0, 1.0]], False)])
    down = np.arange(61) % 2 == 0
    for comp, want in ((True, 0.667), (False, 2.0)):
        for s in range(5):
            t, _ = nz(gt, down, np.random.default_rng(s), compound=comp)
            assert [round(r[0], 3) for r in level_runs(gt, t)] == [want], (comp, s)
    # p_levels 0: no level errors, and no octave errors either (the profiles replace them)
    calm = BeatNoise(**base, p_double=1.0, p_levels=0.0, level_profiles=[prof([[2.0, 1.0]], None)])
    t, _ = calm(gt, down, np.random.default_rng(0))
    assert np.allclose(t, gt)


def test_old_configs_unchanged():
    """A config without level profiles draws exactly as before: the meter's kind is ignored, and
    noisy_beats calls any noise without it."""
    from src.meter import noisy_beats
    nz = BeatNoise.from_json(ROOT / "configs" / "beat_noise.json")
    assert not nz.level_profiles
    gt = {"t": 0.5 * np.arange(80), "down": np.arange(80) % 4 == 0, "ts_names": ["6/8"] * 80}
    a = nz(gt["t"], gt["down"], np.random.default_rng(3))
    b = noisy_beats(nz, gt, np.random.default_rng(3))
    c = nz(gt["t"], gt["down"], np.random.default_rng(3), compound=True)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and all(np.array_equal(x, y) for x, y in zip(a, c))
    seen = []
    assert noisy_beats(lambda t, d, rng: seen.append(1) or (t, d), gt, np.random.default_rng(0))[0] is gt["t"]
    assert seen == [1]


def test_compound_weight():
    from scripts.train_meter import mixed_weights, weights
    perfs = [{"gt": {"t": np.zeros(100), "ts_names": [ts] * 100}, "piece": f"p{k}"}
             for k, ts in enumerate(["6/8", "3/4", "4/4", "12/8"])]
    w1, w3 = weights(perfs, "piece"), weights(perfs, "piece", 3.0)
    assert np.allclose(w1, 0.25)
    assert np.isclose(w3[0] / w3[1], 3.0) and np.isclose(w3[3] / w3[2], 3.0) and np.isclose(w3.sum(), 1.0)
    m = mixed_weights({"a": perfs[:2], "gangqinpu": perfs[2:]}, "piece", {"gangqinpu": 0.5}, 3.0)
    assert np.isclose(m[:2].sum(), 0.5) and np.isclose(m[2:].sum(), 0.5)    # the sources' shares stay
    assert np.isclose(m[3] / m[2], 3.0)
    t = (ROOT / "scripts" / "train_meter.py").read_text(encoding="utf-8")
    assert "--compound-weight" in t and "args.compound_weight" in t


def test_fits_and_recipes():
    for f, flag in (("scripts/fit_beat_noise.py", "--levels"), ("scripts/fit_pm2s_noise.py", "--levels")):
        x = (ROOT / f).read_text(encoding="utf-8")
        assert flag in x and "levels=args.levels" in x and "compound=" in x, f
    j = (ROOT / "pace" / "v2_jobs.sh").read_text()
    for m, want in (("meter_v5_lv", "configs/beat_noise_bt_gangqinpu_levels.json"), ("meter_v5_cw", "--compound-weight 3"),
                    ("meter_v5_lvcw", "--compound-weight 3")):
        line = next(ln for ln in j.splitlines() if ln.strip().startswith(f"{m})"))
        assert want in line, m
    assert "--beat-noise configs/beat_noise_levels.json" in next(
        ln for ln in j.splitlines() if ln.strip().startswith("meter_v5_lvcw)"))
    assert "meter_v5_lv|meter_v5_cw|meter_v5_lvcw) echo \"--tune --tempo-prior\"" in j
    # on this machine's ASAP data, if built: the compound kinds come from the meter data
    if (ROOT / "data/processed/meter/train.pkl").exists() and (ROOT / "data/processed/train_transcribed/beats_tracked.npz").exists():
        from scripts import fit_beat_noise as F
        kinds = F.compound_of()
        cs = F.comparisons("train_transcribed", kinds)
        assert cs and any(c["compound"] is True for c in cs) and all("levels" in c for c in cs if c.get("octave") != "none")
        n = fit_noise(cs, levels=True)
        assert n.level_profiles and 0 < n.p_levels <= 1


def test_report():
    from scripts.meter_ts_report import main
    with tempfile.TemporaryDirectory() as t:
        runs = Path(t) / "runs"
        for run, right in (("a", 0.5), ("b", 1.0)):
            (runs / run).mkdir(parents=True)
            sc = [{"true_ts": "6/8", "decoded_ts": "6/8" if k < 10 * right else "3/4", "grid_level_ok": 1.0,
                   "grid_beat_f": 0.8, "grid_downbeat_f": 0.7} for k in range(10)]
            sc += [{"true_ts": "4/4", "decoded_ts": "4/4", "grid_level_ok": 1.0, "grid_beat_f": 0.9, "grid_downbeat_f": 0.9}]
            (runs / run / "meter_scores_gangqinpu_real.json").write_text(json.dumps(sc))
        out = Path(t) / "ts.md"
        assert main(["--runs", "a,b", "--runs-dir", str(runs), "--out", str(out)]) == 0
        md = out.read_text()
        assert "| a | 6/8 | 10 | 0.500 | 6/8 5, 3/4 5 |" in md and "| b | 6/8 | 10 | 1.000 |" in md
        assert "| a | compound | 10 | 0.500 |" in md
        js = json.loads(out.with_suffix(".json").read_text())
        assert js["gangqinpu_real"]["b"]["6/8"]["right_ts"] == 1.0


def test_batch():
    """pace/v2_compound.sh: 6 jobs; the training after the fits; the resume, the systems and the
    scoring in order; the scoring after the earlier batches' scoring still queued."""
    import shutil
    import subprocess

    from tests.batch_env import batch_env
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    for queued in (True, False):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "pace").mkdir()
            shutil.copy(ROOT / "pace" / "v2_compound.sh", tmp / "pace")
            for f in ("outputs/runs/nl_gq/best.pt", "outputs/runs/meter_v4_gq/eval_meter.json", "data/processed/meter/train.pkl",
                      "data/processed/meter/train_gangqinpu.pkl", "outputs/system_eval/manifest.json",
                      "outputs/system_eval_gangqinpu/manifest.json", "bd/MIDI2ScoreTF.ckpt"):
                (tmp / f).parent.mkdir(parents=True, exist_ok=True)
                (tmp / f).write_text("x")
            (tmp / "bin").mkdir()
            (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                                'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
            q = ('#!/bin/bash\nfor a in "$@"; do case "$prev" in -n) case "$a" in dcfinal) echo 7005;; '
                 'ptfinal) echo 7011;; esac;; esac; prev=$a; done\n') if queued else "#!/bin/bash\nexit 0\n"
            (tmp / "bin" / "squeue").write_text(q)
            for b in ("sbatch", "squeue"):
                (tmp / "bin" / b).chmod(0o755)
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_compound.sh'], cwd=tmp,
                               env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd")),
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            by = {cmd.split("-J ")[1].split()[0]: (jid, cmd)
                  for jid, cmd in (ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines())}
            assert list(by) == ["cmfit", "cmtrain", "cmtrain2", "cmasap", "cmgq", "cmfinal"]
            f = by["cmfit"][1]
            assert "tests.test_beat_levels" in f and "fit_beat_noise.py --levels" in f
            assert "fit_pm2s_noise.py --from-stored beat_this --levels" in f
            assert f"--dependency=afterok:{by['cmfit'][0]}" in by["cmtrain"][1]
            assert "meters meter_v5_lv meter_v5_cw meter_v5_lvcw" in by["cmtrain"][1]
            assert f"--dependency=afterany:{by['cmtrain'][0]}" in by["cmtrain2"][1]
            assert "trackers meter_v5_lv meter_v5_cw meter_v5_lvcw" in by["cmtrain2"][1]
            for j in ("cmasap", "cmgq"):
                s = by[j][1]
                assert f"--dependency=afterany:{by['cmtrain2'][0]}" in s
                for m in ("meter_v5_lv", "meter_v5_cw", "meter_v5_lvcw"):
                    assert f" nl_gq+{m}:tidy,d95,missing" in s, (j, m)
            assert "--manifest outputs/system_eval_gangqinpu/manifest.json" in by["cmgq"][1]
            fin = by["cmfinal"][1]
            assert f"afterany:{by['cmasap'][0]}:{by['cmgq'][0]}" + (":7005:7011" if queued else "") in fin
            assert fin.count("beyer_dai.py score") == 2 and "meter_ts_report.py --runs meter_v4_gq,meter_v5_lv,meter_v5_cw,meter_v5_lvcw" in fin
    c = (ROOT / "scripts" / "compare_systems.py").read_text(encoding="utf-8")
    for name in ("_meter_v5_lv_d95_tidy\"", "_meter_v5_cw_d95_tidy\"", "_meter_v5_lvcw_d95_tidy\""):
        assert name in c, name


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
