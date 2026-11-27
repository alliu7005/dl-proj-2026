"""Tests for synthetic performances of written scores (src/synthetic.py, scripts/build_synthetic_meter.py,
scripts/fetch_pdmx.py, train_meter.py --synthetic).

    python -m tests.test_synthetic        # or: pytest tests/test_synthetic.py

Covers: a score written with music21 (a 3/4 run, an irregular bar, a 6/8 run; a tie across a bar
line) parsed by score.py and rendered: one performance per run of regular bars, its ground truth as
annotation_grid gives it (beats per bar from beat_layout, downbeats at bar starts, quarters from the
first beat, the time signature per beat), increasing beat times at a plausible tempo, the tie joined
into one note, every note on the keyboard and inside the performance; renders differ with the seed;
the training sample accepts it (targets on every beat); the build script end to end on two files
(the split by name, the report); the archive unpacking of the fetcher (a tar with a score in it).
"""
from __future__ import annotations

import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_score(path: Path, seed: int = 0):
    """3/4 x 9 bars, one 2/4 bar written in 3/4 (irregular), 6/8 x 8 bars; a tied note across bars 1-2."""
    import music21 as m21
    rng = np.random.default_rng(seed)
    s = m21.stream.Score()
    p = m21.stream.Part()
    p.insert(0, m21.instrument.Piano())
    k = 0
    for i in range(9):
        m = m21.stream.Measure(number=k + 1)
        if i == 0:
            m.timeSignature = m21.meter.TimeSignature("3/4")
        if i == 1:
            n = m21.note.Note(60, quarterLength=1.0)
            n.tie = m21.tie.Tie("stop")
            m.append(n)
            m.append(m21.note.Note(64, quarterLength=2.0))
        elif i == 0:
            m.append(m21.note.Note(55, quarterLength=2.0))
            n = m21.note.Note(60, quarterLength=1.0)
            n.tie = m21.tie.Tie("start")
            m.append(n)
        else:
            for _ in range(3):
                m.append(m21.note.Note(int(rng.integers(55, 80)), quarterLength=1.0))
        p.append(m)
        k += 1
    m = m21.stream.Measure(number=k + 1)          # only two quarters in a 3/4 bar: irregular
    m.append(m21.note.Note(60, quarterLength=2.0))
    p.append(m)
    k += 1
    for i in range(8):
        m = m21.stream.Measure(number=k + 1)
        if i == 0:
            m.timeSignature = m21.meter.TimeSignature("6/8")
        for _ in range(6):
            m.append(m21.note.Note(int(rng.integers(50, 85)), quarterLength=0.5))
        p.append(m)
        k += 1
    s.insert(0, p)
    s.write("musicxml", fp=str(path))
    return path


def test_render():
    from src.meter import TS_CLASSES, sample, targets
    from src.preprocessing.score import load_score
    from src.synthetic import render
    with tempfile.TemporaryDirectory() as d:
        measures = load_score(write_score(Path(d) / "s.musicxml"))
    perfs = render(measures, np.random.default_rng(0), "s")
    assert len(perfs) == 1, [p["performance"] for p in perfs]   # music21 pads the short bar: one run, 3/4 then 6/8
    a = perfs[0]
    assert a["gt"]["ts_names"] == ["3/4"] * 30 + ["6/8"] * 16                # 6/8: 2 dotted beats a bar
    assert a["gt"]["down"].sum() == 18 and a["gt"]["down"][0] and a["gt"]["down"][30]
    assert np.allclose(a["gt"]["q"], np.r_[np.arange(30), 30 + np.arange(16) * 1.5])
    assert (a["gt"]["ts"][:30] == TS_CLASSES.index("3/4")).all() and (a["gt"]["ts"][30:] == TS_CLASSES.index("6/8")).all()
    # a bar shorter than its time signature, or a meter outside the classes, ends a run; short runs are dropped
    from fractions import Fraction

    from src.preprocessing.score import Measure
    from src.synthetic import _runs
    ms = [Measure(Fraction(3), "3/4", 0)] * 3 + [Measure(Fraction(2), "3/4", 0)] + [Measure(Fraction(7, 2), "7/8", 0)] \
        + [Measure(Fraction(4), "4/4", 0)] * 2
    assert _runs(ms) == [(0, 3), (5, 7)]
    assert render(ms, np.random.default_rng(0)) == []
    for p in perfs:
        t, notes = p["gt"]["t"], p["notes"]
        assert np.all(np.diff(t) > 0) and p["dataset"] == "synthetic" and p["beats"] is None
        qpm = 60 * np.diff(p["gt"]["q"]) / np.diff(t)
        assert 20 < np.median(qpm) < 260
        assert notes.shape[1] == 4 and (notes[:, 0] >= 21).all() and (notes[:, 0] <= 108).all()
        assert (notes[:, 2] > notes[:, 1]).all() and (np.diff(notes[:, 1]) >= 0).all()
        assert notes[:, 1].min() > t[0] - 0.2 and (notes[:, 3] >= 1).all() and (notes[:, 3] <= 127).all()
        tg = targets(t, p["gt"])
        assert (tg["bar"] >= 0).all() and (tg["ts"] >= 0).all()
        f, tg2, bt, _ = sample(p, np.random.default_rng(1))
        assert len(f) == len(t) and (tg2["bar"] == tg["bar"]).all()
    # the tie: one C4 of 2 quarters at the start (bar 1's last beat + bar 2's first), not two notes
    c4 = a["notes"][a["notes"][:, 0] == 60]
    first = c4[np.argmin(c4[:, 1])]
    t = a["gt"]["t"]
    assert abs(first[1] - t[2]) < 0.06 and first[2] > t[3] - 0.05 * 0 - (t[4] - t[3]) * 0.6
    assert not np.any(np.abs(c4[:, 1] - t[3]) < 0.04)                    # no new onset where the tie continues
    other = render(measures, np.random.default_rng(5), "s")
    assert not np.allclose(other[0]["gt"]["t"], a["gt"]["t"])


def test_build_and_fetch():
    from scripts.build_synthetic_meter import main as build, split_of
    from scripts.fetch_pdmx import unpack
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "mxl").mkdir()
        for i in range(2):
            write_score(d / "mxl" / f"score{i}.musicxml", seed=i)
        (d / "mxl" / "junk.musicxml").write_text("not xml")
        build(["--scores", str(d / "mxl"), "--csv", str(d / "none.csv"), "--workers", "2", "--renders", "2",
               "--data", str(d / "data"), "--max", "0"])
        rep = json.loads((d / "data" / "meter" / "synthetic_report.json").read_text())
        assert rep["why"]["kept"] == 2 and sum(v for k, v in rep["why"].items() if k != "kept") == 1, rep
        assert sum(rep["performances"].values()) == 2 * 2                  # 2 scores (one run each) x 2 renders
        assert rep["time_signatures_beats"]["3/4"] > 0 and rep["time_signatures_beats"]["6/8"] > 0
        assert split_of("score0") in ("train", "val") and split_of("x") == split_of("x")
        with tarfile.open(d / "a.tar.gz", "w:gz") as t:
            t.add(d / "mxl" / "score0.musicxml", arcname="data/mxl/0/score0.musicxml")
            t.add(d / "data" / "meter" / "synthetic_report.json", arcname="data/report.json")
        assert unpack(d / "a.tar.gz", d / "out") == 1 and (d / "out" / "data__mxl__0__score0.musicxml").exists()
        assert unpack(d / "a.tar.gz", d / "none", keep={"other"}) == 0
        from scripts.fetch_pdmx import flat
        assert flat("./mxl/1/11/x.mxl") == "mxl__1__11__x.mxl"
        from scripts.fetch_pdmx import piano_scores
        (d / "PDMX.csv").write_text(
            "path,mxl,n_tracks,tracks,license_conflict,subset:deduplicated\n"
            "a,./mxl/1/1/solo.mxl,2,0-0,False,True\n"
            "b,./mxl/1/1/band.mxl,3,0-24-33,False,True\n"
            "c,./mxl/1/1/violin.mxl,2,0-40,False,True\n"
            "d,./mxl/1/1/conflict.mxl,1,0,True,True\n"
            "e,./mxl/1/1/dup.mxl,1,1,False,False\n")
        assert piano_scores(d / "PDMX.csv") == {"solo"} and piano_scores(d / "PDMX.csv", dedup=False) == {"solo", "dup"}


def test_pilot_graph():
    """pace/v2_pilot.sh with stand-ins for sbatch / squeue: it stops when an earlier round's file is
    missing; with everything there but PDMX, 6 jobs (B skipped, said so); with PDMX's scores and
    round 7's PM2S meter models, 8 jobs in order with the right dependencies, the PM2S ensemble
    built with its own tuning, nothing published, the report last waiting for every job."""
    import shutil
    import subprocess

    from tests.batch_env import batch_env
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_pilot.sh", tmp / "pace")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "squeue").write_text("#!/bin/bash\n")
        for f in (tmp / "bin").iterdir():
            f.chmod(0o755)

        def run():
            if (tmp / "jobs.txt").exists():
                (tmp / "jobs.txt").unlink()
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_pilot.sh'], cwd=tmp,
                               env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd")),
                               capture_output=True, text=True)
            jobs = ([(ln.split("\t")[0], ln.split("\t")[1]) for ln in (tmp / "jobs.txt").read_text().splitlines()]
                    if (tmp / "jobs.txt").exists() else [])
            return r, {cmd.split("-J ")[1].split()[0]: (jid, cmd) for jid, cmd in jobs}
        r, by = run()
        assert r.returncode != 0 and not by and "missing:" in r.stdout
        for f in ("data/processed/vocab.json", "data/processed/meter/train_gangqinpu.pkl",
                  "data/processed/meter/tracked_train_gangqinpu_beat_this.npz",
                  "data/processed/meter/tracked_test_gangqinpu_beat_this.npz", "configs/beat_noise_bt_gangqinpu.json",
                  "outputs/system_eval/manifest.json", "outputs/system_eval_gangqinpu/manifest.json", "bd/MIDI2ScoreTF.ckpt"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        r, by = run()
        assert r.returncode == 0, r.stdout + r.stderr
        assert list(by) == ["q1tests", "q1prep", "q1pm2s", "q1asap", "q1gq", "q1final"], list(by)
        assert "B skipped" in r.stdout and "PM2S ensemble skipped" in r.stdout
        for m in ("meter_v4_gq_pm", "meter_v4_gq_pm_s1", "meter_v4_gq_pm_g30"):
            (tmp / "outputs" / "runs" / m).mkdir(parents=True)
            (tmp / "outputs" / "runs" / m / "best.pt").write_text("x")
        (tmp / "data" / "raw" / "pdmx" / "mxl").mkdir(parents=True)
        (tmp / "data" / "raw" / "pdmx" / "mxl" / "a.mxl").write_text("x")
        r, by = run()
        assert r.returncode == 0, r.stdout + r.stderr
        assert list(by) == ["q1tests", "q1prep", "q1synth", "q1short", "q1pm2s", "q1asap", "q1gq", "q1final"], list(by)
        tests, prep, synth, short, pm2s, asap, gq, final = (by[k] for k in by)
        assert "tests.test_synthetic" in tests[1] and "--dependency" not in tests[1]
        assert f"afterok:{tests[0]}" in prep[1] and "ensemble meter_v4_gq_tta tta meter_v4_gq" in prep[1]
        assert "ENS_KIND=pm2s PUBLISH=0 bash pace/v2_jobs.sh ensemble meter_v4_gq_pm_ens3 none meter_v4_gq_pm meter_v4_gq_pm_s1 meter_v4_gq_pm_g30" in prep[1]
        assert f"afterok:{tests[0]}" in synth[1] and "build_synthetic_meter.py --scores data/raw/pdmx/mxl --max 3000" in synth[1]
        assert f"afterok:{synth[0]}" in short[1] and "meters meter_v4_gq_p12k meter_v4_gq_syn_p12k" in short[1]
        assert "--name pm2s_pilot3e4 --lr 3e-4" in pm2s[1] and "--clip 1.0" in pm2s[1] and "--steps 16000" in pm2s[1]
        for j in (asap, gq):
            assert f"afterany:{prep[0]}" in j[1] and "nl_gq+meter_v4_gq_tta:tidy,d95,missing" in j[1]
            assert "nl_base_gn+meter_v4_gq_pm_ens3:tidy,d95,pm,rpm,missing" in j[1]
        assert "--manifest outputs/system_eval_gangqinpu/manifest.json" in gq[1] and "sf_base_reg_cab_c128:meter:sb" in gq[1]
        assert f"afterany:{asap[0]}:{gq[0]}:{prep[0]}:{short[0]}:{pm2s[0]}" in final[1] and "pilot_report.py" in final[1]
        for _, cmd in (prep, short, asap, gq):
            assert "PUBLISH=0" in cmd


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
