"""The rhythm prior pilot: the beats' families decoded as a sequence (src/beat_decode.py viterbi,
decode_subs with ``prior``), the prior counted from labels (scripts/fit_rhythm_prior.py), the tuner's
settings with it, the report and pace/v2_rhythm.sh.

    python -m tests.test_rhythm_prior
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.beat_decode import OTHER, STRAIGHT, TRIPLET, BeatDecode, Families, decode_item, decode_subs, viterbi  # noqa: E402
from tests.test_beat_decode import Space, logp_for  # noqa: E402

# straight beats are common and stay straight; triplets are rare and come in runs
TABLE = {"log_init": np.log([0.9, 0.08, 0.02]).tolist(),
         "log_trans": np.log([[0.95, 0.04, 0.01], [0.3, 0.68, 0.02], [0.5, 0.1, 0.4]]).tolist()}


def test_viterbi():
    emit = np.array([[0.0, -5.0, -9.0], [-0.5, 0.0, -9.0], [0.0, -5.0, -9.0]])
    assert viterbi(emit, np.zeros(3), np.zeros((3, 3)), 1.0).tolist() == [0, 1, 0]       # no prior: each beat's best
    assert viterbi(emit, np.array(TABLE["log_init"]), np.array(TABLE["log_trans"]), 1.0).tolist() == [0, 0, 0]
    emit2 = np.array([[0.0, -2.0, -9.0], [-3.0, 0.0, -9.0], [-3.0, 0.0, -9.0], [-3.0, 0.0, -9.0]])   # a clear run
    assert viterbi(emit2, np.array(TABLE["log_init"]), np.array(TABLE["log_trans"]), 1.0).tolist() == [0, 1, 1, 1]


def _beats(prefs_per_beat):
    """Notes of consecutive beats: each beat's notes' (preferred class, alternative, margin)."""
    prefs, groups = [], []
    for b, ps in enumerate(prefs_per_beat):
        prefs += ps
        groups += [b] * len(ps)
    return logp_for(8, prefs), np.array(groups)


def test_lone_triplet_flipped():
    f = Families(Space())
    # classes: 1: 1/2, 2: 1/4, 3: 3/4, 4: 1/3, 5: 2/3; the middle beat leans triplet (2/3 over 3/4 by 0.6)
    logp, g = _beats([[(2, None, 0)], [(5, 3, 0.6)], [(2, None, 0)], [(3, None, 0)]])
    x = np.full(len(g), np.nan)
    base, fam0 = decode_subs(logp, x, g, f, BeatDecode(0.0, True, False, 0.0))
    assert base.tolist() == [2, 5, 2, 3] and fam0.tolist() == [STRAIGHT, TRIPLET, STRAIGHT, STRAIGHT]
    off, _ = decode_subs(logp, x, g, f, BeatDecode(0.0, True, False, 0.0, prior=0.0, table=TABLE))
    assert off.tolist() == base.tolist()                                    # weight 0: as before
    pick, fam = decode_subs(logp, x, g, f, BeatDecode(0.0, True, False, 0.0, prior=1.0, table=TABLE))
    assert pick.tolist() == [2, 3, 2, 3] and fam.tolist() == [STRAIGHT] * 4  # the lone triplet goes straight
    # a sure triplet run survives the prior
    logp, g = _beats([[(2, None, 0)], [(4, 2, 4.0)], [(5, 3, 4.0)], [(4, 2, 4.0)]])
    pick, fam = decode_subs(logp, np.full(len(g), np.nan), g, f, BeatDecode(0.0, True, False, 0.0, prior=1.0, table=TABLE))
    assert fam.tolist() == [STRAIGHT, TRIPLET, TRIPLET, TRIPLET]


def test_decode_item_orders_beats_in_time():
    f = Families(Space())
    # notes listed out of time order: beat ids follow the notes, the chain follows bar and quarter
    sub = logp_for(8, [(2, None, 0), (5, 3, 0.6), (3, None, 0)])
    pick = {"delta": np.array([1, 1, 1]), "pos": np.array([2, 1, 0]), "sub": sub.argmax(1),
            "dur": np.array([0, 0, 0]), "staff": np.array([0, 0, 0])}
    s, _ = decode_item(sub, np.zeros((3, 4)), pick, np.array([3, 3, 3]), np.array([2.25, 1.66, 0.75]),
                       np.array([True] * 3), f, BeatDecode(0.0, True, False, 0.0, prior=1.0, table=TABLE))
    assert s.tolist() == [2, 3, 3]


def test_params_roundtrip():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "beat_decode.json"
        p.write_text(json.dumps({"best": BeatDecode(4.0, True, False, 0.1, prior=2.0, table=TABLE).to_dict()}))
        q = BeatDecode.load(p)
        assert q.prior == 2.0 and q.sequential and q.table["log_trans"] == TABLE["log_trans"]
        p.write_text(json.dumps({"best": {"weight": 4.0, "consistent": False, "by_staff": False, "anchor": 0.0}}))
        q = BeatDecode.load(p)                                               # an older file: no prior
        assert q.prior == 0.0 and q.table is None and not q.sequential


def test_fit_counts():
    from scripts.fit_rhythm_prior import beat_families, count, table
    fam_of_class = np.array([STRAIGHT, STRAIGHT, STRAIGHT, STRAIGHT, TRIPLET, TRIPLET, TRIPLET, OTHER])
    # labels: [_, delta, pos, sub]; two bars; beat (bar 3, q 1) has a 2/3, (3, 2) a 1/5
    lab = np.array([[0, 1, 0, 0], [0, 1, 1, 5], [0, 1, 1, 1], [0, 1, 2, 7], [0, 2, 0, 2], [0, 1, 3, -1]])
    seq = beat_families(lab, np.array([3, 3, 3, 3, 3, 3]), np.array([True] * 6), fam_of_class)
    assert seq == [STRAIGHT, TRIPLET, OTHER, STRAIGHT]                       # bar 3 q 0, 1, 2; bar 4 q 0
    init, trans = count([seq, [STRAIGHT, STRAIGHT]])
    assert init.tolist() == [2, 0, 0] and trans[0, 1] == 1 and trans[1, 2] == 1 and trans[2, 0] == 1 and trans[0, 0] == 1
    t = table(init, trans)
    assert np.allclose(np.exp(t["log_trans"]).sum(1), 1) and np.isclose(np.exp(t["log_init"]).sum(), 1)


def test_tuner_settings():
    from scripts.tune_beat_decode import settings
    plain = settings([0.0, 4.0])
    withp = settings([0.0, 4.0], [1.0, 2.0], TABLE)
    assert len(withp) == len(plain) + 2 * 2 * 2 * 2                          # per beat-consistent setting, two priors
    assert all(p.consistent for p in withp if p.prior > 0) and not any(p.prior for p in plain)


def test_report_runs_empty():
    from scripts.rhythm_report import main
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        assert main(["--runs-dir", str(tmp), "--prior", str(tmp / "p.json"), "--asap", str(tmp / "a.json"),
                     "--gq", str(tmp / "g.json"), "--out", str(tmp / "out"), "--rare-pieces", "0"]) == 0
        assert "rhythm prior" in (tmp / "out" / "report.md").read_text()


def test_rhythm_batch():
    from tests.batch_env import batch_env
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_rhythm.sh", tmp / "pace")
        for f in ("outputs/runs/nl_gq/best.pt", "outputs/runs/nl_gq/status.json", "outputs/runs/pm2s_hos/best.pt",
                  "outputs/runs/meter_v4_gq_pmhs/eval_meter.json", "data/processed/train/notelabels.npz",
                  "data/processed/val/notelabels.npz", "outputs/system_eval_gangqinpu/manifest.json",
                  "outputs/system_eval/manifest.json", "bd/MIDI2ScoreTF.ckpt"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "squeue").write_text('#!/bin/bash\necho "970 x5final"\necho "971 other"\n')
        for f in (tmp / "bin").iterdir():
            f.chmod(0o755)
        r = subprocess.run(["bash", "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_rhythm.sh'], cwd=tmp,
                           env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd")),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        by = {ln.split("\t")[1].split("-J ")[1].split()[0]: ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()}
        assert list(by) == ["rptests", "rpfit", "rptune", "rpasap", "rpgq", "rpfinal"]
        assert f"afterok:{by['rpfit'][0]}" in by["rptune"][1]
        assert "--run nl_gq_bd0" in by["rptune"][1] and "--run nl_gq_rp --prior-file outputs/rhythm_prior/prior.json" in by["rptune"][1]
        assert "nl_gq_rp+meter_v4_gq_pmhs:tidy,d95,jf,bd,pm,rpm" in by["rpgq"][1] and "PM2S_TAG=hos" in by["rpasap"][1]
        assert f"afterany:{by['rpasap'][0]}:{by['rpgq'][0]}:970" in by["rpfinal"][1] and "971" not in by["rpfinal"][1]


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
