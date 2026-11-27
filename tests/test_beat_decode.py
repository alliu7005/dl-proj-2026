"""Tests for the tenth round's model-side work: the labeller's beat-level decoding (src/beat_decode.py,
scripts/tune_beat_decode.py), the contradicted-label audit and mask (src/notelabels.py contradicted,
the dataset's mask_contradicted, scripts/audit_labels.py, train_labeller --mask-contradicted), the meter
decoder's local tempo term (src/meter.py local_tempo_logprior, eval_meter --local-tempo), the
switches, publishing a tuned beat_decode.json, and pace/v2_decode.sh.

    python -m tests.test_beat_decode        # or: pytest tests/test_beat_decode.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.beat_decode import (STRAIGHT, TRIPLET, BeatDecode, Families, decode_item, decode_subs,  # noqa: E402
                             family_of, straight_lengths)

V = ROOT / "data" / "processed" / "vocab.json"


class Space:
    """A small label space: splits 1/2, 1/4, 3/4, 1/3, 2/3, 1/6, 1/5; lengths 1/4, 1/3, 1/2, 1."""
    subs = [Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 3), Fraction(2, 3), Fraction(1, 6), Fraction(1, 5)]
    durs = [Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(1)]


def logp_for(n_sub, prefs):
    """Log-probabilities with each note's preferred split (class index) ``margin`` above the rest."""
    out = np.full((len(prefs), n_sub), -10.0)
    for i, (k, alt, margin) in enumerate(prefs):
        out[i, k] = 0.0
        if alt is not None:
            out[i, alt] = -margin
    return out


def test_families():
    f = Families(Space())
    # class 0 = on the quarter; 1: 1/2, 2: 1/4, 3: 3/4, 4: 1/3, 5: 2/3, 6: 1/6, 7: 1/5
    assert f.masks[STRAIGHT].tolist() == [True, True, True, True, False, False, False, False]
    assert f.masks[TRIPLET].tolist() == [True, True, False, False, True, True, True, False]   # 1/2 in both
    assert f.masks[2].tolist() == [True, False, False, False, False, False, False, True]
    assert family_of(Fraction(3, 8)) == STRAIGHT and family_of(Fraction(5, 6)) == TRIPLET and family_of(Fraction(2, 5)) == 2
    assert f.straight_dur.tolist() == [True, False, True, True] and f.triplet_dur.tolist() == [False, True, False, False]


def test_decode_subs():
    f = Families(Space())
    # one beat as in the song: the model says 1/2 (sure), 2/3 (sure-ish over 3/4); played at 0.5, 0.74
    logp = logp_for(8, [(1, None, 0), (5, 3, 2.0)])
    x = np.array([0.5, 0.74])
    g = np.array([0, 0])
    own, fam = decode_subs(logp, x, g, f, BeatDecode(weight=0.0, consistent=False, anchor=0.0))
    assert own.tolist() == [1, 5] and fam.tolist() == [-1, -1]          # weight 0, no constraint: the argmax
    pick, _ = decode_subs(logp, x, g, f, BeatDecode(weight=500.0, consistent=False, anchor=0.0))
    assert pick.tolist() == [1, 3]                                       # the timing outweighs a margin of 2
    pick, _ = decode_subs(logp, x, g, f, BeatDecode(weight=10.0, consistent=False, anchor=0.0))
    assert pick.tolist() == [1, 5]                                       # ... a small weight does not
    # a mixed beat (1/4 and 2/3 written): one family per beat, the one the notes score higher together
    logp = logp_for(8, [(2, 4, 1.0), (5, 3, 0.5), (0, None, 0)])
    pick, fam = decode_subs(logp, np.full(3, np.nan), np.array([0, 0, 0]), f, BeatDecode(0.0, True, False, 0.0))
    assert pick.tolist() == [2, 3, 0] and fam.tolist() == [STRAIGHT] * 3
    # a beat played early: its on-the-quarter note at -0.06 shifts the others (anchor)
    logp = logp_for(8, [(0, None, 0), (4, 2, 0.3)])
    x = np.array([-0.06, 0.19])                                          # 0.25 of the beat as played: 1/4
    p0, _ = decode_subs(logp, x, np.array([0, 0]), f, BeatDecode(60.0, False, False, 0.0))
    p1, _ = decode_subs(logp, x, np.array([0, 0]), f, BeatDecode(60.0, False, False, 0.1))
    assert p0.tolist() == [0, 2] and p1.tolist() == [0, 2]
    x = np.array([-0.06, 0.24])                                          # 0.30 as played: nearer 1/3 with the anchor
    p0, _ = decode_subs(logp, x, np.array([0, 0]), f, BeatDecode(60.0, False, False, 0.0))
    p1, _ = decode_subs(logp, x, np.array([0, 0]), f, BeatDecode(60.0, False, False, 0.1))
    assert p0.tolist() == [0, 2] and p1.tolist() == [0, 4]
    # notes alone (group -1) are decoded alone; lengths: a triplet length in a straight beat
    assert straight_lengths(np.array([[0.0, 1.0, 0.5, -1.0]]), np.array([1]), np.array([STRAIGHT]), f).tolist() == [2]
    assert straight_lengths(np.array([[0.0, 1.0, 0.5, -1.0]]), np.array([1]), np.array([TRIPLET]), f).tolist() == [1]


def test_decode_item():
    f = Families(Space())
    sub = logp_for(8, [(1, None, 0), (5, 3, 0.5), (5, 3, 0.5)])
    dur = np.zeros((3, 4))
    dur[:, 1] = 1.0                                                      # likeliest length 1/3 for all
    pick = {"delta": np.array([1, 1, 1]), "pos": np.array([2, 2, 3]), "sub": sub.argmax(1),
            "dur": np.array([1, 1, 1]), "staff": np.array([0, 0, 0])}
    pos_est = np.array([2.5, 2.75, 3.66])                                # the third note: another beat, a triplet
    s, d = decode_item(sub, dur, pick, np.array([3, 3, 3]), pos_est, np.array([True, True, True]), f,
                       BeatDecode(weight=200.0, consistent=True, anchor=0.0))
    assert s.tolist() == [1, 3, 5]
    assert d.tolist() == [0, 0, 1]          # the straight beat's 1/3 lengths -> 1/4 (their best straight); the triplet's stays
    # a note not rendered keeps its argmax and joins no beat; a bar correction drops the timing term
    s, _ = decode_item(sub, dur, dict(pick, delta=np.array([1, 0, 1])), np.array([3, 3, 3]), pos_est,
                       np.array([True, True, False]), f, BeatDecode(weight=20.0, consistent=False, anchor=0.0))
    assert s.tolist() == [1, 5, 5]


def test_params_file():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        assert BeatDecode.load(d) is None
        (d / "beat_decode.json").write_text(json.dumps({"best": {"weight": 2.0, "consistent": False, "by_staff": False,
                                                                 "anchor": 0.0}, "table": []}))
        p = BeatDecode.load(d)
        assert p == BeatDecode(2.0, False, False, 0.0) and p.active
        assert not BeatDecode(0.0, False).active


def test_contradicted():
    from src.notelabels import STRAIGHT_PLAYED_TRIPLET, TRIPLET_PLAYED_STRAIGHT, contradicted
    def row(bar, num, den):
        return [1, bar, num, den, 1, 4, 0, 0]
    # bar 0, quarter 1: written 1, 1 1/3, 1 2/3 (triplets), played 1.0, 1.26, 1.73 -> straight: contradicted
    # quarter 2: written 2, 2 1/3, 2 2/3 played 2.02, 2.36, 2.70 -> as written
    # quarter 3: written 3 1/4, 3 3/4 played 3.33, 3.67 -> straight written, triplets played
    # bar 1, quarter 0: one note off the quarter only -> no judgement
    rows = np.array([row(0, 1, 1), row(0, 4, 3), row(0, 5, 3), row(0, 2, 1), row(0, 7, 3), row(0, 8, 3),
                     row(0, 13, 4), row(0, 15, 4), row(1, 1, 3), [0, 0, 0, 1, 1, 4, 0, 0]])
    pos = np.array([1.0, 1.26, 1.73, 2.02, 2.36, 2.70, 3.33, 3.67, 0.25, 0.0])
    bar = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    c = contradicted(rows, bar, pos)
    assert c.tolist() == [0, TRIPLET_PLAYED_STRAIGHT, TRIPLET_PLAYED_STRAIGHT, 0, 0, 0,
                          STRAIGHT_PLAYED_TRIPLET, STRAIGHT_PLAYED_TRIPLET, 0, 0]
    # one note far from both: not every note near the sixteenths -> left
    pos2 = pos.copy()
    pos2[2] = 1.6
    assert contradicted(rows, bar, pos2)[1] == 0
    # the grid's bar differs from the written one: no judgement
    assert contradicted(rows, np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0]), pos)[1:3].tolist() == [0, 0]


def test_local_tempo():
    from src.meter import Q_STEPS, DecodeConfig, local_tempo_logprior
    t = np.arange(0, 10, 0.5)
    t[11:] -= 0.5 - 0.15          # one step of 0.15 s among steps of 0.5 s
    q = np.arange(len(t), dtype=float)
    q[11:] += 2.0                  # decoded as three quarters (a jump)
    cfg = DecodeConfig(local_weight=1.0, local_sigma=0.25)
    lp = local_tempo_logprior(t, q, cfg)
    one, three = list(Q_STEPS).index(1.0), list(Q_STEPS).index(3.0)
    assert lp[3, one] > lp[3, three]                    # a usual step: one quarter is the plausible tempo
    assert lp[10, one] < lp[3, one]                     # the short step is unusual at any reading ...
    assert np.all(local_tempo_logprior(t, q, DecodeConfig()) == 0)   # off by default
    assert np.all(local_tempo_logprior(t[:3], q[:3], cfg) == 0)


def test_switches():
    from scripts.system_scores import system_name
    from src.release import BEAT_DECODE
    assert BEAT_DECODE is True
    assert system_name(None, "", "", "nl+m", "synth", 0.95, True, "audio", timing_check=True, beat_decode=True,
                       timing_beats=True, readable=True, fix_jumps=True) == "v2_nl_m_d95_tidy_tc_tb_rd_jf_bd"
    for f in ("demo/app.py", "scripts/audio_to_score.py"):
        assert "--no-beat-decode" in (ROOT / f).read_text(encoding="utf-8"), f
    jobs = (ROOT / "pace" / "v2_jobs.sh").read_text()
    assert 'bd) out="$out --beat-decode" ;;' in jobs and "beatdecode)" in jobs and "localtempo)" in jobs
    assert 'nl_gq_mc) echo "--size base --grid-noise 0.8 --gangqinpu-share 0.5 --mask-contradicted" ;;' in jobs
    assert "--mask-contradicted" in (ROOT / "scripts" / "train_labeller.py").read_text(encoding="utf-8")
    assert "--local-tempo" in (ROOT / "scripts" / "eval_meter.py").read_text(encoding="utf-8")
    assert "--beat-decode" in (ROOT / "scripts" / "eval_labeller.py").read_text(encoding="utf-8")
    x = (ROOT / "scripts" / "export_models.py").read_text(encoding="utf-8")
    assert '"beat_decode.json"' in x


def test_pipeline_and_data():
    """With the repo's data: the dataset masks contradicted notes; the pipeline's switch (untuned: as before)."""
    if not V.exists() or not (ROOT / "data" / "processed" / "val" / "notelabels.npz").exists():
        print("  (skipped: no data/processed/val)")
        return
    from src.dataset import ScoreSegmentDataset
    ds = ScoreSegmentDataset(str(ROOT / "data" / "processed" / "val"), str(V), notelabels=True)
    found = 0
    for i in range(0, min(len(ds), 400), 4):
        plain = ds[i]["labels"].numpy()
        ds.mask_contradicted = True
        it = ds[i]
        ds.mask_contradicted = False
        c = it["contradicted"].numpy()
        lab = it["labels"].numpy()
        assert (lab[c > 0, 2:5] == -1).all() and (lab[c == 0] == plain[c == 0]).all()
        assert (lab[:, [0, 1, 5]] == plain[:, [0, 1, 5]]).all()          # keep, bar, staff untouched
        found += int((c > 0).sum())
    assert found > 0
    from scripts.system_inputs import write_midi
    from src.pipeline import V2Pipeline
    from tests.test_glissando import chords
    from tests.test_v2_system import make_v2
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        lab, met = make_v2(tmp)
        notes = np.array(chords(0.5, 10, step=0.25), np.float32)
        midi = tmp / "p.mid"
        write_midi(notes, np.zeros((0, 2), np.float32), midi)
        times = np.arange(0.5, 10.5, 0.5)
        beats = (times, np.arange(len(times)) % 4 == 0)
        pipe = V2Pipeline(lab, met, vocab=V, device="cpu", window_notes=48)
        assert pipe.beat_params is None and not pipe.beat_decode
        off = pipe.run(midi, tmp / "off", beats=beats)
        untuned = pipe.run(midi, tmp / "un", beats=beats, beat_decode=True)
        assert off.stats["beat_decode"] is False and untuned.stats["beat_decode"] == "not tuned"
        assert (tmp / "off" / "tokens.txt").read_text() == (tmp / "un" / "tokens.txt").read_text()
        Path(lab).parent.joinpath("beat_decode.json").write_text(json.dumps({"best": {"weight": 4.0, "consistent": True}}))
        pipe = V2Pipeline(lab, met, vocab=V, device="cpu", window_notes=48)
        on = pipe.run(midi, tmp / "on", beats=beats, beat_decode=True)
        assert on.stats["beat_decode"]["weight"] == 4.0 and on.stats["notes_kept"] == off.stats["notes_kept"]


def test_tuner():
    """scripts/tune_beat_decode.py's scoring on made-up predictions: the argmax row, a decoding that fixes a note."""
    from scripts.tune_beat_decode import score, settings
    f = Families(Space())
    sub = logp_for(8, [(1, None, 0), (5, 3, 0.5)])
    w = {"sub": sub, "dur": np.zeros((2, 4)), "use": np.array([True, True]), "own": np.array([True, True]),
         "pick": {"delta": np.array([1, 1]), "pos": np.array([0, 0]), "sub": sub.argmax(1), "dur": np.array([2, 0]),
                  "staff": np.array([0, 0])},
         "bar_est": np.array([1, 1]), "pos_est": np.array([0.5, 0.75]),
         "labels": np.array([[1, 1, 0, 1, 2, 0], [1, 1, 0, 3, 0, 0]])}
    base = score([w], None, f)
    fixed = score([w], BeatDecode(weight=500.0, consistent=True, anchor=0.0), f)
    assert base["pos_acc"] == 0.5 and fixed["pos_acc"] == 1.0
    assert base["mixed_beats"] == 0.0                                    # 1/2 counts for both families
    s = settings([0.0, 4.0])
    assert len(s) == 2 + 2 * 2 * 2 and s[0] == BeatDecode(0.0, False, False, 0.0)


def test_batch():
    """pace/v2_decode.sh: 8 jobs, the dependencies, the systems."""
    import shutil
    import subprocess

    from tests.batch_env import batch_env
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_decode.sh", tmp / "pace")
        for f in ("outputs/runs/nl_gq/best.pt", "outputs/runs/meter_v4_gq/best.pt", "outputs/runs/meter_v4_gq/decode.json",
                  "outputs/system_eval/manifest.json", "outputs/system_eval_gangqinpu/manifest.json", "bd/MIDI2ScoreTF.ckpt"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "sbatch").chmod(0o755)
        r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_decode.sh'], cwd=tmp,
                           env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd")),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        by = {cmd.split("-J ")[1].split()[0]: (jid, cmd)
              for jid, cmd in (ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines())}
        assert list(by) == ["dctests", "dcaudit", "dctune", "dctrain", "dceval", "dcasap", "dcgq", "dcfinal"]
        assert "tests.test_beat_decode" in by["dctests"][1]
        for j in ("dcaudit", "dctune", "dctrain"):
            assert f"afterok:{by['dctests'][0]}" in by[j][1], j
        assert "audit_labels.py" in by["dcaudit"][1]
        assert "beatdecode nl_gq" in by["dctune"][1] and "localtempo meter_v4_gq" in by["dctune"][1]
        assert "labellers nl_gq_mc" in by["dctrain"][1]
        assert f"afterok:{by['dctrain'][0]}" in by["dceval"][1] and "beatdecode nl_gq_mc" in by["dceval"][1]
        m = "nl_gq+meter_v4_gq:tidy,d95,tc"
        for j in ("dcasap", "dcgq"):
            s = by[j][1]
            assert f"afterany:{by['dctune'][0]}:{by['dceval'][0]}" in s
            for spec in (f"{m},missing", f"{m},tb,missing", f"{m},rd,missing", f"{m},jf,missing", f"{m},bd,missing",
                         f"{m},tb,rd,jf,missing", f"{m},tb,rd,jf,bd,missing", "nl_gq_mc+meter_v4_gq:tidy,d95,tc,missing",
                         "nl_gq+meter_v4_gq_lt:tidy,d95,tc,missing"):
                assert f" {spec}" in s, (j, spec)
        assert "--manifest outputs/system_eval_gangqinpu/manifest.json" in by["dcgq"][1]
        f = by["dcfinal"][1]
        assert f"afterany:{by['dcasap'][0]}:{by['dcgq'][0]}" in f and f.count("beyer_dai.py score") == 2


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
