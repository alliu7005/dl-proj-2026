"""Tests for the meter decoder's piece-tempo prior (src/meter.py piece_tempo_ref,
piece_tempo_logprior, grid_from_logp), its tuning (scripts/eval_meter.py tune_piece,
--piece-tempo), the jobs' piecetempo task, and pace/v2_tempo.sh.

    python -m tests.test_piece_tempo        # or: pytest tests/test_piece_tempo.py

A synthetic piece: 4/4, a quarter every 0.5 s, 16 bars; for four bars the tracker ticks the eighths
and marks a downbeat every four of its beats, and the model leans to a quarter per tracked beat
there (as Beat This! and the meter model did in the user's "blue bird" and "unlasting"): decoded
as it is, those bars come out at double the note values (eight bars of 4/4 in four bars' time);
with the prior, at the piece's tempo.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.meter import (Q_STEPS, TS_CLASSES, DecodeConfig, annotation_grid, grid_from_logp, grid_scores,  # noqa: E402
                       piece_tempo_logprior, piece_tempo_ref)

SEC = (24, 40)   # the true beats where the tracker ticks eighths


def piece(lean: float = 0.6):
    """(tracked beat times, tracker downbeats, log-probabilities {bar, step, ts}, ground truth)."""
    t_true = 0.5 * np.arange(64)
    gt = annotation_grid({"performance_beats": t_true.tolist(), "performance_downbeats": t_true[::4].tolist(),
                          "perf_time_signatures": {"0.0": ["4/4"]}})
    bt, down, in_sec = [], [], []
    for k, x in enumerate(t_true):
        bt.append(x)
        in_sec.append(SEC[0] <= k < SEC[1])
        if SEC[0] <= k < SEC[1]:          # the eighth after it
            bt.append(x + 0.25)
            in_sec.append(True)
    bt = np.array(bt)
    in_sec = np.array(in_sec)
    # the tracker's downbeats: the true ones outside; every fourth of its beats inside (its own level)
    for j, x in enumerate(bt):
        if in_sec[j]:
            down.append((np.flatnonzero(in_sec)[0] - j) % 4 == 0)
        else:
            down.append(any(abs(x - y) < 1e-9 for y in t_true[::4]))
    down = np.array(down)
    K = len(bt)
    pb = np.where(down, 0.9, 0.05)
    bar = np.log(np.stack([1 - pb, pb], 1))
    step = np.full((K, len(Q_STEPS)), 0.01)
    one, half = list(Q_STEPS).index(1), list(Q_STEPS).index(0.5)
    step[~in_sec, one] = 0.85
    step[in_sec, one] = lean                # a quarter per tracked beat: wrong inside
    step[in_sec, half] = 0.85 - lean
    step = np.log(step / step.sum(1, keepdims=True))
    ts = np.full((K, len(TS_CLASSES)), 0.01)
    ts[:, TS_CLASSES.index("4/4")] = 0.9
    ts = np.log(ts / ts.sum(1, keepdims=True))
    return bt, down, {"bar": bar, "step": step, "ts": ts}, gt


def quarters_in_section(g, bt):
    """Written quarters between the section's first and last tracked beat (true: 15.5)."""
    a = np.flatnonzero(np.isin(bt, 0.5 * np.arange(SEC[0], SEC[1])))
    return float(np.interp(bt[a[-1]] + 0.25, g["t"], g["q"]) - np.interp(bt[a[0]], g["t"], g["q"]))


def test_ref_and_logprior():
    t = np.cumsum(np.r_[0, np.full(30, 0.5), np.full(10, 0.25)])    # 30 steps at 120 qpm, 10 at 240 qpm
    q = np.arange(len(t), dtype=float)                               # a quarter per step
    ref = piece_tempo_ref(t, q)
    assert abs(math.exp(ref) - 120) < 3, math.exp(ref)
    assert piece_tempo_ref(t[:3], q[:3]) is None
    cfg = DecodeConfig(piece_weight=1.0, piece_sigma=0.25)
    lp = piece_tempo_logprior(t, ref, cfg)
    one, half = list(Q_STEPS).index(1), list(Q_STEPS).index(0.5)
    assert lp[0, one] > lp[0, half]          # at 0.5 s a step: a quarter is the piece's tempo
    assert lp[35, half] > lp[35, one]        # at 0.25 s: an eighth is
    assert np.all(lp[-1] == 0)
    assert np.all(piece_tempo_logprior(t, ref, DecodeConfig()) == 0)        # off by default
    assert np.all(piece_tempo_logprior(t, None, cfg) == 0)
    assert DecodeConfig().piece_weight == 0.0
    d = DecodeConfig.from_dict({"piece_weight": 2, "piece_sigma": 0.15, "unknown": 1}).to_dict()
    assert d["piece_weight"] == 2.0 and d["piece_sigma"] == 0.15


def test_decoding_fixes_the_stretch():
    bt, down, lp, gt = piece()
    plain = grid_from_logp(bt, lp, DecodeConfig())
    fixed = grid_from_logp(bt, lp, DecodeConfig(piece_weight=2.0, piece_sigma=0.25))
    assert quarters_in_section(plain, bt) > 25, quarters_in_section(plain, bt)        # double the note values
    assert abs(quarters_in_section(fixed, bt) - 16) < 1.5, quarters_in_section(fixed, bt)
    sp, sf = grid_scores(plain, gt), grid_scores(fixed, gt)
    assert sf["downbeat_f"] > sp["downbeat_f"] and sf["beat_f"] >= sp["beat_f"], (sp, sf)
    assert len(fixed["bars"]) == 16 and len(plain["bars"]) > 16
    # a model sure of its reading is not overruled by a light prior
    bt2, _, lp2, _ = piece(lean=0.84)
    assert quarters_in_section(grid_from_logp(bt2, lp2, DecodeConfig(piece_weight=0.5, piece_sigma=0.5)), bt2) > 25


def test_tune_piece():
    from scripts.eval_meter import PIECE_SEARCH, tune_piece
    assert 0.0 in PIECE_SEARCH["piece_weight"] and DecodeConfig().piece_sigma in PIECE_SEARCH["piece_sigma"]
    bt, down, lp, gt = piece()
    groups = [("asap", [{"gt": gt}], [lp], [(bt, down)])]
    cfg, info = tune_piece(groups, DecodeConfig())
    assert cfg.piece_weight > 0 and info["piece_select_tuned"] > info["piece_select_off"], info
    per = info["piece_per_source"]
    assert per["tuned"]["asap"]["select"] > per["off"]["asap"]["select"]
    # nothing to fix: it stays off (ties go to off)
    clean_lp = dict(lp)
    st = lp["step"].copy()
    one = list(Q_STEPS).index(1)
    st[:] = np.log(0.01)
    st[:, one] = np.log(0.9)
    t = 0.5 * np.arange(64)
    clean = {"bar": np.log(np.stack([np.where(np.arange(64) % 4 == 0, 0.1, 0.95),
                                     np.where(np.arange(64) % 4 == 0, 0.9, 0.05)], 1)),
             "step": st[:64], "ts": lp["ts"][:64]}
    cfg2, _ = tune_piece([("asap", [{"gt": gt}], [clean], [(t, np.arange(64) % 4 == 0)])], DecodeConfig())
    assert cfg2.piece_weight == 0.0
    del clean_lp


def test_jobs_task():
    j = (ROOT / "pace" / "v2_jobs.sh").read_text()
    assert "piecetempo)" in j and 'eval_meter.py --run "$pt" --piece-tempo' in j and 'pt="${m}_pt"' in j
    e = (ROOT / "scripts" / "eval_meter.py").read_text(encoding="utf-8")
    assert "--piece-tempo" in e and "tune_piece(groups, cfg)" in e


def test_batch():
    """pace/v2_tempo.sh: 5 jobs; the tuning after the tests; the systems after the tuning (and
    pace/v2_decode.sh's tuning still queued: the beat decoding); the scoring after the systems and
    the earlier batches' scoring still queued."""
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
            shutil.copy(ROOT / "pace" / "v2_tempo.sh", tmp / "pace")
            for f in ("outputs/runs/nl_gq/best.pt", "outputs/runs/meter_v4_gq/best.pt", "outputs/runs/meter_v4_gq/decode.json",
                      "outputs/system_eval/manifest.json", "outputs/system_eval_gangqinpu/manifest.json",
                      "bd/MIDI2ScoreTF.ckpt"):
                (tmp / f).parent.mkdir(parents=True, exist_ok=True)
                (tmp / f).write_text("x")
            (tmp / "bin").mkdir()
            (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                                'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
            q = ('#!/bin/bash\nfor a in "$@"; do case "$prev" in -n) case "$a" in dctune) echo 7001;; '
                 'dcfinal) echo 7005;; d2final) echo 7009;; esac;; esac; prev=$a; done\n') if queued else "#!/bin/bash\nexit 0\n"
            (tmp / "bin" / "squeue").write_text(q)
            for b in ("sbatch", "squeue"):
                (tmp / "bin" / b).chmod(0o755)
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_tempo.sh'], cwd=tmp,
                               env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd")),
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            by = {cmd.split("-J ")[1].split()[0]: (jid, cmd)
                  for jid, cmd in (ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines())}
            assert list(by) == ["pttests", "pttune", "ptasap", "ptgq", "ptfinal"]
            assert "tests.test_piece_tempo" in by["pttests"][1]
            assert f"--dependency=afterok:{by['pttests'][0]}" in by["pttune"][1]
            assert "piecetempo meter_v4_gq" in by["pttune"][1]
            for j in ("ptasap", "ptgq"):
                s = by[j][1]
                want = f"--dependency=afterok:{by['pttune'][0]}" + (",afterany:7001" if queued else " ")
                assert want in s, (j, s[:200])
                for spec in ("nl_gq+meter_v4_gq_pt:tidy,d95,missing", "nl_gq+meter_v4_gq_pt:tidy,d95,bd,missing",
                             "nl_gq+meter_v4_gq:tidy,d95,tc,tb,rd,jf,bd,tv,missing",
                             "nl_gq+meter_v4_gq_pt:tidy,d95,tc,tb,rd,jf,bd,tv,missing"):
                    assert f" {spec}" in s, (j, spec)
            assert "--manifest outputs/system_eval_gangqinpu/manifest.json" in by["ptgq"][1]
            f = by["ptfinal"][1]
            assert f"afterany:{by['ptasap'][0]}:{by['ptgq'][0]}" + (":7005:7009" if queued else "") in f
            assert f.count("beyer_dai.py score") == 2 and f.count("compare_systems.py") == 2
    # the systems are compared: compare_systems.py has the pairs
    c = (ROOT / "scripts" / "compare_systems.py").read_text(encoding="utf-8")
    for name in ("_meter_v4_gq_pt_d95_tidy\"", "_meter_v4_gq_pt_d95_tidy_bd\"", "_tc_tb_rd_jf_bd_tv\""):
        assert name in c, name


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
