"""Simpler rhythms: the complexity of a position / length, the penalty in the labeller's choice, plain
written lengths, the systems, the report's counts, pace/v2_simplicity.sh.

    python -m tests.test_simplicity
"""
from __future__ import annotations

import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
F = Fraction


def test_rhythm_complexity():
    from src.labeller import rhythm_complexity
    assert [rhythm_complexity(F(x)) for x in ("0", "1/2", "1/4", "3/4", "3/2", "4")] == [0] * 6
    assert [rhythm_complexity(F(x)) for x in ("1/8", "1/3", "1/6", "2/3")] == [1] * 4
    assert rhythm_complexity(F(1, 12)) == 1.5 and rhythm_complexity(F(1, 16)) == 1.5
    assert rhythm_complexity(F(1, 5)) == 2 and rhythm_complexity(F(1, 24)) == 2


def test_penalty_in_placed_notes():
    import torch

    from src.labeller import complexity_penalties, placed_notes
    from src.notelabels import N_FLAGS, N_POS, LabelSpace
    space = LabelSpace(["bar", "sub_1/2", "sub_1/3", "sub_1/4", "d_1/6", "d_1/4", "d_1/2", "d_1"])
    ps, pd = complexity_penalties(space)
    assert ps.tolist() == [0, 0, 1, 0] and pd.tolist() == [1, 0, 0, 0]
    out = {"keep": torch.full((1, 1), 5.0), "delta": torch.zeros(1, 1, 3), "pos": torch.zeros(1, 1, N_POS),
           "sub": torch.zeros(1, 1, space.n_sub), "dur": torch.zeros(1, 1, space.n_dur),
           "staff": torch.zeros(1, 1, 4), "flags": torch.full((1, 1, N_FLAGS), -5.0)}
    out["delta"][..., 1] = 5
    out["pos"][0, 0, 0] = 5
    out["sub"][0, 0, space.sub_index[F(1, 3)]] = 1.0       # a triplet a little likelier than the eighth
    out["sub"][0, 0, space.sub_index[F(1, 2)]] = 0.6
    out["dur"][0, 0, space.dur_index[F(1, 6)]] = 1.0
    out["dur"][0, 0, space.dur_index[F(1, 4)]] = 0.7
    args = (out, 0, np.array([60.0]), np.array([1]), np.array([True]), space)
    p0 = placed_notes(*args)[0]
    assert (p0.pos, p0.total) == (F(1, 3), F(1, 6))
    p1 = placed_notes(*args, simplicity=1.0)[0]
    assert (p1.pos, p1.total) == (F(1, 2), F(1, 4))
    out["sub"][0, 0, space.sub_index[F(1, 3)]] = 3.0       # a clear triplet stays one
    assert placed_notes(*args, simplicity=1.0)[0].pos == F(1, 3)


def test_simplify_lengths():
    from src.notelabels import Placed
    from src.readable import simplify_lengths
    pl = [Placed(0, F(1, 2), 48, F(1, 6), 1), Placed(0, F(1), 50, F(1, 2), 1),     # detached off-beat eighth
          Placed(0, F(1, 3), 72, F(1, 6), 0), Placed(0, F(2, 3), 74, F(1, 3), 0),  # a real triplet: kept
          Placed(0, F(2), 76, F(1, 8), 0), Placed(0, F(7, 2), 77, F(1, 5), 0)]     # gap 3/2 > a beat; the last note
    assert simplify_lengths(pl, ["4/4"]) == 3
    assert pl[0].total == F(1, 2)                       # up to the next onset
    assert pl[2].total == F(1, 6) and pl[3].total == F(1, 3)
    assert pl[4].total == F(1, 4)                       # the nearest plain value that ends by the next onset
    assert pl[5].total == F(1, 4)                       # no next onset: the nearest plain value


def test_system_names():
    from scripts.system_scores import system_name
    base = dict(delete_above=0.95, tidy=True)
    assert system_name(None, "meter", "synth", "nl_gq+meter_v4_gq", simplicity=2.0, simplify=True, **base) \
        == "v2_nl_gq_meter_v4_gq_d95_tidy_sp20_sl"
    assert system_name(None, "meter", "synth", "nl_gq+meter_v4_gq", simplicity=1.0, **base).endswith("_tidy_sp10")


def test_rare_shares():
    import music21 as m21

    from scripts.simplicity_report import rare_shares
    s = m21.stream.Score()
    part = m21.stream.Part()
    m = m21.stream.Measure(number=1)
    m.append(m21.meter.TimeSignature("4/4"))
    for ql in (1.0, 0.5, 0.5):
        m.append(m21.note.Note("C4", quarterLength=ql))
    t = m21.duration.Tuplet(3, 2, "eighth")
    for _ in range(3):
        n = m21.note.Note("E4", quarterLength=F(1, 3))
        n.duration.appendTuplet(t)
        m.append(n)
    a, b = m21.note.Note("G4", quarterLength=0.5), m21.note.Note("G4", quarterLength=0.5)
    a.tie, b.tie = m21.tie.Tie("start"), m21.tie.Tie("stop")
    m.append([a])
    part.append(m)
    m2 = m21.stream.Measure(number=2)
    m2.append(b)
    m2.append(m21.note.Rest(quarterLength=3.5))
    part.append(m2)
    s.append(part)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "x.musicxml"
        s.write("musicxml", fp=str(f))
        on, ln, n = rare_shares(str(f))
    # 3 plain notes, 3 triplet eighths (2 on rare positions, all 3 rare lengths), one tied quarter
    assert n == 7 and abs(on - 2 / 7) < 1e-9 and abs(ln - 3 / 7) < 1e-9, (on, ln, n)


def test_simplicity_batch():
    import shutil
    import subprocess

    from tests.batch_env import batch_env
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_simplicity.sh", tmp / "pace")
        (tmp / "bd").mkdir()
        (tmp / "bd" / "MIDI2ScoreTF.ckpt").write_text("x")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "sbatch").chmod(0o755)
        r = subprocess.run(["bash", "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_simplicity.sh'], cwd=tmp,
                           env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd")),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        by = {ln.split("\t")[1].split("-J ")[1].split()[0]: ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()}
        assert list(by) == ["x1tests", "x1asap", "x1gq", "x1final"]
        for o in ("sp10", "sp20", "sl", "sp20,sl", "jf"):
            assert f"nl_gq+meter_v4_gq:tidy,d95,{o},missing" in by["x1asap"][1]
        assert f"afterany:{by['x1asap'][0]}:{by['x1gq'][0]}" in by["x1final"][1] and "simplicity_report.py" in by["x1final"][1]
    # the job library turns the options into system_scores.py flags
    r = subprocess.run(["bash", "-c", 'eval "$(sed -n "/^v2_flags()/,/^}/p" pace/v2_jobs.sh)"; v2_flags tidy,sp20,sl'],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.stdout.split() == ["--tidy", "--simplicity", "2", "--simplify-lengths"], r.stdout + r.stderr


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
