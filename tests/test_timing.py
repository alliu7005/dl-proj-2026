"""Tests for the timing check (src/timing.py, V2Pipeline timing_check, the scripts' switches).

    python -m tests.test_timing        # or: pytest tests/test_timing.py

A steady grid (one quarter a beat, 0.5 s) and placed notes that record the performed note each came
from: a triplet-written note played on a straight eighth moves there (its triplet length with it), a
swung one (played between 2/3 and 3/4) stays, one played on the next bar's downbeat moves into that
bar, a 6/8 bar and grace notes are left alone, a note played between the two stays; the pipeline's
switch, the system name, the demo's and the CLI's switches. check_beats: the song's syncopated beat
(1 2/3, 2 1/3, 2 2/3 written; played straight) written straight as a whole beat, the beat measured
from where it was played, played triplets and a sextuplet run left, 6/8 left.
"""
from __future__ import annotations

import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.notelabels import Placed  # noqa: E402
from src.timing import check_beats, check_timing, is_triplet, quarter_beat  # noqa: E402

V = ROOT / "data" / "processed" / "vocab.json"


def test_check_timing():
    bars = ["4/4", "4/4", "6/8", "4/4"]
    bar_q = np.array([0.0, 4.0, 8.0, 11.0])
    grid = {"t": np.arange(0, 8.0, 0.5), "q": np.arange(0, 16.0), "bar_q": bar_q, "bars": bars}
    sec = lambda q: q * 0.5                                                   # noqa: E731

    onsets = np.array([sec(1.5), sec(2.70), sec(3.98), sec(9.5), sec(1.42), sec(4.25), sec(5.58), sec(12.5)])
    p_eighth = Placed(0, Fraction(4, 3), 60, Fraction(1, 3), 0, src=0)       # 1 1/3, played on 1 1/2
    p_swing = Placed(0, Fraction(8, 3), 62, Fraction(1, 3), 0, src=1)        # 2 2/3, played at 2.70: stays
    p_next = Placed(0, Fraction(11, 3), 64, Fraction(2, 3), 0, src=2)        # 3 2/3, played on the next downbeat
    p_68 = Placed(2, Fraction(3, 2), 65, Fraction(1, 2), 0, src=3)           # 6/8: never touched (not a triplet anyway)
    p_68t = Placed(2, Fraction(4, 3), 65, Fraction(1, 3), 0, src=3)          # 6/8 and a third: still never
    p_between = Placed(0, Fraction(4, 3), 67, Fraction(1, 3), 0, src=4)      # played at 1.42: near neither
    p_grace = Placed(1, Fraction(1, 3), 69, Fraction(0), 0, grace=True, src=5)
    p_16th = Placed(1, Fraction(5, 3), 71, Fraction(1, 6), 0, src=6)         # bar 2 at 1 2/3, played at 1.58: stays
    p_last = Placed(3, Fraction(4, 3), 72, Fraction(1, 3), 0, src=7)         # last bar, played at 12.5 -> 1 1/2
    placed = [p_eighth, p_swing, p_next, p_68, p_68t, p_between, p_grace, p_16th, p_last]
    moved = check_timing(placed, onsets, grid, bars)
    assert (p_eighth.bar, p_eighth.pos, p_eighth.total) == (0, Fraction(3, 2), Fraction(1, 4)), p_eighth
    assert p_swing.pos == Fraction(8, 3)                                      # the labeller's swing stays
    assert (p_next.bar, p_next.pos) == (1, Fraction(0)) and p_next.total == Fraction(3, 4)
    assert p_68.pos == Fraction(3, 2) and p_68t.pos == Fraction(4, 3)
    assert p_between.pos == Fraction(4, 3) and p_grace.pos == Fraction(1, 3)
    # 1 2/3 played at 1.58: the nearest straight position is 1 1/2 (0.08 away: more than TOL) -> stays
    assert p_16th.pos == Fraction(5, 3)
    assert p_last.pos == Fraction(3, 2)
    assert moved == 3
    # a note without its performed note (src -1) or with no grid: nothing
    lone = Placed(0, Fraction(4, 3), 60, Fraction(1, 3), 0)
    assert check_timing([lone], onsets, grid, bars) == 0 and lone.pos == Fraction(4, 3)
    assert check_timing([Placed(0, Fraction(4, 3), 60, Fraction(1, 3), 0, src=0)], onsets, {"t": [], "q": []}, bars) == 0
    assert quarter_beat("4/4") and quarter_beat("3/4") and not quarter_beat("6/8") and not quarter_beat("12/8")
    assert is_triplet(Fraction(1, 3)) and is_triplet(Fraction(5, 6)) and not is_triplet(Fraction(1, 2))
    assert not is_triplet(Fraction(0))


def test_check_beats():
    bars = ["4/4", "4/4", "6/8", "4/4"]
    grid = {"t": np.arange(0, 8.0, 0.5), "q": np.arange(0, 16.0), "bar_q": np.array([0.0, 4.0, 8.0, 11.0]),
            "bars": bars}
    sec = lambda q: q * 0.5                                                   # noqa: E731
    on, placed = [], []

    def note(bar, pos, total, played, pitch=60):
        on.append(sec(played))
        p = Placed(bar, Fraction(pos), pitch, Fraction(total), 0, src=len(on) - 1)
        placed.append(p)
        return p
    # bar 1, beats 2-3 as in the song's bars 64-71: written 1 1/2 (a 1/6), 1 2/3, 2 1/3, 2 2/3
    a = note(0, Fraction(3, 2), Fraction(1, 6), 1.49)
    b = note(0, Fraction(5, 3), Fraction(1, 3), 1.71)                         # halfway: no one note decides
    c0 = note(0, 2, 1, 1.98)
    c = note(0, Fraction(7, 3), Fraction(1, 3), 2.24)                         # 0.09 from 2 1/3: under MARGIN
    d = note(0, Fraction(8, 3), Fraction(1, 3), 2.73)
    # bar 2, beat 1: played triplets -> stay; beat 3: the beat played 0.06 early, notes at 2.43 / 2.68
    t1 = note(1, Fraction(1, 3), Fraction(1, 3), 4.34)
    t2 = note(1, Fraction(2, 3), Fraction(1, 3), 4.66)
    e0 = note(1, 2, Fraction(1, 2), 5.94)
    e1 = note(1, Fraction(7, 3), Fraction(1, 3), 6.43)                        # 0.49 / 0.74 of the beat as played
    e2 = note(1, Fraction(8, 3), Fraction(1, 3), 6.68)
    # bar 2, beat 4: a sextuplet run played unevenly -> judged against sixths, left
    r1 = note(1, Fraction(19, 6), Fraction(1, 6), 7.15)
    r2 = note(1, Fraction(10, 3), Fraction(1, 6), 7.31)
    r3 = note(1, Fraction(7, 2), Fraction(1, 6), 7.52)
    # 6/8: never
    s = note(2, Fraction(4, 3), Fraction(1, 3), 9.5)
    assert check_timing(placed, np.array(on), grid, bars) == 0                # one note at a time: nothing is clear
    changed = check_beats(placed, np.array(on), grid, bars)
    assert (a.pos, a.total) == (Fraction(3, 2), Fraction(1, 4))
    assert (b.pos, b.total) == (Fraction(7, 4), Fraction(1, 4))
    assert (c.pos, d.pos) == (Fraction(9, 4), Fraction(11, 4)) and c0.pos == 2
    assert (t1.pos, t2.pos) == (Fraction(1, 3), Fraction(2, 3))
    assert (e1.pos, e2.pos) == (Fraction(5, 2), Fraction(11, 4)) and e0.pos == 2
    assert (r1.pos, r2.pos, r3.pos) == (Fraction(19, 6), Fraction(10, 3), Fraction(7, 2))
    assert s.pos == Fraction(4, 3)
    assert changed == 6
    # nothing to judge: no grid, no performed notes
    assert check_beats(placed, np.array(on), {"t": [], "q": []}, bars) == 0
    assert check_beats([Placed(0, Fraction(4, 3), 60, Fraction(1, 3), 0)], np.array(on), grid, bars) == 0


def test_pipeline_and_switches():
    """V2Pipeline: off unless asked; on, its count in the stats; the scripts' switches."""
    from scripts.system_scores import system_name
    from src.release import TIMING_BEATS, TIMING_CHECK
    assert TIMING_CHECK is True and TIMING_BEATS is True
    assert system_name(None, "", "", "nl+m", "synth", 0.95, True, "audio", timing_check=True) == "v2_nl_m_d95_tidy_tc"
    assert system_name(None, "", "", "nl+m", "synth", 0.95, True, "audio", timing_check=True, timing_beats=True,
                       readable=True) == "v2_nl_m_d95_tidy_tc_tb_rd"
    for f in ("demo/app.py", "scripts/audio_to_score.py"):
        x = (ROOT / f).read_text(encoding="utf-8")
        assert "--no-timing-check" in x and "--no-timing-beats" in x, f
    jobs = (ROOT / "pace" / "v2_jobs.sh").read_text()
    assert 'tc) out="$out --timing-check" ;;' in jobs and 'tb) out="$out --timing-beats" ;;' in jobs
    if not V.exists():
        print("  (pipeline part skipped: no data/processed/vocab.json)")
        return
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
        assert not pipe.timing_check and not pipe.timing_beats
        off = pipe.run(midi, tmp / "off", beats=beats)
        on = pipe.run(midi, tmp / "on", beats=beats, timing_check=True, timing_beats=True)
        assert off.stats["timing_beats"] is False and off.stats["timing_beats_fixed"] == 0
        assert on.stats["timing_beats"] is True and on.stats["timing_beats_fixed"] >= 0
        assert off.stats["timing_check"] is False and off.stats["timing_fixed"] == 0
        assert on.stats["timing_check"] is True and on.stats["timing_fixed"] >= 0
        assert on.stats["notes_kept"] == off.stats["notes_kept"]              # it moves notes, never drops them


def test_batch():
    """pace/v2_timing.sh: 3 jobs, the systems after the tests, the scoring after the systems."""
    import shutil
    import subprocess

    from tests.batch_env import batch_env
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_timing.sh", tmp / "pace")
        for f in ("outputs/runs/nl_gq/best.pt", "outputs/runs/meter_v4_gq/best.pt", "outputs/system_eval/manifest.json",
                  "outputs/system_eval_gangqinpu/manifest.json", "bd/MIDI2ScoreTF.ckpt"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "sbatch").chmod(0o755)
        r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_timing.sh'], cwd=tmp,
                           env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd")),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        by = {cmd.split("-J ")[1].split()[0]: (jid, cmd)
              for jid, cmd in (ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines())}
        assert list(by) == ["tctests", "tcsys", "tcfinal"]
        assert "tests.test_timing" in by["tctests"][1]
        s = by["tcsys"][1]
        assert f"afterok:{by['tctests'][0]}" in s and s.count("systems nl_gq+meter_v4_gq:tidy,d95,tc") == 2
        assert "--manifest outputs/system_eval_gangqinpu/manifest.json" in s
        f = by["tcfinal"][1]
        assert f"afterany:{by['tcsys'][0]}" in f and f.count("beyer_dai.py score") == 2 and "compare_systems.py" in f


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
