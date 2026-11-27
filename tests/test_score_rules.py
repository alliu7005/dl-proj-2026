"""Tests for the score-level rules: PKSpell (src/pkspell.py), the engraving rules (src/readable.py
engrave_score: clefs, one-voice stems), swing (src/swing.py), their system options
(scripts/system_scores.py system_name, pace/v2_jobs.sh v2_flags) and scripts/postprocess_scores.py.

    python -m tests.test_score_rules        # or: pytest tests/test_score_rules.py

PKSpell's weights are downloaded once (pace/v2_pilot3.sh fetches them on the login node); without
them and without a network, the tests that need them say so and skip.
"""
from __future__ import annotations

import copy
import itertools
import json
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.notelabels import Placed  # noqa: E402
from src.pkspell import kmeans_1d  # noqa: E402
from src.swing import detect_swing, write_swing  # noqa: E402


def have_weights() -> bool:
    try:
        from src.pkspell import fetch_weights
        fetch_weights()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  (skipped: PKSpell weights not available: {type(e).__name__})")
        return False


def test_kmeans_1d():
    """The exact 1-D k-means: the partition into contiguous groups with the least squared error."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        x = rng.choice([0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4], size=int(rng.integers(5, 40))).astype(float)
        lab = kmeans_1d(x, 4)
        vals = np.unique(x)
        if len(vals) <= 4:
            assert len(set(lab)) == len(vals)
            continue
        def sse(cuts):
            groups = np.split(vals, cuts)
            return sum(((x[np.isin(x, g)] - x[np.isin(x, g)].mean()) ** 2).sum() for g in groups)
        best = min(sse(list(c)) for c in itertools.combinations(range(1, len(vals)), 3))
        got = sum(((x[lab == k] - x[lab == k].mean()) ** 2).sum() for k in set(lab))
        assert abs(got - best) < 1e-9, (x, lab)
        order = [lab[np.argmax(x == v)] for v in vals]
        assert order == sorted(order)                       # labels follow the values


def test_pkspell():
    """On a correctly spelled score with every spelling scrambled to sharps: the key back and the
    spellings far above the key-signature rule v2 used (C# minor's leading tone is B#, not C)."""
    if not have_weights():
        return
    import music21 as m21
    from music21 import corpus

    from src.pkspell import predict, respell
    sc = corpus.parse("chopin/mazurka06-2.krn")
    true = [p.name for n in sc.flatten().notes for p in n.pitches]
    s2 = copy.deepcopy(sc)
    for n in s2.flatten().notes:
        for p in n.pitches:
            p.name = m21.pitch.Pitch(p.midi).name
    midis = [p.midi for n in s2.flatten().notes for p in n.pitches]
    info = respell(s2)
    got = [p.name for n in s2.flatten().notes for p in n.pitches]
    assert info["key"] == 4 and info["notes"] == len(true)
    assert sum(a == b for a, b in zip(got, true)) / len(true) > 0.8
    assert [p.midi for n in s2.flatten().notes for p in n.pitches] == midis      # no pitch moved
    assert all(k.sharps == 4 for k in s2.recurse().getElementsByClass(m21.key.KeySignature))
    names, keys = predict([0, 4, 7], [1.0, 1.0, 2.0])
    assert names == ["C", "E", "G"] and len(keys) == 3
    assert predict([], []) == ([], [])


def _part(notes, clef="treble"):
    """A one-staff music21 part of 4/4 bars from (offset, quarter length, MIDI pitches)."""
    import music21 as m21
    p = m21.stream.PartStaff()
    m = m21.stream.Measure(number=1)
    m.timeSignature = m21.meter.TimeSignature("4/4")
    m.clef = m21.clef.TrebleClef() if clef == "treble" else m21.clef.BassClef()
    for off, ql, ps in notes:
        el = m21.chord.Chord(ps, quarterLength=ql) if len(ps) > 1 else m21.note.Note(ps[0], quarterLength=ql)
        m.insert(off, el)
    p.append(m)
    return p


def test_engraving():
    import music21 as m21

    from src.readable import choose_clefs, engrave_stems
    sc = m21.stream.Score()
    # treble: E5 (above B4) down, C4 (below) up, B4 (on the line) down, a chord F4-C6 (C6 farther) down,
    # two eighths in beat 3: G4 and E5 share one direction (E5 is farther: down), a whole note none
    sc.insert(0, _part([(0, 1, [76]), (1, 1, [60]), (2, 0.5, [67]), (2.5, 0.5, [76]), (3, 1, [71])]))
    sc.insert(0, _part([(0, 1, [65, 84]), (1, 3, [55])], clef="bass"))
    n = engrave_stems(sc)
    up = {(el.offset, tuple(p.midi for p in el.pitches)): el.stemDirection for el in sc.parts[0].flatten().notes}
    assert up[(0.0, (76,))] == "down" and up[(1.0, (60,))] == "up" and up[(3.0, (71,))] == "down"
    assert up[(2.0, (67,))] == up[(2.5, (76,))] == "down"
    low = {el.offset: el.stemDirection for el in sc.parts[1].flatten().notes}
    assert low[0.0] == "down" and low[1.0] == "down"       # bass clef: middle line D3; G3 is above it
    assert n == 7
    # a measure with two voices keeps stem_voices's choice
    two = m21.stream.Score()
    p = _part([])
    v1, v2 = m21.stream.Voice(), m21.stream.Voice()
    v1.insert(0, m21.note.Note(76, quarterLength=4))
    v2.insert(0, m21.note.Note(72, quarterLength=1))
    v2[0].stemDirection = "down"
    p.getElementsByClass(m21.stream.Measure)[0].insert(0, v1)
    p.getElementsByClass(m21.stream.Measure)[0].insert(0, v2)
    two.insert(0, p)
    assert engrave_stems(two) == 0
    # clefs: a lower staff mostly above middle C goes to treble, an upper staff deep down to bass
    sc2 = m21.stream.Score()
    sc2.insert(0, _part([(0, 1, [45]), (1, 1, [47])]))
    sc2.insert(0, _part([(0, 2, [64, 67]), (2, 2, [62, 65])], clef="bass"))
    assert choose_clefs(sc2) == 2
    clefs = [p.getElementsByClass(m21.stream.Measure)[0].clef for p in sc2.parts]
    assert isinstance(clefs[0], m21.clef.BassClef) and isinstance(clefs[1], m21.clef.TrebleClef)
    assert choose_clefs(sc2) == 0


def _placed(positions, bars=8, ts="4/4"):
    out = []
    for b in range(bars):
        for pos, tot in positions:
            out.append(Placed(bar=b, pos=Fraction(pos), pitch=60, total=Fraction(tot), staff=0))
    return out, [ts] * bars


def test_swing():
    F = Fraction
    # swung sixteenths everywhere: 0 (1/3) 1/3 (1/6) 1/2 (1/3) 5/6 (1/6) per beat
    beat = [(0, F(1, 3)), (F(1, 3), F(1, 6)), (F(1, 2), F(1, 3)), (F(5, 6), F(1, 6))]
    pl, ts = _placed([(b + p, t) for b in range(4) for p, t in beat])
    assert detect_swing(pl, ts)[0] == "16th"
    n, kind = write_swing(pl, ts)
    assert kind == "16th" and n == len(pl)
    assert sorted({p.pos - int(p.pos) for p in pl}) == [0, F(1, 4), F(1, 2), F(3, 4)]
    assert all(p.total == F(1, 4) for p in pl)
    # straight sixteenths and real triplets are left alone
    pl, ts = _placed([(b + F(k, 4), F(1, 4)) for b in range(4) for k in range(4)])
    assert write_swing(pl, ts) == (0, None)
    pl, ts = _placed([(b + F(k, 3), F(1, 3)) for b in range(4) for k in range(3)])
    assert write_swing(pl, ts) == (0, None)
    # mostly straight already, the leftovers at 5/6 and late 32nds (7/8): straightened
    straight = [(b + F(k, 4), F(1, 4)) for b in range(4) for k in range(4)]
    left = [(0, F(1, 2)), (F(1, 2), F(1, 3)), (F(5, 6), F(1, 6)), (1, F(1, 2)), (F(3, 2), F(3, 8)), (F(15, 8), F(1, 8))]
    pl, ts = _placed(straight + left, bars=8)
    kind, m = detect_swing(pl, ts)
    assert kind == "16th" and m[F(7, 8)] == F(3, 4)
    write_swing(pl, ts)
    assert not {p.pos - int(p.pos) for p in pl} & {F(5, 6), F(7, 8)}
    # but not where real 32nd runs fill 1/8 and 5/8 as well
    runs = [(b + F(k, 8), F(1, 8)) for b in range(2) for k in range(8)]
    pl, ts = _placed(straight + left + runs, bars=8)
    assert detect_swing(pl, ts)[0] is None
    # swung eighths: 2/3 off-beats
    pl, ts = _placed([(b + p, t) for b in range(4) for p, t in ((0, F(2, 3)), (F(2, 3), F(1, 3)))])
    n, kind = write_swing(pl, ts)
    assert kind == "8th" and {p.pos - int(p.pos) for p in pl} == {0, F(1, 2)} and all(p.total == F(1, 2) for p in pl)
    # compound meters are not touched
    pl, ts = _placed([(b + p, t) for b in range(2) for p, t in beat], ts="6/8")
    assert write_swing(pl, ts) == (0, None)


def test_system_options():
    from scripts.system_scores import system_name
    base = system_name(None, "meter", "none", "nl_gq+meter_v4_gq", delete_above=0.95, tidy=True)
    assert base == "v2_nl_gq_meter_v4_gq_d95_tidy"
    assert system_name(None, "meter", "none", "nl_gq+meter_v4_gq", delete_above=0.95, tidy=True,
                       pkspell=True, engraving=True, swing=True) == base + "_ps_eg_sw"
    out = subprocess.run(["bash", "-c", 'eval "$(sed -n "/^v2_flags()/,/^}/p" pace/v2_jobs.sh)"; v2_flags tidy,d95,ps,eg,sw'],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.split() == ["--tidy", "--delete-above", "0.95", "--pkspell", "--engraving",
                                                          "--swing"], out.stdout + out.stderr


def test_postprocess():
    """A copy of a system with the engraving rules (and PKSpell when its weights are there), its
    summary renamed, scores the original had."""
    import music21 as m21

    from scripts.postprocess_scores import main as post
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        src = out / "v2_x" / "clean"
        src.mkdir(parents=True)
        sc = m21.stream.Score()
        sc.insert(0, _part([(0, 1, [76]), (1, 1, [60])]))
        sc.insert(0, _part([(0, 2, [48, 55])], clef="bass"))
        sc.write("musicxml", fp=str(src / "a.musicxml"))
        (out / "v2_x" / "summary.json").write_text(json.dumps(
            {"system": "v2_x", "pieces": [{"slug": "a", "condition": "clean", "musicxml": str(src / "a.musicxml")}]}))
        flags = ["--engraving"] + (["--pkspell"] if have_weights() else [])
        assert post(["--system", "v2_x", "--out", d] + flags) == 0
        name = "v2_x" + ("_ps" if "--pkspell" in flags else "") + "_eg"
        summ = json.loads((out / name / "summary.json").read_text())
        assert summ["system"] == name and summ["postprocessed_from"] == "v2_x" and not summ["postprocess"]["failed"]
        assert summ["pieces"][0]["musicxml"].endswith(f"{name}/clean/a.musicxml")
        got = m21.converter.parse(str(out / name / "clean" / "a.musicxml"))
        assert [el.stemDirection for el in got.parts[0].flatten().notes] == ["down", "up"]


def test_pilot3_graph():
    """pace/v2_pilot3.sh with stand-ins for sbatch / squeue / python3 (the weights' fetch): 4 jobs,
    waiting for the first two pilots' by name, the release post-processed three ways on both test
    sets, swing through the pipeline, nothing published, the report last."""
    import shutil

    from tests.batch_env import batch_env
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_pilot3.sh", tmp / "pace")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "squeue").write_text('#!/bin/bash\nfor a in "$@"; do case "$prev" in -n) case "$a" in '
                                            'q1prep) echo 900;; q1asap) echo 901;; q1gq) echo 902;; r2asap) echo 903;; '
                                            'r2gq) echo 904;; q1final) echo 905;; r2final) echo 906;; esac;; esac; prev=$a; done\n')
        (tmp / "bin" / "python3").write_text("#!/bin/bash\necho 'PKSpell weights: (stand-in)'\n")
        for f in (tmp / "bin").iterdir():
            f.chmod(0o755)
        (tmp / "bd").mkdir()
        (tmp / "bd" / "MIDI2ScoreTF.ckpt").write_text("x")
        r = subprocess.run(["bash", "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_pilot3.sh'], cwd=tmp,
                           env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd")),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        by = {ln.split("\t")[1].split("-J ")[1].split()[0]: ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()}
        assert list(by) == ["p3tests", "p3post", "p3swing", "p3final"], list(by)
        tests, post, swing, final = (by[k] for k in by)
        assert "tests.test_score_rules" in tests[1]
        assert f"afterok:{tests[0]},afterany:901:902:903:904" in post[1]
        for out in ("outputs/system_eval;", "outputs/system_eval_gangqinpu;"):
            for f in ("--pkspell --out", "--engraving --out", "--pkspell --engraving --out"):
                assert f"--system v2_nl_gq_meter_v4_gq_d95_tidy {f} {out}" in post[1], (f, out)
        assert f"afterok:{tests[0]},afterany:900" in swing[1] and "nl_gq+meter_v4_gq:tidy,d95,ps,eg,sw,missing" in swing[1]
        assert "PUBLISH=0" in swing[1] and "--manifest outputs/system_eval_gangqinpu/manifest.json" in swing[1]
        assert f"afterany:{post[0]}:{swing[0]}:905:906" in final[1] and "pilot_report.py" in final[1]


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
