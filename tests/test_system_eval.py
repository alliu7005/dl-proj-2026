"""Tests for the system-level comparison with Beyer & Dai (Baseline 3).

    python -m tests.test_system_eval        # or: pytest tests/test_system_eval.py

Covers the inputs (scripts/system_inputs.py: real transcriptions shifted onto the ASAP timeline and
cut to their piece; the manifest on the local data build), our system on them (scripts/system_scores.py
with a tiny model, the tracked beats given to a beat model), the aggregation and tables of
scripts/beyer_dai.py (their metrics themselves need their environment: pace/bd_setup.sh checks
those on PACE), and the job graph of pace/system_eval.sh (stand-in sbatch).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import beyer_dai, system_inputs, system_scores  # noqa: E402
from src.beats import beats_path, save_beats  # noqa: E402
from src.preprocessing.performance import load_performance  # noqa: E402
from tests.test_pipeline import make_checkpoint, make_performance  # noqa: E402


def notes_of(k, t0=1.0, step=0.4, base=60):
    """k notes as load_performance gives them: pitch, onset, offset (onset + 0.3 s), velocity."""
    return np.array([[base + (i % 12), t0 + step * i, t0 + step * i + 0.3, 60 + i % 30] for i in range(k)], np.float32)


def test_real_input_shift_and_crop():
    """The transcription of a recording that holds two pieces: shifted by the offset onto the ASAP
    timeline and cut to this piece, the other piece's notes are gone and the pedal is clipped."""
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        clean = notes_of(20)                                   # the ASAP performance: 1.0 .. 8.9 s
        system_inputs.write_midi(clean, np.zeros((0, 2)), tmp / "clean.mid")
        offset = 3.0                                           # the recording starts 3 s earlier
        other = notes_of(15, t0=30.0, base=40)                 # the next piece in the same recording
        rec = np.concatenate([clean, other]).copy()
        rec[:, 1:3] += offset
        pedal = np.array([[offset + 0.5, offset + 2.0], [offset + 8.5, offset + 40.0]])
        system_inputs.write_midi(rec, pedal, tmp / "rec.mid")
        n = system_inputs.real_input(tmp / "clean.mid", tmp / "rec.mid", offset, tmp / "out" / "x.mid")
        got, ped = load_performance(tmp / "out" / "x.mid")
        assert n == len(got) == 20, n
        assert np.allclose(got[:, 1:3], clean[:, 1:3], atol=0.01) and np.array_equal(got[:, 0], clean[:, 0])
        hi = float(clean[:, 2].max()) + system_inputs.MARGIN
        assert len(ped) == 2 and abs(ped[1, 1] - hi) < 0.01 and abs(ped[0, 0] - 0.5) < 0.01, ped
    assert system_inputs.slug("Bach/Fugue/bwv_846/Shi05M.mid") == "Bach_Fugue_bwv_846_Shi05M"


def test_manifest_on_local_data():
    """The clean entries: every test performance once, its MIDI and reference score on disk."""
    data, asap = ROOT / "data" / "processed", ROOT / "data" / "raw" / "asap_bd"
    if not (data / "test" / "index.npy").exists() or not (asap / "metadata.csv").exists():
        print("  (skipped: data/processed/test or data/raw/asap_bd missing)")
        return
    with tempfile.TemporaryDirectory() as t:
        entries = system_inputs.build(data, asap, Path(t) / "none", Path(t) / "none", Path(t))
    clean = [e for e in entries if e["condition"] == "clean"]
    assert len(clean) == len({e["performance"] for e in clean}) >= 1
    assert all(Path(e["midi"]).exists() and Path(e["reference"]).name == "xml_score.musicxml" for e in clean)
    if not (data / "test_transcribed" / "report.json").exists():
        assert all(e["condition"] == "clean" for e in entries)


def test_system_scores_end_to_end():
    """Our pipeline on a manifest: one MusicXML per input (or the export error recorded), the
    tracked beats given to a beat model on the real input only."""
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        make_performance(tmp / "perf.mid", n=60)
        perf = "X/Y/perf.mid"
        manifest = [{"condition": "clean", "performance": perf, "slug": "X_Y_perf", "midi": str(tmp / "perf.mid"),
                     "reference": "ref.musicxml"},
                    {"condition": "real", "performance": perf, "slug": "X_Y_perf", "midi": str(tmp / "perf.mid"),
                     "reference": "ref.musicxml"}]
        (tmp / "manifest.json").write_text(json.dumps(manifest))
        times = 0.5 + 0.5 * np.arange(40)
        (tmp / "tracked").mkdir()
        save_beats(beats_path(tmp / "tracked", "tracked"), {perf: (times, np.arange(40) % 4 == 0)})
        for run, extra in (("plain", {}), ("beat", {"n_features": 15, "beat_features": True})):
            (tmp / "runs" / run).mkdir(parents=True)
            make_checkpoint(tmp / "runs" / run, **extra)
            system_scores.main(["--run", run, "--runs-dir", str(tmp / "runs"), "--vocab", str(ROOT / "data" / "processed" / "vocab.json"),
                                "--manifest", str(tmp / "manifest.json"), "--out", str(tmp / "out"), "--device", "cpu",
                                "--tracked", str(tmp / "tracked"), "--mode", "guarded"])
            summary = json.loads((tmp / "out" / f"{run}_guarded" / "summary.json").read_text())
            pieces = {p["condition"]: p for p in summary["pieces"]}
            assert set(pieces) == {"clean", "real"} and summary["mode"] == "guarded"
            for c, p in pieces.items():
                assert p["musicxml"] is None or Path(p["musicxml"]).exists(), p
                assert p["musicxml"] is not None or any(e.startswith("MusicXML") for e in p["errors"]), p
            if run == "beat":
                assert pieces["real"]["beats"] == "given: 40 beats, 10 downbeats", pieces["real"]
                assert pieces["clean"]["beats"] == "none (MIDI input)", pieces["clean"]
            else:
                assert pieces["real"]["beats"] == "not used by this model"


def rated_stub():
    """Two systems on three clean performances and one real one; B fails clean p3 and real r1."""
    def r(onset, dele):
        return {"muster": {"PitchER": 1.0, "MissRate": 2.0, "ExtraRate": 3.0, "OnsetER": onset, "OffsetER": 5.0,
                           "MeanER": 6.0},
                "sim": {"NoteDeletion": dele, "NoteInsertion": 0.1, "NoteDuration": 0.5, "StaffAssignment": 0.05,
                        "StemDirection": 0.2, "NoteSpelling": 0.08, "n_Note": 100.0}, "error": None}
    fail = {"muster": None, "sim": None, "error": "no prediction"}
    rated = {("A", "clean", "p1"): r(10.0, 0.10), ("A", "clean", "p2"): r(20.0, 0.20), ("A", "clean", "p3"): r(60.0, 0.30),
             ("B", "clean", "p1"): r(30.0, 0.40), ("B", "clean", "p2"): r(50.0, 0.20), ("B", "clean", "p3"): fail,
             ("A", "real", "r1"): r(15.0, 0.1), ("B", "real", "r1"): fail}
    entries = [{"condition": "clean", "performance": p} for p in ("p1", "p2", "p3")] + \
              [{"condition": "real", "performance": "r1"}]
    return rated, entries


def test_aggregate_and_table():
    rated, entries = rated_stub()
    res = beyer_dai.aggregate(rated, entries, ["A", "B"])
    a, b = res["clean"]["A"], res["clean"]["B"]
    # all: every performance a system scored; common: those both scored (p1, p2)
    assert a["all"]["n"] == 3 and abs(a["all"]["muster"]["OnsetER"] - 30.0) < 1e-9
    assert a["common"]["n"] == b["common"]["n"] == 2 and abs(a["common"]["muster"]["OnsetER"] - 15.0) < 1e-9
    assert abs(b["common"]["muster"]["OnsetER"] - 40.0) < 1e-9 and b["failed"] == ["p3"] and a["failed"] == []
    assert abs(a["common"]["sim"]["NoteDeletion"] - 15.0) < 1e-9  # per-note rates x 100, as their printout
    assert res["real"]["B"]["all"]["n"] == 0 and res["real"]["B"]["all"]["muster"]["OnsetER"] is None
    assert res["real"]["A"]["common"]["n"] == 0  # nothing both scored
    text = beyer_dai.table(res)
    assert "## clean, common" in text and "## real, all" in text
    assert text.count("Beyer & Dai, paper (Table 3)") == 2 and "| 15.55 |" in text  # with clean only
    row = next(ln for ln in text.splitlines() if ln.startswith("| A | 2 |"))
    assert "| 15.00 |" in row, row
    assert "Not scored (clean): B: 1" in text and "Not scored (real): B: 1" in text
    # a missing prediction is rated without touching their code
    out = beyer_dai.rate(("/no/repo", None, "ref.musicxml"))
    assert out == {"muster": None, "sim": None, "error": "no prediction"}


def test_system_eval_batch():
    """pace/system_eval.sh: inputs first, our systems and Beyer & Dai after them, the scoring after
    all of them (afterany), the tsearch system with the longer walltime; nothing without the setup."""
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "system_eval.sh", tmp / "pace")
        bin_ = tmp / "bin"
        bin_.mkdir()
        (bin_ / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                     'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (bin_ / "sbatch").chmod(0o755)
        bd = tmp / "bd"
        env = dict(os.environ, SUBMITS=str(tmp / "jobs.txt"), BD=str(bd))
        cmd = [shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/system_eval.sh']
        r = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True)
        assert r.returncode != 0 and "bash pace/bd_setup.sh first" in r.stdout and not (tmp / "jobs.txt").exists()
        (bd / "MIDI2ScoreTransformer").mkdir(parents=True)
        (bd / "MIDI2ScoreTF.ckpt").write_bytes(b"x")
        (tmp / "outputs" / "runs" / "sf_base_reg_cab").mkdir(parents=True)
        (tmp / "outputs" / "runs" / "sf_base_reg_cab" / "best.pt").write_bytes(b"")
        r = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        jobs = [ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()]
        by = {j[1].split("-J ")[1].split()[0] + ("_tsearch" if "--mode tsearch_beat" in j[1] else ""): j for j in jobs}
        inp = by["sysinputs"][0]
        assert "system_inputs.py" in by["sysinputs"][1] and "--gres" not in by["sysinputs"][1]
        assert "--mode meter" in by["sys_sf_base_reg_cab"][1] and "-t 3:00:00" in by["sys_sf_base_reg_cab"][1]
        assert "--mode tsearch_beat" in by["sys_sf_base_reg_cab_tsearch"][1] and "-t 8:00:00" in by["sys_sf_base_reg_cab_tsearch"][1]
        assert "skip s2s_m_cab:meter: outputs/runs/s2s_m_cab/best.pt not found" in r.stdout
        for name in ("sys_sf_base_reg_cab", "sys_sf_base_reg_cab_tsearch", "bdpredict"):
            assert f"--dependency=afterok:{inp}" in by[name][1], by[name]
        assert "conda activate bd311" in by["bdpredict"][1] and f"--ckpt {bd}/MIDI2ScoreTF.ckpt" in by["bdpredict"][1]
        dep = ":".join(by[n][0] for n in ("sys_sf_base_reg_cab", "sys_sf_base_reg_cab_tsearch", "bdpredict"))
        sc = by["sysscore"][1]
        assert f"--dependency=afterany:{dep}" in sc and "conda activate bd311" in sc
        assert "--systems beyer_dai,sf_base_reg_cab_meter,sf_base_reg_cab_tsearch_beat" in sc, sc
        # ONLY=bd / ONLY=score: no inputs, none of our systems; Beyer & Dai (bd only), then the scoring
        (tmp / "jobs.txt").unlink()
        env2 = dict(env, ONLY="score")
        r = subprocess.run(cmd, cwd=tmp, env=env2, capture_output=True, text=True)
        assert r.returncode != 0 and "manifest.json missing" in r.stdout and not (tmp / "jobs.txt").exists()
        (tmp / "outputs" / "system_eval").mkdir(parents=True)
        (tmp / "outputs" / "system_eval" / "manifest.json").write_text("[]")
        for only, names in (("score", ["sysscore"]), ("bd", ["bdpredict", "sysscore"])):
            if (tmp / "jobs.txt").exists():
                (tmp / "jobs.txt").unlink()
            r = subprocess.run(cmd, cwd=tmp, env=dict(env, ONLY=only, RUNS="sf_base_reg_cab:meter s2s_m_cab:meter"),
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            jobs = [ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()]
            assert [j[1].split("-J ")[1].split()[0] for j in jobs] == names, jobs
            sc = jobs[-1][1]
            assert "--systems beyer_dai,sf_base_reg_cab_meter,s2s_m_cab_meter" in sc, sc
            if only == "score":
                assert "--dependency" not in sc
            else:
                assert "--dependency" not in jobs[0][1] and f"--dependency=afterany:{jobs[0][0]}" in sc
        r = subprocess.run(cmd, cwd=tmp, env=dict(env, ONLY="nope"), capture_output=True, text=True)
        assert r.returncode != 0 and "unknown ONLY" in r.stdout


def test_split_chords():
    """score_similarity needs single notes: every chord becomes its notes at the same offset, each
    keeping its pitch and tie and the chord's duration, stem and articulations; nothing else moves."""
    from music21 import articulations, chord, meter, note, stream, tie
    m = stream.Measure()
    m.append(meter.TimeSignature("4/4"))
    c = chord.Chord(["C4", "E4", "G4"], quarterLength=2.0)
    c.stemDirection = "down"
    c.articulations.append(articulations.Staccato())
    c[1].tie = tie.Tie("start")
    m.insert(0, c)
    m.insert(2.0, note.Note("A4", quarterLength=1.0))
    m.insert(3.0, chord.Chord(["D5", "F5"], quarterLength=1.0))
    sc = stream.Score([stream.Part([m])])
    beyer_dai.split_chords(sc)
    assert not list(sc.recurse().getElementsByClass(chord.Chord))
    got = sorted((float(n.getOffsetInHierarchy(sc)), n.pitch.nameWithOctave, float(n.duration.quarterLength))
                 for n in sc.recurse().notes)
    assert got == [(0.0, "C4", 2.0), (0.0, "E4", 2.0), (0.0, "G4", 2.0), (2.0, "A4", 1.0), (3.0, "D5", 1.0),
                   (3.0, "F5", 1.0)], got
    first = [n for n in sc.recurse().notes if float(n.getOffsetInHierarchy(sc)) == 0.0]
    assert all(n.stemDirection == "down" and any(isinstance(a, articulations.Staccato) for a in n.articulations)
               for n in first)
    assert [n.pitch.name for n in first if n.tie is not None and n.tie.type == "start"] == ["E"]


def test_cli():
    for argv in (["predict", "--repo", "r"], ["score", "--repo", "r"]):  # required arguments
        try:
            beyer_dai.main(argv)
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError(f"{argv} accepted")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
