"""Tests for the system-level comparison with Beyer & Dai (Baseline 3).

    python -m tests.test_system_eval        # or: pytest tests/test_system_eval.py

Covers the inputs (scripts/system_inputs.py: real transcriptions shifted onto the ASAP timeline and
cut to their piece; the manifest on the local data build), our systems on them (scripts/system_scores.py
with a tiny model, the tracked beats given to a beat model; v2 with tiny models, the rendered-MIDI
beats on clean; a failed score's tokens and error kept), the aggregation and tables of
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
from tests.batch_env import batch_env  # noqa: E402
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
    real = (data / "test_transcribed" / "report.json").exists()
    maestro, transcribed = ROOT / "data" / "raw" / "maestro", ROOT / "data" / "transcribed_transkun"
    if real and not ((maestro / "maestro-v3.0.0.csv").exists() and transcribed.exists()):
        print("  (skipped: test_transcribed exists but data/raw/maestro or data/transcribed_transkun is missing)")
        return
    with tempfile.TemporaryDirectory() as t:
        entries = system_inputs.build(data, asap, transcribed, maestro, Path(t))
        clean = [e for e in entries if e["condition"] == "clean"]
        assert len(clean) == len({e["performance"] for e in clean}) >= 1
        assert all(Path(e["midi"]).exists() and Path(e["reference"]).name == "xml_score.musicxml" for e in clean)
        if real:  # every real input written (cut to its piece), its performance a clean one too
            got = [e for e in entries if e["condition"] == "real"]
            assert got and all(Path(e["midi"]).exists() and e["notes"] > 0 for e in got), got[:2]
            assert {e["performance"] for e in got} <= {e["performance"] for e in clean}
        else:
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


def test_system_scores_v2_and_failures():
    """--v2: V2Pipeline on every input, the tracked beats on real, the rendered-MIDI beats
    (beats_synth.npz) on clean; --midi-beats synth gives a seq2seq beat model those too (system
    <run>_<mode>_sb). A piece whose score cannot be written keeps its tokens and its error."""
    from tests.test_v2_system import make_v2
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        make_performance(tmp / "perf.mid", n=80)
        perf = "X/Y/perf.mid"
        manifest = [{"condition": c, "performance": perf, "slug": "X_Y_perf", "midi": str(tmp / "perf.mid"),
                     "reference": "ref.musicxml"} for c in ("clean", "real")]
        (tmp / "manifest.json").write_text(json.dumps(manifest))
        (tmp / "tracked").mkdir()
        save_beats(beats_path(tmp / "tracked", "tracked"), {perf: (0.5 + 0.5 * np.arange(40), np.arange(40) % 4 == 0)})
        save_beats(tmp / "synth.npz", {perf: (0.4 + 0.5 * np.arange(41), np.arange(41) % 3 == 0)})
        make_v2(tmp / "runs")
        common = ["--runs-dir", str(tmp / "runs"), "--vocab", str(ROOT / "data" / "processed" / "vocab.json"),
                  "--manifest", str(tmp / "manifest.json"), "--out", str(tmp / "out"), "--device", "cpu",
                  "--tracked", str(tmp / "tracked"), "--synth-beats", str(tmp / "synth.npz")]
        # --beat-decode before the labeller's decoding is tuned: refused, nothing written under the _bd name
        try:
            system_scores.main(["--v2", "nl+m", "--beat-decode", "--conditions", "real"] + common)
        except SystemExit as e:
            assert "beat_decode.json" in str(e), e
        else:
            raise AssertionError("an untuned --beat-decode should be refused")
        assert not (tmp / "out" / "v2_nl_m_bd").exists()
        (tmp / "runs" / "nl" / "beat_decode.json").write_text(json.dumps({"best": {"weight": 1.0, "anchor": 0.1}}))
        system_scores.main(["--v2", "nl+m", "--beat-decode", "--conditions", "real"] + common)
        bd = json.loads((tmp / "out" / "v2_nl_m_bd" / "summary.json").read_text())
        assert bd["beat_decode"]["weight"] == 1.0 and len(bd["pieces"]) == 1
        (tmp / "runs" / "nl" / "beat_decode.json").unlink()
        shutil.rmtree(tmp / "out" / "v2_nl_m_bd")
        system_scores.main(["--v2", "nl+m"] + common)
        summary = json.loads((tmp / "out" / "v2_nl_m" / "summary.json").read_text())
        assert summary["beat_decode"] is None
        pieces = {p["condition"]: p for p in summary["pieces"]}
        assert summary["v2"] == "nl+m" and summary["midi_beats"] == "synth" and summary["mode"] is None
        assert pieces["real"]["beats"] == "given: 40 beats, 10 downbeats", pieces["real"]
        assert pieces["clean"]["beats"] == "given: 41 beats, 14 downbeats", pieces["clean"]
        for p in pieces.values():
            assert p["musicxml"] is None or Path(p["musicxml"]).exists(), p
            assert p["bars"] > 0 and p["notes_kept"] >= 0 and p["emptied_bars"] >= 0, p
        (tmp / "runs" / "beat").mkdir()
        make_checkpoint(tmp / "runs" / "beat", n_features=15, beat_features=True)
        system_scores.main(["--run", "beat", "--mode", "none", "--midi-beats", "synth"] + common)
        sb = json.loads((tmp / "out" / "beat_none_sb" / "summary.json").read_text())
        assert {p["condition"]: p["beats"] for p in sb["pieces"]}["clean"] == "given: 41 beats, 14 downbeats"
        # a score that cannot be written: its tokens and error are kept, a stale score removed
        from src.pipeline import Result, V2Pipeline
        real_run = V2Pipeline.run
        V2Pipeline.run = lambda self, *a, **k: Result(["bar", "ts_4/4"], {}, {"export": {"emptied_bars": []}},
                                                      ["MusicXML: KeyError: 3"])
        try:
            system_scores.main(["--v2", "nl+m", "--conditions", "real"] + common)
        finally:
            V2Pipeline.run = real_run
        d = tmp / "out" / "v2_nl_m" / "real"
        assert not (d / "X_Y_perf.musicxml").exists()
        assert (d / "X_Y_perf.tokens.txt").read_text() == "bar ts_4/4"
        assert (d / "X_Y_perf.error.txt").read_text() == "MusicXML: KeyError: 3"
        # --only-missing: only that piece again; the summary keeps the clean one
        system_scores.main(["--v2", "nl+m", "--only-missing"] + common)
        summary = json.loads((tmp / "out" / "v2_nl_m" / "summary.json").read_text())
        got = {p["condition"]: p for p in summary["pieces"]}
        assert set(got) == {"clean", "real"} and got["real"]["musicxml"] and (d / "X_Y_perf.musicxml").exists(), got
        assert not (d / "X_Y_perf.error.txt").exists() and not (d / "X_Y_perf.tokens.txt").exists()
        assert got["clean"] == pieces["clean"]  # not run again
        # --clean-beats audio: clean MIDI with the recording's beats (here the tracked ones), system v2_nl_m_ab
        system_scores.main(["--v2", "nl+m", "--conditions", "clean", "--clean-beats", "audio"] + common)
        ab = json.loads((tmp / "out" / "v2_nl_m_ab" / "summary.json").read_text())
        assert ab["clean_beats"] == "audio" and [p["beats"] for p in ab["pieces"]] == ["given: 40 beats, 10 downbeats"]
        try:
            system_scores.main(["--run", "beat", "--clean-beats", "audio"] + common)
        except SystemExit:
            pass
        else:
            raise AssertionError("--clean-beats audio is for v2 only")
        # --clean-beats pm2s --real-beats pm2s: the beats from the notes on both (PM2S stood in for:
        # tests/test_pm2s.py runs the model itself), system v2_nl_m_pm_rpm
        real_pm2s = V2Pipeline.pm2s_beats
        seen = []

        def stand_in(self, notes):
            seen.append(len(notes))
            return 0.5 + 0.5 * np.arange(36), np.arange(36) % 3 == 0
        V2Pipeline.pm2s_beats = stand_in
        try:
            system_scores.main(["--v2", "nl+m", "--clean-beats", "pm2s", "--real-beats", "pm2s"] + common)
        finally:
            V2Pipeline.pm2s_beats = real_pm2s
        pm = json.loads((tmp / "out" / "v2_nl_m_pm_rpm" / "summary.json").read_text())
        assert pm["clean_beats"] == "pm2s" and pm["real_beats"] == "pm2s" and len(seen) == 2
        assert [p["beats"] for p in pm["pieces"]] == ["PM2S on the notes: 36 beats, 12 downbeats"] * 2, pm["pieces"]
        try:
            system_scores.main(["--run", "beat", "--clean-beats", "pm2s"] + common)
        except SystemExit:
            pass
        else:
            raise AssertionError("--clean-beats pm2s is for v2 only")
        (tmp / "out" / "v2_nl_m_pm_rpm" / "summary.json").unlink()
        (tmp / "out" / "v2_nl_m_ab" / "summary.json").unlink()
        assert beyer_dai.all_systems(tmp / "out") == ["beat_none_sb", "v2_nl_m"]
        (tmp / "out" / "beyer_dai").mkdir()
        (tmp / "out" / "beyer_dai" / "summary.json").write_text("{}")
        (tmp / "out" / "junk").mkdir()  # no summary.json: not a system
        assert beyer_dai.all_systems(tmp / "out") == ["beyer_dai", "beat_none_sb", "v2_nl_m"]
        try:
            system_scores.main(["--v2", "nl"] + common)
        except SystemExit:
            pass
        else:
            raise AssertionError("--v2 without +<meter run> should be refused")


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
        env = batch_env(SUBMITS=str(tmp / "jobs.txt"), BD=str(bd))
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


def test_checkpoint_allow_list():
    """PyTorch's weights_only loading refuses pickled classes: their checkpoint's own classes (a
    module named like theirs, e.g. config.MyModelConfig) are allowed and it loads; a class from
    anywhere else stops the load."""
    import torch
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "config.py").write_text("class MyModelConfig:\n    def __init__(self, d=4):\n        self.d = d\n")
        (tmp / "elsewhere.py").write_text("class Thing:\n    pass\n")
        sys.path.insert(0, str(tmp))
        try:
            import config
            import elsewhere
            torch.save({"hyper_parameters": {"enc": config.MyModelConfig(7)}, "state_dict": {}}, tmp / "ok.ckpt")
            torch.save({"x": elsewhere.Thing()}, tmp / "bad.ckpt")
            assert beyer_dai.allow_their_classes(str(tmp / "ok.ckpt")) == ["config.MyModelConfig"]
            assert torch.load(tmp / "ok.ckpt", weights_only=True)["hyper_parameters"]["enc"].d == 7
            try:
                beyer_dai.allow_their_classes(str(tmp / "bad.ckpt"))
            except SystemExit as e:
                assert "elsewhere.Thing" in str(e), e
            else:
                raise AssertionError("a foreign class was allowed")
        finally:
            sys.path.remove(str(tmp))
            sys.modules.pop("config", None)
            sys.modules.pop("elsewhere", None)


def test_reference_row():
    """The reference against itself is scored like a system (the metrics' floor) but never
    removes performances from the common set."""
    rated, entries = rated_stub()
    for e in entries:
        rated[(beyer_dai.REFERENCE, e["condition"], e["performance"])] = \
            {"muster": None, "sim": None, "error": "x"} if e["performance"] == "p1" else rated[("A", "clean", "p1")]
    res = beyer_dai.aggregate(rated, entries, ["A", "B", beyer_dai.REFERENCE])
    assert res["clean"]["A"]["common"]["n"] == 2  # p1, p2 still common although the reference failed on p1
    assert res["clean"][beyer_dai.REFERENCE]["all"]["n"] == 2
    assert beyer_dai.pred_path(Path("o"), beyer_dai.REFERENCE, {"reference": "r.musicxml"}) == Path("r.musicxml")


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
