"""Tests for rendering performances to audio for the beat tracker (src/render.py) and the rendered
tracker in the meter model's data (scripts/track_meter_data.py beat_this_render, train_meter.py
--tracked in order of preference).

    python -m tests.test_render        # or: pytest tests/test_render.py

Stand-in renderer and tracker (no FluidSynth, no Beat This!) except test_fluidsynth, which renders
for real when FluidSynth and a SoundFont are found ($FLUIDSYNTH / $SOUNDFONTS, or the usual
places) and is skipped otherwise.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_tracked import build, small_release  # noqa: E402

SCALE = np.array([[60 + k, 0.5 * k + 1.0, 0.5 * k + 1.4, 80] for k in range(8)], np.float64)


def test_write_midi():
    import pretty_midi

    from src.render import write_midi
    with tempfile.TemporaryDirectory() as t:
        f = write_midi(SCALE, Path(t) / "x.mid", pedal=np.array([[1.0, 3.0]]))
        pm = pretty_midi.PrettyMIDI(str(f))
        notes = pm.instruments[0].notes
        assert [n.pitch for n in notes] == list(range(60, 68)) and abs(notes[3].start - 2.5) < 1e-3
        assert [n.velocity for n in notes] == [80] * 8
        cc = [(c.number, c.value, round(c.time, 3)) for c in pm.instruments[0].control_changes]
        assert cc == [(64, 127, 1.0), (64, 0, 3.0)], cc


def test_render_track_clocks():
    """An ASAP performance renders its own MIDI file and is tracked by its fold model; a segment
    renders its notes from just before its first one, and its beats come back on the
    performance's clock, cropped to its span."""
    from scripts.track_meter_data import PAD, render_track
    from src.render import write_midi
    with tempfile.TemporaryDirectory() as t:
        asap = Path(t) / "asap"
        (asap / "Bach").mkdir(parents=True)
        write_midi(SCALE, asap / "Bach" / "p1.mid")
        seg = SCALE.copy()
        seg[:, 1:3] += 10.0                                   # a segment from 11 s on
        perfs = [{"performance": "Bach/p1.mid", "dataset": "asap", "notes": SCALE, "gt": {}},
                 {"performance": "gangqinpu/x", "segment": 1, "segments": 2, "span": (10.5, 14.0),
                  "dataset": "gangqinpu", "notes": seg, "gt": {}},
                 {"performance": "gangqinpu/empty", "dataset": "gangqinpu", "notes": SCALE[:1], "gt": {}}]
        rendered, tracked = [], []

        def render(source, wav, key):
            rendered.append((key, source if isinstance(source, Path) else float(np.min(source[:, 1]))))
            Path(wav).write_bytes(b"x")

        def track(wav, ck):
            tracked.append(ck)
            return np.arange(0.0, 20.0, 0.5), np.arange(40) % 4 == 0   # on the render's clock
        out = render_track(perfs, render, track,
                           lambda p: "fold3" if p["dataset"] == "asap" else "final0", asap)
        assert rendered[0] == ("Bach/p1.mid", asap / "Bach" / "p1.mid")
        assert rendered[1] == ("gangqinpu/x#1", PAD)            # notes moved to start PAD s in
        assert tracked == ["fold3", "final0"] and "gangqinpu/empty" not in out
        assert np.allclose(out["Bach/p1.mid"][0], np.arange(0.0, 20.0, 0.5))
        t0 = 11.0 - PAD
        want = np.arange(0.0, 20.0, 0.5) + t0
        assert np.allclose(out["gangqinpu/x#1"][0], want[(want >= 10.5) & (want <= 14.0)])

        def broken(source, wav, key):
            raise RuntimeError("no piano")
        logs = []
        assert render_track(perfs[:1], broken, track, lambda p: "final0", asap, log=logs.append) == {}
        assert "no piano" in logs[0]


def test_track_and_train_in_order_of_preference():
    """track_meter_data --trackers beat_this_render stores every performance's rendered beats (its
    own report, the other trackers' kept); train_meter --tracked beat_this,beat_this_render gives a
    performance its recording's beats where there is one, else its render's."""
    from scripts.fetch_gangqinpu_audio import audio_path
    from scripts.track_meter_data import RENDER, main as track, tracked_path
    from scripts.train_meter import attach_tracked, main as train
    from src.beats import load_beats
    from src.meter import perf_key
    tmp = Path(tempfile.mkdtemp())
    try:
        release = small_release(tmp)
        data = build(tmp, release)
        truth = {}
        for split in ("train", "val", "test"):
            for p in pickle.load(open(data / "meter" / f"{split}_gangqinpu.pkl", "rb")):
                truth[p["performance"].split("/", 1)[1]] = p["gt"]
        audio = tmp / "audio"
        f = audio_path(audio, "train", "tr1")                 # one recording downloaded
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
        track(["--data", str(data), "--release", str(release), "--audio", str(audio), "--trackers", "beat_this",
               "--device", "cpu"], beat_this=lambda path: (truth[Path(path).stem]["t"] + 0.01, truth[Path(path).stem]["down"]))
        rendered = {}   # wav -> (key, where its first note is in the render): renders run ahead of tracking

        def render(source, wav, key):
            rendered[str(wav)] = (key, float(np.min(source[:, 1])))
            Path(wav).write_bytes(b"x")

        def tracked(wav, ck):     # the true beats, on the render's clock, 20 ms late
            key, at = rendered[str(wav)]
            gt = truth[key.split("/", 1)[1].split("#")[0]]
            first = min(float(np.min(p["notes"][:, 1])) for s in ("train", "val", "test")
                        for p in pickle.load(open(data / "meter" / f"{s}_gangqinpu.pkl", "rb"))
                        if perf_key(p) == key)
            return gt["t"] - (first - at) + 0.02, gt["down"]
        (tmp / "folds.split").write_text("x\t0\n")         # read for ASAP only (none here)
        track(["--data", str(data), "--trackers", RENDER, "--device", "cpu", "--folds", str(tmp / "folds.split")],
              render=render, track=tracked)
        rb = load_beats(tracked_path(data / "meter", "train_gangqinpu", RENDER))
        assert len(rb) == 8 and np.allclose(rb["gangqinpu/tr2"][0], truth["tr2"]["t"] + 0.02)
        assert (data / "meter" / "tracked_report_render.json").exists()
        rep = json.loads((data / "meter" / "tracked_report_render.json").read_text())["splits"]
        assert rep["train_gangqinpu"][RENDER]["beat_f"] > 0.99
        # preference: the recording's beats for tr1, the render's for the others
        perfs = pickle.load(open(data / "meter" / "train_gangqinpu.pkl", "rb"))
        n1 = attach_tracked(perfs, load_beats(tracked_path(data / "meter", "train_gangqinpu", "beat_this")), keep=True)
        n2 = attach_tracked(perfs, rb, keep=True)
        assert (n1, n2) == (1, 7)
        by = {perf_key(p): p["tracked"][0] for p in perfs}
        assert np.allclose(by["gangqinpu/tr1"], truth["tr1"]["t"] + 0.01)
        assert np.allclose(by["gangqinpu/tr2"], truth["tr2"]["t"] + 0.02)
        runs = tmp / "runs"
        tiny = ["--gangqinpu", "--no-periscope", "--steps", "2", "--eval-every", "2", "--d-model", "32", "--layers", "1",
                "--workers", "0", "--batch", "4", "--chunk", "64", "--device", "cpu", "--normal-priority",
                "--data", str(data), "--runs-dir", str(runs)]
        train(["--name", "pref", *tiny, "--tracked", f"beat_this,{RENDER}", "--tracked-share", "1"])
        cfg = json.loads((runs / "pref" / "config.json").read_text())
        assert cfg["data"]["tracked_by_source"] == {"gangqinpu": 8}, cfg["data"]
        try:
            train(["--name", "bad", *tiny, "--tracked", "beat_that"])
            raise AssertionError("an unknown tracker should be refused")
        except SystemExit as e:
            assert "beat_that" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_shards_resume_and_merge():
    """--shard K/N: every N-th performance, saved as a part; the same shard again renders only what
    is missing; --merge N joins the parts (and refuses while one is missing); several renders at
    once give the same beats as one at a time."""
    from scripts.track_meter_data import RENDER, main as track, part_path, render_track, tracked_path
    from src.beats import load_beats
    tmp = Path(tempfile.mkdtemp())
    try:
        release = small_release(tmp)
        data = build(tmp, release)
        (tmp / "folds.split").write_text("x\t0\n")
        calls = []

        def render(source, wav, key):
            calls.append(key)
            Path(wav).write_bytes(b"x")

        def tracker(wav, ck):
            return np.arange(0.0, 60.0, 0.5), np.arange(120) % 4 == 0
        common = ["--data", str(data), "--trackers", RENDER, "--device", "cpu", "--folds", str(tmp / "folds.split"),
                  "--splits", "train"]
        whole = tracked_path(data / "meter", "train_gangqinpu", RENDER)
        try:
            track([*common, "--merge", "2"])
        except SystemExit:
            raise AssertionError("nothing to merge yet should pass quietly")
        assert not whole.exists()
        track([*common, "--shard", "0/2", "--workers", "3", "--save-every", "2"], render=render, track=tracker)
        assert len(load_beats(part_path(whole, 0, 2))) == 4 and len(calls) == 4
        try:
            track([*common, "--merge", "2"])
            raise AssertionError("a missing part should be refused")
        except SystemExit as e:
            assert "part1of2" in str(e)
        calls.clear()
        track([*common, "--shard", "0/2"], render=render, track=tracker)       # done already
        assert calls == []
        track([*common, "--shard", "1/2"], render=render, track=tracker)
        track([*common, "--merge", "2"])
        assert len(load_beats(whole)) == 8
        rep = json.loads((data / "meter" / "tracked_report_render.json").read_text())
        assert rep["shards"] == 2 and rep["splits"]["train_gangqinpu"]["tracked"] == 8
        try:
            track(["--data", str(data), "--trackers", f"pm2s,{RENDER}", "--shard", "0/2"])
            raise AssertionError("--shard with other trackers should be refused")
        except SystemExit as e:
            assert "--shard" in str(e)
        perfs = pickle.load(open(data / "meter" / "train_gangqinpu.pkl", "rb"))
        one = render_track(perfs, render, tracker, lambda p: "final0")
        many = render_track(perfs, render, tracker, lambda p: "final0", workers=4)
        assert one.keys() == many.keys() and all(np.allclose(one[k][0], many[k][0]) for k in one)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _found_fluidsynth():
    exe = os.environ.get("FLUIDSYNTH") or shutil.which("fluidsynth") or r"C:\tools\fluidsynth\bin\fluidsynth.exe"
    fonts = os.environ.get("SOUNDFONTS") or r"C:\Program Files\MuseScore 4\sound\MS Basic.sf3"
    fonts = [f for f in fonts.split(os.pathsep) if Path(f).exists()]
    return (exe, fonts) if exe and Path(exe).exists() and fonts else (None, None)


def test_fluidsynth():
    """A real render (skipped without FluidSynth and a SoundFont): audible, as long as the notes,
    each note's onset heard where it was played."""
    exe, fonts = _found_fluidsynth()
    if exe is None:
        print("  (skipped: no FluidSynth / SoundFont)")
        return
    import soundfile as sf

    from src.render import Renderer
    r = Renderer(fonts, exe)
    with tempfile.TemporaryDirectory() as t:
        wav = Path(t) / "x.wav"
        assert r(SCALE, wav, "k") == Path(fonts[0])
        x, sr = sf.read(str(wav))
        x = x.mean(axis=1) if x.ndim == 2 else x
        assert sr == 22050 and 4.5 < len(x) / sr < 8.0, len(x) / sr
        hop = int(0.01 * sr)
        env = np.array([np.abs(x[i:i + hop]).max() for i in range(0, len(x) - hop, hop)])
        # silence until the first note, then an attack where each note was played (the one before
        # still rings, so an attack is a rise of half again, not a jump from silence)
        assert env[: int(0.95 / 0.01)].max() < 1e-3 and env[100:103].max() > 0.05 * env.max()
        rise = np.nonzero((env[1:] > 1.5 * env[:-1] + 1e-4) & (env[1:] > 0.05 * env.max()))[0] * 0.01
        heard = [np.min(np.abs(rise - on)) < 0.03 for on in SCALE[:, 1]]
        assert sum(heard) >= 6, (rise, heard)
        assert not (Path(t) / "x.mid").exists()              # its temporary MIDI file removed


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
