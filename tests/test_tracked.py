"""Tests for real tracker output in the meter model's training and testing: the recordings'
download (scripts/fetch_gangqinpu_audio.py), the trackers run on the data
(scripts/track_meter_data.py), the real / simulated mix in training (train_meter.py --tracked) and
the real beats at test (eval_meter.py --real-tracked).

    python -m tests.test_tracked        # or: pytest tests/test_tracked.py

No network and no real tracker: a stand-in HTTP opener, and stand-in trackers that return the
ground-truth beats moved by a known amount (so what was stored can be checked).
"""
from __future__ import annotations

import io
import json
import pickle
import shutil
import sys
import tempfile
import urllib.error
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_gangqinpu import beats_of, row_of, write_release  # noqa: E402


def small_release(tmp: Path) -> Path:
    rows = {"train": [row_of(f"tr{k}", beats_of(40, ts, sec_per_beat=0.4 + 0.05 * k))
                      for k, ts in enumerate(["3/4", "4/4", "6/8", "2/4"] * 2)],
            "validation": [row_of(f"va{k}", beats_of(30, ts), split="val", source_page_url=f"https://example.invalid/p{k}.htm")
                           for k, ts in enumerate(["3/4", "4/4"])],
            "test": [row_of(f"te{k}", beats_of(30, ts), split="test") for k, ts in enumerate(["6/8", "4/4"])]}
    write_release(tmp / "release", rows)
    return tmp / "release"


class Response(io.BytesIO):
    def __init__(self, data: bytes, kind: str):
        super().__init__(data)
        self.headers = {"Content-Type": kind}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_audio():
    """Every val / test recording and a fixed draw of training ones; HTML and short answers refused
    and retried; what is there skipped on the next run; a log line per attempt; the page as Referer."""
    from scripts.fetch_gangqinpu_audio import USER_AGENT, audio_path, main, plan
    tmp = Path(tempfile.mkdtemp())
    try:
        release = small_release(tmp)
        jobs = plan(release, 3)
        assert [s for s, _ in jobs].count("val") == 2 and [s for s, _ in jobs].count("test") == 2
        train = [r["sample_id"] for s, r in jobs if s == "train"]
        assert len(train) == 3 and train == [r["sample_id"] for s, r in plan(release, 3) if s == "train"]  # fixed draw
        seen = []

        def opener(req, timeout):
            seen.append((req.full_url, req.get_header("User-agent"), req.get_header("Referer")))
            if "va0" in req.full_url:
                return Response(b"<html>gone</html>", "text/html")
            if "te1" in req.full_url and sum("te1" in u for u, _, _ in seen) == 1:
                raise urllib.error.URLError("timeout")             # the first try fails, the second works
            return Response(b"ID3" + b"x" * 4000, "audio/mpeg")
        out = tmp / "audio"
        main(["--release", str(release), "--out", str(out), "--train", "3", "--retries", "2"], opener=opener,
             sleep=lambda s: None)
        s = json.loads((out / "fetch_summary.json").read_text())
        assert s == {"planned": 7, "fetched": 6, "already_there": 0, "failed": 1}, s
        assert not audio_path(out, "val", "va0").exists() and audio_path(out, "test", "te1").exists()
        assert all(ua == USER_AGENT for _, ua, _ in seen)
        refs = {u.rsplit("/", 1)[1].split(".")[0]: ref for u, _, ref in seen}
        assert refs["va1"] == "https://example.invalid/p1.htm" and refs["te0"] is None   # the page, when known
        log = [json.loads(ln) for ln in (out / "download_log.jsonl").read_text().splitlines()]
        assert len(log) == 7 and sum(not x["ok"] for x in log) == 1 and "HTML" in [x for x in log if not x["ok"]][0]["error"]
        n = len(seen)
        main(["--release", str(release), "--out", str(out), "--train", "3", "--retries", "1"], opener=opener,
             sleep=lambda s: None)
        assert len(seen) == n + 1                                  # only the missing one is tried again
        assert json.loads((out / "fetch_summary.json").read_text())["already_there"] == 6
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build(tmp: Path, release: Path) -> Path:
    from scripts.build_meter_data import main as build_meter
    data = tmp / "processed"
    build_meter(["--data", str(data), "--gangqinpu", str(release), "--gangqinpu-only"])
    return data


def test_track_meter_data():
    """Beat This! (stand-in) on the recordings that are there, PM2S (stand-in) on every
    performance's notes; each segment keeps the beats in its span; the report compares them, and
    like for like on the segments both have."""
    from scripts.fetch_gangqinpu_audio import audio_path
    from scripts.track_meter_data import main, tracked_path
    from src.beats import load_beats
    from src.meter import perf_key
    tmp = Path(tempfile.mkdtemp())
    try:
        release = small_release(tmp)
        data = build(tmp, release)
        audio = tmp / "audio"
        for split, sid in (("test", "te0"), ("train", "tr1")):
            audio_path(audio, split, sid).parent.mkdir(parents=True, exist_ok=True)
            audio_path(audio, split, sid).write_bytes(b"x")
        truth = {}
        for split in ("train", "val", "test"):
            for p in pickle.load(open(data / "meter" / f"{split}_gangqinpu.pkl", "rb")):
                truth[p["performance"].split("/", 1)[1]] = p["gt"]

        def beat_this(path):    # the true beats, 10 ms late
            gt = truth[Path(path).stem]
            return gt["t"] + 0.01, gt["down"]

        def pm2s(notes):       # every other true beat (half tempo), found from the notes' span
            n = np.asarray(notes)
            for gt in truth.values():
                if abs(gt["t"][0] - n[:, 1].min()) < 1.0 and abs(gt["t"][-1] - n[:, 1].max()) < 1.0:
                    return gt["t"][::2], gt["down"][::2]
            return np.zeros(0), np.zeros(0, bool)
        main(["--data", str(data), "--release", str(release), "--audio", str(audio), "--device", "cpu"],
             beat_this=beat_this, pm2s=pm2s)
        bt = load_beats(tracked_path(data / "meter", "test_gangqinpu", "beat_this"))
        pm = load_beats(tracked_path(data / "meter", "test_gangqinpu", "pm2s"))
        assert list(bt) == ["gangqinpu/te0"] and set(pm) == {"gangqinpu/te0", "gangqinpu/te1"}
        assert np.allclose(bt["gangqinpu/te0"][0], truth["te0"]["t"] + 0.01)
        assert "gangqinpu/tr1" in load_beats(tracked_path(data / "meter", "train_gangqinpu", "beat_this"))
        rep = json.loads((data / "meter" / "tracked_report.json").read_text())["splits"]["test_gangqinpu"]
        assert rep["beat_this"]["segments"] == 1 and rep["pm2s"]["segments"] == 2
        assert rep["beat_this"]["beat_f"] > 0.99 and rep["both"]["pm2s"]["segments"] == 1
        assert rep["both"]["pm2s"]["wrong_octave"] == 1.0           # half tempo, seen as such
        perfs = pickle.load(open(data / "meter" / "test_gangqinpu.pkl", "rb"))
        assert all(perf_key(p) in pm for p in perfs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mix_in_training_and_real_at_test():
    """with_tracked: about --tracked-share of the draws get the real beats, the others none (the
    simulated tracker); train_meter --tracked attaches them and records how many; a missing file
    only warns; eval_meter --real-tracked tests on the stored Beat This! beats (and their
    performances only), and refuses to go with --tracker pm2s."""
    from scripts.eval_meter import main as evaluate
    from scripts.track_meter_data import main as track
    from scripts.train_meter import main as train, with_tracked
    perf = {"performance": "x", "beats": None, "tracked": (np.arange(4.0), np.zeros(4, bool))}
    rng = np.random.default_rng(0)
    got = [with_tracked(perf, rng, 0.5)["beats"] is not None for _ in range(2000)]
    assert 0.45 < np.mean(got) < 0.55
    assert with_tracked(perf, rng, 0.0)["beats"] is None and with_tracked({"performance": "y"}, rng, 1.0) == {"performance": "y"}
    tmp = Path(tempfile.mkdtemp())
    try:
        release = small_release(tmp)
        data = build(tmp, release)
        truth = {}
        for split in ("train", "val", "test"):
            for p in pickle.load(open(data / "meter" / f"{split}_gangqinpu.pkl", "rb")):
                truth[p["performance"].split("/", 1)[1]] = p["gt"]
        audio = tmp / "audio"
        from scripts.fetch_gangqinpu_audio import audio_path
        for split, prefix in (("train", "tr"), ("test", "te")):
            for k in range(8 if split == "train" else 2):
                f = audio_path(audio, split, f"{prefix}{k}")
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"x")
        track(["--data", str(data), "--release", str(release), "--audio", str(audio), "--trackers", "beat_this",
               "--device", "cpu"], beat_this=lambda path: (truth[Path(path).stem]["t"], truth[Path(path).stem]["down"]))
        runs = tmp / "runs"
        common = ["--data", str(data), "--runs-dir", str(runs)]
        tiny = ["--gangqinpu", "--no-periscope", "--steps", "4", "--eval-every", "2", "--d-model", "32", "--layers", "1",
                "--workers", "0", "--batch", "4", "--chunk", "64", "--device", "cpu", "--normal-priority"]
        train(["--name", "mix", *tiny, "--tracked", "beat_this", "--tracked-share", "0.5", *common])
        cfg = json.loads((runs / "mix" / "config.json").read_text())
        assert cfg["data"]["tracked_by_source"] == {"gangqinpu": 8} and cfg["args"]["tracked"] == "beat_this"
        train(["--name", "nopm", *tiny, "--tracked", "pm2s", *common])          # none stored: warns, trains on
        assert json.loads((runs / "nopm" / "config.json").read_text())["data"]["tracked_by_source"] == {"gangqinpu": 0}
        evaluate(["--run", "mix", "--conditions", "gangqinpu", "--real-tracked", "--device", "cpu", *common])
        res = json.loads((runs / "mix" / "eval_meter.json").read_text())
        assert res["gangqinpu_real"]["tracker"] == "beat_this_real" and res["gangqinpu_real"]["performances"] == 2
        assert abs(res["gangqinpu_real"]["tracker_beat_f"] - 1.0) < 1e-9   # the stand-in gave the true beats
        try:
            evaluate(["--run", "mix", "--conditions", "gangqinpu", "--real-tracked", "--tracker", "pm2s", *common])
            raise AssertionError("--real-tracked with --tracker pm2s should be refused")
        except SystemExit as e:
            assert "real-tracked" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_aux_features():
    """A second tracker's beats seen from the first one's: agreeing, beating twice as fast, on the
    off-beats (the first one off by half a beat), none at all; augmentation stretches them too."""
    from src.meter import N_AUX_FEATURES, N_BEAT_FEATURES, Augment, aux_features, beat_features
    t = 0.5 * np.arange(16)
    down = np.arange(16) % 4 == 0
    same = aux_features(t, (t + 0.01, down))
    assert same.shape == (16, N_AUX_FEATURES) and (same[:, 0] == 1).all() and (same[:, 1] == 1).all()
    assert (same[:, 2] == 0).all() and same[0, 4] == 1 and same[1, 4] == 0
    fast = aux_features(t, (0.25 * np.arange(32), np.zeros(32, bool)))
    assert (fast[:-1, 1] == 1).all() and np.allclose(fast[:-1, 2], 1.0) and np.allclose(fast[:-1, 3], 0.5)
    off = aux_features(t, (t + 0.25, down))                  # its beats in the middle of ours
    assert (off[:, 1] == 0).all() and np.allclose(off[:, 3], 0.5)
    none = aux_features(t, None)
    assert (none == 0).all()
    empty = aux_features(t, (np.zeros(0), np.zeros(0, bool)))
    assert (empty[:, 0] == 1).all() and (empty[:, 1:] == 0).all()
    notes = np.array([[60, 0.0, 0.3, 80], [64, 1.0, 1.3, 80]], np.float32)
    f = beat_features(notes, t, down, (t + 0.25, down))
    assert np.allclose(f[:, N_BEAT_FEATURES:], off)
    perf = {"notes": notes, "gt": {"t": t, "down": down}, "beats": (t, down), "aux": (t + 0.25, down)}
    out = Augment(prob=1.0)(perf, np.random.default_rng(0))
    s = out["beats"][0][1] / t[1]
    assert np.allclose(out["aux"][0], (t + 0.25) * s)


def test_fit_from_stored():
    """fit_pm2s_noise.py --from-stored beat_this: Beat This!'s errors on Gangqinpu fitted from its
    stored real beats (here: a stand-in on the off-beats for a third of the performances, so the
    off-beat rate comes out measured); too few tracked performances -> no profile written."""
    from scripts.fit_pm2s_noise import main as fit
    from scripts.track_meter_data import main as track
    tmp = Path(tempfile.mkdtemp())
    try:
        release = small_release(tmp)
        data = build(tmp, release)
        truth = {}
        for split in ("train", "val", "test"):
            for p in pickle.load(open(data / "meter" / f"{split}_gangqinpu.pkl", "rb")):
                truth[p["performance"].split("/", 1)[1]] = p["gt"]
        audio = tmp / "audio"
        from scripts.fetch_gangqinpu_audio import audio_path
        for k in range(8):
            f = audio_path(audio, "train", f"tr{k}")
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"x")

        def beat_this(path):    # on the off-beats for tr0, tr3, tr6
            gt = truth[Path(path).stem]
            k = int(Path(path).stem[2:])
            if k % 3 == 0:
                mids = (gt["t"][:-1] + gt["t"][1:]) / 2
                return mids, np.zeros(len(mids), bool)
            return gt["t"], gt["down"]
        track(["--data", str(data), "--release", str(release), "--audio", str(audio), "--trackers", "beat_this",
               "--device", "cpu"], beat_this=beat_this)
        out = tmp / "bt_gq.json"
        fit(["--data", str(data), "--from-stored", "beat_this", "--min-beats", "16", "--min-performances", "5",
             "--out", str(out)])
        got = json.loads(out.read_text())
        assert got["tracker"] == "beat_this" and got["fit_on"]["performances"] == 8
        assert abs(got["p_offbeat"] - 3 / 8) < 1e-9, got["p_offbeat"]
        none = tmp / "none.json"
        fit(["--data", str(data), "--from-stored", "beat_this", "--min-beats", "16", "--min-performances", "50",
             "--out", str(none)])
        assert not none.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_both_trackers_end_to_end():
    """A model that reads both trackers (train_meter.py --aux pm2s): PM2S's real beats attached and
    mixed in; Gangqinpu simulated with its own fitted profile (--beat-noise-gangqinpu, a missing
    file only warns); tested with PM2S's beats as evidence (eval_meter.py: "aux" in its results,
    tuning included), never with --tracker pm2s; the pipeline gives it PM2S's beats of the notes."""
    import os
    from scripts.eval_meter import main as evaluate
    from scripts.fit_pm2s_noise import main as fit
    from scripts.track_meter_data import main as track
    from scripts.train_meter import main as train
    from src.meter import N_ALL_FEATURES, build_meter_model, uses_aux
    from tests.test_pm2s import random_weights
    tmp = Path(tempfile.mkdtemp())
    old_env = os.environ.get("PM2S_WEIGHTS")
    try:
        release = small_release(tmp)
        data = build(tmp, release)
        truth = {}
        for split in ("train", "val", "test"):
            for p in pickle.load(open(data / "meter" / f"{split}_gangqinpu.pkl", "rb")):
                truth[p["performance"].split("/", 1)[1]] = p["gt"]

        def pm2s(notes):     # half tempo, from the notes' span
            n = np.asarray(notes)
            for gt in truth.values():
                if abs(gt["t"][0] - n[:, 1].min()) < 1.0 and abs(gt["t"][-1] - n[:, 1].max()) < 1.0:
                    return gt["t"][::2], gt["down"][::2]
            return np.zeros(0), np.zeros(0, bool)
        track(["--data", str(data), "--release", str(release), "--trackers", "pm2s", "--device", "cpu"], pm2s=pm2s)
        noise_pm = tmp / "pm.json"
        from scripts import fit_pm2s_noise

        class StandIn:
            weights = "stand-in"

            def __init__(self, *a, **k):
                pass

            def __call__(self, notes):
                return pm2s(notes)
        real = fit_pm2s_noise.PM2SBeats
        fit_pm2s_noise.PM2SBeats = StandIn
        try:
            fit(["--data", str(data), "--min-beats", "16", "--out", str(noise_pm)])
        finally:
            fit_pm2s_noise.PM2SBeats = real
        runs = tmp / "runs"
        common = ["--data", str(data), "--runs-dir", str(runs)]
        train(["--name", "both", "--gangqinpu", "--no-periscope", "--steps", "4", "--eval-every", "2", "--d-model", "32",
               "--layers", "1", "--workers", "0", "--batch", "4", "--chunk", "64", "--device", "cpu", "--normal-priority",
               "--aux", "pm2s", "--aux-noise", str(noise_pm), "--beat-noise-gangqinpu", str(tmp / "missing.json"),
               *common])
        cfg = json.loads((runs / "both" / "config.json").read_text())
        assert cfg["model"]["n_features"] == N_ALL_FEATURES and uses_aux(cfg["model"])
        assert cfg["data"]["tracked_aux_by_source"] == {"gangqinpu": 8}
        os.environ["PM2S_WEIGHTS"] = str(random_weights(tmp / "pm2s.pth"))
        evaluate(["--run", "both", "--conditions", "gangqinpu", "--device", "cpu", "--tune", "--tune-max", "2", *common])
        res = json.loads((runs / "both" / "eval_meter.json").read_text())
        assert res["gangqinpu"]["aux"] == "pm2s"
        try:
            evaluate(["--run", "both", "--conditions", "gangqinpu", "--device", "cpu", "--tracker", "pm2s", *common])
            raise AssertionError("a model reading PM2S as evidence should not be tested with --tracker pm2s")
        except SystemExit as e:
            assert "second tracker" in str(e)
        # the pipeline: a meter model that reads PM2S gets its beats of the notes, one that does not, none
        import torch
        from src.pipeline import V2Pipeline
        ck = torch.load(runs / "both" / "best.pt", map_location="cpu", weights_only=False)
        calls = []
        pipe = V2Pipeline.__new__(V2Pipeline)
        pipe.meter, pipe.device = build_meter_model(ck["config"]).eval(), torch.device("cpu")
        pipe.meter.load_state_dict(ck["model"])
        from src.meter import DecodeConfig
        pipe.decode_cfg = DecodeConfig()
        pipe.pm2s_beats = lambda notes: calls.append(len(notes)) or (0.5 * np.arange(20), np.arange(20) % 4 == 0)
        notes = np.stack([np.full(40, 60.0), 0.25 * np.arange(40), 0.25 * np.arange(40) + 0.2, np.full(40, 80.0)], 1)
        g = pipe.grid(notes.astype(np.float32), (0.5 * np.arange(20), np.arange(20) % 4 == 0))
        assert calls == [40] and len(g["bars"]) > 0
        cfg_plain = dict(ck["config"], n_features=N_ALL_FEATURES - 5)
        pipe.meter = build_meter_model(cfg_plain).eval()
        pipe.grid(notes.astype(np.float32), (0.5 * np.arange(20), np.arange(20) % 4 == 0))
        assert calls == [40]                                   # no second tracker for this one
    finally:
        if old_env is None:
            os.environ.pop("PM2S_WEIGHTS", None)
        else:
            os.environ["PM2S_WEIGHTS"] = old_env
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
