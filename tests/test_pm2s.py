"""Tests for PM2S's beat tracker as a beat source (src/pm2s_beats.py and what uses it).

    python -m tests.test_pm2s        # or: pytest tests/test_pm2s.py

No download: the model runs on random weights of its architecture (saved to a temporary file), so
these check the plumbing, not the beats. Covers the post-processing on known probabilities (the
gap filling, merged downbeats), the weights' hash check, the tracker's output (sorted, within the
notes, downbeats flagged on beats), V2Pipeline's beat_source, system_scores.py's --clean-beats pm2s
/ --real-beats pm2s names and the batch options.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import pm2s_beats as pb  # noqa: E402


def test_complement_fills_gaps():
    beats = np.arange(0.0, 10.0, 0.5)
    gappy = np.delete(beats, [5, 12, 13])                      # one beat missing, then two
    out = pb.complement(gappy)
    assert len(out) == len(beats) and np.allclose(out, beats)
    assert np.allclose(pb.complement(beats), beats)            # nothing to fill


def test_postprocess():
    onsets = np.arange(0.0, 8.0, 0.25)                         # 32 notes, a beat every other note
    beat_p = np.where(np.arange(32) % 2 == 0, 0.9, 0.05)
    down_p = np.where(np.arange(32) % 8 == 0, 0.9, 0.05)
    beats, downs = pb.postprocess(beat_p, down_p, onsets)
    assert np.allclose(beats, onsets[::2]) and np.allclose(downs, onsets[::8])
    # a downbeat no beat was found at is merged in as a beat
    beat_p2 = beat_p.copy()
    beat_p2[8] = 0.05
    beats2, _ = pb.postprocess(beat_p2, down_p, onsets)
    assert np.any(np.isclose(beats2, onsets[8]))
    # too few beats to fill: returned as they are
    few, _ = pb.postprocess(np.array([0.9, 0.1]), np.array([0.9, 0.1]), np.array([0.0, 0.3]))
    assert np.allclose(few, [0.0])


def random_weights(path: Path) -> Path:
    import torch
    torch.manual_seed(0)
    torch.save(pb._model().state_dict(), path)
    return path


def notes_of(n: int = 120) -> np.ndarray:
    rng = np.random.default_rng(0)
    on = np.cumsum(rng.uniform(0.1, 0.4, n))
    return np.stack([rng.integers(40, 80, n), on, on + 0.3, rng.integers(40, 100, n)], axis=1).astype(np.float32)


def test_tracker_on_random_weights():
    with tempfile.TemporaryDirectory() as t:
        tr = pb.PM2SBeats(random_weights(Path(t) / "w.pth"))
        notes = notes_of()
        b, d, onsets = tr.probabilities(notes[::-1].copy())     # unsorted input is sorted first
        assert len(b) == len(d) == len(notes) and np.all(np.diff(onsets) >= 0)
        assert np.all((b >= 0) & (b <= 1)) and np.all((d >= 0) & (d <= 1))
        times, down = tr(notes)
        assert down.dtype == bool and len(times) == len(down)
        assert np.all(np.diff(times) > 0) or len(times) < 2
        if len(times):
            assert times[0] >= notes[:, 1].min() - 1e-6 and times[-1] <= notes[:, 1].max() + 1e-6
        e_t, e_d = tr(notes[:1])                                 # one note: nothing to track
        assert len(e_t) == 0 and len(e_d) == 0


def test_weights_hash_check():
    with tempfile.TemporaryDirectory() as t:
        target = Path(t) / "sub" / "w.pth"
        real = pb.urllib.request.urlretrieve
        pb.urllib.request.urlretrieve = lambda url, dst: Path(dst).write_bytes(b"not the weights")
        try:
            pb.fetch_weights(target)
            raise AssertionError("a file with the wrong hash must be refused")
        except RuntimeError as e:
            assert "SHA-256" in str(e) and not target.exists() and not target.with_name("w.pth.part").exists()
        finally:
            pb.urllib.request.urlretrieve = real
        target.write_bytes(b"x")                                 # already there: used as it is
        assert pb.fetch_weights(target) == target
    import os
    old = os.environ.get("PM2S_WEIGHTS")
    os.environ["PM2S_WEIGHTS"] = "/somewhere/mine.pth"
    try:
        assert pb.default_weights_path() == Path("/somewhere/mine.pth")
    finally:
        if old is None:
            del os.environ["PM2S_WEIGHTS"]
        else:
            os.environ["PM2S_WEIGHTS"] = old


def test_pipeline_beat_source():
    """V2Pipeline.run(beat_source=...): pm2s takes the beats from the notes (no Beat This!),
    an unknown source is refused before anything runs."""
    from src.pipeline import BEAT_SOURCES, V2Pipeline
    assert BEAT_SOURCES == ("beat_this", "pm2s")
    pipe = V2Pipeline.__new__(V2Pipeline)
    try:
        pipe.run("x.mid", tempfile.gettempdir(), beat_source="madmom")
        raise AssertionError("unknown beat source should be refused")
    except ValueError as e:
        assert "beat_source" in str(e)


def test_system_names_and_batch_options():
    from scripts.system_scores import system_name
    assert system_name(None, "meter", "none", "nl+m", "pm2s", 0.95, True) == "v2_nl_m_d95_tidy_pm"
    assert system_name(None, "meter", "none", "nl+m", "pm2s", 0.95, True, "pm2s") == "v2_nl_m_d95_tidy_pm_rpm"
    assert system_name(None, "meter", "none", "nl+m", real_beats="pm2s") == "v2_nl_m_rpm"
    assert system_name(None, "meter", "none", "nl+m") == "v2_nl_m"
    jobs = (ROOT / "pace" / "v2_jobs.sh").read_text()
    assert 'pm) out="$out --clean-beats pm2s"' in jobs and 'rpm) out="$out --real-beats pm2s"' in jobs
    batch = (ROOT / "pace" / "v2_gangqinpu.sh").read_text()
    assert "compare_beats.py" in batch and ":tidy,d95,pm,rpm" in batch


def test_noise_with_many_extra_beats():
    """A fitted tracker that adds more beats than it finds (extra > 0.5: 6 of PM2S's 120 profiles
    on Gangqinpu) is simulated, with about that share of spurious beats; up to 0.5 the draws are
    the ones they always were (Beat This!'s profiles, the fixed-seed test condition)."""
    from src.beats import BeatNoise
    times, down = np.arange(0, 200, 0.5), np.arange(400) % 4 == 0
    heavy = BeatNoise(p_double=0, p_half=0, miss=0.0, extra=0.7, jitter=0.0)
    t, _ = heavy(times, down, np.random.default_rng(0))
    share = 1 - len(times) / len(t)
    assert 0.6 < share < 0.75, share
    light = BeatNoise(p_double=0, p_half=0, miss=0.1, extra=0.3, jitter=0.01)
    a = light(times, down, np.random.default_rng(5))
    b = light(times, down, np.random.default_rng(5))
    assert np.array_equal(a[0], b[0]) and len(a[0]) > 0      # (tests/test_beats.py: the fitted draws unchanged)


def test_train_for_pm2s():
    """The 2 x 2 at the meter level, end to end on a stand-in Gangqinpu: PM2S's errors fitted
    (scripts/fit_pm2s_noise.py, with a stand-in tracker: the true beats, jittered, every fourth
    one missed), a tiny model trained on them (--beat-noise, --real-beats simulated), its decoder
    tuned on them (--tune-noise) and evaluated with Beat This!'s simulated beats and with PM2S's
    (--tracker pm2s: <condition>_pm2s, the Beat This! results kept)."""
    import json
    import os
    import shutil
    from scripts import fit_pm2s_noise
    from scripts.build_meter_data import main as build
    from scripts.eval_meter import main as evaluate
    from scripts.train_meter import main as train
    from tests.test_gangqinpu import beats_of, row_of, write_release
    tmp = Path(tempfile.mkdtemp())
    old_env = {k: os.environ.get(k) for k in ("PM2S_WEIGHTS", "PM2S_TAG")}
    try:
        write_release(tmp / "release", {
            "train": [row_of(f"tr{k}", beats_of(40, ts, sec_per_beat=0.4 + 0.05 * k))
                      for k, ts in enumerate(["3/4", "4/4", "6/8", "2/4"] * 2)],
            "validation": [row_of(f"va{k}", beats_of(30, ts), split="val") for k, ts in enumerate(["3/4", "4/4"])],
            "test": [row_of(f"te{k}", beats_of(30, ts), split="test") for k, ts in enumerate(["6/8", "4/4"])]})
        data, runs = tmp / "processed", tmp / "runs"
        build(["--data", str(data), "--gangqinpu", str(tmp / "release"), "--gangqinpu-only"])

        class StandIn:
            weights = "stand-in"

            def __init__(self, *a, **k):
                self.rng = np.random.default_rng(0)

            def __call__(self, notes):   # the segment's true beats are not known here: every other note onset
                on = np.unique(np.round(np.asarray(notes)[:, 1], 3))[::2]
                keep = np.arange(len(on)) % 4 != 3
                t = on[keep] + self.rng.normal(0, 0.005, keep.sum())
                return t, np.arange(len(t)) % 3 == 0
        real_tracker = fit_pm2s_noise.PM2SBeats
        fit_pm2s_noise.PM2SBeats = StandIn
        try:
            fit_pm2s_noise.main(["--data", str(data), "--min-beats", "16", "--out", str(tmp / "noise_pm2s.json")])
        finally:
            fit_pm2s_noise.PM2SBeats = real_tracker
        fitted = json.loads((tmp / "noise_pm2s.json").read_text())
        assert fitted["fitted"] and fitted["tracker"] == "pm2s" and fitted["fit_on"]["performances"] == 8
        assert "tracker_on_val_gangqinpu" in fitted
        common = ["--data", str(data), "--runs-dir", str(runs)]
        train(["--name", "pm", "--gangqinpu", "--no-periscope", "--steps", "6", "--eval-every", "3", "--d-model", "32",
               "--layers", "1", "--workers", "0", "--batch", "4", "--chunk", "64", "--device", "cpu", "--normal-priority",
               "--beat-noise", str(tmp / "noise_pm2s.json"), "--real-beats", "simulated", *common])
        cfg = json.loads((runs / "pm" / "config.json").read_text())
        assert cfg["args"]["real_beats"] == "simulated" and cfg["args"]["beat_noise"].endswith("noise_pm2s.json")
        os.environ["PM2S_WEIGHTS"] = str(random_weights(tmp / "pm2s.pth"))   # other weights: a tag (weights_tag)
        os.environ["PM2S_TAG"] = "rand"
        evaluate(["--run", "pm", "--conditions", "gangqinpu", "--device", "cpu", "--tune", "--tune-max", "2",
                  "--tune-noise", str(tmp / "noise_pm2s.json"), *common])
        evaluate(["--run", "pm", "--conditions", "gangqinpu", "--device", "cpu", "--tracker", "pm2s", *common])
        res = json.loads((runs / "pm" / "eval_meter.json").read_text())
        assert res["gangqinpu"]["tracker"] == "beat_this" and res["gangqinpu_pm2s_rand"]["tracker"] == "pm2s"
        assert res["gangqinpu_pm2s_rand"]["pm2s_fallbacks"] >= 0 and "gangqinpu_pm2s" not in res
        assert (runs / "pm" / "grids_gangqinpu.pkl").exists() and (runs / "pm" / "grids_gangqinpu_pm2s_rand.pkl").exists()
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
