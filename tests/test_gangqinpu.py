"""Tests for Gangqinpu as meter-model data (src/preprocessing/gangqinpu.py and what uses it).

    python -m tests.test_gangqinpu        # or: pytest tests/test_gangqinpu.py

The release is gated, so a small stand-in in its layout (data/<split>-*.parquet with the same
columns) is written here. Covers the notes (interpolated ones dropped, sorted), the beat filters
(each reason a bar is dropped, 1-based beat numbers, repeats and skipped measures), the grids
against the beats they came from, the reading of a release (only Gangqinpu's rows, one per audio
recording, our split names), build_meter_data.py --gangqinpu-only, the batch shares in
train_meter.py and the new evaluation condition.
"""
from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import gangqinpu as gq  # noqa: E402
from src.preprocessing.gangqinpu import GridFilter  # noqa: E402
from src.meter import TS_CLASSES  # noqa: E402


def beats_of(n_bars: int, ts: str = "3/4", sec_per_beat: float = 0.5, t0: float = 1.0, base: int = 1,
             first_measure: int = 1, **over) -> list:
    """A clean beat list: n_bars bars of ``ts`` at a steady tempo, beat numbers from ``base``."""
    from src.beats import beat_layout
    n, _ = beat_layout(ts)
    out, t = [], t0
    for m in range(first_measure, first_measure + n_bars):
        for j in range(n):
            out.append({"measure": m, "beat_in_measure": j + base, "performance_time_sec": t, "is_downbeat": j == 0,
                        "is_pickup": False, "time_valid": True, "time_signature_inferred": False,
                        "method": "matched_anchor", "confidence": 0.95, "label": "db" if j == 0 else "b",
                        "time_signature": ts, "beats_per_bar": n, **over})
            t += sec_per_beat
    return out


def notes_over(t0: float, t1: float, step: float = 0.25) -> list:
    return [{"time": float(t), "duration": 0.2, "pitch": 60 + (k % 12), "velocity": 70}
            for k, t in enumerate(np.arange(t0, t1, step))]


def row_of(sample_id: str, beats: list, split: str = "train", **over) -> dict:
    t1 = max(b["performance_time_sec"] for b in beats) + 1.0
    notes = notes_over(0.5, t1)
    return {"sample_id": sample_id, "split": split, "group_id": f"g{sample_id}", "score_group_id": f"s{sample_id}",
            "genre": "pop", "audio_url": f"https://example.invalid/{sample_id}.mp3", "source": "gangqinpu",
            "performance_notes": notes,
            "alignment": {"performance_note_indices": [0, 1, 2], "interpolated_mask": [False, True, False]},
            "beats": beats, **over}


def test_notes():
    row = row_of("a", beats_of(8))
    row["performance_notes"] = row["performance_notes"][:3][::-1]   # unsorted
    notes = gq.performance_notes(row)
    assert notes.shape == (2, 4) and notes.dtype == np.float32
    assert np.all(np.diff(notes[:, 1]) >= 0)
    times = sorted(n["time"] for n in row["performance_notes"])
    assert np.allclose(notes[:, 1], [times[0], times[2]])            # the interpolated note (index 1) is gone
    assert np.allclose(notes[:, 2] - notes[:, 1], 0.2, atol=1e-6) and notes[0, 0] in (60, 61, 62)
    row["alignment"] = None                                            # no alignment: every note kept
    assert len(gq.performance_notes(row)) == 3
    assert gq.performance_notes({"performance_notes": []}).shape == (0, 4)


def test_clean_beats_make_one_grid():
    beats = beats_of(10, "3/4")
    (t, bars, idx, tss), why = gq.trusted_beats(beats)
    assert len(t) == 30 and why["kept"] == 30 and set(idx) == {0, 1, 2}      # 1-based numbers -> from 0
    perfs, _ = gq.performance(row_of("a", beats))
    assert len(perfs) == 1
    p = perfs[0]
    gt = p["gt"]
    assert np.allclose(gt["t"], [b["performance_time_sec"] for b in beats])
    assert np.allclose(np.diff(gt["q"]), 1.0) and gt["down"].sum() == 10
    assert set(gt["ts"]) == {TS_CLASSES.index("3/4")}
    assert p["dataset"] == "gangqinpu" and p["real_notes"] and p["beats"] is None and p["piece"] == "gangqinpu/sa"
    assert p["notes"][:, 1].min() >= gt["t"][0] - 0.5 and p["notes"][:, 1].max() <= gt["t"][-1] + 0.5
    # compound meter: 6/8 has 2 dotted beats per bar of 1.5 quarters
    perfs, _ = gq.performance(row_of("b", beats_of(12, "6/8", base=0)))
    assert len(perfs) == 1 and np.allclose(np.diff(perfs[0]["gt"]["q"]), 1.5)


def test_dropped_bars_split_the_run():
    flt = GridFilter(min_beats=8)
    for reason, change in [("low_confidence", {"confidence": 0.3}), ("untimed_method", {"method": "extrapolated"}),
                           ("time_invalid", {"time_valid": False}), ("pickup", {"is_pickup": True}),
                           ("inferred_time_signature", {"time_signature_inferred": True})]:
        beats = beats_of(10, "4/4")
        for b in beats:
            if b["measure"] == 5:
                b.update(change)
        perfs, why = gq.performance(row_of("a", beats), flt)
        assert why[reason] == 4 and why["kept"] == 36, (reason, why)
        assert len(perfs) == 2 and [len(p["gt"]["t"]) for p in perfs] == [16, 20], reason
    # inferred time signatures can be kept
    beats = beats_of(10, "4/4", time_signature_inferred=True)
    assert gq.performance(row_of("a", beats), GridFilter(keep_inferred=True))[1]["kept"] == 40
    # a beat number the time signature cannot have, times running backwards, an absurd tempo
    beats = beats_of(10, "3/4")
    beats[4]["beat_in_measure"] = 5
    assert gq.trusted_beats(beats)[1]["beats_vs_time_signature"] == 3
    beats = beats_of(10, "3/4")
    beats[7]["performance_time_sec"] = 0.0
    assert gq.trusted_beats(beats)[1]["time_order"] == 3
    beats = beats_of(10, "3/4")
    beats[9]["performance_time_sec"] = beats[10]["performance_time_sec"] - 0.01   # 6000 qpm inside bar 4
    assert gq.trusted_beats(beats)[1]["tempo"] == 3
    # short runs are dropped (min_beats), their notes with them
    assert gq.performance(row_of("a", beats_of(3)))[0] == []


def test_their_beats_onto_ours():
    """The release counts 6/4 in six quarter beats (ASAP and beat_layout: two dotted halves), and
    a few beats sit between beats (4.5): each is mapped by its place in the bar in quarters, only
    those on one of our beats kept. The release's beat numbers are floats from 1."""
    beats = []
    for m in range(1, 9):                                     # 8 bars of 6/4 counted in quarters
        for j in range(6):
            beats.append({"measure": m, "beat_in_measure": float(j + 1), "performance_time_sec": 1.0 + (m - 1) * 3.0 + j * 0.5,
                          "is_downbeat": j == 0, "is_pickup": False, "time_valid": True,
                          "time_signature_inferred": False, "method": "matched_anchor", "confidence": 0.95,
                          "label": "db" if j == 0 else "b", "time_signature": "6/4", "beats_per_bar": 6})
    (t, bars, idx, tss), why = gq.trusted_beats(beats)
    assert len(t) == 16 and list(idx[:4]) == [0, 1, 0, 1] and why["between_our_beats"] == 32
    assert np.allclose(t[:2], [1.0, 2.5])                      # quarters 1 and 4 of the bar
    perfs, _ = gq.performance(row_of("a", beats), GridFilter(min_beats=8))
    gt = perfs[0]["gt"]
    assert np.allclose(np.diff(gt["q"]), 3.0) and set(gt["ts"]) == {TS_CLASSES.index("6/4")}
    # a beat between beats is dropped alone, its bar kept
    beats = beats_of(10, "4/4")
    beats.insert(5, {**beats[5], "beat_in_measure": 2.5, "performance_time_sec": beats[5]["performance_time_sec"] - 0.25})
    (t, _, _, _), why = gq.trusted_beats(beats)
    assert len(t) == 40 and why["between_our_beats"] == 1


def test_repeats_and_skipped_measures():
    beats = beats_of(4, "2/4") + beats_of(4, "2/4", t0=5.0)              # measures 1-4 played twice
    (t, bars, idx, tss), _ = gq.trusted_beats(beats)
    assert len(t) == 16 and list(np.unique(bars)) == list(range(8))        # the repeat continues the run
    perfs, _ = gq.performance(row_of("a", beats), GridFilter(min_beats=8))
    assert len(perfs) == 1 and len(perfs[0]["gt"]["t"]) == 16
    skip = beats_of(5, "2/4") + beats_of(5, "2/4", t0=6.0, first_measure=9)  # measures 6-8 missing
    perfs, _ = gq.performance(row_of("a", skip), GridFilter(min_beats=8))
    assert len(perfs) == 2                                                  # what lies between is unknown


def write_release(root: Path, rows_by_split: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    (root / "data").mkdir(parents=True)
    for split, rows in rows_by_split.items():
        pq.write_table(pa.Table.from_pylist(rows), root / "data" / f"{split}-00000-of-00001.parquet")


def test_release_and_build_script():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        dup = row_of("dup", beats_of(10))
        dup["audio_url"] = "https://example.invalid/a.mp3"
        asap_row = row_of("asap1", beats_of(10), source="asap")
        write_release(tmp / "release", {
            "train": [row_of("a", beats_of(10)), dup, asap_row, row_of("short", beats_of(2))],
            "validation": [row_of("v", beats_of(10, "6/8"), split="val")],
            "test": [row_of("t", beats_of(12, "2/4"), split="test")]})
        rows = list(gq.iter_rows(tmp / "release", "train"))
        assert [r["sample_id"] for r in rows] == ["a", "short"]             # the duplicate audio and ASAP row go
        perfs, rep = gq.build_split(tmp / "release", "train")
        assert rep["rows"] == 2 and rep["performances"] == 1 and rep["segments"] == 1 and rep["genres"] == {"pop": 1}
        try:
            gq.build_split(tmp / "nothing", "train")
            raise AssertionError("no release: should fail")
        except FileNotFoundError:
            pass
        from scripts.build_meter_data import main as build
        data = tmp / "processed"
        build(["--data", str(data), "--gangqinpu", str(tmp / "release"), "--gangqinpu-only"])
        for split, n in (("train", 1), ("val", 1), ("test", 1)):
            got = pickle.load(open(data / "meter" / f"{split}_gangqinpu.pkl", "rb"))
            assert len(got) == n and got[0]["dataset"] == "gangqinpu"
        assert not (data / "meter" / "train.pkl").exists()                 # --gangqinpu-only: nothing else
        rep = json.loads((data / "meter" / "gangqinpu_report.json").read_text())
        assert set(rep["splits"]) == {"train", "val", "test"} and rep["filter"]["min_confidence"] == 0.5


def test_batch_shares():
    from scripts.train_meter import mixed_weights, weights

    def perf(piece, n):
        return {"piece": piece, "gt": {"t": np.zeros(n)}}
    sources = {"asap": [perf("x", 10), perf("x", 30), perf("y", 20)], "real": [], "periscope": [perf("z", 40)],
               "gangqinpu": [perf(f"g{k}", 10) for k in range(100)]}
    plain = mixed_weights(sources, "piece", {})
    assert np.allclose(plain, weights([p for v in sources.values() for p in v], "piece"))   # as before
    w = mixed_weights(sources, "piece", {"gangqinpu": 0.3})
    assert np.isclose(w.sum(), 1.0) and np.isclose(w[4:].sum(), 0.3)
    rest = weights([p for k in ("asap", "periscope") for p in sources[k]], "piece") * 0.7
    assert np.allclose(w[:4], rest)
    only = mixed_weights({"asap": [], "gangqinpu": sources["gangqinpu"]}, "piece", {"gangqinpu": 0.3})
    assert np.isclose(only.sum(), 1.0)


def test_train_and_evaluate():
    """A tiny meter model trained on Gangqinpu alone (--gangqinpu, validated on its own group) and
    evaluated on its test condition."""
    import shutil
    from scripts.build_meter_data import main as build
    from scripts.eval_meter import CONDITIONS, main as evaluate
    from scripts.train_meter import main as train
    assert CONDITIONS["gangqinpu"] == "test_gangqinpu"
    tmp = Path(tempfile.mkdtemp())
    try:
        write_release(tmp / "release", {
            "train": [row_of(f"tr{k}", beats_of(40, ts, sec_per_beat=0.4 + 0.05 * k))
                      for k, ts in enumerate(["3/4", "4/4", "6/8", "2/4"] * 2)],
            "validation": [row_of(f"va{k}", beats_of(30, ts), split="val") for k, ts in enumerate(["3/4", "4/4"])],
            "test": [row_of(f"te{k}", beats_of(30, ts), split="test") for k, ts in enumerate(["6/8", "4/4"])]})
        data, runs = tmp / "processed", tmp / "runs"
        build(["--data", str(data), "--gangqinpu", str(tmp / "release"), "--gangqinpu-only"])
        common = ["--data", str(data), "--runs-dir", str(runs)]
        train(["--name", "gq", "--gangqinpu", "--no-periscope", "--steps", "10", "--eval-every", "5", "--d-model", "32",
               "--layers", "1", "--workers", "0", "--batch", "4", "--chunk", "64", "--device", "cpu",
               "--normal-priority", *common])
        cfg = json.loads((runs / "gq" / "config.json").read_text())
        assert cfg["data"]["train_by_source"]["gangqinpu"] == 8 and cfg["data"]["val_by_source"] == {"gangqinpu": 2}
        evaluate(["--run", "gq", "--conditions", "gangqinpu", "--device", "cpu", *common])
        res = json.loads((runs / "gq" / "eval_meter.json").read_text())
        assert "gangqinpu" in res
        try:  # without the built files, --gangqinpu says what to run
            train(["--name", "gq2", "--gangqinpu", "--steps", "1", "--device", "cpu", "--normal-priority",
                   "--data", str(tmp / "empty"), "--runs-dir", str(runs)])
            raise AssertionError("--gangqinpu without data should stop")
        except SystemExit as e:
            assert "build_meter_data.py --gangqinpu" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_batch_script():
    """pace/v2_gangqinpu.sh: tests first; the data (download, build, the baseline on its test set)
    and the refined grids after them; two chained training jobs after the data; the new systems
    after training; the final scoring after both branches. Every job activates the environment."""
    import shutil
    import subprocess
    from tests.batch_env import batch_env
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_gangqinpu.sh", tmp / "pace")
        for f in ("outputs/runs/meter_v3/best.pt", "outputs/runs/meter_v3_tp/decode.json",
                  "outputs/system_eval/manifest.json"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        (tmp / "bd").mkdir()
        (tmp / "bd" / "MIDI2ScoreTF.ckpt").write_text("x")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "sbatch").chmod(0o755)
        def run_batch(env=None):
            if (tmp / "jobs.txt").exists():
                (tmp / "jobs.txt").unlink()
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_gangqinpu.sh'],
                               cwd=tmp, env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd"),
                                                      **(env or {})),
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            return [ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()]
        jobs = run_batch()
        by = {cmd.split("-J ")[1].split()[0]: (jid, cmd) for jid, cmd in jobs}
        assert list(by) == ["gqtests", "gqdata", "gqtrack", "gqrefine", "gqtrain", "gqtrain2", "gqeval", "gqfinal"], list(by)
        assert len(by) <= 8                                                   # PACE's per-user job limit
        tests = by["gqtests"][0]
        assert "--dependency" not in by["gqtests"][1] and "tests.test_gangqinpu" in by["gqtests"][1]
        data = by["gqdata"][1]
        assert f"afterok:{tests}" in data and "fetch_gangqinpu.py" in data
        assert "build_meter_data.py --gangqinpu data/raw/gangqinpu --gangqinpu-only" in data and "gqbase meter_v3_tp" in data
        assert (data.index("build_meter_data.py") < data.index("fit_pm2s_noise.py") < data.index("gqbase")
                < data.index("fetch_gangqinpu_audio.py --release data/raw/gangqinpu --out data/raw/gangqinpu/audio --train 3000"))
        track = by["gqtrack"][1]                                 # the real trackers, once, before training
        assert f"afterok:{by['gqdata'][0]}" in track and "--gres=gpu:1" in track
        assert "track_meter_data.py --release data/raw/gangqinpu --audio data/raw/gangqinpu/audio" in track
        assert track.index("track_meter_data.py") < track.index("fit_pm2s_noise.py --from-stored beat_this")
        ref = by["gqrefine"][1]
        assert f"afterok:{tests}" in ref and "refine meter_v3_tp meter_v3_tp_rf 0.05 0.25" in ref
        assert "refine meter_v3_tp meter_v3_tp_fill 0 0.25" in ref and "refine oracle meter_oracle_rf" in ref
        assert "nl_base_gn+meter_v3_tp_rf:tidy,d95" in ref and "--gres=gpu:1" in ref
        assert f"afterok:{by['gqtrack'][0]}" in by["gqtrain"][1]
        assert "meters meter_v4_gq meter_v4_gq_pm meter_v4_gq_both" in by["gqtrain"][1]   # Beat This!, PM2S, both
        assert f"afterany:{by['gqtrain'][0]}" in by["gqtrain2"][1]
        ev = by["gqeval"][1]                                     # the 2 x 2: both trackers, both models
        assert f"afterany:{by['gqtrain2'][0]}" in ev
        assert 'PM2S_RUNS="meter_v4_gq_pm" bash pace/v2_jobs.sh trackers meter_v4_gq meter_v4_gq_pm meter_v4_gq_both meter_v3_tp' in ev
        specs = ev.split("systems", 1)[1]                    # each model with its own tracker(s)
        assert "nl_base_gn+meter_v4_gq:tidy,d95 " in specs + " " and "nl_base_gn+meter_v4_gq_both:tidy,d95" in specs
        assert "nl_base_gn+meter_v4_gq_pm:tidy,d95,pm,rpm" in specs and "meter_v4_gq:tidy,d95,pm" not in specs
        assert f"afterany:{by['gqrefine'][0]}:{by['gqeval'][0]}" in by["gqfinal"][1]
        for name, (_, cmd) in by.items():
            if name != "gqfinal":
                assert "--wrap=source pace/common.sh" in cmd, name                # the environment first
        # again after a stop: one more training job, then the evaluation; or the evaluation alone
        names = lambda js: [cmd.split("-J ")[1].split()[0] for _, cmd in js]  # noqa: E731
        jobs = run_batch({"STAGE": "train"})
        assert names(jobs) == ["gqtrain3", "gqeval", "gqfinal"], names(jobs)
        assert "--dependency" not in jobs[0][1] and "meters meter_v4_gq meter_v4_gq_pm meter_v4_gq_both" in jobs[0][1]
        assert f"--dependency=afterany:{jobs[0][0]}" in jobs[1][1] and f"--dependency=afterany:{jobs[1][0]}" in jobs[2][1]
        jobs = run_batch({"STAGE": "eval"})
        assert names(jobs) == ["gqeval", "gqfinal"] and "--dependency" not in jobs[0][1]
        assert f"--dependency=afterany:{jobs[0][0]}" in jobs[1][1]
        # fewer training recordings, or none
        jobs = run_batch({"AUDIO_TRAIN": "0"})
        assert "--train 0" in {cmd.split("-J ")[1].split()[0]: cmd for _, cmd in jobs}["gqdata"]
        # the run flags and tasks it relies on exist
        jobs_sh = (ROOT / "pace" / "v2_jobs.sh").read_text()
        for needle in ("meter_v4_gq)", "meter_v4_gq_both)", "meter_v4_gq_pm)", "  refine)", "  gqbase)", "  trackers)",
                       "--beat-noise configs/beat_noise_pm2s.json --real-beats simulated --tracked pm2s --tracked-share 0.5",
                       "--tracked beat_this --tracked-share 0.5 --beat-noise-gangqinpu configs/beat_noise_bt_gangqinpu.json",
                       "--aux pm2s --aux-noise configs/beat_noise_pm2s.json",
                       "--tune-noise configs/beat_noise_pm2s.json", "--real-tracked --conditions gangqinpu"):
            assert needle in jobs_sh, needle


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
