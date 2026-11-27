"""Tests for Gangqinpu's written scores as note-labeller data (src/preprocessing/gangqinpu.py
score_measures / labeller_performance, scripts/build_gangqinpu_windows.py and what uses them).

    python -m tests.test_gangqinpu_scores        # or: pytest tests/test_gangqinpu_scores.py

The release is gated, so a stand-in row in its layout is written here: a 3/4 piece with a pickup,
a note held across a bar line, a tie written inside a bar, a grace note, a duplicate, a
zero-length note, a second part, and a performance note the alignment interpolated. Covers the
score's tokens, the windows and their labels (every performed note paired, rendering the true
labels gives the target back), the noise for aligned transcriptions (extra notes only), the build
script end to end, the labeller trained with --gangqinpu-share, and its evaluation on the new
conditions (the true grid, and a meter model's grids that cover only part of a piece).
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import gangqinpu as gq  # noqa: E402

SPB, T0, PICKUP_Q = 0.5, 1.5, 2.0   # seconds per beat; the first full bar's time; the pickup's onset (quarters)


def t_of(q: float) -> float:
    """Performance time of written quarter ``q`` (a steady tempo)."""
    return T0 + (q - 3.0) * SPB


def piece_row(sample_id: str = "1", n_bars: int = 40, split: str = "train", genre: str = "pop") -> dict:
    """A 3/4 piece: a one-beat pickup, then ``n_bars`` bars of three quarters (staff 0) over a
    dotted half (staff 1), with the special cases of the module docstring in bars 2-9."""
    notes, perf, midi = [], [], []

    def add(pitch, q, dur, staff=0, part=0, grace=False, performed=True, in_midi=True, dt=0.0):
        pi = None
        if performed:
            pi = len(perf)
            perf.append({"time": t_of(q) + dt, "duration": max(dur, 0.25) * SPB * 0.9, "pitch": pitch, "velocity": 70})
        if in_midi:
            midi.append({"pitch": pitch, "onset_quarter": q, "duration_quarter": dur})
        k = 0 if q < 3.0 else 1 + int((q - 3.0) // 3)
        notes.append({"id": f"n{len(notes)}", "part_index": part, "pitch": pitch, "onset_quarter": q,
                      "duration_quarter": dur, "measure_index": k, "staff": staff, "voice": 0, "is_grace": grace,
                      "source_order": len(notes), "performance_note_index": pi})

    add(67, PICKUP_Q, 1.0)
    for b in range(1, n_bars + 1):
        q0 = 3.0 + 3 * (b - 1)
        if b == 4:   # a tie written inside the bar: 76 for two beats as two written quarters
            add(76, q0, 1.0)
            midi[-1]["duration_quarter"] = 2.0
            add(76, q0 + 1, 1.0, performed=False, in_midi=False)
            add(74, q0 + 2, 1.0)
        else:
            for j in range(3):
                add(72 + j, q0 + j, 1.0)
        if b == 2:
            add(50, q0, 5.0, staff=1)      # held into bar 3
        else:
            add(48, q0, 3.0, staff=1)
        if b == 5:
            add(71, q0, 0.0, grace=True, dt=-0.06)
        if b == 6:
            add(72, q0, 1.0, performed=False, in_midi=False)   # a second copy of the bar's first note
        if b == 7:
            add(90, q0 + 1, 0.0)                               # length 0, performed: dropped with its note
        if b == 8:
            add(36, q0, 3.0, staff=0, part=1)                  # the second part -> staff 2
    interp = [False] * len(perf)
    n9 = next(n for n in notes if n["pitch"] == 73 and abs(n["onset_quarter"] - (3.0 + 3 * 8 + 1)) < 1e-9)
    interp[n9["performance_note_index"]] = True               # the alignment interpolated it: not played
    measures = [{"part_index": p, "onset_quarter": PICKUP_Q, "duration_quarter": 1.0, "time_signature": "3/4",
                 "time_signature_inferred": False, "key_fifths": -1, "is_pickup": True} for p in (0, 1)]
    for b in range(1, n_bars + 1):
        for p in (0, 1):
            measures.append({"part_index": p, "onset_quarter": 3.0 + 3 * (b - 1), "duration_quarter": 3.0,
                             "time_signature": "3/4", "time_signature_inferred": False,
                             "key_fifths": None if b > 1 else -1, "is_pickup": False})
    beats = []
    for b in range(1, n_bars + 1):
        for j in range(3):
            beats.append({"measure": b, "beat_in_measure": j + 1, "performance_time_sec": t_of(3.0 + 3 * (b - 1) + j),
                          "is_downbeat": j == 0, "is_pickup": False, "time_valid": True,
                          "time_signature_inferred": False, "method": "matched_anchor", "confidence": 0.95,
                          "label": "db" if j == 0 else "b", "time_signature": "3/4", "beats_per_bar": 3})
    return {"sample_id": sample_id, "split": split, "group_id": f"g{sample_id}", "score_group_id": f"s{sample_id}",
            "genre": genre, "source": "gangqinpu", "audio_url": f"https://example.invalid/{sample_id}.mp3",
            "performance_notes": perf, "performance_pedal": [{"time": 2.0, "value": 127}, {"time": 4.0, "value": 0}],
            "alignment": {"performance_note_indices": list(range(len(perf))), "interpolated_mask": interp},
            "notation": {"version": 2, "midi_origin_offset_quarter": 0.0, "notes": notes, "measures": measures,
                         "parts": [{"id": "P1", "staff_count": 2}, {"id": "P2", "staff_count": 1}],
                         "alignment_complete": True, "issues": []},
            "score_notes": midi, "beats": beats}


def test_score_measures():
    row = piece_row(n_bars=10)
    measures, written, why = gq.score_measures(row)
    assert len(measures) == 11 and measures[0].length == 1 and measures[1].length == 3
    assert all(m.time_signature == "3/4" and m.key == -1 for m in measures)    # a missing key keeps the last
    assert why["held_across_bar_line"] == 1 and why["tie_in_bar"] == 1
    assert why["duplicate_note"] == 1 and why["zero_length_note"] == 1 and why["grace_note"] == 1
    # bar 2's held note: three beats in bar 2, two more tied in bar 3
    held = [(k, e) for k, m in enumerate(measures) for e in m.events if e.pitch == 50]
    assert [(k, float(e.offset), float(e.duration), e.tie) for k, e in held] == [(2, 0.0, 3.0, False), (3, 0.0, 2.0, True)]
    tied = [e for e in measures[4].events if e.pitch == 76]
    assert [(float(e.offset), e.tie) for e in sorted(tied, key=lambda e: e.offset)] == [(0.0, False), (1.0, True)]
    assert [e.staff for e in measures[8].events if e.pitch == 36] == [2]
    assert [e.grace for e in measures[5].events if e.pitch == 71] == [True]
    # struck written notes: tie continuations and dropped notes aside; quarters from measure 0
    assert len(written) == 1 + 10 * 4 - 1 + 1 + 1   # the pickup, 4 a bar (bar 4: 3, its second 76 is tied),
    assert written[0] == (0, 0.0, False)            # bar 8's second part, bar 5's grace note
    assert sum(g for _, _, g in written) == 1
    toks = gq.score_tokens(measures)
    assert toks[:6] == ["ts_3/4", "key_-1", "pos_2", "staff_0", "p_67", "d_1"]   # a pickup: no bar line, at beat 3
    assert toks.count("bar") == 10 and toks.count("tie") == 2 and "staff_2" in toks and "grace" in toks


def test_windows_and_labels():
    import yaml

    from src.eval.metrics import Accumulator
    from src.notelabels import align_notes, oracle_placed, render, window_labels
    from src.preprocessing.segment import SegmentConfig, segment_windows

    row = piece_row()
    notes, pedal, tl, measures, info = gq.labeller_performance(row)
    # the interpolated note and the zero-length note's performance note are not input
    assert info["input_notes"] == info["performance_notes"] - 2 and len(notes) == info["input_notes"]
    assert np.all(np.diff(notes[:, 1]) >= 0) and pedal.tolist() == [[2.0, 4.0]]
    seg = dict(yaml.safe_load(open(ROOT / "configs" / "preprocess.yaml"))["segment"])
    seg["window_notes"] = tuple(seg["window_notes"])
    segs, st = segment_windows(notes, pedal, tl, measures, SegmentConfig(**seg), gq.window_meta(row, "val"),
                               train=False)
    assert len(segs) >= 1 and st["low_pitch_f1"] == 0
    assert segs[0]["source"] == gq.ALIGNED_SOURCE and segs[0]["performance"] == "gangqinpu/1"
    assert segs[0]["split"] == "val_gangqinpu" and segs[0]["genre"] == "pop"
    acc, paired, live = Accumulator(), 0, 0
    for s in segs:
        lab, grid = window_labels(s["notes"], s["tokens"])
        on = lab[:, 0] >= 0
        live, paired = live + on.sum(), paired + (lab[on, 0] == 1).sum()
        assert len(grid[0]) >= 2
        sn, win, match = align_notes(s["notes"][:, 1].astype(np.float64), s["notes"][:, 0], s["tokens"])
        acc.add(s["tokens"], render(win, oracle_placed(sn, match)))
    assert paired == live, (paired, live)                 # every performed note has its written note
    r = acc.result()
    # the true labels render the target back, except the one note nobody played (interpolated) and
    # bar 4's tie inside the bar: the renderer cuts a note only at bar lines, so it writes the tied
    # quarters as one half note (as for every source); the tie across the bar line comes back
    assert r["staff_acc"] == 1.0 and r["bar_f1"] == 1.0 and r["grace_f1"] == 1.0, r
    assert r["onset_r"] == (r["gt_notes"] - 2) / r["gt_notes"] and r["onset_p"] == 1.0, r
    assert r["tie_r"] == 0.5 and r["tie_p"] == 1.0 and r["duration_acc"] == 1 - 1 / (r["gt_notes"] - 3), r


def test_extra_notes_only():
    from src.preprocessing.noise import NoiseModel

    nm = NoiseModel.from_json(str(ROOT / "configs" / "noise_stats.json"))
    nm.cfg.clean_prob = 0.0
    extra = nm.hallucinations_only()
    assert extra.cfg.drop_table is None and extra.cfg.onset_std == 0 and extra.cfg.halluc_bank == nm.cfg.halluc_bank
    rng = np.random.default_rng(0)
    notes = np.array([[60 + (k % 12), 0.1 * k, 0.3, 80] for k in range(2000)], np.float32)
    pedal = np.array([[0.5, 3.0]], np.float32)
    out, ped, lab = extra(notes, pedal, rng, return_labels=True)
    real = out[lab == 0]
    assert len(real) == len(notes) and (lab == 1).sum() > 0         # nothing dropped, extra notes added
    assert np.array_equal(real[np.lexsort((real[:, 0], real[:, 1]))], notes)   # untouched
    assert np.array_equal(ped, pedal)
    full, _, flab = nm(notes, pedal, np.random.default_rng(0), return_labels=True)
    assert (flab == 0).sum() < len(notes)                           # the full model drops notes


def write_release(root: Path, rows_by_split: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    (root / "data").mkdir(parents=True)
    for split, rows in rows_by_split.items():
        pq.write_table(pa.Table.from_pylist(rows), root / "data" / f"{split}-00000-of-00001.parquet")


def have_data() -> bool:
    d = ROOT / "data" / "processed"
    return all((d / s / "notelabels.npz").exists() for s in ("val",)) and (d / "vocab.json").exists()


def test_build_train_evaluate():
    """The build script on a stand-in release, the labels, a labeller trained with
    --gangqinpu-share and evaluated on the Gangqinpu conditions."""
    if not have_data():
        print("  (skipped: no data/processed/val with notelabels.npz)")
        return
    from scripts.build_gangqinpu_windows import main as build
    from scripts.build_notelabels import main as labels
    from scripts.eval_labeller import main as evaluate
    from scripts.train_labeller import main as train
    from src.preprocessing.store import SegmentStore

    tmp = Path(tempfile.mkdtemp())
    try:
        d, src = tmp / "data", ROOT / "data" / "processed"
        d.mkdir()
        for f in ("vocab.json", "feature_stats.json"):
            shutil.copy(src / f, d / f)
        for split in ("train", "val"):     # the usual data: val stands in for both (small)
            shutil.copytree(src / "val", d / split)
        bad = piece_row("bad", n_bars=10)
        bad["notation"] = None
        write_release(tmp / "release", {
            "train": [piece_row("1"), piece_row("2", genre="classical"), bad],
            "validation": [piece_row("3", split="val")],
            "test": [piece_row("4", split="test"), piece_row("5", split="test")]})
        cfg = tmp / "preprocess.yaml"
        cfg.write_text((ROOT / "configs" / "preprocess.yaml").read_text().replace("out_dir: data/processed",
                                                                                   f"out_dir: {d.as_posix()}"))
        build(["--release", str(tmp / "release"), "--config", str(cfg), "--workers", "1"])
        for split in ("train", "val", "test"):
            rep = json.loads((d / f"{split}_gangqinpu" / "report.json").read_text())
            assert rep["windows"] > 0 and rep["tokens_not_in_vocab"]["count"] == 0
        rep = json.loads((d / "train_gangqinpu" / "report.json").read_text())
        assert rep["rows"] == 3 and rep["performances"] == 2 and rep["skipped_performances"] == {"no_notation": 1}
        assert rep["genres"] == {"pop": 1, "classical": 1} and rep["converter"]["held_across_bar_line"] == 2
        st = SegmentStore(d / "train_gangqinpu")
        assert {m["source"] for m in st.meta} == {gq.ALIGNED_SOURCE} and {m["genre"] for m in st.meta} == {"pop", "classical"}
        n_train = len(st)
        assert n_train > len(SegmentStore(d / "test_gangqinpu")) / 2   # training windows overlap
        labels(["--data", str(d), "--splits", "train_gangqinpu,val_gangqinpu,test_gangqinpu", "--workers", "1"])

        runs = tmp / "runs"
        base = ["--name", "g", "--size", "small", "--max-steps", "4", "--eval-every", "2", "--eval-loss-windows", "8",
                "--eval-decode-windows", "4", "--workers", "0", "--device", "cpu", "--batch-notes", "2048",
                "--warmup", "2", "--normal-priority", "--runs-dir", str(runs), "--data", str(d)]
        train(base + ["--gangqinpu-share", "0.5"])
        run = runs / "g"
        log = (run / "log.csv").read_text().splitlines()
        assert "val_gq_onset_f1" in log[0] and "val_gq_loss" in log[0], log[0]
        c = json.loads((run / "config.json").read_text())
        assert c["data"]["gangqinpu_share"] == 0.5 and c["data"]["gangqinpu_windows"] == n_train
        try:
            train(["--name", "g2"] + base[2:] + ["--gangqinpu-share", "1.5"])
            raise AssertionError("--gangqinpu-share 1.5 was accepted")
        except SystemExit as e:
            assert "between 0 and 1" in str(e)

        # the true grid on both conditions; then a meter model's grids covering half of each piece
        evaluate(["--run", "g", "--runs-dir", str(runs), "--data", str(d), "--conditions", "gangqinpu,gangqinpu_real",
                  "--device", "cpu", "--normal-priority"])
        res = json.loads((run / "eval_pieces_best.json").read_text())
        assert 0 <= res["gangqinpu"]["grid_true"]["onset_f1"] <= 1 and "gangqinpu_real" in res
        meter = runs / "m"
        meter.mkdir()
        for cond in ("gangqinpu", "gangqinpu_real"):
            grids = {}
            for sid in ("4", "5"):
                perfs, _ = gq.performance(piece_row(sid, split="test"))
                gt = perfs[0]["gt"]
                keep = gt["t"] <= np.median(gt["t"])                 # the first half of the piece only
                q = np.asarray(gt["q"], np.float64)[keep]
                nb = int(q.max() // 3) + 1                           # eval_meter.py's grid: bars, bar lines
                g = {"t": np.asarray(gt["t"])[keep], "q": q, "bars": ["3/4"] * nb, "bar_q": np.arange(nb) * 3.0,
                     "span": (float(gt["t"][keep][0]) - 0.5, float(gt["t"][keep][-1]) + 0.5),
                     "performance": f"gangqinpu/{sid}", "key": -1}
                grids[f"gangqinpu/{sid}"] = g
            pickle.dump(grids, open(meter / f"grids_{cond}.pkl", "wb"))
        evaluate(["--run", "g", "--runs-dir", str(runs), "--data", str(d), "--conditions", "gangqinpu,gangqinpu_real",
                  "--grid", "meter:m", "--device", "cpu", "--normal-priority"])
        res = json.loads((run / "eval_pieces_best.json").read_text())
        r = res["gangqinpu_real"]["grid_meter_m"]
        assert r["windows_outside_grids"] > 0 and 0 <= r["onset_f1"] <= 1
        assert "grid_true" in res["gangqinpu"]                      # the other grid's result stays
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_whole_piece_inputs():
    """scripts/gangqinpu_inputs.py on a stand-in release with stand-in trackers: one piece per
    score, the genres in proportion; the reference MusicXML reads back as the written score; the
    aligned notes as MIDI; the transcription cut to their span; the beats stored by performance."""
    from scripts.gangqinpu_inputs import choose, main as make
    from src.beats import load_beats
    from src.preprocessing.performance import load_performance
    from src.preprocessing.score import load_score

    rows = [dict(piece_row(str(k), n_bars=10, split="test", genre=g), score_group_id=f"s{k // 2}")
            for k, g in enumerate(["pop"] * 12 + ["jazz"] * 4 + ["folk"] * 4)]
    picked = choose(rows, 5)
    assert len({r["score_group_id"] for r in picked}) == 5                  # one per score
    assert sorted(r["genre"] for r in picked) == ["folk", "jazz", "pop", "pop", "pop"]
    assert [r["sample_id"] for r in choose(rows, 5)] == [r["sample_id"] for r in picked]   # seeded
    tmp = Path(tempfile.mkdtemp())
    try:
        write_release(tmp / "release", {"test": rows[:6]})
        for r in rows[:6]:
            if r["sample_id"] != "5":                                         # no recording: left out
                (tmp / "audio" / "test").mkdir(parents=True, exist_ok=True)
                (tmp / "audio" / "test" / f"{r['sample_id']}.mp3").write_bytes(b"x")
        seen = []

        def beat_this(wav):
            seen.append(Path(wav).name)
            t = np.array([t_of(3.0 + k) for k in range(30)])
            return t, (np.arange(30) % 3 == 0)

        def transcribe(wav, mid):      # the "transcription": the aligned notes plus a stray note long after
            from scripts.system_inputs import write_midi
            row = next(r for r in rows if f"{r['sample_id']}.mp3" == Path(wav).name)
            notes, pedal, *_ = gq.labeller_performance(row)
            write_midi(np.concatenate([notes, [[60, 500.0, 500.5, 80]]]), pedal, Path(mid))

        out = tmp / "eval"
        make(["--release", str(tmp / "release"), "--audio", str(tmp / "audio"), "--n", "10", "--out", str(out),
              "--device", "cpu"], beat_this=beat_this, transcribe=transcribe)
        man = json.loads((out / "manifest.json").read_text())
        perfs = sorted({e["performance"] for e in man})
        # rows 0-5 share scores pairwise: one per score, among those with a recording (not 5)
        assert len(perfs) == 3 and "gangqinpu/5" not in perfs, perfs
        assert sorted(e["condition"] for e in man) == ["gangqinpu"] * 3 + ["gangqinpu_real"] * 3
        e = next(x for x in man if x["condition"] == "gangqinpu")
        measures = load_score(e["reference"])
        assert len(measures) == 11 and measures[1].time_signature == "3/4"
        n_written = sum(1 for m in measures for ev in m.events if not ev.tie and not ev.grace)
        m0, _, _ = gq.score_measures(rows[int(e["performance"].split("/")[1])])
        assert n_written == sum(1 for m in m0 for ev in m.events if not ev.tie and not ev.grace)
        notes, _ = load_performance(e["midi"])
        assert len(notes) == e["notes"] > 0
        real = next(x for x in man if x["condition"] == "gangqinpu_real" and x["performance"] == e["performance"])
        rn, _ = load_performance(real["midi"])
        assert len(rn) == len(notes) and rn[:, 1].max() < 100           # the stray note is cut off
        beats = load_beats(out / "beats_gangqinpu.npz")
        assert sorted(beats) == perfs and len(beats[perfs[0]][0]) == 30
        # without a transcriber: the aligned notes only
        make(["--release", str(tmp / "release"), "--audio", str(tmp / "audio"), "--n", "2", "--out", str(out),
              "--no-transcribe", "--device", "cpu"], beat_this=beat_this)
        man = json.loads((out / "manifest.json").read_text())
        assert {e["condition"] for e in man} == {"gangqinpu"} and len(man) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_batch_and_jobs():
    """pace/v2_gqlabel.sh: 8 jobs, each after what it needs; pace/v2_jobs.sh: nl_gq's options, the
    labellers on the meter grids of every condition, SYS_ARGS passed to every whole-piece system."""
    import subprocess

    from tests.batch_env import batch_env
    if shutil.which("bash") is None:
        print("  (skipped: no bash)")
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "pace").mkdir()
        for f in ("v2_gqlabel.sh", "v2_jobs.sh"):
            shutil.copy(ROOT / "pace" / f, tmp / "pace")
        for f in ("outputs/runs/meter_v3_tp/decode.json", "outputs/runs/meter_v4_gq/best.pt",
                  "outputs/runs/meter_v4_gq_pm/best.pt", "outputs/runs/meter_v3_s1/best.pt",
                  "outputs/runs/nl_base_gn/best.pt", "outputs/system_eval/manifest.json", "data/processed/vocab.json",
                  "data/processed/train/notelabels.npz", "data/processed/meter/train_gangqinpu.pkl",
                  "configs/beat_noise_bt_gangqinpu.json", "data/raw/gangqinpu/data/test-0.parquet",
                  "data/raw/gangqinpu/audio/test/1.mp3", "bd/MIDI2ScoreTF.ckpt"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "sbatch").chmod(0o755)

        def run_batch(env=None):
            if (tmp / "jobs.txt").exists():
                (tmp / "jobs.txt").unlink()
            r = subprocess.run([shutil.which("bash"), "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_gqlabel.sh'],
                               cwd=tmp, env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd"),
                                                      **(env or {})), capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            return {cmd.split("-J ")[1].split()[0]: (jid, cmd)
                    for jid, cmd in (ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines())}
        by = run_batch()
        assert list(by) == ["lbtests", "lbdata", "lbinputs", "lbtrain", "lbmeter", "lbsys", "lbsys2", "lbfinal"], list(by)
        tests = by["lbtests"][0]
        assert "tests.test_gangqinpu_scores" in by["lbtests"][1] and "--dependency" not in by["lbtests"][1]
        d = by["lbdata"][1]
        assert f"afterok:{tests}" in d and "--gres" not in d
        assert d.index("build_gangqinpu_windows.py") < d.index("build_notelabels.py --splits train_gangqinpu,val_gangqinpu,test_gangqinpu") \
            < d.index("tempo meter_v3_s1 meter_v3_s1_tp")
        assert f"afterok:{tests}" in by["lbinputs"][1] and "gangqinpu_inputs.py --release data/raw/gangqinpu --n 60" in by["lbinputs"][1]
        tr = by["lbtrain"][1]
        assert f"afterok:{by['lbdata'][0]}" in tr and "labellers nl_gq" in tr and "evals nl_gq nl_base_gn" in tr
        # the current default scored on Gangqinpu before nl_gq is published against it
        assert tr.index("eval_labeller.py --run nl_base_gn --conditions gangqinpu") < tr.index("labellers nl_gq")
        assert 'METERS="meter_v3_tp meter_v4_gq"' in tr
        assert f"afterok:{tests}" in by["lbmeter"][1] and "meters meter_v4_gql" in by["lbmeter"][1]
        s1 = by["lbsys"][1]
        assert f"afterany:{by['lbtrain'][0]}:{by['lbinputs'][0]}" in s1
        assert "systems nl_gq+meter_v3_tp:tidy,d95 nl_gq+meter_v4_gq:tidy,d95 nl_base_gn+meter_v3_s1_tp:tidy,d95" in s1
        gqsys = ('SYS_ARGS="--manifest outputs/system_eval_gangqinpu/manifest.json --out outputs/system_eval_gangqinpu '
                 '--conditions gangqinpu,gangqinpu_real" bash pace/v2_jobs.sh systems')
        assert gqsys + " nl_gq+meter_v4_gq:tidy,d95" in s1
        s2 = by["lbsys2"][1]
        assert f"afterok:{by['lbinputs'][0]}" in s2 and gqsys + " nl_base_gn+meter_v3_tp:tidy,d95" in s2
        assert "nl_base_gn+meter_v4_gq_pm:tidy,d95,pm,rpm sf_base_reg_cab_c128:meter:sb" in s2
        assert "beyer_dai.py predict" in s2 and "--manifest outputs/system_eval_gangqinpu/manifest.json" in s2
        fin = by["lbfinal"][1]
        assert f"afterany:{by['lbsys'][0]}:{by['lbsys2'][0]}:{by['lbmeter'][0]}" in fin
        assert "systems nl_base_gn+meter_v4_gql:tidy,d95" in fin and fin.count("beyer_dai.py score") == 2
        assert "--results outputs/system_eval_gangqinpu/results.json --out outputs/system_eval_gangqinpu/paired.md" in fin
        by = run_batch({"STAGE": "eval"})
        assert list(by) == ["lbsys", "lbsys2", "lbfinal"] and "--dependency" not in by["lbsys"][1]

        # pace/v2_jobs.sh with a stand-in python: the options it passes
        (tmp / "bin" / "python").write_text('#!/bin/bash\necho "$*" >> "$CALLS"\nname=""; prev=""\n'
                                            'for a in "$@"; do [ "$prev" = "--name" ] || [ "$prev" = "--run" ] && name=$a; prev=$a; done\n'
                                            'case "$1" in scripts/train_labeller.py) mkdir -p "outputs/runs/$name"; '
                                            'echo \'{"phase": "done"}\' > "outputs/runs/$name/status.json" ;; esac\nexit 0\n')
        (tmp / "bin" / "python").chmod(0o755)
        for f in ("outputs/runs/nl_base_gn/status.json", "outputs/runs/meter_v3_tp/status.json"):
            (tmp / f).write_text('{"phase": "done"}')
        (tmp / "outputs/runs/meter_v3_tp/eval_meter.json").write_text("{}")

        def jobs(args, env=None):
            calls = tmp / "calls.txt"
            if calls.exists():
                calls.unlink()
            subprocess.run([shutil.which("bash"), "-c", f'export PATH="$PWD/bin:$PATH"; bash pace/v2_jobs.sh {args}'],
                           cwd=tmp, env=batch_env(CALLS=str(calls), SLURM_JOB_ID="7", **(env or {})),
                           capture_output=True, text=True)
            return calls.read_text().splitlines() if calls.exists() else []
        calls = jobs("labellers nl_gq")
        assert any("--name nl_gq --size base --grid-noise 0.8 --gangqinpu-share 0.5" in c for c in calls), calls
        calls = jobs("evals nl_base_gn", {"METERS": "meter_v3_tp", "GRIDS_OLD": "", "NEW_NLS": ""})
        assert any("--grid meter:meter_v3_tp --conditions clean,synthetic,real,periscope,gangqinpu,gangqinpu_real" in c
                   for c in calls), calls
        calls = jobs("systems nl_base_gn+meter_v3_tp:tidy,d95",
                     {"SYS_ARGS": "--manifest m.json --out o --conditions gangqinpu,gangqinpu_real"})
        assert calls == ["scripts/system_scores.py --v2 nl_base_gn+meter_v3_tp --tidy --delete-above 0.95 "
                         "--manifest m.json --out o --conditions gangqinpu,gangqinpu_real"], calls
        assert "--gangqinpu --gangqinpu-share 0.5 --steps 100000" in (tmp / "pace" / "v2_jobs.sh").read_text()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
