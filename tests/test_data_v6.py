"""Tests for the v6 data (pace/v2_data.sh): the capped, performer-diverse PianoCoRe selection
(scripts/fetch_pianocore.py select), the recordings' links (scripts/fetch_pianocore_audio.py), the
re-transcription and its two-step alignment (scripts/retranscribe_pianocore.py,
src/preprocessing/pianocore.py pianocore_performance), the build from several folders that never
adds to the test set (scripts/build_dataset.py), and the labeller's weighing by work
(src/dataset.py piece_counts).

    python -m tests.test_data_v6        # or: pytest tests/test_data_v6.py

Uses real PianoCoRe rows where data/raw/pianocore/periscope has them (else skips that test);
stand-in downloader and transcriber.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PC = ROOT / "data/raw/pianocore/periscope"


def test_select_caps_and_performers():
    from scripts.fetch_pianocore import select
    rows = pd.DataFrame({"id": [f"p{i}" for i in range(12)],
                         "work_key": ["A"] * 8 + ["B"] * 4,
                         "performer": ["X", "X", "X", "Y", "Y", "Z", None, None, "X", "X", "Y", "Y"]})
    s = select(rows, per_work=100, total_per_work=5, have_per_work={"A": 2}, seed=0)
    a = s[s.work_key == "A"]
    assert len(a) == 3                                          # 5 per work, 2 fetched already
    assert len(set(a["performer"].fillna(a["id"]))) == 3          # three different pianists before any second
    b = s[s.work_key == "B"]
    assert len(b) == 4 and sorted(b["performer"]) == ["X", "X", "Y", "Y"]
    first_two = b["performer"].tolist()[:2]
    assert sorted(first_two) == ["X", "Y"]                       # one of each before the seconds
    assert len(select(rows, per_work=2, total_per_work=0, have_per_work={}, seed=0)) == 4


def test_links():
    from scripts.fetch_pianocore_audio import atepp_links, link_of, main

    class Resp:
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    csv_text = ("artist,perf_id,youtube_links\nR,00007,https://www.youtube.com/watch?v=2pNrdpv2n0Q\n"
                "S,00008,not a link\n").encode()
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        links = atepp_links(t / "meta.csv", opener=lambda url, timeout=None: Resp(csv_text))
        assert links == {"00007": "https://www.youtube.com/watch?v=2pNrdpv2n0Q"}
        assert link_of("ATEPP_00007", links).endswith("2pNrdpv2n0Q") and link_of("ATEPP_00008", links) is None
        assert link_of("GiantMIDI_9rDGc69FQcY", {}) == "https://www.youtube.com/watch?v=9rDGc69FQcY"
        assert link_of("PERiScoPe_123", links) is None
        d = t / "atepp"
        d.mkdir()
        pd.DataFrame({"id": ["ATEPP_00007", "ATEPP_00008", "GiantMIDI_9rDGc69FQcY"]}).to_parquet(d / "selection.parquet")
        calls = []

        def download(url, stem):
            calls.append(url)
            if "9rDGc69FQcY" in url:
                raise RuntimeError("Video unavailable")
            Path(str(stem) + ".mp3").write_bytes(b"x")
        rep = main(["--dir", str(d), "--atepp-meta", str(t / "meta.csv"), "--sleep", "0"], download=download)
        assert rep == {"ATEPP_00007": "ok", "ATEPP_00008": "no link", "GiantMIDI_9rDGc69FQcY": "unavailable"}
        calls.clear()
        main(["--dir", str(d), "--atepp-meta", str(t / "meta.csv"), "--sleep", "0"], download=download)
        assert calls == []                                       # done or unavailable: nothing again


def test_retranscribe_two_step_alignment():
    """Our transcription of the recording, 2.5 s later on its clock than the original: put back on
    the original's clock (and the recording's offset kept); the score map still from the original, so
    the build gives the same input notes and time map; a transcription of something else is dropped."""
    import pyarrow.parquet as pq

    from scripts.retranscribe_pianocore import main
    from src.preprocessing.performance import load_performance
    from src.render import write_midi
    files = sorted(PC.glob("rows-*.parquet"))
    if not files:
        print("  (skipped: no PianoCoRe rows here)")
        return
    from src.preprocessing.pianocore import load_unfolded_score, pianocore_performance, score_midi_shift
    xml = {r["score_xml_path"]: r for f in PC.glob("scores-*.parquet") for r in pq.read_table(f).to_pylist()}
    row = None
    for f in files[:5]:
        for r in pq.read_table(f).slice(0, 8).to_pylist():
            if r["score_xml_path"] not in xml:
                continue
            ms = load_unfolded_score(xml[r["score_xml_path"]]["score_xml_bytes"], Path(r["score_xml_path"]).suffix)
            shift, match = score_midi_shift(r["refined_score_midi_bytes"], ms)
            if match >= 0.9 and not isinstance(pianocore_performance(r, ms, shift), str):
                row, measures, sh = r, ms, shift
                break
        if row is not None:
            break
    assert row is not None, "no usable PianoCoRe row"
    tmp = Path(tempfile.mkdtemp())
    try:
        src = tmp / "atepp"
        (src / "audio").mkdir(parents=True)
        other = dict(row, id="other")
        import pyarrow as pa
        pq.write_table(pa.Table.from_pylist([dict(row, id="good"), other]), src / "rows-0-0.parquet")
        for pid in ("good", "other"):
            (src / "audio" / f"{pid}.mp3").write_bytes(b"x")
        orig, pedal = load_performance(row["performance_midi_bytes"])

        def transcribe(audio, midi):
            n = orig.copy()
            if Path(audio).stem == "good":
                n[:, 1:3] += 2.5                                  # the recording starts 2.5 s earlier
                keep = np.random.default_rng(0).random(len(n)) > 0.1   # and misses 10% of the notes
                write_midi(n[keep], midi)
            else:                                                 # something else entirely
                rng = np.random.default_rng(1)
                n[:, 0] = rng.integers(21, 109, len(n))
                n[:, 1:3] += rng.uniform(0, 200, (len(n), 1))
                write_midi(n, midi)
        rep = main(["--dir", str(src), "--out", str(tmp / "atepp_tk")], transcribe=transcribe)
        assert rep["good"].startswith("kept") and rep["other"].startswith("no match"), rep
        out = pq.read_table(tmp / "atepp_tk" / "rows-0-0.parquet").to_pylist()
        assert [r["id"] for r in out] == ["good"] and abs(out[0]["audio_offset_s"] - 2.5) < 0.02
        ours, _ = load_performance(out[0]["transcription_midi_bytes"])
        assert abs(ours[:, 1].min() - orig[:, 1].min()) < 0.05        # back on the original's clock
        a = pianocore_performance(row, measures, sh)
        b = pianocore_performance(out[0], measures, sh)
        assert not isinstance(b, str), b
        assert len(b[0]) < len(a[0]) and abs(len(b[0]) - 0.9 * len(a[0])) < 0.05 * len(a[0])   # our notes
        ts = np.linspace(float(orig[:, 1].min()) + 1, float(orig[:, 1].max()) - 1, 7)
        assert np.allclose(np.asarray(a[2].tm(ts)), np.asarray(b[2].tm(ts)))      # the same time map to the score
        assert len(a[2].bars) == len(b[2].bars)
        assert json.loads((tmp / "atepp_tk" / "retranscribe_report.json").read_text())["counts"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_build_reads_extra_folders():
    """load_pianocore over several folders marks the extra ones' rows; pianocore_dirs keeps only
    extra folders that exist."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from scripts.build_dataset import load_pianocore, pianocore_dirs
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        for name, ids in (("main", ["a"]), ("more", ["b"])):
            (t / name).mkdir()
            pq.write_table(pa.Table.from_pylist([{"id": i, "work_key": "W", "score_xml_path": "s.xml", "asap_folders": ""}
                                                 for i in ids]), t / name / "rows-0-0.parquet")
        dirs = pianocore_dirs({"dir": str(t / "main"), "extra_dirs": [str(t / "more"), str(t / "absent")]})
        assert dirs == [str(t / "main"), str(t / "more")]
        by = load_pianocore(dirs)
        assert [(m["id"], m["extra"]) for _, _, m in by["s.xml"]] == [("a", False), ("b", True)]


def test_piece_counts():
    from src.dataset import piece_counts
    meta = [{"composition": "A", "performance": "1"}, {"composition": "A", "performance": "1"},
            {"composition": "A", "performance": "2"}, {"composition": "B", "performance": "3"}]
    assert piece_counts(meta).tolist() == [2, 2, 2, 1]


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
