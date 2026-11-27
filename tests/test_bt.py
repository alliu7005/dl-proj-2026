"""Tests for our Beat This! (retrained without ASAP, fine-tuned on piano, cross-fitted): the folds and
model sets (src/bt_models.py), the export of our data in its format (scripts/export_beat_this.py),
the download of its published data (scripts/fetch_beat_this_data.py), tracking with a model set
(scripts/track_meter_data.py --models, scripts/track_eval_beats.py) and its training
(scripts/train_beat_this.py; the training itself only where pytorch-lightning is installed).

    python -m tests.test_bt        # or: pytest tests/test_bt.py

Stand-in renderer, recordings and trackers; Beat This!'s real spectrogram.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_tracked import build, small_release  # noqa: E402

SR = 22050


def tone_wav(path: Path, seconds: float):
    import soundfile as sf
    t = np.arange(int(seconds * SR)) / SR
    x = 0.2 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype(np.float32), SR, format="WAV")


def model_set(folder: Path, k: int, pieces: dict) -> Path:
    from src.bt_models import write_models
    folder.mkdir(parents=True, exist_ok=True)
    for name in ["full", *[f"fold{f}" for f in range(k)]]:
        (folder / f"{name}.ckpt").write_bytes(b"x")
    return write_models(folder, k, "full.ckpt", {f: f"fold{f}.ckpt" for f in range(k)}, pieces)


def test_folds_and_model_sets():
    from src.bt_models import ModelSet, fold_of, piece_of
    assert fold_of("Bach/Fugue/bwv_846", 4) == fold_of("Bach/Fugue/bwv_846", 4) in range(4)
    assert len({fold_of(f"p{i}", 4) for i in range(40)}) == 4                   # every fold used
    assert piece_of("Bach/Fugue/bwv_846/Shi05M.mid") == "Bach/Fugue/bwv_846"
    assert piece_of({"performance": "gangqinpu/7#1", "piece": "gangqinpu/g3"}) == "gangqinpu/g3"
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "set"
        model_set(d, 2, {"A": 0, "B": 1})
        ms = ModelSet(d)
        assert ms.checkpoint_for({"performance": "x", "piece": "A"}).endswith("fold0.ckpt")
        assert ms.checkpoint_for({"performance": "y", "piece": "B"}).endswith("fold1.ckpt")
        assert ms.checkpoint_for({"performance": "z", "piece": "C"}).endswith("full.ckpt")   # never trained on
        (d / "fold1.ckpt").unlink()
        try:
            ModelSet(d)
            raise AssertionError("a missing checkpoint should be refused")
        except FileNotFoundError:
            pass


def test_mix_repeats():
    from scripts.export_beat_this import mix_repeats
    shares = {"classical_real": 1, "classical_render": 1, "pop_real": 1, "pop_render": 1}
    r = mix_repeats({"classical_real": 5, "classical_render": 200, "pop_real": 100, "pop_render": 150}, shares, 4)
    assert r == {"classical_real": 4, "classical_render": 1, "pop_real": 2, "pop_render": 1}, r   # 5 h: capped at 4
    assert mix_repeats({"pop_render": 0.0}, shares, 4) == {}
    r = mix_repeats({"pop_real": 50, "pop_render": 100}, {"pop_real": 2, "pop_render": 1}, 8)
    assert r == {"pop_real": 4, "pop_render": 1}, r                         # a share of 2 against 1


def export(tmp: Path, data: Path, renderer, extra=()):
    from scripts.export_beat_this import main
    return main(["--data", str(data), "--out", str(tmp / "ft"), "--audio", str(tmp / "audio"), "--k", "2",
                 "--datasets", "gangqinpu_render,gangqinpu_real", "--workers", "2", *extra], renderer=renderer)


def test_export():
    """Renders and recordings, cut to the aligned segments, as Beat This! reads them: spectrograms
    in an uncompressed npz, beats with 1 on the downbeats, single.split (train / val), the folds
    (training items only, by piece), pieces.json; run again, nothing is redone."""
    from scripts.export_beat_this import beat_numbers
    from scripts.fetch_gangqinpu_audio import audio_path
    from src.bt_models import fold_of
    assert list(beat_numbers(np.array([0, 1, 0, 0, 1, 0], bool))) == [2, 1, 2, 3, 1, 2]
    tmp = Path(tempfile.mkdtemp())
    try:
        data = build(tmp, small_release(tmp))
        truth = {}
        for split in ("train", "val"):
            for p in pickle.load(open(data / "meter" / f"{split}_gangqinpu.pkl", "rb")):
                truth[p["performance"]] = p
        for split, sid in (("train", "tr1"), ("val", "va0")):                # two recordings downloaded
            tone_wav(audio_path(tmp / "audio", split, sid), truth[f"gangqinpu/{sid}"]["span"][1] + 2)
        calls = []

        def renderer(source, wav, key):
            calls.append(key)
            notes = source[0] if isinstance(source, tuple) else source
            tone_wav(Path(wav), float(np.max(notes[:, 2])) + 1.0)
        rep = export(tmp, data, renderer)
        assert rep["gangqinpu_render"]["items"] == 10 and rep["gangqinpu_real"]["items"] == 2, rep
        out = tmp / "ft"
        for ds in ("gangqinpu_render", "gangqinpu_real"):
            ann = out / "annotations" / ds
            assert json.loads((ann / "info.json").read_text()) == {"has_downbeats": True}
            single = dict(line.split("\t") for line in (ann / "single.split").read_text().split("\n") if line)
            folds = dict(line.split("\t") for line in (ann / "8-folds.split").read_text().split("\n") if line)
            assert set(single.values()) == {"train", "val"} and set(folds) == {s for s, v in single.items() if v == "train"}
            with zipfile.ZipFile(out / "audio" / "spectrograms" / f"{ds}.npz") as z:
                assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
                names = {n.split("/")[0] for n in z.namelist()}
                spect = np.load(io.BytesIO(z.read(next(iter(z.namelist())))))
            assert names == set(single) and spect.dtype == np.float16 and spect.shape[1] == 128
            for stem in single:
                b = np.loadtxt(ann / "annotations" / "beats" / f"{stem}.beats")
                assert b.ndim == 2 and b.shape[1] == 2 and (b[:, 1] == 1).any() and b[0, 0] >= 0
        # the recording's cut: its beats on the cut's clock, as long as the segment
        p = truth["gangqinpu/tr1"]
        b = np.loadtxt(out / "annotations" / "gangqinpu_real" / "annotations" / "beats" / "gangqinpu_tr1.beats")
        assert np.allclose(b[:, 0], np.asarray(p["gt"]["t"]) - p["span"][0], atol=1e-3)
        # the mix: pop recordings (one training item here) listed more often, up to --max-repeat
        rep_json = json.loads((out / "export_report.json").read_text())
        r = rep_json["mix"]["pop_real"]["repeat"]
        assert rep_json["mix"]["pop_render"]["repeat"] == 1 and 1 < r <= 4, rep_json["mix"]
        rows = (out / "annotations" / "gangqinpu_real" / "8-folds.split").read_text().splitlines()
        assert rows.count(f"gangqinpu_tr1\t{fold_of(truth['gangqinpu/tr1']['piece'], 2)}") == r
        val_rows = [x for x in (out / "annotations" / "gangqinpu_real" / "single.split").read_text().splitlines()
                    if x.endswith("\tval")]
        assert val_rows == ["gangqinpu_va0\tval"]                             # validation items once
        assert set(rep_json["folds"]) == {"without fold 0", "without fold 1"}
        pieces = json.loads((out / "pieces.json").read_text())
        assert pieces["k"] == 2 and all(f == fold_of(pc, 2) for pc, f in pieces["pieces"].items())
        assert len(pieces["pieces"]) == 8                                    # the training pieces only
        calls.clear()
        rep = export(tmp, data, renderer)
        assert calls == [] and rep["gangqinpu_render"]["items"] == 0         # nothing redone
        try:
            from scripts.export_beat_this import main
            main(["--data", str(data), "--out", str(out), "--k", "3", "--datasets", "gangqinpu_render"], renderer=renderer)
            raise AssertionError("another k for the same export should be refused")
        except SystemExit as e:
            assert "k=2" in str(e)
        try:   # Beat This!'s own data module reads it (where pytorch-lightning is installed)
            from beat_this.dataset.dataset import BeatDataModule
        except ImportError:
            print("  (BeatDataModule check skipped: no pytorch-lightning)")
            return
        dm = BeatDataModule(out, batch_size=2, num_workers=0, augmentations={}, fold=0)
        dm.setup("fit")
        listed = sum(len((out / "annotations" / ds / "8-folds.split").read_text().splitlines())
                     for ds in ("gangqinpu_render", "gangqinpu_real"))
        assert len(dm.train_items) + len(dm.val_items) == listed            # repeats kept, nothing skipped
        x = dm.train_dataset[0]
        assert x["spect"].shape[-1] == 128
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_track_with_model_set():
    """--models: each Gangqinpu performance tracked by its piece's fold model (recordings and
    renders), the never-trained ones by the full model; stored as beat_this_<tag>; training takes
    those names; track_eval_beats tracks the test recordings with the full model and stops on a
    test piece that was fine-tuned on."""
    from scripts.fetch_gangqinpu_audio import audio_path
    from scripts.track_eval_beats import main as track_eval
    from scripts.track_meter_data import main as track, tracked_path
    from scripts.train_meter import main as train
    from src.beats import load_beats
    from src.bt_models import fold_of
    tmp = Path(tempfile.mkdtemp())
    try:
        data = build(tmp, small_release(tmp))
        train_perfs = pickle.load(open(data / "meter" / "train_gangqinpu.pkl", "rb"))
        pieces = {p["piece"]: fold_of(p["piece"], 2) for p in train_perfs}
        model_set(tmp / "bt_ft", 2, pieces)
        for split, sid in (("train", "tr1"), ("test", "te0")):
            audio_path(tmp / "audio", split, sid).parent.mkdir(parents=True, exist_ok=True)
            audio_path(tmp / "audio", split, sid).write_bytes(b"x")
        used = {}

        def beat_this(path, ck):
            used[Path(path).stem] = Path(ck).stem
            return np.arange(0.0, 60.0, 0.5), np.arange(120) % 4 == 0
        track(["--data", str(data), "--audio", str(tmp / "audio"), "--trackers", "beat_this", "--models",
               str(tmp / "bt_ft"), "--tag", "ft", "--device", "cpu"], beat_this=beat_this)
        tr1 = next(p for p in train_perfs if p["performance"] == "gangqinpu/tr1")
        assert used == {"tr1": f"fold{fold_of(tr1['piece'], 2)}", "te0": "full"}, used
        assert "gangqinpu/tr1" in load_beats(tracked_path(data / "meter", "train_gangqinpu", "beat_this_ft"))
        assert not tracked_path(data / "meter", "train_gangqinpu", "beat_this").exists()   # the released one untouched
        rendered = []

        def render(source, wav, key):
            Path(wav).write_bytes(b"x")

        def tracked(wav, ck):
            rendered.append(Path(ck).stem)
            return np.arange(0.0, 60.0, 0.5), np.arange(120) % 4 == 0
        track(["--data", str(data), "--trackers", "beat_this_render", "--models", str(tmp / "bt_ft"), "--tag", "ft",
               "--splits", "train", "--device", "cpu"], render=render, track=tracked)
        assert sorted(rendered) == sorted(f"fold{fold_of(p['piece'], 2)}" for p in train_perfs)
        assert len(load_beats(tracked_path(data / "meter", "train_gangqinpu", "beat_this_render_ft"))) == 8
        runs = tmp / "runs"
        train(["--name", "ft", "--gangqinpu", "--no-periscope", "--steps", "2", "--eval-every", "2", "--d-model", "32",
               "--layers", "1", "--workers", "0", "--batch", "4", "--chunk", "64", "--device", "cpu", "--normal-priority",
               "--data", str(data), "--runs-dir", str(runs), "--tracked", "beat_this_ft,beat_this_render_ft",
               "--tracked-share", "1"])
        assert json.loads((runs / "ft" / "config.json").read_text())["data"]["tracked_by_source"] == {"gangqinpu": 8}
        try:
            track(["--data", str(data), "--trackers", "beat_this", "--tag", "ft"])
            raise AssertionError("--tag without --models should be refused")
        except SystemExit:
            pass
        # the evaluation's recordings: the full model; a fine-tuned test piece is a leak
        man = tmp / "eval" / "manifest.json"
        man.parent.mkdir()
        man.write_text(json.dumps([{"condition": "gangqinpu", "performance": "gangqinpu/te0", "group": "g9"},
                                   {"condition": "gangqinpu_real", "performance": "gangqinpu/te0", "group": "g9"}]))
        cks = []
        track_eval(["--models", str(tmp / "bt_ft"), "--tag", "ft", "--splits", "", "--gangqinpu", str(man),
                    "--audio", str(tmp / "audio")], track=lambda w, ck: (cks.append(Path(ck).stem), (np.arange(4.0), np.ones(4, bool)))[1])
        assert cks == ["full"] and list(load_beats(man.parent / "beats_gangqinpu_ft.npz")) == ["gangqinpu/te0"]
        model_set(tmp / "leaky", 2, {"gangqinpu/g9": 1})
        try:
            track_eval(["--models", str(tmp / "leaky"), "--tag", "x", "--splits", "", "--gangqinpu", str(man),
                        "--audio", str(tmp / "audio")], track=lambda w, ck: (np.arange(4.0), np.ones(4, bool)))
            raise AssertionError("a fine-tuned test piece should be refused")
        except SystemExit as e:
            assert "leak" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_two_model_sets_one_render():
    """--models a,b --tag ft,ftm: each performance rendered once and tracked by each set's model
    that never heard its piece; recordings tracked by both; a file per set; shards merged per set."""
    from scripts.fetch_gangqinpu_audio import audio_path
    from scripts.track_meter_data import main as track, part_path, tracked_path
    from src.beats import load_beats
    from src.bt_models import fold_of
    tmp = Path(tempfile.mkdtemp())
    try:
        data = build(tmp, small_release(tmp))
        perfs = pickle.load(open(data / "meter" / "train_gangqinpu.pkl", "rb"))
        pieces = {p["piece"]: fold_of(p["piece"], 2) for p in perfs}
        model_set(tmp / "bt_ft", 2, pieces)
        model_set(tmp / "bt_ftm", 2, pieces)
        audio_path(tmp / "audio", "train", "tr1").parent.mkdir(parents=True, exist_ok=True)
        audio_path(tmp / "audio", "train", "tr1").write_bytes(b"x")
        renders, used = [], []

        def render(source, wav, key):
            renders.append(key)
            Path(wav).write_bytes(b"x")

        def tracker(wav, ck):
            used.append(Path(ck).parent.name + "/" + Path(ck).stem)
            return np.arange(0.0, 60.0, 0.5), np.arange(120) % 4 == 0
        common = ["--data", str(data), "--audio", str(tmp / "audio"), "--models", f"{tmp / 'bt_ft'},{tmp / 'bt_ftm'}",
                  "--tag", "ft,ftm", "--device", "cpu", "--splits", "train"]
        track([*common, "--trackers", "beat_this_render", "--shard", "0/1"], render=render, track=tracker)
        assert len(renders) == 8                                            # rendered once each
        assert sorted(used) == sorted(f"{s}/fold{fold_of(p['piece'], 2)}" for p in perfs for s in ("bt_ft", "bt_ftm"))
        for tag in ("ft", "ftm"):
            assert len(load_beats(part_path(tracked_path(data / "meter", "train_gangqinpu", f"beat_this_render_{tag}"), 0, 1))) == 8
        track([*common, "--trackers", "beat_this_render", "--merge", "1"])
        for tag in ("ft", "ftm"):
            assert len(load_beats(tracked_path(data / "meter", "train_gangqinpu", f"beat_this_render_{tag}"))) == 8
        used.clear()
        track([*common, "--trackers", "beat_this"], beat_this=tracker)        # the downloaded recording
        tr1_fold = pieces[next(p["piece"] for p in perfs if p["performance"] == "gangqinpu/tr1")]
        assert sorted(used) == [f"bt_ft/fold{tr1_fold}", f"bt_ftm/fold{tr1_fold}"], used
        for tag in ("ft", "ftm"):
            assert list(load_beats(tracked_path(data / "meter", "train_gangqinpu", f"beat_this_{tag}"))) == ["gangqinpu/tr1"]
        try:
            track([*common[:-4], "--tag", "ft", "--trackers", "beat_this"], beat_this=tracker)
            raise AssertionError("one tag for two sets should be refused")
        except SystemExit as e:
            assert "--tag" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class Served(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch():
    """The record's files: ASAP left out, the annotations laid out whatever folder they are nested
    in, each dataset's npz unpacked, a cut-off download resumed, a corrupt one fetched again, and
    run again, nothing fetched."""
    from scripts.fetch_beat_this_data import RECORD, main
    blobs = {}
    a = io.BytesIO()
    with zipfile.ZipFile(a, "w") as z:
        for ds in ("ballroom", "asap", "gtzan"):
            z.writestr(f"beat_this_annotations-1.0/{ds}/info.json", '{"has_downbeats": true}')
            z.writestr(f"beat_this_annotations-1.0/{ds}/annotations/beats/x.beats", "0.5\t1\n")
    blobs["beat_this_annotations.zip"] = a.getvalue()
    for ds in ("ballroom", "asap"):
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            z.writestr(f"{ds}.npz", f"spectrograms of {ds}" * 100)
        blobs[f"{ds}.zip"] = b.getvalue()
    record = {"files": [{"key": k, "size": len(v), "checksum": "md5:" + hashlib.md5(v).hexdigest(),
                         "links": {"self": f"https://z/{k}"}} for k, v in blobs.items()]}
    calls, cut = [], {"ballroom.zip": True}

    def opener(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if url == RECORD:
            return Served(json.dumps(record).encode())
        name = url.rsplit("/", 1)[1]
        rng = None if isinstance(req, str) else req.get_header("Range")
        calls.append((name, rng))
        data = blobs[name]
        if rng:
            return Served(data[int(rng.split("=")[1].rstrip("-")):], 206)
        if cut.pop(name, False):
            return Served(data[: len(data) // 2])                            # cut off half way
        return Served(data)
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "orig"
        have = main(["--out", str(out)], opener=opener)
        assert have == ["ballroom"] and not (out / "audio" / "spectrograms" / "asap.npz").exists()
        assert (out / "audio" / "spectrograms" / "ballroom.npz").read_text().startswith("spectrograms of ballroom")
        assert sorted(p.name for p in (out / "annotations").iterdir()) == ["asap", "ballroom", "gtzan"]
        assert any(n == "ballroom.zip" and r for n, r in calls)                # resumed with a range
        assert not list(out.glob("*.zip"))                                     # zips removed once unpacked
        calls.clear()
        main(["--out", str(out)], opener=opener)
        assert calls == []


def test_train_beat_this():
    """--fold with --no-val refused (it would train on the held-out fold); --assemble checks the
    runs cover every fold; where pytorch-lightning is installed, a tiny training from a model's
    weights and its final.ckpt loading in Beat This!'s inference."""
    from scripts.train_beat_this import main
    try:
        main(["--name", "x", "--fold", "0", "--no-val"])
        raise AssertionError("--fold with --no-val should be refused")
    except SystemExit as e:
        assert "held-out fold" in str(e)
    tmp = Path(tempfile.mkdtemp())
    try:
        runs = tmp / "runs"
        for r, fold in (("full", None), ("f0", 0), ("f1", 1)):
            (runs / r).mkdir(parents=True)
            (runs / r / "final.ckpt").write_bytes(b"x")
            (runs / r / "status.json").write_text(json.dumps({"fold": fold, "done": True}))
        (tmp / "pieces.json").write_text(json.dumps({"k": 2, "pieces": {"A": 0, "B": 1}}))
        main(["--assemble", "set", "--full", "full", "--fold-runs", "f0,f1", "--pieces", str(tmp / "pieces.json"),
              "--runs-dir", str(runs)])
        m = json.loads((runs / "set" / "models.json").read_text())
        assert m["k"] == 2 and set(m["folds"]) == {"0", "1"} and m["pieces"] == {"A": 0, "B": 1}
        try:
            main(["--assemble", "bad", "--full", "full", "--fold-runs", "f0", "--pieces", str(tmp / "pieces.json"),
                  "--runs-dir", str(runs)])
            raise AssertionError("one fold run for two folds should be refused")
        except SystemExit:
            pass
        try:
            import pytorch_lightning  # noqa: F401
        except ImportError:
            print("  (training skipped: no pytorch-lightning)")
            return
        data = build(tmp, small_release(tmp))

        def renderer(source, wav, key):
            notes = source[0] if isinstance(source, tuple) else source
            tone_wav(Path(wav), float(np.max(notes[:, 2])) + 1.0)
        from scripts.export_beat_this import main as export_main
        export_main(["--data", str(data), "--out", str(tmp / "ft"), "--k", "2", "--datasets", "gangqinpu_render"],
                    renderer=renderer)
        from beat_this.model.pl_module import PLBeatThis
        import torch
        init = tmp / "init.ckpt"
        m = PLBeatThis()
        torch.save({"state_dict": m.state_dict(), "hyper_parameters": dict(m.hparams)}, init)
        main(["--name", "tiny", "--runs-dir", str(runs), "--data", str(tmp / "ft"), "--init", str(init), "--finetune",
              "--fold", "0", "--max-epochs", "1", "--batch-size", "2", "--accumulate", "1", "--train-length", "200",
              "--num-workers", "0", "--device", "cpu"])
        st = json.loads((runs / "tiny" / "status.json").read_text())
        assert st["done"] and st["fold"] == 0 and (runs / "tiny" / "final.ckpt").exists()
        from beat_this.inference import load_model
        load_model(str(runs / "tiny" / "final.ckpt"))
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
