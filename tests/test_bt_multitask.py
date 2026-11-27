"""Tests for the multi-task Beat This! (src/bt_multitask.py): its per-frame time-signature targets,
its head on the real model, its loss, the export's .meter files, and (where pytorch-lightning is
installed: PACE) its dataset.

    python -m tests.test_bt_multitask        # or: pytest tests/test_bt_multitask.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def renderer(source, wav, key):
    from tests.test_bt import tone_wav
    notes = source[0] if isinstance(source, tuple) else source
    tone_wav(Path(wav), float(np.max(notes[:, 2])) + 1.0)


def test_frame_targets():
    from src.bt_multitask import frame_targets
    t = np.array([1.0, 1.5, 2.0, 2.5])
    ts = frame_targets(t, np.array([2, 2, 3, 3]), start_frame=40, n_frames=100)
    times = (40 + np.arange(100)) / 50                                # 0.8 s .. 2.78 s
    assert (ts[times < 1.0] == -1).all() and (ts[times >= 2.5] == -1).all()   # outside the annotated beats
    assert (ts[(times >= 1.0) & (times < 2.0)] == 2).all() and (ts[(times >= 2.0) & (times < 2.5)] == 3).all()
    assert (frame_targets(t[:1], np.array([2]), 0, 10) == -1).all()


def tiny():
    from beat_this.model.beat_tracker import BeatThis
    return BeatThis(transformer_dim=64, n_layers=1, head_dim=32, stem_dim=32,
                    dropout={"frontend": 0.0, "transformer": 0.0})


def test_head_loss_and_stripping():
    """The head on Beat This!'s transformer output; its loss over annotated frames only (0 without
    any); the weights without it load into a plain Beat This!."""
    import torch

    from src.bt_multitask import add_head, meter_loss, n_ts, strip_head
    torch.manual_seed(0)
    m = add_head(tiny(), 64)
    out = m(torch.randn(2, 120, 128))
    assert out["beat"].shape == (2, 120) and out["meter_ts"].shape == (2, 120, n_ts())
    batch = {"meter_ts": torch.full((2, 120), -1)}
    none = meter_loss(out, batch)
    assert float(none["loss"]) == 0.0 and none["acc"] is None and none["valid_frames"] == 0
    batch["meter_ts"][0, :50] = 2
    some = meter_loss(out, batch)
    assert float(some["loss"]) > 0 and some["valid_frames"] == 50
    some["loss"].backward()                                            # gradients reach the head
    assert m.meter_ts.weight.grad is not None and m.meter_ts.weight.grad.abs().sum() > 0
    sd = {f"model.{k}": v for k, v in m.state_dict().items()}          # as in a Lightning checkpoint
    plain = strip_head(sd)
    assert not [k for k in plain if "meter" in k] and len(plain) == len(sd) - 2
    tiny().load_state_dict({k[len("model."):]: v for k, v in plain.items()})   # strict: nothing missing or extra


def test_export_writes_meter_targets():
    """Each exported item's .meter file: the same beats as its .beats file, the time signature of
    its bar (3/4 here)."""
    from scripts.export_beat_this import main
    from src.meter import TS_CLASSES
    from tests.test_tracked import build, small_release
    tmp = Path(tempfile.mkdtemp())
    try:
        data = build(tmp, small_release(tmp))
        main(["--data", str(data), "--out", str(tmp / "ft"), "--k", "2", "--datasets", "gangqinpu_render"],
             renderer=renderer)
        ann = tmp / "ft" / "annotations" / "gangqinpu_render" / "annotations"
        m = np.loadtxt(ann / "meter" / "gangqinpu_tr0.meter", ndmin=2)
        b = np.loadtxt(ann / "beats" / "gangqinpu_tr0.beats", ndmin=2)
        assert m.shape[1] == 2 and np.allclose(m[:, 0], b[:, 0])     # the same beats
        assert set(m[:, 1].astype(int)) == {TS_CLASSES.index("3/4")}
        assert len(list((ann / "meter").glob("*.meter"))) == len(list((ann / "beats").glob("*.beats")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dataset():
    """Where pytorch-lightning is installed (PACE): the patched dataset gives every item its frames'
    time-signature targets, the head's loss on a real batch is finite, and tempo augmentation is
    refused."""
    try:
        import pytorch_lightning  # noqa: F401
    except ImportError:
        print("  (skipped: no pytorch-lightning)")
        return
    import torch

    from scripts.export_beat_this import main as export
    from src.bt_multitask import add_head, meter_loss, patch_dataset
    from tests.test_tracked import build, small_release
    tmp = Path(tempfile.mkdtemp())
    try:
        data = build(tmp, small_release(tmp))
        export(["--data", str(data), "--out", str(tmp / "ft"), "--k", "2", "--datasets", "gangqinpu_render"],
               renderer=renderer)
        patch_dataset()
        from beat_this.dataset.dataset import BeatDataModule
        dm = BeatDataModule(tmp / "ft", batch_size=2, num_workers=0, train_length=300, augmentations={}, fold=0)
        dm.setup("fit")
        item = dm.train_dataset[0]
        assert item["meter_ts"].shape == (300,) and (item["meter_ts"] >= 0).any()
        batch = next(iter(dm.train_dataloader()))
        m = meter_loss(add_head(tiny(), 64)(batch["spect"].float()), batch)
        assert torch.isfinite(m["loss"]) and m["valid_frames"] > 0
        try:
            BeatDataModule(tmp / "ft", batch_size=2, num_workers=0, train_length=300,
                           augmentations={"tempo": {"min": -4, "max": 4, "stride": 4}}, fold=0).setup("fit")
            raise AssertionError("tempo augmentation with meter targets should be refused")
        except ValueError as e:
            assert "tempo" in str(e)
        print("  dataset and head ok;", json.dumps({"valid_frames": m["valid_frames"]}))
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
