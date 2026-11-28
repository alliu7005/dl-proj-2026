"""Beat This! with a time-signature head (multi-task): the tracker also learns, for every frame, the
time signature of its bar, from the score alignment. Its beats stay the output; the head only
shapes what it learns. A time signature fixes the notated beat (src/beats.py beat_layout: a quarter
in 3/4, a dotted quarter in 6/8, a half in 2/2, an eighth in 3/8), so it carries the beat level the
meter model gets wrong most (whole pieces at half or double note values), which beat times alone do
not. The head is dropped from the inference checkpoint, so everything downstream (the meter model,
the labeller, Beat This!'s own loader) runs on it unchanged.

  * the export (scripts/export_beat_this.py) writes annotations/<dataset>/annotations/meter/<item>.meter
    beside each .beats file: per beat, its time and time-signature class (src/meter.py TS_CLASSES;
    -1 outside them). Beat This!'s own datasets have none: their targets are ignored (-1).
  * ``frame_targets``: per frame of an excerpt, the class of the beat it lies in (-1 outside the
    annotated beats), cut exactly as Beat This! cuts the excerpt.
  * ``patch_dataset``: Beat This!'s dataset class made to load the .meter files and add the targets
    to every item ("meter_ts"; padded with -1). Its tempo augmentation would move the beats but not
    the targets: refused with meter files present.
  * ``multitask_module``: Beat This!'s Lightning module with the head on the transformer's output
    and its cross-entropy added to the loss (``meter_weight``), its accuracy logged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FPS = 50


def n_ts() -> int:
    from src.meter import TS_CLASSES
    return len(TS_CLASSES)


def write_meter(path: Path, t: np.ndarray, ts: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.stack([np.asarray(t, np.float64), np.asarray(ts, np.float64)], 1),
               fmt=["%.4f", "%d"], delimiter="\t")


def read_meter(path: Path):
    a = np.loadtxt(path, ndmin=2)
    return a[:, 0], a[:, 1].astype(np.int64)


def frame_targets(t: np.ndarray, ts: np.ndarray, start_frame: int, n_frames: int, fps: int = FPS) -> np.ndarray:
    """The time-signature class of every frame of the excerpt [start_frame, start_frame + n_frames):
    that of the beat it lies in (from one beat to the next); -1 before the first annotated beat and
    after the last."""
    out = np.full(n_frames, -1, np.int64)
    if len(t) < 2:
        return out
    times = (start_frame + np.arange(n_frames)) / fps
    k = np.searchsorted(np.asarray(t, np.float64), times, side="right") - 1
    ok = (k >= 0) & (k < len(t) - 1)
    out[ok] = np.asarray(ts)[k[ok]]
    return out


def patch_dataset():
    """Beat This!'s BeatTrackingDataset (as its BeatDataModule builds it) loading the .meter files
    and giving every item per-frame "meter_ts". Returns the class it replaced."""
    from .bt_compat import patch_numpy_format
    patch_numpy_format()
    import beat_this.dataset.dataset as bd
    base = bd.BeatTrackingDataset
    if getattr(base, "_meter", False):
        return base

    class MeterTrackingDataset(base):
        _meter = True

        def _load_dataset_item(self, item_name):
            dataset, stem = item_name.split("/", 1)
            f = self.annotation_basepath / dataset / "annotations" / "meter" / f"{stem}.meter"
            # refused before Beat This!'s own loading: with tempo augmentation it skips (returns None) an
            # item whose tempo-stretched spectrograms are missing, and the refusal would never be reached
            if f.exists() and "tempo" in self.augmentations:
                raise ValueError("meter targets with tempo augmentation: the beats move, the targets would not "
                                 "(fine-tune without it: --no-tempo / --finetune)")
            item = super()._load_dataset_item(item_name)
            if item is None:
                return None
            if f.exists():
                item["meter"] = read_meter(f)
            return item

        def __getitem__(self, index):
            if not isinstance(index, (int, np.integer)):
                return [self[i] for i in index]
            out = super().__getitem__(index)
            n = len(out["padding_mask"])
            meter = self.items[index].get("meter")
            ts = np.full(n, -1, np.int64) if meter is None else frame_targets(*meter, int(out["start_frame"]), n, self.fps)
            ts[~np.asarray(out["padding_mask"], bool)] = -1
            out["meter_ts"] = ts
            return out

    bd.BeatTrackingDataset = MeterTrackingDataset
    return base


def add_head(model, transformer_dim: int = 512):
    """The time-signature head on a BeatThis model (its forward then also returns "meter_ts" logits
    [batch, frames, classes]); its weights are model.meter_ts.*."""
    import torch.nn as nn
    if hasattr(model, "meter_ts"):
        return model
    model.meter_ts = nn.Linear(transformer_dim, n_ts())
    cls = type(model)

    class WithMeter(cls):
        def forward(self, x):
            x = self.frontend(x)
            x = self.transformer_blocks(x)
            out = self.task_heads(x)
            out["meter_ts"] = self.meter_ts(x)
            return out
    model.__class__ = WithMeter
    return model


def strip_head(state_dict: dict) -> dict:
    """The weights without the time-signature head (what Beat This!'s inference loads)."""
    return {k: v for k, v in state_dict.items() if ".meter_ts." not in k and not k.startswith("meter_ts.")}


def meter_loss(pred: dict, batch: dict) -> dict:
    """Cross-entropy of the head over the annotated frames (0 where a batch has none), its accuracy,
    and how many frames were annotated."""
    import torch.nn.functional as F
    logits, target = pred["meter_ts"], batch["meter_ts"].long()
    valid = target >= 0
    if valid.any():
        return {"loss": F.cross_entropy(logits[valid], target[valid]),
                "acc": (logits[valid].argmax(-1) == target[valid]).float().mean(), "valid_frames": int(valid.sum())}
    return {"loss": logits.sum() * 0.0, "acc": None, "valid_frames": 0}


def multitask_module():
    """Beat This!'s PLBeatThis with the time-signature head and loss (needs pytorch-lightning)."""
    from beat_this.model.pl_module import PLBeatThis

    class MultiTask(PLBeatThis):
        def __init__(self, *args, meter_weight: float = 0.2, **kw):
            super().__init__(*args, **kw)
            add_head(self.model, kw.get("transformer_dim", 512))
            self.meter_weight = float(meter_weight)

        def _compute_loss(self, batch, model_prediction):
            losses = super()._compute_loss(batch, model_prediction)
            if "meter_ts" in batch and "meter_ts" in model_prediction:
                m = meter_loss(model_prediction, batch)
                losses["total"] = losses["total"] + self.meter_weight * m["loss"]
                step = "train" if self.training else "val"
                n = len(batch["spect"])
                self.log(f"{step}_loss_meter_ts", m["loss"].item(), on_step=False, on_epoch=True, batch_size=n)
                if m["acc"] is not None:
                    self.log(f"{step}_meter_ts_acc", m["acc"].item(), on_step=False, on_epoch=True, batch_size=n)
            return losses
    return MultiTask
