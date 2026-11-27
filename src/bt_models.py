"""Our Beat This! models: retrained without ASAP, fine-tuned on piano, cross-fitted.

A tracker fine-tuned on a performance tracks it too well: a meter model trained on those beats would
learn to trust beats more than it should (it is tested on music no tracker heard). So every
training piece is given one of K folds (``fold_of``, by piece: all performances of a piece, and all
segments of a performance, together), K models are fine-tuned each leaving one fold out, and every
performance is tracked by the model that never heard its fold. A model fine-tuned on every fold
("full") tracks what none of them trained on: the validation and test pieces, and uploads.

A model set is a folder (outputs/beat_this/<set>/) with models.json:
    {"k": K, "full": "<checkpoint>", "folds": {"0": "<checkpoint>", ...},
     "pieces": {"<piece>": fold, ...}}    # the pieces the fine-tuning trained on, with their fold
Checkpoint paths are relative to the folder (or absolute). scripts/export_beat_this.py writes the
pieces, scripts/train_beat_this.py the checkpoints.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

DEFAULT_K = 4


def fold_of(piece: str, k: int = DEFAULT_K) -> int:
    """The fold of a piece: the same on every machine and every run."""
    return zlib.crc32(str(piece).encode()) % k


def piece_of(perf: dict | str) -> str:
    """The piece a meter-model performance (src/meter.py) belongs to; for an ASAP performance path
    (e.g. Bach/Fugue/bwv_846/Shi05M.mid), its folder."""
    from posixpath import dirname   # '/' on every system: the fold map is shared between machines
    if isinstance(perf, dict):
        return str(perf.get("piece") or dirname(str(perf["performance"]).replace("\\", "/")))
    return dirname(str(perf).replace("\\", "/"))


class ModelSet:
    """``checkpoint_for(perf)``: the fold model that never trained on the performance's piece, the
    full model for a piece the fine-tuning never saw."""

    def __init__(self, folder):
        self.folder = Path(folder)
        meta = json.loads((self.folder / "models.json").read_text())
        self.k = int(meta["k"])
        self.full = self._path(meta["full"])
        self.folds = {int(f): self._path(c) for f, c in meta.get("folds", {}).items()}
        self.pieces = {str(p): int(f) for p, f in meta.get("pieces", {}).items()}
        missing = [str(c) for c in [self.full, *self.folds.values()] if not c.exists()]
        if missing:
            raise FileNotFoundError(f"{self.folder}/models.json: checkpoints missing {missing}")
        absent = sorted(set(self.pieces.values()) - set(self.folds))
        if absent:
            raise ValueError(f"{self.folder}/models.json: no model for folds {absent} of the trained pieces")

    def _path(self, c: str) -> Path:
        p = Path(c)
        return p if p.is_absolute() else self.folder / p

    def checkpoint_for(self, perf: dict | str) -> str:
        fold = self.pieces.get(piece_of(perf))
        return str(self.full if fold is None else self.folds[fold])


def write_models(folder, k: int, full: str, folds: dict, pieces: dict) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "models.json"
    out.write_text(json.dumps({"k": k, "full": str(full), "folds": {str(f): str(c) for f, c in folds.items()},
                               "pieces": pieces}, indent=1))
    return out
