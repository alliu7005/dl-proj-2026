"""Baseline 2: beat-snap quantizer + per-note XGBoost corrections (lightweight supervised model).

Starts from the beat-snap placement (``quantize.analyze_window``) and learns four per-note
classifiers from the training split:
  keep    : is this performed note a score note? (drops hallucinations, extra trill/tremolo notes)
  staff   : upper / lower staff
  dur     : written duration class (instead of snapping the performed duration)
  shift   : onset correction relative to the snapped position (e.g. 16th vs triplet, rubato)

Labels come from matching each snapped note to the ground-truth target of its window (same pitch,
nearest onset within half a quarter, one-to-one). Training inputs: clean train windows with
synthetic noise, plus real Kong transcriptions (train_transcribed).
Features: the 56 local-context features of the error detector + placement features.
"""
from __future__ import annotations

import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import xgboost as xgb

from ..eval.metrics import flatten
from .note_features import note_context_features
from .quantize import PieceTiming, analyze_window, render_window

DUR_CLASSES = [Fraction(x) for x in ("1/8", "1/6", "1/4", "1/3", "3/8", "1/2", "2/3", "3/4", "1",
                                     "3/2", "2", "3", "4", "6")]
SHIFTS = [Fraction(x) for x in ("-1/2", "-1/3", "-1/4", "-1/6", "-1/8", "-1/12", "0", "1/12", "1/8",
                                "1/6", "1/4", "1/3", "1/2")]
MATCH_TOL = Fraction(1, 2)


def _nearest(x: Fraction, classes) -> int:
    return int(np.argmin([abs(float(x - c)) for c in classes]))


def placement_features(notes: np.ndarray, a: dict, timing: PieceTiming) -> np.ndarray:
    """Per in-window note: position in bar (raw / snapped / residual), position within the beat,
    grid type, performed & snapped duration (log quarters), bar length, meter."""
    rows = []
    for k, pr, p, dr, d in zip(a["bar"], a["pos_raw"], a["pos"], a["dur_raw"], a["dur"]):
        ts = timing.ts(k)
        num, den = (int(x) for x in ts.split("/"))
        pf = float(p)
        rows.append([pr, pf, pr - pf, pf % 1.0, float(p.denominator == 3), np.log(max(dr, 1e-3)),
                     np.log(float(d)), float(timing.lengths[k]), num, den])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 10)


def window_features(notes, pedal, meta, timing):
    a = analyze_window(notes, meta, timing)
    ctx = note_context_features(notes, pedal)[a["idx"]] if len(a["idx"]) else np.zeros((0, 56), np.float32)
    return a, np.concatenate([ctx, placement_features(notes, a, timing)], axis=1)


def window_labels(notes, a, gt_tokens):
    """keep / staff / duration class / shift class per in-window note (-1 = not applicable)."""
    gt_notes = [n for n in flatten(gt_tokens)[0] if not n[4]]  # (onset, pitch, dur, staff, grace, ...)
    by_pitch = defaultdict(list)
    for j, n in enumerate(gt_notes):
        by_pitch[n[1]].append(j)
    used = set()
    n_in = len(a["idx"])
    keep = np.zeros(n_in, np.int32)
    staff, dur, shift = (np.full(n_in, -1, np.int32) for _ in range(3))
    order = np.argsort([float(x) for x in a["abs"]], kind="stable")
    for t in order:
        pitch, onset = int(notes[a["idx"][t], 0]), a["abs"][t]
        best, bd = None, MATCH_TOL
        for j in by_pitch.get(pitch, ()):
            d = abs(gt_notes[j][0] - onset)
            if j not in used and d <= bd:
                best, bd = j, d
        if best is None:
            continue
        used.add(best)
        g = gt_notes[best]
        keep[t] = 1
        staff[t] = min(g[3], 1)
        dur[t] = _nearest(g[2], DUR_CLASSES)
        shift[t] = _nearest(g[0] - onset, SHIFTS)
    return keep, staff, dur, shift


class XGBQuantizer:
    def __init__(self, models: dict | None = None, keep_threshold: float = 0.5):
        self.models = models or {}
        self.keep_threshold = keep_threshold

    @staticmethod
    def _clf(n_class: int):
        kw = dict(n_estimators=400, learning_rate=0.1, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                  device="cuda", tree_method="hist")
        return xgb.XGBClassifier(**kw) if n_class == 2 else xgb.XGBClassifier(objective="multi:softprob", **kw)

    def fit(self, X, labels: dict):
        for name, y in labels.items():
            m = y >= 0
            classes = np.unique(y[m])
            remap = {c: i for i, c in enumerate(classes)}
            clf = self._clf(len(classes))
            clf.fit(X[m], np.array([remap[c] for c in y[m]]))
            self.models[name] = (clf, classes)
        return self

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        meta = {}
        for name, (clf, classes) in self.models.items():
            clf.save_model(d / f"{name}.json")
            meta[name] = classes.tolist()
        json.dump({"classes": meta, "keep_threshold": self.keep_threshold}, open(d / "meta.json", "w"))

    @classmethod
    def load(cls, d: Path) -> "XGBQuantizer":
        meta = json.load(open(d / "meta.json"))
        models = {}
        for name, classes in meta["classes"].items():
            clf = cls._clf(len(classes))
            clf.load_model(d / f"{name}.json")
            models[name] = (clf, np.array(classes))
        return cls(models, meta["keep_threshold"])

    def _predict(self, name, X):
        clf, classes = self.models[name]
        return classes[np.argmax(clf.predict_proba(X), axis=1)], clf.predict_proba(X)

    def predict_window(self, notes, pedal, meta, timing) -> list[str]:
        a, X = window_features(notes, pedal, meta, timing)
        if not len(X):
            return render_window([], pedal, meta, timing)
        keep_p = self.models["keep"][0].predict_proba(X)[:, -1]
        staff, _ = self._predict("staff", X)
        dur, _ = self._predict("dur", X)
        shift, _ = self._predict("shift", X)
        placed = []
        for t, i in enumerate(a["idx"]):
            if keep_p[t] < self.keep_threshold:
                continue
            k, p = a["bar"][t], a["pos"][t] + SHIFTS[int(shift[t])]
            # a shift across a bar line moves the note to the neighbouring bar
            while p < 0 and k > int(meta["bar_start"]):
                k -= 1
                p += timing.lengths[k]
            while p >= timing.lengths[k] and k + 1 < len(timing.lengths):
                p -= timing.lengths[k]
                k += 1
            p = min(max(p, Fraction(0)), max(timing.lengths[k] - Fraction(1, 48), Fraction(0)))
            placed.append((k, p, int(notes[i, 0]), DUR_CLASSES[int(dur[t])], int(staff[t])))
        return render_window(placed, pedal, meta, timing)
