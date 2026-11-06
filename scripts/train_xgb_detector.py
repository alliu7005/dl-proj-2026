"""XGBoost baseline: detect transcription-error notes (hallucinated / wrong-pitch) from local context.

Two detectors with identical features, evaluated on the same test sets:
  * trained_on_synthetic: clean ASAP train segments + our synthetic noise (labels = injected notes)
  * trained_on_real:      Kong transcriptions of MAESTRO-test recordings (labels from matching
                          against ground truth); recordings that are ASAP test performances excluded
Test sets:
  * synthetic: our noise on the ASAP test split
  * real:      Kong transcriptions of ASAP test performances (test_transcribed); a transcribed note
               is an error if no ground-truth note of the same pitch starts within 50 ms.
               Also reported on the subset whose audio Kong's model never saw.

    python scripts/train_xgb_detector.py

Writes outputs/xgb_<detector>.json (models) and outputs/xgb_results.json (all metrics).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.note_features import FEATURE_NAMES, note_context_features  # noqa: E402
from src.preprocessing.noise import NoiseModel  # noqa: E402
from src.preprocessing.performance import load_performance  # noqa: E402
from src.preprocessing.segment import pitch_f1  # noqa: E402
from src.preprocessing.store import SegmentStore  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402

MATCH_TOL = 0.05


def label_by_matching(pred: np.ndarray, pred_on: np.ndarray, gt: np.ndarray, tol: float = MATCH_TOL):
    """1 = no unused ground-truth note of the same pitch starts within ``tol`` s of the prediction."""
    lab = np.ones(len(pred), np.int8)
    by_pitch: dict[int, list[int]] = {}
    for j, g in enumerate(gt):
        by_pitch.setdefault(int(g[0]), []).append(j)
    used: set[int] = set()
    for k in np.argsort(pred_on):
        best, bd = None, tol
        for j in by_pitch.get(int(pred[k, 0]), ()):
            d = abs(gt[j, 1] - pred_on[k])
            if j not in used and d <= bd:
                best, bd = j, d
        if best is not None:
            used.add(best)
            lab[k] = 0
    return lab


def synthetic(store: SegmentStore, nm: NoiseModel, n_seg: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(store), size=min(n_seg, len(store)), replace=False)
    X, y = [], []
    for i in idx:
        notes, pedal, _ = store.get(i)
        noisy, ped, lab = nm(notes, pedal, rng, return_labels=True)
        X.append(note_context_features(noisy, ped))
        y.append(lab)
    return np.concatenate(X), np.concatenate(y)


def real_asap(store: SegmentStore, asap_root: Path):
    """Real transcriptions of ASAP test performances; labels by matching to the performance MIDI."""
    cache: dict[str, np.ndarray] = {}
    X, y, g, split = [], [], [], []
    for i, m in enumerate(store.meta):
        notes, pedal, _ = store.get(i)
        if m["performance"] not in cache:
            cache[m["performance"]] = load_performance(asap_root / m["performance"])[0]
        X.append(note_context_features(notes, pedal))
        y.append(label_by_matching(notes, notes[:, 1] + m["perf_t0"], cache[m["performance"]]))
        g.append(np.full(len(notes), i))
        split.append(np.full(len(notes), m.get("maestro_split", "?")))
    return np.concatenate(X), np.concatenate(y), np.concatenate(g), np.concatenate(split)


def real_maestro(pairs_csv: str, exclude_gt: set[str], window: float = 8.0):
    """Real Kong errors on MAESTRO-test recordings, cut into windows. Returns X, y, recording id."""
    rows = [r for r in csv.DictReader(open(pairs_csv)) if r["maestro_split"] == "test"
            and not any(r["gt"].endswith(e) for e in exclude_gt)]
    X, y, rec = [], [], []
    for ri, r in enumerate(rows):
        pred, pped = load_performance(r["pred"])
        gt, _ = load_performance(r["gt"])
        lab = label_by_matching(pred, pred[:, 1], gt)
        seg = pred.copy()
        seg[:, 2] = pred[:, 2] - pred[:, 1]  # offset -> duration (segment format)
        for t0 in np.arange(0, seg[:, 1].max() + 1e-6, window):
            m = (seg[:, 1] >= t0) & (seg[:, 1] < t0 + window)
            if m.sum() < 2:
                continue
            w = seg[m].copy()
            w[:, 1] -= t0
            ped = pped[(pped[:, 1] > t0) & (pped[:, 0] < t0 + window)] - t0 if len(pped) else pped
            X.append(note_context_features(w, ped))
            y.append(lab[m])
            rec.append(np.full(m.sum(), ri))
    return np.concatenate(X), np.concatenate(y), np.concatenate(rec), len(rows)


def metrics(y, s, thr=None):
    out = {"n": int(len(y)), "error_rate": float(y.mean()),
           "roc_auc": float(roc_auc_score(y, s)), "avg_precision": float(average_precision_score(y, s))}
    if thr is not None:
        pred = s >= thr
        tp = (pred & (y == 1)).sum()
        out.update(precision=float(tp / max(pred.sum(), 1)), recall=float(tp / max(y.sum(), 1)),
                   flagged=float(pred.mean()))
    return out


def downstream(store, vocab, groups, scores, thr):
    """Mean pitch-F1 (input vs target score) before / after deleting notes scored >= thr."""
    before, after = [], []
    for i in range(len(store)):
        notes, _, tok = store.get(i)
        toks = vocab.decode(tok.tolist())
        sc = scores[groups == i]
        before.append(pitch_f1(notes[:, 0], toks))
        after.append(pitch_f1(notes[sc < thr, 0], toks))
    return {"pitch_f1_before": float(np.mean(before)), "pitch_f1_after": float(np.mean(after))}


def fit(Xtr, ytr, Xva, yva):
    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, eval_metric="aucpr", early_stopping_rounds=100,
        device="cuda", tree_method="hist")
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    s = model.predict_proba(Xva)[:, 1]
    prec, rec, thr = precision_recall_curve(yva, s)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return model, float(thr[np.argmax(f1[:-1])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--asap", default="data/raw/asap_bd")
    ap.add_argument("--noise", default="configs/noise_stats.json")
    ap.add_argument("--pairs", default="data/transcribed/pairs.csv")
    ap.add_argument("--split-file", default="configs/splits/beyer_dai.json")
    ap.add_argument("--train-segments", type=int, default=25000)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nm = NoiseModel.from_json(args.noise)
    nm.cfg.clean_prob = 0.0
    vocab = Vocab.load(data / "vocab.json")

    t = time.time()
    tr, va, te = (SegmentStore(data / s) for s in ("train", "val", "test"))
    Xtr, ytr = synthetic(tr, nm, args.train_segments, 0)
    Xva, yva = synthetic(va, nm, len(va), 1)
    Xte, yte = synthetic(te, nm, len(te), 2)
    real_store = SegmentStore(data / "test_transcribed")
    Xre, yre, gre, sre = real_asap(real_store, Path(args.asap))
    pieces = json.load(open(args.split_file))["pieces"]
    asap_test_audio = {r["maestro_midi_performance"].replace("{maestro}/", "")
                       for r in csv.DictReader(open(Path(args.asap) / "metadata.csv"))
                       if pieces.get(r["folder"]) == "test" and r["maestro_midi_performance"]}
    Xm, ym, rm, n_rec = real_maestro(args.pairs, asap_test_audio)
    val_rec = rm % 5 == 0  # hold out every 5th recording for early stopping / threshold
    print(f"features [{time.time() - t:.0f}s]: synthetic train {Xtr.shape} (errors {ytr.mean():.2%}); "
          f"real MAESTRO-test {Xm.shape} from {n_rec} recordings (errors {ym.mean():.2%}); "
          f"real ASAP test {Xre.shape} (errors {yre.mean():.2%})")

    t = time.time()
    models = {"trained_on_synthetic": fit(Xtr, ytr, Xva, yva),
              "trained_on_real": fit(Xm[~val_rec], ym[~val_rec], Xm[val_rec], ym[val_rec])}
    print(f"trained both in {time.time() - t:.0f}s")

    vi = FEATURE_NAMES.index("velocity")
    unseen = sre != "train"
    res = {"real_test_error_rate": float(yre.mean()), "real_test_notes": int(len(yre)),
           "low_velocity_heuristic": {"synthetic_test": metrics(yte, -Xte[:, vi]),
                                      "real_test": metrics(yre, -Xre[:, vi]),
                                      "real_test_unseen_audio": metrics(yre[unseen], -Xre[unseen, vi])}}
    for name, (model, thr) in models.items():
        model.save_model(out / f"xgb_{name}.json")
        ste, sre_ = model.predict_proba(Xte)[:, 1], model.predict_proba(Xre)[:, 1]
        imp = model.get_booster().get_score(importance_type="gain")
        res[name] = {
            "threshold": thr, "trees": int(model.best_iteration + 1),
            "synthetic_test": metrics(yte, ste, thr),
            "real_test": {**metrics(yre, sre_, thr), **downstream(real_store, vocab, gre, sre_, thr)},
            "real_test_unseen_audio": metrics(yre[unseen], sre_[unseen], thr),
            "top_features": sorted(((FEATURE_NAMES[int(k[1:])], round(v, 1)) for k, v in imp.items()),
                                   key=lambda kv: -kv[1])[:10],
        }
    json.dump(res, open(out / "xgb_results.json", "w"), indent=1)

    print(f"\n{'detector':24s} {'synthetic test':>15s} {'real test':>15s} {'real, unseen audio':>20s}")
    print(f"{'':24s} {'AUC   AP':>15s} {'AUC   AP':>15s} {'AUC   AP':>20s}")
    rows = [("low-velocity heuristic", *res["low_velocity_heuristic"].values())]
    rows += [(n, res[n]["synthetic_test"], res[n]["real_test"], res[n]["real_test_unseen_audio"]) for n in models]
    for n, a, b, c in rows:
        print(f"{n:24s} {a['roc_auc']:8.3f} {a['avg_precision']:.3f} {b['roc_auc']:8.3f} {b['avg_precision']:.3f}"
              f" {c['roc_auc']:13.3f} {c['avg_precision']:.3f}")
    print(f"(AP of random guessing = error rate: synthetic {yte.mean():.3f}, real {yre.mean():.3f}, "
          f"unseen {yre[unseen].mean():.3f})")
    for n in models:
        r = res[n]["real_test"]
        print(f"{n}: flags {r['flagged']:.2%} of real notes, precision {r['precision']:.2f}, recall "
              f"{r['recall']:.2f}; input-vs-score pitch F1 {r['pitch_f1_before']:.4f} -> {r['pitch_f1_after']:.4f}")
        print("   top features:", ", ".join(f for f, _ in res[n]["top_features"][:6]))


if __name__ == "__main__":
    main()
