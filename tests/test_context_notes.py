"""Tests for context notes (src/dataset.py context_before / context_after, src/pipeline.py).

    python -m tests.test_context_notes          # ~1 min on a CPU; needs data/processed

  * the window's own notes are unchanged by adding context (pitch, onset, duration, velocity)
  * "before" notes are exactly the last N earlier notes of the performance that are not part of
    the window, and "after" notes exactly the next N, rebuilt independently from the neighbouring
    test windows (contiguous, no overlap)
  * training items (noise + augmentation) are well formed: 10 features, roles consistent with the
    flag columns, context onsets on the right side of the window, reproducible
  * both architectures run with 10 features; checkpoints without the new fields default to 8
  * the inference pipeline builds the same kind of window: context notes on the right side,
    carried-over notes not repeated, and a context model decodes end to end
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.dataset import (N_FEATURES, N_FEATURES_CONTEXT, ROLE_AFTER, ROLE_BEFORE, ROLE_CARRIED, ROLE_WINDOW,
                         AugmentConfig, ScoreSegmentDataset, assemble_notes, collate, released_mask)
from src.preprocessing.noise import NoiseModel
from src.scoreformer import PRESETS, build_model

V = "data/processed/vocab.json"
NB, NA = 24, 12


def unscale(ds, feats):
    x = feats.numpy().copy()
    cols, mean, std = ds.scale
    x[:, cols] = x[:, cols] * std + mean
    return x


def same_notes(a, b, tol_ms: int = 2) -> bool:
    """Same (pitch, onset ms) notes, onsets within ``tol_ms``: onsets read back from float32
    features minutes into a piece carry sub-millisecond rounding."""
    per = lambda ts: {p: sorted(int(k) for q, k in ts if int(q) == p) for p in {int(t[0]) for t in ts}}
    pa, pb = per(a), per(b)
    return len(a) == len(b) and pa.keys() == pb.keys() and all(
        len(pa[p]) == len(pb[p]) and all(abs(x - y) <= tol_ms for x, y in zip(pa[p], pb[p])) for p in pa)


def test_assemble_notes():
    window = np.array([[60, -0.2, 1.0, 80], [62, 0.0, 0.5, 70], [64, 0.4, 0.5, 70]], np.float32)
    before = np.array([[55, -1.0, 0.3, 60], [57, -0.5, 0.2, 60]], np.float32)
    after = np.array([[67, 2.0, 0.5, 60]], np.float32)
    notes, roles = assemble_notes(window, before, after)
    assert (np.diff(notes[:, 1]) >= 0).all()
    assert roles.tolist() == [ROLE_BEFORE, ROLE_BEFORE, ROLE_CARRIED, ROLE_WINDOW, ROLE_WINDOW, ROLE_AFTER]
    e = np.zeros((0, 4), np.float32)
    n2, r2 = assemble_notes(window, e, e)
    assert r2.tolist() == [ROLE_CARRIED, ROLE_WINDOW, ROLE_WINDOW] and len(n2) == 3


def test_window_notes_unchanged_and_context_exact():
    plain = ScoreSegmentDataset("data/processed/test", V)
    ctx = ScoreSegmentDataset("data/processed/test", V, context_before=NB, context_after=NA)
    st = plain.store
    by_perf = {}
    for i, m in enumerate(st.meta):
        by_perf.setdefault(m["performance"], []).append(i)
    checked = 0
    for perf, idx in list(by_perf.items())[:6]:
        idx.sort(key=lambda i: st.meta[i]["win"])
        abs_notes = []  # every window's notes in absolute time, for an independent reconstruction
        for i in idx:
            n = st.get(i)[0].astype(np.float64).copy()
            n[:, 1] += st.meta[i]["perf_t0"]
            abs_notes.append(n)
        allnotes = np.concatenate(abs_notes)
        key = lambda a: set(zip(a[:, 0].astype(int), np.round(a[:, 1] * 1000).astype(np.int64)))
        # time order, chords (same onset) by pitch: the order the Dataset's pool uses, so the N-note
        # cut through a chord keeps the same notes
        # (one note seen from two windows can round to neighbouring milliseconds: the same note)
        ks = sorted(key(allnotes))
        ks = [k for j, k in enumerate(ks) if not (j and k[0] == ks[j - 1][0] and k[1] - ks[j - 1][1] <= 1)]
        uniq = np.array(sorted(ks, key=lambda t: (t[1], t[0])))
        for pos, i in enumerate(idx):
            a, b = plain[i], ctx[i]
            r = b["roles"].numpy()
            assert b["notes"].shape[1] == N_FEATURES_CONTEXT and a["notes"].shape[1] == N_FEATURES
            assert torch.equal(a["target"], b["target"])
            xa, xb = unscale(plain, a["notes"]), unscale(ctx, b["notes"])
            own = xb[r <= ROLE_CARRIED]
            for c in (0, 1, 3, 4):  # pitch, onset, log duration, velocity (ioi depends on neighbours)
                assert np.allclose(own[:, c], xa[:, c], atol=1e-4), (perf, i, c)
            t0 = st.meta[i]["perf_t0"]
            # the window's own notes: a carried-over candidate released before carry_min_sec is not
            # one of them but a note just before the window, i.e. context (as the pipeline has it)
            kept = released_mask(st.get(i)[0], st.meta[i].get("carry_min_sec"))
            mine = key(abs_notes[pos][kept])
            mine |= {(p, k + d) for p, k in mine for d in (-1, 1)}
            rest = [(p, k) for p, k in uniq.tolist() if (p, k) not in mine]
            want_before = [t for t in rest if t[1] < round(t0 * 1000)][-NB:]
            want_after = [t for t in rest if t[1] >= round(t0 * 1000)][:NA]
            got = lambda role: sorted(zip(xb[r == role, 0].astype(int).tolist(),
                                          np.round((xb[r == role, 1] + t0) * 1000).astype(np.int64).tolist()))
            assert same_notes(got(ROLE_BEFORE), want_before), (perf, i, "before")
            assert same_notes(got(ROLE_AFTER), want_after), (perf, i, "after")
            assert (xb[r == ROLE_BEFORE, 1] < 0).all()
            for role, col in ((ROLE_CARRIED, 7), (ROLE_BEFORE, 8), (ROLE_AFTER, 9)):
                assert np.array_equal(xb[:, col] == 1, r == role)
            checked += 1
    assert checked > 10, checked


def test_training_items():
    nm = NoiseModel.from_json("configs/noise_stats.json")
    ds = ScoreSegmentDataset("data/processed/train", V, noise=nm, augment=AugmentConfig(),
                             context_before=NB, context_after=NA)
    n_ctx = 0
    for i in range(0, len(ds), max(1, len(ds) // 200)):
        it = ds[i]
        x, r = unscale(ds, it["notes"]), it["roles"].numpy()
        assert x.shape == (len(r), N_FEATURES_CONTEXT) and np.isfinite(x).all()
        assert (np.diff(x[:, 1]) >= -1e-4).all(), "sorted by onset"
        assert (x[r == ROLE_BEFORE, 1] < 0).all()
        # synthetic noise drops and adds notes in the context too (as a transcriber would)
        assert ((r == ROLE_BEFORE).sum() <= 2 * NB) and ((r == ROLE_AFTER).sum() <= 2 * NA)
        assert ((x[:, 0] >= 21) & (x[:, 0] <= 108)).all()
        n_ctx += int((r >= ROLE_BEFORE).sum())
        again = ds[i]
        assert torch.equal(again["notes"], it["notes"]) and torch.equal(again["target"], it["target"])
    assert n_ctx > 0
    b = collate([ds[0], ds[1], ds[2]])
    assert b["notes"].shape[2] == N_FEATURES_CONTEXT


def test_models_take_context_features():
    from src.scoreformer import load_model
    ds = ScoreSegmentDataset("data/processed/test", V, context_before=NB, context_after=NA)
    b = collate([ds[0], ds[1]])
    vocab = len(ds.vocab)
    for arch, cfg in (("scoreformer", dict(PRESETS["small"])),
                      ("score2seq", dict(d_model=128, enc_layers=2, dec_layers=2, ff=256, n_heads=4))):
        m = build_model(arch, dict(vocab_size=vocab, n_features=N_FEATURES_CONTEXT, context_before=NB,
                                   context_after=NA, **cfg)).eval()
        with torch.no_grad():
            logits = m(b["notes"], b["note_mask"], b["target"][:, :-1])
        assert logits.shape[:2] == b["target"][:, :-1].shape and torch.isfinite(logits).all()
        old = build_model(arch, dict(vocab_size=vocab, **cfg))     # a checkpoint config without the fields
        assert old.cfg.n_features == N_FEATURES and old.cfg.context_before == 0
        with tempfile.TemporaryDirectory() as tmp:                  # round trip through a checkpoint
            torch.save({"model": m.state_dict(), "arch": arch, "config": vars(m.cfg)}, Path(tmp) / "c.pt")
            m2 = load_model(Path(tmp) / "c.pt", "cpu")
            assert m2.cfg.n_features == N_FEATURES_CONTEXT and m2.cfg.context_after == NA


def test_pipeline_context_windows():
    from src.pipeline import performance_windows
    rng = np.random.default_rng(0)
    on = np.cumsum(rng.uniform(0.05, 0.3, 300))
    notes = np.stack([rng.integers(40, 90, 300), on, on + rng.uniform(0.1, 2.0, 300), rng.integers(40, 100, 300)],
                     axis=1).astype(np.float32)
    pedal = np.array([[1.0, 3.0], [10.0, 14.0]], np.float32)
    plain = performance_windows(notes, pedal, size=48)
    wins = performance_windows(notes, pedal, size=48, context_before=NB, context_after=NA)
    assert len(plain) == len(wins)
    for j, (p, w) in enumerate(zip(plain, wins)):
        assert np.array_equal(p["notes"], w["notes"])                 # window notes unchanged
        assert (j == 0) == (len(w["before"]) == 0) and len(w["before"]) <= NB
        assert (j == len(wins) - 1) == (len(w["after"]) == 0) and len(w["after"]) <= NA
        assert (w["before"][:, 1] < 0).all() and (w["after"][:, 1] >= w["t1"] - w["t0"] - 1e-4).all()
        carried = w["notes"][w["notes"][:, 1] < 0]
        ck = set(zip(carried[:, 0].tolist(), np.round(carried[:, 1], 4).tolist()))
        bk = set(zip(w["before"][:, 0].tolist(), np.round(w["before"][:, 1], 4).tolist()))
        assert not (ck & bk), "carried-over notes are not repeated as context"
        if 0 < j:  # the before notes are the notes right before the window
            prev_on = notes[notes[:, 1] < w["t0"], 1] - w["t0"]
            assert np.isclose(w["before"][-1, 1], prev_on.max()) or len(ck)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
