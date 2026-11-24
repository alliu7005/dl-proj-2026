"""Tests for the metrical labels (src/metrical.py) used by the auxiliary training target.

    python -m tests.test_metrical          # ~30 s on a CPU; needs data/processed

  * phase classes and score onsets parse correctly (tie continuations and grace notes left out)
  * on performances rendered from real windows' scores with known labels, strong rubato, hand
    asynchrony, dropped and extra notes, the aligner's labels are almost always right (precision)
    and cover almost all notes (recall)
  * on the real data (ASAP test / train windows) most notes get a label
  * transfer() carries labels through synthetic noise
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from src.metrical import IGNORE, align, phase_class, score_onsets, transfer
from src.preprocessing.noise import NoiseModel
from src.preprocessing.store import SegmentStore
from src.preprocessing.vocab import Vocab

V = "data/processed/vocab.json"


def test_phase_and_onsets():
    assert [phase_class(Fraction(x)) for x in ("0", "2", "1/2", "5/2", "1/3", "2/3", "1/4", "3/4", "1/8", "1/6")] \
        == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    toks = ["bar", "ts_3/4", "key_0", "pos_0", "p_60", "d_1", "p_64", "d_1", "tie",
            "pos_1", "sub_1/2", "p_62", "grace", "p_67", "d_1/2", "bar", "pos_0", "p_65", "d_3"]
    on = score_onsets(toks)
    # the tied 64 and the grace 62 start no new sound; the second bar starts 3 quarters later
    assert on == [(0.0, 60, 1, 0), (1.5, 67, 0, 1), (3.0, 65, 1, 0)]
    # a window starting mid-bar: positions still count from that bar's line
    assert score_onsets(["ts_4/4", "key_0", "pos_2", "p_60", "d_1", "bar", "pos_0", "p_62", "d_1"]) == \
        [(2.0, 60, 0, 0), (4.0, 62, 1, 0)]


def render(tokens, rng, drop=0.03, extra=0.03):
    """A synthetic performance of a window's score: tempo drifting +-30%, chord notes up to 40 ms
    apart, 3% of notes dropped and 3% extra notes. Returns (onsets, pitches, true downbeat, true phase)."""
    on = score_onsets(tokens)
    q = np.array([x[0] for x in on])
    spq = 0.5 * np.exp(np.cumsum(rng.normal(0, 0.04, len(q))))           # seconds per quarter, drifting
    uq = np.unique(q)
    tq = np.concatenate([[0], np.cumsum(np.diff(uq) * np.interp(uq[1:], q, spq))])
    t = np.interp(q, uq, tq) + rng.uniform(-0.02, 0.02, len(q))          # hand / chord asynchrony
    keep = rng.random(len(q)) >= drop
    t, p = t[keep], np.array([x[1] for x in on])[keep]
    d, ph = np.array([x[2] for x in on])[keep], np.array([x[3] for x in on])[keep]
    n_extra = rng.binomial(len(t), extra)
    et = rng.uniform(t.min(), t.max(), n_extra)
    ep = rng.integers(21, 109, n_extra)
    order = np.argsort(np.concatenate([t, et]), kind="stable")
    t = np.concatenate([t, et])[order] - min(t.min(), et.min() if n_extra else t.min())
    p = np.concatenate([p, ep])[order]
    d = np.concatenate([d, np.full(n_extra, IGNORE)])[order]
    ph = np.concatenate([ph, np.full(n_extra, IGNORE)])[order]
    return t, p, d, ph


def test_alignment_recovers_known_labels():
    vocab = Vocab.load(V)
    store = SegmentStore("data/processed/test")
    rng = np.random.default_rng(0)
    right = wrong = real = 0
    for i in rng.choice(len(store), 150, replace=False):
        toks = vocab.decode(store.get(int(i))[2].tolist())
        if len(score_onsets(toks)) < 8:
            continue
        t, p, d_true, ph_true = render(toks, rng)
        d, ph = align(t, p, toks)
        lab = (d != IGNORE) & (d_true != IGNORE)
        ok = (d == d_true) & (ph == ph_true)
        right += int((lab & ok).sum())
        wrong += int((lab & ~ok).sum()) + int(((d != IGNORE) & (d_true == IGNORE)).sum())  # labelled an extra note
        real += int((d_true != IGNORE).sum())
    precision, recall = right / (right + wrong), right / real
    print(f"    synthetic performances: label precision {precision:.3f}, recall {recall:.3f}")
    assert precision > 0.97 and recall > 0.9, (precision, recall)


def test_real_data_coverage():
    """Most performed notes get a label: on clean ASAP windows (the performance MIDI) above a firm
    floor; on real transcriptions (PERiScoPe, *_transcribed: missed and extra notes) reported, with
    a loose sanity floor."""
    vocab = Vocab.load(V)
    for split, floor in (("test", 0.85), ("train", 0.88)):
        store = SegmentStore(f"data/processed/{split}")
        by_source = {}
        for i, m in enumerate(store.meta):
            by_source.setdefault("clean" if m.get("source", "clean") == "clean" else "real", []).append(i)
        rng = np.random.default_rng(1)
        for source, idx in by_source.items():
            lab = tot = 0
            for i in rng.choice(idx, min(200, len(idx)), replace=False):
                notes, _, toks = store.get(int(i))
                d, ph = align(notes[:, 1], notes[:, 0], vocab.decode(toks.tolist()))
                live = notes[:, 1] >= 0
                assert ((d == IGNORE) == (ph == IGNORE)).all() and (d[~live] == IGNORE).all()
                lab += int((d[live] != IGNORE).sum())
                tot += int(live.sum())
            print(f"    {split} ({source}): {lab / tot:.1%} of performed window notes labelled")
            assert lab / tot > (floor if source == "clean" else 0.5), (split, source, lab / tot)


def test_transfer_through_noise():
    vocab = Vocab.load(V)
    store = SegmentStore("data/processed/train")
    nm = NoiseModel.from_json("configs/noise_stats.json")
    nm.cfg.clean_prob = 0.0
    rng = np.random.default_rng(2)
    kept = total = 0
    for i in rng.choice(len(store), 60, replace=False):
        notes, pedal, toks = store.get(int(i))
        d, ph = align(notes[:, 1], notes[:, 0], vocab.decode(toks.tolist()))
        noisy, _ = nm(notes, pedal, rng)
        d2, ph2 = transfer((d, ph), notes, noisy)
        assert len(d2) == len(noisy) and ((d2 == IGNORE) == (ph2 == IGNORE)).all()
        assert set(np.unique(d2)) <= {IGNORE, 0, 1}
        kept += int((d2 != IGNORE).sum())
        total += int((d != IGNORE).sum())
    print(f"    labels kept through synthetic noise: {kept / total:.1%}")
    assert kept / total > 0.85, kept / total


def test_dataset_labels():
    """Labels in Dataset items: the precomputed file equals on-the-fly alignment, labels stay on
    their notes through context assembly, noise and augmentation, and collation pads them."""
    import torch

    from src.dataset import ROLE_CARRIED, AugmentConfig, ScoreSegmentDataset, collate
    from src.metrical import LABEL_FILE, align as align_
    from pathlib import Path
    has_file = (Path("data/processed/test") / LABEL_FILE).exists()
    fly = ScoreSegmentDataset("data/processed/test", V, metrical=True)
    fly._mfile = None  # force on-the-fly alignment
    filed = ScoreSegmentDataset("data/processed/test", V, metrical=True)
    for i in range(0, len(fly), max(1, len(fly) // 60)):
        a, b = fly[i], filed[i]
        assert torch.equal(a["downbeat"], b["downbeat"]) and torch.equal(a["phase"], b["phase"])
    print(f"    precomputed labels {'used and' if has_file else 'absent; on-the-fly'} identical to on-the-fly alignment")
    # with context notes: window notes keep their labels, context and carried notes get -1
    ctx = ScoreSegmentDataset("data/processed/test", V, metrical=True, context_before=16, context_after=8)
    plain = ScoreSegmentDataset("data/processed/test", V, metrical=True)
    for i in range(0, len(ctx), max(1, len(ctx) // 60)):
        a, b = plain[i], ctx[i]
        r = b["roles"]
        assert (b["downbeat"][r > ROLE_CARRIED] == -1).all() and (b["phase"][r > ROLE_CARRIED] == -1).all()
        assert torch.equal(b["downbeat"][r <= ROLE_CARRIED], a["downbeat"])  # same notes, same order
        assert torch.equal(b["phase"][r <= ROLE_CARRIED], a["phase"])
    # training: noise and augmentation; every label sits on a note that could carry it
    nm = NoiseModel.from_json("configs/noise_stats.json")
    tr = ScoreSegmentDataset("data/processed/train", V, noise=nm, augment=AugmentConfig(), metrical=True,
                             context_before=16, context_after=8)
    n_lab = 0
    for i in range(0, len(tr), max(1, len(tr) // 150)):
        it = tr[i]
        assert len(it["downbeat"]) == len(it["notes"]) == len(it["phase"])
        lab = it["downbeat"] >= 0
        assert torch.equal(lab, it["phase"] >= 0) and (it["roles"][lab] == 0).all()
        assert set(it["downbeat"].unique().tolist()) <= {-1, 0, 1} and int(it["phase"].max()) < 5
        n_lab += int(lab.sum())
        again = tr[i]
        assert torch.equal(again["downbeat"], it["downbeat"])
    assert n_lab > 0
    b = collate([tr[0], tr[1], tr[2]])
    assert b["downbeat"].shape == b["note_mask"].shape and (b["downbeat"][~b["note_mask"]] == -1).all()
    from scripts.train import TruncatingCollate
    tb = TruncatingCollate(20, 4096)([dict(tr[0]), dict(tr[1])])
    assert tb["notes"].shape[1] == tb["downbeat"].shape[1] == 20


def test_aux_loss_and_heads():
    import torch

    from src.dataset import ScoreSegmentDataset, collate
    from src.metrical import N_PHASE, aux_counts, aux_loss, aux_scores
    from src.scoreformer import PRESETS, build_model
    ds = ScoreSegmentDataset("data/processed/test", V, metrical=True)
    b = collate([ds[0], ds[1]])
    for arch, cfg in (("scoreformer", dict(PRESETS["small"])),
                      ("score2seq", dict(d_model=128, enc_layers=2, dec_layers=2, ff=256, n_heads=4))):
        m = build_model(arch, dict(vocab_size=len(ds.vocab), aux_metrical=True, **cfg))
        logits, aux = m(b["notes"], b["note_mask"], b["target"][:, :-1], return_aux=True)
        assert aux.shape == (*b["notes"].shape[:2], 1 + N_PHASE)
        loss = aux_loss(aux, b["downbeat"], b["phase"])
        loss.backward()
        assert torch.isfinite(loss) and loss.item() > 0
        enc_grad = [p.grad for n, p in m.named_parameters() if ("encoder" in n or "enc" in n) and p.grad is not None]
        assert enc_grad and any(g.abs().sum() > 0 for g in enc_grad), "aux loss reaches the encoder"
        plain = build_model(arch, dict(vocab_size=len(ds.vocab), **cfg))
        assert plain.aux is None and plain(b["notes"], b["note_mask"], b["target"][:, :-1], return_aux=True)[1] is None
    none = torch.full_like(b["downbeat"], -1)
    assert aux_loss(aux, none, none).item() == 0.0                  # nothing labelled: zero, still a graph
    perfect = torch.zeros(1, 4, 1 + N_PHASE)
    db, ph = torch.tensor([[1, 0, 0, -1]]), torch.tensor([[0, 2, 1, -1]])
    perfect[0, :, 0] = torch.tensor([5.0, -5.0, -5.0, 5.0])
    for k, c in enumerate([0, 2, 1, 0]):
        perfect[0, k, 1 + c] = 5.0
    sc = aux_scores(aux_counts(perfect, db, ph))
    assert sc == {"downbeat_f1": 1.0, "phase_acc": 1.0}


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
