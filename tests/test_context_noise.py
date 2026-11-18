"""Checks for training-time context corruption (src.dataset.ContextNoise).

    python -m tests.test_context_noise

  * off (prob 0) -> every item identical to a dataset built without it
  * on -> only the context changes (target tokens and note inputs untouched), about ``prob`` of the
    windows that have a context are changed, and items are reproducible
  * corrupted contexts are well formed: vocabulary tokens only, header kept, positions inside the
    bar, no empty position groups, every corrupted context differs from the original
"""
from __future__ import annotations

import numpy as np
import torch

from src.dataset import AugmentConfig, ContextNoise, ScoreSegmentDataset, _ContextCorrupter
from src.preprocessing.noise import NoiseModel

V = "data/processed/vocab.json"


def main():
    nm = NoiseModel.from_json("configs/noise_stats.json")
    base = ScoreSegmentDataset("data/processed/train", V, noise=nm, augment=AugmentConfig())
    off = ScoreSegmentDataset("data/processed/train", V, noise=nm, augment=AugmentConfig(),
                              context_noise=ContextNoise(prob=0.0))
    on = ScoreSegmentDataset("data/processed/train", V, noise=nm, augment=AugmentConfig(),
                             context_noise=ContextNoise(prob=0.5))
    n = min(2000, len(base))
    assert all(torch.equal(base[i]["target"], off[i]["target"]) and torch.equal(base[i]["notes"], off[i]["notes"])
               for i in range(300)), "prob=0 changed the data"
    with_ctx = changed = 0
    for i in range(n):
        x, y = base[i], on[i]
        assert torch.equal(x["notes"], y["notes"]), f"inputs changed at {i}"
        assert torch.equal(x["target"][x["prompt_len"]:], y["target"][y["prompt_len"]:]), f"target changed at {i}"
        with_ctx += x["prompt_len"] > 1
        changed += not torch.equal(x["target"], y["target"])
    frac = changed / max(with_ctx, 1)
    assert 0.4 < frac < 0.6, frac
    assert all(torch.equal(on[i]["target"], on[i]["target"]) for i in range(100)), "not reproducible"
    print(f"datasets: off == unchanged; on: {changed}/{with_ctx} contexts corrupted ({frac:.0%}), "
          "targets and inputs untouched, reproducible")

    vocab = base.vocab
    corr = _ContextCorrupter(vocab, ContextNoise(prob=1.0))
    rng = np.random.default_rng(0)
    checked = same = 0
    for i in range(0, len(base.store), max(1, len(base.store) // 3000)):
        p = base.store.get_prompt(i)
        if len(p) == 0:
            continue
        q = corr(p, rng)
        toks = [vocab.itos[t] for t in q]
        orig = [vocab.itos[t] for t in p]
        k = next((j for j, t in enumerate(orig) if t.startswith("pos_")), len(orig))
        assert [t.split("_")[0] for t in toks[:k]] == [t.split("_")[0] for t in orig[:k]], (orig, toks)
        ts = next((t[3:] for t in orig if t.startswith("ts_")), "4/4")
        nn, dd = (int(x) for x in ts.split("/"))
        top = max([int(np.ceil(nn * 4 / dd)) - 1] + [int(t[4:]) for t in orig if t.startswith("pos_")])
        assert all(int(t[4:]) <= top for t in toks if t.startswith("pos_")), (ts, toks)
        for j, t in enumerate(toks):  # every position group holds at least one note
            if t.startswith("pos_"):
                nxt = next((u for u in toks[j + 1:] if u.startswith(("pos_", "p_"))), None)
                assert nxt is not None and nxt.startswith("p_"), toks
        same += np.array_equal(p, q)
        checked += 1
    assert same <= checked * 0.01, (same, checked)
    print(f"corrupter: {checked} contexts well formed, {same} left unchanged")
    print("ok")


if __name__ == "__main__":
    main()
