"""Checks for the Scoreformer and the chained (own-prompt) evaluation, on real val windows.

    python -m tests.test_scoreformer

1. decoder is causal: changing token t changes no logits before t
2. padding invariance: an item alone == the same item inside a padded batch
3. KV-cached generation == step-by-step generation (left-padded prefixes of different lengths)
4. bf16 autocast runs (the dtype path used on GPU), checked here with CPU autocast
5. chained prompts: prompt_from_output(stitched ground-truth windows) == the stored training
   prompt, token for token, for every window that directly follows its predecessor
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_pieces import stitch  # noqa: E402
from src.dataset import ScoreSegmentDataset, collate  # noqa: E402
from src.model import token_loss  # noqa: E402
from src.preprocessing.segment import prompt_from_output  # noqa: E402
from src.scoreformer import PRESETS, Scoreformer, ScoreformerConfig  # noqa: E402

D = Path("data/processed")


def model_checks():
    torch.manual_seed(0)
    ds = ScoreSegmentDataset(str(D / "val"), str(D / "vocab.json"))
    st = json.loads((D / "feature_stats.json").read_text())
    cfg = ScoreformerConfig(vocab_size=len(ds.vocab), feat_mean=st["mean"], feat_std=st["std"], dropout=0.0,
                            **PRESETS["small"])
    m = Scoreformer(cfg).eval()
    items = [ds[i] for i in (0, 5, 17, 40)]
    b = collate(items)
    dec = b["target"][:, :-1]
    lg = m(b["notes"], b["note_mask"], dec)
    loss = token_loss(lg, b["target"], b["loss_mask"]).item()
    assert abs(loss - torch.log(torch.tensor(float(len(ds.vocab)))).item()) < 0.5, f"initial loss {loss}"

    t = 50
    d2 = dec.clone()
    d2[:, t] = 7
    lg2 = m(b["notes"], b["note_mask"], d2)
    assert torch.allclose(lg[:, :t], lg2[:, :t], atol=1e-5), "decoder is not causal"

    i = 1
    n, T = len(items[i]["notes"]), len(items[i]["target"]) - 1
    solo = m(b["notes"][i:i + 1, :n], b["note_mask"][i:i + 1, :n], b["target"][i:i + 1, :T])
    assert torch.allclose(solo[0], lg[i, :T], atol=1e-4), "padding changes the output"

    pre = [it["target"][:it["prompt_len"]] for it in items]
    assert len({len(p) for p in pre}) > 1
    g1 = m.generate(b["notes"], b["note_mask"], pre, eos_id=2, max_new=30)
    g2 = m.generate_naive(b["notes"], b["note_mask"], pre, eos_id=2, max_new=30)
    assert g1 == g2, "KV-cached generation differs from step-by-step generation"

    with torch.autocast("cpu", dtype=torch.bfloat16):
        lg_bf = m(b["notes"], b["note_mask"], dec)
        g3 = m.generate(b["notes"], b["note_mask"], pre, eos_id=2, max_new=5)
    assert torch.isfinite(lg_bf.float()).all() and len(g3) == len(pre)
    print("model checks: causal, padding-invariant, KV cache == naive, bf16 ok")


def prompt_checks(split: str):
    ds = ScoreSegmentDataset(str(D / split), str(D / "vocab.json"))
    perfs = defaultdict(list)
    for i, meta in enumerate(ds.store.meta):
        perfs[meta["performance"]].append(i)
    checked = 0
    for idx in perfs.values():
        idx.sort(key=lambda i: ds.store.meta[i]["win"])
        hist = []
        for j, i in enumerate(idx):
            if j and ds.store.meta[idx[j - 1]].get("s_hi") == ds.store.meta[i].get("s_lo"):
                built = prompt_from_output(hist)
                stored = ds.vocab.decode(ds.store.get_prompt(i).tolist())
                assert built == stored, f"{ds.store.meta[i]['performance']} win {ds.store.meta[i]['win']}:\n{built}\n{stored}"
                checked += 1
            hist = stitch(hist, ds.vocab.decode(ds.store.get(i)[2].tolist()))
    print(f"chained prompts ({split}): {checked}/{checked} identical to the training prompts")


if __name__ == "__main__":
    model_checks()
    for s in ("val", "test"):
        if (D / s / "index.npy").exists():
            prompt_checks(s)
