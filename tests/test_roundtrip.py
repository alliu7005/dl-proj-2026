"""Round trip: score tokens -> MusicXML (detokenize) -> parse -> tokens. Run after build_dataset.py.

    python -m tests.test_roundtrip --split val --n 300
"""
from __future__ import annotations

import argparse
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from src.preprocessing.detokenize import parse_tokens, tokens_to_musicxml
from src.preprocessing.score import load_score, tokenize_bars
from src.preprocessing.store import SegmentStore
from src.preprocessing.vocab import Vocab


def canonical(tokens):
    """Drop pedal tokens, the position tokens left without any note after removing them, and staff
    tokens that repeat the current staff (a window starting mid-bar states its first staff)."""
    toks, staff = [], 0
    for t in tokens:
        if t.startswith("ped_"):
            continue
        if t == "bar":
            staff = 0
        elif t.startswith("staff_"):
            if int(t[6:]) == staff:
                continue
            staff = int(t[6:])
        toks.append(t)
    out, pending = [], []
    for t in toks:
        if t.startswith(("pos_", "sub_")):
            if t.startswith("pos_"):
                pending = []
            pending.append(t)
        else:
            if t.startswith(("p_", "staff_")):
                out += pending
                pending = []
            out.append(t)
    return out


def note_set(tokens):
    out = Counter()
    for b_i, b in enumerate(parse_tokens(tokens)):
        for n in b.notes:
            out[(b_i, n.pos, n.staff, n.pitch, n.dur, n.grace, n.tie, n.arp, tuple(sorted(n.orns)))] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    store = SegmentStore(Path(args.data) / args.split)
    vocab = Vocab.load(Path(args.data) / "vocab.json")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(store), size=min(args.n, len(store)), replace=False)
    exact, f1s, fails = 0, [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for i in idx:
            toks = canonical(vocab.decode(store.get(i)[2].tolist()))
            try:
                path = Path(tmp) / f"{i}.musicxml"
                tokens_to_musicxml(toks, path)
                ms = load_score(path)
                back = tokenize_bars(ms, [[k] for k in range(len(ms))])
                if toks and toks[0] != "bar" and back and back[0] == "bar":
                    back = back[1:]  # window started mid-bar: the rendered score has a full first bar
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", i, repr(e)[:200])
                continue
            a, b = note_set(toks), note_set(back)
            inter = sum((a & b).values())
            f1s.append(2 * inter / max(sum(a.values()) + sum(b.values()), 1))
            exact += canonical(back) == toks
            if canonical(back) != toks and args.verbose:
                import difflib
                sm = difflib.SequenceMatcher(None, toks, canonical(back), autojunk=False)
                print(i, [(op, toks[a0:a1], canonical(back)[b0:b1]) for op, a0, a1, b0, b1
                          in sm.get_opcodes() if op != "equal"][:4])
    print(f"segments={len(idx)} exact_token_match={exact / len(idx):.3f} "
          f"note_f1 mean={np.mean(f1s):.4f} min={np.min(f1s):.3f} failures={fails}")
    assert fails == 0 and np.mean(f1s) > 0.98


if __name__ == "__main__":
    main()
