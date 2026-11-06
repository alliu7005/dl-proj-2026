"""Data-quality audit of the processed dataset. Prints a report; re-run after pipeline changes.

    python scripts/audit_data.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.detokenize import parse_tokens  # noqa: E402
from src.preprocessing.store import SegmentStore  # noqa: E402
from src.preprocessing.vocab import UNK, Vocab  # noqa: E402

D = Path("data/processed")


def section(t):
    print(f"\n=== {t} " + "=" * (70 - len(t)))


def main():
    vocab = Vocab.load(D / "vocab.json")
    st = {s: SegmentStore(D / s) for s in ("train", "val", "test")}
    tr = st["train"]
    # target tokens only (each segment's prompt, tokens[tok_lo:tgt_lo], is context, not target)
    tgt_mask = np.zeros(len(tr.tokens), dtype=bool)
    for a, b in tr.index[:, [6, 5]]:
        tgt_mask[a:b] = True
    toks_tr = np.asarray(tr.tokens)[tgt_mask]
    tr_tgt = np.cumsum(tr.index[:, 5] - tr.index[:, 6])
    tgt_starts = np.concatenate([[0], tr_tgt[:-1]])  # segment starts within toks_tr
    itos = np.array(vocab.itos)

    section("1. Target token composition (train)")
    kinds = Counter()
    for t, c in zip(*np.unique(toks_tr, return_counts=True)):
        name = itos[t]
        kinds[name.split("_")[0] if "_" in name else name] += int(c)
    total = sum(kinds.values())
    for k, c in kinds.most_common():
        print(f"  {k:8s} {c / total:6.1%}")
    seg_len = tr.index[:, 5] - tr.index[:, 6]
    n_notes = tr.index[:, 1] - tr.index[:, 0]
    print(f"  tokens/segment: median {np.median(seg_len):.0f}, p99 {np.percentile(seg_len, 99):.0f}; "
          f"tokens per input note: {seg_len.sum() / n_notes.sum():.2f}")

    section("2. Pedal targets")
    ped_on, ped_off = vocab.stoi["ped_on"], vocab.stoi["ped_off"]
    per_seg_ped = np.add.reduceat(np.isin(toks_tr, [ped_on, ped_off]).astype(int), tgt_starts)
    bars = np.array([m["n_bars"] for m in tr.meta])  # bars touched (first/last may be partial)
    print(f"  pedal tokens per bar touched: mean {per_seg_ped.sum() / bars.sum():.2f}; segments with any pedal: "
          f"{(per_seg_ped > 0).mean():.1%}")
    # zero-length or very short presses (pedal flutter) after quantization
    short, presses = 0, 0
    for i in np.random.default_rng(0).choice(len(tr), 3000, replace=False):
        bars_ = parse_tokens(vocab.decode(tr.get(i)[2].tolist()))
        for b in bars_:
            ev = sorted(b.pedal)
            for (p0, t0), (p1, t1) in zip(ev, ev[1:]):
                if t0 == "ped_on" and t1 == "ped_off":
                    presses += 1
                    short += (p1 - p0) <= Fraction(1, 4)
    print(f"  pedal presses lasting <= 1/4 quarter (flutter, sampled): {short / max(presses, 1):.1%}")

    section("3. Target notes without an input onset")
    tie_id = vocab.stoi["tie"]
    n_tie = int((toks_tr == tie_id).sum())
    p_ids = [i for i, t in enumerate(vocab.itos) if t.startswith("p_")]
    n_p = int(np.isin(toks_tr, p_ids).sum())
    print(f"  tie continuations: {n_tie / n_p:.1%} of target notes (no new onset in the performance)")
    grace_id = vocab.stoi["grace"]
    print(f"  grace notes: {(toks_tr == grace_id).sum() / n_p:.2%} of target notes")
    orn = [vocab.stoi[t] for t in vocab.itos if t.startswith("orn_")]
    print(f"  ornament marks (trill/tremolo/...): {np.isin(toks_tr, orn).sum() / n_p:.2%} of target notes "
          "(performed as many notes, written as one)")

    section("4. Diversity: how many distinct targets does the model see?")
    by_comp = defaultdict(list)
    for i, m in enumerate(tr.meta):
        by_comp[m["composition"]].append(i)
    distinct = set()
    for i in range(len(tr)):
        distinct.add(hash(tr.get(i)[2].tobytes()))
    print(f"  train segments {len(tr)}, distinct target sequences {len(distinct)} "
          f"({len(distinct) / len(tr):.1%}); compositions {len(by_comp)}")
    segs = np.array(sorted((len(v) for v in by_comp.values()), reverse=True))
    print(f"  segments per composition: top-10 hold {segs[:10].sum() / segs.sum():.1%} of all; "
          f"median {np.median(segs):.0f}, max {segs.max()}")
    perfs = Counter(m["composition"] for m in tr.meta)
    perf_per_comp = Counter()
    for m in tr.meta:
        perf_per_comp[(m["composition"], m["performance"])] += 1
    ppc = Counter(c for c, _ in perf_per_comp)
    v = np.array(list(ppc.values()))
    print(f"  performances per composition: median {np.median(v):.0f}, max {v.max()} "
          f"({max(ppc, key=ppc.get)})")
    del perfs

    section("5. Composer balance (segments)")
    comp = {s: Counter(m["piece"].split("/")[0] for m in st[s].meta) for s in st}
    names = sorted(set().union(*comp.values()), key=lambda n: -comp["train"][n])
    print(f"  {'composer':14s} {'train':>7s} {'val':>7s} {'test':>7s}")
    for n in names:
        print(f"  {n:14s} " + " ".join(f"{comp[s][n] / max(sum(comp[s].values()), 1):7.1%}" for s in st))

    section("6. Pitch / key coverage")
    pitches = np.asarray(tr.notes[:, 0]).astype(int)
    print(f"  input pitch range {pitches.min()}-{pitches.max()}; notes outside 36-96: "
          f"{((pitches < 36) | (pitches > 96)).mean():.1%}")
    key_counts = Counter(itos[t] for t in toks_tr if itos[t].startswith("key_"))
    print("  key signatures (train, segment starts): " +
          ", ".join(f"{k[4:]}:{c}" for k, c in sorted(key_counts.items(), key=lambda kv: int(kv[0][4:]))))

    section("7. Input timing")
    win = np.array([m["window_sec"] for m in tr.meta])
    print(f"  segment duration (s): median {np.median(win):.1f}, p95 {np.percentile(win, 95):.1f}, "
          f"max {win.max():.1f}")
    print(f"  input notes/segment: median {np.median(n_notes):.0f}, p99 {np.percentile(n_notes, 99):.0f}")
    dur = np.asarray(tr.notes[:, 2])
    print(f"  input durations: median {np.median(dur):.2f}s; > 4 s: {(dur > 4).mean():.1%}")

    section("8. Overlapping same-pitch notes in performance input (sampled)")
    ov = tot = 0
    for i in np.random.default_rng(1).choice(len(tr), 3000, replace=False):
        n = tr.get(i)[0]
        for p in np.unique(n[:, 0]):
            x = n[n[:, 0] == p]
            x = x[np.argsort(x[:, 1])]
            ov += int(((x[:-1, 1] + x[:-1, 2]) > x[1:, 1] + 1e-3).sum())
            tot += len(x)
    print(f"  notes overlapping the next same-pitch note: {ov / tot:.2%}")

    section("9. Vocabulary coverage")
    for s in ("val", "test"):
        t = np.asarray(st[s].tokens)  # prompts + targets
        print(f"  {s}: <unk> rate {(t == vocab.stoi[UNK]).mean():.4%}")
    d_ids = sorted((t for t in vocab.itos if t.startswith("d_")), key=lambda t: Fraction(t[2:]))
    print(f"  {len(d_ids)} duration tokens, from {d_ids[0]} to {d_ids[-1]}")

    section("10. Build report: what was dropped")
    rep = json.load(open(D / "report.json"))
    print("  skipped at load:", rep["skipped_at_load"])
    print("  dropped segments:", rep["dropped_segments"])


if __name__ == "__main__":
    main()
