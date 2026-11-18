"""Unit tests for src/decoding.py (chained / guarded decoding), with a stand-in model.

    python -m tests.test_decoding        # or: pytest tests/test_decoding.py

No data or checkpoint needed. A "window" here is a small dict; the fake model writes one bar per
window holding the window's input pitches, and can be told to fail (output a degenerate loop or
nothing) whenever it is given a context, which is exactly the cascade guarded decoding must stop.
"""
from __future__ import annotations

import math

from src.decoding import decode_pieces, input_agreement, pedal_down, stitch
from src.preprocessing.segment import prompt_from_output


def window(*pitches, fail=None, cont=True):
    return {"pitches": list(pitches), "fail": fail, "cont": cont}


def good_output(w):
    toks = ["bar", "ts_4/4", "key_0", "pos_0"]
    for p in w["pitches"]:
        toks += [f"p_{p}", "d_1"]
    return toks


class FakeModel:
    """decode(windows, contexts). With a context, windows marked fail='loop'/'empty' degenerate."""

    def __init__(self):
        self.calls = []  # (n windows, n with context)

    def __call__(self, windows, contexts):
        self.calls.append((len(windows), sum(bool(c) for c in contexts)))
        out = []
        for w, ctx in zip(windows, contexts):
            if ctx and w["fail"] == "loop":
                out.append(["pos_0"] * 300)  # a loop with no notes, far longer than the target
            elif ctx and w["fail"] == "empty":
                out.append(["bar", "ts_4/4", "key_0"])  # bar lines, no notes
            else:
                out.append(good_output(w))
        return out


def pitches(w):
    return w["pitches"]


def test_stitch_and_pedal():
    assert not pedal_down([]) and pedal_down(["ped_on"]) and not pedal_down(["ped_on", "ped_off"])
    hist = ["bar", "pos_0", "ped_on", "p_60", "d_1"]
    # pedal already down: the next window's opening ped_on is dropped, everything else kept
    assert stitch(hist, ["bar", "pos_0", "ped_on", "p_62", "d_1", "ped_off"]) == \
        hist + ["bar", "pos_0", "p_62", "d_1", "ped_off"]
    # pedal up: nothing dropped
    assert stitch(["bar", "ped_on", "ped_off"], ["ped_on", "p_1"]) == ["bar", "ped_on", "ped_off", "ped_on", "p_1"]


def test_input_agreement():
    assert input_agreement([60, 64, 67], ["p_60", "d_1", "p_64", "d_1", "p_67", "d_1"]) == 1.0
    assert input_agreement([60, 64, 67], ["pos_0"] * 50) == 0.0          # loop without notes
    assert input_agreement([], []) == 1.0
    assert math.isclose(input_agreement([60, 60, 62], ["p_60", "p_62"]), 0.8)  # multiset: one 60 missing
    assert input_agreement([60.0, 61.9], ["p_60", "p_62"]) == 1.0         # float pitches from features


def test_chained_contexts_and_contiguity():
    seen = []

    def decode(ws, ctxs):
        seen.extend(zip([w["pitches"][0] for w in ws], ctxs))
        return [good_output(w) for w in ws]

    piece = [window(60), window(62), window(64, cont=False), window(65)]
    outs, hist, fb = decode_pieces({"a": piece}, decode, contiguous=lambda prev, w: w["cont"])
    ctx = dict(seen)
    assert ctx[60] == []                                                  # first window: no context
    assert ctx[62] == prompt_from_output(good_output(piece[0]))           # previous bar fragment
    assert ctx[64] == []                                                  # not contiguous: no context
    assert ctx[65] == prompt_from_output(stitch(good_output(piece[0]) + good_output(piece[1]),
                                                good_output(piece[2])))
    assert fb == [] and len(outs) == 4
    assert hist["a"] == sum((good_output(w) for w in piece), [])


def test_batches_across_pieces():
    model = FakeModel()
    pieces = {"a": [window(60), window(62), window(64)], "b": [window(70)], "c": [window(80), window(81)]}
    outs, _, _ = decode_pieces(pieces, model)
    assert [n for n, _ in model.calls] == [3, 2, 1]                        # step j = one batch
    assert set(outs) == {("a", 0), ("a", 1), ("a", 2), ("b", 0), ("c", 0), ("c", 1)}


def test_chained_cascade_and_guard():
    # window 1 fails whenever it has a context; chained then also feeds that garbage forward
    piece = [window(60, 64), window(62, 65, fail="loop"), window(64, 67, fail="empty"), window(65, 69)]
    outs_c, hist_c, fb_c = decode_pieces({"a": piece}, FakeModel())
    assert input_agreement(piece[1]["pitches"], outs_c[("a", 1)]) == 0.0
    assert input_agreement(piece[2]["pitches"], outs_c[("a", 2)]) == 0.0
    assert fb_c == []

    model = FakeModel()
    outs_g, hist_g, fb_g = decode_pieces({"a": piece}, model, guard_margin=0.05, pitches=pitches)
    assert fb_g == [("a", 1), ("a", 2)]                                     # exactly the broken windows
    for j, w in enumerate(piece):
        assert outs_g[("a", j)] == good_output(w)
    assert hist_g["a"] == sum((good_output(w) for w in piece), [])          # history uses kept outputs
    # every window decoded twice (with and without context), except the first (no context anyway)
    assert sum(n for n, _ in model.calls) == 2 * len(piece)


def test_guard_margin_and_ties():
    def decode(ws, ctxs):  # with a context, window 1 drops one of its two notes: agreement 2/3
        return [["p_62", "d_1"] if c and w["pitches"] == [62, 63] else good_output(w) for w, c in zip(ws, ctxs)]

    piece = [window(60), window(62, 63)]
    gap = 1.0 - 2 / 3                                                       # no-context output agrees fully
    _, _, fb = decode_pieces({"a": piece}, decode, guard_margin=gap + 0.01, pitches=pitches)
    assert fb == []                                                         # gap below the margin: keep chained
    _, _, fb = decode_pieces({"a": piece}, decode, guard_margin=gap - 0.01, pitches=pitches)
    assert fb == [("a", 1)]                                                 # gap above the margin: fall back
    # equal agreement never falls back (chained output kept)
    _, _, fb = decode_pieces({"a": [window(60), window(62)]}, FakeModel(), guard_margin=0.0, pitches=pitches)
    assert fb == []
    # an infinite margin is exactly chained decoding
    piece = [window(60), window(62, fail="loop"), window(64)]
    assert decode_pieces({"a": piece}, FakeModel(), guard_margin=math.inf, pitches=pitches)[0] == \
        decode_pieces({"a": piece}, FakeModel())[0]


def test_guard_reuses_alone_outputs():
    piece = [window(60), window(62, fail="loop"), window(64)]
    alone = {("a", j): good_output(w) for j, w in enumerate(piece)}
    model = FakeModel()
    outs, _, fb = decode_pieces({"a": piece}, model, guard_margin=0.05, pitches=pitches, alone=alone)
    assert fb == [("a", 1)] and outs[("a", 1)] == good_output(piece[1])
    assert sum(n for n, _ in model.calls) == len(piece)                     # no extra no-context decodes


def test_guard_requires_pitches():
    try:
        decode_pieces({"a": [window(60)]}, FakeModel(), guard_margin=0.1)
    except ValueError:
        return
    raise AssertionError("guarded decoding without pitches() should raise")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
