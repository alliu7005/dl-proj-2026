"""Unit tests for src/decoding.py (chained / guarded / meter decoding), with a stand-in model.

    python -m tests.test_decoding        # or: pytest tests/test_decoding.py

No data or checkpoint needed. A "window" here is a small dict; the fake model writes one bar per
window holding the window's input pitches, and can be told to fail (output a degenerate loop or
nothing) whenever it is given a context, which is exactly the cascade guarded decoding must stop.
A window can also be given a wrong time signature (``ts=``), which meter mode must correct; forced
starts are honoured like a real model's prefix would be.
"""
from __future__ import annotations

import math

from src.decoding import (FALLBACK_TS, Scored, decode_pieces, decode_pieces_meter, decode_pieces_tsearch,
                          forced_starts, input_agreement, pedal_down, smooth_meters, stitch, ts_candidates,
                          window_meter)
from src.preprocessing.segment import prompt_from_output


def window(*pitches, fail=None, cont=True, ts="ts_4/4"):
    return {"pitches": list(pitches), "fail": fail, "cont": cont, "ts": ts}


def good_output(w, ts=None):
    toks = ["bar", ts or w["ts"], "key_0", "pos_0"]
    for p in w["pitches"]:
        toks += [f"p_{p}", "d_1"]
    return toks


def free(windows):
    return [[] for _ in windows]


class FakeModel:
    """decode(windows, contexts, starts). With a context, windows marked fail='loop'/'empty'
    degenerate. A forced start [bar] ts_X makes the window use ts_X (as a real decoder would)."""

    def __init__(self):
        self.calls = []  # (n windows, n with context)
        self.seen = []   # (first pitch, context, start)

    def __call__(self, windows, contexts, starts):
        self.calls.append((len(windows), sum(bool(c) for c in contexts)))
        out = []
        for w, ctx, st in zip(windows, contexts, starts):
            self.seen.append((w["pitches"][0], ctx, st))
            if ctx and w["fail"] == "loop":
                o = ["pos_0"] * 300  # a loop with no notes, far longer than the target
            elif ctx and w["fail"] == "empty":
                o = ["bar", "ts_4/4", "key_0"]  # bar lines, no notes
            else:
                o = good_output(w, ts=window_meter(st))
            out.append(list(st) + o[len(st):] if o[:len(st)] != list(st) else o)
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

    def decode(ws, ctxs, starts):
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
    def decode(ws, ctxs, starts):  # with a context, window 1 drops one of its two notes: agreement 2/3
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


def test_window_meter_and_smoothing():
    assert window_meter(["bar", "ts_3/4", "key_0", "pos_0"]) == "ts_3/4"
    assert window_meter(["ts_6/8", "key_1", "pos_2"]) == "ts_6/8"            # a window starting mid-bar
    assert window_meter(["pos_0"] * 5) is None
    A, B = "ts_4/4", "ts_3/4"
    C = "ts_2/4"
    assert smooth_meters([A, A, B, A, A]) == {2: A}                          # isolated outlier: corrected
    assert smooth_meters([A, A, A, B, B, B]) == {}                           # a genuine change: kept
    assert smooth_meters([A, B, B, A, A, A]) == {1: A, 2: A}                 # a 2-window blip: corrected,
    assert smooth_meters([A, A, B, B, B, A, A]) == {}                        # a 3-window section: kept
    assert smooth_meters([A, B]) == {}                                       # too little to judge
    assert smooth_meters([A, B, A]) == {1: A}
    assert smooth_meters([A, None, A, A]) == {1: A}                          # no time signature: given one
    assert smooth_meters([None, A]) == {0: A} and smooth_meters([None]) == {}
    assert smooth_meters([B, A, A]) == {0: A}                                # a blip at the start
    assert smooth_meters([A, A, A, A, B]) == {4: A}                          # ... or the end
    assert smooth_meters([A, A, A, B, B]) == {}                              # a possible real change at the end
    assert smooth_meters([A, B, C, A, B]) == {}                              # no consistent meter at all
    assert smooth_meters([A, B, A, B, A]) == {1: A, 3: A}
    assert smooth_meters([]) == {}


def test_forced_starts():
    model = FakeModel()
    piece = [window(60), window(62), window(64)]
    outs, _, _ = decode_pieces({"a": piece}, model, starts={("a", 1): ["bar", "ts_3/4"]})
    assert outs[("a", 1)][:2] == ["bar", "ts_3/4"] and window_meter(outs[("a", 1)]) == "ts_3/4"
    assert [st for _, _, st in model.seen] == [[], ["bar", "ts_3/4"], []]    # only window 1 is forced
    # guarded: the no-context decoding of a forced window is forced too
    model = FakeModel()
    decode_pieces({"a": piece}, model, guard_margin=0.05, pitches=pitches, starts={("a", 1): ["ts_3/4"]})
    assert all(st == ["ts_3/4"] for p, ctx, st in model.seen if p == 62)


def test_meter_mode_corrects_and_propagates():
    piece = [window(60), window(62), window(64, ts="ts_3/4"), window(65), window(67)]
    model = FakeModel()
    outs, hist, fbs, forced = decode_pieces_meter({"a": piece}, model, guard_margin=0.05, pitches=pitches)
    assert forced == {("a", 2): ["bar", "ts_4/4"]}                          # the outlier window
    assert all(window_meter(outs[("a", j)]) == "ts_4/4" for j in range(len(piece)))
    # the window after the corrected one was decoded (pass 2) with the corrected context
    ctx_after = [ctx for p, ctx, st in model.seen if p == 65 and ctx][-1]
    assert "ts_4/4" in ctx_after and "ts_3/4" not in ctx_after
    assert hist["a"] == sum((good_output(w, ts="ts_4/4") for w in piece), [])
    # consistent meters: nothing forced, and only one pass is decoded
    model = FakeModel()
    steady = [window(60), window(62), window(64)]
    _, _, _, forced = decode_pieces_meter({"a": steady}, model, guard_margin=0.05, pitches=pitches)
    assert forced == {} and sum(n for n, _ in model.calls) == 2 * len(steady)  # guarded pass only
    # a genuine change of meter is left alone
    change = [window(60), window(61), window(62), window(63, ts="ts_3/4"), window(64, ts="ts_3/4"),
              window(65, ts="ts_3/4")]
    assert decode_pieces_meter({"a": change}, FakeModel(), guard_margin=0.05, pitches=pitches)[3] == {}


def test_meter_reuses_first_pass():
    piece = [window(60), window(62), window(64, ts="ts_3/4"), window(65), window(67)]
    alone = {}  # shared, as scripts/evaluate_pieces.py does: the guarded pass fills it
    first = decode_pieces({"a": piece}, FakeModel(), guard_margin=0.05, pitches=pitches, alone=alone)
    assert len(alone) == len(piece)
    model = FakeModel()
    outs, _, _, forced = decode_pieces_meter({"a": piece}, model, guard_margin=0.05, pitches=pitches,
                                             first_pass=first, alone=alone)
    assert forced == {("a", 2): ["bar", "ts_4/4"]}
    # pass 2 only (pass 1 reused), and only from the first forced window (2) on: windows 2-4 decoded
    # with context, plus a no-context decode for the forced window alone (the others' are reused)
    assert sum(n for n, _ in model.calls) == (len(piece) - 2) + 1
    assert {p for p, _, _ in model.seen} == {64, 65, 67}                    # windows 0-1 kept as they were
    for j in (0, 1):
        assert outs[("a", j)] == first[0][("a", j)]
    # two pieces, one of them consistent: it is not decoded again at all
    alone = {}
    pieces = {"a": piece, "b": [window(70), window(72), window(74)]}
    first = decode_pieces(pieces, FakeModel(), guard_margin=0.05, pitches=pitches, alone=alone)
    model = FakeModel()
    outs, _, fbs, forced = decode_pieces_meter(pieces, model, guard_margin=0.05, pitches=pitches,
                                               first_pass=first, alone=alone)
    assert set(forced) == {("a", 2)} and not any(p >= 70 for p, _, _ in model.seen)
    assert all(outs[("b", j)] == first[0][("b", j)] for j in range(3))
    model = FakeModel()                                                     # nothing to fix: no decoding
    steady = [window(60), window(62), window(64)]
    first = decode_pieces({"a": steady}, FakeModel(), guard_margin=0.05, pitches=pitches)
    assert decode_pieces_meter({"a": steady}, model, guard_margin=0.05, pitches=pitches, first_pass=first)[3] == {}
    assert model.calls == []


class ScoringModel(FakeModel):
    """FakeModel that also scores its outputs like a real model: generated tokens cost 1 nat each
    in the window's right meter (``best``, default its ``ts``) and 2 in any other, so the right
    meter is the more probable reading."""

    def __call__(self, windows, contexts, starts):
        out = []
        for w, o, st in zip(windows, super().__call__(windows, contexts, starts), starts):
            n = len(o) - len(st)
            per = 1.0 if window_meter(o) == w.get("best", w["ts"]) else 2.0
            out.append(Scored(o, -per * n, n))
        return out


def piece_of(n, ts, best, **kw):
    return [dict(window(60 + k, 64 + k, ts=ts), best=best, **kw) for k in range(n)]


def guarded_first(pieces, model):
    alone = {}
    first = decode_pieces(pieces, model, guard_margin=0.0, pitches=pitches, alone=alone)
    return first, alone


def test_ts_candidates_and_forced_starts():
    outs = {("a", 0): ["bar", "ts_3/4", "pos_0"], ("a", 1): ["ts_3/4", "pos_1"], ("a", 2): ["bar", "ts_6/8"]}
    alone = {("a", 0): ["bar", "ts_6/8"], ("a", 1): ["ts_2/4"], ("a", 2): ["bar", "ts_3/4"]}
    pieces = {"a": [0, 1, 2]}
    # 3/4 three times, 6/8 twice, 2/4 once: the most frequent first; ties alphabetical
    assert ts_candidates(pieces, (outs, alone), 3) == {"a": ["ts_3/4", "ts_6/8", "ts_2/4"]}
    # fewer distinct ones than asked: topped up with FALLBACK_TS, without repeats
    assert ts_candidates({"b": [0]}, ({("b", 0): ["ts_4/4"]},), 3) == {"b": ["ts_4/4", "ts_3/4", "ts_6/8"]}
    assert ts_candidates({"c": [0]}, ({("c", 0): ["pos_0"]},), 2) == {"c": list(FALLBACK_TS[:2])}
    # forced: [bar] only where the first-pass output opened with a bar line
    st = forced_starts(pieces, outs, lambda p, j: "ts_4/4" if j < 2 else None)
    assert st == {("a", 0): ["bar", "ts_4/4"], ("a", 1): ["ts_4/4"]}


def test_tsearch_picks_the_probable_meter():
    """The windows' greedy guess is 3/4, but the notes read better in 4/4 (the model scores them
    higher): tsearch writes the whole piece in 4/4; a piece whose guess is right keeps it."""
    m = ScoringModel()
    pieces = {"wrong": piece_of(4, "ts_3/4", "ts_4/4"), "right": piece_of(3, "ts_6/8", "ts_6/8")}
    first, alone = guarded_first(pieces, m)
    res, cands = decode_pieces_tsearch(pieces, m, guard_margin=0.0, pitches=pitches, first_pass=first, alone=alone)
    assert cands["wrong"][0] == "ts_3/4" and "ts_4/4" in cands["wrong"], cands  # 4/4 from FALLBACK_TS
    outs, hist, fbs, chosen = res["tsearch"]
    assert chosen == {"wrong": "ts_4/4", "right": "ts_6/8"}, chosen
    assert all(window_meter(outs[("wrong", j)]) == "ts_4/4" for j in range(4))
    assert all(window_meter(outs[("right", j)]) == "ts_6/8" for j in range(3))
    assert hist["wrong"].count("ts_4/4") == 4 and not fbs
    # without beat scores, tsearch_beat is tsearch
    assert res["tsearch_beat"][3] == chosen


def test_tsearch_beat_rule():
    """tsearch_beat follows the beat agreement when every candidate is rated (the model only breaks
    ties), and falls back to the model's choice when the beats cannot rate every candidate."""
    m = ScoringModel()
    pieces = {"p": piece_of(3, "ts_3/4", "ts_3/4", beat="ts_6/8"), "q": piece_of(2, "ts_3/4", "ts_3/4")}
    first, alone = guarded_first(pieces, m)

    def beat_score(w, tokens):
        if "beat" not in w:
            return None  # no beats for this piece
        return 1.0 if window_meter(tokens) == w["beat"] else 0.4
    res, _ = decode_pieces_tsearch(pieces, m, guard_margin=0.0, pitches=pitches, first_pass=first, alone=alone,
                                   beat_score=beat_score)
    assert res["tsearch"][3] == {"p": "ts_3/4", "q": "ts_3/4"}          # the model's preference
    assert res["tsearch_beat"][3] == {"p": "ts_6/8", "q": "ts_3/4"}     # the beats decide p; q has none
    # equal beat agreement for every candidate: the model's probability decides
    res, _ = decode_pieces_tsearch(pieces, m, guard_margin=0.0, pitches=pitches, first_pass=first, alone=alone,
                                   beat_score=lambda w, t: 0.7)
    assert res["tsearch_beat"][3] == res["tsearch"][3]


def test_tsearch_without_scores():
    """A decode returning plain lists gives every candidate the same score: the first one, the
    piece's most frequent time signature, is kept."""
    m = FakeModel()
    pieces = {"a": piece_of(3, "ts_3/4", "ts_4/4")}
    first, alone = guarded_first(pieces, m)
    res, cands = decode_pieces_tsearch(pieces, m, guard_margin=0.0, pitches=pitches, first_pass=first, alone=alone)
    assert res["tsearch"][3] == {"a": cands["a"][0]} == {"a": "ts_3/4"}


def test_tsearch_keeps_guarding():
    """Forced time signatures do not switch the guard off: a window that degenerates with a
    context still falls back to its no-context output in every candidate's decoding."""
    m = ScoringModel()
    ws = piece_of(3, "ts_3/4", "ts_4/4")
    ws[1]["fail"] = "loop"
    pieces = {"a": ws}
    first, alone = guarded_first(pieces, m)
    res, _ = decode_pieces_tsearch(pieces, m, guard_margin=0.0, pitches=pitches, first_pass=first, alone=alone)
    outs, _, fbs, chosen = res["tsearch"]
    assert ("a", 1) in fbs and chosen == {"a": "ts_4/4"} and "p_61" in outs[("a", 1)]


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
