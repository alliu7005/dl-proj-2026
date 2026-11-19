"""Decoding whole pieces window by window: chained and guarded. Shared by the evaluation
(scripts/evaluate_pieces.py) and the demo pipeline (src/pipeline.py), so both run exactly the same
logic; tests/test_decoding.py checks it with a stand-in model.

A piece is transcribed one window (~128 notes) at a time. Before each window the decoder reads a
*context*: the score just before the window (``segment.prompt_from_output`` of everything decoded
so far), which tells it where in the bar the window starts and which time / key signature applies.

  chained  the context is the model's own output so far, as on any new recording. Mistakes carry
           over, and scripts/analyze_chained.py showed they cascade: one degenerate window (no
           notes, or a loop many times longer than its target) wrecks the next window's context.
  guarded  chained, but every window is also decoded with no context, and that output is kept
           when it agrees with the window's INPUT notes (known at inference) better by more than
           ``guard_margin`` pitch F1. The kept output also becomes the history the next context is
           built from, so a degenerate window can no longer take the following ones down with it.
  meter    guarded, then a check that the time signatures agree along the piece: a short run of
           windows whose time signature differs from the windows around it (``smooth_meters``) is
           decoded again with the surrounding time signature forced as its first token(s), and the
           piece is decoded again from there, so the corrected meter also reaches the context of
           the windows after it. A wrong time signature mis-sizes every bar of the window, so it
           shifts every position after it (onset F1, bar F1).
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Hashable, Iterable, Sequence, TypeVar

from .preprocessing.segment import prompt_from_output

W = TypeVar("W")  # a window, in whatever form the caller uses (dataset index, dict of arrays, ...)
# (windows, contexts, starts) -> outputs. A context of [] = none. starts[k] = tokens the output of
# window k must begin with (forced after the context; [] = free); they are part of the returned output.
Decode = Callable[[list, list, list], list]


def pedal_down(tokens: list[str]) -> bool:
    """Is the sustain pedal down at the end of ``tokens``?"""
    for t in reversed(tokens):
        if t in ("ped_on", "ped_off"):
            return t == "ped_on"
    return False


def stitch(history: list[str], window: list[str]) -> list[str]:
    """Append a window's tokens to the piece so far, dropping the window's opening ``ped_on`` if
    the pedal is already down (the only cleanup consecutive windows need)."""
    if pedal_down(history):
        first = next((k for k, t in enumerate(window) if t.startswith("ped_")), None)
        if first is not None and window[first] == "ped_on":
            window = window[:first] + window[first + 1:]
    return history + window


TOKENS_PER_NOTE, EXTRA_TOKENS = 8, 64
# guarded mode keeps the no-context output when its input agreement beats the chained output's by
# more than this. scripts/simulate_guard.py on all six post-milestone models: 0 beats 0.02 / 0.05 /
# 0.1 / 0.2 on note, onset, duration and bar F1 in nearly every model and condition (s2s clean:
# onset F1 0.170 at 0 vs 0.148 at 0.05). Ties keep the chained output.
DEFAULT_GUARD_MARGIN = 0.0


def max_new_tokens(n_notes: int, prefix_len: int, max_tokens: int) -> int:
    """Output budget for one window: ``8 * notes + 64`` tokens, within the model's ``max_tokens``.
    Real targets average 3.4 tokens per input note and none of the train / val / test windows come
    near this (the largest is 80% of it), so it never cuts a legitimate output; it only stops a
    degenerate loop from running to ``max_tokens``. The evaluation and the demo both use it."""
    return max(16, min(max_tokens - prefix_len, TOKENS_PER_NOTE * n_notes + EXTRA_TOKENS))


def input_agreement(pitches: Iterable, tokens: Sequence[str]) -> float:
    """Pitch F1 (multiset) between a window's input notes and the pitches of a decoded output.
    Needs no ground truth, so it is available at inference. Transcription errors are rare, so a
    healthy output scores high; a degenerate one (no notes, or a loop) scores low."""
    a = Counter(int(round(float(p))) for p in pitches)
    b = Counter(int(t[2:]) for t in tokens if t.startswith("p_"))
    tp = sum((a & b).values())
    n = sum(a.values()) + sum(b.values())
    return 2 * tp / n if n else 1.0


def window_meter(tokens: Sequence[str]) -> str | None:
    """The time signature a window's output declares (every window opens with ``[bar] ts_X``)."""
    return next((t for t in tokens if t.startswith("ts_")), None)


def _runs(meters: Sequence[str | None]) -> list[list]:
    """[[time signature, first window, length], ...] for maximal runs of equal values."""
    runs = []
    for j, m in enumerate(meters):
        if runs and runs[-1][0] == m:
            runs[-1][2] += 1
        else:
            runs.append([m, j, 1])
    return runs


def smooth_meters(meters: Sequence[str | None], max_blip: int = 2) -> dict[int, str]:
    """Windows whose time signature is a short blip in an otherwise consistent piece.

    Works on runs of windows with the same time signature (``None`` = the window declared none):
      1. a run of at most ``max_blip`` windows with the same time signature on both sides takes it
         (A A B A A, A B B A A A);
      2. then, at the start or end of the piece, a run of at most ``max_blip`` windows takes its
         neighbour's time signature if that run is at least twice as long (B A A); a window with no
         time signature takes its neighbour's whenever there is one.
    A genuine change of meter (A A A B B B) has no short run between equal ones and is kept, and so
    is a short run at the end that could be a real change (A A A B B). Returns {j: time signature}.
    """
    cur = list(meters)
    while True:                        # 1. interior blips, one at a time (each merges runs: terminates)
        runs = _runs(cur)
        count = Counter(m for m in cur if m is not None)
        cands = [(n, -count[runs[k - 1][0]], j, runs[k - 1][0]) for k, (m, j, n) in enumerate(runs)
                 if 0 < k < len(runs) - 1 and n <= max_blip and runs[k - 1][0] == runs[k + 1][0]
                 and runs[k - 1][0] is not None]
        if not cands:
            break
        n, _, j, m = min(cands)        # shortest blip first, then the more common meter, then leftmost
        cur[j:j + n] = [m] * n
    runs = _runs(cur)
    if len(runs) >= 2:                 # 2. blips at either end
        for k, nb in ((0, 1), (len(runs) - 1, len(runs) - 2)):
            m, j, n = runs[k]
            other = runs[nb]
            if other[0] is not None and (m is None or (n <= max_blip and other[2] >= 2 * n)):
                cur[j:j + n] = [other[0]] * n
    for k, m in enumerate(cur):        # interior windows with no time signature at all
        if m is None:
            left = next((cur[i] for i in range(k - 1, -1, -1) if cur[i] is not None), None)
            right = next((cur[i] for i in range(k + 1, len(cur)) if cur[i] is not None), None)
            if left is not None and left == right:
                cur[k] = left
    return {j: m for j, (m, old) in enumerate(zip(cur, meters)) if m != old and m is not None}


def decode_pieces(pieces: dict[Hashable, list[W]], decode: Decode, *,
                  guard_margin: float | None = None,
                  pitches: Callable[[W], Iterable] | None = None,
                  contiguous: Callable[[W, W], bool] | None = None,
                  alone: dict | None = None,
                  starts: dict | None = None,
                  keep: dict | None = None,
                  progress: Callable[[int, int], None] | None = None):
    """Decode every piece window by window; window ``j`` of all pieces is decoded in one batch.

    pieces        piece id -> its windows, in order
    decode        (windows, contexts, starts) -> one token list per window (see ``Decode``)
    guard_margin  None: chained. A number: guarded (needs ``pitches``)
    pitches       window -> its input pitches (for input_agreement)
    contiguous    (previous window, window) -> whether the window continues right where the previous
                  one ended; if not (a window was dropped by preprocessing) it starts with no context.
                  Default: always contiguous
    alone         optional {(piece id, j): tokens} of no-context outputs already decoded (e.g.
                  evaluation's "none" mode), reused by guarded mode instead of decoding again; the
                  no-context outputs guarded mode decodes are added to it
    starts        optional {(piece id, j): tokens} each such window's output must begin with
                  (forced in both its contextual and its no-context decoding)
    keep          optional {(piece id, j): (tokens, fell back)} outputs to use as they are, not
                  decoded (e.g. a piece's windows before the first one that changed)
    progress      called with (j + 1, number of steps) after each step

    Returns (outputs {(piece id, j): tokens}, histories {piece id: stitched tokens},
    fallbacks [(piece id, j) of windows where guarded mode kept the no-context output]).
    """
    if guard_margin is not None and pitches is None:
        raise ValueError("guarded decoding needs pitches(window)")
    contiguous = contiguous or (lambda prev, w: True)
    alone = {} if alone is None else alone
    starts, keep = starts or {}, keep or {}
    history = {p: [] for p in pieces}
    outputs, fallbacks = {}, []
    n_steps = max((len(ws) for ws in pieces.values()), default=0)
    for j in range(n_steps):
        for p, ws in pieces.items():  # kept outputs: no decoding, same history
            if j < len(ws) and (p, j) in keep:
                o, fell_back = keep[(p, j)]
                outputs[(p, j)] = o
                history[p] = stitch(history[p], o)
                if fell_back:
                    fallbacks.append((p, j))
        step = [(p, ws[j], j and contiguous(ws[j - 1], ws[j])) for p, ws in pieces.items()
                if j < len(ws) and (p, j) not in keep]
        if not step:
            if progress:
                progress(j + 1, n_steps)
            continue
        contexts = [prompt_from_output(history[p]) if cont else [] for p, _, cont in step]
        first = [list(starts.get((p, j), [])) for p, _, _ in step]
        out = decode([w for _, w, _ in step], contexts, first)
        if guard_margin is not None:
            solo = [alone.get((p, j)) for p, _, _ in step]
            todo = [k for k, s in enumerate(solo) if s is None]
            if todo:
                for k, o in zip(todo, decode([step[k][1] for k in todo], [[] for _ in todo],
                                             [first[k] for k in todo])):
                    solo[k] = o
                    alone[(step[k][0], j)] = o
            for k, (p, w, _) in enumerate(step):
                x = list(pitches(w))
                if input_agreement(x, solo[k]) > input_agreement(x, out[k]) + guard_margin:
                    out[k] = solo[k]
                    fallbacks.append((p, j))
        for (p, _, _), o in zip(step, out):
            outputs[(p, j)] = o
            history[p] = stitch(history[p], o)
        if progress:
            progress(j + 1, n_steps)
    return outputs, history, fallbacks


def decode_pieces_meter(pieces: dict[Hashable, list[W]], decode: Decode, *, guard_margin: float,
                        pitches: Callable[[W], Iterable], contiguous: Callable[[W, W], bool] | None = None,
                        alone: dict | None = None, max_blip: int = 2, first_pass: tuple | None = None,
                        progress: Callable[[int, int], None] | None = None):
    """Guarded decoding with consistent time signatures ("meter" mode, module docstring).

    Pass 1 decodes guarded. Every window whose time signature ``smooth_meters`` overrules gets that
    time signature forced (after its ``bar`` token, if it opened with one), and pass 2 decodes each
    affected piece again from its first forced window on, so the windows after a corrected one see
    the corrected context; earlier windows and untouched pieces keep their pass-1 outputs, and the
    no-context outputs of windows that are not forced are reused. ``first_pass`` = the
    (outputs, histories, fallbacks) of a guarded decoding with the same margin, if one was already
    run (pass 1 is exactly that), so it is not decoded twice; pass the ``alone`` dict that decoding
    used (it holds its no-context outputs) so pass 2 can reuse them too.

    Returns (outputs, histories, fallbacks of the final pass, forced {(piece id, j): start tokens}).
    """
    alone = {} if alone is None else alone
    kw = dict(guard_margin=guard_margin, pitches=pitches, contiguous=contiguous)
    if first_pass is None:
        first_pass = decode_pieces(pieces, decode, alone=alone, progress=progress, **kw)
    outs, hist, fbs = first_pass
    forced = {}
    for p, ws in pieces.items():
        meters = [window_meter(outs[(p, j)]) for j in range(len(ws))]
        for j, ts in smooth_meters(meters, max_blip).items():
            forced[(p, j)] = (["bar"] if outs[(p, j)][:1] == ["bar"] else []) + [ts]
    if not forced:
        return outs, hist, fbs, forced
    reuse = {k: v for k, v in alone.items() if k not in forced}
    first_forced = {}
    for p, j in forced:
        first_forced[p] = min(j, first_forced.get(p, j))
    fell = set(fbs)
    keep = {(p, j): (outs[(p, j)], (p, j) in fell) for p, ws in pieces.items() for j in range(len(ws))
            if j < first_forced.get(p, len(ws))}
    outs, hist, fbs = decode_pieces(pieces, decode, alone=reuse, starts=forced, keep=keep, progress=progress, **kw)
    return outs, hist, fbs, forced
