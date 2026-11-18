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
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Hashable, Iterable, Sequence, TypeVar

from .preprocessing.segment import prompt_from_output

W = TypeVar("W")  # a window, in whatever form the caller uses (dataset index, dict of arrays, ...)
Decode = Callable[[list, list], list]  # (windows, contexts) -> outputs; a context of [] = none


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


def decode_pieces(pieces: dict[Hashable, list[W]], decode: Decode, *,
                  guard_margin: float | None = None,
                  pitches: Callable[[W], Iterable] | None = None,
                  contiguous: Callable[[W, W], bool] | None = None,
                  alone: dict | None = None,
                  progress: Callable[[int, int], None] | None = None):
    """Decode every piece window by window; window ``j`` of all pieces is decoded in one batch.

    pieces        piece id -> its windows, in order
    decode        (windows, contexts) -> one token list per window; a context is a token list,
                  ``[]`` meaning no context
    guard_margin  None: chained. A number: guarded (needs ``pitches``)
    pitches       window -> its input pitches (for input_agreement)
    contiguous    (previous window, window) -> whether the window continues right where the previous
                  one ended; if not (a window was dropped by preprocessing) it starts with no context.
                  Default: always contiguous
    alone         optional {(piece id, j): tokens} of no-context outputs already decoded (e.g.
                  evaluation's "none" mode), reused by guarded mode instead of decoding again
    progress      called with (j + 1, number of steps) after each step

    Returns (outputs {(piece id, j): tokens}, histories {piece id: stitched tokens},
    fallbacks [(piece id, j) of windows where guarded mode kept the no-context output]).
    """
    if guard_margin is not None and pitches is None:
        raise ValueError("guarded decoding needs pitches(window)")
    contiguous = contiguous or (lambda prev, w: True)
    alone = alone or {}
    history = {p: [] for p in pieces}
    outputs, fallbacks = {}, []
    n_steps = max((len(ws) for ws in pieces.values()), default=0)
    for j in range(n_steps):
        step = [(p, ws[j], j and contiguous(ws[j - 1], ws[j])) for p, ws in pieces.items() if j < len(ws)]
        contexts = [prompt_from_output(history[p]) if cont else [] for p, _, cont in step]
        out = decode([w for _, w, _ in step], contexts)
        if guard_margin is not None:
            solo = [alone.get((p, j)) for p, _, _ in step]
            todo = [k for k, s in enumerate(solo) if s is None]
            if todo:
                for k, o in zip(todo, decode([step[k][1] for k in todo], [[] for _ in todo])):
                    solo[k] = o
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
