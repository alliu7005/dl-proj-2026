"""Window-level score metrics on our token format (the same protocol for every method).

Each method produces, for every evaluation window, a token sequence in the target grammar (see
``preprocessing.score``). We compare it with the ground-truth target of that window.

Positions: a window's tokens describe consecutive bars, the first possibly starting mid-bar. Every
note gets an absolute position in quarter notes from the start of the window's first bar
(``bar offset + pos``), using each bar's nominal length from its time signature.

Metrics (micro-averaged over all notes / events of a split):
  note_f1        pitch-only match (multiset): note deletion / insertion
  onset_f1       pitch + exact onset position (quarters)                      -> "rhythm F1"
  onset_f1_shift same, after the best constant shift of the predicted window (forgives a window
                 whose start position is misplaced but whose rhythm is right)
  duration_acc   exact written duration, over onset-matched non-grace notes
  staff_acc      correct staff, over onset-matched notes
  bar_f1         bar-line positions (quarters from the first bar line)
  ts_acc/key_acc first time / key signature of the window
  grace/arp/tie/ornament P,R,F1   flag on onset-matched notes vs. ground truth (TP/FP/FN counted
                 over all predicted / ground-truth flags, so a missed note costs recall)
  pedal P,R,F1   ped_on/ped_off events matched within 1/2 quarter
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction

from ..preprocessing.detokenize import parse_tokens


def _bar_len(ts: str) -> Fraction:
    try:
        n, d = ts.split("/")
        return Fraction(int(n) * 4, int(d))
    except (ValueError, ZeroDivisionError):
        return Fraction(4)


def flatten(tokens: list[str]):
    """-> notes [(onset, pitch, dur, staff, grace, tie, arp, has_orn)], bar lines [positions],
    pedal [(position, kind)], first ts, first key."""
    bars = parse_tokens(tokens)
    notes, barlines, pedal = [], [], []
    offset = Fraction(0)
    starts_with_bar = bool(tokens) and tokens[0] == "bar"
    for j, b in enumerate(bars):
        if j > 0 or starts_with_bar:
            barlines.append(offset)
        for n in b.notes:
            notes.append((offset + n.pos, n.pitch, n.dur, n.staff, n.grace, n.tie, n.arp, bool(n.orns)))
        for pos, kind in b.pedal:
            pedal.append((offset + pos, kind))
        offset += _bar_len(b.ts)
    ts = next((t for t in tokens if t.startswith("ts_")), None)
    key = next((t for t in tokens if t.startswith("key_")), None)
    return notes, barlines, pedal, ts, key


def _match(gt, pr, shift=Fraction(0)):
    """One-to-one match on (pitch, onset + shift). Returns list of (gt_idx, pr_idx)."""
    by = defaultdict(list)
    for j, n in enumerate(pr):
        by[(n[1], n[0] + shift)].append(j)
    out = []
    for i, n in enumerate(gt):
        lst = by.get((n[1], n[0]))
        if lst:
            # several candidates (grace + main note, doubled unisons): take the best-agreeing one
            k = max(range(len(lst)), key=lambda k: (pr[lst[k]][4] == n[4], pr[lst[k]][2] == n[2],
                                                    pr[lst[k]][3] == n[3], pr[lst[k]][5] == n[5], -k))
            out.append((i, lst.pop(k)))
    return out


def _best_shift(gt, pr, max_cands: int = 12):
    """Constant shift (quarters) maximizing exact onset matches; candidates from same-pitch pairs."""
    if not gt or not pr:
        return Fraction(0)
    cand = Counter()
    by_pitch = defaultdict(list)
    for n in pr:
        by_pitch[n[1]].append(n[0])
    for n in gt[:64]:
        for o in by_pitch.get(n[1], [])[:64]:
            cand[n[0] - o] += 1
    best, best_m = Fraction(0), len(_match(gt, pr))
    for s, _ in cand.most_common(max_cands):
        m = len(_match(gt, pr, s))
        if m > best_m:
            best, best_m = s, m
    return best


class Accumulator:
    """Accumulates counts over windows; ``result()`` gives the metric dict."""

    def __init__(self):
        self.c = Counter()

    def add(self, gt_tokens: list[str], pr_tokens: list[str]):
        c = self.c
        g, gb, gp, gts, gkey = flatten(gt_tokens)
        p, pb, pp, pts, pkey = flatten(pr_tokens)
        c["windows"] += 1
        c["gt_notes"] += len(g)
        c["pr_notes"] += len(p)
        c["note_tp"] += sum((Counter(n[1] for n in g) & Counter(n[1] for n in p)).values())
        m = _match(g, p)
        c["onset_tp"] += len(m)
        c["onset_tp_shift"] += len(_match(g, p, _best_shift(g, p)))
        for gi, pi in m:
            if not g[gi][4]:
                c["dur_n"] += 1
                c["dur_ok"] += g[gi][2] == p[pi][2]
            c["staff_n"] += 1
            c["staff_ok"] += g[gi][3] == p[pi][3]
        for k, idx in (("grace", 4), ("tie", 5), ("arp", 6), ("ornament", 7)):
            c[f"{k}_gt"] += sum(n[idx] for n in g)
            c[f"{k}_pr"] += sum(n[idx] for n in p)
            c[f"{k}_tp"] += sum(g[gi][idx] and p[pi][idx] for gi, pi in m)
        c["bar_gt"] += len(gb)
        c["bar_pr"] += len(pb)
        c["bar_tp"] += len(set(gb) & set(pb))
        c["ts_ok"] += gts == pts
        c["key_ok"] += gkey == pkey
        used = set()
        for pos, kind in gp:
            for j, (q, k2) in enumerate(pp):
                if j not in used and k2 == kind and abs(q - pos) <= Fraction(1, 2):
                    used.add(j)
                    c["ped_tp"] += 1
                    break
        c["ped_gt"] += len(gp)
        c["ped_pr"] += len(pp)

    @staticmethod
    def _prf(tp, n_pr, n_gt):
        p = tp / n_pr if n_pr else float("nan")
        r = tp / n_gt if n_gt else float("nan")
        f = 2 * tp / (n_pr + n_gt) if (n_pr + n_gt) else float("nan")
        return p, r, f

    def result(self) -> dict:
        c = self.c
        out = {"windows": c["windows"], "gt_notes": c["gt_notes"]}
        for name, tp in (("note", "note_tp"), ("onset", "onset_tp"), ("onset_shift", "onset_tp_shift")):
            out[f"{name}_p"], out[f"{name}_r"], out[f"{name}_f1"] = self._prf(c[tp], c["pr_notes"], c["gt_notes"])
        out["duration_acc"] = c["dur_ok"] / c["dur_n"] if c["dur_n"] else float("nan")
        out["staff_acc"] = c["staff_ok"] / c["staff_n"] if c["staff_n"] else float("nan")
        out["bar_p"], out["bar_r"], out["bar_f1"] = self._prf(c["bar_tp"], c["bar_pr"], c["bar_gt"])
        out["ts_acc"] = c["ts_ok"] / c["windows"]
        out["key_acc"] = c["key_ok"] / c["windows"]
        for k in ("grace", "tie", "arp", "ornament", "ped"):
            out[f"{k}_p"], out[f"{k}_r"], out[f"{k}_f1"] = self._prf(c[f"{k}_tp"], c[f"{k}_pr"], c[f"{k}_gt"])
        return out
