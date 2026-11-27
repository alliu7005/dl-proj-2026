"""Simpler rhythms (pace/v2_simplicity.sh) -> outputs/simplicity/report.md (+ report.json).

    python scripts/simplicity_report.py

v2 writes the rhythm it hears: a detached off-beat eighth comes out as a 1/6-beat note and rests,
a rushed run as 32nds or a quintuplet, where the arrangement writes plain eighths and sixteenths
(the Lemon demo: 28 of 69 bars with a rare rhythm). Measured here for each setting:
1. How often the scores use a rare rhythm (src/labeller.py rhythm_complexity > 0: 32nds, triplets,
   sextuplets, quintuplets ...), as onsets and as written lengths, against the reference scores
   of the same pieces: how much more complicated v2 writes than the real sheets.
2. MUSTER mean error and its onset / offset parts, score similarity's duration error, and MUSTER
   paired against the release per performance (95% bootstrap interval over pieces).
Needs music21 (the dl-proj environment).
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REL = "v2_nl_gq_meter_v4_gq_d95_tidy"
VARIANTS = ("", "_sp10", "_sp20", "_sl", "_sp20_sl", "_jf")
SETS = (("asap", "outputs/system_eval", ("real", "clean")),
        ("gq", "outputs/system_eval_gangqinpu", ("gangqinpu_real", "gangqinpu")))


def rare_shares(path: str) -> tuple[float, float, int] | None:
    """(share of notes on a rare position in the beat, share with a rare written length, notes) of a
    MusicXML score; ties joined, grace notes left out."""
    import music21 as m21

    from src.labeller import rhythm_complexity
    try:
        s = m21.converter.parse(path)
    except Exception:  # noqa: BLE001
        return None
    on = ln = n = 0
    for note in s.flatten().notes:
        if note.duration.isGrace or (note.tie is not None and note.tie.type in ("stop", "continue")):
            continue
        off = Fraction(note.offset).limit_denominator(96)
        q = off - (off.numerator // off.denominator)
        total = Fraction(note.duration.quarterLength).limit_denominator(96)
        t = note
        while t.tie is not None and t.tie.type in ("start", "continue"):
            nxt = t.next("Note")
            if nxt is None or getattr(nxt, "pitch", None) != getattr(t, "pitch", None):
                break
            total += Fraction(nxt.duration.quarterLength).limit_denominator(96)
            t = nxt
        n += 1
        on += rhythm_complexity(q) > 0
        ln += rhythm_complexity(total) > 0
    return (on / n, ln / n, n) if n else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-pieces", type=int, default=40, help="per condition, for the rare-rhythm counts")
    ap.add_argument("--out", default="outputs/simplicity")
    args = ap.parse_args(argv)
    from scripts.compare_systems import load, paired
    L = ["# Simpler rhythms", "",
         "`_sp10` / `_sp20`: rarer rhythms cost 1 / 2 x their complexity in the labeller's choice; `_sl`: rare written "
         "lengths made plain; `_jf`: bars stretched by a jump shortened (the demo's setting, and where Lemon's 3/4 "
         "bars came from).", ""]
    rep = {}
    for key, root, conds in SETS:
        root = Path(root)
        res_f = root / "results.json"
        res = json.loads(res_f.read_text())["results"] if res_f.exists() else {}
        muster = load(res_f) if res_f.exists() else {}
        manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else []
        refs = {(e["condition"], e["performance"]): e.get("reference") for e in manifest}
        for cond in conds:
            L += [f"## {cond}", "",
                  "| system | rare onsets | rare lengths | MUSTER | onset / offset error | duration error (sim) | "
                  "MUSTER vs the release |", "|---|---|---|---|---|---|---|"]
            ref_rows = []
            for (c, perf), ref in refs.items():
                if c == cond and ref and len(ref_rows) < args.max_pieces:
                    r = rare_shares(ref)
                    if r:
                        ref_rows.append(r)
            if ref_rows:
                L.append(f"| the reference scores | {sum(r[0] for r in ref_rows) / len(ref_rows):.1%} | "
                         f"{sum(r[1] for r in ref_rows) / len(ref_rows):.1%} | - | - | - | - |")
            for v in VARIANTS:
                name = REL + v
                summ = root / name / "summary.json"
                if not summ.exists():
                    L.append(f"| `{name}` | not run | | | | | |")
                    continue
                pieces = [p for p in json.loads(summ.read_text())["pieces"] if p["condition"] == cond and p.get("musicxml")]
                rows = [r for r in (rare_shares(p["musicxml"]) for p in pieces[: args.max_pieces]) if r]
                r_all = res.get(cond, {}).get(name, {}).get("all") or {}
                mu, sim = r_all.get("muster") or {}, r_all.get("sim") or {}
                pr = paired(muster.get((name, cond), {}), muster.get((REL, cond), {})) if v else None
                diff = "-" if not pr else (("**" if pr["hi"] < 0 or pr["lo"] > 0 else "") + f"{pr['mean']:+.2f}"
                                           + ("**" if pr["hi"] < 0 or pr["lo"] > 0 else "") + f" [{pr['lo']:+.2f}, {pr['hi']:+.2f}]")
                ro = f"{sum(r[0] for r in rows) / len(rows):.1%}" if rows else "-"
                rl = f"{sum(r[1] for r in rows) / len(rows):.1%}" if rows else "-"
                fmt = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
                L.append(f"| `{name}` | {ro} | {rl} | {fmt(mu.get('MeanER'))} | {fmt(mu.get('OnsetER'))} / "
                         f"{fmt(mu.get('OffsetER'))} | {fmt(sim.get('NoteDuration'))} | {diff} |")
                rep[f"{name}|{cond}"] = {"rare_onsets": ro, "rare_lengths": rl, "muster": mu, "sim": sim,
                                         "paired": pr}
            L.append("")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
