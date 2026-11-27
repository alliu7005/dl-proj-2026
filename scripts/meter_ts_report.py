"""How meter models do per true time signature: compound meters (6/8, 12/8, 9/8) against simple
ones, from each run's per-performance scores (outputs/runs/<run>/meter_scores_<condition>.json,
scripts/eval_meter.py) -> outputs/meter_ts.md (and .json).

    python scripts/meter_ts_report.py --runs meter_v4_gq,meter_v5_lv,meter_v5_cw,meter_v5_lvcw

Per run, condition and true time signature: performances, the share decoded with the right time
signature, what the rest were decoded as, the right note-value level, beat and downbeat F; a
"compound" row pools every compound meter. Gangqinpu's real recordings (gangqinpu_real: Beat
This!'s real beats, eval_meter.py --real-tracked) are where meter_v4_gq got 6/8 right on 49% of
the songs (23 of 79 read as 3/4).

Standard library only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

TS = ("4/4", "3/4", "2/4", "6/8", "12/8", "9/8")
CONDITIONS = ("gangqinpu_real", "gangqinpu", "periscope", "real", "clean")


def is_compound(ts: str) -> bool:
    try:
        n, _ = (int(x) for x in ts.split("/"))
    except (ValueError, AttributeError):
        return False
    return n % 3 == 0 and n > 3


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else None


def rows(scores: list) -> list:
    """[(label, stats)] per true time signature in TS present, then "compound"."""
    out = []
    groups = [(ts, [s for s in scores if s.get("true_ts") == ts]) for ts in TS]
    groups.append(("compound", [s for s in scores if is_compound(str(s.get("true_ts")))]))
    for label, ss in groups:
        if not ss:
            continue
        dec = Counter(s.get("decoded_ts") for s in ss)
        right = (sum(is_compound(str(s.get("decoded_ts"))) for s in ss) if label == "compound"
                 else dec.get(label, 0))
        out.append((label, {"n": len(ss), "right_ts": right / len(ss),
                            "decoded": dict(dec.most_common(3)),
                            "level_ok": mean([s.get("grid_level_ok") for s in ss]),
                            "beat_f": mean([s.get("grid_beat_f") for s in ss]),
                            "downbeat_f": mean([s.get("grid_downbeat_f") for s in ss])}))
    return out


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="meter_v4_gq,meter_v5_lv,meter_v5_cw,meter_v5_lvcw")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--out", default="outputs/meter_ts.md")
    args = ap.parse_args(argv)
    result = {}
    lines = ["# Meter models per true time signature", "",
             "From `outputs/runs/<run>/meter_scores_<condition>.json` (scripts/eval_meter.py). \"right ts\": the "
             "share of performances decoded with their time signature (compound: with any compound one).", ""]
    for cond in args.conditions.split(","):
        found = False
        for run in args.runs.split(","):
            f = Path(args.runs_dir) / run / f"meter_scores_{cond}.json"
            if not f.exists():
                continue
            if not found:
                lines += [f"## {cond}", "", "| run | true ts | n | right ts | decoded as | right level | beat F | downbeat F |",
                          "|---|---|---|---|---|---|---|---|"]
                found = True
            rs = rows(json.loads(f.read_text()))
            result.setdefault(cond, {})[run] = dict(rs)
            for label, s in rs:
                lines.append(f"| {run} | {label} | {s['n']} | {fmt(s['right_ts'])} | "
                             + ", ".join(f"{k} {v}" for k, v in s["decoded"].items())
                             + f" | {fmt(s['level_ok'])} | {fmt(s['beat_f'])} | {fmt(s['downbeat_f'])} |")
        if found:
            lines.append("")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    print("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
