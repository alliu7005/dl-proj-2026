"""Round 7's report: every meter model, labeller and whole-piece system of rounds 6-7 side by side.

    python scripts/analyze_round7.py               # -> outputs/round7/report.md (+ report.json)

Reads only what the evaluations wrote (nothing is recomputed; a file that is not there is noted
and its section skipped), so it runs on the login node and on whatever part of a batch finished:
  1. meter models (outputs/runs/<meter>/eval_meter.json): downbeat F, time-signature accuracy, beat
     F, right note-value level, per test condition (the tracker each was evaluated with is in the
     condition's name: _pm2s, _real)
  2. Gangqinpu by genre (meter_scores_gangqinpu*.json per performance, the genre from
     data/processed/meter/test_gangqinpu.pkl): where the Gangqinpu training data helps
  3. time signatures (eval_meter.json ts_confusion): the most common wrong decodings, and the share
     of wrongly decoded bars written as 4/4 (the bias Gangqinpu's 81% 4/4 could bring)
  4. the note labellers on each meter model's grids (outputs/runs/<labeller>/eval_pieces_best.json):
     onset F1 per condition and grid
  5. whole pieces (outputs/system_eval/results.json): MUSTER mean error per system and condition,
     and the paired comparisons of round 7 (scripts/compare_systems.py's bootstrap over pieces)
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_systems import load as load_systems, paired  # noqa: E402

METERS = ("meter_v3_tp", "meter_v4_gq", "meter_v4_gq_posh", "meter_v4_gq_pm", "meter_v4_gq_both",
          "meter_v4_gq_pm_s1", "meter_v4_gq_pm_g30", "meter_v3_tp_rf", "meter_v3_tp_fill")
LABELLERS = ("nl_base_gn", "nl_base_gn_pm")
REL = "v2_nl_base_gn_meter_v3_tp_d95_tidy"          # the release (Beat This!'s beats)
PM = "v2_nl_base_gn_meter_v4_gq_pm_d95_tidy_pm_rpm"  # round 6's best (PM2S's beats)
PAIRS = [  # (A, B, what A changes); negative A - B: A better
    (REL + "_pm_rpm", REL, "PM2S's beats instead of Beat This!'s (the release's models)"),
    (PM, REL + "_pm_rpm", "Gangqinpu-trained meter model for PM2S (meter_v4_gq_pm), same beats"),
    ("v2_nl_base_gn_meter_v4_gq_d95_tidy", REL, "Gangqinpu-trained meter model (meter_v4_gq), Beat This!"),
    (PM, REL, "round 6's best against the release"),
    (PM, "beyer_dai", "round 6's best against Beyer & Dai"),
    ("v2_nl_base_gn_meter_v4_gq_pm_s1_d95_tidy_pm_rpm", PM, "meter_v4_gq_pm's second seed (chance)"),
    ("v2_nl_base_gn_meter_v4_gq_pm_g30_d95_tidy_pm_rpm", PM, "Gangqinpu 0.3 of every batch instead of 0.5"),
    ("v2_nl_base_gn_pm_meter_v4_gq_pm_d95_tidy_pm_rpm", PM, "labeller trained on PM2S's timing (nl_base_gn_pm)"),
    ("v2_nl_base_gn_pm_meter_v3_tp_d95_tidy_pm_rpm", REL + "_pm_rpm", "nl_base_gn_pm on the release's meter model"),
    ("v2_nl_base_gn_meter_v4_gq_posh_d95_tidy", "v2_nl_base_gn_meter_v4_gq_d95_tidy", "position-in-bar head"),
    ("v2_nl_base_gn_meter_v4_gq_both_d95_tidy", "v2_nl_base_gn_meter_v4_gq_d95_tidy", "PM2S's beats as evidence too"),
    ("v2_nl_base_gn_meter_v3_tp_rf_d95_tidy", REL, "beat times from the notes (snap + fill)"),
    ("v2_nl_base_gn_meter_v3_tp_fill_d95_tidy", REL, "beat times from the notes (fill only)"),
    ("beyer_dai", REL, "Beyer & Dai against the release"),
]


def f3(x) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def meters_section(runs: Path, notes: list[str], out: dict) -> list[str]:
    rows, conds = [], []
    for m in METERS:
        f = runs / m / "eval_meter.json"
        if not f.exists():
            notes.append(f"meter `{m}`: no eval_meter.json")
            continue
        d = {c: v for c, v in json.loads(f.read_text()).items() if isinstance(v, dict) and "grid_downbeat_f" in v}
        out.setdefault("meters", {})[m] = {c: {k: v.get(k) for k in ("grid_downbeat_f", "grid_ts_acc", "grid_beat_f",
                                                                     "grid_level_ok", "performances")} for c, v in d.items()}
        conds += [c for c in d if c not in conds]
        rows.append((m, d))
    if not rows:
        return []
    lines = ["## 1. Meter models", "", "Per test condition: downbeat F / time-signature accuracy / beat F / right "
             "note-value level. `_pm2s`: PM2S's beats; `_real`: Beat This!'s real beats of the recordings; "
             "the others: the condition's usual beats (simulated Beat This! on clean MIDI, ...).", ""]
    for c in conds:
        body = [[f"`{m}`", f3(d[c].get("grid_downbeat_f")), f3(d[c].get("grid_ts_acc")), f3(d[c].get("grid_beat_f")),
                 f3(d[c].get("grid_level_ok")), d[c].get("performances", "-")] for m, d in rows if c in d]
        lines += [f"### {c}", ""] + table(["meter", "downbeat F", "time sig.", "beat F", "right level", "n"], body) + [""]
    return lines


def genre_section(runs: Path, data: Path, notes: list[str], out: dict) -> list[str]:
    pkl = data / "meter" / "test_gangqinpu.pkl"
    if not pkl.exists():
        notes.append(f"genres: no {pkl}")
        return []
    from src.meter import perf_key
    genre = {}
    for p in pickle.load(open(pkl, "rb")):
        genre[perf_key(p)] = p.get("genre") or "unknown"
        genre.setdefault(p["performance"], p.get("genre") or "unknown")
    lines = ["## 2. Gangqinpu test set by genre", "", "Mean over performances of each genre: downbeat F / "
             "time-signature accuracy (n = performances).", ""]
    any_rows = False
    for m in METERS:
        for f in sorted((runs / m).glob("meter_scores_gangqinpu*.json")) if (runs / m).exists() else []:
            per = json.loads(f.read_text())
            by = defaultdict(list)
            for r in per:
                g = genre.get(r["performance"]) or genre.get(str(r["performance"]).split("#")[0]) or "unknown"
                by[g].append(r)
            cond = f.stem[len("meter_scores_"):]
            body = []
            for g, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
                db = [r["grid_downbeat_f"] for r in rs if r.get("grid_downbeat_f") is not None]
                ts = [r["grid_ts_acc"] for r in rs if r.get("grid_ts_acc") is not None]
                body.append([g, len(rs), f3(sum(db) / len(db) if db else None), f3(sum(ts) / len(ts) if ts else None)])
                out.setdefault("genres", {}).setdefault(m, {}).setdefault(cond, {})[g] = {
                    "n": len(rs), "downbeat_f": sum(db) / len(db) if db else None, "ts_acc": sum(ts) / len(ts) if ts else None}
            lines += [f"### `{m}`, {cond}", ""] + table(["genre", "n", "downbeat F", "time sig."], body) + [""]
            any_rows = True
    return lines if any_rows else []


def ts_section(runs: Path, out: dict) -> list[str]:
    lines = ["## 3. Time signatures: what the wrong bars are decoded as", "",
             "Per test condition, the 5 most common (true -> decoded) mistakes in ground-truth bars, and the "
             "share of all wrongly decoded bars that were decoded as 4/4.", ""]
    any_rows = False
    for m in METERS:
        f = runs / m / "eval_meter.json"
        if not f.exists():
            continue
        body = []
        for c, v in json.loads(f.read_text()).items():
            conf = v.get("ts_confusion") if isinstance(v, dict) else None
            if not conf:
                continue
            wrong = [(a, b, n) for a, b, n in conf if a != b]
            tot = sum(n for _, _, n in wrong)
            to44 = sum(n for _, b, n in wrong if b == "4/4") / tot if tot else float("nan")
            body.append([c, f3(to44), ", ".join(f"{a}->{b} {n}" for a, b, n in wrong[:5])])
            out.setdefault("ts", {}).setdefault(m, {})[c] = {"wrong_to_44": to44, "top": wrong[:5]}
        if body:
            lines += [f"### `{m}`", ""] + table(["condition", "wrong -> 4/4", "top mistakes"], body) + [""]
            any_rows = True
    return lines if any_rows else []


def labeller_section(runs: Path, notes: list[str], out: dict) -> list[str]:
    lines = ["## 4. Note labellers on each grid", "", "Onset F1 of the labelled windows per condition "
             "(`grid_true`: the score's grid; `grid_meter_<m>`: meter model m's decoded grid).", ""]
    any_rows = False
    for nl in LABELLERS:
        f = runs / nl / "eval_pieces_best.json"
        if not f.exists():
            notes.append(f"labeller `{nl}`: no eval_pieces_best.json")
            continue
        d = json.loads(f.read_text())
        modes = sorted({mode for c in d.values() if isinstance(c, dict) for mode in c
                        if mode.startswith("grid_")})
        conds = [c for c in d if isinstance(d[c], dict) and any(m in d[c] for m in modes)]
        body = []
        for mode in modes:
            row = [f"`{mode}`"]
            for c in conds:
                r = d[c].get(mode) or {}
                row.append(f3(r.get("onset_f1")))
                out.setdefault("labellers", {}).setdefault(nl, {}).setdefault(mode, {})[c] = r.get("onset_f1")
            body.append(row)
        lines += [f"### `{nl}`", ""] + table(["grid"] + conds, body) + [""]
        any_rows = True
    return lines if any_rows else []


def systems_section(results: Path, notes: list[str], out: dict) -> list[str]:
    if not results.exists():
        notes.append(f"whole pieces: no {results}")
        return []
    res = load_systems(results)
    systems = sorted({s for s, _ in res})
    conds = [c for c in ("real", "clean") if any((s, c) in res for s in systems)]
    body = []
    for s in systems:
        row = [f"`{s}`"]
        for c in conds:
            v = list(res.get((s, c), {}).values())
            row.append(f"{sum(v) / len(v):.2f} ({len(v)})" if v else "-")
            out.setdefault("systems", {}).setdefault(s, {})[c] = sum(v) / len(v) if v else None
        body.append(row)
    body.sort(key=lambda r: float(r[1].split()[0]) if r[1] != "-" else 99)
    lines = ["## 5. Whole pieces", "", "MUSTER mean error (lower is better), mean over performances (n). "
             "Sorted by the first condition.", ""] + table(["system"] + conds, body) + [""]
    lines += ["### Paired comparisons", "", "A - B per performance (negative: A better), 95% interval from a "
              "bootstrap over pieces; **bold**: the interval excludes 0.", ""]
    for c in conds:
        rows = []
        for a, b, what in PAIRS:
            if (a, c) in res and (b, c) in res:
                r = paired(res[(a, c)], res[(b, c)])
                if r:
                    sure = r["hi"] < 0 or r["lo"] > 0
                    d = f"{r['mean']:+.2f}"
                    rows.append([what, f"`{a}`", f"`{b}`", f"**{d}**" if sure else d,
                                 f"[{r['lo']:+.2f}, {r['hi']:+.2f}]", f"{r['better']}/{r['n']}", r["pieces"]])
                    out.setdefault("paired", {}).setdefault(c, []).append({"what": what, "a": a, "b": b, **r})
            else:
                missing = [s for s in (a, b) if (s, c) not in res]
                notes.append(f"pair '{what}' ({c}): not scored: {', '.join(missing)}")
        if rows:
            lines += [f"#### {c}", ""] + table(["what", "A", "B", "A - B", "95% interval", "A better on", "pieces"], rows) + [""]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--results", default="outputs/system_eval/results.json")
    ap.add_argument("--out", default="outputs/round7")
    args = ap.parse_args(argv)
    runs, outdir = Path(args.runs_dir), Path(args.out)
    notes, out = [], {}
    lines = ["# Round 7 report", ""]
    for part in (meters_section(runs, notes, out), genre_section(runs, Path(args.data), notes, out),
                 ts_section(runs, out), labeller_section(runs, notes, out),
                 systems_section(Path(args.results), notes, out)):
        lines += part
    if notes:
        lines += ["## Not in this report", ""] + [f"- {n}" for n in notes] + [""]
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "report.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {outdir / 'report.md'} and report.json")


if __name__ == "__main__":
    main()
