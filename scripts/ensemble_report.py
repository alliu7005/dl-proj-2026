"""The eleventh round's report (pace/v2_ensemble.sh): meter ensembles against the release's single
meter model, on the meter scores and on whole pieces -> outputs/round11/report.md (+ report.json).

    python3 scripts/ensemble_report.py
    python3 scripts/ensemble_report.py --base meter_v4_gq --seeds meter_v4_gq meter_v4_gq_s1 meter_v4_gq_s2 \
        --candidates meter_v4_gq_tta meter_v4_gq_ens3 meter_v4_gq_ens3t

1. Meter models: downbeat F / time-signature accuracy / right note-value level per test condition
   (outputs/runs/<meter>/eval_meter.json), the single seeds' mean and range next to each ensemble.
2. Whole pieces: MUSTER mean error of v2 (the labeller --labeller with each meter model, the
   release's settings: tidy, d95) on ASAP (real, clean) and Gangqinpu (gangqinpu_real, gangqinpu),
   paired against the release's meter model per performance, bootstrap over pieces
   (scripts/compare_systems.py paired).
3. The verdict, the rule the project uses for a change (scripts/rule_verdict.py): a candidate beats
   the release when it is better beyond chance on at least one condition and worse beyond chance on
   none. The single seeds against the base say how far chance alone moves a meter model.

Standard library only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_systems import load, paired  # noqa: E402

METER_CONDS = ("clean", "real", "periscope", "gangqinpu", "gangqinpu_real")
PIECE_CONDS = (("real", "asap"), ("clean", "asap"), ("gangqinpu_real", "gq"), ("gangqinpu", "gq"))


def meter_scores(runs: Path, m: str) -> dict:
    f = runs / m / "eval_meter.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    return {c: (d[c]["grid_downbeat_f"], d[c]["grid_ts_acc"], d[c].get("grid_level_ok"), d[c].get("performances"))
            for c in METER_CONDS if c in d}


def fmt(x, nd=3):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def system(labeller: str, meter: str, suffix: str) -> str:
    return f"v2_{labeller}_{meter}{suffix}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="meter_v4_gq", help="the release's meter model")
    ap.add_argument("--seeds", nargs="+", default=["meter_v4_gq", "meter_v4_gq_s1", "meter_v4_gq_s2"])
    ap.add_argument("--candidates", nargs="+",
                    default=["meter_v4_gq_tta", "meter_v4_gq_ens3", "meter_v4_gq_ens3t", "meter_v4_gq_ens4"])
    ap.add_argument("--labeller", default="nl_gq")
    ap.add_argument("--suffix", default="_d95_tidy", help="the release's settings in the system names")
    ap.add_argument("--extra", nargs=2, action="append", metavar=("A", "B"), default=None,
                    help="more whole-piece pairs (system names), e.g. the deletion threshold")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--asap", default="outputs/system_eval/results.json")
    ap.add_argument("--gq", default="outputs/system_eval_gangqinpu/results.json")
    ap.add_argument("--out", default="outputs/round11")
    args = ap.parse_args(argv)
    runs = Path(args.runs_dir)
    meters = list(dict.fromkeys(args.seeds + args.candidates))
    lines = ["# Round 11: meter ensembles", "",
             "The release's meter model is one training run of its recipe; seeds of one recipe differ by more "
             "than most changes measured. Here: more seeds of `{}`'s recipe, and the models averaged before the "
             "decoder (`src/meter.py` MeterEnsemble; `_tta`: one model on 5 views of each performance, transposed "
             "and stretched).".format(args.base), ""]
    report = {"meters": {}, "pieces": {}, "verdict": {}}

    # 1. meter scores
    lines += ["## 1. Meter models", "",
              "Downbeat F / time-signature accuracy / right note-value level (n performances). `gangqinpu_real`: "
              "Beat This!'s real beats of the recordings.", ""]
    head = "| meter | " + " | ".join(METER_CONDS) + " |"
    lines += [head, "|---|" + "---|" * len(METER_CONDS)]
    sc = {m: meter_scores(runs, m) for m in meters}
    for m in meters:
        if not sc[m]:
            lines.append(f"| `{m}` | " + " | ".join("not evaluated" for _ in METER_CONDS) + " |")
            continue
        cells = []
        for c in METER_CONDS:
            v = sc[m].get(c)
            cells.append("-" if v is None else f"{fmt(v[0])} / {fmt(v[1])} / {fmt(v[2])} ({v[3]})")
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
        report["meters"][m] = sc[m]
    seeds = [m for m in args.seeds if sc.get(m)]
    if len(seeds) >= 2:
        cells = []
        for c in METER_CONDS:
            vals = [sc[m][c] for m in seeds if c in sc[m]]
            if not vals:
                cells.append("-")
                continue
            db, ts = [v[0] for v in vals], [v[1] for v in vals]
            cells.append(f"{fmt(sum(db) / len(db))} [{fmt(min(db))}-{fmt(max(db))}] / "
                         f"{fmt(sum(ts) / len(ts))} [{fmt(min(ts))}-{fmt(max(ts))}]")
        lines.append(f"| single seeds: mean [range] (downbeat F / ts) | " + " | ".join(cells) + " |")
    lines.append("")

    # 2. whole pieces, paired against the release's meter model
    res = {}
    for key, path in (("asap", args.asap), ("gq", args.gq)):
        res[key] = load(path) if Path(path).exists() else {}
    base_sys = system(args.labeller, args.base, args.suffix)
    others = [m for m in meters if m != args.base]
    pairs = [(system(args.labeller, m, args.suffix), base_sys, m) for m in others]
    pairs += [(a, b, f"{a} vs {b}") for a, b in (args.extra or [])]
    lines += ["## 2. Whole pieces", "",
              f"MUSTER mean error (lower is better), `{base_sys}` = the release. A - B per performance "
              "(negative: A better), 95% bootstrap interval over pieces; **bold**: the interval excludes 0.", ""]
    lines += ["| A (against B = the release unless named) | " + " | ".join(c for c, _ in PIECE_CONDS) + " |",
              "|---|" + "---|" * len(PIECE_CONDS)]
    means = {}
    for a, b, label in [(base_sys, None, args.base)] + pairs:
        cells, verdict = [], {"better": [], "worse": [], "missing": []}
        for cond, key in PIECE_CONDS:
            ra = res[key].get((a, cond))
            if not ra:
                cells.append("-")
                verdict["missing"].append(cond)
                continue
            mean_a = sum(ra.values()) / len(ra)
            means[(a, cond)] = mean_a
            if b is None:
                cells.append(f"{mean_a:.2f} (n {len(ra)})")
                continue
            rb = res[key].get((b, cond))
            r = paired(ra, rb) if rb else None
            if not r:
                cells.append(f"{mean_a:.2f} (no B)")
                verdict["missing"].append(cond)
                continue
            sure = r["hi"] < 0 or r["lo"] > 0
            d = f"{r['mean']:+.2f}"
            cells.append(f"{mean_a:.2f}: {'**' + d + '**' if sure else d} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
            if r["hi"] < 0:
                verdict["better"].append(cond)
            elif r["lo"] > 0:
                verdict["worse"].append(cond)
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
        if b is not None:
            report["verdict"][label] = verdict
    report["pieces"] = {f"{s}|{c}": v for (s, c), v in means.items()}
    lines.append("")

    # 3. verdict
    lines += ["## 3. Verdict", "",
              "Beats the release: better beyond chance on at least one condition, worse beyond chance on none "
              "(the rule scripts/rule_verdict.py applies to the correction rules). A single seed that 'beats' it "
              "measures chance, not a better recipe.", ""]
    for label, v in report["verdict"].items():
        tested = len(PIECE_CONDS) - len(v["missing"])
        if tested == 0:
            word = "not measured"
        elif v["better"] and not v["worse"]:
            word = "**beats the release**"
        elif v["worse"] and not v["better"]:
            word = "worse than the release"
        elif v["worse"]:
            word = "mixed (better on some conditions, worse on others)"
        else:
            word = "level with the release (within chance everywhere)"
        extra = (f"; better on {', '.join(v['better'])}" if v["better"] else "") + \
                (f"; worse on {', '.join(v['worse'])}" if v["worse"] else "") + \
                (f"; not measured on {', '.join(v['missing'])}" if v["missing"] else "")
        lines.append(f"- `{label}`: {word}{extra}")
    lines.append("")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print("\n".join(lines))
    print(f"wrote {out / 'report.md'} and report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
