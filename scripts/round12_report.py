"""The twelfth round's report (pace/v2_round12.sh) -> outputs/round12/report.md (+ report.json).

    python3 scripts/round12_report.py

1. The duration chooser (src/dur_chooser.py) on the test windows (true grid): rendered duration
   accuracy and note F1 with the labeller's likeliest length against the chooser's
   (outputs/runs/<labeller>/dur_chooser.json, scripts/train_dur_chooser.py).
2. Whole pieces, MUSTER mean error, A - B per performance with a 95% bootstrap interval over pieces
   (scripts/compare_systems.py paired), on ASAP (real, clean) and Gangqinpu (gangqinpu_real,
   gangqinpu):
     - the chooser on the release (_dc against the release);
     - the new defaults (one meter per piece: meter_v4_gq_om6; swing: _sw; and both with the chooser);
     - seq2seq trained with Gangqinpu against the seq2seq model before, and against v2's release.
3. The verdict per change: better beyond chance on at least one condition, worse beyond chance on none.

Standard library only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_systems import load, paired  # noqa: E402

PIECE_CONDS = (("real", "asap"), ("clean", "asap"), ("gangqinpu_real", "gq"), ("gangqinpu", "gq"))
REL = "v2_nl_gq_meter_v4_gq_d95_tidy"
SEQ_OLD = "sf_base_reg_cab_c128_meter_sb"
SEQ_NEW = "sf_c128_gq30_meter_sb"
PAIRS = (
    ("v2_nl_gq_meter_v4_gq_d95_tidy_dc", REL, "duration chooser"),
    ("v2_nl_gq_meter_v4_gq_om6_d95_tidy_sw", REL, "new defaults: one meter + swing"),
    ("v2_nl_gq_meter_v4_gq_om6_d95_tidy_sw_dc", REL, "new defaults + duration chooser"),
    ("v2_nl_gq_meter_v4_gq_om6_d80_tidy_sw_dc", REL, "the same at d80 (MIDI input's setting; read clean)"),
    (SEQ_NEW, SEQ_OLD, "seq2seq with Gangqinpu, against seq2seq before"),
    (SEQ_NEW, REL, "seq2seq with Gangqinpu, against v2's release"),
)


def verdict(v: dict, n: int) -> str:
    if len(v["missing"]) == n:
        return "not measured"
    if v["better"] and not v["worse"]:
        return "**better**"
    if v["worse"] and not v["better"]:
        return "worse"
    if v["worse"]:
        return "mixed"
    return "level (within chance)"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeller", default="nl_gq")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--asap", default="outputs/system_eval/results.json")
    ap.add_argument("--gq", default="outputs/system_eval_gangqinpu/results.json")
    ap.add_argument("--out", default="outputs/round12")
    args = ap.parse_args(argv)
    lines = ["# Round 12: duration chooser, new defaults, seq2seq with Gangqinpu", ""]
    report = {"chooser_windows": {}, "pairs": {}}

    lines += ["## 1. Duration chooser on the test windows (true grid)", ""]
    f = Path(args.runs_dir) / args.labeller / "dur_chooser.json"
    if f.exists():
        d = json.loads(f.read_text())
        fit = d.get("fit", {})
        lines += [f"Selection set: the true length picked {fit.get('val_acc', float('nan')):.3f} of the time among the "
                  f"top {d.get('k')} (the labeller's likeliest: {fit.get('val_acc_top1', float('nan')):.3f}).", "",
                  "| test set | lengths changed | duration acc: likeliest / chooser | note F1: likeliest / chooser | "
                  "onset F1: likeliest / chooser |", "|---|---|---|---|---|"]
        for cond, r in d.get("test", {}).items():
            a, c = r["argmax"], r["chooser"]
            lines.append(f"| {cond} | {r['changed']:.1%} | {a['duration_acc']:.3f} / {c['duration_acc']:.3f} | "
                         f"{a['note_f1']:.3f} / {c['note_f1']:.3f} | {a['onset_f1']:.3f} / {c['onset_f1']:.3f} |")
        report["chooser_windows"] = d.get("test", {})
    else:
        lines.append(f"not trained ({f} missing)")
    lines.append("")

    res = {k: (load(p) if Path(p).exists() else {}) for k, p in (("asap", args.asap), ("gq", args.gq))}
    lines += ["## 2. Whole pieces", "",
              "MUSTER mean error (lower is better). A - B per performance (negative: A better), 95% bootstrap "
              "interval over pieces; **bold**: beyond chance.", "",
              "| change | A | B | " + " | ".join(c for c, _ in PIECE_CONDS) + " | verdict |",
              "|---|---|---|" + "---|" * len(PIECE_CONDS) + "---|"]
    for a, b, label in PAIRS:
        cells, v = [], {"better": [], "worse": [], "missing": []}
        for cond, key in PIECE_CONDS:
            ra, rb = res[key].get((a, cond)), res[key].get((b, cond))
            r = paired(ra, rb) if ra and rb else None
            if not r:
                cells.append("-")
                v["missing"].append(cond)
                continue
            sure = r["hi"] < 0 or r["lo"] > 0
            dd = f"{r['mean']:+.2f}"
            mean_a = sum(ra.values()) / len(ra)
            cells.append(f"{mean_a:.2f}: {'**' + dd + '**' if sure else dd} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
            if r["hi"] < 0:
                v["better"].append(cond)
            elif r["lo"] > 0:
                v["worse"].append(cond)
        word = verdict(v, len(PIECE_CONDS))
        lines.append(f"| {label} | `{a}` | `{b}` | " + " | ".join(cells) + f" | {word} |")
        report["pairs"][label] = {"a": a, "b": b, **v, "verdict": word}
    lines += ["", "The d80 row is MIDI input's setting: only its clean column is the comparison that matters "
              "(the release keeps 0.95 for recordings).", ""]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print("\n".join(lines))
    print(f"wrote {out / 'report.md'} and report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
