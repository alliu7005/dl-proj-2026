"""The thirteenth round's report (pace/v2_round13.sh) -> outputs/round13/report.md (+ report.json).

    python3 scripts/round13_report.py

1. How much the grid is worth, on whole pieces (MUSTER mean error, A - B per performance against the
   release, 95% bootstrap interval over pieces): v2 with a perfect meter model on its tracked beats
   (_og), with the annotated beats too (_ogt), on the seq2seq system's grid (_s2g), and with the
   labeller's bar lines fed back to the decoder (_rf); and the seq2seq system itself, for scale.
2. Pretraining: the meter model fine-tuned for 12k steps from the masked-beat pretraining on
   Aria-MIDI (meter_v4_gq_pt_p12k) against the same recipe from scratch (meter_v4_gq_p12k, the
   pilot's): downbeat F / time-signature accuracy per test set (eval_meter.json).

Standard library only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_systems import load, paired  # noqa: E402

PIECE_CONDS = (("real", "asap"), ("clean", "asap"), ("gangqinpu_real", "gq"), ("gangqinpu", "gq"))
METER_CONDS = ("clean", "real", "periscope", "gangqinpu", "gangqinpu_real")
REL = "v2_nl_gq_meter_v4_gq_d95_tidy"
SYSTEMS = (
    (REL + "_og", "a perfect meter model on the tracked beats (ceiling of meter fixes)"),
    (REL + "_ogt", "the annotated beats and a perfect meter model (ceiling of beat + meter fixes)"),
    (REL + "_s2g", "the seq2seq system's grid"),
    (REL + "_rf", "the labeller's bar lines fed back to the decoder"),
    ("sf_base_reg_cab_c128_meter_sb", "the seq2seq system itself (for scale)"),
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--asap", default="outputs/system_eval/results.json")
    ap.add_argument("--gq", default="outputs/system_eval_gangqinpu/results.json")
    ap.add_argument("--out", default="outputs/round13")
    args = ap.parse_args(argv)
    res = {k: (load(p) if Path(p).exists() else {}) for k, p in (("asap", args.asap), ("gq", args.gq))}
    L = ["# Round 13: what a better grid is worth, and meter pretraining", "",
         "## 1. The grid, on whole pieces", "",
         f"MUSTER mean error (lower is better); each row against the release `{REL}`: A - B per performance "
         "(negative: A better), 95% bootstrap interval over pieces, **bold** beyond chance. The oracle rows "
         "are ceilings (they read the answer), not systems.", "",
         "| grid | " + " | ".join(c for c, _ in PIECE_CONDS) + " |", "|---|" + "---|" * len(PIECE_CONDS)]
    rep = {"grid": {}, "meter": {}}
    base = []
    for cond, key in PIECE_CONDS:
        r = res[key].get((REL, cond))
        base.append(f"{sum(r.values()) / len(r):.2f} (n {len(r)})" if r else "-")
    L.append("| the release (meter model) | " + " | ".join(base) + " |")
    for sysname, label in SYSTEMS:
        cells, row = [], {}
        for cond, key in PIECE_CONDS:
            ra, rb = res[key].get((sysname, cond)), res[key].get((REL, cond))
            r = paired(ra, rb) if ra and rb else None
            if not r:
                cells.append("-")
                continue
            sure = r["hi"] < 0 or r["lo"] > 0
            d = f"{r['mean']:+.2f}"
            cells.append(f"{sum(ra.values()) / len(ra):.2f}: {'**' + d + '**' if sure else d} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
            row[cond] = r
        L.append(f"| {label} (`{sysname}`) | " + " | ".join(cells) + " |")
        rep["grid"][sysname] = row
    L += ["", "## 2. Meter pretraining on Aria-MIDI (12k fine-tuning steps each)", ""]
    pt = Path(args.runs_dir) / "meter_pt" / "best.pt"
    log = Path(args.runs_dir) / "meter_pt" / "pretrain_log.json"
    if log.exists():
        lg = json.loads(log.read_text())
        if lg:
            L.append(f"Pretraining: masked-beat reconstruction error {lg[0]['val']:.3f} -> {min(x['val'] for x in lg):.3f} "
                     f"over {lg[-1]['step']} steps (1.0 = predicting the mean).")
            L.append("")
    elif not pt.exists():
        L += ["not pretrained (outputs/runs/meter_pt missing)", ""]
    L += ["Downbeat F / time-signature accuracy.", "", "| meter | " + " | ".join(METER_CONDS) + " |",
          "|---|" + "---|" * len(METER_CONDS)]
    for m in ("meter_v4_gq_p12k", "meter_v4_gq_pt_p12k", "meter_v4_gq"):
        f = Path(args.runs_dir) / m / "eval_meter.json"
        if not f.exists():
            L.append(f"| `{m}` | " + " | ".join("-" for _ in METER_CONDS) + " |")
            continue
        d = json.loads(f.read_text())
        cells = [f"{d[c]['grid_downbeat_f']:.3f} / {d[c]['grid_ts_acc']:.3f}" if c in d else "-" for c in METER_CONDS]
        rep["meter"][m] = {c: d[c] for c in METER_CONDS if c in d}
        L.append(f"| `{m}` | " + " | ".join(cells) + " |")
    L += ["", "`meter_v4_gq` (60k steps, the release) for scale. Go for a full pretrained run: the pretrained "
          "12k model above the scratch one by >= 0.01 on (downbeat F + ts) / 2 on average over the real test "
          "sets, and not below it by more than 0.02 on any.", ""]
    a, b = rep["meter"].get("meter_v4_gq_pt_p12k"), rep["meter"].get("meter_v4_gq_p12k")
    if a and b:
        diffs = {c: ((a[c]["grid_downbeat_f"] + a[c]["grid_ts_acc"]) - (b[c]["grid_downbeat_f"] + b[c]["grid_ts_acc"])) / 2
                 for c in ("real", "periscope", "gangqinpu", "gangqinpu_real") if c in a and c in b}
        if diffs:
            mean = sum(diffs.values()) / len(diffs)
            go = mean >= 0.01 and min(diffs.values()) >= -0.02
            L.append("pretrained minus scratch: " + ", ".join(f"{c} {v:+.3f}" for c, v in diffs.items())
                     + f"; mean {mean:+.3f}: {'**go**' if go else 'no go'}")
            rep["pretrain_verdict"] = {"diffs": diffs, "mean": mean, "go": go}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
