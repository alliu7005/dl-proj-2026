"""The rhythm prior pilot's report (pace/v2_rhythm.sh) -> outputs/rhythm/report.md (+ report.json).

    python scripts/rhythm_report.py

1. The prior (outputs/rhythm_prior/prior.json): how often a written beat is triplet, classical
   against pop, and how strongly triplets come in runs.
2. The tuning (nl_gq_bd0 / nl_gq_rp beat_decode.json): the setting chosen, position accuracy, and
   the share of beats decoded triplet against the labels' share, without and with the prior.
3. Whole pieces (round 15's candidate: meter_v4_gq_pmhs, PM2S's beats, _jf): with the prior's
   decoding against the decoding without it and against no beat decoding; rare rhythms against the
   reference scores (needs music21).
4. The verdict. Go: rare rhythms down on Gangqinpu, MUSTER not worse beyond chance anywhere.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_systems import load  # noqa: E402
from scripts.round14_report import pair_rows  # noqa: E402
from scripts.round15_report import cand, rare_rows  # noqa: E402

BD0 = "v2_nl_gq_bd0_meter_v4_gq_pmhs_d95_tidy_jf_bd_pm_rpm_hos"
RP = "v2_nl_gq_rp_meter_v4_gq_pmhs_d95_tidy_jf_bd_pm_rpm_hos"


def tuning_rows(runs: Path) -> list:
    L = ["| decoding | chosen setting | position acc (argmax) | triplet beats decoded / true, per split |", "|---|---|---|---|"]
    for name in ("nl_gq_bd0", "nl_gq_rp"):
        f = runs / name / "beat_decode.json"
        if not f.exists():
            L.append(f"| `{name}` | not tuned | - | - |")
            continue
        d = json.loads(f.read_text())
        best = {k: v for k, v in d["best"].items() if k != "table"}
        row = next((r for r in d.get("table", []) if isinstance(r["setting"], dict)
                    and {k: v for k, v in r["setting"].items() if k != "table"} == best), None)
        trip = ("; ".join(f"{s}: {row[s]['triplet_beats']:.3f} / {row[s]['triplet_beats_true']:.3f}" for s in d.get("splits", [])
                          if s in row and "triplet_beats" in row[s]) if row else "-")
        L.append(f"| `{name}` | {best} | {d['pos_acc']:.4f} ({d['argmax_pos_acc']:.4f}) | {trip or '-'} |")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--prior", default="outputs/rhythm_prior/prior.json")
    ap.add_argument("--asap", default="outputs/system_eval/results.json")
    ap.add_argument("--gq", default="outputs/system_eval_gangqinpu/results.json")
    ap.add_argument("--rare-pieces", type=int, default=30)
    ap.add_argument("--out", default="outputs/rhythm")
    args = ap.parse_args(argv)
    runs = Path(args.runs_dir)
    res = {k: (load(p) if Path(p).exists() else {}) for k, p in (("asap", args.asap), ("gq", args.gq))}
    C = cand("meter_v4_gq_pmhs")
    L = ["# The rhythm prior: fewer stray triplets", "",
         "Whole pieces: MUSTER mean error (lower is better), A - B per performance (negative: A better), 95% bootstrap "
         "interval over pieces, **bold** beyond chance.", "", "## 1. The prior (written scores only)", ""]
    pf = Path(args.prior)
    if pf.exists():
        pj = json.loads(pf.read_text())
        L += ["| split | beats | straight | triplet | other | P(triplet after triplet) | P(triplet after straight) |",
              "|---|---|---|---|---|---|---|"]
        for s, d in pj.get("by_split", {}).items():
            L.append(f"| {s} | {d['beats']} | {d['straight_share']:.1%} | {d['triplet_share']:.1%} | {d['other_share']:.1%} | "
                     f"{d['triplet_after_triplet']:.2f} | {d['triplet_after_straight']:.3f} |")
    else:
        L.append("not fitted")
    L += ["", "## 2. The tuning (validation windows)", ""] + tuning_rows(runs) + [""]
    L += ["## 3. Whole pieces", ""]
    rows, rep = pair_rows(res, [(RP, BD0), (RP, C), (BD0, C)])
    L += rows + [""]
    if args.rare_pieces:
        L += ["Rare rhythms (share of notes on a rare position / with a rare written length):", ""]
        L += rare_rows([C, BD0, RP], args.rare_pieces) + [""]
    v = rep.get(f"{RP} vs {BD0}", {})
    w = rep.get(f"{RP} vs {C}", {})
    worse = (v.get("worse") or []) + (w.get("worse") or [])
    measured = v.get("verdict") not in (None, "not measured")
    L += ["## 4. Verdict", "",
          ("not measured yet" if not measured else
           f"MUSTER worse beyond chance on: {', '.join(sorted(set(worse))) or 'none'}. "
           "Read the rare-rhythm table for Gangqinpu: **go** if the prior's column is below the others there and "
           "nothing is worse beyond chance; else no go."), ""]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
