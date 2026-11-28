"""The fifteenth round's report (pace/v2_round15.sh) -> outputs/round15/report.md (+ report.json).

    python3 scripts/round15_report.py

1. The release candidate: nl_gq + meter_v4_gq_pmhs on PM2S's beats with stretched bars shortened
   (_jf), against the release, against the same without _jf (round 9's), and against seq2seq.
2. Its meter from the Aria-MIDI pretraining (three seeds, ensembles) in the candidate.
3. seq2seq given PM2S's beats (_spm) against Beat This!'s (_sb).
4. v2 on the new seq2seq's grid (_s2g_sf_c128_gq30).
5. The best system per condition. Standard library only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_systems import load  # noqa: E402
from scripts.round14_report import PIECE_CONDS, pair_rows  # noqa: E402

REL = "v2_nl_gq_meter_v4_gq_d95_tidy"
SEQ = "sf_c128_gq30_meter_sb"
SEQ_PM = "sf_c128_gq30_meter_spm_hos"
PMHS = "v2_nl_gq_meter_v4_gq_pmhs_d95_tidy_pm_rpm_hos"   # round 9's
S2G = "v2_nl_gq_meter_v4_gq_d95_tidy_s2g_sf_c128_gq30"
PTR = ["meter_v4_gq_pmhs_ptr", "meter_v4_gq_pmhs_ptr_s1", "meter_v4_gq_pmhs_ptr_s2", "meter_v4_gq_pmhs_ptr_ens3",
       "meter_v4_gq_pmhs_ens4"]
METER_CONDS = ("clean", "real", "periscope", "gangqinpu", "gangqinpu_real")


def cand(meter: str, d: str = "d95") -> str:
    return f"v2_nl_gq_{meter}_{d}_tidy_jf_pm_rpm_hos"


def meter_table(runs: Path, meters: list) -> list:
    """Downbeat F / ts accuracy on PM2S's beats (eval_meter.json's <condition>_pm2s_hos, else <condition>)."""
    L = ["| meter | " + " | ".join(METER_CONDS) + " |", "|---|" + "---|" * len(METER_CONDS)]
    for m in meters:
        f = runs / m / "eval_meter.json"
        d = json.loads(f.read_text()) if f.exists() else {}
        cells = []
        for c in METER_CONDS:
            r = d.get(f"{c}_pm2s_hos") or d.get(c)
            cells.append(f"{r['grid_downbeat_f']:.3f} / {r['grid_ts_acc']:.3f}" if r else "-")
        L.append(f"| `{m}` | " + " | ".join(cells) + " |")
    return L


def best_rows(res: dict, systems: list) -> list:
    L = ["| condition | best | MUSTER | the release | seq2seq |", "|---|---|---|---|---|"]
    for cond, key in PIECE_CONDS:
        have = {s: res[key][(s, cond)] for s in systems if (s, cond) in res[key]}
        if not have:
            continue
        mean = {s: sum(v.values()) / len(v) for s, v in have.items()}
        b = min(mean, key=mean.get)
        fmt = lambda s: f"{mean[s]:.2f}" if s in mean else "-"  # noqa: E731
        L.append(f"| {cond} | `{b}` | {mean[b]:.2f} | {fmt(REL)} | {fmt(SEQ)} |")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--asap", default="outputs/system_eval/results.json")
    ap.add_argument("--gq", default="outputs/system_eval_gangqinpu/results.json")
    ap.add_argument("--out", default="outputs/round15")
    args = ap.parse_args(argv)
    runs = Path(args.runs_dir)
    res = {k: (load(p) if Path(p).exists() else {}) for k, p in (("asap", args.asap), ("gq", args.gq))}
    C = cand("meter_v4_gq_pmhs")
    rep = {}
    L = ["# Round 15: the release candidate on PM2S's beats, its meter pretrained, seq2seq with PM2S, v2 on seq2seq's grid", "",
         "Whole pieces: MUSTER mean error (lower is better), A - B per performance (negative: A better), 95% bootstrap "
         "interval over pieces, **bold** beyond chance.", "",
         "## 1. The candidate", ""]
    rows, rep["candidate"] = pair_rows(res, [(C, REL), (C, PMHS), (C, SEQ), (PMHS, SEQ)])
    L += rows + ["", f"MIDI input's setting (read clean): `{cand('meter_v4_gq_pmhs', 'd80')}` against the same at d95:", ""]
    rows, rep["d80"] = pair_rows(res, [(cand("meter_v4_gq_pmhs", "d80"), C)])
    L += rows + ["", "## 2. Its meter pretrained (on PM2S's beats: downbeat F / time-signature accuracy)", ""]
    L += meter_table(runs, ["meter_v4_gq_pmhs"] + PTR) + [""]
    rows, rep["meters"] = pair_rows(res, [(cand(m), C) for m in PTR] + [(cand(m), SEQ) for m in PTR[3:]])
    L += rows + ["", "## 3. seq2seq given PM2S's beats", ""]
    rows, rep["seq2seq_pm2s"] = pair_rows(res, [(SEQ_PM, SEQ)])
    L += rows + ["", "## 4. v2 on the new seq2seq's grid", ""]
    rows, rep["s2g"] = pair_rows(res, [(S2G, REL), (S2G, SEQ)])
    L += rows + ["", "## 5. The best system per condition (of this round's and the two to beat)", ""]
    L += best_rows(res, [REL, SEQ, SEQ_PM, PMHS, S2G, C] + [cand(m) for m in PTR]) + [""]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
