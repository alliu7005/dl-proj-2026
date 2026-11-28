"""The fifteenth round's report (pace/v2_round15.sh) -> outputs/round15/report.md (+ report.json).

    python3 scripts/round15_report.py

1. The release candidate: nl_gq + meter_v4_gq_pmhs on PM2S's beats with stretched bars shortened
   (_jf), against the release, against the same without _jf (round 9's), and against seq2seq.
2. Its meter from the Aria-MIDI pretraining (three seeds, ensembles) in the candidate.
3. seq2seq given PM2S's beats (_spm) against Beat This!'s (_sb).
4. v2 on the new seq2seq's grid (_s2g_sf_c128_gq30).
5. The team's published release (Hugging Face, 2026-11-27: nl_gq_mc + meter_v4_gq, _jf_bd) and the
   candidate with model's beat decoding (pace/v2_round15b.sh).
6. How often each writes a rare rhythm (triplets, 32nds, ...; scripts/simplicity_report.py
   rare_shares, with music21; skipped without it), against the reference scores.
7. The best system per condition.
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
TEAM = "v2_nl_gq_mc_hf_meter_v4_gq_d95_tidy_jf_bd"   # the release on Hugging Face (pace/v2_round15b.sh)
SETS = (("asap", "outputs/system_eval", ("real", "clean")),
        ("gq", "outputs/system_eval_gangqinpu", ("gangqinpu_real", "gangqinpu")))
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


def cand_bd(meter: str = "meter_v4_gq_pmhs") -> str:
    return f"v2_nl_gq_{meter}_d95_tidy_jf_bd_pm_rpm_hos"


def rare_rows(systems: list, max_pieces: int) -> list:
    """Rare onsets / rare written lengths per condition: the reference scores and each system."""
    try:
        from scripts.simplicity_report import rare_shares
        import music21  # noqa: F401
    except ImportError:
        return ["(skipped: no music21)"]
    L = ["| condition | the reference scores | " + " | ".join(f"`{s}`" for s in systems) + " |",
         "|---|---|" + "---|" * len(systems)]
    mean = lambda rows: (f"{sum(r[0] for r in rows) / len(rows):.1%} / {sum(r[1] for r in rows) / len(rows):.1%}"  # noqa: E731
                         if rows else "-")
    for _, root, conds in SETS:
        root = Path(root)
        man = root / "manifest.json"
        manifest = json.loads(man.read_text()) if man.exists() else []
        for cond in conds:
            refs = [e.get("reference") for e in manifest if e["condition"] == cond and e.get("reference")][:max_pieces]
            cells = [mean([r for r in map(rare_shares, refs) if r])]
            for sname in systems:
                f = root / sname / "summary.json"
                pieces = ([p["musicxml"] for p in json.loads(f.read_text())["pieces"] if p["condition"] == cond and p.get("musicxml")]
                          if f.exists() else [])
                cells.append(mean([r for r in map(rare_shares, pieces[:max_pieces]) if r]))
            if any(c != "-" for c in cells):
                L.append(f"| {cond} | " + " | ".join(cells) + " |")
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
    ap.add_argument("--rare-pieces", type=int, default=30, help="per condition, for the rare-rhythm counts (0: none)")
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
    L += rows + ["", "## 5. The team's published release, and the candidate with its beat decoding", ""]
    rows, rep["team"] = pair_rows(res, [(C, TEAM), (TEAM, REL), (cand_bd(), C), (cand_bd(), TEAM), (SEQ, TEAM)])
    L += rows + [""]
    if args.rare_pieces:
        L += ["## 6. Rare rhythms (share of notes on a rare position / with a rare written length)", ""]
        L += rare_rows([REL, TEAM, C, cand_bd(), SEQ], args.rare_pieces) + [""]
    L += ["## 7. The best system per condition (of this round's and the two to beat)", ""]
    L += best_rows(res, [REL, SEQ, SEQ_PM, PMHS, S2G, C, cand_bd(), TEAM] + [cand(m) for m in PTR]) + [""]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
