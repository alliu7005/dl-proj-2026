"""The pilot's report (pace/v2_pilot.sh): every idea for beating the release, at small scale, each
with a go / no-go for its full-scale run -> outputs/pilot/report.md (+ report.json).

    python3 scripts/pilot_report.py

A. Meter ensembles from models already trained (no training): the release's meter model on 5
   views (meter_v4_gq_tta); round 7's three PM2S meter models averaged (meter_v4_gq_pm_ens3) and
   meter_v4_gq_pm on 5 views (_pm_tta). Meter scores, and whole pieces paired against the members.
   Go: better beyond chance on a condition and worse on none against the best member.
B. Synthetic performances of written scores (src/synthetic.py; PDMX): meter_v4_gq's recipe for 12k
   steps without (meter_v4_gq_p12k) and with them as a quarter of every batch
   (meter_v4_gq_syn_p12k), one seed each: indicative only. Go: the real-data conditions' mean
   (downbeat F + time signature) / 2 up by at least 0.01 and none down by more than 0.02.
C. PM2S retrained with stable optimisation (the ninth round's settings) for 16k steps at two rates:
   does the training loss stay down (the seventh round's rose from step 8k) and validation rise?
   Go: neither run's beat loss above its minimum by more than 10% after step 4k.
D. The deletion threshold at 0.8 on the release (round 7, michael: small but sure).
E. Per-piece choice between seq2seq and v2: its ceiling (the better of the two on every
   performance, known only with the reference) against the better system alone.
I (pace/v2_pilot3.sh). Score-level rules: PKSpell's key and spellings (_ps), engraving (_eg: clefs,
   one-voice stems), swing written straight (_sw), and all three, against the release.
F-H (pace/v2_pilot2.sh). The 2/4-against-4/4 problem: F one meter per piece after decoding (no
   training: meter_v4_gq_om5 / _om6); G the repetition features (meter_v4_gq_rep_p12k); H
   bar-length-tolerant training (meter_v4_gq_tol_p12k), each against meter_v4_gq_p12k.

Standard library only.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_systems import load, paired  # noqa: E402

CONDS = ("clean", "real", "periscope", "gangqinpu", "gangqinpu_real",
         "clean_pm2s", "real_pm2s", "periscope_pm2s", "gangqinpu_pm2s")
REAL_DATA = ("real", "periscope", "gangqinpu", "gangqinpu_real")   # B's go: real performances, not simulated
PIECES = (("real", "asap"), ("clean", "asap"), ("gangqinpu_real", "gq"), ("gangqinpu", "gq"))
SEQ = "sf_base_reg_cab_c128_meter_sb"


def meter(runs, m):
    f = runs / m / "eval_meter.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    return {c: d[c] for c in CONDS if c in d}


def sel(r):
    return (r["grid_downbeat_f"] + r["grid_ts_acc"]) / 2


def meter_table(runs, names):
    cs = [c for c in CONDS if any(c in meter(runs, m) for m in names)]
    if not cs:
        return ["Not evaluated (" + ", ".join(f"`{m}`" for m in names) + ")."]
    lines = ["| meter | " + " | ".join(cs) + " |", "|---|" + "---|" * len(cs)]
    for m in names:
        sc = meter(runs, m)
        cells = [f"{sc[c]['grid_downbeat_f']:.3f} / {sc[c]['grid_ts_acc']:.3f}" if c in sc else "-" for c in cs]
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    return lines


def piece_rows(res, pairs):
    """Rows of paired comparisons (A - B MUSTER mean per performance) and each pair's verdict."""
    lines = ["| A | B | " + " | ".join(c for c, _ in PIECES) + " | verdict |", "|---|---|" + "---|" * (len(PIECES) + 1)]
    verdicts = {}
    for a, b in pairs:
        cells, better, worse, n = [], [], [], 0
        for cond, key in PIECES:
            ra, rb = res[key].get((a, cond)), res[key].get((b, cond))
            r = paired(ra, rb) if ra and rb else None
            if not r:
                cells.append("-")
                continue
            n += 1
            sure = r["hi"] < 0 or r["lo"] > 0
            d = f"{r['mean']:+.2f}"
            cells.append(f"{'**' + d + '**' if sure else d} [{r['lo']:+.2f}, {r['hi']:+.2f}] (n {r['n']})")
            if r["hi"] < 0:
                better.append(cond)
            elif r["lo"] > 0:
                worse.append(cond)
        v = ("not measured" if n == 0 else "**better**" if better and not worse else "worse" if worse and not better
             else "mixed" if worse else "level (within chance)")
        verdicts[(a, b)] = {"verdict": v, "better": better, "worse": worse, "measured": n}
        lines.append(f"| `{a}` | `{b}` | " + " | ".join(cells) + f" | {v} |")
    return lines, verdicts


def pm2s_log(runs, name):
    f = runs / name / "log.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    out = []
    for r in rows:
        try:
            out.append({"step": int(float(r["step"])), "loss_beat": float(r["loss_beat"]),
                        "grad_norm": float(r.get("grad_norm") or "nan"), "select": float(r.get("select") or "nan")})
        except (KeyError, ValueError):
            continue
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--asap", default="outputs/system_eval/results.json")
    ap.add_argument("--gq", default="outputs/system_eval_gangqinpu/results.json")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="outputs/pilot")
    args = ap.parse_args(argv)
    runs = Path(args.runs_dir)
    res = {k: (load(p) if Path(p).exists() else {}) for k, p in (("asap", args.asap), ("gq", args.gq))}
    go = {}
    L = ["# Pilot: every idea at small scale", "",
         "Whole pieces: MUSTER mean error, A - B per performance (negative: A better), 95% bootstrap interval "
         "over pieces, **bold** beyond chance. Meter scores: downbeat F / time-signature accuracy. One run per "
         "setting: a pilot says which ideas earn a full-scale run, not how large their gain is.", ""]

    # A. ensembles
    L += ["## A. Meter ensembles (no training)", ""]
    L += meter_table(runs, ["meter_v4_gq", "meter_v4_gq_tta", "meter_v4_gq_pm", "meter_v4_gq_pm_s1",
                            "meter_v4_gq_pm_g30", "meter_v4_gq_pm_ens3", "meter_v4_gq_pm_tta"]) + [""]
    pm = "_d95_tidy_pm_rpm"
    pairs = [("v2_nl_gq_meter_v4_gq_tta_d95_tidy", "v2_nl_gq_meter_v4_gq_d95_tidy")]
    for nl in ("nl_base_gn", "nl_gq"):
        for member in ("meter_v4_gq_pm", "meter_v4_gq_pm_s1", "meter_v4_gq_pm_g30"):
            pairs.append((f"v2_{nl}_meter_v4_gq_pm_ens3{pm}", f"v2_{nl}_{member}{pm}"))
        pairs.append((f"v2_{nl}_meter_v4_gq_pm_tta{pm}", f"v2_{nl}_meter_v4_gq_pm{pm}"))
    rows, vA = piece_rows(res, pairs)
    L += rows + [""]
    ens_better = [k for k, v in vA.items() if "ens3" in k[0] and v["verdict"] == "**better**"]
    ens_worse = [k for k, v in vA.items() if "ens3" in k[0] and v["verdict"] in ("worse", "mixed")]
    tta = vA.get(pairs[0], {}).get("verdict")
    go["A_ensemble"] = bool(ens_better) and not ens_worse
    go["A_tta"] = tta == "**better**"
    L += [f"- Seeds averaged: {'**go**' if go['A_ensemble'] else 'no go'} (better than {len(ens_better)} member "
          f"comparison(s), worse or mixed in {len(ens_worse)}).",
          f"- Views (TTA) on the release's model: {'**go**' if go['A_tta'] else 'no go'} ({tta or 'not measured'}).", ""]

    # B. synthetic data
    L += ["## B. Synthetic performances of written scores", ""]
    rep = Path(args.data) / "meter" / "synthetic_report.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        top = ", ".join(f"{k} {v}" for k, v in list(r.get("time_signatures_beats", {}).items())[:8])
        L += [f"Built from {r.get('scores_seen')} scores: {r.get('why')}; performances {r.get('performances')}, "
              f"beats {r.get('beats')}; time signatures (beats): {top}.", ""]
    else:
        L += ["No synthetic data was built (data/raw/pdmx missing, or the build failed: slurm-q1synth-*.out).", ""]
    base, syn = meter(runs, "meter_v4_gq_p12k"), meter(runs, "meter_v4_gq_syn_p12k")
    L += meter_table(runs, ["meter_v4_gq_p12k", "meter_v4_gq_syn_p12k", "meter_v4_gq"]) + [""]
    diffs = {c: sel(syn[c]) - sel(base[c]) for c in CONDS if c in base and c in syn}
    if diffs:
        L += ["(downbeat F + ts) / 2, with minus without: " + ", ".join(f"{c} {d:+.3f}" for c, d in diffs.items()), ""]
    real = [diffs[c] for c in REAL_DATA if c in diffs]
    go["B_synthetic"] = bool(real) and sum(real) / len(real) >= 0.01 and min(real) > -0.02
    L += [f"- {'**go**' if go['B_synthetic'] else 'no go'}: real-data mean "
          f"{(sum(real) / len(real)) if real else float('nan'):+.3f}, worst {min(real) if real else float('nan'):+.3f} "
          "(go: mean >= +0.01, none below -0.02).", ""]

    # C. PM2S
    L += ["## C. PM2S retrained with stable optimisation (16k steps)", ""]
    stable = []
    for name in ("pm2s_pilot3e4", "pm2s_pilot1e4"):
        log = pm2s_log(runs, name)
        if not log:
            L.append(f"- `{name}`: no log.csv (not trained: slurm-q1pm2s-*.out)")
            continue
        L += [f"`{name}`:", "", "| step | loss_beat | grad_norm | validation (beat + downbeat F) / 2 |", "|---|---|---|---|"]
        L += [f"| {r['step']} | {r['loss_beat']:.3f} | {r['grad_norm']:.2f} | {r['select']:.3f} |" for r in log]
        after = [r["loss_beat"] for r in log if r["step"] >= 4000]
        lo = min(r["loss_beat"] for r in log)
        ok = bool(after) and max(after[after.index(min(after)):]) <= 1.1 * lo
        stable.append(ok)
        L += ["", f"stable: {'yes' if ok else 'no'} (min loss {lo:.3f}; best validation "
                  f"{max(r['select'] for r in log):.3f})", ""]
    go["C_pm2s"] = bool(stable) and all(stable)
    L += [f"- {'**go**' if go['C_pm2s'] else 'no go'} for the full ninth round (pace/v2_pm2s2.sh).", ""]

    # D. deletion threshold
    L += ["## D. Deleting above p 0.8 instead of 0.95 (the release)", ""]
    rows, vD = piece_rows(res, [("v2_nl_gq_meter_v4_gq_d80_tidy", "v2_nl_gq_meter_v4_gq_d95_tidy")])
    L += rows + [""]
    go["D_d80"] = list(vD.values())[0]["verdict"] == "**better**"

    # F-H. bar length (pace/v2_pilot2.sh)
    def bar_table(names):
        cs = [c for c in CONDS if any(c in meter(runs, m) for m in names)]
        if not cs:
            return ["Not evaluated (" + ", ".join(f"`{m}`" for m in names) + ")."]
        rows = ["| meter | " + " | ".join(cs) + " |", "|---|" + "---|" * len(cs)]
        for m in names:
            sc = meter(runs, m)
            rows.append(f"| `{m}` | " + " | ".join(
                f"{sc[c]['grid_downbeat_f']:.3f} / {sc[c]['grid_ts_acc']:.3f} / {sc[c].get('grid_ts_acc_equiv', float('nan')):.3f}"
                f" / {sc[c].get('grid_ts_changes_per_100_bars', float('nan')):.1f}" if c in sc else "-" for c in cs) + " |")
        return rows
    L += ["## F. One meter per piece (no training)", "",
          "Downbeat F / time signature / time signature with 2/4 = 4/4 = 2/2 / time-signature changes per 100 bars.", ""]
    L += bar_table(["meter_v4_gq", "meter_v4_gq_om5", "meter_v4_gq_om6"]) + [""]
    rows, vF = piece_rows(res, [("v2_nl_gq_meter_v4_gq_om5_d95_tidy", "v2_nl_gq_meter_v4_gq_d95_tidy"),
                                ("v2_nl_gq_meter_v4_gq_om6_d95_tidy", "v2_nl_gq_meter_v4_gq_d95_tidy")])
    L += rows + [""]
    base_m = meter(runs, "meter_v4_gq")
    goF = {}
    for om in ("meter_v4_gq_om5", "meter_v4_gq_om6"):
        sc = meter(runs, om)
        common = [c for c in REAL_DATA + ("clean",) if c in sc and c in base_m]
        if not common:
            continue
        d_ts = [sc[c]["grid_ts_acc"] - base_m[c]["grid_ts_acc"] for c in common]
        d_db = [sc[c]["grid_downbeat_f"] - base_m[c]["grid_downbeat_f"] for c in common]
        whole = vF.get((f"v2_nl_gq_{om}_d95_tidy", "v2_nl_gq_meter_v4_gq_d95_tidy"), {})
        goF[om] = (sum(d_ts) / len(d_ts) > 0 and min(d_db) > -0.01 and whole.get("verdict") not in ("worse", "mixed"))
        L.append(f"- `{om}`: time signature {sum(d_ts) / len(d_ts):+.3f} on average (worst {min(d_ts):+.3f}), downbeat F "
                 f"worst {min(d_db):+.3f}, whole pieces {whole.get('verdict', 'not measured')}: "
                 f"{'**go**' if goF[om] else 'no go'}")
    go["F_one_meter"] = any(goF.values())
    L += ["", "Go: time signature better on average, downbeat F not down by more than 0.01 anywhere, whole pieces not "
          "worse beyond chance. MUSTER barely sees bar lines; this rule is about the written score.", ""]
    for key, run, what in (("G_rep_features", "meter_v4_gq_rep_p12k", "G. Repetition features (12k steps)"),
                           ("H_tolerant", "meter_v4_gq_tol_p12k", "H. Bar-length-tolerant training (12k steps)")):
        L += [f"## {what}", ""] + bar_table(["meter_v4_gq_p12k", run]) + [""]
        b0, b1 = meter(runs, "meter_v4_gq_p12k"), meter(runs, run)
        common = [c for c in REAL_DATA if c in b0 and c in b1]
        if not common:
            L += ["Not trained (slurm-r2train-*.out).", ""]
            go[key] = False
            continue
        if key == "H_tolerant":   # judged on what it is for: the bar length, the other reading allowed
            d = [b1[c]["grid_ts_acc_equiv"] - b0[c]["grid_ts_acc_equiv"] for c in common]
        else:
            d = [sel(b1[c]) - sel(b0[c]) for c in common]
        d_db = [b1[c]["grid_downbeat_f"] - b0[c]["grid_downbeat_f"] for c in common]
        go[key] = sum(d) / len(d) >= 0.01 and min(d_db) > -0.02
        L += [f"- {'time signature (2/4 = 4/4)' if key == 'H_tolerant' else '(downbeat F + ts) / 2'} with minus without: "
              + ", ".join(f"{c} {x:+.3f}" for c, x in zip(common, d)) + f"; downbeat F worst {min(d_db):+.3f}: "
              f"{'**go**' if go[key] else 'no go'} (go: mean >= +0.01, downbeat F not down by more than 0.02)", ""]

    # I. score-level rules (pace/v2_pilot3.sh)
    L += ["## I. Spelling (PKSpell), engraving (clefs, stems), swing", "",
          "Mean over performances: MUSTER mean error, and score similarity's spelling, stem and staff errors (%). "
          "PKSpell was trained on ASAP's scores (our ASAP test pieces included): Gangqinpu is its honest test.", ""]
    base_sys = "v2_nl_gq_meter_v4_gq_d95_tidy"
    variants = [base_sys] + [base_sys + x for x in ("_ps", "_eg", "_ps_eg", "_sw", "_ps_eg_sw")]
    tables = {"asap": json.loads(Path(args.asap).read_text())["results"] if Path(args.asap).exists() else {},
              "gq": json.loads(Path(args.gq).read_text())["results"] if Path(args.gq).exists() else {}}
    L += ["| system | " + " | ".join(f"{c}: MUSTER / spelling / stem / staff" for c, _ in PIECES) + " |",
          "|---|" + "---|" * len(PIECES)]
    metric = {}
    for v in variants:
        cells = []
        for cond, key in PIECES:
            r = tables[key].get(cond, {}).get(v, {}).get("all")
            if not r:
                cells.append("-")
                continue
            sim, mu = r.get("sim", {}), r.get("muster", {})
            metric[(v, cond)] = sim
            cells.append(f"{mu.get('MeanER', float('nan')):.2f} / {sim.get('NoteSpelling', float('nan')):.1f} / "
                         f"{sim.get('StemDirection', float('nan')):.1f} / {sim.get('StaffAssignment', float('nan')):.1f}")
        L.append(f"| `{v}` | " + " | ".join(cells) + " |")
    L.append("")
    rows, vI = piece_rows(res, [(v, base_sys) for v in variants[1:]])
    L += ["MUSTER, paired against the release:", ""] + rows + [""]

    def drop(v, field, conds):
        d = [metric[(v, c)][field] - metric[(base_sys, c)][field] for c in conds
             if (v, c) in metric and (base_sys, c) in metric and field in metric[(v, c)]]
        return sum(d) / len(d) if d else None
    for key, v, field, conds in (("I_pkspell", base_sys + "_ps", "NoteSpelling", ("gangqinpu_real", "gangqinpu")),
                                 ("I_engraving", base_sys + "_eg", "StemDirection", tuple(c for c, _ in PIECES)),
                                 ("I_swing", base_sys + "_sw", None, ())):
        verdict = vI.get((v, base_sys), {}).get("verdict", "not measured")
        if field:
            dd = drop(v, field, conds)
            go[key] = dd is not None and dd < 0 and verdict not in ("worse", "mixed")
            L.append(f"- `{v}`: {field} {'n/a' if dd is None else f'{dd:+.2f}'} points on {', '.join(conds)}; "
                     f"MUSTER {verdict}: {'**go**' if go[key] else 'no go'}")
        else:
            go[key] = verdict not in ("worse", "mixed", "not measured")
            L.append(f"- `{v}`: MUSTER {verdict} (it changes only swung pieces; Lemon: 78% -> 96% of the "
                     f"sheet's onsets in bars 2-7): {'**go**' if go[key] else 'no go'} (go: not worse anywhere)")
    L += ["", "Go: PKSpell lowers the spelling error on Gangqinpu, engraving the stem error, neither makes MUSTER "
          "worse beyond chance; swing must not make anything worse.", ""]

    # E. per-piece choice
    L += ["## E. Per-piece choice between seq2seq and v2 (its ceiling)", ""]
    for v2 in ("v2_nl_gq_meter_v4_gq_d95_tidy",):
        for cond, key in PIECES:
            a, b = res[key].get((SEQ, cond)), res[key].get((v2, cond))
            if not a or not b:
                continue
            com = sorted(set(a) & set(b))
            ma = sum(a[p] for p in com) / len(com)
            mb = sum(b[p] for p in com) / len(com)
            mo = sum(min(a[p], b[p]) for p in com) / len(com)
            L.append(f"- {cond}: seq2seq {ma:.2f}, v2 {mb:.2f}, the better per performance {mo:.2f} "
                     f"(ceiling {min(ma, mb) - mo:.2f} below the better system; n {len(com)})")
    L += ["", "A selector that needs the reference cannot be built; one from the systems' own confidence gets part "
          "of the ceiling at best. No go unless the ceiling is large.", ""]

    L += ["## Go / no go", ""] + [f"- {k}: {'**go**' if v else 'no go'}" for k, v in go.items()] + [""]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps({"go": go, "ensembles": {f"{a}|{b}": v for (a, b), v in vA.items()}},
                                                indent=1))
    print("\n".join(L))
    print(f"wrote {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
