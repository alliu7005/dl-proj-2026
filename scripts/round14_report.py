"""The fourteenth round's report (pace/v2_round14.sh) -> outputs/round14/report.md (+ report.json).

    python3 scripts/round14_report.py

1. The meter model's full recipe from the Aria-MIDI pretraining (meter_v4_gq_ptr, three seeds) and
   their ensembles: meter scores (eval_meter.json) and whole pieces, paired against the release
   (meter_v4_gq) and the pretrained ensemble against round 11's (meter_v4_gq_ens3t).
2. More pretraining data: the pilot's 12k recipe from a pretraining on ~150k files
   (meter_v4_gq_pt150_p12k) against 32k (meter_v4_gq_pt_p12k) and none (meter_v4_gq_p12k).
3. The labeller from the masked-note pretraining (nl_gq_ptl_p20k) against the same 20k-step recipe
   from scratch (nl_gq_p20k, the articulation pilot's): test windows and whole pieces.
The beat errors are their own report (scripts/beat_errors.py: outputs/beat_errors/report.md).
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
SUFFIX = "_d95_tidy"


def meter_table(runs: Path, meters: list) -> list:
    L = ["| meter | " + " | ".join(METER_CONDS) + " |", "|---|" + "---|" * len(METER_CONDS)]
    for m in meters:
        f = runs / m / "eval_meter.json"
        if not f.exists():
            L.append(f"| `{m}` | " + " | ".join("-" for _ in METER_CONDS) + " |")
            continue
        d = json.loads(f.read_text())
        L.append(f"| `{m}` | " + " | ".join(f"{d[c]['grid_downbeat_f']:.3f} / {d[c]['grid_ts_acc']:.3f}" if c in d else "-"
                                         for c in METER_CONDS) + " |")
    return L


def pair_rows(res: dict, pairs: list) -> tuple[list, dict]:
    L = ["| A | B | " + " | ".join(c for c, _ in PIECE_CONDS) + " | verdict |", "|---|---|" + "---|" * len(PIECE_CONDS) + "---|"]
    out = {}
    for a, b in pairs:
        cells, better, worse, n = [], [], [], 0
        for cond, key in PIECE_CONDS:
            ra, rb = res[key].get((a, cond)), res[key].get((b, cond))
            r = paired(ra, rb) if ra and rb else None
            if not r:
                cells.append("-")
                continue
            n += 1
            sure = r["hi"] < 0 or r["lo"] > 0
            d = f"{r['mean']:+.2f}"
            cells.append(f"{sum(ra.values()) / len(ra):.2f}: {'**' + d + '**' if sure else d} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
            (better if r["hi"] < 0 else worse if r["lo"] > 0 else []).append(cond)
        word = ("not measured" if not n else "**better**" if better and not worse else "worse" if worse and not better
                else "mixed" if worse else "level (within chance)")
        L.append(f"| `{a}` | `{b}` | " + " | ".join(cells) + f" | {word} |")
        out[f"{a} vs {b}"] = {"better": better, "worse": worse, "verdict": word}
    return L, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--asap", default="outputs/system_eval/results.json")
    ap.add_argument("--gq", default="outputs/system_eval_gangqinpu/results.json")
    ap.add_argument("--out", default="outputs/round14")
    args = ap.parse_args(argv)
    runs = Path(args.runs_dir)
    res = {k: (load(p) if Path(p).exists() else {}) for k, p in (("asap", args.asap), ("gq", args.gq))}
    sysname = lambda nl, m: f"v2_{nl}_{m}{SUFFIX}"  # noqa: E731
    rep = {}
    L = ["# Round 14: pretrained meters (full recipe, ensembled), more pretraining data, a pretrained labeller", "",
         "Meter scores: downbeat F / time-signature accuracy. Whole pieces: MUSTER mean error, A - B per performance "
         "(negative: A better), 95% bootstrap interval over pieces, **bold** beyond chance.", "",
         "## 1. The meter's full recipe from the pretraining, and its ensembles", ""]
    ptr = ["meter_v4_gq_ptr", "meter_v4_gq_ptr_s1", "meter_v4_gq_ptr_s2", "meter_v4_gq_ptr_ens3", "meter_v4_gq_ptr_ens3t"]
    L += meter_table(runs, ["meter_v4_gq", "meter_v4_gq_ens3t"] + ptr) + [""]
    rows, rep["meters"] = pair_rows(res, [(sysname("nl_gq", m), sysname("nl_gq", "meter_v4_gq")) for m in ptr]
                                    + [(sysname("nl_gq", "meter_v4_gq_ptr_ens3t"), sysname("nl_gq", "meter_v4_gq_ens3t"))])
    L += rows + [""]
    L += ["## 2. More pretraining data (12k fine-tuning steps each)", ""]
    for name, d in (("meter_pt", "32k files (unique)"), ("meter_pt150", "~150k files (deduped)")):
        f = runs / name / "pretrain_log.json"
        if f.exists():
            lg = json.loads(f.read_text())
            if lg:
                L.append(f"- `{name}` ({d}): masked-beat error {lg[0]['val']:.3f} -> {min(x['val'] for x in lg):.3f} over "
                         f"{lg[-1]['step']} steps")
    L += [""] + meter_table(runs, ["meter_v4_gq_p12k", "meter_v4_gq_pt_p12k", "meter_v4_gq_pt150_p12k"]) + [""]
    a, b = (runs / m / "eval_meter.json" for m in ("meter_v4_gq_pt150_p12k", "meter_v4_gq_pt_p12k"))
    if a.exists() and b.exists():
        da, db = json.loads(a.read_text()), json.loads(b.read_text())
        diffs = {c: ((da[c]["grid_downbeat_f"] + da[c]["grid_ts_acc"]) - (db[c]["grid_downbeat_f"] + db[c]["grid_ts_acc"])) / 2
                 for c in ("real", "periscope", "gangqinpu", "gangqinpu_real") if c in da and c in db}
        if diffs:
            m = sum(diffs.values()) / len(diffs)
            go = m >= 0.01 and min(diffs.values()) >= -0.02
            L.append("150k minus 32k, (downbeat F + ts) / 2: " + ", ".join(f"{c} {v:+.3f}" for c, v in diffs.items())
                     + f"; mean {m:+.3f}: {'**go** (pretrain on more)' if go else 'no go (32k is enough)'}")
            rep["pretrain_data"] = {"diffs": diffs, "mean": m, "go": go}
    L += ["", "## 3. The labeller from the masked-note pretraining (20k steps each)", ""]
    f = runs / "nl_pt" / "pretrain_log.json"
    if f.exists():
        lg = json.loads(f.read_text())
        if lg:
            L += [f"Pretraining: hidden notes' pitch right {lg[-1]['val_pitch_acc']:.1%} of the time at step {lg[-1]['step']}.", ""]
    L += ["| test set | labeller | duration acc | note F1 | onset F1 |", "|---|---|---|---|---|"]
    for cond in METER_CONDS[:4]:
        for nl in ("nl_gq_p20k", "nl_gq_ptl_p20k"):
            f = runs / nl / "eval_pieces_best.json"
            r = json.loads(f.read_text()).get(cond, {}).get("grid_true") if f.exists() else None
            if r:
                L.append(f"| {cond} | `{nl}` | {r['duration_acc']:.3f} | {r['note_f1']:.3f} | {r['onset_f1']:.3f} |")
    L.append("")
    rows, rep["labeller"] = pair_rows(res, [(sysname("nl_gq_ptl_p20k", "meter_v4_gq"), sysname("nl_gq_p20k", "meter_v4_gq"))])
    L += rows + ["", "The beat errors: outputs/beat_errors/report.md.", ""]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
