"""The articulation pilot (pace/v2_articulation.sh) -> outputs/articulation/report.md (+ report.json).

    python scripts/articulation_report.py

Does the labeller write simpler, truer lengths when it learns that played and written lengths
differ? nl_gq's recipe for 20k steps without (nl_gq_p20k) and with (nl_gq_art_p20k) half its
training windows played with another articulation (src/dataset.py articulate: detached, legato,
held, stretch by stretch, the targets unchanged).
1. Test windows on the true grid (eval_pieces_best.json): rendered duration accuracy, note F1,
   onset F1, the dur head's accuracy.
2. Whole pieces with meter_v4_gq (tidy, d95), ASAP and Gangqinpu: how often the scores use a rare
   rhythm (scripts/simplicity_report.py rare_shares) against the reference scores, MUSTER and its
   offset part, score similarity's duration error, MUSTER paired (with minus without).
3. Go: rare lengths down on whole pieces, MUSTER not worse beyond chance anywhere, window duration
   accuracy not down by more than 0.01 anywhere.
Needs music21 (the dl-proj environment).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE, ART = "nl_gq_p20k", "nl_gq_art_p20k"
SETS = (("asap", "outputs/system_eval", ("real", "clean")),
        ("gq", "outputs/system_eval_gangqinpu", ("gangqinpu_real", "gangqinpu")))


def system(nl: str) -> str:
    return f"v2_{nl}_meter_v4_gq_d95_tidy"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--max-pieces", type=int, default=40)
    ap.add_argument("--out", default="outputs/articulation")
    args = ap.parse_args(argv)
    from scripts.compare_systems import load, paired
    from scripts.simplicity_report import rare_shares
    L = ["# Articulation pilot: written lengths that do not follow played ones", "",
         f"`{BASE}`: nl_gq's recipe, 20k steps. `{ART}`: the same with half the training windows played detached, "
         "legato or held (the targets unchanged).", "", "## 1. Test windows (true grid)", "",
         "| test set | labeller | duration acc | note F1 | onset F1 | dur head |", "|---|---|---|---|---|---|"]
    rep = {"windows": {}, "pieces": {}}
    win = {}
    for nl in (BASE, ART):
        f = Path(args.runs_dir) / nl / "eval_pieces_best.json"
        win[nl] = json.loads(f.read_text()) if f.exists() else {}
    dur_drop = []
    for cond in sorted(set(win[BASE]) | set(win[ART])):
        vals = {}
        for nl in (BASE, ART):
            modes = win[nl].get(cond, {})
            mode = next((m for m in ("grid_true", "true") if m in modes), next(iter(modes), None))
            r = modes.get(mode) if mode else None
            if not r or "duration_acc" not in r:
                continue
            vals[nl] = r
            L.append(f"| {cond} | `{nl}` | {r['duration_acc']:.3f} | {r['note_f1']:.3f} | {r['onset_f1']:.3f} | "
                     f"{r.get('head_dur_acc', float('nan')):.3f} |")
        if len(vals) == 2:
            dur_drop.append(vals[ART]["duration_acc"] - vals[BASE]["duration_acc"])
            rep["windows"][cond] = {nl: {k: vals[nl][k] for k in ("duration_acc", "note_f1", "onset_f1")} for nl in vals}
    L += ["", "## 2. Whole pieces (with meter_v4_gq, tidy, d95)", ""]
    rare_down, worse = [], []
    for key, root, conds in SETS:
        root = Path(root)
        res_f = root / "results.json"
        res = json.loads(res_f.read_text())["results"] if res_f.exists() else {}
        muster = load(res_f) if res_f.exists() else {}
        manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else []
        for cond in conds:
            L += [f"### {cond}", "", "| system | rare onsets | rare lengths | MUSTER | offset error | duration error (sim) |",
                  "|---|---|---|---|---|---|"]
            refs = [e["reference"] for e in manifest if e["condition"] == cond and e.get("reference")][: args.max_pieces]
            rr = [r for r in (rare_shares(x) for x in refs) if r]
            if rr:
                L.append(f"| the reference scores | {sum(r[0] for r in rr) / len(rr):.1%} | {sum(r[1] for r in rr) / len(rr):.1%} | - | - | - |")
            lens = {}
            for nl in (BASE, ART):
                name = system(nl)
                summ = root / name / "summary.json"
                if not summ.exists():
                    L.append(f"| `{name}` | not run | | | | |")
                    continue
                pieces = [p for p in json.loads(summ.read_text())["pieces"] if p["condition"] == cond and p.get("musicxml")]
                rows = [r for r in (rare_shares(p["musicxml"]) for p in pieces[: args.max_pieces]) if r]
                r_all = res.get(cond, {}).get(name, {}).get("all") or {}
                mu, sim = r_all.get("muster") or {}, r_all.get("sim") or {}
                fmt = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
                ro = sum(r[0] for r in rows) / len(rows) if rows else None
                rl = sum(r[1] for r in rows) / len(rows) if rows else None
                lens[nl] = rl
                L.append(f"| `{name}` | {'-' if ro is None else f'{ro:.1%}'} | {'-' if rl is None else f'{rl:.1%}'} | "
                         f"{fmt(mu.get('MeanER'))} | {fmt(mu.get('OffsetER'))} | {fmt(sim.get('NoteDuration'))} |")
                rep["pieces"][f"{name}|{cond}"] = {"rare_onsets": ro, "rare_lengths": rl, "muster": mu, "sim": sim}
            pr = paired(muster.get((system(ART), cond), {}), muster.get((system(BASE), cond), {}))
            if pr:
                sure = pr["hi"] < 0 or pr["lo"] > 0
                L += ["", f"MUSTER, with minus without: {'**' if sure else ''}{pr['mean']:+.2f}{'**' if sure else ''} "
                      f"[{pr['lo']:+.2f}, {pr['hi']:+.2f}] (n {pr['n']})"]
                if pr["lo"] > 0:
                    worse.append(cond)
            if lens.get(BASE) is not None and lens.get(ART) is not None:
                rare_down.append(lens[ART] - lens[BASE])
            L.append("")
    go = bool(rare_down) and sum(rare_down) / len(rare_down) < 0 and not worse and all(d >= -0.01 for d in dur_drop)
    L += ["## 3. Verdict", "",
          f"- rare written lengths, with minus without: {', '.join(f'{d:+.1%}' for d in rare_down) or 'not measured'}",
          f"- MUSTER worse beyond chance on: {', '.join(worse) or 'none'}",
          f"- window duration accuracy, with minus without: {', '.join(f'{d:+.3f}' for d in dur_drop) or 'not measured'}",
          f"- **{'go' if go else 'no go'}** for nl_gq's full recipe with --articulation 0.5 "
          "(go: rare lengths down, MUSTER not worse beyond chance anywhere, window duration accuracy not down by more than 0.01)", ""]
    rep["go"] = go
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
