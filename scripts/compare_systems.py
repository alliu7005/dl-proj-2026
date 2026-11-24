"""Paired comparisons of whole-piece systems (outputs/system_eval/results.json, beyer_dai.py score):
for each pair, the difference in MUSTER mean error per performance, with a 95% interval from a
bootstrap over PIECES (performances of one piece are not independent), and on how many
performances the first system is better. The tables in outputs/system_eval/results.md say how
each system does; this says which differences are larger than chance.

    python3 scripts/compare_systems.py                       # the default pairs -> outputs/system_eval/paired.md
    python3 scripts/compare_systems.py --pair A B --pair C D
    python3 scripts/compare_systems.py --ablation outputs/system_eval_ablation/results.json

Standard library only (runs on PACE's login node).
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

BASE = "v2_nl_base_gn_meter_v3"            # the round-3 system: each round-4 change is measured from it
TP = BASE + "_tp"
RELEASE = TP + "_d95_tidy"                 # the released configuration (src/release.py)
SEQ = "sf_base_reg_cab_c128_meter_sb"      # the best seq2seq system
PAIRS = [  # (A, B, what A changes)
    (TP, BASE, "tempo-prior decoder (meter_v3_tp)"),
    (BASE + "_tidy", BASE, "length tidying"),
    (BASE + "_d80", BASE, "delete only above p 0.8"),
    (BASE + "_d95", BASE, "delete only above p 0.95"),
    (BASE + "_keepall", BASE, "keep every note"),
    ("v2_nl_base_gn_s1_meter_v3", BASE, "the labeller's second seed (chance)"),
    ("v2_nl_base_gn_f1_meter_v3_tp", TP, "labeller kept on onset F1"),
    ("v2_nl_base_gn1_f1_meter_v3_tp", TP, "labeller kept on onset F1, all windows tracker-timed"),
    (SEQ, TP, "seq2seq (best) against v2 with the tempo prior"),
    ("beyer_dai", TP, "Beyer & Dai against v2 with the tempo prior"),
    (RELEASE, BASE, "the release: tempo prior + d95 + tidy"),
    (TP + "_d80_tidy", RELEASE, "the release at d80"),
    ("v2_nl_base_gn_s1_meter_v3_tp_d95_tidy", RELEASE, "the release, labeller's second seed (chance)"),
    (SEQ, RELEASE, "seq2seq (best) against the release"),
    ("beyer_dai", RELEASE, "Beyer & Dai against the release"),
]


def load(path):
    """{(system, condition): {performance: MUSTER mean}}"""
    out = defaultdict(dict)
    for e in json.loads(Path(path).read_text())["pieces"]:
        m = e.get("muster") or {}
        if m.get("MeanER") is not None:
            out[(e["system"], e["condition"])][e["performance"]] = m["MeanER"]
    return out


def paired(a, b, n_boot=4000, seed=0):
    """A - B on the performances both scored: n, pieces, mean, 95% interval, how many A is better on."""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    by_piece = defaultdict(list)
    for p in common:
        by_piece[p.rsplit("/", 1)[0]].append(a[p] - b[p])
    groups = list(by_piece.values())
    diffs = [x for g in groups for x in g]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [x for g in (rng.choice(groups) for _ in groups) for x in g]
        boots.append(sum(s) / len(s))
    boots.sort()
    return {"n": len(diffs), "pieces": len(groups), "mean": sum(diffs) / len(diffs),
            "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot) - 1],
            "better": sum(x < 0 for x in diffs)}


def row(label, a, b, r):
    sure = r["hi"] < 0 or r["lo"] > 0
    d = "{:+.2f}".format(r["mean"])
    return "| {} | `{}` | `{}` | {} | [{:+.2f}, {:+.2f}] | {}/{} | {} |".format(
        label, a, b, "**" + d + "**" if sure else d, r["lo"], r["hi"], r["better"], r["n"], r["pieces"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="outputs/system_eval/results.json")
    ap.add_argument("--ablation", default="outputs/system_eval_ablation/results.json",
                    help="the beat ablation's results (its _ab system against the same system), if there")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"), help="instead of the default pairs")
    ap.add_argument("--out", default="outputs/system_eval/paired.md")
    args = ap.parse_args(argv)
    if not Path(args.results).exists():
        sys.exit("{} not found: run beyer_dai.py score first".format(args.results))
    res = load(args.results)
    pairs = [(a, b, "") for a, b in args.pair] if args.pair else PAIRS
    lines = ["# Paired comparisons of whole-piece systems", "",
             "MUSTER mean error, system A minus system B, per performance (negative: A is better). "
             "Interval: 95% bootstrap over pieces. Bold: the interval excludes 0. `better on`: "
             "performances where A's error is lower. From `{}` (`scripts/compare_systems.py`).".format(args.results), ""]
    for cond in ("real", "clean"):
        rows = []
        for a, b, label in pairs:
            if (a, cond) in res and (b, cond) in res:
                r = paired(res[(a, cond)], res[(b, cond)])
                if r:
                    rows.append(row(label or "{} vs {}".format(a, b), a, b, r))
        if rows:
            lines += ["## {}".format(cond), "",
                      "| what | A | B | A - B | 95% interval | A better on | pieces |", "|---|---|---|---|---|---|---|"]
            lines += rows + [""]
    if Path(args.ablation).exists():
        ab = load(args.ablation)
        rows = []
        for (s, cond) in sorted(ab):
            if s.endswith("_ab") and (s[:-3], cond) in ab:
                r = paired(ab[(s, cond)], ab[(s[:-3], cond)])
                if r:
                    rows.append(row("the recording's beats instead of the rendered MIDI's ({})".format(cond),
                                    s, s[:-3], r))
        if rows:
            lines += ["## beat ablation", "", "| what | A | B | A - B | 95% interval | A better on | pieces |",
                      "|---|---|---|---|---|---|---|"] + rows + [""]
    text = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    print(text)
    print("wrote " + args.out)


if __name__ == "__main__":
    main()
