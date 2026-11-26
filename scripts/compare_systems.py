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
GQ_NL = "v2_nl_gq"                         # the labeller trained with Gangqinpu's scores
PMHS = GQ_NL + "_meter_v4_gq_pmhs_d95_tidy_pm_rpm_hos"   # v2 with PM2S retrained stably and its meter model
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
    ("v2_nl_base_gn_meter_v3_th_d95_tidy", RELEASE, "the release with the tempo head choosing the level (frozen trunk)"),
    ("v2_nl_base_gn_meter_v3_ft_d95_tidy", RELEASE, "the release with meter_v3 fine-tuned with the tempo head"),
    (SEQ, RELEASE, "seq2seq (best) against the release"),
    ("beyer_dai", RELEASE, "Beyer & Dai against the release"),
    # sixth round (pace/v2_gangqinpu.sh): each change against the release
    ("v2_nl_base_gn_meter_v3_tp_rf_d95_tidy", RELEASE, "the release, beat times from the notes (snap + fill)"),
    ("v2_nl_base_gn_meter_v3_tp_fill_d95_tidy", RELEASE, "the release, missed beats filled from the notes"),
    (RELEASE + "_pm_rpm", RELEASE, "the release with PM2S's beats instead of Beat This!'s"),
    ("v2_nl_base_gn_meter_v4_gq_d95_tidy", RELEASE, "meter model trained with Gangqinpu (Beat This!)"),
    ("v2_nl_base_gn_meter_v4_gq_both_d95_tidy", RELEASE, "meter model trained with Gangqinpu (Beat This! + PM2S)"),
    ("v2_nl_base_gn_meter_v4_gq_pm_d95_tidy_pm_rpm", RELEASE + "_pm_rpm", "meter model trained with Gangqinpu for PM2S, both on PM2S"),
    ("v2_nl_base_gn_meter_v4_gq_both_d95_tidy", "v2_nl_base_gn_meter_v4_gq_d95_tidy", "PM2S as evidence on Beat This!'s beats"),
    (SEQ, "v2_nl_base_gn_meter_v4_gq_d95_tidy", "seq2seq (best) against v2 with the Gangqinpu meter model"),
    ("beyer_dai", "v2_nl_base_gn_meter_v4_gq_d95_tidy", "Beyer & Dai against v2 with the Gangqinpu meter model"),
    # seventh round (pace/v2_pm2s.sh): PM2S retrained without our test pieces (tag ho)
    (RELEASE + "_pm_rpm_ho", RELEASE, "the release with PM2S's beats, PM2S trained without our test pieces"),
    (RELEASE + "_pm_rpm_ho", RELEASE + "_pm_rpm", "PM2S without our test pieces vs the released PM2S (the leak)"),
    ("v2_nl_base_gn_meter_v4_gq_pm_d95_tidy_pm_rpm_ho", RELEASE, "meter_v4_gq_pm with the held-out PM2S's beats"),
    ("v2_nl_base_gn_meter_v4_gq_pmho_d95_tidy_pm_rpm_ho", RELEASE, "meter model trained for the held-out PM2S, with it"),
    ("v2_nl_base_gn_meter_v4_gq_pmho_d95_tidy_pm_rpm_ho", "v2_nl_base_gn_meter_v4_gq_pm_d95_tidy_pm_rpm_ho",
     "meter model trained for the held-out PM2S vs for the released one"),
    (SEQ, "v2_nl_base_gn_meter_v4_gq_pmho_d95_tidy_pm_rpm_ho", "seq2seq (best) against v2 with the held-out PM2S"),
    ("beyer_dai", "v2_nl_base_gn_meter_v4_gq_pmho_d95_tidy_pm_rpm_ho", "Beyer & Dai against v2 with the held-out PM2S"),
    # eighth round (pace/v2_gqlabel.sh): the labeller trained with Gangqinpu's scores (nl_gq), on ASAP
    # and on Gangqinpu's test pieces (outputs/system_eval_gangqinpu)
    (GQ_NL + "_meter_v3_tp_d95_tidy", RELEASE, "the release with the labeller trained with Gangqinpu"),
    (GQ_NL + "_meter_v4_gq_d95_tidy", "v2_nl_base_gn_meter_v4_gq_d95_tidy",
     "labeller trained with Gangqinpu, both on the Gangqinpu meter model"),
    (GQ_NL + "_meter_v4_gq_d95_tidy", RELEASE, "both models trained with Gangqinpu"),
    ("v2_nl_base_gn_meter_v4_gq_d95_tidy", RELEASE, "meter model trained with Gangqinpu (Beat This!)"),
    ("v2_nl_base_gn_meter_v4_gq_pm_d95_tidy_pm_rpm", "v2_nl_base_gn_meter_v4_gq_d95_tidy",
     "PM2S's beats against Beat This!'s (each with its Gangqinpu meter model)"),
    ("v2_nl_base_gn_meter_v3_s1_tp_d95_tidy", RELEASE, "the release, the meter model's second seed (chance)"),
    ("v2_nl_base_gn_meter_v4_gql_d95_tidy", "v2_nl_base_gn_meter_v4_gq_d95_tidy",
     "the Gangqinpu meter model trained longer (100k steps)"),
    (SEQ, GQ_NL + "_meter_v4_gq_d95_tidy", "seq2seq (best) against v2 with both models trained with Gangqinpu"),
    ("beyer_dai", GQ_NL + "_meter_v4_gq_d95_tidy", "Beyer & Dai against v2 with both models trained with Gangqinpu"),
    (SEQ, RELEASE, "seq2seq (best) against the release"),
    ("beyer_dai", RELEASE, "Beyer & Dai against the release"),
    # ninth round (pace/v2_pm2s2.sh): PM2S retrained with stable optimisation (tag hos), with nl_gq
    (RELEASE + "_pm_rpm_hos", RELEASE + "_pm_rpm_ho", "PM2S retrained stably vs its first retraining"),
    (RELEASE + "_pm_rpm_hos", RELEASE + "_pm_rpm", "PM2S retrained stably vs the released PM2S (what the leak was worth)"),
    (RELEASE + "_pm_rpm_hos", RELEASE, "the release with the stably retrained PM2S"),
    (PMHS, GQ_NL + "_meter_v4_gq_d95_tidy", "PM2S + its meter model vs the default (Beat This!; on clean MIDI: rendered)"),
    (PMHS, "v2_nl_base_gn_meter_v4_gq_pmho_d95_tidy_pm_rpm_ho", "the stable PM2S path vs the first one"),
    (GQ_NL + "_meter_v4_gq_d95_tidy_rsb", GQ_NL + "_meter_v4_gq_d95_tidy", "Beat This! on the rendered MIDI vs on the recording"),
    (PMHS, GQ_NL + "_meter_v4_gq_d95_tidy_rsb", "MIDI input: PM2S + its meter model vs Beat This! on the rendered MIDI"),
    (SEQ, PMHS, "seq2seq (best) against v2 with the stable PM2S"),
    ("beyer_dai", PMHS, "Beyer & Dai against v2 with the stable PM2S"),
    # the timing check (src/timing.py; pace/v2_pm2s2.sh p2sys, or pace/v2_timing.sh)
    (GQ_NL + "_meter_v4_gq_d95_tidy_tc", GQ_NL + "_meter_v4_gq_d95_tidy",
     "the timing check: triplets the timing contradicts written straight"),
    # the beat-level timing check (src/timing.py check_beats), the readability rules and the short-bar
    # fix (src/readable.py), each on the timing check (pace/v2_decode.sh)
    (GQ_NL + "_meter_v4_gq_d95_tidy_tc_tb", GQ_NL + "_meter_v4_gq_d95_tidy_tc",
     "the beat-level timing check: beats played straight written straight, not as the labeller's triplets"),
    (GQ_NL + "_meter_v4_gq_d95_tidy_tc_rd", GQ_NL + "_meter_v4_gq_d95_tidy_tc",
     "the readability rules: 16th rests closed, small overlaps trimmed (stems, hidden rests)"),
    (GQ_NL + "_meter_v4_gq_d95_tidy_tc_jf", GQ_NL + "_meter_v4_gq_d95_tidy_tc",
     "the short-bar fix: bars the meter decoder stretched with a jump shortened"),
    (GQ_NL + "_meter_v4_gq_d95_tidy_tc_tb_rd_jf", GQ_NL + "_meter_v4_gq_d95_tidy_tc",
     "all three (the demo's default) against the timing check alone"),
    (GQ_NL + "_meter_v4_gq_d95_tidy_tc_tb_rd_jf", GQ_NL + "_meter_v4_gq_d95_tidy",
     "the rules against the plain system"),
    # tenth round (pace/v2_decode.sh): the deep-learning side of the same problems
    (GQ_NL + "_meter_v4_gq_d95_tidy_tc_bd", GQ_NL + "_meter_v4_gq_d95_tidy_tc",
     "beat-level decoding of the labeller (its probabilities + the timing, tuned on validation)"),
    (GQ_NL + "_meter_v4_gq_d95_tidy_tc_bd", GQ_NL + "_meter_v4_gq_d95_tidy_tc_tb",
     "beat-level decoding against the whole-beat rule"),
    ("v2_nl_gq_mc_meter_v4_gq_d95_tidy_tc", GQ_NL + "_meter_v4_gq_d95_tidy_tc",
     "the labeller retrained without the contradicted labels (written triplet, played straight)"),
    (GQ_NL + "_meter_v4_gq_lt_d95_tidy_tc", GQ_NL + "_meter_v4_gq_d95_tidy_tc",
     "the meter decoder with local tempo consistency"),
    (GQ_NL + "_meter_v4_gq_d95_tidy_tc_tb_rd_jf_bd", GQ_NL + "_meter_v4_gq_d95_tidy",
     "the demo's default against the plain system"),
]


def load(path):
    """{(system, condition): {performance: MUSTER mean}}"""
    out = defaultdict(dict)
    for e in json.loads(Path(path).read_text())["pieces"]:
        m = e.get("muster") or {}
        if m.get("MeanER") is not None:
            out[(e["system"], e["condition"])][e["performance"]] = m["MeanER"]
    return out


def piece_of(performance):
    """The piece a performance is of: ASAP's folder (``Bach/Prelude/bwv_875/Ahfat01M.mid`` ->
    ``Bach/Prelude/bwv_875``), whose performances are not independent. A Gangqinpu performance
    (``gangqinpu/<sample id>``, scripts/gangqinpu_inputs.py: one per score) is its own piece; the
    first version grouped all of them into one ("gangqinpu"), which gave zero-width intervals."""
    if performance.startswith("gangqinpu/"):
        return performance
    return performance.rsplit("/", 1)[0]


def paired(a, b, n_boot=4000, seed=0):
    """A - B on the performances both scored: n, pieces, mean, 95% interval, how many A is better on."""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    by_piece = defaultdict(list)
    for p in common:
        by_piece[piece_of(p)].append(a[p] - b[p])
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
                    help="the beat ablation's results (its _ab system against the same system), if there; '' = none")
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
    for cond in ("real", "clean", "gangqinpu_real", "gangqinpu"):
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
    if args.ablation and Path(args.ablation).exists():
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
