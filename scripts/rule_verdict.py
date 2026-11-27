"""Which of the correction rules the demo keeps, decided from the whole-piece benchmark.

    python scripts/rule_verdict.py --labeller nl_gq_mc
    python scripts/rule_verdict.py --labeller auto        # the labeller the Hugging Face release serves
    python scripts/publish_model.py --settings outputs/rule_verdict.json   # the demo then follows it

The project keeps as much of the decision with the models as it can: a rule that corrects a model's
output after the fact stays on only when the benchmark shows the models need it. The deep-learning
configuration is the labeller + its beat-level decoding (src/beat_decode.py, tuned on validation)
on the released meter model; each correction rule is measured added to it, on whole pieces
(scripts/system_scores.py, Beyer & Dai's MUSTER mean error per performance, paired, 95% bootstrap
interval over pieces: scripts/compare_systems.py paired), on the demo's input: real recordings
(ASAP "real", Gangqinpu "gangqinpu_real").

  beat_decode   the model side: on unless it makes the system significantly worse on a condition
  timing_check  src/timing.py check_timing           on only when it makes the system significantly
  timing_beats  src/timing.py check_beats            better on a condition and significantly worse
  fix_jumps     src/readable.py fix_jumps            on none (the burden of proof is on the rule)

A rule whose systems are missing keeps its current setting ("unknown"). The engraving rules (length
tidying, readable lengths and stems, glissando lines, grace notes and rolls played back) decide how
a rhythm is written, not what it is; they are not the models' job and stay on (src/release.py).
The verdict goes to outputs/rule_verdict.json and .md; publish_model.py --settings puts its rules in
the release manifest, which the demo and scripts/audio_to_score.py read (src/release.py
rule_settings), so the demo turns off what the models made unnecessary.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_systems import load, paired  # noqa: E402

METER = "meter_v4_gq"
CONDITIONS = (("outputs/system_eval/results.json", "real"),
              ("outputs/system_eval_gangqinpu/results.json", "gangqinpu_real"))
CORRECTIONS = {"timing_check": "tc", "timing_beats": "tb", "fix_jumps": "jf"}   # the rule -> its system suffix


def base_name(labeller: str, meter: str = METER) -> str:
    """The system without rules, as scripts/system_scores.py names it (tidy, delete above p 0.95)."""
    return f"v2_{labeller}_{meter}_d95_tidy"


def with_suffixes(base: str, *flags: str) -> str:
    """The system with these options, in system_name's order (tc, tb, rd, jf, bd)."""
    order = ("tc", "tb", "rd", "jf", "bd")
    return base + "".join("_" + f for f in order if f in flags)


def significant(r: dict | None) -> int:
    """-1: A significantly better (lower error), +1: significantly worse, 0: neither / unknown."""
    if not r:
        return 0
    return -1 if r["hi"] < 0 else (1 if r["lo"] > 0 else 0)


def verdict(results: dict, labeller: str, meter: str = METER, conditions=None) -> dict:
    """``results``: {(system, condition): {performance: MUSTER mean}} (compare_systems.load, merged
    over the result files). Returns {"rules": {rule: bool or None}, "evidence": ..., ...}."""
    conditions = conditions or [c for _, c in CONDITIONS]
    base = base_name(labeller, meter)
    dl = with_suffixes(base, "bd")
    ev, rules = {}, {}

    def compare(a, b):
        out = {}
        for c in conditions:
            if (a, c) in results and (b, c) in results:
                r = paired(results[(a, c)], results[(b, c)])
                if r:
                    out[c] = dict(r, A=a, B=b, verdict=significant(r))
        return out

    def complete(e):   # "worse on none" needs every condition measured: else undecided (the setting stays)
        return len(e) == len(conditions)
    # the model side: the beat decoding stays unless it hurts
    e = compare(dl, base)
    ev["beat_decode"] = e
    rules["beat_decode"] = None if not complete(e) else not any(v["verdict"] > 0 for v in e.values())
    # the corrections: on only when they help and hurt nowhere
    for rule, flag in CORRECTIONS.items():
        e = compare(with_suffixes(base, flag, "bd"), dl)
        ev[rule] = e
        if not complete(e):
            rules[rule] = None
        else:
            helps = any(v["verdict"] < 0 for v in e.values())
            hurts = any(v["verdict"] > 0 for v in e.values())
            rules[rule] = helps and not hurts
    # for the report: every rule the demo has (corrections and engraving) against the models alone
    ev["all_rules"] = compare(with_suffixes(base, "tc", "tb", "rd", "jf", "bd"), dl)
    return {"labeller": labeller, "meter": meter, "conditions": list(conditions), "rules": rules,
            "evidence": ev, "date": datetime.date.today().isoformat(),
            "deep_learning_config": dl,
            "rule": "beat_decode: on unless significantly worse; corrections: on only if significantly better on a "
                    "condition and significantly worse on none (95% bootstrap over pieces, MUSTER mean error)"}


def report(v: dict) -> str:
    lines = [f"# Rule verdict ({v['date']})", "",
             f"Deep-learning configuration: `{v['deep_learning_config']}` (labeller `{v['labeller']}`, meter `{v['meter']}`). "
             "MUSTER mean error, A minus B per performance (negative: A better), 95% bootstrap over pieces.", "",
             "| rule | on? | condition | A - B | 95% interval | pieces |", "|---|---|---|---|---|---|"]
    for rule in ("beat_decode", *CORRECTIONS, "all_rules"):
        on = v["rules"].get(rule)
        mark = {True: "on", False: "**off**", None: "unknown (kept as it is)"}.get(on, "") if rule in v["rules"] else "-"
        e = v["evidence"].get(rule) or {}
        if not e:
            lines.append(f"| {rule} | {mark} | - | - | - | - |")
        for c, r in e.items():
            lines.append(f"| {rule} | {mark} | {c} | {r['mean']:+.2f} | [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['pieces']} |")
    return "\n".join(lines) + "\n"


def released_labeller() -> str:
    """The run the Hugging Face release's labeller came from (its manifest)."""
    import huggingface_hub as hub

    from src.release import HF_REPO
    m = json.loads(Path(hub.hf_hub_download(repo_id=HF_REPO, filename="manifest.json", repo_type="model")).read_text())
    return m["models"]["v2_labeller"]["source_run"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labeller", default="auto", help="its run name, or auto: the one the release serves")
    ap.add_argument("--meter", default=METER)
    ap.add_argument("--out", default="outputs/rule_verdict.json")
    args = ap.parse_args(argv)
    lab = released_labeller() if args.labeller == "auto" else args.labeller
    results, conds = {}, []
    for path, cond in CONDITIONS:
        if Path(path).exists():
            results.update({k: v for k, v in load(path).items() if k[1] == cond})
            conds.append(cond)
        else:
            print(f"note: no {path}: {cond} not judged")
    if not conds:
        raise SystemExit("no results: run beyer_dai.py score first")
    v = verdict(results, lab, args.meter, conds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(v, indent=1))
    md = report(v)
    out.with_suffix(".md").write_text(md)
    print(md)
    print(f"wrote {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
