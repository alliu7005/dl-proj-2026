"""What an evaluation computes and what is still missing from a run's results.

One definition of the evaluation's options, mode names and test sets, shared by
scripts/evaluate_pieces.py (which runs it) and pace/reevaluate.sh (which decides what to submit),
so the two can never disagree about whether a result already exists.

Standard library only (and Python 3.6+ syntax), so the PACE login node can run it without the
conda environment:

    python3 -m src.eval.plan --run s2s --conditions real      # what an evaluation would still compute
    python3 -m src.eval.plan --run s2s --extra-margins -0.05,0.05

Exit status: 0 = something is missing (printed one condition per line), 3 = nothing is missing,
4 = the run's training has not finished (outputs/runs/<run>/status.json), 2 = bad arguments.

A result counts as present only if it was computed the same way as the one requested: the same
guard margin for guarded / meter modes, all performances (not a --limit-perfs subset), and the
current EVAL_VERSION. Results written before these were recorded count as version 1, all
performances: every evaluation run since the per-window output cap (the code version 1 describes).
"""
import argparse
import json
import sys
from pathlib import Path

# condition -> data split under data/processed (a missing split is skipped, not an error)
SPLITS = {"clean": "test", "synthetic": "test", "real": "test_transcribed",
          "real_kong": "test_transcribed_kong", "periscope": "test_periscope"}
DEFAULT_MODES = "oracle,chained,none,guarded,meter"
DEFAULT_CONDITIONS = "clean,synthetic,real,real_kong,periscope"
# results derived from another condition's decoding, never requested on their own
DERIVED = {"real_unseen_audio": "real"}
DECODED = ("chained", "guarded", "meter")  # modes decoded piece by piece (src/decoding.py)
# guarded mode keeps the no-context output when its input agreement beats the chained output's by
# more than this. scripts/simulate_guard.py on all six post-milestone models: 0 beats 0.02 / 0.05 /
# 0.1 / 0.2 on note, onset, duration and bar F1 in nearly every model and condition (s2s clean:
# onset F1 0.170 at 0 vs 0.148 at 0.05). Ties keep the chained output.
DEFAULT_GUARD_MARGIN = 0.0
# Bump when decoding or scoring changes what a mode's numbers mean: every older result then counts
# as missing, and pace/reevaluate.sh re-runs it. 1 = per-window output cap (src/decoding.py
# max_new_tokens), guarded with margin recorded, meter with the forced time signature banned
# elsewhere in the window.
EVAL_VERSION = 1
MISSING, NOTHING_MISSING, TRAINING = 0, 3, 4  # exit statuses of the command line


def join_dash_values(argv, flags=("--extra-margins",)):
    """``--extra-margins -0.05,-0.02`` -> ``--extra-margins=-0.05,-0.02``. Before Python 3.13,
    argparse reads a value that starts with "-" and is not a single number as another option
    ("expected one argument"); the joined form works on every version."""
    out, k = [], 0
    while k < len(argv):
        if argv[k] in flags and k + 1 < len(argv) and argv[k + 1].startswith("-"):
            out.append("{}={}".format(argv[k], argv[k + 1]))
            k += 2
        else:
            out.append(argv[k])
            k += 1
    return out


def build_parser():
    """The options of scripts/evaluate_pieces.py (this module's command line reads the same ones)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--modes", default=DEFAULT_MODES)
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    ap.add_argument("--limit-perfs", type=int, default=0, help="performances per condition (0 = all)")
    ap.add_argument("--musicxml", type=int, default=0, help="write MusicXML for the first N pieces")
    ap.add_argument("--guard-margin", type=float, default=DEFAULT_GUARD_MARGIN,
                    help="guarded / meter: use the no-prompt output if its input agreement is higher by more than this")
    ap.add_argument("--extra-margins", default="",
                    help="comma-separated guard margins to evaluate as well; each adds modes guarded@<m> "
                         "(and meter@<m> if meter is run), so margins can be compared in one job")
    ap.add_argument("--only-missing", action="store_true",
                    help="compute only the condition / mode results not already in the run's results "
                         "(same margin, all performances, current EVAL_VERSION); see src/eval/plan.py")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--normal-priority", action="store_true")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--data-dir", default="data/processed")
    return ap


def parse_args(argv):
    args = build_parser().parse_args(join_dash_values(list(argv)))
    for name in args.modes.split(","):
        if name not in ("oracle", "none") + DECODED:
            raise SystemExit("unknown mode {!r} (modes: oracle, none, {})".format(name, ", ".join(DECODED)))
    for cond in args.conditions.split(","):
        if cond not in SPLITS:
            raise SystemExit("unknown condition {!r} (conditions: {})".format(cond, ", ".join(SPLITS)))
    return args


def extra_margins(text):
    return [float(m) for m in text.split(",") if m.strip()]


def decode_passes(modes, guard_margin, extras):
    """[(mode name, kind, guard margin)] for the piece-by-piece modes requested, in decoding order:
    chained once, then guarded and meter at the main margin (named "guarded" / "meter") and at each
    extra margin ("guarded@-0.05", "meter@-0.05")."""
    passes = []
    for kind in DECODED:
        if kind not in modes:
            continue
        for k, m in enumerate([guard_margin] + [m for m in extras if m != guard_margin]):
            if kind == "chained" and k:
                break
            passes.append((kind if k == 0 else "{}@{:g}".format(kind, m), kind, m))
    return passes


def mode_names(modes, guard_margin, extras):
    """Every result name the request produces: oracle / none first, then decode_passes."""
    return [m for m in modes if m not in DECODED] + [n for n, _, _ in decode_passes(modes, guard_margin, extras)]


def margins_of(modes, guard_margin, extras):
    return {n: m for n, _, m in decode_passes(modes, guard_margin, extras) if not n.startswith("chained")}


def stamp(result, margin=None, limit_perfs=0):
    """Record how a result was computed (what present() compares)."""
    result["eval_version"] = EVAL_VERSION
    result["limit_perfs"] = int(limit_perfs)
    if margin is not None:
        result["guard_margin"] = margin
    return result


def present(result, margin=None, limit_perfs=0):
    """Is this stored result the one requested?"""
    if not isinstance(result, dict):
        return False
    if result.get("eval_version", 1) != EVAL_VERSION or int(result.get("limit_perfs", 0)) != int(limit_perfs):
        return False
    if margin is not None and ("guard_margin" not in result or abs(float(result["guard_margin"]) - margin) > 1e-9):
        return False  # guarded / meter results always record their margin: one without is unknown
    return True


def missing(results, conditions, names, margins, available, limit_perfs=0):
    """{condition: [mode names still to compute]} for the conditions whose data exists.
    results: metrics[condition][mode] (eval_pieces_<ckpt>.json); available(cond) -> bool."""
    out = {}
    for cond in conditions:
        if not available(cond):
            continue
        have = results.get(cond, {})
        need = [n for n in names if not present(have.get(n), margins.get(n), limit_perfs)]
        if cond == "real":  # its unseen-audio subset is written by the same decoding, at the same
            # time as the real result whose margin was just checked; before this module it was
            # written without a guard_margin, so only a recorded one is compared
            sub = results.get("real_unseen_audio", {})

            def sub_ok(n):
                r = sub.get(n)
                return present(r, None, limit_perfs) and (
                    margins.get(n) is None or "guard_margin" not in r or present(r, margins[n], limit_perfs))
            need += [n for n in names if n not in need and not sub_ok(n)]
            need = [n for n in names if n in need]  # keep the decoding order
        if need:
            out[cond] = need
    return out


def split_available(data_dir):
    data_dir = Path(data_dir)
    return lambda cond: (data_dir / SPLITS[cond] / "index.npy").exists()


def results_path(run_dir, ckpt):
    return Path(run_dir) / "eval_pieces_{}.json".format(Path(ckpt).stem)


def load_results(run_dir, ckpt):
    p = results_path(run_dir, ckpt)
    return json.loads(p.read_text()) if p.exists() else {}


def training_state(run_dir):
    """The training phase recorded by scripts/train_model.py in status.json ("train", "cooldown",
    "done"). Runs trained before status.json existed have none: they all finished, so "done"."""
    p = Path(run_dir) / "status.json"
    if not p.exists():
        return "done"
    try:
        return json.loads(p.read_text()).get("phase", "done")
    except ValueError:  # being rewritten right now: a training job is running
        return "train"


def plan(args):
    """(training phase, {condition: [missing modes]}) for parsed evaluate_pieces arguments."""
    run_dir = Path(args.runs_dir) / args.run
    modes = args.modes.split(",")
    extras = extra_margins(args.extra_margins)
    todo = missing(load_results(run_dir, args.ckpt), args.conditions.split(","),
                   mode_names(modes, args.guard_margin, extras), margins_of(modes, args.guard_margin, extras),
                   split_available(args.data_dir), args.limit_perfs)
    return training_state(run_dir), todo


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    phase, todo = plan(args)
    if phase != "done":
        print("training not finished (phase {})".format(phase))
        return TRAINING
    if not todo:
        print("nothing missing")
        return NOTHING_MISSING
    for cond, names in todo.items():
        print("{}: {}".format(cond, ",".join(names)))
    return MISSING


if __name__ == "__main__":
    sys.exit(main())
