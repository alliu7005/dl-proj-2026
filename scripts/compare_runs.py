"""Compare every evaluated run in readable tables (standard library only: runs on the PACE login node).

    python3 scripts/compare_runs.py                    # all runs -> outputs/compare/
    python3 scripts/compare_runs.py --runs s2s,s2s_c   # only these runs

Reads outputs/runs/<run>/eval_pieces_best.json (scripts/evaluate_pieces.py), config.json and
log.csv, and writes to outputs/compare/:

  summary.md    the runs (size, settings, validation loss, how training ended), then per test set
                the system output (meter mode: what the demo runs), onset / bar F1 in every
                context mode, the note labeller (v2) on every grid, the time-signature search
                (meter vs ts_true vs tsearch / tsearch_beat: onset F1, bar F1, time signature),
                and the meter models (outputs/runs/<run>/eval_meter.json: their grids' beat /
                downbeat F and time-signature accuracy). Bold = best in its column.
  margins.md    the guard-margin sweeps (runs evaluated with --extra-margins), per test set
  results.csv   every number: one row per run, test set and mode, for a spreadsheet

Markdown tables are padded so they line up as plain text too; open the .md files in VS Code
(Ctrl+Shift+V) or on GitHub to see them rendered. summary.md is also printed.
"""
import argparse
import csv
import datetime
import fnmatch
import json
import sys
from pathlib import Path

CONDITIONS = [  # (key in the results, description)
    ("clean", "ASAP test, clean performance MIDI"),
    ("synthetic", "ASAP test, fixed synthetic transcription noise"),
    ("periscope", "PERiScoPe test, real piano transcriptions"),
    ("real", "ASAP test performances transcribed from audio by Transkun V2"),
    ("real_unseen_audio", "the subset of `real` whose audio Transkun never trained on"),
    ("clean_gt_beats", "beat models only: clean MIDI with the annotated beats (what the beat-snap baselines get)"),
    ("real_gt_beats", "beat models only: the real transcriptions with the annotated beats (upper bound)"),
    ("real_no_beats", "beat models only: the real transcriptions without any beats (MIDI input, no tracker)"),
]
MODES = ["oracle", "chained", "none", "guarded", "meter", "piece"]
SYSTEM = "meter"  # the mode the demo runs (guarded + time-signature consistency)
TS_MODES = ["meter", "ts_true", "tsearch", "tsearch_beat"]  # the time-signature search table
# note labeller (v2) results, one per grid it was given (scripts/eval_labeller.py)
LABELLER_MODES = [("grid_true", "true grid")]
TS_HELP = ("ts_true = guarded with each window's true time signature forced (the most a time-signature "
           "choice can give); tsearch = the whole piece decoded under its 3 likeliest time signatures, keeping "
           "the one the model finds most probable; tsearch_beat = the same candidates, keeping the one whose "
           "written beats agree best with the tracked beats (models trained with beats; otherwise as tsearch); "
           "changed = share of windows whose time signature tsearch changed from guarded decoding")
METRICS = [("note_f1", "note F1"), ("onset_f1", "onset F1"), ("onset_shift_f1", "onset-shift F1"),
           ("duration_acc", "duration"), ("staff_acc", "staff"), ("bar_f1", "bar F1"), ("ts_acc", "time sig")]
MODE_HELP = ("oracle = the true score before each window (upper bound); chained = the model's own "
             "previous output; none = no context; guarded = chained, falling back to no context when "
             "that output matches the input notes better; meter = guarded + consistent time signatures; "
             "piece = meter with one time signature per piece unless it changes for a long stretch")
DASH = "-"  # ASCII: prints the same in every terminal
METER_CONDITIONS = [  # (key in eval_meter.json, description)
    ("clean", "ASAP test, clean MIDI, the tracker's errors simulated; against the annotations"),
    ("real", "ASAP test transcribed by Transkun V2, Beat This!'s own beats; against the annotations"),
    ("periscope", "PERiScoPe test (104 unseen works), real transcriptions, the tracker simulated; against the "
                  "score's beats timed by the note alignment"),
]
METER_HELP = ("The grid each meter model decodes from the tracked beats of a whole piece (scripts/eval_meter.py): "
              "beat F / downbeat F of the notated beats it implies, time sig = share of bars with the right time "
              "signature, per performance and averaged (per piece: each piece once, however often it was played). "
              "default = the same model with the untuned decoder (the tuned one: decode.json, chosen on validation). "
              "tracker = the tracked beats themselves, the input. Data: A = ASAP, R = real ASAP transcriptions "
              "with Beat This!'s beats, P = PERiScoPe")


def num(x, pct=False, digits=3):
    if not isinstance(x, (int, float)) or isinstance(x, bool) or x != x:  # missing or NaN
        return DASH
    return "{:.0%}".format(x) if pct else "{:.{}f}".format(x, digits)


def numeric(cell):
    t = str(cell).strip("*").rstrip("%M")
    try:
        float(t)
        return True
    except ValueError:
        return t == DASH


def table(header, rows, best=None):
    """A padded markdown table: number columns right-aligned, text left-aligned.
    best: {column index: "max" | "min"} -> that column's best value in bold."""
    rows = [list(r) for r in rows]
    right = [k > 0 and all(numeric(r[k]) for r in rows) for k in range(len(header))]
    for col, how in (best or {}).items():
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col].rstrip("%")))
            except ValueError:
                pass
        if len(vals) > 1:
            target = max(vals) if how == "max" else min(vals)
            for r in rows:
                try:
                    if float(r[col].rstrip("%")) == target:
                        r[col] = "**{}**".format(r[col])
                except ValueError:
                    pass
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]

    def line(cells):
        return "| " + " | ".join(str(c).rjust(w) if r else str(c).ljust(w)
                                 for c, w, r in zip(cells, widths, right)) + " |"
    sep = "|" + "|".join(("-" * (w + 1) + ":") if r else (":" + "-" * (w + 1)) for w, r in zip(widths, right)) + "|"
    return "\n".join([line(header), sep] + [line(r) for r in rows])


def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}


def run_info(d):
    """Settings and training outcome of one run directory (missing files -> dashes)."""
    cfg = load_json(d / "config.json")
    args, model = cfg.get("args", {}), cfg.get("model", {})
    rows = []
    try:
        with open(d / "log.csv", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        pass
    losses = []
    for r in rows:
        try:
            losses.append(float(r["val_loss"]))
        except (KeyError, TypeError, ValueError):
            pass
    last = int(float(rows[-1]["step"])) if rows and rows[-1].get("step") else None
    spe, total = cfg.get("steps_per_epoch"), cfg.get("total_steps")
    status = load_json(d / "status.json")
    if status.get("phase") not in (None, "done"):
        ended = "training ({})".format(status["phase"])
    elif "stopped_early" in status:
        ended = "early stop" if status["stopped_early"] else "full"
    elif last is not None and total:
        ended = "early stop" if last < total else "full"
    else:
        ended = DASH
    arch = cfg.get("arch") or args.get("arch") or "score2seq"
    name = {"score2seq": "Score2Seq", "labeller": "Note labeller {}".format(args.get("size", ""))}.get(
        arch, "Scoreformer {}".format(args.get("size", "")))
    cb, ca = model.get("context_before", 0), model.get("context_after", 0)
    tt = cfg.get("data", {}).get("train_transcribed")
    return {
        "model": name.strip(),
        "params": "{:.1f}M".format(cfg["params"] / 1e6) if cfg.get("params") else DASH,
        "context": "{}/{}".format(cb, ca) if cb or ca else DASH,
        "aux": "yes" if model.get("aux_metrical") else DASH,
        "dropout": args.get("dropout", DASH), "wd": args.get("weight_decay", DASH), "lr": args.get("lr", DASH),
        "best_val": min(losses) if losses else None,
        "epochs": "{:.1f}".format(last / spe) if last is not None and spe else DASH,
        "ended": ended,
        "real_train": DASH if tt is None else ("yes" if tt else "no"),
        "beats": beats_label(model, cfg.get("data", {}).get("beats") or {}),
        "seed": args.get("seed", 0),
    }


def beats_label(model, rec):
    """How a run was given beats: - (none), tracker errors (simulated on the ground truth, the
    tracker's own beats on the real transcriptions), or exact (the ground truth; "exact, all" when
    the real transcriptions got the ground truth too)."""
    if not model.get("beat_features"):
        return DASH
    if rec.get("noise") == "none":
        return "exact, all" if rec.get("train_transcribed") == "gt" else "exact"
    return "tracker errors"


def find_runs(runs_dir, only, exclude):
    out = []
    for f in sorted(Path(runs_dir).glob("*/eval_pieces_best.json")):
        name = f.parent.name
        if only and name not in only:
            continue
        if any(fnmatch.fnmatch(name, pat) for pat in exclude):
            continue
        out.append(name)
    return out


def meter_info(d):
    """Settings of one meter-model run (missing files -> dashes)."""
    cfg = load_json(d / "config.json")
    model, data = cfg.get("model", {}), cfg.get("data", {})
    by = data.get("train_by_source")
    if by:
        src = "+".join(label for key, label in (("asap", "A"), ("real", "R"), ("periscope", "P")) if by.get(key))
    else:  # the first meter model recorded only the real-transcription count
        src = "A+R" if data.get("real_transcriptions") else ("A" if cfg else DASH)
    status = load_json(d / "status.json")
    return {"params": "{:.1f}M".format(cfg["params"] / 1e6) if cfg.get("params") else DASH,
            "notes": DASH if not cfg else ("yes" if model.get("n_note_features") else "no"),
            "data": src, "seed": cfg.get("args", {}).get("seed", DASH) if cfg else DASH,
            "ended": DASH if not status else ("done" if status.get("phase") == "done" else "training")}


def meter_section(meters):
    """The meter models' tables: {run: (eval_meter.json, meter_info)}."""
    if not meters:
        return []
    out = ["## Meter models (v2) per test set", "", METER_HELP + ".", ""]
    for cond, desc in METER_CONDITIONS:
        rows = []
        for r in sorted(meters):
            v = meters[r][0].get(cond)
            if not isinstance(v, dict):
                continue
            mi, dflt, pm = meters[r][1], v.get("default", {}), v.get("piece_mean", {})
            wrong = [c for c in v.get("ts_confusion", []) if c[0] != c[1]][:2]
            rows.append([r, mi["data"], mi["notes"], mi["params"], mi["seed"],
                         num(v.get("grid_beat_f")), num(v.get("grid_downbeat_f")), num(v.get("grid_ts_acc")),
                         num(pm.get("grid_downbeat_f")), num(pm.get("grid_ts_acc")),
                         num(dflt.get("grid_downbeat_f")), num(dflt.get("grid_ts_acc")),
                         num(v.get("tracker_beat_f")), num(v.get("tracker_downbeat_f")),
                         str(v.get("performances", DASH)),
                         ", ".join("{}->{} {}".format(*c) for c in wrong) or DASH])
        if rows:
            hdr = ["run", "data", "note encoder", "params", "seed", "beat F", "downbeat F", "time sig",
                   "downbeat F per piece", "time sig per piece", "downbeat F default", "time sig default",
                   "tracker beat F", "tracker downbeat F", "n", "most bars wrong"]
            out += ["### {}: {}".format(cond, desc), "",
                    table(hdr, rows, {k: "max" for k in range(5, 12)}), ""]
    return out


def find_meters(runs_dir, only, exclude):
    out = {}
    for f in sorted(Path(runs_dir).glob("*/eval_meter.json")):
        name = f.parent.name
        if (only and name not in only) or any(fnmatch.fnmatch(name, pat) for pat in exclude):
            continue
        out[name] = (load_json(f), meter_info(f.parent))
    return out


def summary(runs, res, info, now, meters=None):
    out = ["# Run comparison", "",
           "Generated {} from `outputs/runs/*/eval_pieces_best.json` (each run's `best.pt`). Higher is "
           "better, except validation loss. **Bold** = best in its column; {} = not evaluated."
           .format(now, DASH), "",
           "## Runs", ""]
    hdr = ["run", "model", "params", "context notes", "aux head", "beats", "dropout", "wd", "lr", "seed",
           "best val loss", "epochs", "ended", "trained on real transcriptions"]
    rows = [[r, info[r]["model"], info[r]["params"], info[r]["context"], info[r]["aux"], info[r]["beats"],
             info[r]["dropout"], info[r]["wd"], info[r]["lr"], info[r]["seed"], num(info[r]["best_val"], digits=4),
             info[r]["epochs"], info[r]["ended"], info[r]["real_train"]] for r in runs]
    out += [table(hdr, rows, {10: "min"}), ""]

    out += ["## System output ({} mode) per test set".format(SYSTEM), "",
            "The decoding the demo runs. fallback = share of windows where the no-context output was kept; "
            "ts fixed = share of windows whose time signature meter mode corrected.", ""]
    for cond, desc in CONDITIONS:
        have = [r for r in runs if SYSTEM in res[r].get(cond, {})]
        if not have:
            continue
        hdr = ["run"] + [m[1] for m in METRICS] + ["fallback", "ts fixed"]
        rows = []
        for r in have:
            v = res[r][cond][SYSTEM]
            rows.append([r] + [num(v.get(k)) for k, _ in METRICS]
                        + [num(v.get("fallback_rate"), True), num(v.get("meter_fix_rate"), True)])
        out += ["### {}: {}".format(cond, desc), "",
                table(hdr, rows, {k + 1: "max" for k in range(len(METRICS))}), ""]

    out += ["## Context modes per test set: onset F1 and bar F1", "", MODE_HELP + ".", ""]
    for cond, desc in CONDITIONS:
        have = [r for r in runs if res[r].get(cond)]
        if not have:
            continue
        hdr = ["run"] + ["onset {}".format(m) for m in MODES] + ["bar {}".format(m) for m in MODES]
        rows = [[r] + [num(res[r][cond].get(m, {}).get("onset_f1")) for m in MODES]
                + [num(res[r][cond].get(m, {}).get("bar_f1")) for m in MODES] for r in have]
        out += ["### {}: {}".format(cond, desc), "",
                table(hdr, rows, {k + 1: "max" for k in range(2 * len(MODES))}), ""]

    out += ["## Note labeller (v2) per test set", "",
            "One pass labels every performed note (bar, position, written length, staff, flags, or delete) and "
            "the notes are rendered back to tokens (src/labeller.py, scripts/eval_labeller.py). true grid = given "
            "the score's beats timed through the note alignment (like the beat-snap baselines given the annotated "
            "beats); meter model <run> = the grid that meter model decoded from the tracked beats, with its bars, "
            "time signatures and an estimated key signature: nothing from the score (the real system). The "
            "seq2seq models' meter mode is the system they replace.", ""]
    for cond, desc in CONDITIONS:
        rows = []
        for r in runs:
            modes = [(m, dict(LABELLER_MODES).get(m, "meter model " + m[len("grid_meter_"):]))
                     for m in res[r].get(cond, {}) if m.startswith("grid_")]
            for mode, label in sorted(modes, key=lambda x: (x[0] != "grid_true", x[0])):
                v = res[r][cond][mode]
                rows.append([r, label] + [num(v.get(k)) for k, _ in METRICS])
        if rows:
            out += ["### {}: {}".format(cond, desc), "",
                    table(["run", "grid"] + [m[1] for m in METRICS], rows,
                          {k + 2: "max" for k in range(len(METRICS))}), ""]

    out += ["## Time-signature search per test set", "", TS_HELP + ".", ""]
    keys = [("onset_f1", "onset"), ("bar_f1", "bar"), ("ts_acc", "ts")]
    for cond, desc in CONDITIONS:
        have = [r for r in runs if any(m in res[r].get(cond, {}) for m in TS_MODES[1:])]
        if not have:
            continue
        hdr = ["run"] + ["{} {}".format(label, m) for _, label in keys for m in TS_MODES] + ["changed"]
        rows = [[r] + [num(res[r][cond].get(m, {}).get(k)) for k, _ in keys for m in TS_MODES]
                + [num(res[r][cond].get("tsearch", {}).get("ts_switch_rate"), True)] for r in have]
        out += ["### {}: {}".format(cond, desc), "",
                table(hdr, rows, {k + 1: "max" for k in range(len(keys) * len(TS_MODES))}), ""]
    out += meter_section(meters or {})
    return "\n".join(out)


def margin_of(run_res, name, v):
    """A guarded / meter result's guard margin: recorded in the result, else in the mode name
    (guarded@-0.05), else the margin the same mode recorded on another test set (real_unseen_audio
    results written before src/eval/plan.py carry none: they come from the same decoding as real)."""
    if "guard_margin" in v:
        return v["guard_margin"]
    if "@" in name:
        try:
            return float(name.split("@", 1)[1])
        except ValueError:
            return None
    for modes in run_res.values():
        other = modes.get(name)
        if isinstance(other, dict) and "guard_margin" in other:
            return other["guard_margin"]
    return None


def margins(runs, res, now):
    swept = [r for r in runs if any("@" in m for c in res[r].values() for m in c)]
    out = ["# Guard-margin sweeps", "",
           "Generated {}. guarded keeps the no-context output of a window when its pitch agreement with the "
           "input notes beats the chained output's by more than the margin: a negative margin falls back "
           "more often. Default: the `guarded` / `meter` modes (margin in the first column). **Bold** = best "
           "in its column for that run and test set.".format(now), ""]
    if not swept:
        return "\n".join(out + ["No run has been evaluated at more than one margin yet "
                                "(`scripts/evaluate_pieces.py --extra-margins`)."])
    keys = [("note_f1", "note F1"), ("onset_f1", "onset F1"), ("onset_shift_f1", "onset-shift F1"),
            ("bar_f1", "bar F1"), ("ts_acc", "time sig")]
    for r in swept:
        out += ["## {}".format(r), ""]
        for cond, desc in CONDITIONS:
            modes = res[r].get(cond, {})
            rows = []
            for kind in ("guarded", "meter", "piece"):
                entries = [(margin_of(res[r], name, v), name, v) for name, v in modes.items()
                           if name.split("@")[0] == kind and isinstance(v, dict)]
                for m, name, v in sorted(entries, key=lambda e: (e[0] is None, e[0])):
                    label = name if m is None else "{} {}".format(
                        kind, "0" if m == 0 else "{:+g}".format(m))
                    if name == kind:
                        label += " (default)"
                    rows.append([label,
                                 num(v.get("fallback_rate"), True), num(v.get("meter_fix_rate"), True)]
                                + [num(v.get(k)) for k, _ in keys])
            if len(rows) < 2:
                continue
            for ref in ("chained", "none"):  # the two ends of the guard: never / always fall back
                if ref in modes:
                    rows.append([ref, DASH, DASH] + [num(modes[ref].get(k)) for k, _ in keys])
            hdr = ["mode, margin", "fallback", "ts fixed"] + [k[1] for k in keys]
            out += ["### {}: {}".format(cond, desc), "",
                    table(hdr, rows, {k + 3: "max" for k in range(len(keys))}), ""]
    return "\n".join(out)


def write_csv(path, runs, res):
    keys = []
    for r in runs:
        for c in res[r].values():
            for v in c.values():
                for k, x in v.items():
                    if isinstance(x, (int, float)) and k not in keys:
                        keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "condition", "mode"] + keys)
        for r in runs:
            for cond, modes in res[r].items():
                for mode, v in modes.items():
                    w.writerow([r, cond, mode] + [v.get(k, "") for k in keys])


def main(argv=None):
    ap = argparse.ArgumentParser(description="tables comparing every evaluated run")
    ap.add_argument("--runs", default="", help="comma-separated run names (default: every evaluated run)")
    ap.add_argument("--exclude", default="smoke*,_*", help="name patterns to leave out")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--out", default="outputs/compare")
    ap.add_argument("--quiet", action="store_true", help="do not print summary.md")
    args = ap.parse_args(argv)
    only = [r for r in args.runs.split(",") if r]
    exclude = [p for p in args.exclude.split(",") if p]
    runs = find_runs(args.runs_dir, only, exclude)
    meters = find_meters(args.runs_dir, only, exclude)
    if not runs and not meters:
        print("no evaluated runs in {} (looked for */eval_pieces_best.json and */eval_meter.json)".format(args.runs_dir))
        return 1
    res = {r: load_json(Path(args.runs_dir) / r / "eval_pieces_best.json") for r in runs}
    info = {r: run_info(Path(args.runs_dir) / r) for r in runs}
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = summary(runs, res, info, now, meters)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    (out / "margins.md").write_text(margins(runs, res, now) + "\n", encoding="utf-8")
    write_csv(out / "results.csv", runs, res)
    if not args.quiet:
        print(text)
        print()
    print("wrote {0}/summary.md, {0}/margins.md, {0}/results.csv ({1} runs, {2} meter models)".format(
        out, len(runs), len(meters)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
