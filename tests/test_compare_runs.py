"""Tests for scripts/compare_runs.py (the comparison tables).

    python -m tests.test_compare_runs        # or: pytest tests/test_compare_runs.py

Builds stand-in runs shaped like the real ones (a full evaluation, a margin sweep, a run without the
real condition, an old run without the newer config fields, a run still training) and checks every
table, the bold best values, the dashes for what was not evaluated, and the CSV.
"""
from __future__ import annotations

import ast
import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import compare_runs  # noqa: E402

KEYS = ["note_f1", "onset_f1", "onset_shift_f1", "duration_acc", "staff_acc", "bar_f1", "ts_acc"]


def metrics(base, **extra):
    return dict({k: base + 0.01 * i for i, k in enumerate(KEYS)}, **extra)


def make_run(runs, name, results, config=None, log=None, status=None):
    d = runs / name
    d.mkdir(parents=True)
    (d / "eval_pieces_best.json").write_text(json.dumps(results))
    if config is not None:
        (d / "config.json").write_text(json.dumps(config))
    if log is not None:
        with open(d / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["val_loss", "step", "epoch"])
            w.writeheader()
            for loss, step in log:
                w.writerow({"val_loss": loss, "step": step, "epoch": 0})
    if status is not None:
        (d / "status.json").write_text(json.dumps(status))


def full(base, conds=("clean", "synthetic", "periscope", "real", "real_unseen_audio")):
    out = {}
    for c in conds:
        out[c] = {"oracle": metrics(base + 0.3), "chained": metrics(base), "none": metrics(base + 0.05),
                  "guarded": metrics(base + 0.1, fallback_rate=0.45, guard_margin=0.0),
                  "meter": metrics(base + 0.12, fallback_rate=0.45, meter_fix_rate=0.08, guard_margin=0.0)}
    return out


def cfg(params, arch="score2seq", cb=0, ca=0, aux=False, tt=True, **args):
    return {"arch": arch, "params": params, "steps_per_epoch": 100, "total_steps": 2000,
            "args": dict({"dropout": 0.1, "weight_decay": 0.01, "lr": 3e-4, "size": "base"}, **args),
            "model": {"context_before": cb, "context_after": ca, "aux_metrical": aux},
            "data": {"train_transcribed": tt}}


def build(runs):
    sweep = full(0.2)
    for c in sweep:
        for m in (-0.05, 0.05):
            sweep[c]["guarded@{:g}".format(m)] = metrics(0.3 + m, fallback_rate=0.5 - m, guard_margin=m)
            sweep[c]["meter@{:g}".format(m)] = metrics(0.32 + m, fallback_rate=0.5 - m, meter_fix_rate=0.05,
                                                       guard_margin=m)
    for v in sweep["real_unseen_audio"].values():  # the evaluator wrote no margin there before
        v.pop("guard_margin", None)
    make_run(runs, "s2s", sweep, cfg(10e6), [(0.30, 1000), (0.2781, 1500), (0.29, 2000)],
             {"phase": "done", "stopped_early": False})
    make_run(runs, "s2s_ca", full(0.25, ("clean", "synthetic", "periscope")), cfg(10.2e6, cb=64, ca=32, aux=True),
             [(0.2779, 1560)], {"phase": "done", "stopped_early": True})
    make_run(runs, "sf_old", {"clean": {"oracle": metrics(0.5), "none": metrics(0.1)}},
             {"arch": "scoreformer", "params": 44.6e6, "args": {"size": "base"}, "model": {}})  # no data / log
    make_run(runs, "sf_busy", full(0.1, ("clean",)), cfg(44.6e6, arch="scoreformer"), [(0.3, 300)],
             {"phase": "cooldown"})
    make_run(runs, "smoke", full(0.9))     # excluded by default
    make_run(runs, "_test_x", full(0.9))   # excluded by default


def test_tables():
    with tempfile.TemporaryDirectory() as t:
        runs, out = Path(t) / "runs", Path(t) / "compare"
        build(runs)
        assert compare_runs.main(["--runs-dir", str(runs), "--out", str(out), "--quiet"]) == 0
        s = (out / "summary.md").read_text(encoding="utf-8")
        m = (out / "margins.md").read_text(encoding="utf-8")
        assert "smoke" not in s and "_test_x" not in s

        runs_tbl = s.split("## Runs")[1].split("## System")[0]
        line = {r: next(ln for ln in runs_tbl.splitlines() if ln.startswith("| " + r + " ")) for r in
                ("s2s", "s2s_ca", "sf_old", "sf_busy")}
        assert "**0.2779**" in line["s2s_ca"] and "0.2781" in line["s2s"] and "**" not in line["s2s"]
        assert "64/32" in line["s2s_ca"] and "yes" in line["s2s_ca"] and "early stop" in line["s2s_ca"]
        assert "full" in line["s2s"] and "20.0" in line["s2s"] and "15.6" in line["s2s_ca"]
        assert "Scoreformer base" in line["sf_old"] and "44.6M" in line["sf_old"]
        hdr_sep = runs_tbl.strip().splitlines()[1].split("|")[1:-1]  # alignment row: text left, numbers right
        assert [c.startswith(":") for c in hdr_sep] == [True, True, False, True, True, False, False, False,
                                                        False, False, True, True], hdr_sep
        assert "training (cooldown)" in line["sf_busy"]
        assert line["sf_old"].count(compare_runs.DASH) >= 5  # no data record, no log, no status

        # system tables: one per test set with meter results; real has only s2s (s2s_ca lacks it)
        real = s.split("### real: ")[1].split("###")[0]
        assert "| s2s " in real and "s2s_ca" not in real
        clean = s.split("### clean: ")[1].split("###")[0]
        assert "sf_old" not in clean.split("\n\n")[1]  # no meter result -> not in the system table
        s2s_row = next(ln for ln in clean.splitlines() if ln.startswith("| s2s "))
        assert "45%" in s2s_row and "8%" in s2s_row
        best = next(ln for ln in clean.splitlines() if ln.startswith("| s2s_ca "))
        assert best.count("**") == 2 * len(KEYS)  # highest in every metric column

        # context-mode tables include runs without meter, with dashes for the missing modes
        modes = s.split("## Context modes")[1]
        old = next(ln for ln in modes.splitlines() if ln.startswith("| sf_old "))
        assert old.count(compare_runs.DASH) == 8 and "0.510" in old and "**" not in old  # 4 modes x 2 metrics
        ca = next(ln for ln in modes.splitlines() if ln.startswith("| s2s_ca "))
        assert "**0.560**" in ca  # the best oracle onset F1 on clean

        # margin sweeps: only s2s, rows ordered by margin, chained / none as the two references
        assert "## s2s" in m and "s2s_ca" not in m
        rows = [ln.split("|")[1].strip() for ln in m.split("### clean: ")[1].split("###")[0].splitlines()
                if ln.startswith("| ") and not ln.startswith("| mode")]
        assert rows == ["guarded -0.05", "guarded 0 (default)", "guarded +0.05", "meter -0.05",
                        "meter 0 (default)", "meter +0.05", "chained", "none"], rows
        assert "|:-------" in m.split("### clean: ")[1]  # the text column is left-aligned
        # real_unseen_audio as written before margins were recorded there: read from the mode names
        unseen = [ln.split("|")[1].strip() for ln in m.split("### real_unseen_audio: ")[1].splitlines()
                  if ln.startswith("| ") and not ln.startswith("| mode")]
        assert unseen == rows, unseen

        # every line of every table has the same width (aligned as plain text)
        for text in (s, m):
            for block in text.split("\n\n"):
                lines = [ln for ln in block.splitlines() if ln.startswith("|")]
                assert len({len(ln) for ln in lines}) <= 1, block

        with open(out / "results.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5 * 9 + 3 * 5 + 2 + 5, len(rows)  # s2s sweep, s2s_ca, sf_old, sf_busy
        r = next(x for x in rows if x["run"] == "s2s" and x["condition"] == "real" and x["mode"] == "meter@-0.05")
        assert float(r["guard_margin"]) == -0.05 and float(r["meter_fix_rate"]) == 0.05


def test_no_runs_and_selection():
    with tempfile.TemporaryDirectory() as t:
        runs, out = Path(t) / "runs", Path(t) / "compare"
        runs.mkdir()
        assert compare_runs.main(["--runs-dir", str(runs), "--out", str(out), "--quiet"]) == 1
        build(runs)
        assert compare_runs.main(["--runs-dir", str(runs), "--out", str(out), "--quiet", "--runs", "s2s_ca"]) == 0
        s = (out / "summary.md").read_text(encoding="utf-8")
        assert "| s2s_ca " in s and "| s2s " not in s
        assert "No run has been evaluated at more than one margin" in (out / "margins.md").read_text(encoding="utf-8")


def test_login_node_python():
    """Runs with the login node's system python3: standard library only, Python 3.6 syntax."""
    src = (ROOT / "scripts" / "compare_runs.py").read_text(encoding="utf-8")
    ast.parse(src, feature_version=(3, 6))
    stdlib = set(sys.stdlib_module_names)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                assert n.split(".")[0] in stdlib, n


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
