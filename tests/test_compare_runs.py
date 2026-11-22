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
    # beat runs (low scores: they never take a bold value from the runs above) with the
    # time-signature search modes on clean and real
    ts = full(0.05, ("clean", "real"))
    for c in ts:
        ts[c]["ts_true"] = metrics(0.4, fallback_rate=0.4, ts_switch_rate=0.3, guard_margin=0.0)
        ts[c]["tsearch"] = metrics(0.2, fallback_rate=0.4, ts_switch_rate=0.25, guard_margin=0.0)
        ts[c]["tsearch_beat"] = metrics(0.21, fallback_rate=0.4, ts_switch_rate=0.27, guard_margin=0.0)
    beat_cfg = cfg(44.6e6, arch="scoreformer", cb=64, ca=32, aux=True)
    beat_cfg["model"]["beat_features"] = True
    beat_cfg["data"]["beats"] = {"noise": "configs/beat_noise.json", "train_transcribed": "tracked"}
    make_run(runs, "sf_cab", ts, beat_cfg, [(0.31, 1180)], {"phase": "done", "stopped_early": True})
    exact = json.loads(json.dumps(beat_cfg))
    exact["data"]["beats"] = {"noise": "none", "train_transcribed": "gt"}
    exact["args"]["seed"] = 1
    make_run(runs, "sf_cabx", full(0.04, ("clean",)), exact, [(0.32, 1100)], {"phase": "done", "stopped_early": True})
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
        # run, model, params, context, aux, beats, dropout, wd, lr, seed, val loss, epochs, ended, real
        assert [c.startswith(":") for c in hdr_sep] == [True, True, False, True, True, True, False, False, False,
                                                        False, False, False, True, True], hdr_sep
        beats = {r: [c.strip() for c in line[r].split("|")][6] for r in ("s2s", "s2s_ca")}
        assert beats == {"s2s": compare_runs.DASH, "s2s_ca": compare_runs.DASH}, beats
        cab = next(ln for ln in runs_tbl.splitlines() if ln.startswith("| sf_cab "))
        cabx = next(ln for ln in runs_tbl.splitlines() if ln.startswith("| sf_cabx "))
        assert "tracker errors" in cab and "exact, all" in cabx and [c.strip() for c in cabx.split("|")][10] == "1"
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

        # time-signature search: the runs with those modes, per test set; meter alongside
        tsec = s.split("## Time-signature search per test set")[1]
        clean_ts = tsec.split("### clean: ")[1].split("###")[0]
        assert "| sf_cab " in clean_ts and "| s2s " not in clean_ts and "sf_cabx" not in clean_ts
        row = next(ln for ln in clean_ts.splitlines() if ln.startswith("| sf_cab "))
        cells = [c.strip().strip("*") for c in row.split("|")[1:-1]]
        # onset: meter, ts_true, tsearch, tsearch_beat; then bar, ts; then the share tsearch changed
        assert cells[1:5] == ["0.180", "0.410", "0.210", "0.220"] and cells[-1] == "25%", cells
        assert "### real: " in tsec and "### synthetic: " not in tsec

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
        assert len(rows) == 5 * 9 + 3 * 5 + 2 + 5 + 2 * 8 + 5, len(rows)  # s2s sweep, s2s_ca, sf_old, sf_busy, sf_cab, sf_cabx
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


def test_meter_tables():
    """Meter models (outputs/runs/<run>/eval_meter.json): one table per test set, their data and
    note encoder from config.json (the first meter model's older config too), the tuned and the
    default decoder, the per-piece means, the most frequent wrong time signatures; a meter model
    alone (no seq2seq run) is enough for a summary."""
    def cond(base, n=59):
        return {"grid_beat_f": base + 0.2, "grid_downbeat_f": base + 0.1, "grid_ts_acc": base,
                "tracker_beat_f": 0.81, "tracker_downbeat_f": 0.62, "performances": n, "pieces": 14,
                "default": {"grid_downbeat_f": base + 0.05, "grid_ts_acc": base - 0.05},
                "piece_mean": {"grid_downbeat_f": base + 0.08, "grid_ts_acc": base + 0.02},
                "ts_confusion": [["4/4", "4/4", 900], ["6/4", "3/4", 300], ["2/4", "4/4", 120]]}
    with tempfile.TemporaryDirectory() as t:
        runs, out = Path(t) / "runs", Path(t) / "compare"
        for name, res, config in (
                ("meter_base", {"clean": cond(0.39), "real": cond(0.32, 24)},
                 {"params": 4.75e6, "model": {}, "data": {"train": 841, "real_transcriptions": 85}}),
                ("meter_v2", {"clean": cond(0.6), "real": cond(0.5, 24), "periscope": cond(0.7, 480)},
                 {"params": 5.1e6, "args": {"seed": 0}, "model": {"n_note_features": 20},
                  "data": {"train_by_source": {"asap": 756, "real": 85, "periscope": 6000}}}),
                ("meter_v2_asap", {"clean": cond(0.5)},
                 {"params": 5.1e6, "args": {"seed": 0}, "model": {"n_note_features": 20},
                  "data": {"train_by_source": {"asap": 756, "real": 85, "periscope": 0}}})):
            d = runs / name
            d.mkdir(parents=True)
            (d / "eval_meter.json").write_text(json.dumps(res))
            (d / "config.json").write_text(json.dumps(config))
            (d / "status.json").write_text(json.dumps({"phase": "done"}))
        assert compare_runs.main(["--runs-dir", str(runs), "--out", str(out), "--quiet"]) == 0
        s = (out / "summary.md").read_text(encoding="utf-8")
        sec = s.split("## Meter models (v2) per test set")[1]
        clean = sec.split("### clean: ")[1].split("###")[0]
        rows = {ln.split("|")[1].strip(): [c.strip() for c in ln.split("|")[1:-1]]
                for ln in clean.splitlines() if ln.startswith("| meter")}
        assert set(rows) == {"meter_base", "meter_v2", "meter_v2_asap"}
        assert rows["meter_base"][1:4] == ["A+R", "no", "4.8M"] and rows["meter_v2"][1:3] == ["A+R+P", "yes"]
        assert rows["meter_v2_asap"][1] == "A+R"
        assert rows["meter_v2"][7] == "**0.600**" and rows["meter_base"][7] == "0.390"   # tuned time sig
        assert rows["meter_v2"][11] == "**0.550**"                                       # default decoder
        assert rows["meter_v2"][-1] == "6/4->3/4 300, 2/4->4/4 120"
        periscope = sec.split("### periscope: ")[1]
        assert "| meter_v2 " in periscope and "meter_base" not in periscope
        for block in sec.split("\n\n"):
            lines = [ln for ln in block.splitlines() if ln.startswith("|")]
            assert len({len(ln) for ln in lines}) <= 1, block


def test_labeller_tables():
    """The note labeller's table: per run, the true grid first, then the oracle grid, then each
    meter model's grid by name; the onset-scale F1 column (a dash where an older result lacks it);
    the oracle as a meter-model row with no model settings."""
    with tempfile.TemporaryDirectory() as t:
        runs, out = Path(t) / "runs", Path(t) / "compare"
        res = {"clean": {"grid_true": metrics(0.7, onset_scaled_f1=0.8),
                         "grid_meter_meter_v3": metrics(0.3, onset_scaled_f1=0.45),
                         "grid_meter_meter_oracle": metrics(0.5, onset_scaled_f1=0.6),
                         "grid_meter_meter_base": metrics(0.2)}}
        make_run(runs, "nl_base_gn", res, {"arch": "labeller", "params": 25.6e6, "args": {}, "model": {}})
        d = runs / "meter_oracle"
        d.mkdir()
        (d / "eval_meter.json").write_text(json.dumps({"clean": {"grid_beat_f": 0.88, "grid_downbeat_f": 0.9,
                                                                 "grid_ts_acc": 0.99, "performances": 59}}))
        (d / "config.json").write_text(json.dumps({"arch": "meter_oracle"}))
        assert compare_runs.main(["--runs-dir", str(runs), "--out", str(out), "--quiet"]) == 0
        s = (out / "summary.md").read_text(encoding="utf-8")
        sec = s.split("## Note labeller (v2) per test set")[1].split("\n## ")[0]
        hdr = next(ln for ln in sec.splitlines() if ln.startswith("| run"))
        assert "onset-scale F1" in hdr and "oracle grid" in sec and "*_gn" in sec
        rows = [[c.strip() for c in ln.split("|")[1:-1]] for ln in sec.splitlines() if ln.startswith("| nl_base_gn")]
        assert [r[1] for r in rows] == ["true grid", "oracle grid (tracked beats, every meter head right)",
                                        "meter model meter_base", "meter model meter_v3"], [r[1] for r in rows]
        col = [c.strip() for c in hdr.split("|")[1:-1]].index("onset-scale F1")
        assert [r[col] for r in rows] == ["**0.800**", "0.600", compare_runs.DASH, "0.450"], [r[col] for r in rows]
        meter = s.split("## Meter models (v2) per test set")[1]
        orow = next([c.strip() for c in ln.split("|")[1:-1]] for ln in meter.splitlines() if ln.startswith("| meter_oracle"))
        assert orow[1:5] == ["oracle", compare_runs.DASH, compare_runs.DASH, compare_runs.DASH], orow


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
