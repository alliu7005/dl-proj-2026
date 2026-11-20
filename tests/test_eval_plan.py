"""Tests for src/eval/plan.py and ``scripts/evaluate_pieces.py --only-missing``.

    python -m tests.test_eval_plan        # or: pytest tests/test_eval_plan.py

The plan decides what pace/reevaluate.sh submits and what --only-missing computes, so a mistake
either wastes GPU jobs or, worse, keeps a stale result: every rule is checked here. The last test
runs the real evaluator end to end (CPU, one test performance) with a stand-in checkpoint that ends
every window at once, so it needs data/processed/test but takes seconds.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval import plan  # noqa: E402
from src.eval.plan import (DEFAULT_GUARD_MARGIN, EVAL_VERSION, decode_passes, missing, mode_names,  # noqa: E402
                           parse_args, stamp)

ALL = plan.DEFAULT_MODES.split(",")


def full_results(names, margins, version=EVAL_VERSION, limit=0, conds=("clean", "real")):
    out = {}
    for c in list(conds) + (["real_unseen_audio"] if "real" in conds else []):
        out[c] = {n: stamp({"note_f1": 0.5}, margins.get(n), limit) for n in names}
        if version != EVAL_VERSION:
            for r in out[c].values():
                r["eval_version"] = version
    return out


def test_mode_names_and_passes():
    assert decode_passes(ALL, 0.0, []) == [(k, k, 0.0) for k in ("chained", "guarded", "meter", "piece", "ts_true",
                                                                 "tsearch", "tsearch_beat")]
    passes = decode_passes(ALL, 0.0, [-0.05, 0.0, 0.05])  # an extra equal to the main margin is dropped
    assert [p[0] for p in passes] == ["chained"] + [n for k in ("guarded", "meter", "piece", "ts_true", "tsearch",
                                                                "tsearch_beat")
                                                    for n in (k, k + "@-0.05", k + "@0.05")]
    assert dict((n, m) for n, _, m in passes)["meter@-0.05"] == -0.05
    assert mode_names(["none", "guarded"], 0.0, [0.1]) == ["none", "guarded", "guarded@0.1"]
    assert mode_names(["oracle", "chained"], 0.0, [0.1]) == ["oracle", "chained"]  # chained has no margin
    # the same names as the evaluator writes, and as were written before this module existed
    assert DEFAULT_GUARD_MARGIN == 0.0


def test_missing_rules():
    names = mode_names(ALL, 0.0, [])
    margins = plan.margins_of(ALL, 0.0, [])
    assert set(margins) == {"guarded", "meter", "piece", "ts_true", "tsearch", "tsearch_beat"}
    yes = lambda c: True  # noqa: E731
    conds = ["clean", "real", "periscope"]
    res = full_results(names, margins)
    assert missing(res, conds, names, margins, yes) == {"periscope": names}       # a condition never run
    assert missing(res, conds, names, margins, lambda c: c != "periscope") == {}   # its data does not exist
    # results from before eval_version / limit_perfs were recorded: version 1, all performances
    # exactly as the evaluator wrote them then: the unseen-audio subset had no guard_margin either
    legacy = json.loads(json.dumps(res))
    for cname, c in legacy.items():
        for r in c.values():
            del r["eval_version"], r["limit_perfs"]
            if cname == "real_unseen_audio":
                r.pop("guard_margin", None)
    assert missing(legacy, ["clean", "real"], names, margins, yes) == {}
    # a margin sweep's file (guarded / meter at several margins, no oracle / none / chained)
    sweep = {c: {n: r for n, r in legacy[c].items() if n.split("@")[0] in ("guarded", "meter")}
             for c in legacy}
    assert missing(sweep, ["real"], names, margins, yes) == {"real": ["oracle", "none", "chained", "piece", "ts_true",
                                                                      "tsearch", "tsearch_beat"]}
    # a recorded unseen-audio margin is still compared
    bad = json.loads(json.dumps(res))
    bad["real_unseen_audio"]["meter"]["guard_margin"] = 0.05
    assert missing(bad, ["real"], names, margins, yes) == {"real": ["meter"]}
    # another guard margin, no recorded margin, a --limit-perfs subset, an older evaluation version
    other = json.loads(json.dumps(res))
    other["clean"]["guarded"]["guard_margin"] = 0.05
    del other["clean"]["meter"]["guard_margin"]
    assert missing(other, ["clean"], names, margins, yes) == {"clean": ["guarded", "meter"]}
    assert missing(full_results(names, margins, limit=10), ["clean"], names, margins, yes) == {"clean": names}
    assert missing(res, ["clean"], names, margins, yes, limit_perfs=10) == {"clean": names}
    old = full_results(names, margins, version=EVAL_VERSION - 1)
    assert missing(old, ["clean"], names, margins, yes) == {"clean": names}
    # real is complete only with its unseen-audio subset (written by the same decoding)
    part = json.loads(json.dumps(res))
    del part["real_unseen_audio"]["meter"]
    assert missing(part, ["real"], names, margins, yes) == {"real": ["meter"]}
    # extra margins: only the new modes are missing
    n2 = mode_names(ALL, 0.0, [-0.02])
    m2 = plan.margins_of(ALL, 0.0, [-0.02])
    assert missing(res, ["clean"], n2, m2, yes) == {"clean": [k + "@-0.02" for k in ("guarded", "meter", "piece",
                                                                                     "ts_true", "tsearch",
                                                                                     "tsearch_beat")]}


def test_arguments():
    a = parse_args(["--run", "s2s", "--extra-margins", "-0.05,-0.02", "--only-missing"])
    assert a.extra_margins == "-0.05,-0.02" and a.only_missing and a.guard_margin == DEFAULT_GUARD_MARGIN
    for bad in (["--run", "x", "--modes", "guardd"], ["--run", "x", "--conditions", "reel"]):
        try:
            parse_args(bad)
        except SystemExit as e:
            assert "unknown" in str(e), e
        else:
            raise AssertionError(f"{bad} accepted")


def run_plan(*argv):
    r = subprocess.run([sys.executable, "-m", "src.eval.plan", *argv], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def test_command_line():
    """What pace/reevaluate.sh runs on the login node: exit status 0 / 3 / 4 and one line per condition."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data, runs = tmp / "data", tmp / "runs"
        for split in ("test", "test_transcribed"):
            (data / split).mkdir(parents=True)
            (data / split / "index.npy").write_bytes(b"")
        run = runs / "r"
        run.mkdir(parents=True)
        dirs = ["--runs-dir", str(runs), "--data-dir", str(data)]
        code, out = run_plan("--run", "r", *dirs)  # no results yet: clean, synthetic and real (the
        assert code == plan.MISSING, out           # periscope and Kong splits do not exist here)
        assert out.splitlines() == [f"{c}: {','.join(mode_names(ALL, 0.0, []))}" for c in ("clean", "synthetic", "real")]
        names = mode_names(ALL, 0.0, [])
        res = full_results(names, plan.margins_of(ALL, 0.0, []), conds=("clean", "synthetic", "real"))
        (run / "eval_pieces_best.json").write_text(json.dumps(res))
        assert run_plan("--run", "r", *dirs) == (plan.NOTHING_MISSING, "nothing missing")
        code, out = run_plan("--run", "r", "--conditions", "real", "--extra-margins", "-0.05", *dirs)
        assert code == plan.MISSING and out == "real: " + ",".join(
            k + "@-0.05" for k in ("guarded", "meter", "piece", "ts_true", "tsearch", "tsearch_beat")), out
        (run / "status.json").write_text(json.dumps({"phase": "cooldown", "step": 5}))
        assert run_plan("--run", "r", *dirs) == (plan.TRAINING, "training not finished (phase cooldown)")
        (run / "status.json").write_text(json.dumps({"phase": "done", "step": 9}))
        assert run_plan("--run", "r", *dirs)[0] == plan.NOTHING_MISSING
        code, out = run_plan("--run", "r", "--modes", "bogus", *dirs)
        assert code != 0 and code not in (plan.NOTHING_MISSING, plan.TRAINING)


def test_beat_conditions():
    """Beat-only conditions apply to beat models only; a beat model waits for its beats."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data, runs = tmp / "data", tmp / "runs"
        for split in ("test", "test_transcribed"):
            (data / split).mkdir(parents=True)
            (data / split / "index.npy").write_bytes(b"")
        for name, beats in (("plain", False), ("beat", True)):
            (runs / name).mkdir(parents=True)
            (runs / name / "config.json").write_text(json.dumps({"model": {"beat_features": beats}}))
        noise = tmp / "beat_noise.json"
        dirs = ["--runs-dir", str(runs), "--data-dir", str(data), "--beat-noise", str(noise)]
        conds = lambda out: [ln.split(":")[0] for ln in out.splitlines()]  # noqa: E731
        assert plan.has_beats(runs / "beat") and not plan.has_beats(runs / "plain")
        # a model without beats: the beat-only conditions never apply
        assert conds(run_plan("--run", "plain", *dirs)[1]) == ["clean", "synthetic", "real"]
        # a beat model: nothing with beats until they exist; only "none" needs nothing
        assert conds(run_plan("--run", "beat", *dirs)[1]) == ["real_no_beats"]
        for split in ("test", "test_transcribed"):
            (data / split / "beats_gt.npz").write_bytes(b"")
        assert conds(run_plan("--run", "beat", *dirs)[1]) == ["clean_gt_beats", "real_gt_beats", "real_no_beats"]
        noise.write_text("{}")  # the fitted tracker noise: the simulated-tracker conditions
        assert conds(run_plan("--run", "beat", *dirs)[1]) == ["clean", "synthetic", "clean_gt_beats",
                                                              "real_gt_beats", "real_no_beats"]
        (data / "test_transcribed" / "beats_tracked.npz").write_bytes(b"")  # Beat This! on the audio: real
        assert conds(run_plan("--run", "beat", *dirs)[1]) == ["clean", "synthetic", "real", "clean_gt_beats",
                                                              "real_gt_beats", "real_no_beats"]
        assert plan.BEATS["real"] == "tracked" and plan.BEATS["clean"] == "sim" and set(plan.BEAT_ONLY) <= set(plan.BEATS)


def test_login_node_python():
    """src/eval/plan.py must run on the PACE login node's system python3: standard library only,
    and no syntax newer than Python 3.6 (checked by compiling it for 3.6)."""
    import ast
    src = (ROOT / "src" / "eval" / "plan.py").read_text(encoding="utf-8")
    ast.parse(src, feature_version=(3, 6))
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        else:
            continue
        for m in mods:
            assert m.split(".")[0] in stdlib, f"src/eval/plan.py imports {m}"


def make_quiet_checkpoint(path: Path):
    """A small Scoreformer that ends every window at once (its output is always <eos>)."""
    import torch
    from src.preprocessing.vocab import EOS, Vocab
    from src.scoreformer import PRESETS, build_model
    vocab = Vocab.load(str(ROOT / "data" / "processed" / "vocab.json"))
    cfg = dict(vocab_size=len(vocab), dropout=0.0, max_tokens=2048, max_notes=1024, **PRESETS["small"])
    model = build_model("scoreformer", cfg)
    with torch.no_grad():  # constant final features -> the <eos> logit always wins
        model.out_norm.weight.zero_()
        model.out_norm.bias.zero_()
        model.out_norm.bias[0] = 1.0
        model.head.weight.zero_()
        model.head.weight[vocab.stoi[EOS], 0] = 10.0
    torch.save({"model": model.state_dict(), "arch": "scoreformer", "config": cfg}, path)


def evaluate(runs, *argv):
    cmd = [sys.executable, "scripts/evaluate_pieces.py", "--run", "q", "--runs-dir", str(runs), "--device", "cpu",
           "--normal-priority", "--conditions", "clean,real_kong", "--limit-perfs", "1", *argv]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


def test_only_missing_end_to_end():
    if not (ROOT / "data" / "processed" / "test" / "index.npy").exists():
        print("  (skipped: data/processed/test missing)")
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "q").mkdir()
        make_quiet_checkpoint(tmp / "q" / "best.pt")
        evaluate(tmp, "--modes", "none,guarded")
        res = json.loads((tmp / "q" / "eval_pieces_best.json").read_text())
        assert set(res) == {"clean"} and set(res["clean"]) == {"none", "guarded"}, res.keys()
        g = res["clean"]["guarded"]
        assert g["eval_version"] == EVAL_VERSION and g["limit_perfs"] == 1 and g["guard_margin"] == 0.0
        assert "guard_margin" not in res["clean"]["none"] and res["clean"]["none"]["limit_perfs"] == 1
        before = {(json.loads(x)["mode"], json.loads(x)["win"]) for x in open(tmp / "q" / "pred_pieces_best_clean.jsonl")}

        out = evaluate(tmp, "--modes", "none,guarded,meter", "--extra-margins", "0.05", "--only-missing")
        assert "to compute: clean guarded@0.05,meter,meter@0.05" in out, out
        assert "clean: none done" not in out and "clean: guarded done" not in out, out  # kept, not re-run
        res = json.loads((tmp / "q" / "eval_pieces_best.json").read_text())
        assert set(res["clean"]) == {"none", "guarded", "guarded@0.05", "meter", "meter@0.05"}
        assert res["clean"]["meter@0.05"]["guard_margin"] == 0.05 and "meter_fix_rate" in res["clean"]["meter"]
        after = {(json.loads(x)["mode"], json.loads(x)["win"]) for x in open(tmp / "q" / "pred_pieces_best_clean.jsonl")}
        assert before < after and {m for m, _ in after} == set(res["clean"])  # earlier predictions kept

        out = evaluate(tmp, "--modes", "none,guarded,meter", "--extra-margins", "0.05", "--only-missing")
        assert "to compute: nothing" in out and "done" not in out, out
        # the time-signature modes, on their own (guarded's first pass is decoded again for them)
        out = evaluate(tmp, "--modes", "ts_true,tsearch_beat,tsearch", "--only-missing")
        assert "to compute: clean ts_true,tsearch,tsearch_beat" in out, out
        # their guarded first pass is not decoded again: the saved guarded / none predictions are reused
        assert "clean: reusing the saved guarded and none predictions" in out, out
        res = json.loads((tmp / "q" / "eval_pieces_best.json").read_text())
        for m in ("ts_true", "tsearch", "tsearch_beat"):
            r = res["clean"][m]
            assert r["guard_margin"] == 0.0 and 0.0 <= r["ts_switch_rate"] <= 1.0 and "fallback_rate" in r, (m, r)
        modes = {json.loads(x)["mode"] for x in open(tmp / "q" / "pred_pieces_best_clean.jsonl")}
        assert {"ts_true", "tsearch", "tsearch_beat", "meter"} <= modes
        # results are merged per condition under a lock: a second writer adds, never overwrites
        from scripts.evaluate_pieces import merge_results
        merged = merge_results(tmp / "q" / "eval_pieces_best.json", {"periscope": {"none": {"note_f1": 0.1}}})
        assert set(merged["clean"]) >= {"none", "guarded", "tsearch"} and merged["periscope"]["none"]["note_f1"] == 0.1
        assert not (tmp / "q" / "eval_pieces_best.json.tmp").exists()
        # the same request on all performances is not the same result: computed again
        code, text = run_plan("--run", "q", "--runs-dir", str(tmp), "--conditions", "clean", "--modes", "none")
        assert code == plan.MISSING and text == "clean: none", text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
