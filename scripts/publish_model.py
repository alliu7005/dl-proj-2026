"""Publish a finished training run to the Hugging Face model repo (src/release.py HF_REPO): archived
under runs/<run>/ always, and made the default its slot serves (v2_meter, v2_labeller or seq2seq:
what the demo and scripts/audio_to_score.py load) when it is at least as good as the current one.

    python scripts/publish_model.py --run outputs/runs/meter_v4_gq          # archive + promote if better
    python scripts/publish_model.py --run outputs/runs/nl_gq --dry-run      # what it would do
    python scripts/publish_model.py --run outputs/runs/meter_v4_gq --force  # promote whatever the gate says

pace/v2_jobs.sh calls it whenever a training run finishes and is evaluated (PUBLISH=0 turns that
off); the demo pulls the new default when it starts and every hour (demo/app.py --update-every).

What a run is (its config.json "arch"): a meter model, a note labeller, or a seq2seq model. It must
be one the demo can run: a meter model trained for Beat This!'s beats (the demo tracks with it), not
for PM2S's (train_meter.py --tracked pm2s / --aux pm2s); and it must be evaluated (a meter model
tuned too: decode.json), since the gate reads its evaluation:
  meter     eval_meter.json: (grid downbeat F + grid time-signature accuracy) / 2 per test condition
  labeller  eval_pieces_best.json: onset F1 with the true grid per condition
  seq2seq   eval_pieces_best.json: onset F1 in the demo's mode (src/pipeline.py MODES[0]) per condition
The gate compares the mean over the conditions both runs were scored on (at least 2) and promotes when
the candidate is not lower. These are test-set scores: the gate picks the demo's default, not a
reported result (docs/experiments.md has those, paired by performance). A default whose scores are
unknown (not in the manifest, no evaluation here) is not replaced without --force.

Everything goes up as one commit: runs/<run>/ (the slimmed checkpoint, its files, its evaluation),
and when promoted the slot's folder and manifest.json (the release renamed <date>-<slot>-<run> and
tagged so). A run whose checkpoint is in the archive already, with nothing to change, is skipped; the
default's own run with a file added or changed (a labeller's beat_decode.json, tuned after it was
published: scripts/tune_beat_decode.py) updates the slot.

--settings <rule_verdict.json> publishes no model: the benchmark's verdict on the correction rules
(scripts/rule_verdict.py) goes into the manifest's settings, and the demo turns off what it says.
Needs `hf auth login` once (a token with write access) where it runs; without one it says so and
exits 2 (the batch jobs go on).
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_models import VOCABS, export  # noqa: E402
from src.release import HF_REPO, RELEASE  # noqa: E402

SLOT = {"meter": "v2_meter", "labeller": "v2_labeller", "seq2seq": "seq2seq"}
COND = {"meter": ("clean", "real", "periscope", "gangqinpu", "gangqinpu_real"),
        "labeller": ("clean", "real", "periscope", "gangqinpu"),
        "seq2seq": ("clean", "real", "periscope")}
EVAL_FILES = {"meter": ("eval_meter.json", "decode.json"), "labeller": ("eval_pieces_best.json",),
              "seq2seq": ("eval_pieces_best.json",)}
MIN_SHARED = 2
# the trackers whose beats the demo gives the meter model: the released Beat This! (on a recording,
# or on a render of a MIDI file); a meter trained on another tracker's beats reads them wrong
DEMO_TRACKERS = {"beat_this", "beat_this_render"}


def kind_of(run: Path) -> str:
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    arch = cfg.get("arch")
    return arch if arch in ("meter", "labeller") else "seq2seq"


def incompatible(run: Path, kind: str) -> str | None:
    """Why the demo could not run this model (None: it can)."""
    if not (run / "best.pt").exists():
        return f"{run / 'best.pt'} not found"
    if kind == "meter":
        a = json.loads((run / "config.json").read_text()).get("args", {}) if (run / "config.json").exists() else {}
        tracked = {t for t in str(a.get("tracked") or "").split(",") if t and t != "none"}
        if "pm2s" in tracked or a.get("aux", "none") not in (None, "none"):
            return "trained for PM2S's beats (the demo tracks with Beat This!)"
        if tracked - DEMO_TRACKERS:   # e.g. our fine-tuned Beat This! (beat_this_ft): not the demo's tracker
            return (f"trained for another tracker's beats ({', '.join(sorted(tracked - DEMO_TRACKERS))}; "
                    "the demo tracks with the released Beat This!)")
        if not (run / "decode.json").exists():
            return "its decoder is not tuned yet (no decode.json: eval_meter.py --tune)"
    missing = [f for f in EVAL_FILES[kind][:1] if not (run / f).exists()]
    if missing:
        return f"not evaluated yet (no {missing[0]})"
    return None


def scores(run: Path, kind: str) -> dict:
    """{condition: the gate's value} from the run's evaluation ({} if there is none)."""
    from src.pipeline import MODES
    f = run / EVAL_FILES[kind][0]
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    out = {}
    for c in COND[kind]:
        r = d.get(c)
        if not isinstance(r, dict):
            continue
        if kind == "meter" and r.get("grid_downbeat_f") is not None and r.get("grid_ts_acc") is not None:
            out[c] = (float(r["grid_downbeat_f"]) + float(r["grid_ts_acc"])) / 2
        elif kind != "meter":
            m = r.get("grid_true" if kind == "labeller" else MODES[0]) or {}
            if m.get("onset_f1") is not None and m["onset_f1"] == m["onset_f1"]:
                out[c] = float(m["onset_f1"])
    return out


def gate(cand: dict, cur: dict) -> tuple[bool, str]:
    shared = sorted(set(cand) & set(cur))
    if len(shared) < MIN_SHARED:
        return False, f"only {len(shared)} test condition(s) in common with the current default ({shared})"
    a = sum(cand[c] for c in shared) / len(shared)
    b = sum(cur[c] for c in shared) / len(shared)
    per = ", ".join(f"{c} {cand[c]:.3f} vs {cur[c]:.3f}" for c in shared)
    return a >= b, f"mean {a:.3f} vs the current default's {b:.3f} on {len(shared)} conditions ({per})"


def current_scores(entry: dict, kind: str, runs: Path) -> dict:
    """The current default's gate values: kept in the manifest, else from its run's evaluation here."""
    if entry.get("scores"):
        return entry["scores"]
    run = entry.get("eval_run") or entry.get("source_run")
    dec = (entry.get("sources") or {}).get("decode.json")
    if kind == "meter" and dec:   # the release's meter_v3 was evaluated with meter_v3_tp's decoder, there
        run = Path(dec).parent.name
    return scores(runs / run, kind) if run else {}


def remote_manifest(api, repo: str, hf_hub_download) -> dict:
    try:
        p = hf_hub_download(repo_id=repo, filename="manifest.json", repo_type="model")
    except Exception as e:  # noqa: BLE001  (a new repo: nothing released yet)
        if "404" in str(e) or "EntryNotFound" in type(e).__name__ or "RepositoryNotFound" in type(e).__name__:
            return {"release": None, "models": {}}
        raise
    return json.loads(Path(p).read_text())


def publish_settings(path: Path, repo: str, dry_run: bool = False, hub=None) -> int:
    """The rule verdict's decisions -> manifest["settings"]["rules"] (and the evidence beside them):
    one commit of manifest.json; the demo picks it up at its next check. A rule the verdict could not
    decide (None) keeps the manifest's current value."""
    v = json.loads(path.read_text())
    if hub is None:
        try:
            import huggingface_hub as hub
        except ImportError:
            print("not published: needs huggingface_hub (pip install huggingface_hub)")
            return 2
    api = hub.HfApi()
    if not dry_run:
        try:
            api.whoami()
        except Exception as e:  # noqa: BLE001
            print(f"not published: not logged in to Hugging Face ({type(e).__name__}); run `hf auth login` once")
            return 2
    manifest = remote_manifest(api, repo, hub.hf_hub_download)
    settings = manifest.setdefault("settings", {"delete_above": 0.95, "tidy": True})
    rules = dict(settings.get("rules", {}))
    decided = {k: bool(on) for k, on in v.get("rules", {}).items() if on is not None}
    rules.update(decided)
    if rules == settings.get("rules") and settings.get("rule_verdict", {}).get("date") == v.get("date"):
        print("the release's rules are as decided already: nothing to do")
        return 0
    settings["rules"] = rules
    settings["rule_verdict"] = {"date": v.get("date"), "labeller": v.get("labeller"), "rule": v.get("rule"),
                                "evidence": v.get("evidence")}
    print("rules: " + ", ".join(f"{k} {'on' if on else 'OFF'}" for k, on in sorted(rules.items())), flush=True)
    if dry_run:
        print("(dry run: nothing uploaded)")
        return 0
    with tempfile.TemporaryDirectory() as t:
        mf = Path(t) / "manifest.json"
        mf.write_text(json.dumps(manifest, indent=1))
        commit = api.create_commit(repo_id=repo, repo_type="model",
                                   operations=[hub.CommitOperationAdd(path_in_repo="manifest.json", path_or_fileobj=str(mf))],
                                   commit_message=("rule verdict " + str(v.get("date")) + ": " + ", ".join(
                                       f"{k} {'on' if on else 'off'}" for k, on in sorted(decided.items())))[:200])
    print(f"uploaded: {getattr(commit, 'commit_url', commit)}", flush=True)
    return 0


def main(argv=None, hub=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="the finished training run's folder")
    ap.add_argument("--settings", default=None,
                    help="instead of a run: a rule verdict (scripts/rule_verdict.py's JSON) put in the manifest's "
                         "settings, which the demo follows")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--runs-dir", default="outputs/runs", help="where the current default's run may be evaluated")
    ap.add_argument("--force", action="store_true", help="make it the default whatever the gate says")
    ap.add_argument("--no-promote", action="store_true", help="archive only")
    ap.add_argument("--dry-run", action="store_true", help="say what would be uploaded, upload nothing")
    ap.add_argument("--vocab", action="append", default=None)
    args = ap.parse_args(argv)
    if args.settings:
        return publish_settings(Path(args.settings), args.repo, args.dry_run, hub)
    if not args.run:
        ap.error("--run or --settings")
    run = Path(args.run)
    kind = kind_of(run)
    slot = SLOT[kind]
    why = incompatible(run, kind)
    if why:
        print(f"{run.name}: not published: {why}")
        return 1
    if hub is None:
        try:
            import huggingface_hub as hub
        except ImportError:
            print("not published: needs huggingface_hub (pip install huggingface_hub)")
            return 2
    api = hub.HfApi()
    if not args.dry_run:
        try:
            api.whoami()
        except Exception as e:  # noqa: BLE001
            print(f"not published: not logged in to Hugging Face ({type(e).__name__}); run `hf auth login` once")
            return 2
    manifest = remote_manifest(api, args.repo, hub.hf_hub_download)
    models = manifest.setdefault("models", {})
    archive = manifest.setdefault("archive", {})
    cand = scores(run, kind)
    with tempfile.TemporaryDirectory() as t:
        stage = Path(t)
        entry = export(slot, run, stage, kind, args.vocab or list(VOCABS),
                       run / "decode.json" if kind == "meter" else None)
        entry.update(what=f"{RELEASE[slot]['what'].split(' (')[0]} ({run.name})", source_run=run.name,
                     eval_run=run.name, scores=cand)
        sha = entry["files"]["model.pt"]["sha256"]
        cur = models.get(slot)
        if cur is None:
            promote, msg = True, "the slot is empty"
        elif cur.get("files", {}).get("model.pt", {}).get("sha256") == sha:
            changed = sorted(set(entry["files"]) ^ set(cur.get("files", {})) |
                             {f for f in entry["files"] if cur.get("files", {}).get(f, {}).get("sha256")
                              not in (None, entry["files"][f]["sha256"])})
            promote, msg = ((True, f"it is the default, its files changed ({', '.join(changed)})") if changed
                            else (False, "it is the default already"))
        else:
            promote, msg = gate(cand, current_scores(cur, kind, Path(args.runs_dir)))
            if not current_scores(cur, kind, Path(args.runs_dir)):
                promote, msg = False, f"the current default's ({cur.get('source_run')}) scores are unknown"
        if args.force and cur is not None and cur.get("files", {}).get("model.pt", {}).get("sha256") != sha:
            promote, msg = True, f"--force ({msg})"
        if args.no_promote:
            promote, msg = False, "--no-promote"
        known = archive.get(run.name, {}).get("sha256") == sha
        if known and not promote:
            print(f"{run.name}: published already (runs/{run.name}), stays {'the' if msg.startswith('it is') else 'not the'} "
                  f"default: {msg}")
            return 0
        # the commit: the archive, and the slot + manifest when promoted
        ops = []
        for f in sorted((stage / slot).iterdir()):
            ops.append((f"runs/{run.name}/{f.name}", f))
        for f in EVAL_FILES[kind]:
            if (run / f).exists() and not (stage / slot / f).exists():   # decode.json is in the export already
                ops.append((f"runs/{run.name}/{f}", run / f))
        archive[run.name] = {"kind": kind, "sha256": sha, "published": datetime.date.today().isoformat(),
                             "scores": cand, "step": entry.get("best_step") or entry.get("step")}
        release = manifest.get("release")
        if promote:
            for f in sorted((stage / slot).iterdir()):
                ops.append((f"{slot}/{f.name}", f))
            card = Path(__file__).resolve().parents[1] / "docs" / "model_card.md"
            if card.exists():   # the repo's page, as the code it was published with has it
                ops.append(("README.md", card))
            old = models.get(slot, {}).get("source_run")
            models[slot] = entry
            release = f"{datetime.date.today().isoformat()}-{slot}-{run.name}"
            manifest.update(release=release, created=datetime.date.today().isoformat())
            manifest.setdefault("history", []).append({"release": release, "slot": slot, "run": run.name,
                                                       "replaced": old, "why": msg})
            manifest.setdefault("settings", {"delete_above": 0.95, "tidy": True})
        mf = stage / "manifest.json"
        mf.write_text(json.dumps(manifest, indent=1))
        ops.append(("manifest.json", mf))
        verb = "made the default" if promote else "archived only"
        print(f"{run.name} ({kind}): {verb}: {msg}", flush=True)
        for p, _ in ops:
            print(f"  {p}")
        if args.dry_run:
            print("(dry run: nothing uploaded)")
            return 0
        api.create_repo(args.repo, repo_type="model", exist_ok=True)
        commit = api.create_commit(
            repo_id=args.repo, repo_type="model",
            operations=[hub.CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(f)) for p, f in ops],
            commit_message=f"{run.name}: {verb} ({msg})"[:200])
        print(f"uploaded: {getattr(commit, 'commit_url', commit)}", flush=True)
        if promote:
            try:
                api.create_tag(args.repo, tag=release, repo_type="model")
                print(f"tagged {release}")
            except Exception as e:  # noqa: BLE001  (a tag of that name already: the commit stands)
                print(f"not tagged {release}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
