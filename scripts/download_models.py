"""Download the released models into models/ (src/release.py), then check every file against the
manifest's sha256. The demo and scripts/audio_to_score.py use models/ when it is there.

    python scripts/download_models.py                    # the repo in src/release.py HF_REPO, latest
    python scripts/download_models.py --repo <user>/<name> --revision v2-round4
    python scripts/download_models.py --check            # only check what is in models/ already
    python scripts/download_models.py --run meter_v4_gql # an archived run -> models/runs/meter_v4_gql/

Only the default models are fetched (manifest.json, the model card and each slot's folder), not the
archive of every published run (runs/, scripts/publish_model.py). ``update()`` is what the demo calls
when it starts and every hour: it downloads only when the repo's manifest differs from models/'s, so
a new default (publish_model.py promotes a better run when its training finishes) arrives by itself.

A private repo needs a token: `hf auth login` once (or HF_TOKEN in the environment).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.release import HF_REPO, MODELS, RELEASE, ROOT  # noqa: E402

DEFAULT_FILES = ["manifest.json", "README.md"] + [f"{slot}/*" for slot in RELEASE]   # not runs/


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(out: Path) -> list[str]:
    """Problems with the release in ``out`` (empty: every file is there and matches the manifest)."""
    mf = out / "manifest.json"
    if not mf.exists():
        return [f"{mf} not found"]
    bad = []
    for name, m in json.loads(mf.read_text())["models"].items():
        for fname, f in m["files"].items():
            p = out / name / fname
            if not p.exists():
                bad.append(f"{p}: missing")
            elif sha256(p) != f["sha256"]:
                bad.append(f"{p}: sha256 does not match the manifest (a partial download? run again)")
    return bad


def fetch(repo: str, out: Path, revision: str | None = None, allow=None, hub=None) -> None:
    if hub is None:
        import huggingface_hub as hub
    hub.snapshot_download(repo_id=repo, revision=revision, local_dir=str(out), repo_type="model",
                          allow_patterns=allow or DEFAULT_FILES)


def update(repo: str = HF_REPO, out: Path = ROOT / MODELS, hub=None, log=print) -> bool:
    """Bring models/ to the repo's current default when it differs. Returns True when models/ changed.
    Never raises: offline, no huggingface_hub, a failed check: models/ stays as it is (said so)."""
    if not repo:
        return False
    try:
        if hub is None:
            import huggingface_hub as hub
        remote = json.loads(Path(hub.hf_hub_download(repo_id=repo, filename="manifest.json",
                                                     repo_type="model")).read_text())
        mf = Path(out) / "manifest.json"
        local = json.loads(mf.read_text()) if mf.exists() else None
        files = lambda m: {k: v.get("files") for k, v in m.get("models", {}).items()}  # noqa: E731
        same = local is not None and files(local) == files(remote)
        if same and not verify(Path(out)):
            if local.get("settings") != remote.get("settings"):   # the rule verdict changed: no files to fetch
                mf.write_text(json.dumps(remote, indent=1))
                log(f"models: {repo}'s settings changed (rules: {remote.get('settings', {}).get('rules')})")
                return True
            return False
        log(f"models: {repo} has release {remote.get('release')} (here: "
            f"{local.get('release') if local else 'none'}): downloading")
        fetch(repo, Path(out), hub=hub)
        bad = verify(Path(out))
        if bad:
            log("models: the download does not match its manifest: " + "; ".join(bad[:3]))
            return False
        log(f"models: now release {remote.get('release')}")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"models: no update from {repo} ({type(e).__name__}: {str(e)[:120]}); using what is here")
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=HF_REPO, help="the Hugging Face model repo, <user or org>/<name>")
    ap.add_argument("--revision", default=None, help="a tag, branch or commit (default: the latest)")
    ap.add_argument("--out", default=str(ROOT / MODELS))
    ap.add_argument("--check", action="store_true", help="only check models/ against its manifest")
    ap.add_argument("--run", default=None, help="an archived run instead (runs/<run>/ -> <out>/runs/<run>/)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.run:
        try:
            fetch(args.repo, out, args.revision, [f"runs/{args.run}/*"])
        except ImportError:
            raise SystemExit("needs huggingface_hub: pip install huggingface_hub")
        if not (out / "runs" / args.run / "model.pt").exists():
            raise SystemExit(f"{args.repo} has no runs/{args.run}/ (scripts/publish_model.py publishes runs)")
        print(f"ok: {out / 'runs' / args.run}")
        return
    if not args.check:
        if not args.repo:
            raise SystemExit("no model repo: give --repo <user or org>/<name> (or set HF_REPO in src/release.py)")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit("needs huggingface_hub: pip install huggingface_hub")
        print(f"downloading {args.repo}{'@' + args.revision if args.revision else ''} -> {out}", flush=True)
        snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=str(out), repo_type="model",
                          allow_patterns=DEFAULT_FILES)
    bad = verify(out)
    if bad:
        print("\n".join(bad))
        raise SystemExit(1)
    m = json.loads((out / "manifest.json").read_text())
    print(f"ok: release {m['release']} ({m['created']}): " + ", ".join(m["models"]) + f" in {out}")


if __name__ == "__main__":
    main()
