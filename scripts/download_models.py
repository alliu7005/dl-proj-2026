"""Download the released models into models/ (src/release.py), then check every file against the
manifest's sha256. The demo and scripts/audio_to_score.py use models/ when it is there.

    python scripts/download_models.py                    # the repo in src/release.py HF_REPO, latest
    python scripts/download_models.py --repo <user>/<name> --revision v2-round4
    python scripts/download_models.py --check            # only check what is in models/ already

A private repo needs a token: `hf auth login` once (or HF_TOKEN in the environment).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.release import HF_REPO, MODELS, ROOT  # noqa: E402


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=HF_REPO, help="the Hugging Face model repo, <user or org>/<name>")
    ap.add_argument("--revision", default=None, help="a tag, branch or commit (default: the latest)")
    ap.add_argument("--out", default=str(ROOT / MODELS))
    ap.add_argument("--check", action="store_true", help="only check models/ against its manifest")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not args.check:
        if not args.repo:
            raise SystemExit("no model repo: give --repo <user or org>/<name> (or set HF_REPO in src/release.py)")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit("needs huggingface_hub: pip install huggingface_hub")
        print(f"downloading {args.repo}{'@' + args.revision if args.revision else ''} -> {out}", flush=True)
        snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=str(out), repo_type="model")
    bad = verify(out)
    if bad:
        print("\n".join(bad))
        raise SystemExit(1)
    m = json.loads((out / "manifest.json").read_text())
    print(f"ok: release {m['release']} ({m['created']}): " + ", ".join(m["models"]) + f" in {out}")


if __name__ == "__main__":
    main()
