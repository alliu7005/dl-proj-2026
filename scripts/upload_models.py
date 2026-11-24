"""Upload the release (models/, scripts/export_models.py) to a Hugging Face model repo, after
checking every file against its manifest; tag the upload with the release's name, so a download
can ask for exactly this version (download_models.py --revision <tag>).

    hf auth login                                          # once: a token with write access
    python scripts/upload_models.py --repo <user or org>/piano-score-models
    python scripts/upload_models.py --repo <user>/<name> --private     # visible only to you (and an org's members)

Then set HF_REPO in src/release.py to the repo, so download_models.py needs no --repo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_models import verify  # noqa: E402
from src.release import MODELS, ROOT  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="<user or org>/<name>; created if it does not exist")
    ap.add_argument("--models", default=str(ROOT / MODELS))
    ap.add_argument("--private", action="store_true", help="create the repo private (an existing repo keeps its setting)")
    ap.add_argument("--no-tag", action="store_true", help="do not tag the upload with the release's name")
    args = ap.parse_args(argv)
    models = Path(args.models)
    bad = verify(models)
    if bad:
        raise SystemExit("not uploading, the release does not match its manifest:\n" + "\n".join(bad))
    if not (models / "README.md").exists():
        raise SystemExit(f"{models / 'README.md'} (the model card) is missing: run scripts/export_models.py again")
    release = json.loads((models / "manifest.json").read_text())["release"]
    from huggingface_hub import HfApi
    api = HfApi()
    url = api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"repo: {url}", flush=True)
    commit = api.upload_folder(repo_id=args.repo, folder_path=str(models), repo_type="model",
                               commit_message=f"release {release}",
                               ignore_patterns=["*.partial*", ".cache/**"])   # .cache: a download's own bookkeeping
    print(f"uploaded: {getattr(commit, 'commit_url', commit)}", flush=True)
    if not args.no_tag:
        tags = {t.name for t in api.list_repo_refs(args.repo, repo_type="model").tags}
        if release in tags:
            print(f"tag {release} exists already: left on its commit (a new release needs a new --release name "
                  "in export_models.py)")
        else:
            api.create_tag(args.repo, tag=release, repo_type="model")
            print(f"tagged {release}")
    print(f"next: set HF_REPO = \"{args.repo}\" in src/release.py; teammates then run python scripts/download_models.py")


if __name__ == "__main__":
    main()
