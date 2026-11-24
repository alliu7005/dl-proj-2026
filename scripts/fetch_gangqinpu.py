"""Download the Gangqinpu release's Parquet data (the meter model's extra training data; see
src/preprocessing/gangqinpu.py and scripts/build_meter_data.py --gangqinpu).

    python scripts/fetch_gangqinpu.py                 # -> data/raw/gangqinpu/data/*.parquet (~660 MB)

The dataset is gated (CC BY-NC-SA 4.0, non-commercial). Once per account: open
https://huggingface.co/datasets/Wayne42/gangqinpu, accept the licence, then log in where this runs
(on PACE: HF_HOME=$HOME/scratch/.cache/huggingface hf auth login, the cache pace/common.sh uses).
Only data/ is fetched (the default config's Parquet: notes, alignment, beats), at a pinned
revision, so a rebuild reads the same rows; the original audio and scores are not needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = "Wayne42/gangqinpu"
REVISION = "2da2e7c0a784f655c3c9d92c08ec85d841b5829a"   # v1.0.0 as of 2026-10-29


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/gangqinpu")
    ap.add_argument("--revision", default=REVISION)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if sorted((out / "data").glob("*.parquet")):
        print(f"{out / 'data'}: already there")
        return
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError
    except ImportError:
        raise SystemExit("needs huggingface_hub: pip install huggingface_hub")
    try:
        snapshot_download(repo_id=REPO, repo_type="dataset", revision=args.revision, local_dir=str(out),
                          allow_patterns=["data/*.parquet", "README.md", "LICENSE", "BUILD_INFO.json"])
    except GatedRepoError:
        sys.exit(f"{REPO} is gated: accept its licence at https://huggingface.co/datasets/{REPO} and log in "
                 "(hf auth login; on PACE with HF_HOME=$HOME/scratch/.cache/huggingface)")
    files = sorted((out / "data").glob("*.parquet"))
    print(f"{len(files)} files -> {out / 'data'}: " + ", ".join(f.name for f in files))


if __name__ == "__main__":
    main()
