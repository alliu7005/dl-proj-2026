"""A run published to the Hugging Face model repo (scripts/publish_model.py: runs/<run>/) installed as
a finished run under outputs/runs, so the batch scripts can use a model trained on another account.

    python scripts/install_run.py --run nl_gq                           # -> outputs/runs/nl_gq (if missing)
    python scripts/install_run.py --run meter_v4_gq --as meter_v4_gq_hf --unless-same meter_v4_gq

The published folder's model.pt (the checkpoint without its optimizer) becomes best.pt; its other
files (decode.json, eval_meter.json, beat_decode.json, vocab.json, ...) are copied as they are;
status.json says "done" (a published run is a finished one). An existing run is never overwritten.
--unless-same <run>: when that local run's weights are the published ones (the same tensors), nothing
is installed: they are one model, not two.

Exit status: 0 installed or nothing to do (said which), 1 the repo has no such run or the download
failed (offline, no token for a private repo: `hf auth login`).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.release import HF_REPO  # noqa: E402


def same_weights(a: Path, b: Path) -> bool:
    import torch
    sa = torch.load(a, map_location="cpu", weights_only=False)["model"]
    sb = torch.load(b, map_location="cpu", weights_only=False)["model"]
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def install(src: Path, dst: Path, name: str) -> None:
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / ("best.pt" if f.name == "model.pt" else f.name))
    (dst / "status.json").write_text(json.dumps({"phase": "done", "installed_from": f"{HF_REPO}:runs/{name}"}))


def main(argv=None, fetch=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the published run (runs/<run>/ in the repo)")
    ap.add_argument("--as", dest="name", default=None, help="its name under --runs-dir (default: --run)")
    ap.add_argument("--unless-same", default=None, help="a local run: skip when its weights are the published ones")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--runs-dir", default="outputs/runs")
    args = ap.parse_args(argv)
    runs = Path(args.runs_dir)
    dst = runs / (args.name or args.run)
    if (dst / "best.pt").exists():
        print(f"{dst}: there already, kept")
        return 0
    if fetch is None:
        from scripts.download_models import fetch
    with tempfile.TemporaryDirectory() as t:
        try:
            fetch(args.repo, Path(t), None, [f"runs/{args.run}/*"])
        except Exception as e:  # noqa: BLE001
            print(f"{args.repo}: runs/{args.run} not downloaded ({type(e).__name__}: {str(e)[:200]})")
            return 1
        src = Path(t) / "runs" / args.run
        if not (src / "model.pt").exists():
            print(f"{args.repo} has no runs/{args.run}/model.pt")
            return 1
        if args.unless_same:
            local = runs / args.unless_same / "best.pt"
            if local.exists() and same_weights(local, src / "model.pt"):
                print(f"{args.unless_same} is the published {args.run} (the same weights): nothing installed")
                return 0
        install(src, dst, args.run)
    print(f"{dst}: installed from {args.repo}:runs/{args.run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
