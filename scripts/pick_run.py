"""Pick the best of several finished training runs by their own validation score and give it one name.

    python scripts/pick_run.py --runs pm2s_ho2 pm2s_ho3 --as pm2s_hos

Each run's score is the "best" its status.json keeps (the value its best.pt was saved at; higher is
better unless --lower-better). The winner's best.pt, config.json, log.csv and status.json are
linked (copied where links are not possible) into outputs/runs/<as>/, with chosen.json saying which
run it is and every candidate's score, so the jobs after it can be submitted before the choice is
made (pace/v2_pm2s2.sh: two learning rates for PM2S, the next jobs read outputs/runs/pm2s_hos).
Runs without best.pt are left out; with none left it fails. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

FILES = ("best.pt", "config.json", "log.csv", "status.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--as", dest="name", required=True, help="the name the winner is linked under")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--lower-better", action="store_true")
    args = ap.parse_args(argv)
    runs = Path(args.runs_dir)
    scores = {}
    for r in args.runs:
        st = runs / r / "status.json"
        if not (runs / r / "best.pt").exists() or not st.exists():
            print(f"{r}: no best.pt / status.json, left out")
            continue
        s = json.loads(st.read_text())
        if s.get("best") is None:
            print(f"{r}: no best score, left out")
            continue
        scores[r] = {"best": float(s["best"]), "best_step": s.get("best_step"), "phase": s.get("phase")}
        print(f"{r}: best {scores[r]['best']:.4f} at step {s.get('best_step')} ({s.get('phase')})")
    if not scores:
        sys.exit("no run to pick from")
    pick = (min if args.lower_better else max)(scores, key=lambda r: scores[r]["best"])
    dst = runs / args.name
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        if not (dst / "chosen.json").exists():
            sys.exit(f"{dst} exists and is not a pick (no chosen.json): not replacing it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in FILES:
        src = (runs / pick / f).resolve()
        if not src.exists():
            continue
        try:
            os.symlink(src, dst / f)
        except OSError:          # no links (Windows without the privilege): a copy
            shutil.copy(src, dst / f)
    (dst / "chosen.json").write_text(json.dumps({"chosen": pick, "candidates": scores,
                                                 "lower_better": args.lower_better}, indent=1))
    print(f"picked {pick} -> {dst}")


if __name__ == "__main__":
    main()
