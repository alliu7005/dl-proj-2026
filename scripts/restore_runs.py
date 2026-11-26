"""Training-run folders rebuilt from the released models (models/, scripts/download_models.py), for a
machine that never trained them: what the batch scripts and scripts/system_scores.py read.

    python scripts/download_models.py       # once: models/
    python scripts/restore_runs.py          # -> outputs/runs/{nl_base_gn, meter_v3, meter_v3_tp, sf_base_reg_cab_c128}

Each run gets best.pt (the release's model.pt: weights and config, no optimizer state), the files
the release keeps beside it (config.json, vocab.json, feature_stats.json: the seq2seq model's own
input statistics, which src/pipeline.py reads from next to the checkpoint first), and a
status.json saying it is done. meter_v3_tp is meter_v3's checkpoint with the release's tuned
decoder (decode.json, the tempo prior). A run folder that already has a best.pt is left as it is.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

RUNS = {  # run -> (release folder, files copied beside best.pt)
    "nl_base_gn": ("v2_labeller", ["config.json", "vocab.json"]),
    "meter_v3": ("v2_meter", ["config.json"]),
    "meter_v3_tp": ("v2_meter", ["config.json", "decode.json"]),
    "sf_base_reg_cab_c128": ("seq2seq", ["config.json", "vocab.json", "feature_stats.json"]),
}
DONE = {"phase": "done", "step": 0, "total_steps": 0, "best": 0, "best_step": 0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="models")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--only", nargs="*", help="these runs only (default: all of them)")
    args = ap.parse_args(argv)
    models, runs = Path(args.models), Path(args.runs_dir)
    if not (models / "manifest.json").exists():
        raise SystemExit(f"{models}/manifest.json missing: run scripts/download_models.py first")
    for run, (src, extra) in RUNS.items():
        if args.only and run not in args.only:
            continue
        d = runs / run
        if (d / "best.pt").exists():
            print(f"{run}: {d / 'best.pt'} there already, left as it is")
            continue
        if not (models / src / "model.pt").exists():
            print(f"{run}: {models / src / 'model.pt'} missing, skipped")
            continue
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(models / src / "model.pt", d / "best.pt")
        for f in extra:
            if (models / src / f).exists():
                shutil.copy(models / src / f, d / f)
        (d / "status.json").write_text(json.dumps(DONE))
        print(f"{run}: restored from {models / src} ({', '.join(sorted(x.name for x in d.iterdir()))})")


if __name__ == "__main__":
    main()
