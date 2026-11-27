"""A meter "run" that is several finished meter models read together, and / or one model read on
several views of the performance (src/meter.py MeterEnsemble): their probabilities averaged before
the decoder. Everything that takes a meter run (scripts/eval_meter.py, scripts/system_scores.py,
pace/v2_jobs.sh, the pipeline) takes it by name, as any other run.

    python scripts/make_ensemble.py --name meter_v4_gq_ens3 --members meter_v4_gq meter_v4_gq_s1 meter_v4_gq_s2
    python scripts/make_ensemble.py --name meter_v4_gq_tta --members meter_v4_gq --views tta
    python scripts/eval_meter.py --run meter_v4_gq_ens3 --tune --tempo-prior     # its own decoder, tuned

Writes outputs/runs/<name>/: ensemble.json (the members' checkpoints, relative to the runs folder,
and the views), config.json (arch "meter_ensemble"), status.json ("done": it trains nothing). No
best.pt: scripts/publish_model.py does not publish it (the release holds single models). The
decoder is tuned for the ensemble itself (averaged probabilities are less peaked than one
model's), never copied from a member.

Why (docs/experiments.md): meter models of one recipe differ from seed to seed by up to 0.09 in
time-signature accuracy and ~0.9 MUSTER on whole pieces, more than most changes measured, and a
larger model overfits; the release's meter model is one draw.

--views: none (each performance as it is), tta (as it is, transposed -3 / +3 semitones, stretched
x0.9 / x1.1: the training augmentation's range, src/meter.py TTA_VIEWS).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.meter import ENSEMBLE_FILE, IDENTITY_VIEW, TTA_VIEWS  # noqa: E402

VIEWS = {"none": (IDENTITY_VIEW,), "tta": TTA_VIEWS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--members", nargs="+", required=True, help="finished meter runs (their best.pt)")
    ap.add_argument("--views", choices=list(VIEWS), default="none")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--force", action="store_true", help="replace an ensemble of this name")
    args = ap.parse_args(argv)
    runs = Path(args.runs_dir)
    out = runs / args.name
    if (out / args.ckpt).exists():
        raise SystemExit(f"{out} is a trained run (it has {args.ckpt}): pick another --name")
    if (out / ENSEMBLE_FILE).exists() and not args.force:
        old = json.loads((out / ENSEMBLE_FILE).read_text())
        want = [f"{m}/{args.ckpt}" for m in args.members]
        if old.get("members") == want and [list(v) for v in old.get("views", [])] == [list(v) for v in VIEWS[args.views]]:
            print(f"{out}: the same ensemble exists already")
            return 0
        raise SystemExit(f"{out} is a different ensemble already (--force replaces it)")
    if len(args.members) == 1 and args.views == "none":
        raise SystemExit("one member and no views is that member itself")
    if len(set(args.members)) != len(args.members):
        raise SystemExit("a member is listed twice")
    for m in args.members:
        p = runs / m / args.ckpt
        if not p.exists():
            raise SystemExit(f"{p} not found: train {m} first")
        st = runs / m / "status.json"
        if st.exists() and json.loads(st.read_text()).get("phase") != "done":
            raise SystemExit(f"{m} has not finished training ({st})")
        if (runs / m / ENSEMBLE_FILE).exists():
            raise SystemExit(f"{m} is an ensemble: list its members instead")
    out.mkdir(parents=True, exist_ok=True)
    spec = {"members": [f"{m}/{args.ckpt}" for m in args.members], "views": [list(v) for v in VIEWS[args.views]]}
    (out / ENSEMBLE_FILE).write_text(json.dumps(spec, indent=1))
    (out / "config.json").write_text(json.dumps({"arch": "meter_ensemble", **spec, "view_set": args.views}, indent=1))
    (out / "status.json").write_text(json.dumps({"phase": "done", "note": "an ensemble: nothing to train"}))
    for f in ("decode.json", "eval_meter.json"):   # a decoder / results of a different ensemble are stale
        if args.force and (out / f).exists():
            (out / f).unlink()
    print(f"{out}: {len(args.members)} member(s) x {len(VIEWS[args.views])} view(s) "
          f"({', '.join(args.members)}; views {args.views}); next: eval_meter.py --run {args.name} --tune")
    return 0


if __name__ == "__main__":
    sys.exit(main())
