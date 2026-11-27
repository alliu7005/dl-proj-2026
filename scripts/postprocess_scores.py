"""The score-level rules on scores a system wrote already: a copy of the system with PKSpell's key
signature and spellings (src/pkspell.py respell; suffix _ps) and / or the engraving rules (clefs by
range, one-voice stems: src/readable.py engrave_score; _eg), without running the system again.

    python scripts/postprocess_scores.py --system v2_nl_gq_meter_v4_gq_d95_tidy --pkspell
    python scripts/postprocess_scores.py --system v2_nl_gq_meter_v4_gq_d95_tidy --pkspell --engraving \
        --out outputs/system_eval_gangqinpu
    -> <out>/<system>_ps[_eg]/<condition>/<slug>.musicxml + summary.json

The same rules as V2Pipeline(pkspell=..., engraving=...) applies before writing (src/pipeline.py
engrave), so the copy is the system those options make, scored by scripts/beyer_dai.py score like
any other (a folder with a summary.json). Scores music21 cannot read are left out and listed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True)
    ap.add_argument("--pkspell", action="store_true")
    ap.add_argument("--engraving", action="store_true")
    ap.add_argument("--out", default="outputs/system_eval")
    ap.add_argument("--force", action="store_true", help="redo the scores the copy has already")
    args = ap.parse_args(argv)
    if not (args.pkspell or args.engraving):
        ap.error("--pkspell and / or --engraving")
    import music21 as m21

    from src.pkspell import respell
    from src.readable import engrave_score
    out = Path(args.out)
    src = out / args.system
    if not (src / "summary.json").exists():
        raise SystemExit(f"{src}/summary.json not found: run the system first (scripts/system_scores.py)")
    name = args.system + ("_ps" if args.pkspell else "") + ("_eg" if args.engraving else "")
    dst = out / name
    done, failed = 0, []
    for f in sorted(src.glob("*/*.musicxml")):
        target = dst / f.parent.name / f.name
        if target.exists() and not args.force:
            continue
        try:
            score = m21.converter.parse(str(f))
            if args.pkspell:
                respell(score)
            if args.engraving:
                engrave_score(score)
            target.parent.mkdir(parents=True, exist_ok=True)
            score.write("musicxml", fp=str(target), makeNotation=False)
            done += 1
        except Exception as e:  # noqa: BLE001
            failed.append(f"{f.parent.name}/{f.name}: {type(e).__name__}: {str(e)[:120]}")
    summary = json.loads((src / "summary.json").read_text())
    summary["system"] = name
    summary["postprocessed_from"] = args.system
    summary["postprocess"] = {"pkspell": args.pkspell, "engraving": args.engraving, "failed": failed}
    for p in summary.get("pieces", []):
        if p.get("musicxml"):
            p["musicxml"] = str(dst / Path(p["musicxml"]).parent.name / Path(p["musicxml"]).name)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{dst}: {done} scores written, {len(failed)} failed" + ("".join(f"\n  {x}" for x in failed[:5])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
