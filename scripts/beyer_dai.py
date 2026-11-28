"""Baseline 3: Beyer & Dai (ISMIR 2024), "End-to-end Piano Performance-MIDI to Score Conversion with
Transformers", and the system-level comparison with their metrics. Runs in its own environment
(Python 3.11, their patched music21 / score_transformer / muster forks): pace/bd_setup.sh.

    python scripts/beyer_dai.py predict --repo <MIDI2ScoreTransformer> --ckpt <MIDI2ScoreTF.ckpt>
        -> outputs/system_eval/beyer_dai/<condition>/<slug>.musicxml      (their model, every input)
    python scripts/beyer_dai.py score --repo <MIDI2ScoreTransformer> --systems beyer_dai,sf_base_reg_cab_meter
        -> outputs/system_eval/results.json, results.md                   (every system, same metrics)

Inputs: outputs/system_eval/manifest.json (scripts/system_inputs.py): every ASAP test performance
(clean MIDI) and its Transkun V2 transcription (real), each with its ASAP reference score
(xml_score.musicxml). Their model needs only the MIDI; it chunks it itself (512 notes, 64 overlap:
``utils.quantize_path`` with its defaults) and returns a music21 score, written with
makeNotation=False as their code says.

Scoring follows their evaluation (midi2scoretransformer/evaluation/run_eval.py, utils.eval), the
same way for every system:
  muster(est file, reference file)                          MUSTER error rates: pitch, missing,
                                                            extra, onset, offset, mean (%): their
                                                            programs read the MusicXML files as
                                                            they are (chords included)
  score_similarity_normalized(est, reference, full=False)   score-similarity error rates per note:
                                                            deletion, insertion, duration, staff,
                                                            stem direction, spelling (x 100)
score_similarity's alignment refuses chords (score_transformer: "Chords are not supported"), so for
it both scores are parsed and every chord is split into its notes at the same offset (``split_chords``:
pitch, duration, tie, stem, articulations and expressions kept), for every system and the reference
alike. Their ``postprocess_score`` is not applied on reading back: their predictions went through
it before they were written (``quantize_path``), and it cannot read chords (``n.pitch``).
``selftest`` scores a reference against itself through exactly this path (expects no errors).
and the per-performance values are averaged without weighting (their aggregation), skipping a
performance whose metric failed (None). Each table row says how many performances it averages;
rows over the performances every system scored ("common") compare like with like. Their paper's
test-set row (Table 3, "Ours") is printed with the clean results, to check the reproduction.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

MUSTER_KEYS = [("PitchER", "pitch"), ("MissRate", "missing"), ("ExtraRate", "extra"), ("OnsetER", "onset"),
               ("OffsetER", "offset"), ("MeanER", "mean")]
SIM_KEYS = [("NoteDeletion", "deletion"), ("NoteInsertion", "insertion"), ("NoteDuration", "duration"),
            ("StaffAssignment", "staff"), ("StemDirection", "stem"), ("NoteSpelling", "spelling")]
# Beyer & Dai 2024, Table 3, "Ours" on the ASAP test set (59 performances): lower is better
PAPER = {"muster": {"PitchER": 3.11, "MissRate": 7.56, "ExtraRate": 6.44, "OnsetER": 15.55, "OffsetER": 23.84,
                    "MeanER": 11.30},
         "sim": {"NoteDeletion": 12.69, "NoteInsertion": 9.06, "NoteDuration": 51.86, "StaffAssignment": 6.62,
                 "StemDirection": 25.03, "NoteSpelling": 8.69}}


def their_code(repo: str):
    """Make their flat-import package importable (``from tokenizer import ...``)."""
    sys.path.insert(0, str(Path(repo) / "midi2scoretransformer"))


REFERENCE = "reference, against itself"  # the floor of the metrics: MUSTER's own alignment noise
THEIR_MODULES = ("config", "models", "tokenizer", "constants", "utils", "score_utils", "dataset", "chunker")


def pred_path(out: Path, system: str, entry: dict) -> Path:
    if system == REFERENCE:
        return Path(entry["reference"])
    return out / system / entry["condition"] / f"{entry['slug']}.musicxml"


def allow_their_classes(ckpt: str) -> list[str]:
    """PyTorch >= 2.6 loads checkpoints with weights_only=True, which refuses the Python objects
    their checkpoint pickles (e.g. config.MyModelConfig). Instead of switching that check off, list
    the objects the file needs and allow exactly those defined in their own modules; anything else
    stops with its name. Returns the names allowed."""
    import importlib
    from torch.serialization import add_safe_globals, get_unsafe_globals_in_checkpoint
    names = get_unsafe_globals_in_checkpoint(ckpt)
    foreign = [n for n in names if n.rpartition(".")[0].split(".")[0] not in THEIR_MODULES]
    if foreign:
        raise SystemExit(f"{ckpt} needs objects from outside their code: {foreign}; not loading it")
    add_safe_globals([getattr(importlib.import_module(n.rpartition(".")[0]), n.rpartition(".")[2]) for n in names])
    return names


# ------------------------------------------------------------------------------------- predict
def predict(args) -> None:
    their_code(args.repo)
    import torch
    from models.roformer import Roformer
    from utils import quantize_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"allowed in the checkpoint: {allow_their_classes(args.ckpt)}", flush=True)
    model = Roformer.load_from_checkpoint(args.ckpt, map_location=device).to(device).eval()
    entries = load_manifest(args.manifest, args.conditions)
    out, done = Path(args.out), []
    for k, e in enumerate(entries):
        dst = pred_path(out, "beyer_dai", e)
        t, err = time.perf_counter(), None
        if dst.exists():  # an earlier run's score must not stand in for a piece that fails now
            dst.unlink()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = quantize_path(e["midi"], model)
            dst.parent.mkdir(parents=True, exist_ok=True)
            score.write("musicxml", fp=str(dst), makeNotation=False)
        except Exception as ex:  # noqa: BLE001  (one bad piece must not stop the rest)
            err = f"{type(ex).__name__}: {ex}"
        done.append({"condition": e["condition"], "performance": e["performance"],
                     "musicxml": None if err else str(dst), "errors": [err] if err else [],
                     "seconds": round(time.perf_counter() - t, 1)})
        print(f"{k + 1}/{len(entries)} {e['condition']} {e['performance']}: {err or 'ok'} ({done[-1]['seconds']} s)",
              flush=True)
    (out / "beyer_dai").mkdir(parents=True, exist_ok=True)
    (out / "beyer_dai" / "summary.json").write_text(json.dumps({"system": "beyer_dai", "pieces": done}, indent=1))
    print(f"wrote {out / 'beyer_dai'}: {sum(d['musicxml'] is not None for d in done)}/{len(done)} scores")


# --------------------------------------------------------------------------------------- score
def split_chords(score):
    """Replace every chord by its notes at the chord's offset in the same container (in place;
    returns the score): each note keeps its pitch, tie and notehead from the chord's note, and
    the chord's duration, stem direction, articulations and expressions. score_similarity's
    alignment needs single notes; the notation it rates (pitch, duration, staff, stem, spelling)
    is unchanged."""
    import copy
    from music21 import chord
    for c in list(score.recurse().getElementsByClass(chord.Chord)):
        site, off = c.activeSite, c.offset
        if site is None:
            continue
        for cn in c.notes:
            n = copy.deepcopy(cn)
            n.duration = copy.deepcopy(c.duration)
            if c.stemDirection not in (None, "unspecified"):
                n.stemDirection = c.stemDirection
            n.articulations = [copy.deepcopy(a) for a in c.articulations]
            n.expressions = [copy.deepcopy(e) for e in c.expressions]
            site.insert(off, n)
        site.remove(c)
    return score


def rate(job: tuple) -> dict:
    """Their metrics for one predicted score (in a worker process): {muster: {...}, sim: {...}}, a
    metric that failed -> None (its error in "error"), a missing prediction -> both None."""
    repo, est_path, ref_path = job
    their_code(repo)
    out = {"muster": None, "sim": None, "error": None}
    if est_path is None or not Path(est_path).exists():
        out["error"] = "no prediction"
        return out
    from music21 import converter
    from muster import muster
    from utils import score_similarity_normalized
    errors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def sim():
            est = split_chords(converter.parse(est_path, forceSource=True))
            ref = split_chords(converter.parse(ref_path, forceSource=True))
            return score_similarity_normalized(est, ref, full=False)
        for key, fn in (("muster", lambda: muster(str(est_path), str(ref_path))), ("sim", sim)):
            try:
                r = fn()
                vals = {k: (float(v) if isinstance(v, (int, float)) else None) for k, v in r.items()}
                if all(v is None for v in vals.values()):  # muster's silent failure: EMPTY_SCORE
                    raise RuntimeError("every value is None")
                out[key] = vals
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{key}: {type(ex).__name__}: {ex}")
    out["error"] = "; ".join(errors) or None
    return out


def selftest(args) -> None:
    """A reference score against itself through ``rate``: score similarity must be exactly 0,
    MUSTER's pitch and onset errors 0 and its mean error small (its voice / note alignment is not
    exact even for identical scores: the Bach fugue against itself has 0.7% missing, 1.6% extra).
    Then, if any predictions exist, one of each system, to show they can be read and scored."""
    entries = load_manifest(args.manifest, "clean,real")
    ref = entries[0]["reference"]
    r = rate((args.repo, ref, ref))
    print(f"{ref} against itself: {r}")
    assert r["error"] is None, r["error"]
    sim_worst = max(abs(r["sim"][k]) for k, _ in SIM_KEYS if r["sim"].get(k) is not None)
    assert sim_worst < 1e-9, f"score similarity of a score against itself: {sim_worst}"
    assert r["muster"]["PitchER"] == 0 and r["muster"]["OnsetER"] == 0, r["muster"]
    assert r["muster"]["MeanER"] < 2.0, f"MUSTER mean error of a score against itself: {r['muster']['MeanER']}"
    out = Path(args.out)
    for d in sorted(p for p in out.iterdir() if p.is_dir() and p.name != "inputs") if out.exists() else []:
        e = next((e for e in entries if pred_path(out, d.name, e).exists()), None)
        if e is not None:
            r = rate((args.repo, str(pred_path(out, d.name, e)), e["reference"]))
            print(f"{d.name} {e['condition']} {e['performance']}: error {r['error']}; MUSTER onset "
                  f"{(r['muster'] or {}).get('OnsetER')}; sim duration {(r['sim'] or {}).get('NoteDuration')}")
    print("selftest ok")


def mean_over(values: list) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def aggregate(rated: dict, entries: list[dict], systems: list[str]) -> dict:
    """rated[(system, condition, performance)] = rate() result -> per system and condition: the
    unweighted mean over performances of every metric (their aggregation; failed metrics skipped),
    over all performances the system scored ("all") and over the ones every system scored
    ("common"), with the number of performances each mean covers."""
    out = {}
    for cond in sorted({e["condition"] for e in entries}):
        perfs = [e["performance"] for e in entries if e["condition"] == cond]

        def ok(s, p):
            r = rated.get((s, cond, p))
            return bool(r and r["muster"] and r["muster"].get("OnsetER") is not None and r["sim"])
        common = [p for p in perfs if all(ok(s, p) for s in systems if s != REFERENCE)]
        for s in systems:
            row = {}
            for subset, ps in (("all", [p for p in perfs if ok(s, p)]), ("common", common)):
                rs = [rated[(s, cond, p)] for p in ps if ok(s, p)]  # the reference may miss a common one
                row[subset] = {"n": len(rs),
                               "muster": {k: mean_over([r["muster"].get(k) for r in rs]) for k, _ in MUSTER_KEYS},
                               "sim": {k: (None if (m := mean_over([r["sim"].get(k) for r in rs])) is None else 100 * m)
                                       for k, _ in SIM_KEYS}}
            row["failed"] = [p for p in perfs if not ok(s, p)]
            out.setdefault(cond, {})[s] = row
    return out


def table(results: dict) -> str:
    """Markdown tables: one per condition and subset, every system, the paper's row with clean."""
    hdr = ["system", "n"] + [f"MUSTER {lab}" for _, lab in MUSTER_KEYS] + [f"sim {lab}" for _, lab in SIM_KEYS]
    fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    lines = ["# System-level comparison (whole pieces, Beyer & Dai's metrics)", "",
             "Error rates in %, lower is better. MUSTER: pitch / missing / extra / onset / offset / mean; score "
             "similarity (per reference note): deletion / insertion / duration / staff / stem / spelling. Unweighted "
             "mean over performances; n = performances averaged. \"common\" = the performances every system scored. "
             f"\"{REFERENCE}\" = each reference scored against itself: the floor (MUSTER's alignment is not exact "
             "even for identical scores).", ""]
    for cond, rows in results.items():
        for subset in ("common", "all"):
            lines += [f"## {cond}, {subset}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
            for s, row in rows.items():
                r = row[subset]
                lines.append("| " + " | ".join([s, str(r["n"])] + [fmt(r["muster"][k]) for k, _ in MUSTER_KEYS]
                                               + [fmt(r["sim"][k]) for k, _ in SIM_KEYS]) + " |")
            if cond == "clean":
                lines.append("| " + " | ".join(["Beyer & Dai, paper (Table 3)", "59"]
                                               + [fmt(PAPER["muster"][k]) for k, _ in MUSTER_KEYS]
                                               + [fmt(PAPER["sim"][k]) for k, _ in SIM_KEYS]) + " |")
            lines.append("")
        failed = {s: row["failed"] for s, row in rows.items() if row["failed"]}
        if failed:
            lines += [f"Not scored ({cond}): " + "; ".join(f"{s}: {len(f)}" for s, f in failed.items()), ""]
    return "\n".join(lines)


def all_systems(out: Path) -> list[str]:
    """Every system with predictions under ``out`` (a folder with a summary.json), Beyer & Dai first."""
    found = sorted(d.name for d in out.iterdir() if d.is_dir() and (d / "summary.json").exists()) if out.exists() else []
    return sorted(found, key=lambda s: s != "beyer_dai")


PIECE_KEYS = ("system", "condition", "performance")


def load_rated(path: Path) -> dict:
    """{(system, condition, performance): rate() result (+ "pred_mtime")} of a results.json, {} without one."""
    if not path.exists():
        return {}
    try:
        pieces = json.loads(path.read_text()).get("pieces", [])
    except ValueError:
        return {}
    return {tuple(e[k] for k in PIECE_KEYS): {k: v for k, v in e.items() if k not in PIECE_KEYS} for e in pieces}


def score(args) -> None:
    """Every system's predictions rated against the references, results.json and results.md written.
    A prediction rated already (results.json holds it, the file unchanged since: its mtime) keeps its
    rating unless --rescore, so a round's last job rates only what is new. The write merges with the
    results.json on disk under a lock: two jobs scoring at once never drop each other's systems."""
    import fcntl

    from joblib import Parallel, delayed
    entries = load_manifest(args.manifest, args.conditions)
    out = Path(args.out)
    names = all_systems(out) if args.systems == "all" else [s for s in args.systems.split(",") if s]
    systems = names + [REFERENCE]
    res_f = out / "results.json"
    prev = {} if args.rescore else load_rated(res_f)
    jobs = [(s, e) for s in systems for e in entries]
    rated, todo = {}, []
    for s, e in jobs:
        pp = pred_path(out, s, e)
        mtime = round(pp.stat().st_mtime, 3) if pp.exists() else None
        key = (s, e["condition"], e["performance"])
        old = prev.get(key)
        if mtime is not None and old is not None and old.get("pred_mtime") == mtime and old.get("muster") is not None:
            rated[key] = old
        else:
            todo.append((key, (args.repo, str(pp) if mtime is not None else None, e["reference"]), mtime))
    print(f"{len(jobs)} predictions: {len(jobs) - len(todo)} rated already, {len(todo)} to rate", flush=True)
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(rate)(j) for _, j, _ in todo)
    for (key, _, mtime), r in zip(todo, res):
        rated[key] = {**r, "pred_mtime": mtime}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ".results.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merged = load_rated(res_f)          # what other jobs wrote meanwhile
        merged.update(rated)
        everyone = sorted({k[0] for k in merged if k[0] != REFERENCE}, key=lambda s: (s != "beyer_dai", s)) + [REFERENCE]
        results = aggregate(merged, entries, everyone)
        per_piece = [{"system": s, "condition": c, "performance": p, **r} for (s, c, p), r in merged.items()]
        tmp = res_f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"results": results, "pieces": per_piece}, indent=1))
        tmp.replace(res_f)
        text = table(results)
        (out / "results.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"wrote {out / 'results.md'} and {res_f}")


def load_manifest(path: str, conditions: str) -> list[dict]:
    keep = conditions.split(",")
    return [e for e in json.loads(Path(path).read_text()) if e["condition"] in keep]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("predict", "score", "selftest"):
        p = sub.add_parser(name)
        p.add_argument("--repo", required=True, help="the MIDI2ScoreTransformer checkout")
        p.add_argument("--manifest", default="outputs/system_eval/manifest.json")
        p.add_argument("--conditions", default="clean,real")
        p.add_argument("--out", default="outputs/system_eval")
    sub.choices["predict"].add_argument("--ckpt", required=True, help="MIDI2ScoreTF.ckpt")
    sub.choices["score"].add_argument("--systems", required=True,
                                      help="comma-separated folders under --out (beyer_dai, <run>_<mode>, ...), "
                                           "or all: every folder with a summary.json")
    sub.choices["score"].add_argument("--jobs", type=int, default=8)
    sub.choices["score"].add_argument("--rescore", action="store_true",
                                      help="rate every prediction again, even those results.json holds unchanged")
    args = ap.parse_args(argv)
    {"predict": predict, "score": score, "selftest": selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
