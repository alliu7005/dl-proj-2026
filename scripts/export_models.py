"""Make the model release (models/, src/release.py) from the training runs: each checkpoint without
its optimizer state (a third of the size), with the files it needs beside it, a manifest with
sha256 sums, and the model card. Then loads the release the way the demo does, to check it.

    python scripts/export_models.py                     # the release (on PACE: every run is there)
    python scripts/export_models.py --meter outputs/runs/meter_v3 --release v2-round4 \\
        --decode outputs/runs/meter_v3_tp/decode.json   # the round-4 release again
    python scripts/export_models.py --seq2seq outputs/runs/sf_base_reg_cab   # another run for one model
    python scripts/export_models.py --no-seq2seq        # v2 only

The vocabulary for each model is the one it was trained with: its run folder's vocab.json if it has
one (train_model.py / train_labeller.py keep it since 2026-11-23), else the first --vocab candidate
that fits the checkpoint (a seq2seq model: as many tokens as it has outputs; the labeller: as many
positions and lengths as its heads). A seq2seq model trained before the 2026-11-22 data rebuild has
data/processed_pre1122/vocab.json (the rebuild added tokens at the end, so the new one does not fit).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.release import DELETE_ABOVE, RELEASE, TIDY  # noqa: E402

KEEP = ("model", "config", "arch", "step", "best", "best_step")   # everything but the optimizer / resume state
VOCABS = ("data/processed_pre1122/vocab.json", "data/processed/vocab.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def slim(src: Path, dst: Path) -> dict:
    """The checkpoint without its optimizer state; returns what it kept (step, parameters)."""
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    out = {k: ck[k] for k in KEEP if k in ck}
    if "model" not in out or "config" not in out:
        raise ValueError(f"{src}: not a training checkpoint (no model / config)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return {"step": out.get("step"), "best_step": out.get("best_step"),
            "params": int(sum(v.numel() for v in out["model"].values() if hasattr(v, "numel")))}


def fits(vocab_path: Path, kind: str, cfg) -> bool:
    """Whether a vocab.json is the one a checkpoint was trained with (as far as its shape shows)."""
    from src.preprocessing.vocab import Vocab
    vocab = Vocab.load(str(vocab_path))
    if kind == "seq2seq":
        return len(vocab) == int(cfg["vocab_size"] if isinstance(cfg, dict) else cfg.vocab_size)
    from src.notelabels import LabelSpace
    space = LabelSpace(vocab.itos)
    get = (lambda k: cfg[k]) if isinstance(cfg, dict) else (lambda k: getattr(cfg, k))
    return (space.n_sub, space.n_dur) == (int(get("n_sub")), int(get("n_dur")))


def pick_vocab(run: Path, kind: str, candidates) -> Path:
    import torch
    cfg = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["config"]
    tried = []
    for p in [run / "vocab.json"] + [Path(c) for c in candidates]:
        if p.exists():
            if fits(p, kind, cfg):
                return p
            tried.append(str(p))
    raise FileNotFoundError(f"{run}: no vocab.json fits its checkpoint (tried {tried or 'none found'}); "
                            "give the one it trained on with --vocab")


def export(name: str, run: Path, out: Path, kind: str, vocabs, decode: Path | None = None) -> dict:
    """One model: models/<name>/model.pt + the files beside it. Returns its manifest entry."""
    ck = run / "best.pt"
    if not ck.exists():
        raise FileNotFoundError(f"{ck} not found")
    dst = out / name
    if dst.exists():
        shutil.rmtree(dst)
    info = slim(ck, dst / "model.pt")
    extra = {}
    if (run / "config.json").exists():
        extra["config.json"] = run / "config.json"
    if kind in ("seq2seq", "labeller"):
        extra["vocab.json"] = pick_vocab(run, kind, vocabs)
    if kind == "labeller" and (run / "beat_decode.json").exists():   # its tuned beat decoding (src/beat_decode.py)
        extra["beat_decode.json"] = run / "beat_decode.json"
    if kind == "seq2seq":
        for p in [run / "feature_stats.json", Path("data/processed/feature_stats.json")]:
            if p.exists():
                extra["feature_stats.json"] = p
                break
    if kind == "meter":
        d = decode or run / "decode.json"
        if not d.exists():
            raise FileNotFoundError(f"{d}: the meter model's tuned decoder (eval_meter.py --tune) not found")
        extra["decode.json"] = d
    for fname, src in extra.items():
        shutil.copy(src, dst / fname)
    return {"what": RELEASE[name]["what"], "kind": kind, "source_run": run.name,
            "sources": {k: str(v) for k, v in extra.items()}, **info,
            "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(dst.iterdir())}}


def check(out: Path, names) -> dict:
    """Load the release as the demo does (CPU): v2 = labeller + meter, seq2seq on its own."""
    from src.pipeline import ScorePipeline, V2Pipeline
    seen = {}
    if {"v2_labeller", "v2_meter"} <= set(names):
        pipe = V2Pipeline(out / "v2_labeller" / "model.pt", out / "v2_meter" / "model.pt", device="cpu",
                          delete_above=DELETE_ABOVE, tidy=TIDY)
        seen["v2"] = pipe.info
        if not pipe.decode_cfg.tempo_weight:
            print("note: the meter's decoder has no tempo prior (tempo_weight 0)")
    if "seq2seq" in names:
        seen["seq2seq"] = ScorePipeline(out / "seq2seq" / "model.pt", device="cpu").info
    return seen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models")
    ap.add_argument("--labeller", default="outputs/runs/nl_base_gn", help="the v2 note labeller's run")
    ap.add_argument("--meter", default="outputs/runs/meter_v4_gq", help="the v2 meter model's run")
    ap.add_argument("--decode", default="",
                    help="the meter's tuned decoder ('' = the meter run's own: meter_v4_gq's has the tempo prior; "
                         "the round-4 release used outputs/runs/meter_v3_tp/decode.json with meter_v3)")
    ap.add_argument("--seq2seq", default="outputs/runs/sf_base_reg_cab_c128", help="the seq2seq model's run")
    ap.add_argument("--no-seq2seq", action="store_true")
    ap.add_argument("--vocab", action="append", default=None,
                    help=f"vocab.json candidates when a run has none (repeatable; default: {', '.join(VOCABS)})")
    ap.add_argument("--card", default="docs/model_card.md", help="the model card (models/README.md)")
    ap.add_argument("--release", default="v2-gangqinpu-meter", help="the release's name, in the manifest")
    ap.add_argument("--no-check", action="store_true", help="skip loading the release afterwards")
    args = ap.parse_args(argv)
    final = Path(args.out)
    out = final.with_name(final.name + ".partial")   # built here; replaces --out only once it loads
    if out.exists():
        shutil.rmtree(out)
    vocabs = args.vocab or list(VOCABS)
    todo = [("v2_labeller", Path(args.labeller), "labeller", None),
            ("v2_meter", Path(args.meter), "meter", Path(args.decode) if args.decode else None)]
    if not args.no_seq2seq:
        todo.append(("seq2seq", Path(args.seq2seq), "seq2seq", None))
    out.mkdir(parents=True, exist_ok=True)
    models = {}
    for name, run, kind, decode in todo:
        models[name] = export(name, run, out, kind, vocabs, decode)
        size = sum(f["bytes"] for f in models[name]["files"].values()) / 1e6
        print(f"{name}: {run} -> {out / name} ({size:.0f} MB, {models[name]['params'] / 1e6:.1f}M parameters, "
              f"step {models[name]['best_step'] or models[name]['step']})", flush=True)
    if Path(args.card).exists():
        shutil.copy(args.card, out / "README.md")
    else:
        print(f"no model card at {args.card}")
    manifest = {"release": args.release, "created": datetime.date.today().isoformat(),
                "settings": {"delete_above": DELETE_ABOVE, "tidy": TIDY}, "models": models}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if not args.no_check:
        for k, v in check(out, list(models)).items():
            print(f"loads: {k}: {v}", flush=True)
    if final.exists():
        shutil.rmtree(final)
    out.rename(final)
    total = sum(f["bytes"] for m in models.values() for f in m["files"].values()) / 1e6
    print(f"wrote {final} ({total:.0f} MB): upload it as it is (docs/handoff.md, 'Models')")


if __name__ == "__main__":
    main()
