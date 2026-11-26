"""Tune the note labeller's beat-level decoding (src/beat_decode.py) on validation windows.

    python scripts/tune_beat_decode.py --run nl_gq                      # val + val_gangqinpu
    python scripts/tune_beat_decode.py --run nl_gq --splits val --windows 200 --no-write

The labeller runs once over the validation windows (the grid with the beat tracker's timing
errors, configs/beat_noise.json, as the labeller was validated in training; --grid-noise 0: the
true grid); then every setting of the decoding is scored on the same predictions: the share of the
labelled score notes whose bar correction, quarter and split are all right ("position accuracy"),
and of those whose written length is right. The setting with the best position accuracy, averaged
over the splits (each split counts once), is written to <run>/beat_decode.json (V2Pipeline reads
it); the whole table goes with it. Setting "weight 0, consistent off" is the labeller's own
choice, so the table shows what the decoding gains or costs against it on held-out data.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import TruncatingCollate, lower_priority, pick_device  # noqa: E402
from src.beat_decode import FILE, BeatDecode, Families, decode_item, family_of  # noqa: E402
from src.dataset import ROLE_WINDOW, ScoreSegmentDataset  # noqa: E402
from src.labeller import keep_mask, load_labeller  # noqa: E402
from src.notelabels import LABEL_FILE, GridNoise  # noqa: E402
from src.release import DELETE_ABOVE  # noqa: E402

WEIGHTS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def settings(weights) -> list[BeatDecode]:
    out = [BeatDecode(weight=w, consistent=False, anchor=0.0) for w in weights]
    for w, staff, anchor in itertools.product(weights, (False, True), (0.0, 0.1)):
        out.append(BeatDecode(weight=w, consistent=True, by_staff=staff, anchor=anchor))
    return out


@torch.no_grad()
def predictions(model, ds, idx, coll, device, amp, batch_size: int, delete_above: float) -> list[dict]:
    """Per window: the heads' outputs the decoding needs, the grid estimates and the labels."""
    items = [ds[int(i)] for i in idx]
    res = []
    for lo in range(0, len(items), batch_size):
        chunk = items[lo:lo + batch_size]
        b = coll([dict(it) for it in chunk])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(b["notes"].to(device), b["note_mask"].to(device))
        out = {k: v.float().cpu() for k, v in out.items()}
        for r, it in enumerate(chunk):
            n = min(len(it["roles"]), out["keep"].shape[1])
            keep = keep_mask(out["keep"][r, :n], delete_above).numpy()
            res.append({
                "sub": out["sub"][r, :n].numpy().astype(np.float32),
                "dur": out["dur"][r, :n].numpy().astype(np.float32),
                "pick": {k: out[k][r, :n].argmax(-1).numpy() for k in ("delta", "pos", "sub", "dur", "staff")},
                "use": (it["roles"][:n] <= ROLE_WINDOW).numpy() & keep,
                "own": (it["roles"][:n] <= ROLE_WINDOW).numpy(),
                "bar_est": it["bar_est"][:n].numpy(), "pos_est": it["pos_est"][:n].numpy(),
                "labels": b["labels"][r, :n].numpy(),
            })
    return res


def score(preds: list[dict], p: BeatDecode | None, fams: Families) -> dict:
    """Position and length accuracy over the labelled own notes; the share of beats mixing a
    straight-only and a triplet-only split."""
    ok_pos = ok_dur = n_pos = n_dur = mixed = beats = 0
    for w in preds:
        pk = w["pick"]
        if p is not None and p.active:
            sub, dur = decode_item(w["sub"], w["dur"], pk, w["bar_est"], w["pos_est"], w["use"], fams, p)
        else:
            sub, dur = pk["sub"], pk["dur"]
        lab = w["labels"]
        m = w["own"] & (lab[:, 1] >= 0) & (lab[:, 2] >= 0) & (lab[:, 3] >= 0)
        ok_pos += int(((pk["delta"] == lab[:, 1]) & (pk["pos"] == lab[:, 2]) & (sub == lab[:, 3]))[m].sum())
        n_pos += int(m.sum())
        md = w["own"] & (lab[:, 4] >= 0)
        ok_dur += int((dur == lab[:, 4])[md].sum())
        n_dur += int(md.sum())
        fam = np.where(fams.masks[1][sub] & fams.masks[0][sub], -1, fams.fam[sub])   # 0 and 1/2: both families
        keys = {}
        for i in np.flatnonzero(w["use"]):
            keys.setdefault((int(w["bar_est"][i]) + int(pk["delta"][i]), int(pk["pos"][i])), set()).add(int(fam[i]))
        beats += len(keys)
        mixed += sum(1 for v in keys.values() if 0 in v and 1 in v)
    return {"pos_acc": ok_pos / max(n_pos, 1), "dur_acc": ok_dur / max(n_dur, 1),
            "mixed_beats": mixed / max(beats, 1), "notes": n_pos}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--splits", default="val,val_gangqinpu")
    ap.add_argument("--windows", type=int, default=600, help="validation windows per split (a fixed random draw)")
    ap.add_argument("--grid-noise", type=float, default=1.0,
                    help="share of windows whose true grid gets the tracker's timing errors (0: the true grid)")
    ap.add_argument("--beat-noise", default="configs/beat_noise.json")
    ap.add_argument("--weights", default=",".join(str(w) for w in WEIGHTS))
    ap.add_argument("--delete-above", type=float, default=DELETE_ABOVE)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--no-write", action="store_true", help="print the table only")
    ap.add_argument("--normal-priority", action="store_true")
    args = ap.parse_args(argv)
    if not args.normal_priority:
        lower_priority()
    device = pick_device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    run, data = Path(args.runs_dir) / args.run, Path(args.data)
    model = load_labeller(run / args.ckpt, device)
    cfg = model.cfg
    ctx = {"context_before": cfg.context_before, "context_after": cfg.context_after}
    coll = TruncatingCollate(cfg.max_notes, 10 ** 9)
    # the data is read with its own vocab (its tokens), the heads' classes are the labeller's: the
    # splits must agree, lengths are mapped by value (a labeller trained with Gangqinpu has a few more)
    from src.notelabels import LabelSpace
    from src.preprocessing.vocab import Vocab
    mvocab = run / "vocab.json" if (run / "vocab.json").exists() else data / "vocab.json"
    mspace = LabelSpace(Vocab.load(str(mvocab)).itos)
    if (mspace.n_sub, mspace.n_dur) != (cfg.n_sub, cfg.n_dur):
        raise SystemExit(f"{mvocab} gives {mspace.n_sub} splits / {mspace.n_dur} lengths, the labeller has "
                         f"{cfg.n_sub} / {cfg.n_dur}: put the vocab.json it trained on next to it")
    gnoise = GridNoise.from_json(args.beat_noise, args.grid_noise) if args.grid_noise > 0 else None
    preds, fams = {}, None
    for split in args.splits.split(","):
        if not (data / split / LABEL_FILE).exists():
            print(f"skip {split}: no {data / split / LABEL_FILE}")
            continue
        ds = ScoreSegmentDataset(str(data / split), str(data / "vocab.json"), notelabels=True, **ctx)
        ds.grid_noise = gnoise
        if ds.label_space.subs != mspace.subs:
            raise SystemExit(f"{data / 'vocab.json'} and {mvocab} split the quarter differently: the labels would not match")
        dur_map = np.array([mspace.dur_class(d) for d in ds.label_space.durs])
        fams = fams or Families(mspace)
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(ds), min(args.windows, len(ds)), replace=False))
        preds[split] = predictions(model, ds, idx, coll, device, amp, args.batch_size, args.delete_above)
        for w in preds[split]:
            d = w["labels"][:, 4]
            w["labels"][:, 4] = np.where(d >= 0, dur_map[np.maximum(d, 0)], d)
        print(f"{split}: {len(idx)} windows labelled", flush=True)
    if not preds:
        raise SystemExit("no validation split with note labels")
    weights = [float(w) for w in args.weights.split(",")]
    table = []
    for p in [None] + settings(weights):
        row = {"setting": p.to_dict() if p else "argmax"}
        for split, pr in preds.items():
            row[split] = score(pr, p, fams)
        row["pos_acc"] = float(np.mean([row[s]["pos_acc"] for s in preds]))
        row["dur_acc"] = float(np.mean([row[s]["dur_acc"] for s in preds]))
        table.append(row)
        name = "argmax (the labeller's own)" if p is None else (
            f"w={p.weight:<4g} {'beat' if p.consistent else 'note'}{'/staff' if p.by_staff else ''}"
            f"{' anchor' if p.anchor else ''}")
        print(f"  {name:36s} pos {row['pos_acc']:.4f}  dur {row['dur_acc']:.4f}  " + "  ".join(
            f"{s}: pos {row[s]['pos_acc']:.4f} mixed {row[s]['mixed_beats']:.3f}" for s in preds), flush=True)
    base = table[0]
    best = max(table[1:], key=lambda r: (round(r["pos_acc"], 5), round(r["dur_acc"], 5)))
    print(f"best: {best['setting']} pos {best['pos_acc']:.4f} (argmax {base['pos_acc']:.4f}), "
          f"dur {best['dur_acc']:.4f} (argmax {base['dur_acc']:.4f})")
    if best["pos_acc"] < base["pos_acc"]:
        print("note: no setting beats the labeller's own choice on these windows")
    if not args.no_write:
        out = run / FILE
        out.write_text(json.dumps({"best": best["setting"], "pos_acc": best["pos_acc"], "argmax_pos_acc": base["pos_acc"],
                                   "dur_acc": best["dur_acc"], "argmax_dur_acc": base["dur_acc"],
                                   "splits": list(preds), "windows": args.windows, "grid_noise": args.grid_noise,
                                   "ckpt": args.ckpt, "table": table}, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
