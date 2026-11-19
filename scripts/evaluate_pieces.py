"""Evaluate a checkpoint (Scoreformer or Score2Seq) on the test sets, with five context modes.

    python scripts/evaluate_pieces.py --run sf_base                     # all conditions, all modes
    python scripts/evaluate_pieces.py --run sf_base --modes chained --conditions periscope --musicxml 5
    python scripts/evaluate_pieces.py --run base --limit-perfs 10       # Michael's run (scripts/train.py)
    python scripts/evaluate_pieces.py --run s2s --modes guarded          # adds a mode to existing results
    python scripts/evaluate_pieces.py --run s2s --modes guarded --extra-margins -0.05,-0.02,0.05
                                                  # + modes guarded@-0.05 ... with other guard margins
    python scripts/evaluate_pieces.py --run s2s --only-missing           # only results not computed yet

Prompt modes (the decoder's context before each window):
  oracle   the ground-truth score just before the window (= scripts/evaluate.py and the training
           validation). An upper bound: real use never has the correct score of the previous bars.
  chained  whole pieces, window by window: each window's prompt is built from the model's OWN
           output so far (segment.prompt_from_output), exactly as at inference on a new recording.
           Errors (a wrong time signature, a missed bar line) carry into the next windows.
           This is the number to report as the system's performance.
  none     no prompt at all (each window decoded on its own).
  guarded  chained, with a guard against cascades: every window is also decoded without a prompt,
           and the no-prompt output replaces the chained one when it agrees clearly better
           (by more than --guard-margin pitch F1) with the window's INPUT notes, which are known at
           inference. The kept output is also the next window's history, so one degenerate window
           (no notes, or a loop) no longer derails the windows after it.
  meter    guarded, then time signatures made consistent along each piece: a window whose time
           signature disagrees with a clear majority of its neighbours is decoded again with that
           one forced, and the piece re-decoded from there (src/decoding.py decode_pieces_meter).
  piece    meter with no limit on how long a deviating run may be: the piece keeps one time
           signature unless a different one holds for a stretch as long as the rest.
Chained, guarded and meter decoding are src/decoding.py, the same code the demo runs.
All modes score every window against its ground-truth target with src/eval/metrics.py, so the
numbers are comparable with each other and with scripts/run_baselines.py. A window whose
predecessor was dropped by preprocessing starts without a prompt in chained mode.

Conditions (as in scripts/evaluate.py): clean, synthetic, real, real_kong, periscope; for models
trained with beats (src/beats.py) also clean_gt_beats, real_gt_beats and real_no_beats, the beat
source of every condition being src/eval/plan.py BEATS (a simulated tracker on MIDI, Beat This! on
the real audio, the ground truth, or none).
Options, mode names and conditions are defined in src/eval/plan.py. ``--only-missing`` computes
only the condition / mode results the run does not have yet with the same margin, all performances
and the current src/eval/plan.py EVAL_VERSION (what pace/reevaluate.sh submits).

Outputs in outputs/runs/<run>/ (modes already in these files from an earlier run are kept, so
``--modes guarded`` adds to a full evaluation instead of replacing it):
  eval_pieces_<ckpt>.json                      metrics[condition][mode], each stamped with how it was
                                               computed (eval_version, limit_perfs, guard_margin)
  pred_pieces_<ckpt>_<condition>.jsonl         per window and mode: performance, win, mode, pred tokens
  musicxml/<condition>/<performance>_{pred,true}.musicxml   (--musicxml N: first N pieces, from the
                                               best mode run: meter, else guarded, else chained)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import SHOW, TruncatingCollate, lower_priority, pick_device  # noqa: E402
from src.dataset import ROLE_CARRIED, ScoreSegmentDataset  # noqa: E402
from src.decoding import (decode_pieces, decode_pieces_meter, input_agreement,  # noqa: E402,F401
                          max_new_tokens, pedal_down, stitch)
from src.eval.metrics import Accumulator  # noqa: E402
from src.beats import BeatNoise  # noqa: E402
from src.eval.plan import (BEATS, GUARDED, PIECE_MAX_BLIP, SPLITS, condition_available,  # noqa: E402,F401
                           decode_passes, extra_margins, has_beats, join_dash_values, load_results, missing,
                           parse_args, split_available, stamp)
from src.preprocessing.noise import NoiseModel  # noqa: E402
from src.preprocessing.vocab import BOS, EOS, SEP  # noqa: E402
from src.scoreformer import load_model  # noqa: E402

D = Path("data/processed")
def window_notes(item: dict) -> torch.Tensor:
    """The window's own notes (in the window or carried into it), without context notes."""
    x = item["notes"]
    return x[item["roles"] <= ROLE_CARRIED] if "roles" in item else x


def write_xml(tokens: list[str], path: Path) -> str:
    from src.preprocessing.detokenize import tokens_to_musicxml
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tokens_to_musicxml(tokens, str(path))
        return "ok"
    except Exception as e:  # noqa: BLE001  (malformed predictions shouldn't stop the evaluation)
        return f"failed: {type(e).__name__}: {e}"


class Decoder:
    def __init__(self, model, vocab, coll, device, amp, batch_size, max_new):
        self.model, self.vocab, self.coll = model, vocab, coll
        self.device, self.amp, self.bs, self.max_new = device, amp, batch_size, max_new
        self.bos, self.sep, self.eos = vocab.stoi[BOS], vocab.stoi[SEP], vocab.stoi[EOS]
        self.ts_ids = torch.tensor([i for i, t in enumerate(vocab.itos) if t.startswith("ts_")])

    def prefix(self, prompt_tokens: list[str], start: list[str] = ()) -> torch.Tensor:
        """<bos> [prompt <sep>] [start]: ``start`` = tokens the output is forced to begin with."""
        head = [self.bos] + (self.vocab.encode(prompt_tokens, add_special=False) + [self.sep] if prompt_tokens else [])
        return torch.tensor(head + self.vocab.encode(list(start), add_special=False))

    @torch.no_grad()
    def __call__(self, items: list[dict], prefixes: list[torch.Tensor], ban_ts: list[bool] | None = None
                 ) -> list[list[str]]:
        """``ban_ts[k]``: item k may not generate a time signature (its own was forced)."""
        out = []
        order = sorted(range(len(items)), key=lambda k: len(items[k]["notes"]))  # less padding
        res = [None] * len(items)
        for k in range(0, len(order), self.bs):
            idx = order[k:k + self.bs]
            b = self.coll([dict(items[i]) for i in idx])
            # per-window output budgets (the same as the demo's); greedy decoding is independent per
            # window, so decoding to the batch's largest budget and cutting each output to its own
            # equals decoding each window with its own budget
            caps = [max_new_tokens(min(len(window_notes(items[i])), self.coll.max_notes), len(prefixes[i]),
                                   self.max_new) for i in idx]
            banned = None
            if ban_ts is not None and any(ban_ts[i] for i in idx):
                banned = torch.zeros(len(idx), len(self.vocab.itos), dtype=torch.bool)
                for r, i in enumerate(idx):
                    if ban_ts[i]:
                        banned[r, self.ts_ids] = True
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                preds = self.model.generate(b["notes"].to(self.device), b["note_mask"].to(self.device),
                                            [prefixes[i] for i in idx], self.eos, max_new=max(caps), banned=banned)
            for i, p, cap in zip(idx, preds, caps):
                res[i] = self.vocab.decode(list(p)[:cap])
        out.extend(res)
        return out


def main():
    # options, mode names and test sets: src/eval/plan.py (shared with pace/reevaluate.sh)
    args = parse_args(sys.argv[1:])
    if not args.normal_priority:
        lower_priority()
    device = pick_device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda" and not amp:
        print("note: this GPU has no bf16 support (e.g. V100), running in fp32: slower, ask for A100/H100")
    run = Path(args.runs_dir) / args.run
    D = Path(args.data_dir)
    modes = args.modes.split(",")
    # decoding passes: (mode name, kind, guard margin); --extra-margins adds guarded@m / meter@m
    requested = decode_passes(modes, args.guard_margin, extra_margins(args.extra_margins))
    margins = {name: m for name, _, m in requested}
    requested_names = [m for m in modes if m not in margins] + [n for n, _, _ in requested]
    todo = None  # --only-missing: {condition: [mode names]} not yet in this run's results
    if args.only_missing:
        todo = missing(load_results(run, args.ckpt), args.conditions.split(","), requested_names,
                       {n: m for n, m in margins.items() if n != "chained"},
                       split_available(D, has_beats(run), args.beat_noise), args.limit_perfs)
        print("to compute: " + ("; ".join(f"{c} {','.join(n)}" for c, n in todo.items()) or "nothing"), flush=True)
        if not todo:
            return
    model = load_model(run / args.ckpt, device)
    cfg = model.cfg
    # context notes the model was trained with (src/dataset.py); 0 for older checkpoints
    ctx = {k: int(getattr(cfg, k, 0)) for k in ("context_before", "context_after")}
    beat_model = bool(getattr(cfg, "beat_features", False))
    beat_noise = BeatNoise.from_json(args.beat_noise) if beat_model and Path(args.beat_noise).exists() else None

    def beat_args(cond):
        """Dataset beat arguments for this condition (src/eval/plan.py BEATS); {} without beats."""
        if not beat_model:
            return {}
        source = BEATS[cond]
        if source == "sim":  # the fitted tracker errors on the ground truth, fixed seed per window
            return {"beats": "gt", "beat_noise": beat_noise}
        return {"beats": source}
    coll = TruncatingCollate(cfg.max_notes, cfg.max_tokens)
    vocab_path = str(D / "vocab.json")
    nm = NoiseModel.from_json("configs/noise_stats.json")
    nm.cfg.clean_prob = 0.0
    ck_name = Path(args.ckpt).stem
    results = {}

    for cond in args.conditions.split(","):
        split = SPLITS[cond]
        if not (D / split / "index.npy").exists():
            print(f"skip {cond}: {D / split} missing")
            continue
        if not condition_available(cond, D, beat_model, args.beat_noise):
            why = ("only for models trained with beats" if not beat_model else
                   f"its beats ({BEATS[cond]}) are not there yet: scripts/build_beats.py, scripts/track_beats.py, "
                   f"scripts/fit_beat_noise.py")
            print(f"skip {cond}: {why}", flush=True)
            continue
        if todo is not None and cond not in todo:
            print(f"skip {cond}: every requested mode is already in the results", flush=True)
            continue
        all_modes = requested_names if todo is None else todo[cond]  # the results computed here
        decode_modes = [p for p in requested if p[0] in all_modes]
        if cond == "synthetic":  # fixed noise draw (seeded per window), no prompt dropout
            ds = ScoreSegmentDataset(str(D / split), vocab_path, noise=nm, seed=12345, prompt_dropout=0.0, **ctx,
                                     **beat_args(cond))
        else:
            ds = ScoreSegmentDataset(str(D / split), vocab_path, **ctx, **beat_args(cond))
        if len(ds) == 0:
            print(f"skip {cond}: {split} is empty")
            continue
        vocab = ds.vocab
        dec = Decoder(model, vocab, coll, device, amp, args.batch_size, cfg.max_tokens)
        perfs = defaultdict(list)
        for i, m in enumerate(ds.store.meta):
            perfs[m["performance"]].append(i)
        names = sorted(perfs)
        if args.limit_perfs and len(names) > args.limit_perfs:
            names = sorted(np.random.default_rng(0).choice(names, args.limit_perfs, replace=False).tolist())
        for p in names:
            perfs[p].sort(key=lambda i: ds.store.meta[i]["win"])
        windows = [i for p in names for i in perfs[p]]
        items = {i: ds[i] for i in windows}
        gt = {i: vocab.decode(items[i]["target"][items[i]["prompt_len"]:-1].tolist()) for i in windows}
        preds = {}  # (mode, window) -> tokens

        if "oracle" in all_modes:
            out = dec([items[i] for i in windows], [items[i]["target"][:items[i]["prompt_len"]] for i in windows])
            preds.update({("oracle", i): o for i, o in zip(windows, out)})
            print(f"{cond}: oracle done ({len(windows)} windows)", flush=True)
        if "none" in all_modes:
            out = dec([items[i] for i in windows], [dec.prefix([]) for _ in windows])
            preds.update({("none", i): o for i, o in zip(windows, out)})
            print(f"{cond}: none done", flush=True)
        pieces, fallback = {}, {}
        meta = ds.store.meta

        def contiguous(prev, i):  # the window starts where the previous one ended (none was dropped)
            return meta[prev].get("s_hi") is not None and meta[prev]["s_hi"] == meta[i].get("s_lo")

        def decode(ws, contexts, starts):
            # a forced time signature must also stay the window's only one: ban generating another
            ban = [any(t.startswith("ts_") for t in s) for s in starts]
            out = dec([items[i] for i in ws], [dec.prefix(c, s) for c, s in zip(contexts, starts)], ban)
            return [list(s) + o for s, o in zip(starts, out)]  # forced starts are part of the output

        by_piece = {p: perfs[p] for p in names}
        alone = {(p, j): preds[("none", i)] for p in names for j, i in enumerate(perfs[p]) if ("none", i) in preds}
        common = dict(contiguous=contiguous, pitches=lambda i: window_notes(items[i])[:, 0].tolist())
        guarded_pass, forced = {}, {}  # per margin: guarded outputs (meter's first pass); per mode: forced
        for name, kind, margin in decode_modes:
            show = lambda j, n, cm=name: print(f"{cond}: {cm} window {j}/{n}", end="\r", flush=True)
            if kind == "chained":
                outs, history, fbs = decode_pieces(by_piece, decode, progress=show, **common)
            elif kind == "guarded":
                outs, history, fbs = guarded_pass[margin] = decode_pieces(
                    by_piece, decode, guard_margin=margin, alone=alone, progress=show, **common)
            else:  # meter / piece: guarded, then one time signature per run of windows / per piece
                outs, history, fbs, forced[name] = decode_pieces_meter(
                    by_piece, decode, guard_margin=margin, alone=alone, first_pass=guarded_pass.get(margin),
                    progress=show, max_blip=PIECE_MAX_BLIP if kind == "piece" else 2, **common)
            preds.update({(name, perfs[p][j]): o for (p, j), o in outs.items()})
            if kind != "chained":
                fallback[(cond, name)] = len(fbs) / len(windows)
            extra = {"chained": "", "guarded": f", {len(fbs) / len(windows):.1%} of windows fell back to no prompt",
                     "meter": f", time signature corrected in {len(forced.get(name, {})) / len(windows):.1%} "
                              f"of windows",
                     "piece": f", time signature corrected in {len(forced.get(name, {})) / len(windows):.1%} "
                              f"of windows"}[kind]
            print(f"{cond}: {name} done{extra}                ", flush=True)
            # MusicXML from the system's best mode run here: meter, else guarded, else chained
            if name == "meter" or (name == "guarded" and "meter" not in all_modes) or \
                    (name == "chained" and not {"guarded", "meter"} & set(all_modes)):
                pieces = history

        res = {}
        for mode in all_modes:
            acc, acc_unseen = Accumulator(), Accumulator()
            for i in windows:
                acc.add(gt[i], preds[(mode, i)])
                if cond == "real" and ds.store.meta[i].get("maestro_split") != "train":
                    acc_unseen.add(gt[i], preds[(mode, i)])
            guarded = mode.split("@")[0] in GUARDED
            # how it was computed (src/eval/plan.py: --only-missing / pace/reevaluate.sh compare it)
            res[mode] = stamp(acc.result(), margins[mode] if guarded else None, args.limit_perfs)
            if guarded:
                res[mode]["fallback_rate"] = fallback.get((cond, mode), 0.0)
            if mode.split("@")[0] in ("meter", "piece"):
                res[mode]["meter_fix_rate"] = len(forced.get(mode, {})) / len(windows)
            if cond == "real":
                results.setdefault("real_unseen_audio", {})[mode] = stamp(
                    acc_unseen.result(), margins[mode] if guarded else None, args.limit_perfs)
        results[cond] = res
        pred_file = run / f"pred_pieces_{ck_name}_{cond}.jsonl"
        kept = []  # predictions of other modes from an earlier run of this script
        if pred_file.exists():
            kept = [line for line in open(pred_file) if json.loads(line)["mode"] not in all_modes]
        with open(pred_file, "w") as f:
            f.writelines(kept)
            for (mode, i), toks in preds.items():
                m = ds.store.meta[i]
                f.write(json.dumps({"performance": m["performance"], "win": m["win"], "mode": mode,
                                    "pred": toks}) + "\n")
        if args.musicxml and pieces:
            for p in names[:args.musicxml]:
                slug = re.sub(r"[^A-Za-z0-9_-]+", "_", p.rsplit(".", 1)[0])
                true = []
                for i in perfs[p]:
                    true = stitch(true, gt[i])
                d = run / "musicxml" / cond
                print(f"  {slug}: pred {write_xml(pieces[p], d / f'{slug}_pred.musicxml')}, "
                      f"true {write_xml(true, d / f'{slug}_true.musicxml')}", flush=True)

    out = run / f"eval_pieces_{ck_name}.json"
    if out.exists():  # keep modes / conditions evaluated earlier
        old = json.load(open(out))
        for c, r in results.items():
            old.setdefault(c, {}).update(r)
        results = old
    json.dump(results, open(out, "w"), indent=1)
    cols = [(c, m) for c in results for m in results[c]]
    print("\n" + f"{'metric':16s}" + "".join(f"{c + '/' + m:>22s}" for c, m in cols))
    for key in SHOW + ("onset_shift_f1", "arp_f1", "ped_f1"):
        print(f"{key:16s}" + "".join(f"{results[c][m][key]:22.3f}" for c, m in cols))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
