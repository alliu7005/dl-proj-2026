"""Evaluate a checkpoint (Scoreformer or Score2Seq) on the test sets, with three prompt modes.

    python scripts/evaluate_pieces.py --run sf_base                     # all conditions, all modes
    python scripts/evaluate_pieces.py --run sf_base --modes chained --conditions periscope --musicxml 5
    python scripts/evaluate_pieces.py --run base --limit-perfs 10       # Michael's run (scripts/train.py)

Prompt modes (the decoder's context before each window):
  oracle   the ground-truth score just before the window (= scripts/evaluate.py and the training
           validation). An upper bound: real use never has the correct score of the previous bars.
  chained  whole pieces, window by window: each window's prompt is built from the model's OWN
           output so far (segment.prompt_from_output), exactly as at inference on a new recording.
           Errors (a wrong time signature, a missed bar line) carry into the next windows.
           This is the number to report as the system's performance.
  none     no prompt at all (each window decoded on its own).
All modes score every window against its ground-truth target with src/eval/metrics.py, so the
numbers are comparable with each other and with scripts/run_baselines.py. A window whose
predecessor was dropped by preprocessing starts without a prompt in chained mode.

Conditions (as in scripts/evaluate.py): clean, synthetic, real, real_kong, periscope.

Outputs in outputs/runs/<run>/:
  eval_pieces_<ckpt>.json                      metrics[condition][mode]
  pred_pieces_<ckpt>_<condition>.jsonl         per window and mode: performance, win, mode, pred tokens
  musicxml/<condition>/<performance>_{pred,true}.musicxml   (--musicxml N: first N pieces, chained mode)
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
from src.dataset import ScoreSegmentDataset  # noqa: E402
from src.eval.metrics import Accumulator  # noqa: E402
from src.preprocessing.noise import NoiseModel  # noqa: E402
from src.preprocessing.segment import prompt_from_output  # noqa: E402
from src.preprocessing.vocab import BOS, EOS, SEP  # noqa: E402
from src.scoreformer import load_model  # noqa: E402

D = Path("data/processed")
SPLITS = {"clean": "test", "synthetic": "test", "real": "test_transcribed",
          "real_kong": "test_transcribed_kong", "periscope": "test_periscope"}


def pedal_down(tokens: list[str]) -> bool:
    for t in reversed(tokens):
        if t in ("ped_on", "ped_off"):
            return t == "ped_on"
    return False


def stitch(history: list[str], window: list[str]) -> list[str]:
    """Append a window's tokens to the piece so far, dropping the window's opening ``ped_on`` if
    the pedal is already down (the only cleanup consecutive windows need, see README)."""
    if pedal_down(history):
        first = next((k for k, t in enumerate(window) if t.startswith("ped_")), None)
        if first is not None and window[first] == "ped_on":
            window = window[:first] + window[first + 1:]
    return history + window


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

    def prefix(self, prompt_tokens: list[str]) -> torch.Tensor:
        if not prompt_tokens:
            return torch.tensor([self.bos])
        return torch.tensor([self.bos] + self.vocab.encode(prompt_tokens, add_special=False) + [self.sep])

    @torch.no_grad()
    def __call__(self, items: list[dict], prefixes: list[torch.Tensor]) -> list[list[str]]:
        out = []
        order = sorted(range(len(items)), key=lambda k: len(items[k]["notes"]))  # less padding
        res = [None] * len(items)
        for k in range(0, len(order), self.bs):
            idx = order[k:k + self.bs]
            b = self.coll([dict(items[i]) for i in idx])
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                preds = self.model.generate(b["notes"].to(self.device), b["note_mask"].to(self.device),
                                            [prefixes[i] for i in idx], self.eos, max_new=self.max_new)
            for i, p in zip(idx, preds):
                res[i] = self.vocab.decode(p)
        out.extend(res)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--modes", default="oracle,chained,none")
    ap.add_argument("--conditions", default="clean,synthetic,real,real_kong,periscope")
    ap.add_argument("--limit-perfs", type=int, default=0, help="performances per condition (0 = all)")
    ap.add_argument("--musicxml", type=int, default=0, help="write MusicXML for the first N pieces (chained)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--normal-priority", action="store_true")
    args = ap.parse_args()
    if not args.normal_priority:
        lower_priority()
    device = pick_device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda" and not amp:
        print("note: this GPU has no bf16 support (e.g. V100), running in fp32: slower, ask for A100/H100")
    run = Path("outputs/runs") / args.run
    model = load_model(run / args.ckpt, device)
    cfg = model.cfg
    coll = TruncatingCollate(cfg.max_notes, cfg.max_tokens)
    vocab_path = str(D / "vocab.json")
    modes = args.modes.split(",")
    nm = NoiseModel.from_json("configs/noise_stats.json")
    nm.cfg.clean_prob = 0.0
    ck_name = Path(args.ckpt).stem
    results = {}

    for cond in args.conditions.split(","):
        split = SPLITS[cond]
        if not (D / split / "index.npy").exists():
            print(f"skip {cond}: {D / split} missing")
            continue
        if cond == "synthetic":  # fixed noise draw (seeded per window), no prompt dropout
            ds = ScoreSegmentDataset(str(D / split), vocab_path, noise=nm, seed=12345, prompt_dropout=0.0)
        else:
            ds = ScoreSegmentDataset(str(D / split), vocab_path)
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

        if "oracle" in modes:
            out = dec([items[i] for i in windows], [items[i]["target"][:items[i]["prompt_len"]] for i in windows])
            preds.update({("oracle", i): o for i, o in zip(windows, out)})
            print(f"{cond}: oracle done ({len(windows)} windows)", flush=True)
        if "none" in modes:
            out = dec([items[i] for i in windows], [dec.prefix([]) for _ in windows])
            preds.update({("none", i): o for i, o in zip(windows, out)})
            print(f"{cond}: none done", flush=True)
        pieces = {}
        if "chained" in modes:
            history = {p: [] for p in names}
            for j in range(max(len(perfs[p]) for p in names)):
                step = [(p, perfs[p][j]) for p in names if j < len(perfs[p])]
                prefixes = []
                for p, i in step:
                    prev = perfs[p][j - 1] if j else None
                    contiguous = (prev is not None and ds.store.meta[prev].get("s_hi") is not None
                                  and ds.store.meta[prev]["s_hi"] == ds.store.meta[i].get("s_lo"))
                    prefixes.append(dec.prefix(prompt_from_output(history[p]) if contiguous else []))
                out = dec([items[i] for _, i in step], prefixes)
                for (p, i), o in zip(step, out):
                    preds[("chained", i)] = o
                    history[p] = stitch(history[p], o)
                print(f"{cond}: chained window {j + 1}", end="\r", flush=True)
            pieces = history
            print(f"{cond}: chained done                ", flush=True)

        res = {}
        for mode in modes:
            acc, acc_unseen = Accumulator(), Accumulator()
            for i in windows:
                acc.add(gt[i], preds[(mode, i)])
                if cond == "real" and ds.store.meta[i].get("maestro_split") != "train":
                    acc_unseen.add(gt[i], preds[(mode, i)])
            res[mode] = acc.result()
            if cond == "real":
                results.setdefault("real_unseen_audio", {})[mode] = acc_unseen.result()
        results[cond] = res
        with open(run / f"pred_pieces_{ck_name}_{cond}.jsonl", "w") as f:
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
    json.dump(results, open(out, "w"), indent=1)
    cols = [(c, m) for c in results for m in results[c]]
    print("\n" + f"{'metric':16s}" + "".join(f"{c + '/' + m:>22s}" for c, m in cols))
    for key in SHOW + ("onset_shift_f1", "arp_f1", "ped_f1"):
        print(f"{key:16s}" + "".join(f"{results[c][m][key]:22.3f}" for c, m in cols))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
