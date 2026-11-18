"""Build a split whose inputs are REAL transcriptions (Kong or Transkun V2) instead of clean MIDI.

For every ASAP performance in ``--asap-split`` that has MAESTRO audio (optionally only from
``--maestro-splits``) and a transcription in data/transcribed/, the transcription is shifted onto
the ASAP performance timeline (ASAP files can be cut/offset relative to MAESTRO), segmented with
exactly the same bars/windows as the clean split, and written to
data/processed/<asap-split>_transcribed/ with the same vocab. Targets are copied from the matching
clean segment (same performance + bar_start), so they are always ground truth.

    # evaluation: all test performances with audio (meta "maestro_split" says whether the
    # transcriber was trained on that audio: report "train" separately from validation/test)
    python scripts/build_transcribed.py --asap-split test
    # training / model selection on real errors: only audio the transcriber was NOT trained on
    python scripts/build_transcribed.py --asap-split train --maestro-splits test,validation
    python scripts/build_transcribed.py --asap-split val --maestro-splits test,validation
    # Kong transcriptions instead of Transkun V2 (-> <split>_transcribed_kong)
    python scripts/build_transcribed.py --asap-split test --transcribed data/transcribed
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.performance import estimate_offset, load_performance  # noqa: E402
from src.preprocessing.score import load_score  # noqa: E402
from src.preprocessing.segment import SegmentConfig, segment_pair, segment_windows  # noqa: E402
from src.preprocessing.split import composition_id  # noqa: E402
from src.preprocessing.store import SegmentStore, segment_sizes, write_split  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--asap-split", default="test")
    ap.add_argument("--maestro", default="data/raw/maestro")
    ap.add_argument("--transcribed", default="data/transcribed_transkun",
                    help="data/transcribed_transkun (Transkun V2, default) or data/transcribed (Kong)")
    ap.add_argument("--suffix", default=None,
                    help="output dir = <split>_transcribed<suffix>; default '' for Transkun, '_kong' for Kong")
    ap.add_argument("--maestro-splits", default="train,validation,test",
                    help="only use audio from these MAESTRO splits (Kong was trained on 'train')")
    ap.add_argument("--min-offset-match", type=float, default=0.8,
                    help="skip a performance if the transcription can't be aligned this well")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cfg = yaml.safe_load(open(args.config))
    suffix = args.suffix if args.suffix is not None else ("" if "transkun" in args.transcribed else "_kong")
    root, out = Path(cfg["asap_root"]), Path(cfg["out_dir"])
    seg = dict(cfg["segment"])
    seg["window_notes"] = tuple(seg.get("window_notes", SegmentConfig.window_notes))
    seg_cfg = SegmentConfig(**seg)
    # same windows as the clean split (bars mode: train overlaps, val/test don't; windows mode: the
    # clean split's exact cut points are reused below). No pitch-F1 filter since transcriptions are
    # noisy by design -- only windows whose clean counterpart exists are kept.
    stride = seg_cfg.stride if args.asap_split == "train" else seg_cfg.bars_per_segment
    seg_cfg = replace(seg_cfg, stride=stride, min_pitch_f1=0.0, max_input_notes=seg_cfg.max_input_notes * 2)
    windows_mode = seg_cfg.mode == "windows"
    vocab = Vocab.load(out / "vocab.json")
    clean = SegmentStore(out / args.asap_split)
    # a window is identified by (performance, win) in windows mode, (performance, bar_start) otherwise
    wkey = "win" if windows_mode else "bar_start"
    clean_keys = {(m["performance"], m[wkey]): i for i, m in enumerate(clean.meta)}
    clean_perfs = {k[0] for k in clean_keys}

    ann = json.load(open(root / "asap_annotations.json"))
    maestro = {r["midi_filename"]: r for r in
               csv.DictReader(open(Path(args.maestro) / "maestro-v3.0.0.csv", encoding="utf-8"))}
    sp_json = json.load(open(out / "splits.json"))  # the split build_dataset.py actually used
    pieces = sp_json.get("asap", sp_json)
    ok_maestro = set(args.maestro_splits.split(","))
    scores, segs, report = {}, [], []
    for row in csv.DictReader(open(root / "metadata.csv")):
        mm = row["maestro_midi_performance"].replace("{maestro}/", "")
        if pieces.get(row["folder"]) != args.asap_split or mm not in maestro \
                or maestro[mm]["split"] not in ok_maestro:
            continue
        pred = Path(args.transcribed) / Path(maestro[mm]["audio_filename"]).with_suffix(".mid")
        perf = row["midi_performance"]
        if not pred.exists() or perf not in clean_perfs:
            report.append({"performance": perf,
                           "skipped": "no transcription" if not pred.exists() else "not in clean split"})
            continue
        clean_notes, _ = load_performance(root / perf)
        notes, pedal = load_performance(pred)
        d, frac = estimate_offset(clean_notes, notes)
        if frac < args.min_offset_match:
            report.append({"performance": perf, "skipped": f"alignment match {frac:.2f}"})
            continue
        notes[:, 1:3] -= d
        pedal = pedal - d if len(pedal) else pedal
        if row["folder"] not in scores:
            scores[row["folder"]] = load_score(root / row["xml_score"])
        meta = {"dataset": "asap", "piece": row["folder"], "composition": composition_id("asap", row["folder"]),
                "performance": perf, "split": f"{args.asap_split}_transcribed{suffix}",
                "source": "transkun_real" if "transkun" in args.transcribed else "kong_real",
                "maestro_split": maestro[mm]["split"]}
        if windows_mode:  # cut at exactly the clean windows' score-time cut points
            # sort by window id: s_lo is None for a piece's first window, so tuples can't be compared
            wins = sorted(((m["s_lo"], m["s_hi"], m["win"]) for m in clean.meta if m["performance"] == perf),
                          key=lambda w: w[2])
            s, _ = segment_windows(notes, pedal, ann[perf], scores[row["folder"]], seg_cfg, meta,
                                   train=False, windows=wins)
        else:
            s, _ = segment_pair(notes, pedal, ann[perf], scores[row["folder"]], seg_cfg, meta)
        s = [x for x in s if (perf, x[wkey]) in clean_keys]
        for x in s:  # ground truth: same target and prompt as the clean segment (incl. pedal from clean CC64)
            ci = clean_keys[(perf, x[wkey])]
            x["tokens"] = vocab.decode(clean.get(ci)[2].tolist())
            x["prompt"] = vocab.decode(clean.get_prompt(ci).tolist())
        segs += s
        report.append({"performance": perf, "offset_s": round(d, 3), "offset_match": round(frac, 3),
                       "maestro_split": maestro[mm]["split"], "segments": len(s)})

    sizes = tuple(sum(x) for x in zip(*map(segment_sizes, segs))) if segs else (0, 0, 0, 0)
    dst = out / f"{args.asap_split}_transcribed{suffix}"
    write_split(dst, iter(segs), sizes, vocab)
    json.dump(report, open(dst / "report.json", "w"), indent=1)
    skipped = Counter(r["skipped"].split(" ")[0] for r in report if "skipped" in r)
    print(f"{dst}: {len(segs)} segments from {sum('segments' in r for r in report)} performances "
          f"(audio MAESTRO split: {dict(Counter(s['maestro_split'] for s in segs))}); skipped: {dict(skipped)}")


if __name__ == "__main__":
    main()
