"""Reproduce Beyer & Dai (ISMIR 2024) ASAP train/val/test split + their ignore list.

Their split lives in MIDI2ScoreTransformer/midi2scoretransformer/{constants,dataset}.py and is defined
over ACPAS ``piece_id``s. This script replays their filtering on the ACPAS metadata and writes

    configs/splits/beyer_dai.json  =  {"pieces": {asap_folder: split}, "exclude_performances": [...]}

    python scripts/make_bd_split.py --acpas ../external/ACPAS-dataset
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

# Copied verbatim from MIDI2ScoreTransformer/midi2scoretransformer/constants.py
SKIP = {"{ASAP}/Glinka/The_Lark/Denisova10M.mid", "{ASAP}/Glinka/The_Lark/Kleisen07M.mid"}
TO_IGNORE_INDICES = [152, 153, 154, 165, 166, 179, 180, 181, 332, 333, 334, 335, 349, 350,
                     351, 418, 419, 420, 426, 428, 429, 430, 472, 473, 474, 489, 490, 491,
                     516, 517, 518, 519, 520, 521, 522, 540, 541, 560, 609, 774, 798, 799,
                     800, 801, 802, 803, 819, 920, 921, 935, 936, 937, 938, 939, 940, 941,
                     979, 980, 981, 997, 998, 999, 1012, 1013, 1014, 1017, 1018]
TEST_PIECE_IDS = {15, 78, 159, 172, 254, 288, 322, 374, 395, 399, 411, 418, 452, 478}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acpas", required=True, help="clone of github.com/cheriell/ACPAS-dataset")
    ap.add_argument("--asap", default="data/raw/asap_bd")
    ap.add_argument("--out", default="configs/splits/beyer_dai.json")
    args = ap.parse_args()

    acpas = Path(args.acpas)
    rows = []
    for name in ("metadata_R.csv", "metadata_S.csv"):  # pd.concat([real, synthetic])
        rows += list(csv.DictReader(open(acpas / name, newline="", encoding="utf-8")))
    ann = json.load(open(Path(args.asap) / "asap_annotations.json"))
    unaligned = {"{ASAP}/" + k for k, v in ann.items() if not v["score_and_performance_aligned"]}

    rows = [r for r in rows if r["source"] == "ASAP" and r["aligned"] == "True"]
    rows = [r for r in rows if r["performance_MIDI_external"] not in SKIP | unaligned]
    seen, dedup = set(), []
    for r in rows:  # drop_duplicates keeps first
        if r["performance_MIDI_external"] not in seen:
            seen.add(r["performance_MIDI_external"])
            dedup.append(r)
    ignore = set(TO_IGNORE_INDICES)  # reset_index -> positional indices
    kept = [r for i, r in enumerate(dedup) if i not in ignore]
    excluded = [r["performance_MIDI_external"] for i, r in enumerate(dedup) if i in ignore]
    excluded += sorted(SKIP)

    def split_of(pid: int) -> str:
        return "test" if pid in TEST_PIECE_IDS else "val" if pid % 10 == 0 else "train"

    pieces, conflicts = {}, []
    for r in kept + [r for i, r in enumerate(dedup) if i in ignore]:
        folder = r["performance_MIDI_external"].removeprefix("{ASAP}/").rsplit("/", 1)[0]
        sp = split_of(int(r["piece_id"]))
        if pieces.setdefault(folder, sp) != sp:
            conflicts.append(folder)
    perf_counts = Counter(split_of(int(r["piece_id"])) for r in kept)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"source": "Beyer & Dai 2024, MIDI2ScoreTransformer constants.py (TEST_PIECE_IDS etc.)",
               "pieces": dict(sorted(pieces.items())),
               "exclude_performances": sorted(p.removeprefix("{ASAP}/") for p in excluded)},
              open(out, "w"), indent=1)
    print("performances per split:", dict(perf_counts), "| excluded:", len(excluded),
          "| pieces:", Counter(pieces.values()), "| conflicts:", conflicts)


if __name__ == "__main__":
    main()
