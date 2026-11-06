"""Download PianoCoRe metadata only (~35 MB of the 8.6 GB release) and report what is usable.

    python scripts/fetch_pianocore_meta.py

Reads just the metadata columns of every Hugging Face parquet shard over HTTP and writes
``<out>/metadata.parquet`` (one row per performance, plus its shard / row group / row position so
later scripts can fetch MIDI bytes for selected rows only) and ``<out>/row_groups.parquet``
(compressed size of the refined-MIDI columns per row group, for download budgeting).

Each PianoCoRe work is mapped to ASAP folders through the ASAP performances PianoCoRe contains,
and from there to Beyer & Dai's split (``bd_split``: test / train / val, or empty for works that
are not in ASAP). Then prints a funnel: all -> scored -> tier A -> A* -> non-ASAP sources ->
not a Beyer & Dai test piece -> capped performances per work.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.pianocore import (ALIGN_COLS, META_COLS, hf_fs, map_works_to_asap,  # noqa: E402
                                         open_shard, shard_paths, work_key)
from src.preprocessing.split import composition_id, load_fixed_split  # noqa: E402

ASAP_META_URL = ("https://raw.githubusercontent.com/TimFelixBeyer/asap-dataset/"
                 "8cba199e15931975542010a7ea2ff94a6fc9cbee/metadata.csv")


def read_asap_metadata(local_root: str) -> list[dict]:
    p = Path(local_root) / "metadata.csv"
    text = p.read_text(encoding="utf-8") if p.exists() else \
        urllib.request.urlopen(ASAP_META_URL).read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def fetch(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fs = hf_fs()
    metas, rgs = [], []
    for path in tqdm(shard_paths(fs), desc="shards"):
        pf = open_shard(fs, path)
        md = pf.metadata
        df = pf.read(columns=META_COLS).to_pandas()
        sizes = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
        df["shard"] = path.rsplit("/", 1)[-1]
        df["row_group"] = np.repeat(np.arange(len(sizes)), sizes)
        df["row_in_group"] = np.concatenate([np.arange(n) for n in sizes])
        metas.append(df)
        names = {md.schema.column(j).path: j for j in range(md.num_columns)}
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            rgs.append({"shard": df["shard"].iat[0], "row_group": i, "rows": rg.num_rows,
                        "align_bytes": sum(rg.column(names[c]).total_compressed_size for c in ALIGN_COLS)})
    meta, rg = pd.concat(metas, ignore_index=True), pd.DataFrame(rgs)
    out.mkdir(parents=True, exist_ok=True)
    rg.to_parquet(out / "row_groups.parquet")
    return meta, rg


def funnel_row(name, df):
    return {"step": name, "performances": len(df), "works": df["work_key"].nunique(),
            "hours": round(df["performance_duration"].sum() / 3600)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/pianocore")
    ap.add_argument("--asap-root", default="data/raw/asap_bd", help="uses metadata.csv here if present")
    ap.add_argument("--split-file", default="configs/splits/beyer_dai.json")
    ap.add_argument("--caps", type=int, nargs="+", default=[5, 10, 20], help="performances per work")
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    if (out / "metadata.parquet").exists() and not args.refetch:
        meta = pd.read_parquet(out / "metadata.parquet")
    else:
        meta, _ = fetch(out)
    meta["work_key"] = [work_key(*t) for t in zip(meta["composer"], meta["composition"], meta["movement"])]

    # PianoCoRe work -> ASAP folders -> Beyer & Dai split (test wins if a work maps to several)
    fixed, _ = load_fixed_split(args.split_file)
    work_folders = map_works_to_asap(meta, read_asap_metadata(args.asap_root))
    rank = {"test": 0, "val": 1, "train": 2}

    def bd_split(key):
        sp = [fixed.get(composition_id("asap", f), "train") for f in work_folders.get(key, [])]
        return min(sp, key=rank.get) if sp else ""
    splits = {k: bd_split(k) for k in meta["work_key"].unique()}
    meta["asap_folders"] = meta["work_key"].map(lambda k: ";".join(work_folders.get(k, [])))
    meta["bd_split"] = meta["work_key"].map(splits)
    meta.to_parquet(out / "metadata.parquet")

    a = meta[meta["tier_a_star"]]
    usable = a[(a["performance_dataset"] != "ASAP") & (a["bd_split"] != "test")]
    steps = [funnel_row("all rows (PianoCoRe-C)", meta),
             funnel_row("has a score", meta[meta["score_xml_path"].notna()]),
             funnel_row("tier A (note-aligned)", meta[meta["tier_a"]]),
             funnel_row("tier A* (best alignments)", a),
             funnel_row("  minus ASAP performances", a[a["performance_dataset"] != "ASAP"]),
             funnel_row("  minus Beyer & Dai test works", usable)]
    for cap in args.caps:
        capped = usable.groupby("work_key").head(cap)
        steps.append(funnel_row(f"  capped at {cap} per work", capped))
    print(pd.DataFrame(steps).to_string(index=False))

    by_src = usable.groupby(["performance_dataset", "capture_model"]).agg(
        performances=("id", "size"), works=("work_key", "nunique"),
        hours=("performance_duration", lambda x: round(x.sum() / 3600)))
    print("\nusable A* by source:\n" + by_src.to_string())
    in_asap = usable["bd_split"].ne("")
    print(f"\nusable works already in ASAP (train/val): {usable.loc[in_asap, 'work_key'].nunique()}"
          f" | new works not in ASAP: {usable.loc[~in_asap, 'work_key'].nunique()}")
    print(f"ASAP works matched in PianoCoRe: {len(work_folders)} -> "
          f"{len({f for v in work_folders.values() for f in v})} ASAP folders; "
          f"B&D test works found: {sum(v == 'test' for v in splits.values())}")
    json.dump({"funnel": steps, "work_to_asap": work_folders}, open(out / "funnel.json", "w"), indent=1)
    print("wrote", out / "metadata.parquet")


if __name__ == "__main__":
    main()
