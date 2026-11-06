"""Download the PERiScoPe (Transkun V2) part of PianoCoRe needed for training, and nothing else.

    python scripts/fetch_pianocore_meta.py      # first: metadata + Beyer & Dai split mapping
    python scripts/fetch_pianocore.py           # ~6.7k performances, a few GB streamed

Selects tier-A* PERiScoPe rows that are not Beyer & Dai test works, keeps ``--per-work`` per work,
and for each needed parquet row group reads only:
  * ``performance_midi_bytes``             raw Transkun transcription (the model input)
  * refined performance / score MIDI + alignment (only to build the performance -> score time map)
  * ``score_xml_bytes``                    once per score (the targets)

Writes ``<out>/periscope/rows-<shard>-<rg>.parquet`` and ``<out>/periscope/scores-<shard>-<rg>*.parquet``
(per row group, so an interrupted run resumes; scores keyed by ``score_xml_path``, since
``score_id`` is not unique) and ``<out>/periscope/selection.parquet``. Re-running also fetches any
scores missing from row groups already downloaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.pianocore import FETCH_COLS, HF_REPO, hf_fs, open_shard, work_key  # noqa: E402

KEEP_META = ["id", "work_key", "composer", "composition", "movement", "score_id", "score_xml_path",
             "bd_split", "asap_folders", "performance_dataset", "capture_model",
             "performance_duration", "refined_recall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", default="data/raw/pianocore")
    ap.add_argument("--source", default="PERiScoPe", help="performance_dataset to keep")
    ap.add_argument("--per-work", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-row-groups", type=int, default=0,
                    help="only fetch this many row groups (a quick test download; 0 = all)")
    args = ap.parse_args()
    meta_dir = Path(args.meta_dir)
    out = meta_dir / "periscope"
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(meta_dir / "metadata.parquet")
    if "work_key" not in meta:
        meta["work_key"] = [work_key(*t) for t in zip(meta["composer"], meta["composition"], meta["movement"])]
    sel = meta[meta["tier_a_star"] & meta["is_refined"] & (meta["performance_dataset"] == args.source)
               & (meta["bd_split"] != "test") & meta["score_xml_path"].notna()]
    # random (reproducible) performances per work, not the first N in file order
    sel = sel.sample(frac=1, random_state=args.seed).groupby("work_key").head(args.per_work)
    sel = sel.sort_values(["shard", "row_group", "row_in_group"])
    sel.to_parquet(out / "selection.parquet")

    # Scores are keyed by score_xml_path: score_id is NOT unique (every ASAP-sourced score has
    # score_id "ASAP_midi_score").
    seen_paths = set()
    for p in out.glob("scores-*.parquet"):
        seen_paths |= set(pq.read_table(p, columns=["score_xml_path"]).column(0).to_pylist())
    rg = pd.read_parquet(meta_dir / "row_groups.parquet").set_index(["shard", "row_group"])
    groups = list(sel.groupby(["shard", "row_group"]))
    have_rows = {k for k, _ in groups if (out / f"rows-{k[0]}-{k[1]}.parquet").exists()}
    todo = [(k, g) for k, g in groups if k not in have_rows]
    if args.max_row_groups:
        todo = todo[: max(args.max_row_groups - len(have_rows), 0)]
    # row groups already downloaded whose scores are missing (e.g. from the score_id bug)
    repair = [(k, g) for k, g in groups if k in have_rows and not set(g["score_xml_path"]) <= seen_paths]
    # FETCH_COLS ~= refined columns (row_groups.parquet) + the raw performance (~1.2x the refined MIDI)
    est = sum(rg.loc[k, "align_bytes"] for k, _ in todo) * 2.3 / 1e6
    print(f"{len(sel)} performances, {sel['work_key'].nunique()} works, {sel['score_xml_path'].nunique()} scores, "
          f"{sel['performance_duration'].sum() / 3600:.0f} h | {len(groups)} row groups, "
          f"{len(todo)} to fetch (~{est:.0f} MB), {len(repair)} to repair (scores only)")

    fs, streamed = hf_fs(), 0
    for (shard, rgi), g in tqdm(sorted(todo + repair, key=lambda kg: kg[0]), desc="row groups"):
        need_rows = (shard, rgi) not in have_rows
        missing = g[~g["score_xml_path"].isin(seen_paths) & ~g["score_xml_path"].duplicated()]
        cols = (FETCH_COLS if need_rows else []) + (["score_xml_bytes"] if len(missing) else [])
        if not cols:
            continue
        tbl = open_shard(fs, f"{HF_REPO}/data/{shard}").read_row_group(int(rgi), columns=cols)
        streamed += tbl.nbytes
        if len(missing):
            scores = pa.Table.from_pandas(missing[["score_id", "score_xml_path"]].reset_index(drop=True),
                                          preserve_index=False)
            scores = scores.append_column("score_xml_bytes", tbl.column("score_xml_bytes").cast(pa.large_binary())
                                          .take(pa.array(missing["row_in_group"].to_numpy())))
            n = len(list(out.glob(f"scores-{shard}-{rgi}*.parquet")))
            pq.write_table(scores, out / f"scores-{shard}-{rgi}{'' if n == 0 else f'-{n}'}.parquet")
            seen_paths |= set(missing["score_xml_path"])
        if need_rows:
            idx = pa.array(g["row_in_group"].to_numpy())
            rows = pa.Table.from_pandas(g[KEEP_META].reset_index(drop=True), preserve_index=False)
            for c in FETCH_COLS:
                rows = rows.append_column(c, tbl.column(c).cast(pa.large_binary()).take(idx))
            pq.write_table(rows, out / f"rows-{shard}-{rgi}.parquet")  # written last = row group done
    kept = sum(p.stat().st_size for p in out.glob("*.parquet")) / 1e6
    print(f"streamed ~{streamed / 1e6:.0f} MB (decompressed) | kept {kept:.0f} MB in {out}")


if __name__ == "__main__":
    main()
