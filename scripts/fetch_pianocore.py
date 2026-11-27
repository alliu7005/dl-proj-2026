"""Download the PERiScoPe (Transkun V2) part of PianoCoRe needed for training, and nothing else.

    python scripts/fetch_pianocore_meta.py      # first: metadata + Beyer & Dai split mapping
    python scripts/fetch_pianocore.py           # ~6.7k performances, a few GB streamed

Selects tier-A* PERiScoPe rows that are not Beyer & Dai test works, keeps ``--per-work`` per work,
and for each needed parquet row group reads only:
  * ``performance_midi_bytes``             raw Transkun transcription (the model input)
  * refined performance / score MIDI + alignment (only to build the performance -> score time map)
  * ``score_xml_bytes``                    once per score (the targets)

More data, without overfitting by work or by performance (all three are written to their own folder,
which scripts/build_dataset.py reads as an extra source: it never adds to the test set):
    python scripts/fetch_pianocore.py --out-name periscope_more --have periscope --total-per-work 30
    python scripts/fetch_pianocore.py --source ATEPP,GiantMIDI-Piano --out-name atepp --have periscope,periscope_more --total-per-work 30
--have: the folders already fetched (their performances are not fetched again, and count towards
the cap); --total-per-work: at most this many performances of a work over all of them together;
different performers first (a second recording by the same pianist only when a work has no
other). ATEPP and GiantMIDI-Piano rows are then given their recordings and our own transcriptions
(scripts/fetch_pianocore_audio.py, scripts/retranscribe_pianocore.py).

Writes ``<out>/<out-name>/rows-<shard>-<rg>.parquet`` and ``<out>/periscope/scores-<shard>-<rg>*.parquet``
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
             "bd_split", "asap_folders", "performance_dataset", "capture_model", "performer",
             "performance_duration", "refined_recall"]


def select(rows: pd.DataFrame, per_work: int, total_per_work: int, have_per_work: dict, seed: int) -> pd.DataFrame:
    """Per work, in a reproducible random order with different performers first (a pianist's second
    recording of a work only after every other pianist's first): at most ``per_work`` of these, and
    at most ``total_per_work`` counting the ``have_per_work`` already fetched (0: no total cap)."""
    rows = rows.sample(frac=1, random_state=seed).copy()
    performer = rows["performer"] if "performer" in rows else pd.Series([None] * len(rows), index=rows.index)
    key = performer.fillna("").astype(str)
    key = key.where(key != "", rows["id"].astype(str))          # unknown performer: each is its own
    rows["_nth"] = rows.assign(_k=key).groupby(["work_key", "_k"]).cumcount()
    rows = rows.sort_values(["work_key", "_nth"], kind="stable")
    rows["_rank"] = rows.groupby("work_key").cumcount()
    room = rows["work_key"].map(lambda w: max(0, total_per_work - have_per_work.get(w, 0)) if total_per_work else per_work)
    keep = (rows["_rank"] < per_work) & (rows["_rank"] < room)
    return rows[keep].drop(columns=["_nth", "_rank"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", default="data/raw/pianocore")
    ap.add_argument("--source", default="PERiScoPe", help="performance_dataset(s) to keep (comma-separated)")
    ap.add_argument("--per-work", type=int, default=10, help="performances per work from this fetch")
    ap.add_argument("--out-name", default="periscope", help="the folder under --meta-dir")
    ap.add_argument("--have", default="", help="folders already fetched (comma-separated): not fetched again, counted in the cap")
    ap.add_argument("--total-per-work", type=int, default=0,
                    help="at most this many performances per work over --have and this fetch (0: --per-work alone)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-row-groups", type=int, default=0,
                    help="only fetch this many row groups (a quick test download; 0 = all)")
    args = ap.parse_args()
    meta_dir = Path(args.meta_dir)
    out = meta_dir / args.out_name
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(meta_dir / "metadata.parquet")
    if "work_key" not in meta:
        meta["work_key"] = [work_key(*t) for t in zip(meta["composer"], meta["composition"], meta["movement"])]
    sources = [x for x in args.source.split(",") if x]
    sel = meta[meta["tier_a_star"] & meta["is_refined"] & meta["performance_dataset"].isin(sources)
               & (meta["bd_split"] != "test") & meta["score_xml_path"].notna()]
    have_ids, have_per_work, have_performers = set(), {}, {}
    for h in [x for x in args.have.split(",") if x]:
        f = meta_dir / h / "selection.parquet"
        if not f.exists():
            raise SystemExit(f"--have {h}: no {f}")
        hs = pd.read_parquet(f, columns=["id", "work_key"])
        have_ids |= set(hs["id"])
        for w, n in hs["work_key"].value_counts().items():
            have_per_work[w] = have_per_work.get(w, 0) + int(n)
    sel = sel[~sel["id"].isin(have_ids)]
    sel = select(sel, args.per_work, args.total_per_work, have_per_work, args.seed)
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
