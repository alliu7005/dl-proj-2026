"""Build (performance MIDI -> score tokens) segments from ASAP and PianoCoRe's PERiScoPe subset.

    python scripts/build_dataset.py --config configs/preprocess.yaml

Writes to ``out_dir``: {train,val,test,test_periscope}/ (memory-mapped arrays, see
src/preprocessing/store.py), vocab.json, feature_stats.json, splits.json and report.json.

* ASAP: clean Disklavier performances, bars from hand-checked beat annotations. ``test`` is
  Beyer & Dai's ASAP test set.
* PERiScoPe (``pianocore:`` in the config; download with scripts/fetch_pianocore.py): RAW Transkun V2
  transcriptions (``source: transkun_real``, never noised), bars from PianoCoRe's note alignment.
  Works that are also in ASAP inherit ASAP's split; the others are hashed into train / val /
  ``test_periscope`` by work.

Pieces are processed in parallel and written to per-piece shards first; the vocabulary is then
built from the train shards and every split is streamed into flat arrays. Nothing has to fit in RAM
at once.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.performance import load_performance  # noqa: E402
from src.preprocessing.score import load_score  # noqa: E402
from src.preprocessing.segment import SegmentConfig, segment_pair, segment_windows  # noqa: E402
from src.preprocessing.split import (assign_splits, composer_stratified_val, composition_id,  # noqa: E402
                                     load_fixed_split)
from src.preprocessing.store import segment_sizes, write_split  # noqa: E402
from src.preprocessing.vocab import Vocab  # noqa: E402

SPLITS = ("train", "val", "test", "test_periscope")
PIANOCORE_COLS = ["id", "work_key", "score_xml_path", "asap_folders"]


def load_pianocore(pc_dir: Path) -> dict[str, list[tuple[str, int, dict]]]:
    """score_xml_path -> [(rows parquet file, row index, metadata)] from scripts/fetch_pianocore.py.
    Keyed by path: PianoCoRe's score_id is not unique (all ASAP-sourced scores share one)."""
    files = sorted(pc_dir.glob("rows-*.parquet"))
    if not files:  # not downloaded: build ASAP only instead of failing
        print(f"PianoCoRe: no rows-*.parquet in {pc_dir} -> skipped "
              "(run scripts/fetch_pianocore_meta.py and scripts/fetch_pianocore.py to include it)")
        return {}
    import pyarrow.parquet as pq

    by_score = defaultdict(list)
    for f in files:
        for i, r in enumerate(pq.read_table(f, columns=PIANOCORE_COLS).to_pylist()):
            by_score[r["score_xml_path"]].append((str(f), i, r))
    return by_score


def process_pianocore_score(pc_dir: str, score_path: str, items, cfg: SegmentConfig, split: str,
                            cid: str, shard_dir: str, min_score_match: float):
    import warnings
    warnings.filterwarnings("ignore")
    import pyarrow.parquet as pq

    from src.preprocessing.pianocore import load_unfolded_score, pianocore_performance, score_midi_shift

    key = items[0][2]["work_key"]
    tables = {}

    def row(f, i):
        if f not in tables:
            tables[f] = pq.read_table(f)
        return tables[f].slice(i, 1).to_pylist()[0]

    try:
        sc = next(r for p in Path(pc_dir).glob("scores-*.parquet")
                  for r in pq.read_table(p, filters=[("score_xml_path", "=", score_path)]).to_pylist())
        measures = load_unfolded_score(sc["score_xml_bytes"], Path(sc["score_xml_path"]).suffix)
        shift, match = score_midi_shift(row(*items[0][:2])["refined_score_midi_bytes"], measures)
    except Exception as e:  # noqa: BLE001
        return None, [{"piece": key, "error": f"score: {e!r}"[:300]}], (0, 0, 0, 0), Counter()
    if match < min_score_match:
        return None, [{"piece": key, "skipped": f"score_midi_vs_xml_match_{match:.2f}"}], (0, 0, 0, 0), Counter()
    segs, reports = [], []
    for f, i, m in items:
        perf = m["id"]
        try:
            res = pianocore_performance(row(f, i), measures, shift)
            if isinstance(res, str):
                s, st = [], {"skipped": res}
            else:
                notes, pedal, tl, info = res
                meta = {"dataset": "pianocore", "piece": key, "composition": cid, "performance": perf,
                        "split": split, "source": "transkun_real"}
                seed = int(hashlib.sha1(perf.encode()).hexdigest()[:8], 16)
                s, st = segment_windows(notes, pedal, tl, measures, cfg, meta, train=split == "train", seed=seed)
                st.update(info, score_match=round(match, 3))
        except Exception as e:  # noqa: BLE001
            s, st = [], {"error": repr(e)[:300]}
        segs += s
        reports.append({"performance": perf, "dataset": "pianocore", **st})
    return _write_shard(segs, reports, split, f"pc:{score_path}", shard_dir)


def _write_shard(segs, reports, split, name, shard_dir):
    shard = Path(shard_dir) / f"{split}__{hashlib.sha1(name.encode()).hexdigest()[:12]}.pkl"
    with open(shard, "wb") as f:
        pickle.dump(segs, f, protocol=pickle.HIGHEST_PROTOCOL)
    sizes = tuple(sum(x) for x in zip(*map(segment_sizes, segs))) if segs else (0, 0, 0, 0)
    tok_counts = Counter(t for s in segs for t in s["tokens"] + s.get("prompt", [])) \
        if split == "train" else Counter()
    return str(shard), reports, sizes, tok_counts


def load_asap(root: Path, exclude: set[str]):
    with open(root / "asap_annotations.json") as f:
        ann = json.load(f)
    by_piece = defaultdict(list)
    skipped = Counter()
    with open(root / "metadata.csv", newline="") as f:
        for row in csv.DictReader(f):
            a = ann.get(row["midi_performance"])
            if row["midi_performance"] in exclude:
                skipped["excluded_by_split_file"] += 1
            elif a is None:
                skipped["no_annotation"] += 1
            elif not a.get("score_and_performance_aligned", False):
                skipped["not_aligned"] += 1
            elif not isinstance(a.get("downbeats_score_map"), list):
                skipped["no_downbeat_map"] += 1
            else:
                by_piece[row["folder"]].append((row, a))
    return by_piece, skipped


def process_piece(root: str, folder: str, items, cfg: SegmentConfig, split: str, cid: str,
                  shard_dir: str, fallback_root: str | None = None):
    import warnings
    warnings.filterwarnings("ignore")  # music21 MusicXML warnings (overfull measures, wedges, ...)
    root = Path(root)
    measures, err = None, None
    for r in [root] + ([Path(fallback_root)] if fallback_root else []):
        try:
            measures = load_score(r / items[0][0]["xml_score"])
            break
        except Exception as e:  # noqa: BLE001
            err = err or e
    if measures is None:
        return None, [{"piece": folder, "error": f"score: {err!r}"}], (0, 0, 0, 0), Counter()
    segs, reports = [], []
    for row, ann in items:
        perf = row["midi_performance"]
        try:
            notes, pedal = load_performance(root / perf)
            meta = {"dataset": "asap", "piece": folder, "composition": cid, "performance": perf,
                    "split": split, "source": "clean"}
            if cfg.mode == "windows":  # window sizes drawn reproducibly per performance
                seed = int(hashlib.sha1(perf.encode()).hexdigest()[:8], 16)
                s, st = segment_windows(notes, pedal, ann, measures, cfg, meta, train=split == "train", seed=seed)
            else:
                s, st = segment_pair(notes, pedal, ann, measures, cfg, meta)
        except Exception as e:  # noqa: BLE001
            s, st = [], {"error": repr(e)}
        segs += s
        reports.append({"performance": perf, "dataset": "asap", **st})
    return _write_shard(segs, reports, split, folder, shard_dir)


def iter_shards(paths):
    for p in sorted(paths):
        with open(p, "rb") as f:
            segs = pickle.load(f)
        yield from sorted(segs, key=lambda s: (s["performance"], s.get("win", 0), s["bar_start"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process N ASAP pieces and N PianoCoRe scores (debugging)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out = Path(cfg["asap_root"]), Path(cfg["out_dir"])
    shard_dir = out / "_shards"
    if out.exists():
        for sp in SPLITS:
            shutil.rmtree(out / sp, ignore_errors=True)
        shutil.rmtree(shard_dir, ignore_errors=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # --- what to process, and composition-level splits (decided before any segmentation) ---
    fixed, exclude = load_fixed_split(cfg.get("fixed_split"))
    by_piece, skipped = load_asap(root, exclude)
    cids = {f: composition_id("asap", f) for f in by_piece}
    pc = cfg.get("pianocore") or {}
    by_score = load_pianocore(Path(pc["dir"])) if pc.get("enabled") else {}
    pc_cid, pc_composer = {}, {}
    for sid, items in by_score.items():  # works also in ASAP share the ASAP composition id
        m = items[0][2]
        folders = sorted(f for f in (m.get("asap_folders") or "").split(";") if f)
        pc_cid[sid] = composition_id("asap", folders[0]) if folders else composition_id("pianocore", m["work_key"])
        if folders:
            pc_composer[pc_cid[sid]] = folders[0].split("/")[0]
    asap_like = set(cids.values()) | {c for s, c in pc_cid.items() if (by_score[s][0][2].get("asap_folders") or "")}
    if cfg.get("val_strategy") == "composer_stratified":
        # keep the fixed (Beyer & Dai) TEST pieces; val = ~val_frac of each composer's performances
        composer_of = {**pc_composer, **{cids[f]: f.split("/")[0] for f in by_piece}}
        splits = composer_stratified_val(asap_like, composer_of,
                                         {c for c, sp in fixed.items() if sp == "test"},
                                         cfg.get("val_frac", 0.1), seed=cfg["seed"],
                                         weight=Counter({cids[f]: len(by_piece[f]) for f in by_piece}))
    else:
        splits = assign_splits(asap_like, tuple(cfg["split_ratios"]), cfg["seed"], fixed)
    # PianoCoRe-only works (not in ASAP): hashed into train / val / test_periscope by work
    pc_only = {c for c in pc_cid.values() if c not in splits}
    for c, sp in assign_splits(pc_only, tuple(pc.get("split_ratios", (0.85, 0.05, 0.10))), cfg["seed"]).items():
        splits[c] = "test_periscope" if sp == "test" else sp
    if any(splits[c] == "test" for c in pc_cid.values()):
        raise RuntimeError("a PianoCoRe work landed in the Beyer & Dai test split")

    pieces = sorted(by_piece)[: args.limit] if args.limit else sorted(by_piece)
    scores = sorted(by_score)[: args.limit] if args.limit else sorted(by_score)

    seg = dict(cfg["segment"])
    seg["window_notes"] = tuple(seg.get("window_notes", SegmentConfig.window_notes))
    base = SegmentConfig(**seg)
    eval_cfg = replace(base, stride=base.bars_per_segment)  # windows mode: eval is handled by train=False
    shards, sizes = defaultdict(list), defaultdict(lambda: [0, 0, 0, 0])
    reports, train_counts = [], Counter()
    units = []  # (dataset, split, composition, performance ids) per task, for the summary
    with ProcessPoolExecutor(max_workers=cfg.get("workers", 4)) as ex:
        futs = []
        for f in pieces:
            sp = splits[cids[f]]
            futs.append(ex.submit(process_piece, str(root), f, by_piece[f], base if sp == "train" else eval_cfg,
                                  sp, cids[f], str(shard_dir), cfg.get("score_fallback_root")))
            units.append(("asap", sp, cids[f], [r["midi_performance"] for r, _ in by_piece[f]]))
        for s in scores:
            sp = splits[pc_cid[s]]
            futs.append(ex.submit(process_pianocore_score, pc["dir"], s, by_score[s],
                                  base if sp == "train" else eval_cfg, sp, pc_cid[s], str(shard_dir),
                                  pc.get("min_score_match", 0.9)))
            units.append(("pianocore", sp, pc_cid[s], [m["id"] for _, _, m in by_score[s]]))
        for fu in tqdm(as_completed(futs), total=len(futs), desc="pieces"):
            shard, rep, sz, tc = fu.result()
            reports += rep
            train_counts += tc
            if shard:
                sp = Path(shard).name.split("__")[0]
                shards[sp].append(shard)
                sizes[sp] = [x + y for x, y in zip(sizes[sp], sz)]

    vocab = Vocab.from_counts(train_counts, cfg.get("vocab_min_count", 2))
    vocab.save(out / "vocab.json")
    for sp in SPLITS:
        write_split(out / sp, iter_shards(shards[sp]), tuple(sizes[sp]), vocab)
    if sizes["train"][0]:  # mean/std of the continuous note features, applied by the Dataset
        from src.dataset import compute_feature_stats
        compute_feature_stats(out / "train", out / "feature_stats.json")
    if not cfg.get("keep_shards", False):
        shutil.rmtree(shard_dir, ignore_errors=True)

    json.dump({"asap": {f: splits[cids[f]] for f in sorted(by_piece)},
               "pianocore": {by_score[s][0][2]["work_key"]: splits[pc_cid[s]] for s in sorted(by_score)}},
              open(out / "splits.json", "w"), indent=1)
    ok = {r["performance"] for r in reports if r.get("segments")}
    summary = {}
    for sp in SPLITS:
        row = {"segments": sizes[sp][0], "notes": sizes[sp][1], "tokens": sizes[sp][3]}
        for ds in ("asap", "pianocore"):
            us = [u for u in units if u[0] == ds and u[1] == sp]
            row[f"{ds}_performances"] = len({p for u in us for p in u[3] if p in ok})
            row[f"{ds}_compositions"] = len({u[2] for u in us})
        summary[sp] = row
    dropped, carry, skips = Counter(), Counter(), Counter()
    for r in reports:
        for k in ("too_long", "empty", "long_bar", "low_pitch_f1"):
            dropped[f"{r.get('dataset', '?')}:{k}"] += r.get(k, 0)
        for k in ("carry_candidates", "carried_notes", "ties_at_window_start", "ties_with_carried_note"):
            carry[k] += r.get(k, 0)
        if "skipped" in r or "error" in r:
            why = r.get("skipped") or r.get("error", "")
            skips[f"{r.get('dataset', '?')}:{why.split('_0.')[0].split('(')[0][:60]}"] += 1
    errs = [r for r in reports if "error" in r or "skipped" in r]
    json.dump({"summary": summary, "skipped_at_load": skipped, "dropped_segments": dropped,
               "skipped_performances": skips, "carried_over_notes": carry,
               "vocab_size": len(vocab), "problems": errs, "performances": reports},
              open(out / "report.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("skipped at load:", dict(skipped), "| problem performances/scores:", dict(skips))
    print("dropped segments:", {k: v for k, v in dropped.items() if v}, "| vocab:", len(vocab))
    if carry["ties_at_window_start"]:
        print(f"carried-over notes: {carry['carried_notes']} key-held (+ {carry['carry_candidates'] - carry['carried_notes']}"
              f" pedal-sustained candidates) | ties into a window's first beat: "
              f"{carry['ties_at_window_start']}, {carry['ties_with_carried_note']} "
              f"({carry['ties_with_carried_note'] / carry['ties_at_window_start']:.1%}) have a carried note")


if __name__ == "__main__":
    main()
