"""Do PianoCoRe's transcribed performances look like clean (Disklavier) performances?

    python scripts/fetch_pianocore_meta.py        # first: metadata + row-group sizes
    python scripts/compare_pianocore.py           # ~0.5-1 GB streamed, nothing large kept on disk

Our training inputs are clean performance MIDI + synthetic Kong noise. PianoCoRe performances
outside ASAP were transcribed from audio by other models (Aria-AMT, Transkun V2, ATEPP, ByteDance),
so using them would stack two noise sources. This script measures how much of that other noise
survives in the *refined* data (extra notes removed, missed notes interpolated), per source, against
PianoCoRe's own ASAP Disklavier performances processed identically.

Per performance, on aligned non-interpolated notes:
  timing    residual of each score position's onset against a local tempo fit (ms), spread of
            onsets within written chords (ms), fraction of chord notes with identical onsets
  durations log2(performed / expected) for notes released with the pedal up, where expected =
            written length x local seconds-per-beat; fraction of very short notes
  pedal     fraction of time held down, presses per minute
  velocity  mean, std, spread within chords
From metadata (all of A*, not just the sample): extra-note and missing-note rates of the raw
transcription relative to the score.

Writes outputs/pianocore_compare/{per_performance.csv, summary.md, summary.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.pianocore import ALIGN_COLS, HF_REPO, hf_fs, load_aligned, open_shard  # noqa: E402

REF = "ASAP"
WINDOW = 4  # score positions on each side for the local tempo fit
METRICS = {  # name -> (description, unit)
    "timing_resid_ms": ("onset deviation from local tempo (median |res|)", "ms"),
    "chord_spread_ms": ("onset spread within written chords (median)", "ms"),
    "chord_identical": ("chord notes with identical onsets", "frac"),
    "dur_log2_med": ("note length vs written, pedal up (median log2)", "log2"),
    "dur_log2_iqr": ("note length vs written, pedal up (IQR log2)", "log2"),
    "short_notes": ("notes shorter than 30 ms", "frac"),
    "released_under_pedal": ("notes released while pedal down", "frac"),
    "pedal_down": ("time with sustain pedal down", "frac"),
    "pedal_per_min": ("pedal presses per minute", "/min"),
    "vel_mean": ("velocity mean", "0-127"),
    "vel_std": ("velocity std", "0-127"),
    "chord_vel_std": ("velocity spread within chords (median std)", "0-127"),
}
META_METRICS = {
    "extra_rate": "raw notes removed as unmatched (extra)",
    "missing_rate": "score notes with no performed note (interpolated)",
}


def performance_stats(d: dict) -> dict:
    keep = ~d["interpolated"]
    beat, dur_b = d["beat"][keep], d["dur_beats"][keep]
    on, off, vel = d["onset"][keep], d["offset"][keep], d["velocity"][keep]
    if len(on) < 50:
        raise ValueError("too few aligned notes")
    df = pd.DataFrame({"beat": beat, "on": on, "vel": vel})

    # score positions -> median performed onset; local linear tempo fit excluding the point itself
    pos = df.groupby("beat")["on"].median()
    u, t = pos.index.to_numpy(), pos.to_numpy()
    resid, spb = np.full(len(u), np.nan), np.full(len(u), np.nan)
    for k in range(len(u)):
        nb = np.r_[max(0, k - WINDOW):k, k + 1:min(len(u), k + WINDOW + 1)]
        if len(nb) < 3 or np.ptp(u[nb]) == 0:
            continue
        slope, icpt = np.polyfit(u[nb], t[nb], 1)
        if slope <= 0:
            continue
        resid[k], spb[k] = t[k] - (slope * u[k] + icpt), slope

    g = df.groupby("beat")
    chords = g.filter(lambda x: len(x) >= 2)
    spread = chords.groupby("beat")["on"].agg(np.ptp)
    ident = chords.assign(r=np.round(chords["on"] * 2000)).duplicated(["beat", "r"], keep=False)

    # expected duration = written quarters x local seconds per quarter (at the note's position)
    spb_note = pd.Series(spb, index=u).reindex(beat).to_numpy()
    expected = dur_b * spb_note
    ped = d["pedal"]
    under = np.zeros(len(off), dtype=bool)
    if len(ped):
        j = np.searchsorted(ped[:, 0], off) - 1
        under = (j >= 0) & (off < ped[np.clip(j, 0, None), 1])
    ok = (~under) & np.isfinite(expected) & (expected > 0.02)
    ratio = np.log2(np.maximum(off - on, 1e-3)[ok] / expected[ok])
    length = max(float(off.max() - on.min()), 1.0)
    q = lambda a, p: float(np.nanpercentile(a, p)) if np.isfinite(a).any() else np.nan  # noqa: E731
    return {
        "timing_resid_ms": 1000 * q(np.abs(resid), 50),
        "chord_spread_ms": 1000 * float(spread.median()) if len(spread) else np.nan,
        "chord_identical": float(ident.mean()) if len(ident) else np.nan,
        "dur_log2_med": q(ratio, 50) if len(ratio) else np.nan,
        "dur_log2_iqr": q(ratio, 75) - q(ratio, 25) if len(ratio) else np.nan,
        "short_notes": float(np.mean(off - on < 0.03)),
        "released_under_pedal": float(under.mean()),
        "pedal_down": float(np.clip(ped[:, 1] - ped[:, 0], 0, None).sum() / length) if len(ped) else 0.0,
        "pedal_per_min": 60 * len(ped) / length,
        "vel_mean": float(vel.mean()), "vel_std": float(vel.std()),
        "chord_vel_std": float(chords.groupby("beat")["vel"].std().median()) if len(chords) else np.nan,
    }


def choose_sample(a: pd.DataFrame, rg: pd.DataFrame, per_source: int, per_work: int,
                  budget_mb: float, seed: int) -> pd.DataFrame:
    """Greedy over row groups in random order until every source has ``per_source`` rows
    (max ``per_work`` per work), within the download budget."""
    rng = np.random.default_rng(seed)
    rg = rg.set_index(["shard", "row_group"])["align_bytes"]
    groups = list(a.groupby(["shard", "row_group"]))
    order = rng.permutation(len(groups))
    taken, per_src, per_wk, used = [], {}, {}, 0.0
    sources = set(a["performance_dataset"])
    for i in order:
        if all(per_src.get(s, 0) >= per_source for s in sources) or used >= budget_mb:
            break
        key, rows = groups[i]
        pick = []
        for idx, r in rows.sample(frac=1, random_state=int(rng.integers(1 << 31))).iterrows():
            s, w = r["performance_dataset"], r["work_key"]
            if per_src.get(s, 0) < per_source and per_wk.get(w, 0) < per_work:
                pick.append(idx)
                per_src[s] = per_src.get(s, 0) + 1
                per_wk[w] = per_wk.get(w, 0) + 1
        if pick:
            taken += pick
            used += rg[key] / 1e6
    print(f"sample: {per_src} from {len(set(a.loc[taken, ['shard', 'row_group']].itertuples(index=False)))}"
          f" row groups, ~{used:.0f} MB to stream")
    return a.loc[taken]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", default="data/raw/pianocore")
    ap.add_argument("--out", default="outputs/pianocore_compare")
    ap.add_argument("--per-source", type=int, default=150)
    ap.add_argument("--per-work", type=int, default=3)
    ap.add_argument("--budget-mb", type=float, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(Path(args.meta_dir) / "metadata.parquet")
    rg = pd.read_parquet(Path(args.meta_dir) / "row_groups.parquet")
    a = meta[meta["tier_a_star"] & meta["is_refined"]].copy()
    a["extra_rate"] = ((a["performance_note_count"] - (a["refined_performance_note_count"]
                        - a["refined_performance_interpolated_note_count"])) / a["performance_note_count"]).clip(lower=0)
    a["missing_rate"] = a["refined_performance_interpolated_note_count"] / a["refined_performance_note_count"]

    # per-performance stats (resumable: rows already in per_performance.csv are skipped)
    csv_path = out / "per_performance.csv"
    done = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(columns=["id"])
    sample = choose_sample(a, rg, args.per_source, args.per_work, args.budget_mb, args.seed)
    todo = sample[~sample["id"].isin(set(done["id"]))]
    fs, rows = hf_fs(), done.to_dict("records")
    for (shard, rgi), grp in tqdm(list(todo.groupby(["shard", "row_group"])), desc="row groups"):
        tbl = open_shard(fs, f"{HF_REPO}/data/{shard}").read_row_group(int(rgi), columns=ALIGN_COLS)
        for _, r in grp.iterrows():
            rec = {"id": r["id"], "source": r["performance_dataset"], "work_key": r["work_key"]}
            try:
                d = load_aligned(*(tbl.column(c)[int(r["row_in_group"])].as_py() for c in ALIGN_COLS))
                rec.update(performance_stats(d))
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)[:200]
            rows.append(rec)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    per = pd.DataFrame(rows)
    per = per[per["id"].isin(set(sample["id"]))]
    n_err = int(per.get("error", pd.Series(dtype=str)).notna().sum())
    per = per[per.get("error", pd.Series(index=per.index, dtype=str)).isna()]

    # summary: median [IQR] per source; KS distance vs ASAP Disklavier; flag medians outside ASAP IQR
    sources = [REF] + sorted(set(per["source"]) - {REF})
    lines, summary = [], {"n": per["source"].value_counts().to_dict(), "errors": n_err, "metrics": {}}
    hdr = "| metric | " + " | ".join(f"{s} (n={(per['source'] == s).sum()})" for s in sources) + " |"
    lines += [hdr, "|" + "---|" * (len(sources) + 1)]

    def row(name, desc, vals_by_src):
        ref = vals_by_src[REF].dropna()
        lo, hi = ref.quantile(0.25), ref.quantile(0.75)
        cells, rec = [], {}
        for s in sources:
            v = vals_by_src[s].dropna()
            if not len(v):
                cells.append("–")
                continue
            m = v.median()
            ks = ks_2samp(v, ref).statistic if s != REF else 0.0
            flag = " ⚠" if s != REF and not (lo <= m <= hi) else ""
            cells.append(f"{m:.3g} [{v.quantile(.25):.3g}–{v.quantile(.75):.3g}]"
                         + (f" KS {ks:.2f}" if s != REF else "") + flag)
            rec[s] = {"median": m, "q25": v.quantile(.25), "q75": v.quantile(.75), "ks_vs_asap": ks,
                      "outside_asap_iqr": bool(flag)}
        lines.append(f"| {desc} | " + " | ".join(cells) + " |")
        summary["metrics"][name] = rec

    for k, (desc, unit) in METRICS.items():
        row(k, f"{desc} ({unit})", {s: per.loc[per["source"] == s, k].astype(float) for s in sources})
    lines.append("| *from metadata, all A\\* rows* |" + " |" * len(sources))
    for k, desc in META_METRICS.items():
        row(k, desc, {s: a.loc[a["performance_dataset"] == s, k].astype(float) for s in sources})

    md = ("# PianoCoRe refined performances vs ASAP Disklavier\n\n"
          "Median [IQR] over performances. KS = Kolmogorov-Smirnov distance to the ASAP distribution "
          "(0 = identical, 1 = disjoint). ⚠ = median outside ASAP's IQR.\n\n" + "\n".join(lines) + "\n")
    (out / "summary.md").write_text(md, encoding="utf-8")
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    print(md)
    print(f"{len(per)} performances analysed, {n_err} failed -> {out}")


if __name__ == "__main__":
    main()
