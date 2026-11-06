"""Fit noise-model parameters by comparing transcribed MIDI against ground-truth MIDI.

    python scripts/fit_noise.py --pairs data/transcribed/pairs.csv --out configs/noise_stats.json

``pairs.csv`` (written by scripts/transcribe.py) has columns ``pred,gt[,maestro_split]``. Only rows
from the MAESTRO *test* split are used (``--maestro-split``): Kong et al.'s transcriber was trained
on MAESTRO train, so train-split transcriptions are unrealistically clean.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.noise import (NoiseConfig, NoiseModel, VEL_EDGES, attribute_source,  # noqa: E402
                                     chord_sizes, vel_bin)
from src.preprocessing.performance import load_performance  # noqa: E402

ONSET_TOL = 0.05


def match(gt: np.ndarray, pred: np.ndarray, tol: float = ONSET_TOL):
    """Greedy same-pitch onset matching. Returns list of (gt_idx, pred_idx)."""
    pairs = []
    by_pitch = defaultdict(list)
    for j, p in enumerate(pred):
        by_pitch[int(p[0])].append(j)
    used = np.zeros(len(pred), dtype=bool)
    for i, g in enumerate(gt):
        best, bd = -1, tol
        for j in by_pitch.get(int(g[0]), ()):
            d = abs(pred[j, 1] - g[1])
            if not used[j] and d <= bd:
                best, bd = j, d
        if best >= 0:
            used[best] = True
            pairs.append((i, best))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="csv with columns pred,gt (paths)")
    ap.add_argument("--chord-window", type=float, default=0.03)
    ap.add_argument("--out", default="configs/noise_stats.json")
    ap.add_argument("--maestro-split", default="test", help="only use pairs from this MAESTRO split")
    args = ap.parse_args()

    n_gt = n_pred = 0
    miss_by_size = defaultdict(lambda: [0, 0])  # chord size -> [missed, total]
    on_err, dur_ratio, vel_err = [], [], []
    pedal_ext = [0, 0]  # [extended to pedal-up, notes released under pedal]
    halluc_iv = Counter()
    halluc_vel_ratio, halluc_dur_ratio, halluc_on = [], [], []
    ped_err, ped_total, ped_missed = [], 0, 0
    drop_counts = defaultdict(lambda: [0, 0])          # "size|velbin" -> [missed, total]
    notes_per_vbin, sources_per_vbin = Counter(), Counter()
    bank, orphans = [], 0

    rows = [r for r in csv.DictReader(open(args.pairs, newline=""))
            if r.get("maestro_split", args.maestro_split) == args.maestro_split]
    print(f"fitting on {len(rows)} recordings from MAESTRO {args.maestro_split}")
    for row in tqdm(rows):
        gt, gped = load_performance(row["gt"])
        pr, pped = load_performance(row["pred"])
        # sustain pedal: match each gt press to the nearest predicted press within 150 ms
        for on, off in gped:
            ped_total += 1
            if len(pped) == 0:
                ped_missed += 1
                continue
            j = int(np.argmin(np.abs(pped[:, 0] - on)))
            if abs(pped[j, 0] - on) > 0.15:
                ped_missed += 1
            else:
                ped_err += [pped[j, 0] - on, pped[j, 1] - off]
        n_gt += len(gt)
        n_pred += len(pr)
        pairs = match(gt, pr)
        gi = np.array([a for a, _ in pairs], dtype=int)
        pj = np.array([b for _, b in pairs], dtype=int)

        # missing notes vs chord size
        grp = np.concatenate([[0], np.cumsum(np.diff(gt[:, 1]) > args.chord_window)])
        size = np.bincount(grp)[grp]
        hit = np.zeros(len(gt), dtype=bool)
        hit[gi] = True
        for s, h in zip(size, hit):
            miss_by_size[int(min(s, 8))][0] += int(not h)
            miss_by_size[int(min(s, 8))][1] += 1

        if len(pairs):
            g, p = gt[gi], pr[pj]
            on_err += list(p[:, 1] - g[:, 1])
            gd, pd = g[:, 2] - g[:, 1], p[:, 2] - p[:, 1]
            vel_err += list(p[:, 3] - g[:, 3])
            # sustain: was the gt note released while pedal down, and did pred last to pedal-up?
            for k in range(len(g)):
                off = g[k, 2]
                under = gped[(gped[:, 0] < off) & (gped[:, 1] > off)] if len(gped) else gped
                if len(under):
                    pedal_ext[1] += 1
                    if abs(p[k, 2] - under[0, 1]) < 0.1 or p[k, 2] > under[0, 1]:
                        pedal_ext[0] += 1
                else:
                    dur_ratio.append(np.log(max(pd[k], 1e-3) / max(gd[k], 1e-3)))

        # empirical drop table: chord size x velocity bin
        vb = vel_bin(gt[:, 3])
        for sz, v, h in zip(chord_sizes(gt[:, 1], args.chord_window), vb, hit):
            key = f"{min(int(sz), 8)}|{int(v)}"
            drop_counts[key][0] += int(not h)
            drop_counts[key][1] += 1
        notes_per_vbin.update(vb.tolist())
        # error bank: every unmatched prediction relative to its most likely source note
        end = gt[:, 2].copy()
        if len(gped):
            k = np.searchsorted(gped[:, 0], end, side="right") - 1
            under = (k >= 0) & (end < gped[np.maximum(k, 0), 1])
            end[under] = gped[k[under], 1]
        for j in np.setdiff1d(np.arange(len(pr)), pj):
            src = attribute_source(pr[j, 1], gt[:, 1], end, gt[:, 3])
            if src < 0:
                orphans += 1
                continue
            sources_per_vbin[int(vb[src])] += 1
            bank.append([int(pr[j, 0] - gt[src, 0]), round(float(pr[j, 1] - gt[src, 1]), 4),
                         round(float(pr[j, 3] / max(gt[src, 3], 1)), 3),
                         round(float(pr[j, 2] - pr[j, 1]), 4), int(vb[src])])

        # hallucinations: unmatched predictions, relative to closest-sounding gt note
        un = np.setdiff1d(np.arange(len(pr)), pj)
        for j in un:
            t = pr[j, 1]
            cand = np.nonzero((gt[:, 1] <= t + ONSET_TOL) & (gt[:, 2] >= t - ONSET_TOL))[0]
            if len(cand) == 0:
                halluc_iv["none"] += 1
                continue
            src = cand[np.argmin(np.abs(gt[cand, 1] - t) + 0.01 * np.abs(gt[cand, 0] - pr[j, 0]))]
            halluc_iv[str(int(pr[j, 0] - gt[src, 0]))] += 1
            halluc_vel_ratio.append(pr[j, 3] / max(gt[src, 3], 1))
            halluc_dur_ratio.append((pr[j, 2] - pr[j, 1]) / max(gt[src, 2] - gt[src, 1], 1e-3))
            halluc_on.append(t - gt[src, 1])

    sizes = sorted(miss_by_size)
    rates = np.array([miss_by_size[s][0] / max(miss_by_size[s][1], 1) for s in sizes])
    w = np.array([miss_by_size[s][1] for s in sizes], dtype=float)
    slope, base = np.polyfit(np.array(sizes) - 1, rates, 1, w=np.sqrt(w)) if len(sizes) > 1 else (0.0, rates[0])
    n_halluc = sum(halluc_iv.values())
    top_iv = {k: v / n_halluc for k, v in halluc_iv.most_common(12) if k != "none"}

    q = lambda a, lo, hi: (float(np.quantile(a, lo)), float(np.quantile(a, hi))) if len(a) else None  # noqa: E731
    cfg = NoiseConfig(
        drop_base=float(max(base, 0)),
        drop_per_chord_note=float(max(slope, 0)),
        chord_window=args.chord_window,
        halluc_rate=n_halluc / max(n_gt, 1),
        halluc_intervals=top_iv or NoiseConfig().halluc_intervals,
        halluc_vel_ratio=q(halluc_vel_ratio, 0.1, 0.9) or NoiseConfig().halluc_vel_ratio,
        halluc_dur_ratio=q(halluc_dur_ratio, 0.1, 0.9) or NoiseConfig().halluc_dur_ratio,
        halluc_onset_std=float(np.std(halluc_on)) if halluc_on else NoiseConfig().halluc_onset_std,
        onset_std=float(np.std(on_err)) if on_err else NoiseConfig().onset_std,
        offset_log_std=float(np.std(dur_ratio)) if dur_ratio else NoiseConfig().offset_log_std,
        pedal_extend_prob=pedal_ext[0] / max(pedal_ext[1], 1),
        velocity_std=float(np.std(vel_err)) if vel_err else NoiseConfig().velocity_std,
        pedal_time_std=float(np.std(np.clip(ped_err, -0.5, 0.5))) if ped_err else NoiseConfig().pedal_time_std,
        pedal_drop_prob=ped_missed / ped_total if ped_total else NoiseConfig().pedal_drop_prob,
        drop_table={k: round(m / t, 4) for k, (m, t) in sorted(drop_counts.items()) if t >= 30},
        halluc_rate_by_vel={str(v): round(sources_per_vbin[v] / max(notes_per_vbin[v], 1), 5)
                            for v in range(len(VEL_EDGES) + 1)},
        halluc_bank=bank,
    )
    NoiseModel(cfg).to_json(args.out)
    diag = {"n_gt": n_gt, "n_pred": n_pred, "matched": len(on_err),
            "miss_rate_by_chord_size": dict(zip(map(str, sizes), rates.round(4).tolist())),
            "hallucination_intervals_raw": dict(halluc_iv.most_common(25)),
            "onset_err_mean": float(np.mean(on_err)) if on_err else None,
            "pedal_presses_gt": ped_total, "pedal_missed": ped_missed,
            "bank_size": len(bank), "orphan_hallucinations": orphans,
            "bank_interval_top": Counter(r[0] for r in bank).most_common(10),
            "bank_delay_median": float(np.median([r[1] for r in bank])) if bank else None}
    Path(args.out).with_suffix(".diagnostics.json").write_text(json.dumps(diag, indent=1))
    print(json.dumps(diag, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
