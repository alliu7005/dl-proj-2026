"""Evaluate a meter model (v2, Phase 2) and save its grids for the note labeller (Phase 3).

    python scripts/eval_meter.py --run meter_v2 --tune      # tune the decoder on val, then every test set
    python scripts/eval_meter.py --run meter_v2             # every test set (decode.json if tuned before)

For every test performance: the tracked beats, the model's decoded grid (src/meter.py
predict_logp + decode) and its scores against the ground truth (grid_scores: beat F, downbeat F,
time-signature accuracy) next to the tracker's own beat / downbeat F on the same beats.
Conditions:
  clean      ASAP test MIDI, the tracker's errors simulated (a fixed draw per performance, as the
             seq2seq evaluation's simulated-tracker condition), scored against the annotations
  real       the Transkun transcriptions of the ASAP test audio with Beat This!'s own beats
  periscope  PERiScoPe test (104 works no model trained on): real transcriptions, the tracker
             simulated on, and scored against, the score's beats timed through the note alignment
             (scripts/build_meter_data.py; within 70 ms of ASAP's human annotations for 81% of
             beats, the same time signature for >99.9%), one grid per segment of the performance
  gangqinpu  Gangqinpu test (build_meter_data.py --gangqinpu): web piano recordings of many genres,
             real transcriptions, the tracker simulated on the score's beats timed through the
             release's alignment: music outside the classical benchmark (skipped when not built)

--tune: the decoder's weights and penalties (src/meter.py DecodeConfig: bar / time-signature head
weights, the time-signature change and mid-step bar line penalties) chosen on the validation
performances (ASAP val, real val, --tune-periscope PERiScoPe val) by coordinate search on the
training script's selection score (mean of downbeat F and time-signature accuracy), saved to
outputs/runs/<run>/decode.json and used from then on. Every condition is also scored with the
default decoder (``default`` in the results), so the tuning's effect is on record.

--level: the piece's note-value level after decoding (src/meter.py choose_level: the grid, or the
same bar lines at halved / doubled note values, whichever its written tempo fits best), its weights
(a log-normal prior on the written tempo, the model's own estimate with a tempo head, a bonus for
the decoded level) tuned on validation after the decoder's other settings: on their own with
--decode-from <decode.json> (the other settings taken from there, e.g. a run sharing the
checkpoint), else after --tune's. Every condition reports level_ok: the share of performances at
the right level (decoded quarters / true quarters within LEVEL_TOL of 1 in log2).

--oracle: no model; each head certain of its training target on the same tracked beats (src/meter.py
oracle_logp), decoded alike: the best grid these beats allow (``python scripts/eval_meter.py --run
meter_oracle --oracle``). The labeller on it (scripts/eval_labeller.py --grid meter:meter_oracle)
separates what the tracker's beats cost from what the meter model's errors cost.

--snap S / --fill F: the decoded grid's beat times taken from the notes (src/meter.py refine_grid:
an anchor onto a chord onset within S seconds; a notated beat no tracked beat times added at a chord
onset within F beats of where the anchors around it put it), saved into the run's decode.json with
the other settings (0 turns one off), so everything that decodes with that file (the labeller's
grids, scripts/system_scores.py, the demo) gets the same grids. A run sharing another's checkpoint
(pace/v2_jobs.sh refine) compares them on the same model. ``refined`` in the results: anchors
moved and added per performance.

--tracker pm2s: every performance's beats from PM2S on its notes (src/pm2s_beats.py) instead of
the simulated / recorded Beat This! ones, every condition then under <condition>_pm2s in the results
and the file names (the Beat This! results stay). With a model trained for either tracker, the 2 x 2
of tracker in training and at test. PM2S's released weights were trained on all of ASAP (clean /
real here: optimistic); PERiScoPe and Gangqinpu are music it did not train on.

--real-tracked: Beat This!'s real beats where scripts/track_meter_data.py stored them for the
condition (Gangqinpu's test recordings: meter/tracked_test_gangqinpu_beat_this.npz), only those
performances, under <condition>_real: with --tracker pm2s on the same condition, the two real
trackers on the same music.

A model that reads a second tracker (train_meter.py --aux pm2s) gets PM2S's beats of each
performance's notes as that evidence everywhere here (tuning and every condition), as in the
pipeline: "aux": "pm2s" in its results.

Writes outputs/runs/<run>/eval_meter.json (means per condition and per piece, the time-signature
confusions), meter_scores_<condition>.json (per performance) and grids_<condition>.pkl ({key:
grid, with its performance, its span and the key signature estimated from its notes:
src/notelabels.py estimate_key}), which scripts/eval_labeller.py --grid meter:<run> reads.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import pick_device  # noqa: E402
from scripts.train_meter import load  # noqa: E402
from src.beats import MATCH_TOL, BeatNoise, f_measure, load_beats  # noqa: E402
from src.meter import (DecodeConfig, build_meter_model, choose_level, cover_notes, grid_from_logp,  # noqa: E402
                       grid_scores, oracle_logp, perf_key, predict_logp, refine_grid, uses_aux)
from src.notelabels import estimate_key  # noqa: E402

CONDITIONS = {"clean": "test", "real": "test_transcribed", "periscope": "test_periscope",
              "gangqinpu": "test_gangqinpu"}
DECODER_VERSION = 2  # 2: a step longer than a bar keeps the bars it passes (src/meter.py grid_from_states)
VAL = {"asap": "val", "real": "val_transcribed", "periscope": "val_periscope"}
SEED = 12345  # the simulated tracker's draw on the clean performances (fixed: comparable runs)
SEARCH = {  # coordinate search over the decoder's settings (each list holds the default)
    "bar_weight": [0.5, 1.0, 2.0, 4.0],
    "ts_weight": [0.25, 0.5, 1.0, 2.0],
    "ts_change": [math.log(x) for x in (1e-1, 1e-2, 1e-3, 1e-5, 1e-8, 1e-12)],
    "mid_step_bar": [math.log(x) for x in (0.3, 1e-1, 1e-2, 1e-3)],
}
TEMPO_SEARCH = {  # with --tempo-prior: the prior on the written tempo (src/meter.py tempo_logprior)
    "tempo_weight": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    "tempo_mu": [math.log(x) for x in (80.0, 100.0, 116.0, 140.0)],
}
LOCAL_SEARCH = {  # with --local-tempo: local tempo consistency (src/meter.py local_tempo_logprior)
    "local_weight": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    "local_sigma": [0.15, 0.25, 0.4],
}
PIECE_SEARCH = {  # with --piece-tempo: the piece's own tempo (src/meter.py piece_tempo_logprior)
    "piece_weight": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
    "piece_sigma": [0.1, 0.15, 0.25, 0.35, 0.5],
}
VAL_GANGQINPU = {"gangqinpu": "val_gangqinpu"}   # --local-tempo / --piece-tempo also tune on Gangqinpu's validation
LEVEL_SEARCH = {  # with --level: the piece's note-value level (src/meter.py choose_level)
    "level_weight": [0.0, 0.25, 0.5, 1.0, 2.0],
    "level_model_weight": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],   # only with a tempo head
    "level_stay": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
}


def key_of(perf: dict) -> str:
    """A grid's key in grids_<condition>.pkl: the performance, #segment when it has several."""
    return perf_key(perf)


def tracked(perf: dict, k: int, noise, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """The beats the system gets for a performance: the real tracker's, else the simulated one."""
    if perf.get("beats") is not None:
        return np.asarray(perf["beats"][0], np.float64), np.asarray(perf["beats"][1], bool)
    gt = perf["gt"]
    bt, bd = noise(gt["t"], gt["down"], np.random.default_rng((seed, k)))
    return np.asarray(bt, np.float64), np.asarray(bd, bool)


def score(perfs, lps, beats, cfg: DecodeConfig) -> list:
    return [grid_scores(grid_from_logp(bt, lp, cfg), p["gt"]) for p, lp, (bt, _) in zip(perfs, lps, beats)]


def nanmean(xs) -> float:
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def tune_level(groups: list, cfg: DecodeConfig, has_head: bool, rounds: int = 2) -> tuple[DecodeConfig, dict]:
    """Coordinate search of the level settings on the validation performances: each one decoded
    once with ``cfg`` (its level weights off), then each candidate setting choosing the level of
    those grids (cheap: no decoding). Scored by the selection score (the level moves the time
    signatures), ties to the decoded level; level_ok on record."""
    base = DecodeConfig(**{**cfg.to_dict(), "level_weight": 0.0, "level_model_weight": 0.0, "level_stay": 0.0})
    decoded = [[(grid_from_logp(bt, lp, base), lp, p["gt"]) for p, lp, (bt, _) in zip(perfs, lps, beats)]
               for _, perfs, lps, beats in groups]
    cache = {}

    def scores(c: DecodeConfig):
        key = tuple(sorted(c.to_dict().items()))
        if key not in cache:
            cache[key] = [[grid_scores(choose_level(g, c, lp.get("tempo") if lp else None), gt) for g, lp, gt in grp]
                          for grp in decoded]
        return cache[key]

    def value(c):
        return select(scores(c))
    search = {k: v for k, v in LEVEL_SEARCH.items() if has_head or k != "level_model_weight"}
    cur = base
    start = value(cur)
    for _ in range(rounds):
        for name, options in search.items():
            best = max(options, key=lambda v: (value(DecodeConfig(**{**cur.to_dict(), name: v})),
                                               -abs(v - getattr(DecodeConfig(), name))))
            cur = DecodeConfig(**{**cur.to_dict(), name: best})

    def level_ok(c):
        return nanmean([s["level_ok"] for grp in scores(c) for s in grp])
    info = {"level_select_decoded": start, "level_select_tuned": value(cur), "level_ok_decoded": level_ok(base),
            "level_ok_tuned": level_ok(cur), "level_configs_tried": len(cache)}
    return cur, info


def tune_local(groups: list, cfg: DecodeConfig, rounds: int = 2) -> tuple[DecodeConfig, dict]:
    """Coordinate search of the local tempo term (LOCAL_SEARCH) on the validation performances,
    every other setting kept as tuned. Scored by the selection score; ties go to the term off."""
    cache = {}

    def value(c: DecodeConfig) -> float:
        key = tuple(sorted(c.to_dict().items()))
        if key not in cache:
            cache[key] = select([score(p, lp, b, c) for _, p, lp, b in groups])
        return cache[key]
    cur = DecodeConfig(**{**cfg.to_dict(), "local_weight": 0.0})
    start = value(cur)
    for _ in range(rounds):
        for name, options in LOCAL_SEARCH.items():
            best = max(options, key=lambda v: (round(value(DecodeConfig(**{**cur.to_dict(), name: v})), 6),
                                               -abs(v - getattr(DecodeConfig(), name))))
            cur = DecodeConfig(**{**cur.to_dict(), name: best})
    per = {name: (select([score(p, lp, b, DecodeConfig(**{**cfg.to_dict(), "local_weight": 0.0}))]),
                  select([score(p, lp, b, cur)])) for name, p, lp, b in groups}
    info = {"local_select_off": start, "local_select_tuned": value(cur), "local_configs_tried": len(cache),
            "local_per_source": per}
    return cur, info


def tune_piece(groups: list, cfg: DecodeConfig, rounds: int = 2) -> tuple[DecodeConfig, dict]:
    """Coordinate search of the piece-tempo prior (PIECE_SEARCH) on the validation performances,
    every other setting kept as tuned. Scored by the selection score (a stretch written at double
    the note values has twice the bar lines: downbeat F and time-signature accuracy see it); ties
    go to the prior off. The right level (level_ok) and beat F on record, per source."""
    cache = {}

    def scores(c: DecodeConfig):
        key = tuple(sorted(c.to_dict().items()))
        if key not in cache:
            cache[key] = [score(p, lp, b, c) for _, p, lp, b in groups]
        return cache[key]

    def value(c: DecodeConfig) -> float:
        return select(scores(c))
    off = DecodeConfig(**{**cfg.to_dict(), "piece_weight": 0.0})
    cur = off
    for _ in range(rounds):
        for name, options in PIECE_SEARCH.items():
            best = max(options, key=lambda v: (round(value(DecodeConfig(**{**cur.to_dict(), name: v})), 6),
                                               -abs(v - getattr(DecodeConfig(), name))))
            cur = DecodeConfig(**{**cur.to_dict(), name: best})

    def summary(c):
        return {name: {"select": select([s]), "level_ok": nanmean([x["level_ok"] for x in s]),
                       "beat_f": nanmean([x["beat_f"] for x in s])}
                for (name, *_), s in zip(groups, scores(c))}
    info = {"piece_select_off": value(off), "piece_select_tuned": value(cur), "piece_configs_tried": len(cache),
            "piece_per_source": {"off": summary(off), "tuned": summary(cur)}}
    return cur, info


def select(groups_scores: list) -> float:
    """The training script's selection score: mean over the sources of (downbeat F + ts acc) / 2."""
    vals = [(np.mean([s["downbeat_f"] for s in g]) + np.mean([s["ts_acc"] for s in g])) / 2 for g in groups_scores if g]
    return float(np.mean(vals)) if vals else 0.0


def val_groups(model, data: Path, noise, device, n_periscope: int, max_perfs: int = 0, aux_of=None,
               extra: dict | None = None) -> list:
    """The validation performances per source with their tracked beats and the model's
    log-probabilities (computed once; the tracker drawn with another seed than the test draw).
    ``extra``: more sources ({name: split}), e.g. VAL_GANGQINPU."""
    groups = []
    for name, split in {**VAL, **(extra or {})}.items():
        perfs = load(data, split)
        cap = min(n_periscope, max_perfs or n_periscope) if name == "periscope" else max_perfs
        if cap and len(perfs) > cap:
            perfs = [perfs[i] for i in sorted(np.random.default_rng(1).choice(len(perfs), cap, replace=False))]
        if not perfs:
            continue
        beats = [tracked(p, k, noise, seed=SEED + 1) for k, p in enumerate(perfs)]
        lps = [predict_logp(model, p["notes"], bt, bd, device, aux_of(p) if aux_of else None)
               for p, (bt, bd) in zip(perfs, beats)]
        groups.append((name, perfs, lps, beats))
    return groups


def tune(groups: list, rounds: int = 2, tempo: bool = False) -> tuple[DecodeConfig, dict]:
    """Coordinate search of DecodeConfig on the validation performances (``val_groups``)."""
    cfg, cache = DecodeConfig(), {}

    def value(c: DecodeConfig) -> float:
        key = tuple(sorted(c.to_dict().items()))
        if key not in cache:
            cache[key] = select([score(p, lp, b, c) for _, p, lp, b in groups])
        return cache[key]
    start = value(cfg)
    search = {**SEARCH, **(TEMPO_SEARCH if tempo else {})}
    for _ in range(rounds):
        for name, options in search.items():  # ties go to the value nearest the default
            best = max(options, key=lambda v: (value(DecodeConfig(**{**cfg.to_dict(), name: v})),
                                               -abs(v - getattr(DecodeConfig(), name))))
            cfg = DecodeConfig(**{**cfg.to_dict(), name: best})
    info = {"validation": {name: len(p) for name, p, _, _ in groups}, "select_default": start,
            "select_tuned": value(cfg), "configs_tried": len(cache)}
    return cfg, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--conditions", default="clean,real,periscope,gangqinpu")
    ap.add_argument("--local-tempo", action="store_true",
                    help="tune the local tempo consistency term (src/meter.py local_tempo_logprior) on validation, "
                         "Gangqinpu's included, on top of decode.json's settings (or --decode-from's)")
    ap.add_argument("--piece-tempo", action="store_true",
                    help="tune the piece-tempo prior (src/meter.py piece_tempo_logprior) on validation, Gangqinpu's "
                         "included, on top of decode.json's settings (or --decode-from's)")
    ap.add_argument("--tune", action="store_true", help="tune the decoder on the validation performances first")
    ap.add_argument("--tempo-prior", action="store_true",
                    help="--tune also searches the written-tempo prior (src/meter.py tempo_logprior)")
    ap.add_argument("--level", action="store_true",
                    help="also tune the note-value level choice (src/meter.py choose_level) on validation")
    ap.add_argument("--decode-from", default=None,
                    help="start from this decode.json's settings (with --level and no --tune: only the level is tuned)")
    ap.add_argument("--tune-periscope", type=int, default=150, help="PERiScoPe val performances used by --tune")
    ap.add_argument("--tune-max", type=int, default=0, help="at most this many per validation source (0 = all)")
    ap.add_argument("--beat-noise", default="configs/beat_noise.json",
                    help="the simulated tracker at test (Beat This!'s fitted errors)")
    ap.add_argument("--tune-noise", default=None,
                    help="the simulated tracker the decoder is tuned with (default: --beat-noise); "
                         "configs/beat_noise_pm2s.json for a model trained for PM2S")
    ap.add_argument("--tracker", choices=("beat_this", "pm2s"), default="beat_this",
                    help="the beats at test: Beat This!'s (simulated, or recorded on real), or PM2S on the notes")
    ap.add_argument("--real-tracked", action="store_true",
                    help="Beat This!'s real beats stored for the condition (track_meter_data.py), its performances "
                         "that have them only: <condition>_real")
    ap.add_argument("--snap", type=float, default=None,
                    help="move each grid anchor onto a chord onset within this many seconds (saved in decode.json)")
    ap.add_argument("--fill", type=float, default=None,
                    help="add notated beats the tracker missed at chord onsets within this many beats (saved in decode.json)")
    ap.add_argument("--oracle", action="store_true",
                    help="no model: every head certain of the truth on the tracked beats (the best grid they allow)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--runs-dir", default="outputs/runs")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    run, data = Path(args.runs_dir) / args.run, Path(args.data)
    noise = BeatNoise.from_json(args.beat_noise)
    tune_noise = BeatNoise.from_json(args.tune_noise) if args.tune_noise else noise
    aux_tracker = None
    if args.oracle:
        if args.tune:
            raise SystemExit("--oracle has no model to tune a decoder for")
        run.mkdir(parents=True, exist_ok=True)
        (run / "config.json").write_text(json.dumps({"arch": "meter_oracle", "beat_noise": args.beat_noise,
                                                     "note": "no model: each head certain of the truth on the "
                                                             "tracked beats (src/meter.py oracle_logp)"}, indent=1))
        model = None
    else:
        ck = torch.load(run / args.ckpt, map_location=device, weights_only=False)
        model = build_meter_model(ck["config"]).to(device)
        if uses_aux(ck["config"]):
            if args.tracker == "pm2s":
                raise SystemExit(f"{args.run} reads PM2S as its second tracker: test it with Beat This!'s beats "
                                 "(not --tracker pm2s)")
            from src.pm2s_beats import PM2SBeats
            aux_tracker = PM2SBeats(device=str(device) if str(device).startswith("cuda") else "cpu")
        model.load_state_dict(ck["model"])
    aux_of = None
    if aux_tracker is not None:
        aux_cache = {}

        def aux_of(perf):   # PM2S on the performance's notes, once per performance
            key = (perf["performance"], perf.get("segment", 0), len(perf["notes"]))
            if key not in aux_cache:
                aux_cache[key] = aux_tracker(np.asarray(perf["notes"], np.float64))
            return aux_cache[key]
    dfile = run / "decode.json"
    if args.level and model is None:
        raise SystemExit("--level needs a model (the oracle's grids are at the right level already)")
    if (args.tune or args.level or args.local_tempo or args.piece_tempo) and model is None:
        raise SystemExit("--tune, --level, --local-tempo and --piece-tempo need a model")
    if args.tune or args.level or args.local_tempo or args.piece_tempo:
        groups = val_groups(model, data, tune_noise, device, args.tune_periscope, max_perfs=args.tune_max, aux_of=aux_of,
                            extra=VAL_GANGQINPU if args.local_tempo or args.piece_tempo else None)
        if args.tune:
            cfg, info = tune(groups, tempo=args.tempo_prior)
            print(f"decoder tuned on validation ({info['validation']}): selection score {info['select_default']:.3f} "
                  f"-> {info['select_tuned']:.3f} with {cfg.to_dict()}", flush=True)
        else:
            src = Path(args.decode_from) if args.decode_from else dfile
            if not src.exists():
                raise SystemExit(f"--level / --local-tempo / --piece-tempo without --tune start from a tuned decoder: "
                                 f"{src} not found (--decode-from)")
            prev = json.loads(src.read_text())
            cfg, info = DecodeConfig.from_dict(prev["decode"]), {k: v for k, v in prev.items() if k != "decode"}
            info["decode_from"] = str(src)
        if args.level:
            has_head = any(lp is not None and "tempo" in lp for _, _, lps, _ in groups for lp in lps)
            cfg, linfo = tune_level(groups, cfg, has_head)
            info.update(linfo)
            print(f"level tuned on validation: selection score {linfo['level_select_decoded']:.3f} -> "
                  f"{linfo['level_select_tuned']:.3f}, right level {linfo['level_ok_decoded']:.3f} -> "
                  f"{linfo['level_ok_tuned']:.3f} (level_weight {cfg.level_weight}, level_model_weight "
                  f"{cfg.level_model_weight}, level_stay {cfg.level_stay}{'' if has_head else '; no tempo head'})",
                  flush=True)
        if args.local_tempo:
            cfg, tinfo = tune_local(groups, cfg)
            info.update(tinfo)
            print(f"local tempo tuned on validation ({ {n: len(p) for n, p, _, _ in groups} }): selection score "
                  f"{tinfo['local_select_off']:.4f} -> {tinfo['local_select_tuned']:.4f} (local_weight {cfg.local_weight}, "
                  f"local_sigma {cfg.local_sigma}); per source (off, on): {tinfo['local_per_source']}", flush=True)
        if args.piece_tempo:
            cfg, pinfo = tune_piece(groups, cfg)
            info.update(pinfo)
            print(f"piece tempo tuned on validation ({ {n: len(p) for n, p, _, _ in groups} }): selection score "
                  f"{pinfo['piece_select_off']:.4f} -> {pinfo['piece_select_tuned']:.4f} (piece_weight {cfg.piece_weight}, "
                  f"piece_sigma {cfg.piece_sigma}); per source: {pinfo['piece_per_source']}", flush=True)
        dfile.write_text(json.dumps({"decode": cfg.to_dict(), **info}, indent=1))
    elif args.decode_from:
        dfile.write_text(Path(args.decode_from).read_text())
    cfg = DecodeConfig.from_dict(json.loads(dfile.read_text())["decode"]) if dfile.exists() else DecodeConfig()
    if args.snap is not None or args.fill is not None:
        cfg.snap_s = cfg.snap_s if args.snap is None else float(args.snap)
        cfg.fill_tol = cfg.fill_tol if args.fill is None else float(args.fill)
        saved = json.loads(dfile.read_text()) if dfile.exists() else {}
        dfile.write_text(json.dumps({**saved, "decode": cfg.to_dict()}, indent=1))
    out_file = run / "eval_meter.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    if args.real_tracked and args.tracker != "beat_this":
        raise SystemExit("--real-tracked is Beat This!'s stored beats: not with --tracker pm2s (PM2S runs here)")
    pm2s = None
    from src.pm2s_beats import tagged
    if args.tracker == "pm2s":
        from src.pm2s_beats import PM2SBeats
        pm2s = PM2SBeats(device=str(device) if str(device).startswith("cuda") else "cpu")
    for cond_name in args.conditions.split(","):
        perfs = load(data, CONDITIONS[cond_name])
        if not perfs:
            print(f"skip {cond_name}: no {CONDITIONS[cond_name]} performances (scripts/build_meter_data.py)")
            continue
        # results / file names (other PM2S weights than the released: <condition>_pm2s_<tag>)
        cond = cond_name if pm2s is None else tagged(f"{cond_name}_pm2s")
        real = None
        if args.real_tracked:
            from scripts.track_meter_data import tracked_path
            real = load_beats(tracked_path(data / "meter", CONDITIONS[cond_name], "beat_this"))
            perfs = [p for p in perfs if perf_key(p) in real]
            if not perfs:
                print(f"skip {cond_name}: no Beat This! beats stored for {CONDITIONS[cond_name]} (track_meter_data.py)")
                continue
            cond = f"{cond_name}_real"
        fallbacks = 0                                                # PM2S found < 2 beats: the simulated tracker
        grids, scores, default, raw, per = {}, [], [], [], []
        confusion, refined = Counter(), Counter()
        for k, perf in enumerate(perfs):
            if real is not None:
                bt, bd = (np.asarray(x) for x in real[perf_key(perf)])
            elif pm2s is None:
                bt, bd = tracked(perf, k, noise)
            else:  # PM2S on the performance's notes (np arrays as tracked() gives)
                bt, bd = pm2s(np.asarray(perf["notes"], np.float64))
                if len(bt) < 2:
                    bt, bd = tracked(perf, k, noise)
                    fallbacks += 1
            lp = (oracle_logp(bt, perf["gt"]) if model is None
                  else predict_logp(model, perf["notes"], bt, bd, device, aux_of(perf) if aux_of else None))
            grid = grid_from_logp(bt, lp, cfg)
            if len(perf["notes"]):  # beat times onto the notes (refine_grid; off unless --snap / --fill)
                grid = refine_grid(grid, np.asarray(perf["notes"], np.float64)[:, 1], cfg)
            refined.update(grid.get("refined", {}))
            grid["key"] = estimate_key(perf["notes"])  # the key signature the rendered score gets
            grid["performance"] = perf["performance"]
            grid["span"] = tuple(perf.get("span") or (float(bt[0]) - 1.0, float(bt[-1]) + 1.0))
            s = grid_scores(grid, perf["gt"])
            # the labeller's grid: bars in front for the notes before the first bar line (scored above without)
            on = np.asarray(perf["notes"], np.float64)[:, 1] if len(perf["notes"]) else np.zeros(0)
            on = on[on >= grid["span"][0]]
            grids[key_of(perf)] = cover_notes(grid, float(on.min())) if len(on) else grid
            scores.append(s)
            default.append(grid_scores(grid_from_logp(bt, lp, DecodeConfig()), perf["gt"]))
            gt = perf["gt"]
            raw.append({"beat_f": f_measure(gt["t"], bt), "downbeat_f": f_measure(gt["t"][gt["down"]], bt[bd])})
            bar_t = np.asarray(grid["bar_t"])
            for j in np.nonzero(gt["down"])[0]:  # per ground-truth bar: true -> decoded time signature
                b = int(np.searchsorted(bar_t, gt["t"][j] + MATCH_TOL)) - 1
                confusion[(gt["ts_names"][j], grid["bars"][b] if 0 <= b < len(grid["bars"]) else "-")] += 1
            per.append({"performance": key_of(perf), "piece": perf.get("piece"),
                        "true_ts": Counter(gt["ts_names"]).most_common(1)[0][0],
                        "decoded_ts": Counter(grid["bars"]).most_common(1)[0][0] if grid["bars"] else "-",
                        **{f"grid_{a}": v for a, v in s.items()}, **{f"tracker_{a}": v for a, v in raw[-1].items()}})
        res = {f"grid_{key}": nanmean([s[key] for s in scores]) for key in scores[0]}
        res.update({f"tracker_{key}": float(np.mean([s[key] for s in raw])) for key in raw[0]})
        res["default"] = {f"grid_{key}": nanmean([s[key] for s in default]) for key in default[0]}
        by_piece = {}
        for r in per:
            by_piece.setdefault(r["piece"], []).append(r)
        res["piece_mean"] = {f"grid_{key}": nanmean([nanmean([r[f'grid_{key}'] for r in v])
                                                     for v in by_piece.values()]) for key in scores[0]}
        res["level_changed"] = int(sum(g.get("level", 1.0) != 1.0 for g in grids.values()))
        res["performances"], res["pieces"] = len(perfs), len(by_piece)
        res["decode"] = cfg.to_dict()
        res["decoder_version"] = DECODER_VERSION
        res["ts_confusion"] = [[a, b, n] for (a, b), n in confusion.most_common(25)]
        res["refined"] = {k: v / len(perfs) for k, v in refined.items()}   # anchors moved / added per performance
        res["tracker"] = args.tracker + ("_real" if real is not None else "")
        if aux_of is not None:
            res["aux"] = "pm2s"
        if pm2s is not None:
            res["pm2s_fallbacks"] = fallbacks
        results[cond] = res
        with open(run / f"grids_{cond}.pkl", "wb") as f:
            pickle.dump(grids, f, protocol=pickle.HIGHEST_PROTOCOL)
        (run / f"meter_scores_{cond}.json").write_text(json.dumps(per, indent=1))
        wrong = [(a, b, n) for (a, b), n in confusion.most_common() if a != b][:4]
        print(f"{cond} ({len(perfs)} performances, {len(by_piece)} pieces): grid beat F {res['grid_beat_f']:.3f} "
              f"(tracker {res['tracker_beat_f']:.3f}), downbeat F {res['grid_downbeat_f']:.3f} (tracker "
              f"{res['tracker_downbeat_f']:.3f}), time signature {res['grid_ts_acc']:.3f} | default decoder: "
              f"downbeat F {res['default']['grid_downbeat_f']:.3f}, time signature {res['default']['grid_ts_acc']:.3f}"
              f" | right level {res['grid_level_ok']:.3f} ({res['level_changed']} grids moved)"
              " | most bars wrong: " + ", ".join(f"{a}->{b} {n}" for a, b, n in wrong), flush=True)
    out_file.write_text(json.dumps(results, indent=1))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
