"""Train the meter model (v2, Phase 2: src/meter.py) on whole performances.

    python scripts/build_meter_data.py                  # once: data/processed/meter/<split>.pkl
    python scripts/train_meter.py --name meter_v2
    python scripts/train_meter.py --name meter_v2 --resume

Training data (every source that data/processed/meter has):
  ASAP           the clean training performances: the tracker simulated on the annotated beats
                 (configs/beat_noise.json: octave errors, missed / extra beats, jitter, downbeat
                 errors; a new draw every time) and synthetic transcription noise on the notes
                 (--note-noise)
  real           the Transkun transcriptions of ASAP training audio with Beat This!'s own beats
  PERiScoPe      real Transkun transcriptions of 971 more works, the grid from their score
                 alignment (scripts/build_meter_data.py); the tracker simulated, the notes as they
                 are (--no-periscope leaves them out)
  Gangqinpu      with --gangqinpu: ~18k web piano recordings of many genres (pop, jazz, folk,
                 classical), Transkun transcriptions with the grid from their score alignment
                 (build_meter_data.py --gangqinpu); the tracker simulated. --gangqinpu-share is the
                 share of every batch drawn from it (the other sources keep their proportions in
                 the rest), so its size does not drown the classical data; validated on a fixed
                 draw of --val-gangqinpu of its validation performances
Every step draws --batch performances (--balance piece: each piece in proportion to its length,
however many performances it has; beats: each performance in proportion to its length), augments
each (src/meter.py Augment: tempo stretch, transposition, velocity; --no-augment turns it off: on
ASAP alone a model then memorises its 156 pieces within ~2k steps, validation time-signature
accuracy falling while training goes on) and cuts a random stretch of --chunk-min .. --chunk beats
from each. --workers processes make the samples.

Another tracker: --beat-noise <its fitted errors> (scripts/fit_pm2s_noise.py ->
configs/beat_noise_pm2s.json for PM2S) with --real-beats simulated, so the real transcriptions get
the simulated tracker too, not Beat This!'s recorded beats: a model trained for that tracker only.

Real tracker output in the mix: --tracked beat_this / pm2s attaches that tracker's own beats
(scripts/track_meter_data.py: Beat This! on Gangqinpu's recordings, PM2S on Gangqinpu's and
PERiScoPe's notes; never ASAP, which both trackers trained on) to the training performances that
have them, and each time one of them is drawn it gets those beats with probability
--tracked-share, else the simulated tracker: errors that come from the music itself (where the
real tracker goes wrong), and the variety of fresh simulated draws. beat_this_<tag> /
beat_this_render_<tag>: our fine-tuned Beat This!'s, each performance tracked by the model that never
heard its piece (scripts/track_meter_data.py --models). --tracked beat_this_render:
Beat This! on the performances rendered to audio (src/render.py), every source including ASAP
(tracked by the fold model that never heard it). Several, comma-separated, in order of preference:
--tracked beat_this,beat_this_render gives each performance its recording's beats where there is
one, else its render's; with --tracked-share 1 the simulated tracker is left out of training.

A second tracker as evidence: --aux pm2s gives every beat of the first tracker features of PM2S's
beats around it (src/meter.py aux_features: does it agree, does it beat faster, on the off-beats),
PM2S simulated with --aux-noise or, in --tracked-share of the draws that have them, its real beats
(Gangqinpu, PERiScoPe). The model then reads both trackers (n_features = N_ALL_FEATURES).

The tracker's errors per source: --beat-noise-gangqinpu <fitted on Gangqinpu> simulates the first
tracker on Gangqinpu's performances with the errors it makes on that music (scripts/fit_pm2s_noise.py
--from-stored beat_this: Beat This!'s real beats of its recordings), the others keep --beat-noise.

Validation (every --eval-every steps), per source (ASAP val, real val, a fixed --val-periscope
PERiScoPe val performances): the heads' accuracies and the DECODED grids scored against the
ground truth (src/meter.py grid_scores: beat F, downbeat F, time-signature accuracy) next to the
raw tracker's beat / downbeat F on the same beats. best.pt keeps the best mean over the sources of
(downbeat F + time-signature accuracy) / 2: the bar lines and meters the note labeller is given.
Early stopping, resume and atomic checkpoints as in the other training scripts; --max-hours stops
cleanly (last.pt) before a job's time limit, and --resume goes on from there.

The piece's note-value level (src/meter.py choose_level): --tempo-head adds a head estimating the
piece's written tempo (log quarters per minute, Gaussian) from the mean of its beats' states; the
decoder then picks between the decoded grid and the same bar lines at halved / doubled note values
by it. --init starts from another run's checkpoint (the new head starts fresh), --freeze-trunk
trains only that head (the bar / step / time-signature heads stay exactly the checkpoint's).
--select picks what best.pt keeps: meter (the grid score above), tempo (the share of validation
performances at the right level once the head chooses it), both (their mean).

    python scripts/train_meter.py --name meter_v3_th --init outputs/runs/meter_v3/best.pt --tempo-head \
        --freeze-trunk --select tempo --steps 6000 --lr 1e-3

Inputs: per-beat features (src/meter.py beat_features). --note-encoder adds an embedding of every
note pooled into its beat; off by default since the second round: with PERiScoPe it overfits
(best validation at 4k steps, then early stopping) where the beat features alone kept improving
to 26k and won on every test set (docs/experiments.md, "Meter model, second round: results").
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import lower_priority, pick_device  # noqa: E402
from scripts.train_model import learning_rate  # noqa: E402
from src.beats import BeatNoise, f_measure, load_beats  # noqa: E402
from src.meter import (N_ALL_FEATURES, N_BEAT_FEATURES, N_NOTE_FEATURES, Augment, DecodeConfig, MeterConfig,  # noqa: E402
                       build_meter_model, choose_level, collate, grid_from_logp, grid_scores, logp_of, meter_loss,
                       perf_key, sample)
from src.pm2s_beats import weights_tag  # noqa: E402
from src.preprocessing.noise import NoiseModel  # noqa: E402

FIRST_FEATURES = 27  # the first meter model's beat features (--beat-features first)
HEADS = ("bar", "step", "ts")
HEAD_CHOICE = DecodeConfig(level_model_weight=1.0)   # the level chosen by the tempo head alone (validation)


def load(data: Path, split: str) -> list:
    """data/processed/meter/<split>.pkl ([] when it was not built); every performance gets a
    ``piece`` (its folder for the ASAP ones built before pieces were stored)."""
    p = data / "meter" / f"{split}.pkl"
    perfs = pickle.load(open(p, "rb")) if p.exists() else []
    for x in perfs:
        x.setdefault("piece", str(Path(x["performance"]).parent))
    return perfs


def mixed_weights(sources: dict, balance: str, shares: dict, compound_weight: float = 1.0) -> np.ndarray:
    """Sampling weights over the concatenated sources: each source named in ``shares`` gets that
    share of the draws (``weights`` within it), the others share the rest in proportion to their
    own ``weights`` together, as without it. Empty sources take no share. ``compound_weight``: see
    ``weights`` (within each source, so the sources' shares stay)."""
    parts, fixed = [], {k: v for k, v in shares.items() if sources.get(k)}
    rest = [k for k in sources if k not in fixed and sources[k]]
    rest_total = 1.0 - sum(fixed.values()) if rest else 0.0
    if not rest:
        fixed = {k: v / sum(fixed.values()) for k, v in fixed.items()}
    rest_w = (weights([p for k in rest for p in sources[k]], balance, compound_weight) * rest_total if rest
              else np.zeros(0))
    off = 0
    for k, v in sources.items():
        if not v:
            continue
        if k in fixed:
            parts.append(weights(v, balance, compound_weight) * fixed[k])
        else:
            parts.append(rest_w[off:off + len(v)])
            off += len(v)
    w = np.concatenate(parts) if parts else np.zeros(0)
    return w / w.sum()


def weights(perfs: list, balance: str, compound_weight: float = 1.0) -> np.ndarray:
    """Sampling weights: ``beats`` = each performance by its length; ``piece`` = each piece by its
    (mean performance) length, shared by its performances, so a piece played 20 times is not
    drawn 20 times as often as one played once. ``compound_weight``: a performance mostly in a
    compound meter (6/8, 9/8, 12/8, 6/4: src/beats.py piece_compound) counts that many times (1:
    as before). They are rare (Gangqinpu: 6/8 4.5% of the pieces, 12/8 1.2%) and the meter model
    reads them as 3/4 or 4/4 half of the time on real recordings."""
    n = np.array([len(p["gt"]["t"]) for p in perfs], float)
    if balance == "piece":
        count = Counter(p["piece"] for p in perfs)
        n = n / np.array([count[p["piece"]] for p in perfs], float)
    if compound_weight != 1.0:
        from src.beats import piece_compound
        n = n * np.array([compound_weight if piece_compound(p["gt"]) else 1.0 for p in perfs])
    return n / n.sum()


def with_tracked(perf: dict, rng: np.random.Generator, share: float) -> dict:
    """The performance with its real tracker's beats (``tracked``, attached by --tracked) with
    probability ``share``, else as it is (the simulated tracker, or its recorded beats)."""
    out = perf
    if share > 0 and perf.get("tracked") is not None and rng.random() < share:
        out = {**out, "beats": perf["tracked"]}
    if share > 0 and perf.get("tracked_aux") is not None and rng.random() < share:   # the second tracker's (--aux)
        out = {**out, "aux": perf["tracked_aux"]}
    return out


def attach_tracked(perfs: list, beats: dict, key: str = "tracked", keep: bool = False) -> int:
    """Each performance's real tracked beats (keyed by src/meter.py perf_key) as ``key``
    (tracked: the first tracker's, tracked_aux: the second's); how many got them. ``keep``: a
    performance that already has some keeps them (trackers attached in order of preference)."""
    n = 0
    for p in perfs:
        if keep and p.get(key) is not None:
            continue
        b = beats.get(perf_key(p))
        if b is not None and len(b[0]) >= 2:
            p[key] = (np.asarray(b[0], np.float64), np.asarray(b[1], bool))
            n += 1
    return n


class Stream(torch.utils.data.IterableDataset):
    """Endless training batches (collated on the CPU), each worker with its own random stream."""

    def __init__(self, perfs, w, noise, note_noise, batch, chunk, chunk_min, seed, augment=None, tracked_share=0.0,
                 by_dataset=None, aux_noise=None):
        self.perfs, self.w, self.noise, self.note_noise, self.augment = perfs, w, noise, note_noise, augment
        self.batch, self.chunk, self.chunk_min, self.seed = batch, chunk, chunk_min, seed
        self.tracked_share, self.by_dataset, self.aux_noise = tracked_share, by_dataset, aux_noise

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng((self.seed, info.id if info else 0))
        while True:
            length = int(rng.integers(self.chunk_min, self.chunk + 1))
            idx = rng.choice(len(self.perfs), self.batch, p=self.w)
            drawn = [with_tracked(self.perfs[i], rng, self.tracked_share) for i in idx]
            yield collate([sample(p, rng, noise_for(p, self.noise, self.by_dataset), length, self.note_noise,
                                  self.augment, self.aux_noise) for p in drawn])


def noise_for(perf: dict, noise, by_dataset: dict | None):
    """The simulated tracker for a performance: its dataset's own (--beat-noise-gangqinpu), else ``noise``."""
    return (by_dataset or {}).get(perf.get("dataset"), noise)


@torch.no_grad()
def validate(model, groups: dict, device, noise, seed: int = 0, select: str = "meter", by_dataset=None,
             aux_noise=None) -> dict:
    """Per validation source: head accuracies and decoded-grid scores (the tracker simulated with a
    fixed seed per performance; the real one where it exists), the share of performances at the
    right note-value level (level_ok), and with a tempo head the same once it chooses the level
    (head_level_ok) and its mean absolute error in log written tempo. ``select`` (what best.pt
    keeps): meter = the mean over the sources of (downbeat F + time-signature accuracy) / 2; tempo =
    the mean over the sources of head_level_ok; both = the mean of the two."""
    model.eval()
    res, sel, sel_t = {}, [], []
    ok = {h: 0 for h in HEADS}
    n = {h: 0 for h in HEADS}
    for gi, (name, perfs) in enumerate(groups.items()):
        scores, raw, chosen, err = [], [], [], []
        for k, perf in enumerate(perfs):
            f, tg, bt, nin = sample(perf, np.random.default_rng((seed, gi, k)), noise_for(perf, noise, by_dataset),
                                    aux_noise=aux_noise)
            if len(f) < 2:
                continue
            x, mask, notes, _ = collate([(f, None, bt, nin)], device)
            out = model(x, mask, notes)
            lp = logp_of(out)
            for h in HEADS:
                m = tg[h] >= 0
                ok[h] += int((lp[h].argmax(-1) == tg[h])[m].sum())
                n[h] += int(m.sum())
            grid = grid_from_logp(bt, lp, DecodeConfig())
            scores.append(grid_scores(grid, perf["gt"]))
            if "tempo" in lp:
                chosen.append(grid_scores(choose_level(grid, HEAD_CHOICE, lp["tempo"]), perf["gt"])["level_ok"])
                if np.isfinite(tg["tempo"][0]):
                    err.append(abs(float(lp["tempo"][0]) - float(tg["tempo"][0])))
            gt = perf["gt"]
            rng2 = np.random.default_rng((seed, gi, k))  # the same tracked beats again, for the tracker's own scores
            tb, tdn = (perf["beats"] if perf.get("beats") is not None
                       else noise_for(perf, noise, by_dataset)(gt["t"], gt["down"], rng2))
            raw.append({"beat_f": f_measure(gt["t"], tb), "downbeat_f": f_measure(gt["t"][gt["down"]], tb[tdn])})
        if not scores:
            continue
        for key in ("beat_f", "downbeat_f", "ts_acc"):
            res[f"{name}_grid_{key}"] = float(np.mean([s[key] for s in scores]))
        res[f"{name}_level_ok"] = nanmean([s["level_ok"] for s in scores])
        if chosen:
            res[f"{name}_head_level_ok"] = nanmean(chosen)
            res[f"{name}_tempo_mae"] = float(np.mean(err)) if err else float("nan")
            sel_t.append(res[f"{name}_head_level_ok"])
        for key in ("beat_f", "downbeat_f"):
            res[f"{name}_tracker_{key}"] = float(np.mean([s[key] for s in raw]))
        sel.append((res[f"{name}_grid_downbeat_f"] + res[f"{name}_grid_ts_acc"]) / 2)
    model.train()
    res.update({f"{h}_acc": ok[h] / max(n[h], 1) for h in HEADS})
    if select != "meter" and not sel_t:
        raise ValueError(f"--select {select} needs a tempo head (--tempo-head)")
    meter_sel = float(np.mean(sel)) if sel else 0.0
    tempo_sel = float(np.mean(sel_t)) if sel_t else 0.0
    res["select"] = {"meter": meter_sel, "tempo": tempo_sel, "both": (meter_sel + tempo_sel) / 2}[select]
    return res


def nanmean(xs) -> float:
    """The mean of the values that are not nan (0.0 when none are)."""
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--note-encoder", action=argparse.BooleanOptionalAction, default=False,
                    help="pool a learned embedding of every note into its beat (--no-note-encoder: beat features only)")
    ap.add_argument("--d-note", type=int, default=64)
    ap.add_argument("--beat-features", choices=("all", "first"), default="all",
                    help="first = the first meter model's 27 (no pitch-class profile)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=1024, help="most beats per training sample")
    ap.add_argument("--chunk-min", type=int, default=0, help="fewest beats per training sample (0 = chunk / 4)")
    ap.add_argument("--balance", choices=("piece", "beats"), default="piece")
    ap.add_argument("--compound-weight", type=float, default=1.0,
                    help="draw performances in a compound meter (6/8, 9/8, 12/8, 6/4) this many times as often "
                         "(within each source; 1: as their length says)")
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                    help="tempo stretch, transposition and velocity scaling of every training sample")
    ap.add_argument("--periscope", action=argparse.BooleanOptionalAction, default=True,
                    help="train (and validate) on the PERiScoPe performances too, when built")
    ap.add_argument("--val-periscope", type=int, default=150, help="PERiScoPe validation performances (fixed draw)")
    ap.add_argument("--gangqinpu", action=argparse.BooleanOptionalAction, default=False,
                    help="train (and validate) on the Gangqinpu performances too (build_meter_data.py --gangqinpu)")
    ap.add_argument("--gangqinpu-share", type=float, default=0.5,
                    help="share of every batch drawn from Gangqinpu (0 = in proportion to its size, like the others)")
    ap.add_argument("--val-gangqinpu", type=int, default=150, help="Gangqinpu validation performances (fixed draw)")
    ap.add_argument("--beat-noise", default="configs/beat_noise.json",
                    help="the tracker's fitted errors the ground-truth beats get (configs/beat_noise_pm2s.json: PM2S)")
    ap.add_argument("--real-beats", choices=("tracked", "simulated"), default="tracked",
                    help="the real transcriptions: Beat This!'s recorded beats, or the simulated tracker "
                         "(--beat-noise) as the other sources (training for another tracker)")
    ap.add_argument("--tracked", default="none",
                    help="mix real trackers' beats (meter/tracked_*.npz) into training: beat_this, pm2s, beat_this_render, "
                         "or several comma-separated in order of preference (beat_this,beat_this_render)")
    ap.add_argument("--tracked-share", type=float, default=0.5,
                    help="with --tracked / --aux: the chance a performance that has them is drawn with the real beats")
    ap.add_argument("--aux", choices=("none", "pm2s"), default="none",
                    help="a second tracker read as evidence on the first one's beats (src/meter.py aux_features)")
    ap.add_argument("--aux-noise", default="configs/beat_noise_pm2s.json", help="the second tracker's fitted errors")
    ap.add_argument("--beat-noise-gangqinpu", default=None,
                    help="the first tracker's errors fitted on Gangqinpu (fit_pm2s_noise.py --from-stored), for its "
                         "performances; the others keep --beat-noise")
    ap.add_argument("--note-noise", default="configs/noise_stats.json", help="'none' = clean notes")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=5, help="evaluations without a better selection score (0 = off)")
    ap.add_argument("--workers", type=int, default=6, help="processes making training samples (0 = in the main one)")
    ap.add_argument("--tempo-head", action="store_true",
                    help="a piece-level head estimating the written tempo, for the note-value level (choose_level)")
    ap.add_argument("--tempo-loss", type=float, default=1.0, help="weight of the tempo head's loss")
    ap.add_argument("--init", default=None, help="start from this checkpoint's weights (a new head starts fresh)")
    ap.add_argument("--freeze-trunk", action="store_true",
                    help="train only the tempo head (needs --init and --tempo-head): the other heads stay the checkpoint's")
    ap.add_argument("--select", choices=("meter", "tempo", "both"), default="meter",
                    help="what best.pt keeps: the grid score, the level the tempo head picks, or their mean")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop cleanly (last.pt saved, not done) after this long; --resume continues (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--normal-priority", action="store_true")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--runs-dir", default="outputs/runs")
    args = ap.parse_args(argv)
    if not args.normal_priority:
        lower_priority()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    data, run = Path(args.data), Path(args.runs_dir) / args.name
    run.mkdir(parents=True, exist_ok=True)
    sources = {"asap": load(data, "train"), "real": load(data, "train_transcribed"),
               "periscope": load(data, "train_periscope") if args.periscope else []}
    if args.gangqinpu:
        sources["gangqinpu"] = load(data, "train_gangqinpu")
        if not sources["gangqinpu"]:
            raise SystemExit(f"--gangqinpu: no {data / 'meter' / 'train_gangqinpu.pkl'} "
                             "(scripts/build_meter_data.py --gangqinpu <release> --gangqinpu-only)")
    if not 0.0 <= args.gangqinpu_share < 1.0:
        raise SystemExit("--gangqinpu-share must be in [0, 1)")
    tracked_list = [t for t in args.tracked.split(",") if t and t != "none"]
    import re
    bad = [t for t in tracked_list if not re.fullmatch(r"(beat_this|beat_this_render)(_[A-Za-z0-9]+)?|pm2s", t)]
    if bad:
        raise SystemExit(f"--tracked: unknown {bad} (beat_this, pm2s, beat_this_render; beat_this[_render]_<tag>: "
                         "our Beat This!, scripts/track_meter_data.py --models)")
    tracked_counts, aux_counts = {}, {}
    if args.aux != "none":
        from scripts.track_meter_data import tracked_path
        for name, split in (("gangqinpu", "train_gangqinpu"), ("periscope", "train_periscope")):
            if sources.get(name):
                aux_counts[name] = attach_tracked(sources[name], load_beats(tracked_path(data / "meter", split, args.aux)),
                                                  key="tracked_aux")
    if tracked_list:
        from scripts.track_meter_data import tracked_path
        for name, split in (("asap", "train"), ("gangqinpu", "train_gangqinpu"), ("periscope", "train_periscope")):
            if sources.get(name):   # the first tracker in the list that has a performance's beats gives them
                tracked_counts[name] = sum(
                    attach_tracked(sources[name], load_beats(f), keep=True)
                    for f in (tracked_path(data / "meter", split, t) for t in tracked_list) if f.exists())
        if not any(tracked_counts.values()):  # e.g. no recording could be downloaded: trains on, simulated only
            print(f"WARNING --tracked {args.tracked}: no tracked beats in {data / 'meter'} "
                  "(scripts/track_meter_data.py): the simulated tracker only (config.json: tracked_by_source)",
                  flush=True)
    train = [p for v in sources.values() for p in v]
    groups = {"asap": load(data, "val"), "real": load(data, "val_transcribed")}

    def fixed_draw(perfs, n):
        if len(perfs) > n:
            perfs = [perfs[i] for i in sorted(np.random.default_rng(0).choice(len(perfs), n, replace=False))]
        return perfs
    if args.periscope:
        groups["periscope"] = fixed_draw(load(data, "val_periscope"), args.val_periscope)
    if args.gangqinpu:
        groups["gangqinpu"] = fixed_draw(load(data, "val_gangqinpu"), args.val_gangqinpu)
    if args.real_beats == "simulated":  # their notes as they are, the tracker simulated on the true beats
        def simulated(perfs):
            return [{**p, "beats": None, "real_notes": True} for p in perfs]
        sources["real"] = simulated(sources["real"])
        groups["real"] = simulated(groups["real"])
        train = [p for v in sources.values() for p in v]
    groups = {k: v for k, v in groups.items() if v}
    if not train or not groups:
        raise SystemExit(f"{data / 'meter'}: no train / val performances: run scripts/build_meter_data.py first")
    noise = BeatNoise.from_json(args.beat_noise)
    by_dataset = {}
    if args.beat_noise_gangqinpu:
        if Path(args.beat_noise_gangqinpu).exists():
            by_dataset["gangqinpu"] = BeatNoise.from_json(args.beat_noise_gangqinpu)
        else:   # e.g. no recording could be downloaded to fit it on
            print(f"WARNING {args.beat_noise_gangqinpu} missing: Gangqinpu gets --beat-noise too", flush=True)
    aux_noise = BeatNoise.from_json(args.aux_noise) if args.aux != "none" else None
    note_noise = None if args.note_noise == "none" else NoiseModel.from_json(args.note_noise)
    cfg = MeterConfig(n_features=(N_ALL_FEATURES if args.aux != "none" else N_BEAT_FEATURES)
                      if args.beat_features == "all" else FIRST_FEATURES,
                      d_model=args.d_model, layers=args.layers, dropout=args.dropout,
                      n_heads=max(4, args.d_model // 32), ff=4 * args.d_model,
                      n_note_features=N_NOTE_FEATURES if args.note_encoder else 0, d_note=args.d_note).to_dict()
    if args.freeze_trunk and not (args.init and args.tempo_head):
        raise SystemExit("--freeze-trunk trains only the tempo head: give --init <checkpoint> and --tempo-head")
    if args.select != "meter" and not args.tempo_head:
        raise SystemExit(f"--select {args.select} needs --tempo-head")
    init = None
    if args.init:  # the starting checkpoint's architecture wins over the size flags
        init = torch.load(args.init, map_location=device, weights_only=False)
        cfg = dict(init["config"])
    cfg["tempo_head"] = bool(args.tempo_head or cfg.get("tempo_head", False))
    ck = None
    if args.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        cfg = ck["config"]
    elif (run / "last.pt").exists():
        raise SystemExit(f"{run} already has a checkpoint: use --resume, or a different --name")
    model = build_meter_model(cfg).to(device)
    if init is not None and ck is None:
        missing, unexpected = model.load_state_dict(init["model"], strict=False)
        if unexpected or any(not k.startswith("tempo.") for k in missing):
            raise SystemExit(f"--init {args.init}: the checkpoint does not fit (missing {missing}, unexpected {unexpected})")
        print(f"started from {args.init} (new: {', '.join(sorted({k.split('.')[0] for k in missing})) or 'nothing'})",
              flush=True)
    if args.freeze_trunk:
        for name, prm in model.named_parameters():
            prm.requires_grad = name.startswith("tempo.")
    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    step, best, best_step, done = 0, -1.0, 0, False
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best, best_step, done = ck["step"], ck["best"], ck["best_step"], ck.get("done", False)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in trainable)
    counts = {k: len(v) for k, v in sources.items()}
    pieces = {k: len({p["piece"] for p in v}) for k, v in sources.items()}
    json.dump({"args": vars(args), "arch": "meter", "model": cfg, "params": n_params,
               "data": {"train": len(train), "train_by_source": counts, "pieces_by_source": pieces,
                        "val_by_source": {k: len(v) for k, v in groups.items()},
                        "tracked_by_source": tracked_counts, "tracked_aux_by_source": aux_counts,
                        "real_transcriptions": counts["real"]},
               # which PM2S weights its stored PM2S beats came from (src/pm2s_beats.py weights_tag; '' = released)
               "pm2s": ({"tag": weights_tag(), "weights": os.environ.get("PM2S_WEIGHTS") or "released"}
                        if "pm2s" in (*tracked_list, args.aux) else None)},
              open(run / "config.json", "w"), indent=1)
    print(f"meter model: {n_params / 1e6:.2f}M parameters ({n_train / 1e6:.2f}M trained) on {device} (note encoder: "
          f"{'yes' if cfg.get('n_note_features') else 'no'}, {cfg['n_features']} beat features) | training: "
          + ", ".join(f"{k} {counts[k]} performances / {pieces[k]} pieces" for k in counts)
          + " | validation: " + ", ".join(f"{k} {len(v)}" for k, v in groups.items()), flush=True)

    def save(path):
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "best_step": best_step, "done": done, "arch": "meter", "config": cfg}, tmp)
        os.replace(tmp, path)
        st = run / "status.json.tmp"
        st.write_text(json.dumps({"phase": "done" if done else "train", "step": step, "total_steps": args.steps,
                                  "best": best, "best_step": best_step}))
        os.replace(st, run / "status.json")

    shares = {"gangqinpu": args.gangqinpu_share} if args.gangqinpu and args.gangqinpu_share > 0 else {}
    stream = Stream(train, mixed_weights(sources, args.balance, shares, args.compound_weight), noise, note_noise,
                    args.batch, args.chunk,
                    args.chunk_min or max(args.chunk // 4, 2), args.seed * 100003 + step,
                    Augment() if args.augment else None,
                    tracked_share=args.tracked_share if (tracked_list or args.aux != "none") else 0.0,
                    by_dataset=by_dataset, aux_noise=aux_noise)
    loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers,
                                         prefetch_factor=4 if args.workers else None,
                                         persistent_workers=bool(args.workers))
    batches = iter(loader)
    log_path, t0, run_loss = run / "log.csv", time.time(), 0.0
    deadline = t0 + args.max_hours * 3600 if args.max_hours > 0 else float("inf")
    model.train()
    try:
        while not done:
            x, mask, notes, tg = next(batches)
            x, mask, tg = x.to(device), mask.to(device), {h: v.to(device) for h, v in tg.items()}
            notes = tuple(v.to(device) for v in notes)
            for g in opt.param_groups:
                g["lr"] = learning_rate(step, lr=args.lr, warmup=args.warmup, total_steps=args.steps,
                                        min_lr_frac=args.min_lr_frac)
            loss, parts = meter_loss(model(x, mask, notes), tg, args.tempo_loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1
            run_loss += float(loss.detach())
            if step % 100 == 0:
                print(f"step {step}/{args.steps} loss {run_loss / 100:.4f} ({time.time() - t0:.0f} s)", flush=True)
                run_loss = 0.0
            if step % args.eval_every == 0 or step >= args.steps:
                res = validate(model, groups, device, noise, args.seed, args.select, by_dataset, aux_noise)
                res.update(step=step, minutes=round((time.time() - t0) / 60, 1))
                new = not log_path.exists()
                with open(log_path, "a", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(res))
                    if new:
                        wr.writeheader()
                    wr.writerow(res)
                print("  eval " + " | ".join(f"{k}={v:.3f}" for k, v in res.items() if isinstance(v, float)), flush=True)
                if res["select"] > best:
                    best, best_step = res["select"], step
                    save(run / "best.pt")
                if step >= args.steps or (args.patience and (step - best_step) >= args.patience * args.eval_every):
                    done = True
                save(run / "last.pt")
            if not done and time.time() > deadline:
                save(run / "last.pt")
                print(f"time limit ({args.max_hours:g} h) at step {step}: python scripts/train_meter.py --name "
                      f"{args.name} --resume goes on", flush=True)
                return
    except KeyboardInterrupt:
        save(run / "last.pt")
        sys.exit(130)
    what = {"meter": "mean of downbeat F and time-signature accuracy", "tempo": "share at the right level, "
            "the tempo head choosing", "both": "mean of the two"}[args.select]
    print(f"done: {step} steps; best validation score {best:.3f} ({what}) at step {best_step} -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
