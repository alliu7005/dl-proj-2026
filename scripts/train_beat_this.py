"""Beat This! trained or fine-tuned on the datasets chosen, with its own training code (the pip
package's BeatDataModule and PLBeatThis, settings as its launch_scripts/train.py).

    # from scratch on its published data without ASAP (scripts/fetch_beat_this_data.py), as final0 was
    python scripts/train_beat_this.py --name bt_noasap --data data/beat_this_orig --exclude asap --no-val
    # fine-tuned on our piano data (scripts/export_beat_this.py): every fold, and fold k left out
    python scripts/train_beat_this.py --name bt_ft_full --data data/beat_this_ft --init outputs/beat_this/bt_noasap/final.ckpt --finetune
    python scripts/train_beat_this.py --name bt_ft_fold0 --data data/beat_this_ft --init outputs/beat_this/bt_noasap/final.ckpt --finetune --fold 0
    # the multi-task variant (src/bt_multitask.py: a time-signature head too), on every fold, as a set of its own
    python scripts/train_beat_this.py --name bt_ftm_full --data data/beat_this_ft --init outputs/beat_this/bt_noasap/final.ckpt --finetune --multitask
    python scripts/train_beat_this.py --assemble bt_ftm --full bt_ftm_full --full-only
    # the fine-tuned models as one set (src/bt_models.py: each performance tracked by the model that never heard its piece)
    python scripts/train_beat_this.py --assemble bt_ft --full bt_ft_full --fold-runs bt_ft_fold0,bt_ft_fold1,bt_ft_fold2,bt_ft_fold3 --pieces data/beat_this_ft/pieces.json

The datasets of one or more --data folders (Beat This!'s layout: annotations/<dataset>/, audio/
spectrograms/<dataset>.npz) are linked into the run's own data folder, the --exclude'd left out.
--fold k: the datasets' 8-folds.split parts other than k train, part k validates (our export
writes K folds there); without it single.split's train / val (or train on both with --no-val).
--init: start from a checkpoint's weights (a path or a Beat This! name such as final0), a new
optimiser and schedule. --finetune: a fine-tuning recipe (lr 1e-4, 200 warmup steps, 20 epochs,
no precomputed pitch / tempo variants, which our export does not make). --max-hours stops cleanly
before a job's limit; run again, it resumes (last.ckpt). Writes outputs/beat_this/<name>/: last.ckpt
(to resume), final.ckpt (the weights only: beat_this.inference loads it), status.json, metrics.csv.
--multitask: a head for each frame's time signature as well (the export's .meter files;
src/bt_multitask.py), its loss added (--meter-weight); dropped from final.ckpt.
Needs pytorch-lightning (pip install pytorch-lightning).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FT_RECIPE = {"lr": 1e-4, "warmup": 200, "max_epochs": 20, "pitch": False, "tempo": False}


def link(src: Path, dst: Path):
    """dst -> src (a symlink; a copy where links are not allowed, e.g. Windows without the right)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
    except OSError:
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)


def stage(data_dirs: list, run: Path, datasets: list | None, exclude: list) -> list:
    """The run's data folder with the chosen datasets linked in; returns their names. A dataset is
    one with annotations/<name>/ and audio/spectrograms/<name>.npz (or a folder); gtzan (Beat This!'s
    test set) is never trained on."""
    out = run / "data"
    chosen = []
    for d in map(Path, data_dirs):
        for ann in sorted((d / "annotations").glob("*")):
            name = ann.name
            if not ann.is_dir() or name in exclude or (datasets and name not in datasets) or name in chosen:
                continue
            spec = d / "audio" / "spectrograms"
            src = spec / f"{name}.npz" if (spec / f"{name}.npz").exists() else spec / name
            if not src.exists():
                print(f"  {name}: no spectrograms under {spec}, left out", flush=True)
                continue
            link(ann, out / "annotations" / name)
            link(src, out / "audio" / "spectrograms" / src.name)
            chosen.append(name)
    missing = sorted(set(datasets or []) - set(chosen))
    if missing:
        raise SystemExit(f"datasets not found in {data_dirs}: {missing}")
    if not [c for c in chosen if c != "gtzan"]:
        raise SystemExit(f"no dataset to train on in {data_dirs}")
    return chosen


def save_weights(ckpt_path: Path, out: Path):
    """A Lightning checkpoint -> its weights and hyper-parameters only (beat_this.inference.load_model
    reads it; no optimiser state; the multi-task variant's time-signature head left out)."""
    import torch

    from src.bt_multitask import strip_head
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    torch.save({"state_dict": strip_head(ck["state_dict"]), "hyper_parameters": dict(ck["hyper_parameters"])}, out)


def assemble(args) -> Path:
    from src.bt_models import ModelSet, write_models
    runs = Path(args.runs_dir)
    pieces = json.loads(Path(args.pieces).read_text())
    k = int(pieces["k"])
    fold_runs = [r for r in args.fold_runs.split(",") if r]
    if args.full_only:   # one model for everything (evaluation only: no cross-fitted training beats)
        if not (runs / args.full / "final.ckpt").exists():
            raise SystemExit(f"{runs / args.full}: not finished (no final.ckpt)")
        out = write_models(runs / args.assemble, k, str((runs / args.full / "final.ckpt").resolve()), {}, {})
        ModelSet(runs / args.assemble)
        print(f"wrote {out}: {args.full} alone", flush=True)
        return out
    if len(fold_runs) != k:
        raise SystemExit(f"--fold-runs: {len(fold_runs)} runs for k={k} folds")
    for r in [args.full, *fold_runs]:
        if not (runs / r / "final.ckpt").exists():
            raise SystemExit(f"{runs / r}: not finished (no final.ckpt)")
    folds = {}
    for r in fold_runs:
        st = json.loads((runs / r / "status.json").read_text())
        if st.get("fold") is None:
            raise SystemExit(f"{r}: trained without --fold")
        folds[int(st["fold"])] = str((runs / r / "final.ckpt").resolve())
    if sorted(folds) != list(range(k)):
        raise SystemExit(f"--fold-runs cover folds {sorted(folds)}, not 0..{k - 1}")
    out = write_models(runs / args.assemble, k, str((runs / args.full / "final.ckpt").resolve()), folds,
                       pieces["pieces"])
    ModelSet(runs / args.assemble)    # checks it
    print(f"wrote {out}: full {args.full}, folds {folds}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--runs-dir", default="outputs/beat_this")
    ap.add_argument("--data", nargs="+", default=["data/beat_this_ft"])
    ap.add_argument("--datasets", default="", help="only these (comma-separated); default all")
    ap.add_argument("--exclude", default="", help="leave these out (comma-separated), e.g. asap")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--init", default=None, help="start from these weights (a path or a Beat This! name)")
    ap.add_argument("--finetune", action="store_true", help=f"the fine-tuning recipe {FT_RECIPE}")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--no-pitch", action="store_true")
    ap.add_argument("--no-tempo", action="store_true")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--no-val", action="store_true", help="validation pieces trained on too (as final0)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accumulate", type=int, default=8)
    ap.add_argument("--train-length", type=int, default=1500)
    ap.add_argument("--oversampling", type=float, default=0.65, help="length-based oversampling factor")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=None, help="stop cleanly after this long (run again to resume)")
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--assemble", default=None, help="write a model set (models.json) of finished runs")
    ap.add_argument("--full", default=None)
    ap.add_argument("--fold-runs", default="")
    ap.add_argument("--full-only", action="store_true", help="--assemble: the full model alone (for evaluation)")
    ap.add_argument("--multitask", action="store_true", help="a time-signature head too (src/bt_multitask.py)")
    ap.add_argument("--meter-weight", type=float, default=0.2, help="--multitask: the time-signature loss's weight")
    ap.add_argument("--pieces", default="data/beat_this_ft/pieces.json")
    args = ap.parse_args(argv)
    if args.assemble:
        return assemble(args)
    if not args.name:
        raise SystemExit("--name")
    if args.fold is not None and args.no_val:
        # BeatDataModule would add the held-out fold to training: the leak cross-fitting is there to prevent
        raise SystemExit("--fold with --no-val would train on the held-out fold")
    import pytorch_lightning as pl
    import torch

    from src.bt_compat import patch_numpy_format
    patch_numpy_format()   # Beat This!'s spectrogram reader on numpy 2.5 (src/bt_compat.py)
    from beat_this.dataset.dataset import BeatDataModule
    from beat_this.inference import load_checkpoint
    from beat_this.model.pl_module import PLBeatThis

    recipe = {"lr": 8e-4, "warmup": 1000, "max_epochs": 100, "pitch": True, "tempo": True}
    if args.finetune:
        recipe.update(FT_RECIPE)
    lr = args.lr if args.lr is not None else recipe["lr"]
    warmup = args.warmup if args.warmup is not None else recipe["warmup"]
    epochs = args.max_epochs if args.max_epochs is not None else recipe["max_epochs"]
    run = Path(args.runs_dir) / args.name
    run.mkdir(parents=True, exist_ok=True)
    if (run / "final.ckpt").exists():
        print(f"{run}: finished already", flush=True)
        return run
    pl.seed_everything(args.seed, workers=True)
    names = stage(args.data, run, [d for d in args.datasets.split(",") if d], [d for d in args.exclude.split(",") if d])
    aug = {}
    if recipe["pitch"] and not args.no_pitch:
        aug["pitch"] = {"min": -5, "max": 6}
    if recipe["tempo"] and not args.no_tempo:
        aug["tempo"] = {"min": -20, "max": 20, "stride": 4}
    if not args.no_mask:
        aug["mask"] = {"kind": "permute", "min_count": 1, "max_count": 6, "min_len": 0.1, "max_len": 2,
                       "min_parts": 5, "max_parts": 9}
    if args.multitask:
        if "tempo" in aug:
            raise SystemExit("--multitask: without tempo augmentation (--no-tempo, or --finetune)")
        from src.bt_multitask import patch_dataset
        patch_dataset()   # its items carry each frame's meter targets (the export's .meter files)
        from beat_this.dataset.dataset import BeatDataModule  # noqa: F811  (the module's name, now patched)
    dm = BeatDataModule(run / "data", batch_size=args.batch_size, train_length=args.train_length, spect_fps=50,
                        num_workers=args.num_workers, test_dataset="gtzan",
                        length_based_oversampling_factor=args.oversampling, augmentations=aug,
                        no_val=args.no_val, fold=args.fold)
    dm.setup(stage="fit")
    pos_weights = dm.get_train_positive_weights(widen_target_mask=3)
    module = PLBeatThis
    extra = {}
    if args.multitask:
        from src.bt_multitask import multitask_module
        module, extra = multitask_module(), {"meter_weight": args.meter_weight}
    model = module(spect_dim=128, fps=50, transformer_dim=512, ff_mult=4, n_layers=6, stem_dim=32,
                       dropout={"frontend": 0.1, "transformer": 0.2}, lr=lr, weight_decay=0.01,
                       pos_weights=pos_weights, head_dim=32, loss_type="shift_tolerant_weighted_bce",
                       warmup_steps=warmup, max_epochs=epochs, use_dbn=False, eval_trim_beats=5,
                       sum_head=True, partial_transformers=True, **extra)
    resume = run / "last.ckpt"
    if args.init and not resume.exists():
        state = load_checkpoint(args.init, device="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if ".meter_ts." not in k]   # the new head: starts fresh
        if missing or unexpected:
            raise SystemExit(f"--init {args.init}: weights do not fit (missing {missing[:5]}, unexpected {unexpected[:5]})")
    status = {"name": args.name, "datasets": names, "fold": args.fold, "init": args.init, "lr": lr, "warmup": warmup,
              "max_epochs": epochs, "augmentations": sorted(aug), "no_val": args.no_val, "seed": args.seed,
              "train_items": len(dm.train_dataset) if hasattr(dm, "train_dataset") else None,
              "val_items": len(dm.val_dataset) if getattr(dm, "val_dataset", None) is not None else None,
              "pos_weights": pos_weights, "multitask": bool(args.multitask),
              "meter_weight": args.meter_weight if args.multitask else None, "done": False}
    (run / "status.json").write_text(json.dumps(status, indent=1, default=str))
    print(f"{args.name}: {json.dumps(status, default=str)}", flush=True)
    device = args.device
    if device == "auto":
        device = "gpu" if torch.cuda.is_available() else "cpu"
    ckpt = pl.callbacks.ModelCheckpoint(dirpath=str(run), save_top_k=0, save_last=True, every_n_epochs=1)
    trainer = pl.Trainer(max_epochs=epochs, accelerator=device, devices=1,
                         precision=args.precision if device == "gpu" else "32-true",
                         accumulate_grad_batches=args.accumulate, check_val_every_n_epoch=args.val_every,
                         num_sanity_val_steps=0, callbacks=[ckpt],
                         logger=pl.loggers.CSVLogger(str(run), name="", version=""),
                         max_time={"hours": args.max_hours} if args.max_hours else None,
                         enable_progress_bar=False, deterministic=False)
    t = time.perf_counter()
    trainer.fit(model, dm, ckpt_path=str(resume) if resume.exists() else None)
    finished = trainer.current_epoch >= epochs
    trainer.save_checkpoint(str(resume))
    status.update(epochs_done=int(trainer.current_epoch), hours=round((time.perf_counter() - t) / 3600, 2))
    if finished:
        save_weights(resume, run / "final.ckpt")
        status["done"] = True
    (run / "status.json").write_text(json.dumps(status, indent=1, default=str))
    print(f"{args.name}: {'finished' if finished else 'stopped (run again to resume)'} after epoch "
          f"{trainer.current_epoch} of {epochs}", flush=True)
    return run


if __name__ == "__main__":
    main()
