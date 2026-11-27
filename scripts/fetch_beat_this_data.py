"""Beat This!'s published training data, for retraining it (scripts/train_beat_this.py).

    python scripts/fetch_beat_this_data.py                  # every dataset but ASAP -> data/beat_this_orig
    python scripts/fetch_beat_this_data.py --list           # what there is, and the sizes

From Zenodo record 13922116 ("Beat This! Spectrograms for Beat and Downbeat Tracking", CC-BY-4.0):
one zip per dataset holding <dataset>.npz (the spectrograms, with the pitch / tempo variants), and
beat_this_annotations.zip (the beats, splits and folds). Laid out as the training reads it:
  data/beat_this_orig/annotations/<dataset>/...
  data/beat_this_orig/audio/spectrograms/<dataset>.npz
--exclude (default asap: our retrained model must never have heard ASAP, which our benchmark and
our fine-tuning use). Every download is checked against its MD5, resumed if cut off, and its zip
deleted once unpacked; run again, it fetches only what is missing. About 108 GB without ASAP
(harmonix alone 45 GB); --exclude harmonix if space is short.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/13922116"
ANNOTATIONS = "beat_this_annotations"


def listing(opener=urllib.request.urlopen) -> list[dict]:
    """[{name, size, md5, url}] of the record's files."""
    with opener(RECORD, timeout=60) as r:
        meta = json.loads(r.read())
    out = []
    for f in meta["files"]:
        out.append({"name": f["key"], "size": int(f["size"]), "md5": f["checksum"].split(":", 1)[-1],
                    "url": f["links"]["self"]})
    return out


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def download(f: dict, dst: Path, opener=urllib.request.urlopen, retries: int = 5) -> Path:
    """The file, resumed where an earlier try stopped, checked against its MD5."""
    for attempt in range(retries):
        have = dst.stat().st_size if dst.exists() else 0
        if have < f["size"]:
            req = urllib.request.Request(f["url"], headers={"Range": f"bytes={have}-"} if have else {})
            try:
                with opener(req, timeout=120) as r, open(dst, "ab" if have else "wb") as out:
                    if have and getattr(r, "status", 206) != 206:   # no range support: from the start
                        out.truncate(0)
                    shutil.copyfileobj(r, out, 1 << 24)
            except Exception as e:  # noqa: BLE001
                print(f"  {f['name']}: {e} (try {attempt + 1} of {retries})", flush=True)
                time.sleep(min(60, 5 * (attempt + 1)))
                continue
        size = dst.stat().st_size if dst.exists() else 0
        if size == f["size"]:
            if md5(dst) == f["md5"]:
                return dst
            print(f"  {f['name']}: MD5 wrong, downloading again", flush=True)
            dst.unlink()
        elif size > f["size"]:
            dst.unlink()
        # shorter: cut off, the next try goes on from there
    raise SystemExit(f"{f['name']}: failed after {retries} tries")


def unpack_spectrograms(zf: Path, dataset: str, out: Path) -> Path:
    """<dataset>.npz out of its zip, streamed (no second copy in memory)."""
    target = out / "audio" / "spectrograms" / f"{dataset}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zf) as z:
        member = next((n for n in z.namelist() if n.endswith(f"{dataset}.npz")), None)
        if member is None:
            raise SystemExit(f"{zf}: no {dataset}.npz inside ({z.namelist()[:5]})")
        tmp = target.with_suffix(".npz.part")
        with z.open(member) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
        tmp.replace(target)
    return target


def unpack_annotations(zf: Path, out: Path) -> Path:
    """annotations/<dataset>/ from the annotations zip, whatever folder it is nested in (the one
    holding the dataset folders, found by their info.json)."""
    ann = out / "annotations"
    tmp = out / "annotations.part"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zf) as z:
        z.extractall(tmp)
    infos = sorted(tmp.rglob("info.json"), key=lambda p: len(p.parts))
    if not infos:
        raise SystemExit(f"{zf}: no dataset folder (info.json) inside")
    root = infos[0].parent.parent
    ann.mkdir(parents=True, exist_ok=True)
    for d in root.iterdir():
        if d.is_dir():
            shutil.rmtree(ann / d.name, ignore_errors=True)
            shutil.move(str(d), str(ann / d.name))
    shutil.rmtree(tmp, ignore_errors=True)
    return ann


def main(argv=None, opener=urllib.request.urlopen):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/beat_this_orig")
    ap.add_argument("--exclude", default="asap", help="datasets not fetched (comma-separated)")
    ap.add_argument("--only", default="", help="only these datasets (comma-separated)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    files = listing(opener)
    if args.list:
        for f in files:
            print(f"{f['name']:32s} {f['size'] / 1e9:7.1f} GB")
        return files
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exclude = {x for x in args.exclude.split(",") if x}
    only = {x for x in args.only.split(",") if x}
    done = []
    for f in files:
        name = f["name"].removesuffix(".zip")
        if name == ANNOTATIONS:
            if (out / "annotations").is_dir() and any((out / "annotations").glob("*/info.json")):
                continue
            zf = download(f, out / f["name"], opener)
            unpack_annotations(zf, out)
            zf.unlink()
            done.append(name)
            continue
        if name in exclude or (only and name not in only):
            continue
        if (out / "audio" / "spectrograms" / f"{name}.npz").exists():
            continue
        print(f"{name}: {f['size'] / 1e9:.1f} GB", flush=True)
        zf = download(f, out / f["name"], opener)
        unpack_spectrograms(zf, name, out)
        zf.unlink()
        done.append(name)
        print(f"{name}: unpacked", flush=True)
    have = sorted(p.stem for p in (out / "audio" / "spectrograms").glob("*.npz"))
    print(f"fetched {done}; spectrograms here: {have} (excluded: {sorted(exclude)})", flush=True)
    return have


if __name__ == "__main__":
    main()
