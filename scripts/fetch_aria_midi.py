"""Aria-MIDI (Bradshaw & Colton, ICLR 2025; CC-BY-NC-SA 4.0): ~1.2M solo-piano recordings transcribed
to MIDI, for pretraining the meter model (scripts/pretrain_meter.py). Streams one subset's archive
from Hugging Face and keeps the first ``--max-files`` MIDI files, flattened to safe file names.

    python scripts/fetch_aria_midi.py                          # the unique subset (32,522 files)
    python scripts/fetch_aria_midi.py --subset deduped --max-files 60000
    -> data/raw/aria_midi/<subset>/*.mid (+ metadata.json when the archive has one, fetch.json)

Subsets (the dataset's README): unique (one performance per composition), deduped, pruned, full.
The files are themselves transcriptions (like our real inputs), with no score: the pretraining is
self-supervised.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/loubb/aria-midi/resolve/main/"
SUBSETS = {"unique": "aria-midi-v1-unique-ext.tar.gz", "deduped": "aria-midi-v1-deduped-ext.tar.gz",
           "pruned": "aria-midi-v1-pruned-ext.tar.gz", "full": "aria-midi-v1-ext.tar.gz"}


def safe_name(member: str) -> str:
    """A flat file name from an archive path (no directories, only [A-Za-z0-9_.-])."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(member).name)[:200]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="unique", choices=list(SUBSETS))
    ap.add_argument("--max-files", type=int, default=40000)
    ap.add_argument("--out", default="data/raw/aria_midi")
    args = ap.parse_args(argv)
    out = Path(args.out) / args.subset
    out.mkdir(parents=True, exist_ok=True)
    have = len(list(out.glob("*.mid")))
    if have >= args.max_files:
        print(f"{out}: {have} files already")
        return 0
    url = BASE + SUBSETS[args.subset] + "?download=true"
    print(f"streaming {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "dl-proj-fetch/1.0"})
    n, t0, meta = 0, time.time(), False
    with urllib.request.urlopen(req, timeout=120) as r, tarfile.open(fileobj=r, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            name = safe_name(m.name)
            if name.endswith("metadata.json") and not meta:
                f = tar.extractfile(m)
                if f is not None:
                    (out / "metadata.json").write_bytes(f.read())
                    meta = True
                continue
            if not name.lower().endswith((".mid", ".midi")) or m.size > 5_000_000:
                continue
            dst = out / name
            if not dst.exists():
                f = tar.extractfile(m)
                if f is None:
                    continue
                dst.write_bytes(f.read())
            n += 1
            if n % 2000 == 0:
                print(f"  {n} files ({time.time() - t0:.0f} s)", flush=True)
            if n >= args.max_files:
                break
    (out / "fetch.json").write_text(json.dumps({"subset": args.subset, "url": url, "files": n,
                                                "metadata": meta, "license": "CC-BY-NC-SA-4.0"}, indent=1))
    print(f"{out}: {n} MIDI files" + (" + metadata.json" if meta else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
