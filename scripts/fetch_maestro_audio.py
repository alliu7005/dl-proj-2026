"""Download selected MAESTRO v3 recordings without fetching the whole 108 GB zip.

Reads the zip's central directory with HTTP range requests, then pulls only the requested members.

    # 40 MAESTRO test-split recordings for fitting the noise model
    python scripts/fetch_maestro_audio.py --split test --limit 40
    # audio for ASAP performances in a given ASAP split (real-transcription evaluation)
    python scripts/fetch_maestro_audio.py --asap-split test

Files land in data/raw/maestro/<year>/<name>.wav (+ .midi), mirroring the MAESTRO layout.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
PREFIX = "maestro-v3.0.0/"


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.pos, self.block = url, 0, block
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
        self._cache_start, self._cache = -1, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req) as r:
            return r.read()

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        c0, c1 = self._cache_start, self._cache_start + len(self._cache)
        if c0 <= self.pos and self.pos + n <= c1:
            out = self._cache[self.pos - c0:self.pos - c0 + n]
        elif n >= self.block:
            out = self._fetch(self.pos, self.pos + n)
        else:
            self._cache_start = self.pos
            self._cache = self._fetch(self.pos, min(self.pos + self.block, self.size))
            out = self._cache[:n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/maestro")
    ap.add_argument("--split", help="MAESTRO split to fetch (train/validation/test)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--asap-split", help="fetch audio for ASAP performances in this split")
    ap.add_argument("--asap", default="data/raw/asap_bd")
    ap.add_argument("--split-file", default="configs/splits/beyer_dai.json")
    args = ap.parse_args()

    out = Path(args.out)
    meta_csv = out / "maestro-v3.0.0.csv"
    if not meta_csv.exists():
        out.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.replace(".zip", ".csv"), meta_csv)
    rows = list(csv.DictReader(open(meta_csv, encoding="utf-8")))

    wanted = []
    if args.split:
        sel = sorted((r for r in rows if r["split"] == args.split), key=lambda r: r["audio_filename"])
        # spread the selection over years/pianos instead of taking the first N alphabetically
        if args.limit and len(sel) > args.limit:
            step = len(sel) / args.limit
            sel = [sel[int(i * step)] for i in range(args.limit)]
        wanted += sel
    if args.asap_split:
        pieces = json.load(open(args.split_file))["pieces"]
        by_midi = {r["midi_filename"]: r for r in rows}
        for r in csv.DictReader(open(Path(args.asap) / "metadata.csv")):
            mm = r["maestro_midi_performance"].replace("{maestro}/", "")
            if pieces.get(r["folder"]) == args.asap_split and mm in by_midi:
                wanted.append(by_midi[mm])

    members = {m for r in wanted for m in (r["audio_filename"], r["midi_filename"])}
    todo = [m for m in sorted(members) if not (out / m).exists()]
    print(f"{len(wanted)} recordings, {len(todo)} files to download")
    if not todo:
        return
    zf = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(URL), buffer_size=1 << 20))
    for i, m in enumerate(todo):
        info = zf.getinfo(PREFIX + m)
        dst = out / m
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with zf.open(info) as src, open(tmp, "wb") as f:
            shutil.copyfileobj(src, f, length=8 << 20)
        tmp.rename(dst)
        print(f"[{i + 1}/{len(todo)}] {m} ({info.file_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
