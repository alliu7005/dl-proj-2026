"""PDMX (Long et al. 2024, arXiv 2409.10831): ~250k public-domain MusicXML scores from MuseScore, on
Zenodo (CC-BY / public domain; the authors advise the no_license_conflict subset). Downloads the
record's files into data/raw/pdmx (resumable) and unpacks the MusicXML (.mxl) scores, for
scripts/build_synthetic_meter.py. Run on the login node (compute nodes may have no internet).

    python scripts/fetch_pdmx.py --list                    # the record's files and sizes
    python scripts/fetch_pdmx.py                           # the scores' archive -> data/raw/pdmx/mxl/
    python scripts/fetch_pdmx.py --file <name> --max 5000  # one file; unpack at most 5000 scores

Which file: --file, else the one whose name says mxl, else the largest archive. The metadata CSV
(PDMX.csv) comes first: only the solo-piano scores (every track a piano), one arrangement per song,
without a licence conflict, are unpacked (``piano_scores``; --all-scores: every one). Standard
library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RECORD = "15571083"   # PDMX v9 (2025-06-01); --record latest follows newer versions
API = "https://zenodo.org/api/records/"
ARCHIVES = (".tar.gz", ".tgz", ".tar", ".zip")


def files_of(record: str) -> list[dict]:
    url = API + (f"{RECORD}/versions/latest" if record == "latest" else record)
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.load(r)
    return [{"key": f["key"], "size": f.get("size", 0), "url": f["links"]["self"]} for f in d.get("files", [])]


def download(url: str, dst: Path, size: int = 0) -> None:
    """Resumable: a partial file goes on from where it stopped."""
    have = dst.stat().st_size if dst.exists() else 0
    if size and have == size:
        return
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "ab" if have else "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    print(f"{dst}: {dst.stat().st_size / 1e9:.2f} GB")


def flat(name: str) -> str:
    """An archive member's path as one file name (./mxl/1/11/x.mxl -> mxl__1__11__x.mxl)."""
    while name.startswith("./"):
        name = name[2:]
    return name.strip("/").replace("/", "__")


def piano_scores(csv_path: Path, dedup: bool = True) -> set:
    """The stems (file names without the extension) of PDMX's solo-piano scores from its metadata:
    every track a piano (General MIDI programs 0-7: ``tracks`` lists them, e.g. "0-0"), at most two
    tracks, no licence conflict, and with ``dedup`` one arrangement per song (subset:deduplicated)."""
    import csv
    csv.field_size_limit(1 << 30)
    out = set()
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                progs = [int(x) for x in str(r.get("tracks", "")).split("-") if x != ""]
                n = int(float(r.get("n_tracks") or 0))
            except ValueError:
                continue
            if not progs or n > 2 or any(p > 7 for p in progs):
                continue
            if str(r.get("license_conflict", "")).strip().lower() == "true":
                continue
            if dedup and str(r.get("subset:deduplicated", "True")).strip().lower() != "true":
                continue
            if r.get("mxl"):
                out.add(Path(r["mxl"]).stem)
    return out


def unpack(archive: Path, out: Path, limit: int = 0, keep: set | None = None) -> int:
    """The .mxl / .musicxml / .xml scores of an archive into ``out`` (flat names; at most ``limit``;
    with ``keep``, only the scores whose stem is in it)."""
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    ok = (".mxl", ".musicxml", ".xml")
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.lower().endswith(ok) and (keep is None or Path(name).stem in keep):
                    (out / flat(name)).write_bytes(z.read(name))
                    n += 1
                    if limit and n >= limit:
                        break
        return n
    with tarfile.open(archive, "r|*") as t:   # streamed: the first ``limit`` scores without unpacking the rest
        for m in t:
            if m.isfile() and m.name.lower().endswith(ok) and (keep is None or Path(m.name).stem in keep):
                f = t.extractfile(m)
                if f is not None:
                    (out / flat(m.name)).write_bytes(f.read())
                    n += 1
                    if limit and n >= limit:
                        break
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", default=RECORD, help="a Zenodo record id, or latest")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--file", default=None)
    ap.add_argument("--max", type=int, default=0, help="unpack at most this many scores (0 = all)")
    ap.add_argument("--out", default="data/raw/pdmx")
    ap.add_argument("--all-scores", action="store_true", help="unpack every score, not only the solo-piano ones")
    ap.add_argument("--with-duplicates", action="store_true", help="keep every arrangement of a song")
    args = ap.parse_args(argv)
    try:
        files = files_of(args.record)
    except Exception as e:  # noqa: BLE001
        print(f"Zenodo record {args.record}: not reached ({type(e).__name__}: {e}); on PACE run this on the login node")
        return 1
    if args.list or not files:
        for f in files:
            print(f"{f['size'] / 1e9:8.2f} GB  {f['key']}")
        return 0 if files else 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pick = [f for f in files if f["key"] == args.file] if args.file else \
        [f for f in files if "mxl" in f["key"].lower() and f["key"].lower().endswith(ARCHIVES)]
    if not pick:
        pick = sorted([f for f in files if f["key"].lower().endswith(ARCHIVES)], key=lambda f: -f["size"])[:1]
    if not pick:
        print("no archive in the record: --list, then --file <name>")
        return 1
    for f in files:   # the metadata (small)
        if f["key"].lower().endswith(".csv"):
            download(f["url"], out / f["key"], f["size"])
    f = pick[0]
    print(f"downloading {f['key']} ({f['size'] / 1e9:.2f} GB)")
    download(f["url"], out / f["key"], f["size"])
    keep = None
    meta = out / "PDMX.csv"
    if meta.exists() and not args.all_scores:
        keep = piano_scores(meta, dedup=not args.with_duplicates)
        print(f"{len(keep)} solo-piano scores without a licence conflict in {meta}")
    n = unpack(out / f["key"], out / "mxl", args.max, keep)
    print(f"{n} scores in {out / 'mxl'}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
