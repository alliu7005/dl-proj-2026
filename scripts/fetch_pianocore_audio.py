"""The recordings of PianoCoRe performances whose source links to them (ATEPP, GiantMIDI-Piano), for
re-transcribing them with Transkun (scripts/retranscribe_pianocore.py) and for Beat This! (real
classical recordings, scripts/export_beat_this.py).

    python scripts/fetch_pianocore_audio.py --dir data/raw/pianocore/atepp
        -> <dir>/audio/<performance id>.mp3, <dir>/audio/fetch_report.json

Links: ATEPP's metadata (ATEPP-metadata-1.2.csv, its youtube_links column; fetched once from the
ATEPP repository), GiantMIDI-Piano's file names (the YouTube video id). Downloaded with yt-dlp
(pip install yt-dlp; ffmpeg converts to mp3), one at a time with --sleep seconds between, so the
site is not hammered; run again, it fetches only what is missing (and retries failures with
--retry-failed). Videos taken down are recorded as unavailable. Some networks get YouTube's "not a
bot" check: --cookies <a cookies.txt from a logged-in browser>.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ATEPP_META = "https://raw.githubusercontent.com/tangjjbetsy/ATEPP/master/ATEPP-metadata-1.2.csv"
YT = re.compile(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})")


def atepp_links(path: Path, opener=urllib.request.urlopen) -> dict:
    """ATEPP perf_id (5 digits) -> YouTube URL, from its metadata (downloaded to ``path`` once)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with opener(ATEPP_META, timeout=120) as r:
            path.write_bytes(r.read())
    out = {}
    for row in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8", errors="replace"))):
        pid, link = (row.get("perf_id") or "").strip(), (row.get("youtube_links") or "").strip()
        if pid and YT.search(link):
            out[pid.zfill(5)] = link
    return out


def link_of(perf_id: str, atepp: dict) -> str | None:
    """The recording's URL for a PianoCoRe performance id, None if its source gives none."""
    if perf_id.startswith("ATEPP_"):
        return atepp.get(perf_id.split("_", 1)[1].zfill(5))
    if perf_id.startswith("GiantMIDI_"):
        vid = perf_id.split("_", 1)[1]
        return f"https://www.youtube.com/watch?v={vid}" if len(vid) == 11 else None
    return None


def ytdlp_download(url: str, out_stem: Path, cookies: str | None = None) -> Path:
    """The audio of ``url`` as <out_stem>.mp3 (yt-dlp, ffmpeg)."""
    import yt_dlp
    opts = {"format": "bestaudio/best", "outtmpl": str(out_stem) + ".%(ext)s", "quiet": True, "noprogress": True,
            "noplaylist": True, "retries": 3,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}]}
    if cookies:
        opts["cookiefile"] = cookies
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([url])
    out = out_stem.with_suffix(".mp3")
    if not out.exists():
        raise RuntimeError("no mp3 written")
    return out


UNAVAILABLE = ("unavailable", "private", "removed", "terminated", "copyright", "not available", "does not exist")


def main(argv=None, download=None, opener=urllib.request.urlopen):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="a fetch_pianocore.py folder (its selection.parquet)")
    ap.add_argument("--atepp-meta", default="data/raw/pianocore/ATEPP-metadata-1.2.csv")
    ap.add_argument("--sleep", type=float, default=3.0, help="seconds between downloads")
    ap.add_argument("--limit", type=int, default=0, help="at most this many downloads this run (0: all)")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--cookies", default=None)
    args = ap.parse_args(argv)
    d = Path(args.dir)
    sel = pd.read_parquet(d / "selection.parquet", columns=["id"])
    audio = d / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    rep_f = audio / "fetch_report.json"
    report = json.loads(rep_f.read_text()) if rep_f.exists() else {}
    need_atepp = any(str(i).startswith("ATEPP_") for i in sel["id"])
    atepp = atepp_links(Path(args.atepp_meta), opener) if need_atepp else {}
    download = download or (lambda url, stem: ytdlp_download(url, stem, args.cookies))
    done = 0
    for pid in sel["id"].astype(str):
        if (audio / f"{pid}.mp3").exists():
            report[pid] = "ok"
            continue
        if report.get(pid) == "unavailable" or (report.get(pid, "").startswith("error") and not args.retry_failed):
            continue
        url = link_of(pid, atepp)
        if url is None:
            report[pid] = "no link"
            continue
        if args.limit and done >= args.limit:
            break
        try:
            download(url, audio / pid)
            report[pid] = "ok"
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            report[pid] = "unavailable" if any(w in msg for w in UNAVAILABLE) else f"error: {str(e)[:200]}"
            if "not a bot" in msg or "sign in to confirm" in msg:
                print("YouTube asks to confirm this is not a bot: give --cookies (a cookies.txt of a logged-in "
                      "browser) and run again", flush=True)
                rep_f.write_text(json.dumps(report, indent=1))
                raise SystemExit(2)
        done += 1
        if done % 25 == 0:
            rep_f.write_text(json.dumps(report, indent=1))
            counts = pd.Series([v if not v.startswith("error") else "error" for v in report.values()]).value_counts()
            print(f"{done} tried: {counts.to_dict()}", flush=True)
        time.sleep(args.sleep)
    rep_f.write_text(json.dumps(report, indent=1))
    counts = pd.Series([v if not v.startswith("error") else "error" for v in report.values()]).value_counts().to_dict()
    print(f"{d.name}: {counts} of {len(sel)} performances", flush=True)
    return report


if __name__ == "__main__":
    main()
