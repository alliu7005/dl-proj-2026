"""Download the recordings behind Gangqinpu's transcriptions (for Beat This! on real audio:
scripts/track_meter_data.py), from the URLs the release lists.

    python scripts/fetch_gangqinpu_audio.py                    # val + test + 3,000 training recordings
    python scripts/fetch_gangqinpu_audio.py --train 0          # val + test only

The release (scripts/fetch_gangqinpu.py) does not redistribute the audio: every row has an
``audio_url`` (MP3s on the source site's file host, ~3 minutes each, ~3 MB) and the page it came
from (``source_page_url``), and the dataset's own tools/download_sources.py fetches them the same
way. The recordings belong to the site and its uploaders, not to the dataset's licence: for
non-commercial research only, nothing redistributed.

Politely: one request per --interval seconds (default 1), a descriptive User-Agent, the page as
Referer, --retries on failure, HTML answers (an error page, an expired link) refused. Resumable:
files already there are skipped, every attempt is appended to <out>/download_log.jsonl. Files go
to <out>/<split>/<sample_id>.mp3 (performance times are this audio's seconds). The training
recordings are a fixed draw (--train of them; the same ones on every run).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.gangqinpu import iter_rows, split_files  # noqa: E402

USER_AGENT = "dl-proj-2026 research fetcher (course project; non-commercial; Gangqinpu audio_url)"


def audio_path(out: Path, split: str, sample_id: str) -> Path:
    return Path(out) / split / f"{sample_id}.mp3"


def plan(release: Path, n_train: int) -> list[tuple[str, dict]]:
    """(split, row) to fetch: every val / test row, a fixed draw of ``n_train`` training rows."""
    jobs = []
    for split in ("val", "test"):
        if split_files(release, split):
            jobs += [(split, r) for r in iter_rows(release, split, columns=["sample_id", "audio_url", "source_page_url", "source"])]
    if n_train and split_files(release, "train"):
        rows = sorted(iter_rows(release, "train", columns=["sample_id", "audio_url", "source_page_url", "source"]),
                      key=lambda r: r["sample_id"])
        if len(rows) > n_train:
            rows = [rows[i] for i in sorted(np.random.default_rng(0).choice(len(rows), n_train, replace=False))]
        jobs += [("train", r) for r in rows]
    return [(s, r) for s, r in jobs if r.get("audio_url")]


def fetch(url: str, referer: str | None, dst: Path, timeout: float, opener=urllib.request.urlopen) -> int:
    """One file -> dst (written whole or not at all); its size. Raises on HTML or HTTP errors."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **({"Referer": referer} if referer else {})})
    with opener(req, timeout=timeout) as r:
        kind = (r.headers.get("Content-Type") or "").lower()
        if "text/html" in kind:
            raise ValueError(f"HTML instead of audio ({kind})")
        data = r.read()
    if len(data) < 1024:
        raise ValueError(f"only {len(data)} bytes")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dst)
    return len(data)


def main(argv=None, opener=urllib.request.urlopen, sleep=time.sleep):
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="data/raw/gangqinpu")
    ap.add_argument("--out", default="data/raw/gangqinpu/audio")
    ap.add_argument("--train", type=int, default=3000, help="training recordings to fetch (a fixed draw; 0 = none)")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="at most this many recordings (0 = all planned; a quick check)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    jobs = plan(Path(args.release), args.train)
    if args.limit:
        jobs = jobs[:args.limit]
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "download_log.jsonl", "a", encoding="utf-8")
    have = ok = failed = 0
    for k, (split, r) in enumerate(jobs):
        dst = audio_path(out, split, r["sample_id"])
        if dst.exists():
            have += 1
            continue
        err = None
        for attempt in range(args.retries):
            try:
                size = fetch(r["audio_url"], r.get("source_page_url"), dst, args.timeout, opener)
                err = None
                break
            except Exception as e:  # noqa: BLE001  (any failure of one file: logged, the next one goes on)
                err = f"{type(e).__name__}: {e}"
                sleep(args.interval * (2 ** attempt))
        log.write(json.dumps({"sample_id": r["sample_id"], "split": split, "ok": err is None,
                              **({"bytes": size} if err is None else {"error": err[:300]})}) + "\n")
        log.flush()
        ok, failed = ok + (err is None), failed + (err is not None)
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(jobs)}: {ok} fetched, {have} there already, {failed} failed", flush=True)
        sleep(args.interval)
    summary = {"planned": len(jobs), "fetched": ok, "already_there": have, "failed": failed}
    (out / "fetch_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
