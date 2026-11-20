"""Web demo: upload a piano recording, get the transcribed score back.

    python demo/app.py --ckpt outputs/runs/sf_base_reg_cab/best.pt         # then open http://127.0.0.1:7860
    python demo/app.py --ckpt outputs/runs/sf_base_reg_cab/best.pt --device cpu --port 8000

Everything runs in this process (Transkun + the score model, on the GPU if there is one, else the
CPU); the page only uploads the file and shows the result. Jobs run one at a time in a background
thread; outputs go to outputs/demo/<job id>/ (input, transcription.mid, tokens.txt,
score.musicxml, score.mid, result.json). Also accepts .mid files (skips transcription).
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402

from src.pipeline import AUDIO_EXT, MIDI_EXT, MODES, ScorePipeline  # noqa: E402

MAX_MB = 100
STAGES = {"transcribe": (0.0, 0.45), "score": (0.45, 0.95), "engrave": (0.95, 1.0)}
DOWNLOAD_NAMES = {"musicxml": "score.musicxml", "score_midi": "score.mid",
                  "transcription_midi": "transcription.mid", "tokens": "tokens.txt"}


def create_app(pipe: ScorePipeline, out_root: Path) -> FastAPI:
    app = FastAPI(title="Piano to score demo")
    jobs: dict[str, dict] = {}
    todo: queue.Queue = queue.Queue()
    lock = threading.Lock()

    def update(jid, **kw):
        with lock:
            jobs[jid].update(kw)

    def worker():
        while True:
            jid = todo.get()
            job = jobs[jid]
            update(jid, status="running", started=time.time())

            def progress(stage, frac, msg):
                lo, hi = STAGES.get(stage, (0, 1))
                update(jid, stage=stage, message=msg, progress=round(lo + (hi - lo) * frac, 3))

            try:
                res = pipe.run(job["input"], job["dir"], mode=job["mode"], progress=progress)
                update(jid, status="done", progress=1.0, stage="done", message="done", stats=res.stats,
                       errors=res.errors, files={k: str(v) for k, v in res.files.items()},
                       elapsed=round(time.time() - job["started"], 1))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                update(jid, status="error", message=f"{type(e).__name__}: {e}")
            finally:
                todo.task_done()

    threading.Thread(target=worker, daemon=True).start()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/api/info")
    def info():
        return {**pipe.info, "audio": sorted(AUDIO_EXT), "midi": sorted(MIDI_EXT), "max_mb": MAX_MB}

    @app.post("/api/jobs")
    async def submit(file: UploadFile = File(...), mode: str = Form(MODES[0])):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in AUDIO_EXT | MIDI_EXT:
            raise HTTPException(400, f"unsupported file type {ext or '(none)'}; use audio or .mid")
        if mode not in MODES:
            raise HTTPException(400, f"mode must be one of {', '.join(MODES)}")
        data = await file.read()
        if len(data) > MAX_MB * 2**20:
            raise HTTPException(413, f"file larger than {MAX_MB} MB")
        jid = uuid.uuid4().hex[:12]
        d = out_root / jid
        d.mkdir(parents=True)
        src = d / f"input{ext}"
        src.write_bytes(data)
        with lock:
            jobs[jid] = {"id": jid, "status": "queued", "stage": "queued", "progress": 0.0,
                         "message": "waiting for the previous job", "name": file.filename, "mode": mode,
                         "input": str(src), "dir": str(d), "is_midi": ext in MIDI_EXT, "created": time.time()}
        todo.put(jid)
        return {"id": jid}

    @app.get("/api/jobs/{jid}")
    def status(jid: str):
        with lock:
            job = jobs.get(jid)
            if job is None:
                raise HTTPException(404, "unknown job")
            ahead = sum(j["status"] == "queued" and j["created"] < job["created"] for j in jobs.values())
            out = {k: v for k, v in job.items() if k not in ("input", "dir", "files")}
            out["downloads"] = sorted(job.get("files", {}).keys() & DOWNLOAD_NAMES.keys())
            out["queue_ahead"] = ahead
        return out

    @app.get("/api/jobs/{jid}/files/{kind}")
    def download(jid: str, kind: str):
        job = jobs.get(jid)
        if job is None or kind not in job.get("files", {}) or kind not in DOWNLOAD_NAMES:
            raise HTTPException(404, "no such file")
        stem = Path(job["name"] or "score").stem
        return FileResponse(job["files"][kind], filename=f"{stem}_{DOWNLOAD_NAMES[kind]}")

    @app.get("/api/jobs/{jid}/input")
    def original(jid: str):
        job = jobs.get(jid)
        if job is None:
            raise HTTPException(404, "unknown job")
        return FileResponse(job["input"])

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="outputs/runs/sf_base_reg_cab/best.pt")
    ap.add_argument("--vocab", default=None, help="default: next to the checkpoint, else data/processed")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other machines on the network")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--out", default="outputs/demo")
    args = ap.parse_args()
    import os
    os.chdir(ROOT)  # data/processed and outputs/ are relative to the repo
    print(f"loading {args.ckpt} ...", flush=True)
    pipe = ScorePipeline(args.ckpt, args.vocab, args.device)
    print(f"model: {pipe.info}", flush=True)
    app = create_app(pipe, Path(args.out))
    print(f"open http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
