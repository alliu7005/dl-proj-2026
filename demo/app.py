"""Web demo: upload a piano recording, get the transcribed score back.

    python demo/app.py                                    # then open http://127.0.0.1:7860
    python demo/app.py --device cpu --port 8000
    python demo/app.py --labeller outputs/runs/nl_base_gn/best.pt --meter outputs/runs/meter_v3/best.pt                        --ckpt outputs/runs/sf_base_reg_cab/best.pt

Two systems, chosen per upload on the page (each loaded if its checkpoints exist):
  v2       (default) the meter model's grid of the whole piece from Beat This!'s beats, then the
           note labeller on every note (src/pipeline.py V2Pipeline; --labeller, --meter)
  seq2seq  the Scoreformer writing the score window by window (ScorePipeline; --ckpt), with its
           context modes
Everything runs in this process (Transkun, Beat This!, the models; on the GPU if there is one, else
the CPU); the page only uploads the file and shows the result. Jobs run one at a time in a
background thread; outputs go to outputs/demo/<job id>/ (input, transcription.mid, tokens.txt,
score.musicxml, score.mid, result.json). Also accepts .mid files (no transcription; v2 tracks the
beats of the MIDI rendered to audio).
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

from src.pipeline import AUDIO_EXT, MIDI_EXT, MODES, ScorePipeline, V2Pipeline  # noqa: E402

MAX_MB = 100
STAGES = {"transcribe": (0.0, 0.45), "beats": (0.45, 0.5), "grid": (0.5, 0.55), "score": (0.55, 0.95),
          "engrave": (0.95, 1.0)}
SYSTEMS = ("v2", "seq2seq")  # in order of preference for the default
# v2: when to leave a performed note out as a transcription error (the labeller's p(delete) above it).
# Transkun adds a note that was not played for 0.19% of notes (configs/noise_stats.json halluc_rate);
# at the trained 0.5 the labeller leaves out 4.6% of a real recording's notes, a third of them score
# notes on clean MIDI. Above 0.95 it leaves out about as many as Transkun adds (0.2% on clean MIDI,
# 4% of them score notes), the default.
DELETE = {"off": 1.0, "sure": 0.95, "trained": 0.5}
DOWNLOAD_NAMES = {"musicxml": "score.musicxml", "score_midi": "score.mid",
                  "transcription_midi": "transcription.mid", "tokens": "tokens.txt"}


def create_app(pipes: dict, out_root: Path, default: str | None = None) -> FastAPI:
    """``pipes``: {"v2": V2Pipeline, "seq2seq": ScorePipeline}, either or both."""
    if not pipes:
        raise ValueError("no system to serve")
    default = default if default in pipes else next(s for s in SYSTEMS if s in pipes)
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
                pipe = pipes[job["system"]]
                if job["system"] == "v2":
                    res = pipe.run(job["input"], job["dir"], progress=progress, delete_above=DELETE[job["delete"]])
                else:
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
        return {"systems": {k: p.info for k, p in pipes.items()}, "default": default, **pipes[default].info,
                "audio": sorted(AUDIO_EXT), "midi": sorted(MIDI_EXT), "max_mb": MAX_MB}

    @app.post("/api/jobs")
    async def submit(file: UploadFile = File(...), mode: str = Form(MODES[0]), system: str = Form(""),
                     delete: str = Form("sure")):
        ext = Path(file.filename or "").suffix.lower()
        system = system or default
        if ext not in AUDIO_EXT | MIDI_EXT:
            raise HTTPException(400, f"unsupported file type {ext or '(none)'}; use audio or .mid")
        if system not in pipes:
            raise HTTPException(400, f"system must be one of {', '.join(pipes)}")
        if mode not in MODES:
            raise HTTPException(400, f"mode must be one of {', '.join(MODES)}")
        if delete not in DELETE:
            raise HTTPException(400, f"delete must be one of {', '.join(DELETE)}")
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
                         "message": "waiting for the previous job", "name": file.filename, "mode": mode, "system": system,
                         "delete": delete,
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
    ap.add_argument("--ckpt", default="outputs/runs/sf_base_reg_cab/best.pt", help="the seq2seq model")
    ap.add_argument("--labeller", default="outputs/runs/nl_base_gn/best.pt", help="v2: the note labeller")
    ap.add_argument("--meter", default="outputs/runs/meter_v3/best.pt", help="v2: the meter model")
    ap.add_argument("--system", default="v2", choices=list(SYSTEMS), help="the page's default system")
    ap.add_argument("--vocab", default=None, help="default: next to the checkpoint, else data/processed")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other machines on the network")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--out", default="outputs/demo")
    args = ap.parse_args()
    import os
    os.chdir(ROOT)  # data/processed and outputs/ are relative to the repo
    pipes = {}
    if Path(args.labeller).exists() and Path(args.meter).exists():
        print(f"loading v2: {args.labeller} + {args.meter} ...", flush=True)
        pipes["v2"] = V2Pipeline(args.labeller, args.meter, args.vocab, args.device)
    else:
        print(f"no v2 ({args.labeller} / {args.meter} missing)", flush=True)
    if Path(args.ckpt).exists():
        print(f"loading seq2seq: {args.ckpt} ...", flush=True)
        pipes["seq2seq"] = ScorePipeline(args.ckpt, args.vocab, args.device)
    else:
        print(f"no seq2seq ({args.ckpt} missing)", flush=True)
    if not pipes:
        raise SystemExit("no checkpoints found: give --labeller and --meter (v2) and/or --ckpt (seq2seq)")
    for k, p in pipes.items():
        print(f"{k}: {p.info}", flush=True)
    app = create_app(pipes, Path(args.out), args.system)
    print(f"open http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
