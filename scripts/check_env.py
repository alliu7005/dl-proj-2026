"""Checks that this Python can load everything the demo needs, and on Windows says which files an
Application Control policy (Smart App Control, or a policy set by an organisation's IT) blocks.

    python scripts/check_env.py            # every package the demo and the CLI load
    python scripts/check_env.py --quick    # imports only (no GPU / audio probes)

Each package is loaded in its own process, so one blocked file does not hide the others. A blocked
file shows up as "DLL load failed ... An Application Control policy has blocked this file"; the
script then reads Windows' Code Integrity log for the exact paths and prints the ways around it
(docs/demo.md, "Windows: files blocked by Application Control"). Exit code: 0 all fine, 1 something
missing or failing, 2 something blocked by policy.

Standard library only, so it runs before anything is installed.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# (name shown, what it is for, probe run in a fresh interpreter). The probes go a little past the
# import where a package loads its native libraries late (CUDA kernels, numba's LLVM, libsndfile).
CHECKS = [
    ("numpy", "everything", "import numpy; numpy.linalg.svd(numpy.eye(3))"),
    ("torch", "the models", "import torch; torch.nn.functional.conv1d(torch.zeros(1, 1, 8), torch.zeros(1, 1, 3))"),
    ("torch (GPU)", "the models on an NVIDIA GPU (optional: the CPU works)",
     "import torch, sys\n"
     "if not torch.cuda.is_available(): print('no CUDA GPU: the CPU is used'); sys.exit(0)\n"
     "x = torch.zeros(1, 1, 8, 8, device='cuda'); torch.nn.functional.conv2d(x, torch.zeros(1, 1, 3, 3, device='cuda'))\n"
     "(torch.zeros(4, 4, device='cuda') @ torch.zeros(4, 4, device='cuda')).sum().item()"),
    ("torchaudio", "transcription", "import torchaudio"),
    ("soundfile", "reading audio", "import soundfile; soundfile.available_formats()"),
    ("soxr", "resampling", "import soxr, numpy; soxr.resample(numpy.zeros(1600, 'float32'), 16000, 8000)"),
    ("librosa", "audio features", "import numba, llvmlite.binding, librosa"),
    ("pydub", "transcription", "import pydub"),
    ("mir_eval", "transcription", "import mir_eval"),
    ("moduleconf", "transcription", "import moduleconf"),
    ("transkun", "transcription (pip install --no-deps transkun==2.0.1)", "import transkun"),
    ("beat_this", "beat tracking", "import beat_this, rotary_embedding_torch, einops"),
    ("music21", "the score", "import music21"),
    ("pretty_midi", "MIDI", "import pretty_midi, mido"),
    ("huggingface_hub", "downloading the models", "import huggingface_hub"),
    ("fastapi", "the web demo", "import fastapi, uvicorn, multipart"),
]
GPU = {"torch (GPU)"}

BLOCKED = re.compile(r"Application Control policy|WinError 4551|has blocked this file", re.I)
MISSING = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")
DLL = re.compile(r"(DLL load failed while importing \S+: [^\r\n]*|\[WinError \d+\][^\r\n]*)")
LOADED = re.compile(r"attempted to load (.+?) that did not", re.I)


def classify(returncode: int, out: str) -> tuple[str, str]:
    """(status, detail) from a probe's exit code and output: ok / missing / blocked / failed."""
    if returncode == 0:
        return "ok", out.strip().splitlines()[-1] if out.strip() else ""
    if BLOCKED.search(out):
        m = DLL.search(out)
        return "blocked", (m.group(1) if m else "a file was blocked by an Application Control policy").strip()
    m = MISSING.search(out)
    if m:
        return "missing", f"no module named {m.group(1)}"
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    return "failed", (lines[-1] if lines else f"exit code {returncode}")[:300]


def probe(code: str, timeout: int = 300) -> tuple[str, str]:
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "PYTHONWARNINGS": "ignore"})
    except subprocess.TimeoutExpired:
        return "failed", f"no answer within {timeout} s"
    except OSError as e:   # the interpreter itself could not be started
        return ("blocked" if BLOCKED.search(str(e)) else "failed"), str(e)
    return classify(r.returncode, (r.stdout or "") + "\n" + (r.stderr or ""))


def smart_app_control() -> str | None:
    """'on', 'evaluation', 'off' or None (not Windows, or not known)."""
    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\CI\Policy") as k:
            v, _ = winreg.QueryValueEx(k, "VerifiedAndReputablePolicyState")
    except OSError:
        return None
    return {0: "off", 1: "on", 2: "evaluation"}.get(v, f"state {v}")


def blocked_files(limit: int = 200) -> list[str]:
    """The files Code Integrity refused to load recently (events 3033 / 3077), newest first; empty if
    the log cannot be read."""
    if os.name != "nt":
        return []
    ps = ("Get-WinEvent -LogName 'Microsoft-Windows-CodeIntegrity/Operational' -MaxEvents %d -ErrorAction Stop | "
          "Where-Object { $_.Id -in 3033,3077 } | ForEach-Object { $_.Message }" % limit)
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_blocked(r.stdout or "")


def parse_blocked(text: str) -> list[str]:
    out = []
    for m in LOADED.finditer(text):
        f = re.sub(r"^\\Device\\HarddiskVolume\d+", "", m.group(1).strip())
        if f not in out:
            out.append(f)
    return out


ADVICE = """
Some files were blocked by an Application Control policy. Windows refuses to load programs and
libraries it does not trust; Python packages ship many unsigned libraries (.dll / .pyd), so Smart
App Control blocks some of them. Ways around it (docs/demo.md, "Windows: files blocked by
Application Control"):
  1. Run the demo in WSL (Linux on Windows): the policy does not apply to Linux programs, and the
     GPU still works. Nothing in Windows' security is changed.
  2. Turn Smart App Control off (Windows Security > App & browser control > Smart App Control).
     That is a security trade-off for the computer's owner to decide; read Windows' own warning first.
  3. A computer managed by a school or employer: the policy is IT's; ask them, or use WSL if it is
     allowed, or another computer."""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="imports only")
    args = ap.parse_args(argv)
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    sac = smart_app_control()
    if sac:
        print(f"Smart App Control: {sac}")
    results = []
    for name, what, code in CHECKS:
        if args.quick and name in GPU:
            continue
        if args.quick:
            code = ";".join(p for p in code.split(";") if p.strip().startswith("import"))
        status, detail = probe(code)
        results.append((name, status))
        print(f"  {status:8} {name:16} {what}" + (f"\n           {detail}" if detail and status != "ok" else
                                                    (f"  ({detail})" if detail else "")), flush=True)
    blocked = [n for n, s in results if s == "blocked"]
    bad = [n for n, s in results if s in ("missing", "failed")]
    if blocked:
        files = blocked_files()
        if files:
            print("\nfiles Windows refused to load (Code Integrity log, newest first):")
            for f in files[:20]:
                print(f"  {f}")
        else:
            print("\n(the Code Integrity log could not be read: Windows Security > App & browser control > "
                  "Protection history lists the blocked files)")
        print(ADVICE)
    if bad:
        print("\nmissing or failing: " + ", ".join(bad) + ". Install the environment: conda env create -f "
              "environment.yml, then pip install --no-deps transkun==2.0.1 (README.md).")
    if not blocked and not bad:
        print("\neverything loads.")
    return 2 if blocked else 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
