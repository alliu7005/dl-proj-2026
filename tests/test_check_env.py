"""Tests for scripts/check_env.py: telling a file blocked by an Application Control policy (Smart App
Control) from a missing package or another failure, reading the blocked paths out of Windows' Code
Integrity log, and the script running on this machine.

    python -m tests.test_check_env        # or: pytest tests/test_check_env.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_env import CHECKS, classify, parse_blocked  # noqa: E402

BLOCK = """Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\\env\\Lib\\site-packages\\soxr\\__init__.py", line 10, in <module>
    from . import soxr_ext
ImportError: DLL load failed while importing soxr_ext: An Application Control policy has blocked this file.
"""


def test_classify():
    assert classify(0, "") == ("ok", "")
    assert classify(0, "no CUDA GPU: the CPU is used\n") == ("ok", "no CUDA GPU: the CPU is used")
    s, d = classify(1, BLOCK)
    assert s == "blocked" and d.startswith("DLL load failed while importing soxr_ext")
    s, d = classify(1, "OSError: [WinError 4551] An Application Control policy has blocked this file. "
                       "Error loading \"C:\\env\\torch\\lib\\c10.dll\" or one of its dependencies.")
    assert s == "blocked" and "4551" in d
    assert classify(1, "ModuleNotFoundError: No module named 'transkun'") == ("missing", "no module named transkun")
    assert classify(1, "Traceback...\nRuntimeError: CUDA error: no kernel image\n")[0] == "failed"
    assert classify(3, "") == ("failed", "exit code 3")


def test_parse_blocked():
    log = ("Code Integrity determined that a process (\\Device\\HarddiskVolume3\\Users\\a\\env\\python.exe) attempted "
           "to load \\Device\\HarddiskVolume3\\Users\\a\\env\\Lib\\site-packages\\soxr\\soxr_ext.pyd that did not "
           "meet the Enterprise signing level requirements or violated code integrity policy.\n"
           "Code Integrity determined that a process (\\Device\\HarddiskVolume3\\x\\python.exe) attempted to load "
           "\\Device\\HarddiskVolume3\\Users\\a\\env\\Lib\\site-packages\\soxr\\soxr_ext.pyd that did not meet ...\n"
           "Code Integrity determined that a process (...) attempted to load \\Device\\HarddiskVolume3\\Users\\a\\env"
           "\\Library\\bin\\sndfile.dll that did not meet the Windows signing level requirements.")
    assert parse_blocked(log) == ["\\Users\\a\\env\\Lib\\site-packages\\soxr\\soxr_ext.pyd",
                                  "\\Users\\a\\env\\Library\\bin\\sndfile.dll"]
    assert parse_blocked("") == []


def test_checks_cover_the_demo():
    names = {n for n, _, _ in CHECKS}
    for need in ("numpy", "torch", "transkun", "beat_this", "music21", "fastapi", "librosa", "soundfile"):
        assert need in names, need


def test_runs():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_env.py"), "--quick"], capture_output=True,
                       text=True, timeout=600)
    assert r.returncode in (0, 1, 2) and "python " in r.stdout and "numpy" in r.stdout, r.stdout + r.stderr


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
