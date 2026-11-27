"""The articulation augmentation (src/dataset.py articulate), its training flag, the pilot's recipes
and batch (pace/v2_articulation.sh).

    python -m tests.test_articulation
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _notes(n=200, seed=0):
    rng = np.random.default_rng(seed)
    on = np.sort(rng.uniform(0, 20, n)).astype(np.float32)
    pitch = rng.integers(36, 90, n).astype(np.float32)
    return np.stack([pitch, on, rng.uniform(0.1, 0.6, n).astype(np.float32), rng.integers(30, 100, n).astype(np.float32)], 1)


def test_articulate():
    from src.dataset import articulate
    x = _notes()
    y = articulate(x, np.random.default_rng(1))
    assert y.shape == x.shape and y is not x
    assert np.array_equal(y[:, [0, 1, 3]], x[:, [0, 1, 3]])          # pitches, onsets, velocities stay
    assert not np.allclose(y[:, 2], x[:, 2]) and (y[:, 2] >= 0.03 - 1e-6).all()
    r = y[:, 2] / x[:, 2]
    assert (r < 0.61).any() and (r > 1.49).any()                      # detached and held stretches both drawn
    assert np.array_equal(x, _notes())                                 # the input is left as it was
    assert len(articulate(x[:0], np.random.default_rng(0))) == 0
    # legato: every note reaches its hand's next onset (one stretch, one hand, the mode forced)
    import src.dataset as D
    keep = D.ARTIC_MODES
    try:
        D.ARTIC_MODES = (("legato", 1.0),)
        h = x[x[:, 0] >= 60]
        z = articulate(h, np.random.default_rng(2))
        nxt = np.array([np.min(h[h[:, 1] > o, 1], initial=np.inf) for o in h[:, 1]])
        ok = np.isfinite(nxt)
        assert (z[ok, 2] >= np.maximum(h[ok, 2], nxt[ok] - h[ok, 1]) - 1e-4).all()
    finally:
        D.ARTIC_MODES = keep


def test_augment_config_and_flag():
    from src.dataset import AugmentConfig
    assert AugmentConfig().articulation == 0.0                         # off unless asked: earlier runs unchanged
    src = (ROOT / "scripts" / "train_labeller.py").read_text()
    assert "--articulation" in src and src.count("AugmentConfig(articulation=args.articulation)") == 3


def test_recipes():
    lib = (ROOT / "pace" / "v2_jobs.sh").read_text()
    r = subprocess.run(["bash", "-c", 'eval "$(sed -n "/^nl_flags()/,/^}/p" pace/v2_jobs.sh)"; nl_flags nl_gq_p20k; nl_flags nl_gq_art_p20k'],
                       cwd=ROOT, capture_output=True, text=True)
    a, b = r.stdout.strip().splitlines()
    assert "--max-steps 20000" in a and "--articulation" not in a, r.stdout + r.stderr
    assert b == a + " --articulation 0.5", (a, b)
    assert "nl_gq_art_p20k" in lib


def test_articulation_batch():
    import shutil

    from tests.batch_env import batch_env
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "pace").mkdir()
        shutil.copy(ROOT / "pace" / "v2_articulation.sh", tmp / "pace")
        for f in ("data/processed/train_gangqinpu/notelabels.npz", "data/processed/vocab.json",
                  "outputs/runs/meter_v4_gq/best.pt", "outputs/system_eval_gangqinpu/manifest.json",
                  "outputs/system_eval/manifest.json", "bd/MIDI2ScoreTF.ckpt"):
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_text("x")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "sbatch").write_text('#!/bin/bash\nn=$(( $(cat "$SUBMITS.n" 2>/dev/null || echo 100) + 1 ))\n'
                                            'echo $n > "$SUBMITS.n"\nprintf "%s\\t%s\\n" "$n" "$*" >> "$SUBMITS"\necho $n\n')
        (tmp / "bin" / "sbatch").chmod(0o755)
        r = subprocess.run(["bash", "-c", 'export PATH="$PWD/bin:$PATH"; bash pace/v2_articulation.sh'], cwd=tmp,
                           env=batch_env(SUBMITS=str(tmp / "jobs.txt"), USER="me", BD=str(tmp / "bd")),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        by = {ln.split("\t")[1].split("-J ")[1].split()[0]: ln.split("\t") for ln in (tmp / "jobs.txt").read_text().splitlines()}
        assert list(by) == ["a1tests", "a1train", "a1asap", "a1gq", "a1final"]
        assert f"afterok:{by['a1tests'][0]}" in by["a1train"][1] and "labellers nl_gq_p20k nl_gq_art_p20k" in by["a1train"][1]
        assert "nl_gq_art_p20k+meter_v4_gq:tidy,d95,missing" in by["a1gq"][1] and "--conditions gangqinpu" in by["a1gq"][1]
        assert f"afterany:{by['a1asap'][0]}:{by['a1gq'][0]}" in by["a1final"][1] and "articulation_report.py" in by["a1final"][1]


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
