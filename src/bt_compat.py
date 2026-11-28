"""Beat This! 1.1.0 with numpy 2.5.

Its memory-mapped spectrogram reader (beat_this/dataset/mmnpz.py) parses each .npy header with
numpy's private helpers ``np.lib.format._check_version`` and ``_read_array_header``. numpy 2.x
moved them to ``numpy.lib._format_impl`` and ``numpy.lib.format`` stopped exposing private names
(PACE's numpy 2.5.4: AttributeError on the first training item, 2026-11-27). ``patch_numpy_format``
puts them back on ``numpy.lib.format``: call it before Beat This!'s dataset reads a file.
"""
from __future__ import annotations

NAMES = ("_check_version", "_read_array_header")


def patch_numpy_format() -> list[str]:
    """Missing private helpers copied onto numpy.lib.format; returns the names added (none on numpy 1.x)."""
    import numpy as np
    fmt = np.lib.format
    try:
        from numpy.lib import _format_impl as impl
    except ImportError:   # numpy 1.x: numpy.lib.format is the implementation
        return []
    added = []
    for name in NAMES:
        if not hasattr(fmt, name) and hasattr(impl, name):
            setattr(fmt, name, getattr(impl, name))
            added.append(name)
    return added
