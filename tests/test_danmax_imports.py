"""The optional DanMAX imports must not swallow interrupts.

``danmax.py`` guards pyFAI, pytz and the beamline's own ``lib`` helpers, all of
which are optional. Those guards used to catch ``BaseException``, which includes
``KeyboardInterrupt`` and ``SystemExit``: interrupting a slow import was reported
as "optional dependency unavailable", execution continued with the helper names
undefined, and the real failure surfaced later as a ``NameError`` somewhere
unrelated.
"""
import importlib
import sys
from contextlib import contextmanager

import pytest

import tomo_image_stitcher.danmax as danmax


def _drop_fake_lib():
    for name in [m for m in sys.modules if m == "lib" or m.startswith("lib.")]:
        del sys.modules[name]


@contextmanager
def fake_lib(tmp_path, body):
    """Put a package named ``lib`` on sys.path that runs ``body`` on import."""
    pkg = tmp_path / "lib"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(body, encoding="utf-8")

    saved_path = list(sys.path)
    _drop_fake_lib()
    sys.path.insert(0, str(tmp_path))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        _drop_fake_lib()
        # Leave the real module loaded for anything that runs afterwards.
        importlib.reload(danmax)


def test_keyboard_interrupt_is_not_swallowed(tmp_path):
    with fake_lib(tmp_path, "raise KeyboardInterrupt\n"):
        with pytest.raises(KeyboardInterrupt):
            importlib.reload(danmax)


def test_a_broken_lib_package_still_imports_the_module(tmp_path):
    # An ImportError is the case the guard is for, so the module must survive it.
    with fake_lib(tmp_path, "raise ImportError('pretend lib is broken')\n"):
        importlib.reload(danmax)  # must not raise
