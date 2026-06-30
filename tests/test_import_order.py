"""Regression tests for circular import (issue #42).

Importing submodules directly (without a prior ``import voyageai``) used to
trigger an ``ImportError`` / ``AttributeError`` because some modules accessed
``voyageai.error.X`` via late-binding attribute access on the top-level package,
which hadn't finished initializing yet.

Each test below spawns a **fresh** Python subprocess that imports a single
submodule — this is the only reliable way to reproduce the original failure,
because in-process importlib tricks are defeated by the module cache.
"""

import pathlib
import subprocess
import sys

import pytest
import voyageai

# Modules that previously evaluated ``voyageai.error.X`` via late-binding
# attribute access at import time (the root cause of issue #42). They must now
# import error classes directly (``from voyageai.error import X``).
_PATCHED_MODULES = ["util.py", "embeddings_utils.py", "_base.py"]
_PKG_DIR = pathlib.Path(voyageai.__file__).parent

# Each entry is (human-readable label, Python import statement).
_IMPORT_CASES = [
    ("voyageai.util", "from voyageai.util import default_api_key"),
    ("voyageai.embeddings_utils", "from voyageai.embeddings_utils import get_embeddings"),
    (
        "voyageai.api_resources.api_requestor",
        "from voyageai.api_resources.api_requestor import APIRequestor",
    ),
]


@pytest.mark.parametrize("label,import_stmt", _IMPORT_CASES, ids=[c[0] for c in _IMPORT_CASES])
def test_no_circular_import_error(label: str, import_stmt: str) -> None:
    """Importing *label* in a fresh interpreter must not raise ImportError."""
    result = subprocess.run(
        [sys.executable, "-c", import_stmt],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Importing {label} failed (exit {result.returncode}).\n" f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("filename", _PATCHED_MODULES)
def test_no_late_bound_error_access(filename: str) -> None:
    """The subprocess test above can pass even against the pre-fix code, because
    importing any submodule first runs ``voyageai/__init__.py`` to completion,
    which binds ``voyageai.error`` before the late access is reached. This static
    check actually fails on the buggy code and passes on the fix: it asserts the
    previously-affected modules contain no module-scope ``voyageai.error.X``
    attribute access.
    """
    source = (_PKG_DIR / filename).read_text()
    assert "voyageai.error." not in source, (
        f"{filename} uses late-bound 'voyageai.error.X' access (issue #42). "
        "Import the error class directly instead, e.g. "
        "'from voyageai.error import RateLimitError'."
    )
