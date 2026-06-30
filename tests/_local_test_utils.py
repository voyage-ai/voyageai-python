"""Shared helpers for the local-model test modules."""

import importlib.util


def real_deps_available() -> bool:
    """Check if sentence-transformers and torch can be found (without importing).

    Deliberately uses ``importlib.util.find_spec`` rather than
    ``voyageai.local._is_local_available`` so it stays an *independent* signal:
    ``test_has_local_constant`` asserts ``voyageai.HAS_LOCAL`` against this, and
    that assertion is only meaningful if the two are computed differently.
    """
    return (
        importlib.util.find_spec("sentence_transformers") is not None
        and importlib.util.find_spec("torch") is not None
    )
