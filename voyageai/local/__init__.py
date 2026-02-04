"""Local model support for Voyage AI SDK.

This module provides lazy imports to avoid loading torch/sentence-transformers
for API-only users. Install with: pip install voyageai[local]
"""

# Lazy imports - set to None if not available
try:
    import sentence_transformers as _sentence_transformers
    import torch as _torch
except ImportError:  # pragma: no cover - handled lazily in functions
    _sentence_transformers = None  # type: ignore[assignment]
    _torch = None  # type: ignore[assignment]


def _ensure_local_deps() -> tuple:
    """Ensure local model dependencies are available.

    Returns:
        Tuple of (sentence_transformers, torch) modules.

    Raises:
        ImportError: If sentence-transformers or torch are not installed.
    """
    if _sentence_transformers is None or _torch is None:
        raise ImportError(
            "The 'sentence-transformers' and 'torch' packages are required for local models. "
            "Install them with: pip install voyageai[local]"
        )
    return _sentence_transformers, _torch
