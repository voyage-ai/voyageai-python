"""Local model support for Voyage AI SDK.

This module provides lazy imports to avoid loading torch/sentence-transformers
for API-only users. Install with: pip install voyageai[local]
"""

import sys


def _ensure_local_deps() -> tuple:
    """Ensure local model dependencies are available.

    Returns:
        Tuple of (sentence_transformers, torch) modules.

    Raises:
        ImportError: If sentence-transformers or torch are not installed.
    """
    if sys.version_info < (3, 10):
        raise ImportError(
            "Local model support requires Python 3.10 or later. "
            "You are running Python {}.{}. Please use an API-based model or upgrade Python.".format(
                sys.version_info.major, sys.version_info.minor
            )
        )
    try:
        import sentence_transformers
        import torch

        return sentence_transformers, torch
    except ImportError:
        raise ImportError(
            "The 'sentence-transformers' and 'torch' packages are required for local models. "
            "Install them with: pip install voyageai[local]"
        )


def _is_local_available() -> bool:
    """Check if sentence-transformers and torch are installed (lazy check)."""
    try:
        _ensure_local_deps()
        return True
    except ImportError:
        return False
