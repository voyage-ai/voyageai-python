"""Model configuration and thread-safe caching for local models."""

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from voyageai.error import InvalidRequestError


@dataclass(frozen=True)
class LocalModelConfig:
    """Configuration for a local embedding model."""

    huggingface_id: str
    max_tokens: int
    default_dimension: int
    supported_dimensions: tuple
    supported_precisions: tuple
    trust_remote_code: bool = True

    def validate_dimension(self, dimension: Optional[int]) -> int:
        """Validate and return the dimension to use.

        Args:
            dimension: Requested dimension, or None for default.

        Returns:
            The dimension to use.

        Raises:
            InvalidRequestError: If dimension is not supported. Matches the
                hosted API, which rejects invalid output_dimension the same way.
        """
        if dimension is None:
            return self.default_dimension
        if dimension not in self.supported_dimensions:
            raise InvalidRequestError(
                f"Invalid output_dimension {dimension}. "
                f"Supported dimensions: {self.supported_dimensions}"
            )
        return dimension

    def validate_precision(self, precision: Optional[str]) -> Optional[str]:
        """Validate and return the precision to use.

        Args:
            precision: Requested precision, or None for default (float32).

        Returns:
            The precision to use.

        Raises:
            InvalidRequestError: If precision is not supported. Matches the
                hosted API, which rejects invalid output_dtype the same way.
        """
        if precision is None:
            return None
        if precision not in self.supported_precisions:
            raise InvalidRequestError(
                f"Invalid output_dtype '{precision}'. "
                f"Supported dtypes: {self.supported_precisions}"
            )
        return precision


# Supported local models
SUPPORTED_MODELS: Dict[str, LocalModelConfig] = {
    "voyage-4-nano": LocalModelConfig(
        huggingface_id="voyageai/voyage-4-nano",
        max_tokens=32768,
        default_dimension=2048,
        supported_dimensions=(2048, 1024, 512, 256),
        # int8/uint8 omitted: they require fixed calibration ranges to match the
        # hosted API, which we don't ship locally yet (see encode() in
        # sentence_transformer_backend.py).
        supported_precisions=("float32", "float", "binary", "ubinary"),
        trust_remote_code=True,
    ),
}


def get_model_config(model: str) -> LocalModelConfig:
    """Get configuration for a model.

    Args:
        model: Model name.

    Returns:
        LocalModelConfig for the model.

    Raises:
        ValueError: If model is not supported.
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported local model '{model}'. "
            f"Supported models: {list(SUPPORTED_MODELS.keys())}"
        )
    return SUPPORTED_MODELS[model]


class ModelCache:
    """Thread-safe singleton cache for loaded models.

    Avoids reloading models per call, which can be expensive.
    """

    _instance: Optional["ModelCache"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ModelCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # Fully initialize a local instance, then publish it last.
                    # The outer ``if cls._instance is None`` runs without the
                    # lock, so a concurrent caller must never be able to observe
                    # a published-but-half-built instance (one missing _cache /
                    # _cache_lock) and crash dereferencing them.
                    instance = super().__new__(cls)
                    instance._cache = {}
                    instance._cache_lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    def get(self, key: str) -> Optional[Any]:
        """Get a cached model.

        Args:
            key: Cache key (typically model name + device).

        Returns:
            Cached model or None if not found.
        """
        with self._cache_lock:
            return self._cache.get(key)

    def set(self, key: str, model: Any) -> None:
        """Cache a model.

        Args:
            key: Cache key.
            model: Model to cache.
        """
        with self._cache_lock:
            self._cache[key] = model

    def clear(self) -> None:
        """Clear all cached models."""
        with self._cache_lock:
            self._cache.clear()

    def get_or_load(self, key: str, loader: Callable[[], Any]) -> Any:
        """Get a cached model or load it if not cached.

        Thread-safe: holds the lock for the full check-load-store sequence.

        Args:
            key: Cache key.
            loader: Callable to load the model if not cached.

        Returns:
            The cached or newly loaded model.
        """
        with self._cache_lock:
            model = self._cache.get(key)
            if model is None:
                model = loader()
                self._cache[key] = model
            return model
