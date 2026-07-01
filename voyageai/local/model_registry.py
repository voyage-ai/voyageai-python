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
    # Max inputs per embed() call. Matches the hosted API's documented list limit
    # so the same call raises InvalidRequestError on both the API and local paths.
    max_batch_size: int = 1000

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
        max_batch_size=1000,
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
                    # Per-cache-key locks. ``_load_locks`` serialize cold loads
                    # of a given key (so loading one model never blocks callers
                    # of a different, already-cached model); ``_inference_locks``
                    # serialize forward passes on a given cached model
                    # (SentenceTransformer.encode* is not thread-safe).
                    instance._load_locks = {}
                    instance._inference_locks = {}
                    cls._instance = instance
        return cls._instance

    def _keyed_lock(self, registry: Dict[str, threading.Lock], key: str) -> threading.Lock:
        """Return a lock for ``key`` from ``registry``, creating it if needed."""
        with self._cache_lock:
            lock = registry.get(key)
            if lock is None:
                lock = threading.Lock()
                registry[key] = lock
            return lock

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

    def get_inference_lock(self, key: str) -> threading.Lock:
        """Return the per-key lock guarding inference on the cached model.

        ``SentenceTransformer.encode*`` is not contractually thread-safe, so
        concurrent embeds that share one cached model (e.g. the async path
        dispatching to worker threads) must serialize their forward passes.
        """
        return self._keyed_lock(self._inference_locks, key)

    def get_or_load(self, key: str, loader: Callable[[], Any]) -> Any:
        """Get a cached model or load it if not cached.

        Uses a per-key load lock so only concurrent loads of the *same* key
        serialize: a multi-second cold load of one model never blocks callers
        requesting a different, already-cached model. (Holding the single cache
        lock across the whole load would serialize unrelated keys.)

        Args:
            key: Cache key.
            loader: Callable to load the model if not cached.

        Returns:
            The cached or newly loaded model.
        """
        model = self.get(key)
        if model is not None:
            return model
        with self._keyed_lock(self._load_locks, key):
            # Re-check under the per-key lock: another thread may have finished
            # loading this key while we waited.
            model = self.get(key)
            if model is None:
                model = loader()
                self.set(key, model)
            return model
