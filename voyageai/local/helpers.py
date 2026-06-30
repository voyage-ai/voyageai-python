"""Shared helpers for local model routing in sync and async clients."""

from typing import TYPE_CHECKING, List, Optional

from voyageai.local.model_registry import SUPPORTED_MODELS as LOCAL_MODELS
from voyageai.object import EmbeddingsObject

if TYPE_CHECKING:
    from voyageai.local.sentence_transformer_backend import SentenceTransformerBackend


def get_local_backend(model: str) -> "SentenceTransformerBackend":
    """Create a local backend for the given model.

    A fresh SentenceTransformerBackend wrapper is constructed on every call;
    only the heavy underlying SentenceTransformer model is cached (process-wide,
    keyed by model:device, via the ModelCache singleton). The per-call wrapper
    cost is small, so no per-client backend cache is needed.

    Args:
        model: Model name (must be in LOCAL_MODELS).

    Returns:
        A SentenceTransformerBackend wrapping the cached model.
    """
    from voyageai.local.sentence_transformer_backend import SentenceTransformerBackend

    return SentenceTransformerBackend(model)


def embed_local(
    texts: List[str],
    model: str,
    input_type: Optional[str] = None,
    truncation: bool = True,
    output_dtype: Optional[str] = None,
    output_dimension: Optional[int] = None,
) -> EmbeddingsObject:
    """Generate embeddings using a local model.

    Args:
        texts: List of texts to embed.
        model: Model name.
        input_type: "query", "document", or None.
        truncation: Whether to truncate texts exceeding max tokens.
        output_dtype: Output data type.
        output_dimension: Output embedding dimension.

    Returns:
        EmbeddingsObject with embeddings and token count.
    """
    # The API path accepts a bare string and returns a list with one embedding
    # (embeddings == [[...]]). Normalize here so the local path matches; without
    # this, encode() would return a 1-D array and embeddings[0] would be a float.
    if isinstance(texts, str):
        texts = [texts]

    backend = get_local_backend(model)

    embeddings_array, total_tokens = backend.encode(
        texts=texts,
        input_type=input_type,
        output_dtype=output_dtype,
        output_dimension=output_dimension,
        truncation=truncation,
    )

    result = EmbeddingsObject()
    result.embeddings = embeddings_array.tolist()
    result.total_tokens = total_tokens

    return result


def is_local_model(model: str) -> bool:
    """Check if a model name refers to a locally-supported model."""
    return model in LOCAL_MODELS
