"""Sentence-transformers backend for local model inference."""

from typing import List, Optional

import numpy as np

from voyageai.local import _ensure_local_deps
from voyageai.local.model_registry import ModelCache, get_model_config

# Mapping from SDK output_dtype to sentence-transformers precision
DTYPE_TO_PRECISION = {
    "float32": "float32",
    "float": "float32",
    "int8": "int8",
    "uint8": "uint8",
    "binary": "binary",
    "ubinary": "ubinary",
}


class SentenceTransformerBackend:
    """Wrapper for sentence-transformers with SDK-compatible interface."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
    ):
        """Initialize the backend.

        Args:
            model_name: Name of the model (e.g., "voyage-4-nano").
            device: Device to use ("cuda", "cpu", or None for auto-detect).
        """
        sentence_transformers, torch = _ensure_local_deps()

        self.config = get_model_config(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Cache key includes model and device
        cache_key = f"{model_name}:{self.device}"
        cache = ModelCache()

        def load_model():
            return sentence_transformers.SentenceTransformer(
                self.config.huggingface_id,
                trust_remote_code=self.config.trust_remote_code,
                device=self.device,
            )

        self.model = cache.get_or_load(cache_key, load_model)
        self._tokenizer = self.model.tokenizer

    def encode(
        self,
        texts: List[str],
        input_type: Optional[str] = None,
        output_dtype: Optional[str] = None,
        output_dimension: Optional[int] = None,
        truncation: bool = True,
    ) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: List of texts to encode.
            input_type: "query", "document", or None.
            output_dtype: Output data type (float32, int8, uint8, binary, ubinary).
            output_dimension: Dimension to truncate embeddings to (MRL support).
            truncation: Whether to truncate texts exceeding max tokens.

        Returns:
            Numpy array of embeddings.
        """
        # Validate and get dimension
        dimension = self.config.validate_dimension(output_dimension)

        # Validate and map precision
        self.config.validate_precision(output_dtype)
        precision = DTYPE_TO_PRECISION.get(output_dtype) if output_dtype else None

        # Build encode kwargs
        encode_kwargs = {}
        if dimension != self.config.default_dimension:
            encode_kwargs["truncate_dim"] = dimension
        if precision:
            encode_kwargs["precision"] = precision

        # Route based on input_type
        if input_type == "query":
            embeddings = self.model.encode_query(texts, **encode_kwargs)
        elif input_type == "document":
            embeddings = self.model.encode_document(texts, **encode_kwargs)
        else:
            embeddings = self.model.encode(texts, **encode_kwargs)

        return embeddings

    def count_tokens(self, texts: List[str]) -> int:
        """Count total tokens across all texts.

        Args:
            texts: List of texts to count tokens for.

        Returns:
            Total token count.
        """
        total = 0
        for text in texts:
            encoded = self._tokenizer.encode(text, add_special_tokens=True)
            total += len(encoded)
        return total
