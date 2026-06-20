"""Sentence-transformers backend for local model inference."""

from typing import List, Optional, Tuple

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

# Mapping from input_type to the prompt name used by the model
INPUT_TYPE_TO_PROMPT = {
    "query": "query",
    "document": "document",
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
            model = sentence_transformers.SentenceTransformer(
                self.config.huggingface_id,
                trust_remote_code=self.config.trust_remote_code,
                device=self.device,
            )
            # Apply max_tokens as model's max_seq_length
            model.max_seq_length = self.config.max_tokens
            return model

        self.model = cache.get_or_load(cache_key, load_model)
        self._tokenizer = self.model.tokenizer

    def _get_prompt_for_input_type(self, input_type: Optional[str]) -> Optional[str]:
        """Get the instruction prompt for the given input type.

        Args:
            input_type: "query", "document", or None.

        Returns:
            The prompt string, or None if no prompt applies.
        """
        if input_type is None:
            return None
        prompt_name = INPUT_TYPE_TO_PROMPT.get(input_type)
        if prompt_name and prompt_name in self.model.prompts:
            return self.model.prompts[prompt_name]
        return None

    def encode(
        self,
        texts: List[str],
        input_type: Optional[str] = None,
        output_dtype: Optional[str] = None,
        output_dimension: Optional[int] = None,
        truncation: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """Encode texts into embeddings.

        Args:
            texts: List of texts to encode.
            input_type: "query", "document", or None.
            output_dtype: Output data type (float32, float, int8, uint8, binary, ubinary).
            output_dimension: Dimension to truncate embeddings to (MRL support).
            truncation: Whether to truncate texts exceeding max tokens.

        Returns:
            Tuple of (numpy array of embeddings, total token count).
        """
        # Validate and get dimension
        dimension = self.config.validate_dimension(output_dimension)

        # Validate and map precision — use returned value directly
        precision = self.config.validate_precision(output_dtype)
        if precision:
            precision = DTYPE_TO_PRECISION[precision]

        # Build encode kwargs
        encode_kwargs = {}
        if dimension != self.config.default_dimension:
            encode_kwargs["truncate_dim"] = dimension
        if precision:
            encode_kwargs["precision"] = precision

        # Wire truncation through to the model via processing_kwargs
        if not truncation:
            encode_kwargs["processing_kwargs"] = {
                "text": {"truncation": False},
            }

        # Route based on input_type using prompt_name
        if input_type == "query":
            embeddings = self.model.encode_query(texts, **encode_kwargs)
        elif input_type == "document":
            embeddings = self.model.encode_document(texts, **encode_kwargs)
        else:
            embeddings = self.model.encode(texts, **encode_kwargs)

        # Count tokens accounting for instruction prefix
        total_tokens = self._count_tokens_with_prefix(texts, input_type)

        return embeddings, total_tokens

    def _count_tokens_with_prefix(self, texts: List[str], input_type: Optional[str] = None) -> int:
        """Count total tokens across all texts, including instruction prefix.

        Args:
            texts: List of texts to count tokens for.
            input_type: "query", "document", or None.

        Returns:
            Total token count.
        """
        prompt = self._get_prompt_for_input_type(input_type)
        total = 0
        for text in texts:
            full_text = f"{prompt}{text}" if prompt else text
            encoded = self._tokenizer.encode(full_text, add_special_tokens=True)
            total += len(encoded)
        return total

    def count_tokens(self, texts: List[str], input_type: Optional[str] = None) -> int:
        """Count total tokens across all texts.

        Args:
            texts: List of texts to count tokens for.
            input_type: "query", "document", or None.

        Returns:
            Total token count.
        """
        return self._count_tokens_with_prefix(texts, input_type)
