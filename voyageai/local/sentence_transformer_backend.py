"""Sentence-transformers backend for local model inference."""

from typing import List, Optional, Tuple

import numpy as np

from voyageai.error import InvalidRequestError
from voyageai.local import _ensure_local_deps
from voyageai.local.model_registry import ModelCache, get_model_config

# Mapping from SDK output_dtype to sentence-transformers precision.
#
# int8/uint8 are intentionally absent: matching the hosted API requires fixed
# calibration ranges, which we don't have for local models. Without them
# sentence-transformers derives the quantization range from the current batch,
# yielding non-deterministic, API-incompatible vectors — so int8/uint8 are
# rejected up front in ``encode`` rather than returning silently-wrong output.
DTYPE_TO_PRECISION = {
    "float32": "float32",
    "float": "float32",
    "binary": "binary",
    "ubinary": "ubinary",
}

# Output dtypes that require fixed calibration ranges we don't yet ship locally.
_UNSUPPORTED_QUANTIZED_DTYPES = ("int8", "uint8")

# Valid input_type values (None means no instruction prompt).
_VALID_INPUT_TYPES = (None, "query", "document")

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
        # Serializes forward passes on this shared cached model (see encode()).
        self._inference_lock = cache.get_inference_lock(cache_key)

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
            output_dtype: Output data type (float32, float, binary, ubinary).
            output_dimension: Dimension to truncate embeddings to (MRL support).
            truncation: Whether to truncate texts exceeding max tokens.

        Returns:
            Tuple of (numpy array of embeddings, total token count).

        Raises:
            InvalidRequestError: If input_type/output_dtype/output_dimension is
                invalid. Mirrors the hosted API's validation so callers can
                catch the same exception on both paths.
            NotImplementedError: If output_dtype is int8/uint8. These are valid
                against the hosted API but unsupported locally because they
                require fixed calibration ranges we don't ship yet — a genuine
                capability gap rather than an invalid request (see
                DTYPE_TO_PRECISION).
        """
        # Reject invalid input_type up front. The hosted API raises for unknown
        # values; without this the unknown value would silently fall through to
        # the no-prompt branch and degrade embedding quality.
        if input_type not in _VALID_INPUT_TYPES:
            raise InvalidRequestError(
                f"Invalid input_type {input_type!r}. Use 'query', 'document', or None."
            )

        # int8/uint8 need fixed calibration ranges to match the API; reject
        # rather than return non-deterministic, incomparable vectors. This is a
        # local capability gap (the API accepts them), so NotImplementedError —
        # not InvalidRequestError — is the honest signal.
        if output_dtype in _UNSUPPORTED_QUANTIZED_DTYPES:
            raise NotImplementedError(
                f"output_dtype={output_dtype!r} is not supported for local models. "
                "int8/uint8 quantization requires fixed calibration ranges to match "
                "the hosted API; use 'float'/'float32' or 'binary'/'ubinary'."
            )

        # Validate and get dimension
        dimension = self.config.validate_dimension(output_dimension)

        # Validate and map precision — use returned value directly
        precision = self.config.validate_precision(output_dtype)
        if precision:
            precision = DTYPE_TO_PRECISION[precision]

        # Build encode kwargs. Always normalize: the API returns unit-norm
        # embeddings at every dimension, and truncating a unit-normalized vector
        # (Matryoshka / MRL) breaks unit norm, so truncated and full-dim outputs
        # must both be re-normalized. Setting this unconditionally (rather than
        # only on the truncated branch) makes full-dim norm an explicit
        # guarantee instead of relying on the model self-normalizing.
        encode_kwargs = {"normalize_embeddings": True}
        if dimension != self.config.default_dimension:
            encode_kwargs["truncate_dim"] = dimension
        if precision:
            encode_kwargs["precision"] = precision

        # Wire truncation through to the model. When truncation is disabled the
        # hosted API rejects input exceeding the context length (rather than
        # silently degrading); validate length up front and raise the same
        # InvalidRequestError instead of feeding an over-length sequence to the
        # model (which would crash deep inside it or degrade silently).
        if not truncation:
            prompt = self._get_prompt_for_input_type(input_type)
            for text in texts:
                full_text = f"{prompt}{text}" if prompt else text
                n = len(self._tokenizer.encode(full_text, add_special_tokens=True))
                if n > self.config.max_tokens:
                    raise InvalidRequestError(
                        f"Input exceeds the {self.config.max_tokens}-token context length "
                        "and truncation is disabled."
                    )
            encode_kwargs["processing_kwargs"] = {"text": {"truncation": False}}

        # Route based on input_type using prompt_name. Serialize the forward
        # pass: SentenceTransformer.encode* isn't contractually thread-safe, and
        # the async path dispatches concurrent embeds onto worker threads that
        # share this one cached model.
        with self._inference_lock:
            if input_type == "query":
                embeddings = self.model.encode_query(texts, **encode_kwargs)
            elif input_type == "document":
                embeddings = self.model.encode_document(texts, **encode_kwargs)
            else:
                embeddings = self.model.encode(texts, **encode_kwargs)

        # Count tokens accounting for instruction prefix (and truncation cap)
        total_tokens = self.count_tokens(texts, input_type, truncation)

        return embeddings, total_tokens

    def count_tokens(
        self, texts: List[str], input_type: Optional[str] = None, truncation: bool = True
    ) -> int:
        """Count total tokens across all texts, including the instruction prefix.

        The model prepends an instruction prompt for query/document inputs and
        tokenizes it too, so the prefix is counted to match server-side usage.
        When ``truncation`` is enabled the model only processes the first
        ``max_tokens`` tokens of an over-length input, so each text's count is
        capped there — reporting the *processed* token count, as the API does,
        rather than the full pre-truncation length.

        Args:
            texts: List of texts to count tokens for.
            input_type: "query", "document", or None.
            truncation: Whether over-length inputs are truncated to max_tokens.

        Returns:
            Total token count.
        """
        prompt = self._get_prompt_for_input_type(input_type)
        total = 0
        for text in texts:
            full_text = f"{prompt}{text}" if prompt else text
            n = len(self._tokenizer.encode(full_text, add_special_tokens=True))
            total += min(n, self.config.max_tokens) if truncation else n
        return total
