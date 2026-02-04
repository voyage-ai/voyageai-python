import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

from PIL.Image import Image
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

import voyageai
import voyageai.error as error
from voyageai._base import _BaseClient
from voyageai.chunking import apply_chunking
from voyageai.local.model_registry import SUPPORTED_MODELS as LOCAL_MODELS
from voyageai.object import (
    ContextualizedEmbeddingsObject,
    EmbeddingsObject,
    MultimodalEmbeddingsObject,
    RerankingObject,
)
from voyageai.object.multimodal_embeddings import MultimodalInputRequest
from voyageai.video_utils import Video

if TYPE_CHECKING:
    from voyageai.local.sentence_transformer_backend import SentenceTransformerBackend


class Client(_BaseClient):
    """Voyage AI Client

    Args:
        api_key (str): Your API key (not required for local models).
        max_retries (int): Maximum number of retries if API call fails.
        timeout (float): Timeout in seconds.
        base_url (str): Base URL for the API endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 0,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(api_key, max_retries, timeout, base_url)
        self._local_backends: Dict[str, "SentenceTransformerBackend"] = {}

    def _make_retry_controller(self) -> Retrying:
        return Retrying(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(initial=1, max=16),
            retry=(
                retry_if_exception_type(error.RateLimitError)
                | retry_if_exception_type(error.ServiceUnavailableError)
                | retry_if_exception_type(error.Timeout)
            ),
        )

    def _get_local_backend(self, model: str) -> "SentenceTransformerBackend":
        """Get or create a local backend for the given model."""
        if model not in self._local_backends:
            from voyageai.local.sentence_transformer_backend import SentenceTransformerBackend

            self._local_backends[model] = SentenceTransformerBackend(model)
        return self._local_backends[model]

    def _embed_local(
        self,
        texts: List[str],
        model: str,
        input_type: Optional[str] = None,
        truncation: bool = True,
        output_dtype: Optional[str] = None,
        output_dimension: Optional[int] = None,
    ) -> EmbeddingsObject:
        """Generate embeddings using a local model."""
        backend = self._get_local_backend(model)

        embeddings_array = backend.encode(
            texts=texts,
            input_type=input_type,
            output_dtype=output_dtype,
            output_dimension=output_dimension,
            truncation=truncation,
        )

        total_tokens = backend.count_tokens(texts)

        result = EmbeddingsObject()
        result.embeddings = embeddings_array.tolist()
        result.total_tokens = total_tokens

        return result

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: Optional[str] = None,
        truncation: bool = True,
        output_dtype: Optional[str] = None,
        output_dimension: Optional[int] = None,
    ) -> EmbeddingsObject:
        if model is None:
            model = voyageai.VOYAGE_EMBED_DEFAULT_MODEL
            warnings.warn(
                f"The `model` argument is not specified and defaults to {voyageai.VOYAGE_EMBED_DEFAULT_MODEL}. "
                "It will be a required argument in the future. We recommend to specify the model when using this "
                "function. Please see https://docs.voyageai.com/docs/embeddings for the list of latest models "
                "provided by Voyage AI."
            )

        # Check if this is a local model
        if model in LOCAL_MODELS:
            return self._embed_local(
                texts=texts,
                model=model,
                input_type=input_type,
                truncation=truncation,
                output_dtype=output_dtype,
                output_dimension=output_dimension,
            )

        # API models require an API key
        if not self.api_key:
            raise error.AuthenticationError(
                "An API key is required for API-based models. "
                "Set your API key via VOYAGE_API_KEY environment variable or pass it to Client(api_key=...)."
            )

        response = None
        for attempt in self._make_retry_controller():
            with attempt:
                response = voyageai.Embedding.create(
                    input=texts,
                    model=model,
                    input_type=input_type,
                    truncation=truncation,
                    output_dtype=output_dtype,
                    output_dimension=output_dimension,
                    **self._params,
                )

        if response is None:
            raise error.APIConnectionError("Failed to get response after all retry attempts")

        result = EmbeddingsObject(response)
        return result

    def contextualized_embed(
        self,
        inputs: List[List[str]],
        model: str,
        input_type: Optional[str] = None,
        output_dtype: Optional[str] = None,
        output_dimension: Optional[int] = None,
        chunk_fn: Optional[Callable[[str], List[str]]] = None,
    ) -> ContextualizedEmbeddingsObject:
        response = None
        for attempt in self._make_retry_controller():
            with attempt:
                if chunk_fn:
                    inputs = apply_chunking(inputs, chunk_fn)
                response = voyageai.ContextualizedEmbedding.create(
                    inputs=inputs,
                    model=model,
                    input_type=input_type,
                    output_dtype=output_dtype,
                    output_dimension=output_dimension,
                    **self._params,
                )

        if response is None:
            raise error.APIConnectionError("Failed to get response after all retry attempts")

        if chunk_fn:
            return ContextualizedEmbeddingsObject(
                response=response,
                chunk_texts=inputs,
            )
        return ContextualizedEmbeddingsObject(response)

    def rerank(
        self,
        query: str,
        documents: List[str],
        model: str,
        top_k: Optional[int] = None,
        truncation: bool = True,
    ) -> RerankingObject:
        response = None
        for attempt in self._make_retry_controller():
            with attempt:
                response = voyageai.Reranking.create(
                    query=query,
                    documents=documents,
                    model=model,
                    top_k=top_k,
                    truncation=truncation,
                    **self._params,
                )

        if response is None:
            raise error.APIConnectionError("Failed to get response after all retry attempts")

        result = RerankingObject(documents, response)
        return result

    def multimodal_embed(
        self,
        inputs: Union[List[Dict], List[List[Union[str, Image, Video]]]],
        model: str,
        input_type: Optional[str] = None,
        truncation: bool = True,
        output_dtype: Optional[str] = None,
        output_dimension: Optional[int] = None,
    ) -> MultimodalEmbeddingsObject:
        response = None
        for attempt in self._make_retry_controller():
            with attempt:
                response = voyageai.MultimodalEmbedding.create(
                    **MultimodalInputRequest.from_user_inputs(
                        inputs=inputs,
                        model=model,
                        input_type=input_type,
                        truncation=truncation,
                        output_dtype=output_dtype,
                        output_dimension=output_dimension,
                    ).dict(),
                    **self._params,
                )

        if response is None:
            raise error.APIConnectionError("Failed to get response after all retry attempts")

        result = MultimodalEmbeddingsObject(response)
        return result
