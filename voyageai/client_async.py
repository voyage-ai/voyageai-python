import warnings
from typing import Callable, Dict, List, Optional, Union

from PIL.Image import Image
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

import voyageai
import voyageai.error as error
from voyageai._base import _BaseClient
from voyageai.chunking import (
    apply_chunking,
    validate_and_normalize_contextualized_inputs,
)
from voyageai.object import (
    ContextualizedEmbeddingsObject,
    EmbeddingsObject,
    MultimodalEmbeddingsObject,
    RerankingObject,
)
from voyageai.object.multimodal_embeddings import MultimodalInputRequest
from voyageai.video_utils import Video


class AsyncClient(_BaseClient):
    """Voyage AI Async Client

    Args:
        api_key (str): Your API key.
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

    def _make_retry_controller(self) -> AsyncRetrying:
        return AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(initial=1, max=16),
            retry=(
                retry_if_exception_type(error.RateLimitError)
                | retry_if_exception_type(error.ServiceUnavailableError)
                | retry_if_exception_type(error.Timeout)
            ),
        )

    async def embed(
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

        response = None
        async for attempt in self._make_retry_controller():
            with attempt:
                response = await voyageai.Embedding.acreate(
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

    async def contextualized_embed(
        self,
        inputs: Union[List[List[str]], List[str]],
        model: str,
        input_type: Optional[str] = None,
        output_dtype: Optional[str] = None,
        output_dimension: Optional[int] = None,
        chunk_fn: Optional[Callable[[str], List[str]]] = None,
        enable_auto_chunking: bool = False,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> ContextualizedEmbeddingsObject:
        normalized_inputs, extra_kwargs = validate_and_normalize_contextualized_inputs(
            inputs=inputs,
            input_type=input_type,
            chunk_fn=chunk_fn,
            enable_auto_chunking=enable_auto_chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        request_inputs = (
            apply_chunking(normalized_inputs, chunk_fn) if chunk_fn else normalized_inputs
        )

        response = None
        async for attempt in self._make_retry_controller():
            with attempt:
                response = await voyageai.ContextualizedEmbedding.acreate(
                    inputs=request_inputs,
                    model=model,
                    input_type=input_type,
                    output_dtype=output_dtype,
                    output_dimension=output_dimension,
                    **extra_kwargs,
                    **self._params,
                )

        if response is None:
            raise error.APIConnectionError("Failed to get response after all retry attempts")

        if chunk_fn:
            return ContextualizedEmbeddingsObject(
                response=response,
                chunk_texts=request_inputs,
            )
        return ContextualizedEmbeddingsObject(response)

    async def rerank(
        self,
        query: str,
        documents: List[str],
        model: str,
        top_k: Optional[int] = None,
        truncation: bool = True,
    ) -> RerankingObject:
        response = None
        async for attempt in self._make_retry_controller():
            with attempt:
                response = await voyageai.Reranking.acreate(
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

    async def multimodal_embed(
        self,
        inputs: Union[List[Dict], List[List[Union[str, Image, Video]]]],
        model: str,
        input_type: Optional[str] = None,
        truncation: bool = True,
        output_dtype: Optional[str] = None,
        output_dimension: Optional[int] = None,
    ) -> MultimodalEmbeddingsObject:
        response = None
        async for attempt in self._make_retry_controller():
            with attempt:
                response = await voyageai.MultimodalEmbedding.acreate(
                    **MultimodalInputRequest.from_user_inputs(
                        inputs=inputs,
                        model=model,
                        input_type=input_type,
                        truncation=truncation,
                        output_dtype=output_dtype,
                        output_dimension=output_dimension,
                    ).model_dump(),
                    **self._params,
                )
        if response is None:
            raise error.APIConnectionError("Failed to get response after all retry attempts")

        result = MultimodalEmbeddingsObject(response)
        return result
