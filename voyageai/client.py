import warnings
from typing import Callable, Dict, List, Optional, Union

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
from voyageai.object import (
    ContextualizedEmbeddingsObject,
    EmbeddingsObject,
    MultimodalEmbeddingsObject,
    RerankingObject,
)
from voyageai.object.multimodal_embeddings import MultimodalInputRequest
from voyageai.video_utils import Video


class Client(_BaseClient):
    """Voyage AI Client

    Args:
        api_key (str): Your API key.
        max_retries (int): Maximum number of retries if API call fails.
        timeout (float): Timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 0,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(api_key, max_retries, timeout)

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
