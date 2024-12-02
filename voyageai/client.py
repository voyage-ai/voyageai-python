import warnings
from typing import Any, List, Optional, Union, Dict
from tenacity import (
    Retrying,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)
from PIL.Image import Image

import voyageai
from voyageai._base import _BaseClient
import voyageai.error as error
from voyageai.object.multimodal_embeddings import MultimodalInputRequest
from voyageai.object import EmbeddingsObject, RerankingObject, MultimodalEmbeddingsObject


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
    ) ->None:
        super().__init__(api_key, max_retries, timeout)

        self.retry_controller = Retrying(
            reraise=True,
            stop=stop_after_attempt(max_retries),
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

        for attempt in self.retry_controller:
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

        result = EmbeddingsObject(response)
        return result

    def rerank(
        self,
        query: str,
        documents: List[str],
        model: str,
        top_k: Optional[int] = None,
        truncation: bool = True,
    ) -> RerankingObject:

        for attempt in self.retry_controller:
            with attempt:
                response = voyageai.Reranking.create(
                    query=query,
                    documents=documents,
                    model=model,
                    top_k=top_k,
                    truncation=truncation,
                    **self._params,
                )

        result = RerankingObject(documents, response)
        return result

    def multimodal_embed(
        self,
        inputs: Union[List[Dict], List[List[Union[str, Image]]]],
        model: str,
        input_type: Optional[str] = None,
        truncation: bool = True,
    ) -> MultimodalEmbeddingsObject:
        """
        Generate multimodal embeddings for the provided inputs using the specified model.

        :param inputs: Either a list of dictionaries (each with 'content') or a list of lists containing strings and/or PIL images.
        :param model: The model identifier.
        :param input_type: Optional input type.
        :param truncation: Whether to apply truncation.
        :return: An instance of MultimodalEmbeddingsObject.
        """
        for attempt in self.retry_controller:
            with attempt:
                response = voyageai.MultimodalEmbedding.create(
                    **MultimodalInputRequest.from_user_inputs(
                        inputs=inputs,
                        model=model,
                        input_type=input_type,
                        truncation=truncation,
                    ).dict(),
                    **self._params,
                )

        result = MultimodalEmbeddingsObject(response)
        return result