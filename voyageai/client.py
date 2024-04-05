import functools
import warnings
from typing import Any, List, Optional
from tenacity import (
    Retrying,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

import voyageai
import voyageai.error as error
from voyageai.util import default_api_key
from voyageai.object import EmbeddingsObject, RerankingObject


class Client:
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

        self.api_key = api_key or default_api_key()

        self._params = {
            "api_key": self.api_key,
            "request_timeout": timeout,
        }

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

    @property
    @functools.lru_cache()
    def tokenizer(self):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "tokenizers package not found. Please run `pip install tokenizers` "
                "to install the dependency."
            )

        tokenizer = Tokenizer.from_pretrained("voyageai/voyage")
        tokenizer.no_truncation()
        return tokenizer

    def tokenize(
        self,
        texts: List[str],
    ) -> List[Any]:
        return self.tokenizer.encode_batch(texts)

    def count_tokens(
        self,
        texts: List[str],
    ) -> int:
        tokenized = self.tokenize(texts)
        return sum([len(t) for t in tokenized])
