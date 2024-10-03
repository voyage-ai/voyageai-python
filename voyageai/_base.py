from abc import ABC, abstractmethod
from collections.abc import Awaitable
import functools
import warnings
from typing import Any, List, Optional, Union

import voyageai
import voyageai.error as error
from voyageai.util import default_api_key
from voyageai.object import EmbeddingsObject, RerankingObject


class _BaseClient:
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


    @abstractmethod    
    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: Optional[str] = None,
        truncation: bool = True,
    ) -> Union[EmbeddingsObject, Awaitable[EmbeddingsObject]]:
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        model: str,
        top_k: Optional[int] = None,
        truncation: bool = True,
    ) -> Union[RerankingObject, Awaitable[RerankingObject]]:
        pass

    @functools.lru_cache()
    def tokenizer(self, model: str):
        try:
            from tokenizers import Tokenizer  # type: ignore
        except ImportError:
            raise ImportError(
                "The package `tokenizers` is not found. Please run `pip install tokenizers` "
                "to install the dependency."
            )

        try:
            tokenizer = Tokenizer.from_pretrained(f"voyageai/{model}")
            tokenizer.no_truncation()
        except:
            warnings.warn(
                f"Failed to load the tokenizer for `{model}`. Please ensure that it is a valid model name."
            )
            raise
        
        return tokenizer

    def tokenize(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[Any]:

        if model is None:
            warnings.warn(
                "Please specify the `model` when using the tokenizer. Voyage's older models use the same "
                "tokenizer, but new models may use different tokenizers. If `model` is not specified, "
                "the old tokenizer will be used and the results might be different. `model` will be a "
                "required argument in the future."
            )
            model = voyageai.VOYAGE_EMBED_DEFAULT_MODEL

        return self.tokenizer(model).encode_batch(texts)

    def count_tokens(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> int:
        tokenized = self.tokenize(texts, model)
        return sum([len(t) for t in tokenized])
