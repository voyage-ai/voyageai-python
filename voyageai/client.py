import json
import functools
from typing import Any, Dict, Iterable, List, Optional, Union

import voyageai
from voyageai.util import default_api_key
from voyageai.embeddings_object import EmbeddingsObject


class Client:
    """Voyage AI Client

    Args:
        api_key (str): Your API key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
    ) -> None:
        
        self.api_key = api_key or default_api_key()
        # self.max_retries = max_retries
        # self.timeout = timeout
        # self.batch_size = voyageai.VOYAGE_EMBED_BATCH_SIZE

        self._params = {
            "api_key": self.api_key
        }

    def embed(
        self,
        texts: List[str],
        model: str = voyageai.VOYAGE_EMBED_DEFAULT_MODEL,
        input_type: Optional[str] = None,
        truncation: Optional[bool] = None,
    ) -> EmbeddingsObject:
        result = EmbeddingsObject()
        
        response = voyageai.Embedding.create(
            input=texts,
            model=model,
            input_type=input_type,
            truncation=truncation,
            **self._params,
        )
        result.update(response)

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

        tokenizer = Tokenizer.from_pretrained('voyageai/voyage')
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
