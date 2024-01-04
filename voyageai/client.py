import json
from typing import Any, Dict, Iterable, List, Optional, Union

import voyageai
from voyageai.util import default_api_key


class Client:
    """Voyage AI Client

    Args:
        api_key (str): Your API key.
        num_workers (int): Maximal number of threads for parallelized calls.
        max_retries (int): maximal number of retries for requests.
        timeout (int): request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str = None,
        max_retries: int = 3,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or default_api_key()
        # self.batch_size = voyageai.VOYAGE_EMBED_BATCH_SIZE
        self.max_retries = max_retries
        self.timeout = timeout

        self._params = {
            "api_key": self.api_key
        }

    def embed(
        self,
        texts: List[str],
        model: str = voyageai.VOYAGE_EMBED_DEFAULT_MODEL,
        input_type: Optional[str] = None,
        truncation: Optional[bool] = None,
        truncation_side: Optional[str] = None,
    ) -> list:
        result = voyageai.Embedding.create(
            input=texts,
            model=model,
            input_type=input_type,
            truncation=truncation,
            truncation_side=truncation_side,
            **self._params,
        )
        return result

    def tokenize(
        self,
    ) -> List[str]:
        pass

    def count_tokens(
        self,
    ) -> int:
        pass
