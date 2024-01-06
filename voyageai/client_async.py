import json
import functools
from typing import Any, Dict, Iterable, List, Optional, Union

import voyageai
from voyageai.client import Client
from voyageai.embeddings_object import EmbeddingsObject


class AsyncClient(Client):
    """Voyage AI Async Client

    Args:
        api_key (str): Your API key.
    """

    async def embed(
        self,
        texts: List[str],
        model: str = voyageai.VOYAGE_EMBED_DEFAULT_MODEL,
        input_type: Optional[str] = None,
        truncation: Optional[bool] = None,
    ) -> EmbeddingsObject:
        result = EmbeddingsObject()
        
        response = await voyageai.Embedding.acreate(
            input=texts,
            model=model,
            input_type=input_type,
            truncation=truncation,
            **self._params,
        )
        result.update(response)

        return result
