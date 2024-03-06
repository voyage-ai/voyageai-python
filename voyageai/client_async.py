from typing import List, Optional

import voyageai
from voyageai.client import Client
from voyageai.object import EmbeddingsObject, RerankingObject


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
        
        response = await voyageai.Embedding.acreate(
            input=texts,
            model=model,
            input_type=input_type,
            truncation=truncation,
            **self._params,
        )

        result = EmbeddingsObject(response)
        return result

    async def rerank(
        self,
        query: str,
        documents: List[str],
        model: str,
        top_k: Optional[int] = None,
        truncation: bool = True,
    ) -> RerankingObject:

        response = await voyageai.Reranking.acreate(
            query=query,
            documents=documents,
            model=model,
            top_k=top_k,
            truncation=truncation,
            **self._params,
        )

        result = RerankingObject(documents, response)
        return result
