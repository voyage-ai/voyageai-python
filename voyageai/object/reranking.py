from collections import namedtuple
from typing import List, Optional
from voyageai.api_resources import VoyageResponse


RerankingResult = namedtuple(
    "RerankingResult", ["index", "document", "relevance_score"]
)


class RerankingObject:

    def __init__(self, documents: List[str], response: VoyageResponse):
        self.results: List[RerankingResult] = [
            RerankingResult(
                index=d.index,
                document=documents[d.index],
                relevance_score=d.relevance_score,
            )
            for d in response.data
        ]
        self.total_tokens: int = response.usage.total_tokens
