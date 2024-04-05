from typing import List, Optional
from voyageai.api_resources import VoyageResponse


class EmbeddingsObject:

    def __init__(self, response: Optional[VoyageResponse] = None):
        self.embeddings: List[List[float]] = []
        self.total_tokens: int = 0
        if response:
            self.update(response)

    def update(self, response: VoyageResponse):
        for d in response.data:
            self.embeddings.append(d.embedding)
        self.total_tokens += response.usage.total_tokens
