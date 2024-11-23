from typing import List, Optional, Union
from voyageai.api_resources import VoyageResponse


class EmbeddingsObject:

    def __init__(self, response: Optional[VoyageResponse] = None):
        self.embeddings: Union[List[List[float]], List[List[int]]] = []
        self.total_tokens: int = 0
        if response:
            self.update(response)

    def update(self, response: VoyageResponse):
        for d in response.data:
            self.embeddings.append(d.embedding)
        self.total_tokens += response.usage.total_tokens
