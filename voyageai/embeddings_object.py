from typing import List, Optional

from voyageai import voyage_object


class EmbeddingsObject:

    def __init__(self, embeddings: Optional[List[float]] = None, total_tokens: int = 0):
        self.embeddings = embeddings or []
        self.total_tokens = total_tokens
        self._raw_responses = []

    def update(self, response: voyage_object.VoyageObject):
        self._raw_responses.append(response)
        for d in response.data:
            self.embeddings.append(d.embedding)
        self.total_tokens += response.usage.total_tokens
