from dataclasses import dataclass
from typing import List, Optional, Union
from voyageai.api_resources import VoyageResponse
from voyageai import error

@dataclass
class ContextualizedEmbeddingsResult:
    index: int
    embeddings: Union[List[List[float]], List[List[int]]]
    chunk_texts: Optional[List[str]] = None


class ContextualizedEmbeddingsObject:
    def __init__(
        self, 
        response: Optional[VoyageResponse] = None,
        chunk_texts: Optional[List[List[str]]] = None,
    ):
        self.results: List[ContextualizedEmbeddingsResult] = []
        self.total_tokens: int = 0
        self.chunk_texts = chunk_texts
        if response:
            self.update(response)

    def update(self, response: VoyageResponse):
        if self.chunk_texts and len(response.data) != len(self.chunk_texts):
            raise error.ServerError("The server failed to process the request.")
        for i, d in enumerate(response.data):
            embeddings = [embd.embedding for embd in d.data]
            if len(self.chunk_texts[i]) != len(embeddings):
                raise error.ServerError("The server failed to process the request.")
            self.results.append(
                ContextualizedEmbeddingsResult(
                    index=i,
                    embeddings=embeddings,
                    chunk_texts=self.chunk_texts[i],
                )
            )
            self.embeddings.append(d.embedding)
        self.total_tokens += response.usage.total_tokens
