from dataclasses import dataclass
from typing import List, Optional, Union

from voyageai.api_resources import VoyageResponse


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
        self.chunker_version: Optional[str] = None
        if response:
            self.update(response)

    def update(self, response: VoyageResponse):
        client_chunk_texts = self.chunk_texts
        server_chunk_texts: List[List[str]] = []
        for i, d in enumerate(response.data):
            embeddings = [embd.embedding for embd in d.data]
            per_doc_server_texts = [
                getattr(embd, "text", None) for embd in d.data
            ]
            if any(t is not None for t in per_doc_server_texts):
                server_chunk_texts.append(per_doc_server_texts)
            else:
                server_chunk_texts.append([])

            if client_chunk_texts:
                result_chunk_texts = client_chunk_texts[i]
            elif server_chunk_texts[i]:
                result_chunk_texts = server_chunk_texts[i]
            else:
                result_chunk_texts = None

            self.results.append(
                ContextualizedEmbeddingsResult(
                    index=i,
                    embeddings=embeddings,
                    chunk_texts=result_chunk_texts,
                )
            )

        if (
            client_chunk_texts is None
            and any(texts for texts in server_chunk_texts)
        ):
            self.chunk_texts = server_chunk_texts

        self.total_tokens += response.usage.total_tokens
        self.chunker_version = getattr(response, "chunker_version", None)
