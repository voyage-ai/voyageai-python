from dataclasses import dataclass
from typing import List, Optional, Union

from voyageai.api_resources import VoyageResponse
from voyageai.error import ServerError


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
        server_texts_per_doc: List[Optional[List[str]]] = []

        for i, d in enumerate(response.data):
            embeddings = [embd.embedding for embd in d.data]

            if client_chunk_texts is not None:
                result_chunk_texts = client_chunk_texts[i]
            else:
                per_doc_texts = [embd.get("text") for embd in d.data]
                if all(t is not None for t in per_doc_texts):
                    server_texts_per_doc.append(per_doc_texts)
                    result_chunk_texts = per_doc_texts
                elif any(t is not None for t in per_doc_texts):
                    raise ServerError(
                        f"inputs[{i}] returned a partial set of chunk texts; "
                        "expected text on every chunk or none"
                    )
                else:
                    server_texts_per_doc.append(None)
                    result_chunk_texts = None

            self.results.append(
                ContextualizedEmbeddingsResult(
                    index=i,
                    embeddings=embeddings,
                    chunk_texts=result_chunk_texts,
                )
            )

        if client_chunk_texts is None and server_texts_per_doc:
            populated = [t for t in server_texts_per_doc if t is not None]
            if populated and len(populated) != len(server_texts_per_doc):
                raise ServerError(
                    "response returned chunk texts for some documents but not others; "
                    "expected all-or-nothing"
                )
            if populated:
                self.chunk_texts = populated

        self.total_tokens += response.usage.total_tokens
        self.chunker_version = response.get("chunker_version")
