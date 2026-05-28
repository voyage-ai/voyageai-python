import re
from typing import List
from unittest.mock import AsyncMock, patch

import pytest
import voyageai
import voyageai.error as error
from voyageai.api_resources.response import VoyageResponse
from voyageai.chunking import default_chunk_fn


class TestAsyncClient:
    embed_model = "voyage-2"
    context_embed_model = "voyage-context-3"
    rerank_model = "rerank-lite-1"

    sample_query = "This is a test query."
    sample_docs = [
        "This is a test document.",
        "This is a test document 1.",
        "This is a test document 2.",
    ]
    sample_chunked_query = [[sample_query]]
    sample_chunked_docs = [
        ["This is doc 1 chunk 1."],
        ["This is doc 2 chunk 1.", "This is doc 2 chunk 2."],
    ]

    """
    Embedding
    """

    @pytest.mark.asyncio
    async def test_async_client_embed(self):
        vo = voyageai.AsyncClient()
        result = await vo.embed([self.sample_query], model=self.embed_model)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert result.total_tokens > 0

        result = await vo.embed(self.sample_docs, model=self.embed_model)
        assert len(result.embeddings) == 3
        for i in range(3):
            assert len(result.embeddings[i]) == 1024
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_async_client_embed_input_type(self):
        vo = voyageai.AsyncClient()
        query_embd = (
            await vo.embed([self.sample_query], model=self.embed_model, input_type="query")
        ).embeddings[0]
        doc_embd = (
            await vo.embed(self.sample_docs, model=self.embed_model, input_type="document")
        ).embeddings[0]
        assert len(query_embd) == 1024
        assert len(doc_embd) == 1024
        assert query_embd[0] != doc_embd[0]

    @pytest.mark.asyncio
    async def test_async_client_embed_invalid_request(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(error.InvalidRequestError):
            await vo.embed(self.sample_query, model="wrong-model-name")

        with pytest.raises(error.InvalidRequestError):
            await vo.embed(self.sample_docs, model=self.embed_model, truncation="test")

        with pytest.raises(error.InvalidRequestError):
            await vo.embed(self.sample_query, model=self.embed_model, input_type="doc")

    @pytest.mark.asyncio
    async def test_async_client_embed_timeout(self):
        vo = voyageai.AsyncClient(timeout=1, max_retries=1)
        with pytest.raises(error.Timeout):
            await vo.embed([self.sample_query * 100] * 100, model=self.embed_model)

    @pytest.mark.asyncio
    async def test_async_client_embed_output_dtype(self):
        vo = voyageai.AsyncClient()
        result = await vo.embed([self.sample_query], model=self.embed_model)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert isinstance(result.embeddings[0][0], float)
        assert result.total_tokens > 0

        result = await vo.embed(
            [self.sample_query], model=self.embed_model, output_dtype="float", output_dimension=1024
        )
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert isinstance(result.embeddings[0][0], float)

        conversion_enabled_model = "voyage-code-3"

        result = await vo.embed([self.sample_query], model=conversion_enabled_model)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert isinstance(result.embeddings[0][0], float)

        result = await vo.embed(
            [self.sample_query],
            model=conversion_enabled_model,
            output_dtype="int8",
            output_dimension=2048,
        )
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 2048
        assert isinstance(result.embeddings[0][0], int)

        result = await vo.embed(
            [self.sample_query],
            model=conversion_enabled_model,
            output_dtype="ubinary",
            output_dimension=256,
        )
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 32
        assert isinstance(result.embeddings[0][0], int)

    """
    Contextualized embeddings
    """

    @pytest.mark.asyncio
    async def test_async_client_contextualized_embed(self):
        vo = voyageai.AsyncClient()
        result = await vo.contextualized_embed(
            inputs=self.sample_chunked_query, model=self.context_embed_model
        )
        assert len(result.results) == 1
        assert len(result.results[0].embeddings) == 1
        assert len(result.results[0].embeddings[0]) == 1024
        assert result.total_tokens > 0

        result = await vo.contextualized_embed(
            inputs=self.sample_chunked_docs, model=self.context_embed_model
        )
        assert len(result.results) == 2
        assert len(result.results[0].embeddings) == 1
        assert len(result.results[0].embeddings[0]) == 1024
        assert len(result.results[1].embeddings) == 2
        assert len(result.results[1].embeddings[0]) == 1024
        assert len(result.results[1].embeddings[1]) == 1024
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_async_client_contextualized_embed_input_type(self):
        vo = voyageai.AsyncClient()
        query_result = await vo.contextualized_embed(
            inputs=self.sample_chunked_query, model=self.context_embed_model, input_type="query"
        )
        query_embd = query_result.results[0].embeddings[0]
        doc_result = await vo.contextualized_embed(
            inputs=self.sample_chunked_docs, model=self.context_embed_model, input_type="document"
        )
        doc_embd = doc_result.results[0].embeddings[0]
        assert len(query_embd) == 1024
        assert len(doc_embd) == 1024
        assert query_embd[0] != doc_embd[0]

    @pytest.mark.asyncio
    async def test_async_client_contextualized_embed_with_chunking_fn(self):
        vo = voyageai.AsyncClient()
        doc = "I am an unchunked document"
        result = await vo.contextualized_embed(
            inputs=[[doc], [doc, doc]],
            model=self.context_embed_model,
            chunk_fn=default_chunk_fn(chunk_size=1),
        )
        assert len(result.results) == 2
        assert result.total_tokens == len(doc) * 3
        assert result.chunk_texts is not None
        assert len(result.chunk_texts) == 2
        assert len(result.chunk_texts[0]) == len(doc)
        assert len(result.chunk_texts[1]) == len(doc) * 2

    @pytest.mark.asyncio
    async def test_async_client_contextualized_embed_batch_size(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(voyageai.error.InvalidRequestError):
            await vo.contextualized_embed(
                inputs=self.sample_chunked_docs * 1100, model=self.context_embed_model
            )

    @pytest.mark.asyncio
    async def test_async_client_contextualized_embed_context_length(self):
        vo = voyageai.AsyncClient()
        texts = self.sample_chunked_docs + self.sample_chunked_query * 998

        result = await vo.contextualized_embed(inputs=texts, model=self.context_embed_model)
        assert result.total_tokens <= 6015

    @pytest.mark.asyncio
    async def test_async_client_contextualized_embed_invalid_request(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(error.InvalidRequestError):
            await vo.contextualized_embed(
                inputs=self.sample_chunked_query, model="wrong-model-name"
            )

    @pytest.mark.asyncio
    async def test_async_client_contextualized_embed_timeout(self):
        vo = voyageai.AsyncClient(timeout=1, max_retries=1)
        with pytest.raises(error.Timeout):
            await vo.contextualized_embed(
                inputs=[[self.sample_query] * 100] * 100, model=self.context_embed_model
            )

    @pytest.mark.asyncio
    async def test_async_client_contextualized_embed_output_dtype(self):
        vo = voyageai.AsyncClient()
        result = await vo.contextualized_embed(
            inputs=self.sample_chunked_query, model=self.context_embed_model
        )
        assert len(result.results) == 1
        assert len(result.results[0].embeddings) == 1
        assert len(result.results[0].embeddings[0]) == 1024
        assert isinstance(result.results[0].embeddings[0][0], float)
        assert result.total_tokens > 0

        result = await vo.contextualized_embed(
            inputs=self.sample_chunked_query,
            model=self.context_embed_model,
            output_dtype="float",
            output_dimension=1024,
        )
        assert len(result.results) == 1
        assert len(result.results[0].embeddings) == 1
        assert len(result.results[0].embeddings[0]) == 1024
        assert isinstance(result.results[0].embeddings[0][0], float)

        result = await vo.contextualized_embed(
            inputs=self.sample_chunked_query,
            model=self.context_embed_model,
            output_dtype="int8",
            output_dimension=2048,
        )
        assert len(result.results) == 1
        assert len(result.results[0].embeddings) == 1
        assert len(result.results[0].embeddings[0]) == 2048
        assert isinstance(result.results[0].embeddings[0][0], int)

        result = await vo.contextualized_embed(
            inputs=self.sample_chunked_query,
            model=self.context_embed_model,
            output_dtype="ubinary",
            output_dimension=256,
        )
        assert len(result.results) == 1
        assert len(result.results[0].embeddings) == 1
        assert len(result.results[0].embeddings[0]) == 32
        assert isinstance(result.results[0].embeddings[0][0], int)

    # ---- Auto-chunking validation (no network) ----

    @pytest.mark.asyncio
    async def test_async_auto_chunk_flat_list_requires_flag_or_query(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "List[str] inputs requires enable_auto_chunking=True or input_type='query'"
            ),
        ):
            await vo.contextualized_embed(
                inputs=["doc text"], model=self.context_embed_model
            )

    @pytest.mark.asyncio
    async def test_async_auto_chunk_requires_document_input_type(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "enable_auto_chunking=True requires input_type='document'"
            ),
        ):
            await vo.contextualized_embed(
                inputs=["doc text"],
                model=self.context_embed_model,
                input_type="query",
                enable_auto_chunking=True,
            )

    @pytest.mark.asyncio
    async def test_async_chunk_size_requires_auto_chunking(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "chunk_size and chunk_overlap require enable_auto_chunking=True"
            ),
        ):
            await vo.contextualized_embed(
                inputs=self.sample_chunked_docs,
                model=self.context_embed_model,
                chunk_size=128,
            )

    @pytest.mark.asyncio
    async def test_async_empty_inputs_rejected(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(
            ValueError,
            match=re.escape("inputs must not be empty"),
        ):
            await vo.contextualized_embed(
                inputs=[],
                model=self.context_embed_model,
                input_type="document",
                enable_auto_chunking=True,
            )

    @pytest.mark.asyncio
    async def test_async_chunk_size_must_be_at_least_one(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(
            ValueError,
            match=re.escape("chunk_size must be greater than or equal to 1"),
        ):
            await vo.contextualized_embed(
                inputs=["doc"],
                model=self.context_embed_model,
                input_type="document",
                enable_auto_chunking=True,
                chunk_size=0,
            )

    @pytest.mark.asyncio
    async def test_async_chunk_overlap_must_be_non_negative(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(
            ValueError,
            match=re.escape("chunk_overlap must be greater than or equal to 0"),
        ):
            await vo.contextualized_embed(
                inputs=["doc"],
                model=self.context_embed_model,
                input_type="document",
                enable_auto_chunking=True,
                chunk_size=128,
                chunk_overlap=-1,
            )

    @pytest.mark.asyncio
    async def test_async_chunk_overlap_requires_chunk_size(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(
            ValueError,
            match=re.escape("chunk_overlap requires chunk_size"),
        ):
            await vo.contextualized_embed(
                inputs=["doc"],
                model=self.context_embed_model,
                input_type="document",
                enable_auto_chunking=True,
                chunk_overlap=32,
            )

    @pytest.mark.asyncio
    async def test_async_chunk_overlap_must_be_less_than_chunk_size(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "chunk_overlap (64) must be less than chunk_size (64)"
            ),
        ):
            await vo.contextualized_embed(
                inputs=["doc"],
                model=self.context_embed_model,
                input_type="document",
                enable_auto_chunking=True,
                chunk_size=64,
                chunk_overlap=64,
            )

    @pytest.mark.asyncio
    async def test_async_chunk_fn_and_auto_chunking_mutually_exclusive(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "chunk_fn cannot be combined with enable_auto_chunking=True"
            ),
        ):
            await vo.contextualized_embed(
                inputs=["doc"],
                model=self.context_embed_model,
                input_type="document",
                enable_auto_chunking=True,
                chunk_fn=default_chunk_fn(chunk_size=10),
            )

    @pytest.mark.asyncio
    async def test_async_auto_chunking_rejects_pre_chunked_doc(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "inputs[0] has 2 chunks; auto-chunking expects one string per document"
            ),
        ):
            await vo.contextualized_embed(
                inputs=[["chunk1", "chunk2"]],
                model=self.context_embed_model,
                input_type="document",
                enable_auto_chunking=True,
            )

    # ---- Auto-chunking happy path (mocked transport) ----

    @staticmethod
    def _build_fake_response(
        chunk_texts: List[List[str]],
        chunker_version: str = "1.0.0",
    ) -> VoyageResponse:
        return VoyageResponse.construct_from(
            {
                "object": "list",
                "data": [
                    {
                        "object": "list",
                        "index": i,
                        "data": [
                            {
                                "object": "embedding",
                                "index": j,
                                "embedding": [0.1] * 4,
                                "text": text,
                            }
                            for j, text in enumerate(doc_chunks)
                        ],
                    }
                    for i, doc_chunks in enumerate(chunk_texts)
                ],
                "model": "voyage-context-3",
                "usage": {"total_tokens": 42},
                "chunker_version": chunker_version,
            }
        )

    @pytest.mark.asyncio
    async def test_async_auto_chunking_forwards_params_and_normalizes_flat_input(self):
        vo = voyageai.AsyncClient()
        captured: dict = {}
        fake_response = self._build_fake_response([["a", "b"], ["c"]])

        async def fake_acreate(**kwargs):
            captured.update(kwargs)
            return fake_response

        with patch(
            "voyageai.ContextualizedEmbedding.acreate",
            new=AsyncMock(side_effect=fake_acreate),
        ):
            result = await vo.contextualized_embed(
                inputs=["doc one", "doc two"],
                model=self.context_embed_model,
                input_type="document",
                enable_auto_chunking=True,
                chunk_size=256,
                chunk_overlap=32,
            )

        assert captured["inputs"] == [["doc one"], ["doc two"]]
        assert captured["enable_auto_chunking"] is True
        assert captured["chunk_size"] == 256
        assert captured["chunk_overlap"] == 32
        assert result.chunk_texts == [["a", "b"], ["c"]]
        assert result.results[0].chunk_texts == ["a", "b"]
        assert result.results[1].chunk_texts == ["c"]
        assert result.chunker_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_async_auto_chunking_with_list_of_lists_input(self):
        vo = voyageai.AsyncClient()
        captured: dict = {}
        fake_response = self._build_fake_response([["x"]])

        async def fake_acreate(**kwargs):
            captured.update(kwargs)
            return fake_response

        with patch(
            "voyageai.ContextualizedEmbedding.acreate",
            new=AsyncMock(side_effect=fake_acreate),
        ):
            result = await vo.contextualized_embed(
                inputs=[["doc one"]],
                model=self.context_embed_model,
                input_type="document",
                enable_auto_chunking=True,
            )

        assert captured["inputs"] == [["doc one"]]
        assert captured["enable_auto_chunking"] is True
        assert "chunk_size" not in captured
        assert "chunk_overlap" not in captured
        assert result.chunker_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_async_query_flat_list_normalized_without_auto_chunking(self):
        vo = voyageai.AsyncClient()
        captured: dict = {}
        fake_response = VoyageResponse.construct_from(
            {
                "object": "list",
                "data": [
                    {
                        "object": "list",
                        "index": 0,
                        "data": [
                            {"object": "embedding", "index": 0, "embedding": [0.0] * 4}
                        ],
                    }
                ],
                "model": "voyage-context-3",
                "usage": {"total_tokens": 1},
            }
        )

        async def fake_acreate(**kwargs):
            captured.update(kwargs)
            return fake_response

        with patch(
            "voyageai.ContextualizedEmbedding.acreate",
            new=AsyncMock(side_effect=fake_acreate),
        ):
            result = await vo.contextualized_embed(
                inputs=["hello"],
                model=self.context_embed_model,
                input_type="query",
            )

        assert captured["inputs"] == [["hello"]]
        assert "enable_auto_chunking" not in captured
        assert result.chunker_version is None
        assert result.chunk_texts is None

    """
    Reranker
    """

    @pytest.mark.asyncio
    async def test_async_client_rerank(self):
        vo = voyageai.AsyncClient()
        reranking = await vo.rerank(self.sample_query, self.sample_docs, self.rerank_model)
        assert len(reranking.results) == len(self.sample_docs)

        for i in range(len(self.sample_docs)):
            if i + 1 < len(self.sample_docs):
                r = reranking.results[i]
                assert r.relevance_score >= reranking.results[i + 1].relevance_score
                assert r.document == self.sample_docs[r.index]

        assert reranking.total_tokens > 0

    @pytest.mark.asyncio
    async def test_async_client_rerank_invalid_request(self):
        vo = voyageai.AsyncClient()
        with pytest.raises(error.InvalidRequestError):
            await vo.rerank(self.sample_query, self.sample_docs * 400, self.rerank_model)

        with pytest.raises(error.InvalidRequestError):
            await vo.rerank(self.sample_query, self.sample_docs, "wrong-model-name")

        # This does not exceeds single document context length limit, but exceeds the total tokens limit.
        long_docs = [self.sample_docs[0] * 100] * 1000
        with pytest.raises(error.InvalidRequestError):
            await vo.rerank(self.sample_query, long_docs, self.rerank_model)

    """
    Tokenizer
    """

    def test_async_client_tokenize(self):
        vo = voyageai.AsyncClient()
        result = vo.tokenize(self.sample_docs, self.embed_model)
        assert isinstance(result, list)
        assert len(result) == 3
        assert len(result[0].tokens) == 7
        assert len(result[1].tokens) == 9
        assert len(result[2].tokens) == 9

    def test_async_client_count_tokens(self):
        vo = voyageai.AsyncClient()
        total_tokens = vo.count_tokens([self.sample_query], self.embed_model)
        assert total_tokens == 7

        total_tokens = vo.count_tokens(self.sample_docs, self.embed_model)
        assert total_tokens == 25
