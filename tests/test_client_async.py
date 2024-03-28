import pytest
import voyageai
import voyageai.error as error


class TestAsyncClient:

    embed_model = "voyage-2"
    rerank_model = "rerank-lite-1"

    sample_query = "This is a test query."
    sample_docs = [
        "This is a test document.",
        "This is a test document 1.",
        "This is a test document 2."
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
        query_embd = (await vo.embed(
            [self.sample_query], model=self.embed_model, input_type="query")
        ).embeddings[0]
        doc_embd = (await vo.embed(
            self.sample_docs, model=self.embed_model, input_type="document")
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
    async def test_client_embed_timeout(self):
        vo = voyageai.AsyncClient(timeout=1, max_retries=1)
        with pytest.raises(error.Timeout):
            await vo.embed([self.sample_query * 100] * 100, model=self.embed_model)

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
    async def test_client_rerank_invalid_request(self):
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
        result = vo.tokenize(self.sample_docs)
        assert isinstance(result, list)
        assert len(result) == 3
        assert len(result[0].tokens) == 7
        assert len(result[1].tokens) == 9
        assert len(result[2].tokens) == 9

    def test_client_count_tokens(self):
        vo = voyageai.Client()
        total_tokens = vo.count_tokens([self.sample_query])
        assert total_tokens == 7

        total_tokens = vo.count_tokens(self.sample_docs)
        assert total_tokens == 25
