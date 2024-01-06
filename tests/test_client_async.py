import pytest
import voyageai


class TestAsyncClient:

    model = "voyage-01"
    sample_text = "This is a test query."
    sample_texts = [
        "This is a test query.",
        "This is a test query 1.",
        "This is a test query 2."
    ]

    @pytest.mark.asyncio
    async def test_async_client_embed(self):
        vo = voyageai.AsyncClient()
        result = await vo.embed([self.sample_text], model=self.model)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert result.total_tokens > 0

        result = await vo.embed(self.sample_texts, model=self.model)
        assert len(result.embeddings) == 3
        for i in range(3):
            assert len(result.embeddings[i]) == 1024
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_async_client_embed_input_type(self):
        vo = voyageai.AsyncClient()
        query_embd = (await vo.embed(
            [self.sample_text], model=self.model, input_type="query")
        ).embeddings[0]
        doc_embd = (await vo.embed(
            [self.sample_text], model=self.model, input_type="document")
        ).embeddings[0]
        assert len(query_embd) == 1024
        assert len(doc_embd) == 1024
        assert query_embd[0] != doc_embd[0]

    def test_async_client_tokenize(self):
        vo = voyageai.AsyncClient()
        result = vo.tokenize(self.sample_texts)
        assert isinstance(result, list)
        assert len(result) == 3
        assert len(result[0].tokens) == 7
        assert len(result[1].tokens) == 9
        assert len(result[2].tokens) == 9

    def test_client_count_tokens(self):
        vo = voyageai.Client()
        total_tokens = vo.count_tokens([self.sample_text])
        assert total_tokens == 7

        total_tokens = vo.count_tokens(self.sample_texts)
        assert total_tokens == 25
