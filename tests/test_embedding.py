import pytest
import voyageai


class TestEmbedding:

    model = "voyage-2"
    sample_text = "This is a test query."
    sample_texts = [
        "This is a test query.",
        "This is a test query 1.",
        "This is a test query 2.",
    ]

    def test_create_embedding(self):
        response = voyageai.Embedding.create(input=self.sample_text, model=self.model)
        assert "data" in response
        assert len(response["data"]) == 1
        assert len(response["data"][0]["embedding"]) == 1024

    def test_create_embedding_multiple(self):
        response = voyageai.Embedding.create(input=self.sample_texts, model=self.model)
        assert "data" in response
        assert len(response["data"]) == 3
        for i in range(3):
            assert len(response["data"][i]["embedding"]) == 1024

    @pytest.mark.asyncio
    async def test_acreate_embedding(self):
        response = await voyageai.Embedding.acreate(
            input=self.sample_text, model=self.model
        )
        assert "data" in response
        assert len(response["data"]) == 1
        assert len(response["data"][0]["embedding"]) == 1024

    @pytest.mark.asyncio
    async def test_acreate_embedding_multiple(self):
        response = await voyageai.Embedding.acreate(
            input=self.sample_texts, model=self.model
        )
        assert "data" in response
        assert len(response["data"]) == 3
        for i in range(3):
            assert len(response["data"][i]["embedding"]) == 1024

    def test_get_embedding(self):
        embd = voyageai.get_embedding(self.sample_text, model=self.model)
        assert len(embd) == 1024

    def test_get_embeddings(self):
        embds = voyageai.get_embeddings(self.sample_texts, model=self.model)
        assert len(embds) == 3
        for i in range(3):
            assert len(embds[i]) == 1024
        embds = voyageai.get_embeddings(self.sample_texts * 10, model=self.model)
        assert len(embds) == 30
        for i in range(30):
            assert len(embds[i]) == 1024

    @pytest.mark.asyncio
    async def test_aget_embedding(self):
        embd = await voyageai.aget_embedding(self.sample_text, model=self.model)
        assert len(embd) == 1024

    @pytest.mark.asyncio
    async def test_aget_embeddings(self):
        embds = await voyageai.aget_embeddings(self.sample_texts, model=self.model)
        assert len(embds) == 3
        for i in range(3):
            assert len(embds[i]) == 1024
        embds = await voyageai.aget_embeddings(self.sample_texts * 10, model=self.model)
        assert len(embds) == 30
        for i in range(30):
            assert len(embds[i]) == 1024

    def test_get_embedding_input_type(self):
        query_embd = voyageai.get_embedding(
            self.sample_text, model=self.model, input_type="query"
        )
        doc_embd = voyageai.get_embedding(
            self.sample_text, model=self.model, input_type="document"
        )
        assert len(query_embd) == 1024
        assert len(doc_embd) == 1024
        assert query_embd[0] != doc_embd[0]
