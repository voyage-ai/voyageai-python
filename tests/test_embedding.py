import pytest
import voyageai


class TestEmbedding:

    model = "voyage-api-v0"

    def test_create_embedding(self):
        text = "This is a test query."
        response = voyageai.Embedding.create(input=text, model=self.model)
        assert "data" in response
        assert len(response["data"]) == 1
        assert len(response["data"][0]["embedding"]) == 1024

    def test_create_embedding_multiple(self):
        texts = [
            "This is a test query.",
            "This is a test query 1.",
            "This is a test query 2."
        ]
        response = voyageai.Embedding.create(input=texts, model=self.model)
        assert "data" in response
        assert len(response["data"]) == 3
        for i in range(3):
            assert len(response["data"][i]["embedding"]) == 1024

    @pytest.mark.asyncio
    async def test_acreate_embedding(self):
        text = "This is a test query."
        response = await voyageai.Embedding.acreate(input=text, model=self.model)
        assert "data" in response
        assert len(response["data"]) == 1
        assert len(response["data"][0]["embedding"]) == 1024
    
    @pytest.mark.asyncio
    async def test_acreate_embedding_multiple(self):
        texts = [
            "This is a test query.",
            "This is a test query 1.",
            "This is a test query 2."
        ]
        response = await voyageai.Embedding.acreate(input=texts, model=self.model)
        assert "data" in response
        assert len(response["data"]) == 3
        for i in range(3):
            assert len(response["data"][i]["embedding"]) == 1024
