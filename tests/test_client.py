import pytest
import voyageai


class TestClient:

    model = "voyage-2"
    sample_text = "This is a test query."
    sample_texts = [
        "This is a test query.",
        "This is a test query 1.",
        "This is a test query 2."
    ]

    def test_client_embed(self):
        vo = voyageai.Client()
        result = vo.embed([self.sample_text], model=self.model)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert result.total_tokens > 0

        result = vo.embed(self.sample_texts, model=self.model)
        assert len(result.embeddings) == 3
        for i in range(3):
            assert len(result.embeddings[i]) == 1024
        assert result.total_tokens > 0

    def test_client_embed_input_type(self):
        vo = voyageai.Client()
        query_embd = vo.embed(
            [self.sample_text], model=self.model, input_type="query").embeddings[0]
        doc_embd = vo.embed(
            [self.sample_text], model=self.model, input_type="document").embeddings[0]
        assert len(query_embd) == 1024
        assert len(doc_embd) == 1024
        assert query_embd[0] != doc_embd[0]

    def test_client_embed_batch_size(self):
        vo = voyageai.Client()
        with pytest.raises(voyageai.error.InvalidRequestError):
            vo.embed([self.sample_text] * 200, model=self.model)

    def test_client_embed_model_name(self):
        vo = voyageai.Client()
        with pytest.raises(voyageai.error.InvalidRequestError):
            vo.embed(self.sample_text, model="wrong-model-name")

    def test_client_embed_context_length(self):
        vo = voyageai.Client()
        texts = self.sample_texts + [self.sample_text * 1000]
        
        with pytest.raises(voyageai.error.InvalidRequestError):
            vo.embed(texts, model=self.model)
        
        result = vo.embed(texts, model=self.model, truncation=True)
        assert result.total_tokens <= 4096

        with pytest.raises(voyageai.error.InvalidRequestError):
            vo.embed(texts, model=self.model, truncation=False)

    def test_client_embed_malformed(self):
        vo = voyageai.Client()
        with pytest.raises(voyageai.error.InvalidRequestError):
            vo.embed(self.sample_texts, model=self.model, truncation="test")

        with pytest.raises(voyageai.error.InvalidRequestError):
            vo.embed(self.sample_text, input_type="doc")

    def test_client_tokenize(self):
        vo = voyageai.Client()
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
