import pytest
import importlib.metadata

import voyageai
import voyageai.error as error


class TestClient:

    embed_model = "voyage-2"
    rerank_model = "rerank-lite-1"

    sample_query = "This is a test query."
    sample_docs = [
        "This is a test document.",
        "This is a test document 1.",
        "This is a test document 2.",
    ]

    """
    Embedding
    """

    def test_client_embed(self):
        vo = voyageai.Client()
        result = vo.embed([self.sample_query], model=self.embed_model)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert result.total_tokens > 0

        result = vo.embed(self.sample_docs, model=self.embed_model)
        assert len(result.embeddings) == 3
        for i in range(3):
            assert len(result.embeddings[i]) == 1024
        assert result.total_tokens > 0

    def test_client_embed_input_type(self):
        vo = voyageai.Client()
        query_embd = vo.embed(
            [self.sample_query], model=self.embed_model, input_type="query"
        ).embeddings[0]
        doc_embd = vo.embed(
            self.sample_docs, model=self.embed_model, input_type="document"
        ).embeddings[0]
        assert len(query_embd) == 1024
        assert len(doc_embd) == 1024
        assert query_embd[0] != doc_embd[0]

    def test_client_embed_batch_size(self):
        vo = voyageai.Client()
        with pytest.raises(voyageai.error.InvalidRequestError):
            vo.embed(self.sample_docs * 200, model=self.embed_model)

    def test_client_embed_context_length(self):
        vo = voyageai.Client()
        texts = self.sample_docs + [self.sample_query * 1000]

        result = vo.embed(texts, model=self.embed_model, truncation=True)
        assert result.total_tokens <= 4096

        with pytest.raises(error.InvalidRequestError):
            vo.embed(texts, model=self.embed_model, truncation=False)

    def test_client_embed_invalid_request(self):
        vo = voyageai.Client()
        with pytest.raises(error.InvalidRequestError):
            vo.embed(self.sample_query, model="wrong-model-name")

        with pytest.raises(error.InvalidRequestError):
            vo.embed(self.sample_docs, model=self.embed_model, truncation="test")

        with pytest.raises(error.InvalidRequestError):
            vo.embed(self.sample_query, model=self.embed_model, input_type="doc")

    def test_client_embed_timeout(self):
        vo = voyageai.Client(timeout=1, max_retries=1)
        with pytest.raises(error.Timeout):
            vo.embed([self.sample_query * 100] * 100, model=self.embed_model)

    def test_client_embed_output_dtype(self):
        vo = voyageai.Client()
        result = vo.embed([self.sample_query], model=self.embed_model)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert isinstance(result.embeddings[0][0], float)
        assert result.total_tokens > 0

        result = vo.embed([self.sample_query], model=self.embed_model, output_dtype="float", output_dimension=1024)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert isinstance(result.embeddings[0][0], float)

        conversion_enabled_model = "voyage-code-3"

        result = vo.embed([self.sample_query], model=conversion_enabled_model)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert isinstance(result.embeddings[0][0], float)

        result = vo.embed([self.sample_query], model=conversion_enabled_model, output_dtype="int8", output_dimension=2048)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 2048
        assert isinstance(result.embeddings[0][0], int)

        result = vo.embed([self.sample_query], model=conversion_enabled_model, output_dtype="ubinary", output_dimension=256)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 32
        assert isinstance(result.embeddings[0][0], int)

    """
    Reranker
    """

    def test_client_rerank(self):
        vo = voyageai.Client()
        reranking = vo.rerank(self.sample_query, self.sample_docs, self.rerank_model)
        assert len(reranking.results) == len(self.sample_docs)

        for i in range(len(self.sample_docs)):
            if i + 1 < len(self.sample_docs):
                r = reranking.results[i]
                assert r.relevance_score >= reranking.results[i + 1].relevance_score
                assert r.document == self.sample_docs[r.index]

        assert reranking.total_tokens > 0

    def test_client_rerank_top_k(self):
        vo = voyageai.Client()
        reranking = vo.rerank(
            self.sample_query, self.sample_docs, self.rerank_model, top_k=2
        )
        assert len(reranking.results) == 2

    def test_client_rerank_truncation(self):
        long_query = self.sample_query * 1000
        long_docs = [d * 1000 for d in self.sample_docs]
        # print(long_query)
        # print(long_docs)

        vo = voyageai.Client()
        reranking = vo.rerank(long_query, long_docs, self.rerank_model)
        assert reranking.total_tokens <= 4096 * len(long_docs)

        with pytest.raises(error.InvalidRequestError):
            reranking = vo.rerank(
                long_query, self.sample_docs, self.rerank_model, truncation=False
            )

        with pytest.raises(error.InvalidRequestError):
            reranking = vo.rerank(
                self.sample_query, long_docs, self.rerank_model, truncation=False
            )

    def test_client_rerank_invalid_request(self):
        vo = voyageai.Client()
        with pytest.raises(error.InvalidRequestError):
            vo.rerank(self.sample_query, self.sample_docs * 400, self.rerank_model)

        with pytest.raises(error.InvalidRequestError):
            vo.rerank(self.sample_query, self.sample_docs, "wrong-model-name")

        # This does not exceeds single document context length limit, but exceeds the total tokens limit.
        long_docs = [self.sample_docs[0] * 100] * 1000
        with pytest.raises(error.InvalidRequestError):
            vo.rerank(self.sample_query, long_docs, self.rerank_model)

    """
    Tokenizer
    """

    def test_client_tokenize(self):
        vo = voyageai.Client()
        result = vo.tokenize(self.sample_docs, self.embed_model)
        assert isinstance(result, list)
        assert len(result) == 3
        assert len(result[0].tokens) == 7
        assert len(result[1].tokens) == 9
        assert len(result[2].tokens) == 9

        result = vo.tokenize(self.sample_docs, self.rerank_model)
        assert len(result) == 3
        assert len(result[0].tokens) == 7
        assert len(result[1].tokens) == 9
        assert len(result[2].tokens) == 9

    def test_client_tokenize_model(self):
        vo = voyageai.Client()

        result = vo.tokenize([self.sample_query], "voyage-finance-2")
        assert len(result[0].tokens) == 7

        result = vo.tokenize([self.sample_query], "rerank-1")
        assert len(result[0].tokens) == 7

        with pytest.raises(Exception):
            with pytest.warns(UserWarning):
                vo.tokenize(self.sample_docs, "wrong-model")

    def test_client_count_tokens(self):
        vo = voyageai.Client()
        total_tokens = vo.count_tokens([self.sample_query], self.embed_model)
        assert total_tokens == 7

        total_tokens = vo.count_tokens(self.sample_docs, self.embed_model)
        assert total_tokens == 25

        total_tokens = vo.count_tokens([self.sample_query, self.sample_docs[0]], self.rerank_model)
        assert total_tokens == 14

    def test_client_version(self):
        assert voyageai.__version__ == importlib.metadata.version("voyageai")