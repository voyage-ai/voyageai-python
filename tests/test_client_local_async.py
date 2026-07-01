"""Tests for async local model support in AsyncClient."""

import asyncio

import pytest

from tests._local_test_utils import real_deps_available

REAL_DEPS_AVAILABLE = real_deps_available()


@pytest.mark.integration
class TestAsyncLocalModelIntegration:
    """Integration tests for async local models using the standard AsyncClient.

    Run with: pytest -m integration
    """

    @pytest.fixture(scope="session")
    def check_deps(self):
        """Skip if dependencies not installed or model can't load.

        Loading happens here (not at import/collection time) and only the
        documented failure modes are swallowed: a genuinely broken backend
        (download error, OOM, runtime error) must surface, not turn into a
        misleading skip.
        """
        if not REAL_DEPS_AVAILABLE:
            pytest.skip("sentence-transformers or torch not installed")
        try:
            from voyageai.local.sentence_transformer_backend import SentenceTransformerBackend

            SentenceTransformerBackend("voyage-4-nano")
        except TypeError:
            pytest.skip("local model failed to load (requires Python 3.10+)")
        except ImportError:
            pytest.skip("local model dependencies not available")

    @pytest.mark.asyncio
    async def test_seamless_async_local_embedding(self, check_deps):
        """Test that AsyncClient.embed() seamlessly uses local model."""
        from voyageai import AsyncClient

        client = AsyncClient()
        result = await client.embed(["Hello, world!"], model="voyage-4-nano", input_type="document")

        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 2048
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_concurrent_local_embeds_return_correct_results(self, check_deps):
        """Concurrently-dispatched local embeds each return a correct result.

        Note: these are dispatched concurrently (asyncio.gather over
        asyncio.to_thread), but the per-model inference lock serializes the
        actual forward passes on the shared cached model — so this verifies
        correctness under concurrent dispatch, not parallel execution. (A batched
        encode would be the path to real parallelism if throughput matters.)
        """
        from voyageai import AsyncClient

        client = AsyncClient()

        texts = [
            ["First document"],
            ["Second document"],
            ["Third document"],
        ]

        tasks = [client.embed(t, model="voyage-4-nano", input_type="document") for t in texts]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert len(result.embeddings) == 1
            assert len(result.embeddings[0]) == 2048

    @pytest.mark.asyncio
    async def test_async_all_dimensions(self, check_deps):
        """Test all supported dimensions in async context."""
        from voyageai import AsyncClient

        client = AsyncClient()

        for dim in [256, 512, 1024, 2048]:
            result = await client.embed(
                ["Test text"], model="voyage-4-nano", input_type="document", output_dimension=dim
            )
            assert (
                len(result.embeddings[0]) == dim
            ), f"Expected {dim}, got {len(result.embeddings[0])}"

    @pytest.mark.asyncio
    async def test_async_query_vs_document(self, check_deps):
        """Test query and document embeddings are different in async context."""
        from voyageai import AsyncClient

        client = AsyncClient()

        query_result = await client.embed(
            ["What is AI?"], model="voyage-4-nano", input_type="query"
        )
        doc_result = await client.embed(
            ["What is AI?"], model="voyage-4-nano", input_type="document"
        )

        assert query_result.embeddings[0] != doc_result.embeddings[0]

    @pytest.mark.asyncio
    async def test_async_batch_embedding(self, check_deps):
        """Test batch embedding in async context."""
        from voyageai import AsyncClient

        client = AsyncClient()

        result = await client.embed(
            ["First", "Second", "Third"], model="voyage-4-nano", input_type="document"
        )
        assert len(result.embeddings) == 3

    @pytest.mark.asyncio
    async def test_async_invalid_input_type_raises(self, check_deps):
        """Unknown input_type must raise on the async path too, matching the API."""
        from voyageai import AsyncClient
        from voyageai.error import InvalidRequestError

        client = AsyncClient()
        with pytest.raises(InvalidRequestError):
            await client.embed(["test"], model="voyage-4-nano", input_type="doc")

    @pytest.mark.asyncio
    async def test_async_truncation_false_over_length_raises(self, check_deps):
        """truncation=False on over-length input must raise on the async path too."""
        from voyageai import AsyncClient
        from voyageai.error import InvalidRequestError

        client = AsyncClient()
        long_text = "word " * 40000
        with pytest.raises(InvalidRequestError):
            await client.embed([long_text], model="voyage-4-nano", truncation=False)


class TestAsyncLocalInputValidation:
    """Empty / None / oversized-batch input must raise InvalidRequestError on the
    async path too, matching the sync path and the hosted API. These raise before
    any model load, so they need no local deps or network."""

    @pytest.mark.asyncio
    async def test_async_empty_list_raises(self):
        from voyageai import AsyncClient
        from voyageai.error import InvalidRequestError

        with pytest.raises(InvalidRequestError):
            await AsyncClient().embed([], model="voyage-4-nano")

    @pytest.mark.asyncio
    async def test_async_none_input_raises(self):
        from voyageai import AsyncClient
        from voyageai.error import InvalidRequestError

        with pytest.raises(InvalidRequestError):
            await AsyncClient().embed(None, model="voyage-4-nano")

    @pytest.mark.asyncio
    async def test_async_batch_over_limit_raises(self):
        from voyageai import AsyncClient
        from voyageai.error import InvalidRequestError

        with pytest.raises(InvalidRequestError):
            await AsyncClient().embed(["x"] * 1001, model="voyage-4-nano")
