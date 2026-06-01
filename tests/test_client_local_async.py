"""Tests for async local model support in AsyncClient."""

import asyncio

import pytest

# ruff: noqa: F401

# Check if real dependencies are available
try:
    import sentence_transformers
    import torch

    REAL_DEPS_AVAILABLE = True
except ImportError:
    REAL_DEPS_AVAILABLE = False


@pytest.mark.integration
class TestAsyncLocalModelIntegration:
    """Integration tests for async local models using the standard AsyncClient.

    Run with: pytest -m integration
    """

    @pytest.fixture
    def check_deps(self):
        """Skip if dependencies not installed."""
        if not REAL_DEPS_AVAILABLE:
            pytest.skip("sentence-transformers or torch not installed")

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
    async def test_concurrent_local_embeddings(self, check_deps):
        """Test concurrent local embedding calls."""
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
            assert len(result.embeddings[0]) == dim, (
                f"Expected {dim}, got {len(result.embeddings[0])}"
            )

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
