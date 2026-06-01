"""Tests for local model support in Client."""

import pytest

# ruff: noqa: F401

# Check if real dependencies are available
try:
    import sentence_transformers
    import torch

    REAL_DEPS_AVAILABLE = True
except ImportError:
    REAL_DEPS_AVAILABLE = False


class TestLocalModelSupport:
    """Test local model detection and routing."""

    @pytest.mark.skipif(REAL_DEPS_AVAILABLE, reason="Only run when deps not installed")
    def test_import_error_when_deps_missing(self):
        """Test helpful error message when sentence-transformers not installed."""
        from voyageai.local import _ensure_local_deps

        with pytest.raises(ImportError) as exc_info:
            _ensure_local_deps()

        assert "pip install voyageai[local]" in str(exc_info.value)

    def test_has_local_constant(self):
        """Test HAS_LOCAL constant reflects dependency availability."""
        import voyageai

        assert voyageai.HAS_LOCAL == REAL_DEPS_AVAILABLE


@pytest.mark.integration
class TestLocalModelIntegration:
    """Integration tests for local models using the standard Client.

    Run with: pytest -m integration
    """

    @pytest.fixture
    def check_deps(self):
        """Skip if dependencies not installed."""
        if not REAL_DEPS_AVAILABLE:
            pytest.skip("sentence-transformers or torch not installed")

    def test_seamless_local_embedding(self, check_deps):
        """Test that Client.embed() seamlessly uses local model."""
        from voyageai import Client

        # No API key needed for local models
        client = Client()
        result = client.embed(["Hello, world!"], model="voyage-4-nano", input_type="document")

        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 2048
        assert result.total_tokens > 0

    def test_all_dimensions(self, check_deps):
        """Test all supported dimensions work."""
        from voyageai import Client

        client = Client()

        for dim in [256, 512, 1024, 2048]:
            result = client.embed(
                ["Test text"], model="voyage-4-nano", input_type="document", output_dimension=dim
            )
            assert len(result.embeddings[0]) == dim, (
                f"Expected {dim}, got {len(result.embeddings[0])}"
            )

    def test_float32_dtype(self, check_deps):
        """Test float32 output data type (default)."""
        from voyageai import Client

        client = Client()

        result = client.embed(
            ["Test"], model="voyage-4-nano", input_type="document", output_dtype="float32"
        )
        assert isinstance(result.embeddings[0][0], float)

    def test_query_vs_document_different(self, check_deps):
        """Test query and document embeddings are different."""
        from voyageai import Client

        client = Client()

        query_result = client.embed(
            ["What is machine learning?"], model="voyage-4-nano", input_type="query"
        )
        doc_result = client.embed(
            ["What is machine learning?"], model="voyage-4-nano", input_type="document"
        )

        # Embeddings should be different due to different prompts
        assert query_result.embeddings[0] != doc_result.embeddings[0]

    def test_batch_embedding(self, check_deps):
        """Test batch embedding works."""
        from voyageai import Client

        client = Client()

        texts = [
            "First document",
            "Second document",
            "Third document",
        ]
        result = client.embed(texts, model="voyage-4-nano", input_type="document")

        assert len(result.embeddings) == 3
        for emb in result.embeddings:
            assert len(emb) == 2048

    def test_invalid_dimension_raises_error(self, check_deps):
        """Test invalid dimension raises ValueError."""
        from voyageai import Client

        client = Client()

        with pytest.raises(ValueError) as exc_info:
            client.embed(["test"], model="voyage-4-nano", output_dimension=999)

        assert "Invalid output_dimension" in str(exc_info.value)

    def test_invalid_dtype_raises_error(self, check_deps):
        """Test invalid dtype raises ValueError."""
        from voyageai import Client

        client = Client()

        with pytest.raises(ValueError) as exc_info:
            client.embed(["test"], model="voyage-4-nano", output_dtype="invalid")

        assert "Invalid output_dtype" in str(exc_info.value)
