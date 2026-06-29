"""Tests for local model support in Client."""

import importlib.util

import pytest
from voyageai.error import InvalidRequestError


def _real_deps_available() -> bool:
    """Check if sentence-transformers and torch can be found (without importing)."""
    return (
        importlib.util.find_spec("sentence_transformers") is not None
        and importlib.util.find_spec("torch") is not None
    )


REAL_DEPS_AVAILABLE = _real_deps_available()


@pytest.fixture(scope="session")
def local_model_works():
    """Check if the local model can actually load.

    Fails on Python <3.10 due to HuggingFace model code using 3.10+ type
    union syntax.
    """
    if not REAL_DEPS_AVAILABLE:
        pytest.skip("sentence-transformers or torch not installed")
    try:
        from voyageai.local.sentence_transformer_backend import SentenceTransformerBackend

        SentenceTransformerBackend("voyage-4-nano")
        return True
    except TypeError:
        pytest.skip("local model failed to load (requires Python 3.10+)")
    except ImportError:
        pytest.skip("local model dependencies not available")


class TestLocalModelSupport:
    """Test local model detection and routing."""

    @pytest.mark.skipif(REAL_DEPS_AVAILABLE, reason="Only run when deps not installed")
    def test_import_error_when_deps_missing(self):
        """Test helpful error message when sentence-transformers not installed."""
        import sys

        from voyageai.local import _ensure_local_deps

        with pytest.raises(ImportError) as exc_info:
            _ensure_local_deps()

        msg = str(exc_info.value)
        if sys.version_info < (3, 10):
            assert "Python 3.10 or later" in msg
        else:
            assert "pip install voyageai[local]" in msg

    def test_has_local_constant(self):
        """Test HAS_LOCAL reflects dependency availability via independent check."""
        import voyageai

        expected = _real_deps_available()
        assert voyageai.HAS_LOCAL == expected


@pytest.mark.integration
class TestLocalModelIntegration:
    """Integration tests for local models using the standard Client.

    Run with: pytest -m integration
    """

    @pytest.fixture(autouse=True)
    def _require_local_model(self, local_model_works):
        """Skip all tests in this class if local model can't load."""

    def test_seamless_local_embedding(self):
        """Test that Client.embed() seamlessly uses local model."""
        from voyageai import Client

        # No API key needed for local models
        client = Client()
        result = client.embed(["Hello, world!"], model="voyage-4-nano", input_type="document")

        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 2048
        assert result.total_tokens > 0

    def test_all_dimensions(self):
        """Test all supported dimensions work."""
        from voyageai import Client

        client = Client()

        for dim in [256, 512, 1024, 2048]:
            result = client.embed(
                ["Test text"], model="voyage-4-nano", input_type="document", output_dimension=dim
            )
            assert (
                len(result.embeddings[0]) == dim
            ), f"Expected {dim}, got {len(result.embeddings[0])}"

    def test_float32_dtype(self):
        """Test float32 output data type (default)."""
        from voyageai import Client

        client = Client()

        result = client.embed(
            ["Test"], model="voyage-4-nano", input_type="document", output_dtype="float32"
        )
        assert isinstance(result.embeddings[0][0], float)

    def test_float_dtype(self):
        """Test 'float' output_dtype works (alias for float32)."""
        from voyageai import Client

        client = Client()

        result = client.embed(
            ["Test"], model="voyage-4-nano", input_type="document", output_dtype="float"
        )
        assert isinstance(result.embeddings[0][0], float)

    def test_query_vs_document_different(self):
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

    def test_batch_embedding(self):
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

    def test_invalid_dimension_raises_error(self):
        """Invalid dimension raises InvalidRequestError, matching the hosted API."""
        from voyageai import Client

        client = Client()

        with pytest.raises(InvalidRequestError) as exc_info:
            client.embed(["test"], model="voyage-4-nano", output_dimension=999)

        assert "Invalid output_dimension" in str(exc_info.value)

    def test_invalid_dtype_raises_error(self):
        """Invalid dtype raises InvalidRequestError, matching the hosted API."""
        from voyageai import Client

        client = Client()

        with pytest.raises(InvalidRequestError) as exc_info:
            client.embed(["test"], model="voyage-4-nano", output_dtype="invalid")

        assert "Invalid output_dtype" in str(exc_info.value)

    def test_bare_string_returns_nested_list(self):
        """Bare string input must return a list with one embedding, matching the API."""
        from voyageai import Client

        client = Client()

        result = client.embed("Hello, world!", model="voyage-4-nano", input_type="document")

        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 2048
        assert isinstance(result.embeddings[0][0], float)

    def test_invalid_input_type_raises_error(self):
        """An unknown input_type must raise, not silently degrade to no prompt."""
        from voyageai import Client

        client = Client()

        with pytest.raises(InvalidRequestError) as exc_info:
            client.embed(["test"], model="voyage-4-nano", input_type="doc")

        assert "Invalid input_type" in str(exc_info.value)

    def test_int8_dtype_not_supported(self):
        """int8/uint8 require fixed calibration ranges and must be rejected locally."""
        from voyageai import Client

        client = Client()

        for dtype in ("int8", "uint8"):
            with pytest.raises(NotImplementedError) as exc_info:
                client.embed(["test"], model="voyage-4-nano", output_dtype=dtype)
            assert "calibration ranges" in str(exc_info.value)

    def test_truncated_dimensions_are_unit_norm(self):
        """Matryoshka-truncated embeddings must stay unit-norm like the API."""
        import numpy as np
        from voyageai import Client

        client = Client()

        for dim in [256, 512, 1024, 2048]:
            result = client.embed(
                ["Test text"], model="voyage-4-nano", input_type="document", output_dimension=dim
            )
            norm = np.linalg.norm(result.embeddings[0])
            assert norm == pytest.approx(1.0, abs=1e-2), f"dim={dim} not unit-norm (norm={norm})"


class TestKeylessApiGate:
    """A keyless client is allowed (for local models) but every API-backed method
    must fail fast with a clear AuthenticationError instead of late and deep in
    the request layer. No model load or network access is required for these.
    """

    @pytest.fixture
    def keyless_client(self, monkeypatch):
        import voyageai
        from voyageai import Client

        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("VOYAGE_API_KEY_PATH", raising=False)
        monkeypatch.setattr(voyageai, "api_key", None, raising=False)
        monkeypatch.setattr(voyageai, "api_key_path", None, raising=False)
        return Client()

    def test_rerank_requires_api_key(self, keyless_client):
        import voyageai

        with pytest.raises(voyageai.error.AuthenticationError):
            keyless_client.rerank("q", ["a", "b"], model="rerank-2")

    def test_contextualized_embed_requires_api_key(self, keyless_client):
        import voyageai

        with pytest.raises(voyageai.error.AuthenticationError):
            keyless_client.contextualized_embed([["a", "b"]], model="voyage-context-3")

    def test_multimodal_embed_requires_api_key(self, keyless_client):
        import voyageai

        with pytest.raises(voyageai.error.AuthenticationError):
            keyless_client.multimodal_embed([["hello"]], model="voyage-multimodal-3")

    def test_embed_requires_api_key_for_api_model(self, keyless_client):
        import voyageai

        with pytest.raises(voyageai.error.AuthenticationError):
            keyless_client.embed(["hello"], model="voyage-3")
