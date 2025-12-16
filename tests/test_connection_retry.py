import pytest
from unittest.mock import Mock, patch
import voyageai
import voyageai.error as error
from voyageai.api_resources.response import VoyageResponse


class TestConnectionRetry:
    """Test cases to verify APIConnectionError retry behavior."""
    
    sample_query = "This is a test query."
    embed_model = "voyage-2"
    
    def test_sync_client_connection_error_retry(self):
        """Test that sync client retries APIConnectionError."""
        with patch('voyageai.Embedding.create') as mock_create:
            # Set up mock to raise APIConnectionError once, then succeed
            mock_create.side_effect = [
                error.APIConnectionError("Connection aborted", None, None),
                VoyageResponse.construct_from({
                    "data": [{"embedding": [0.1] * 1024}],
                    "usage": {"total_tokens": 10}
                })
            ]
            
            vo = voyageai.Client(max_retries=2)
            result = vo.embed([self.sample_query], model=self.embed_model)
            
            # Verify it was called 2 times (1 initial attempt + 1 retry)
            assert mock_create.call_count == 2
            assert len(result.embeddings) == 1
    
    def test_sync_client_connection_error_max_retries_exceeded(self):
        """Test that sync client fails after max_retries for APIConnectionError."""
        with patch('voyageai.Embedding.create') as mock_create:
            # Set up mock to always raise APIConnectionError
            mock_create.side_effect = error.APIConnectionError(
                "Connection aborted", None, None
            )
            
            vo = voyageai.Client(max_retries=1)
            
            with pytest.raises(error.APIConnectionError):
                vo.embed([self.sample_query], model=self.embed_model)
            
            # Verify it was called 1 time only (max_retries=1 means 1 total attempt)
            assert mock_create.call_count == 1
    
    def test_sync_client_connection_error_no_retry_when_max_retries_zero(self):
        """Test that sync client doesn't retry when max_retries=0."""
        with patch('voyageai.Embedding.create') as mock_create:
            mock_create.side_effect = error.APIConnectionError(
                "Connection aborted", None, None
            )
            
            vo = voyageai.Client(max_retries=0)
            
            with pytest.raises(error.APIConnectionError):
                vo.embed([self.sample_query], model=self.embed_model)
            
            # Verify it was called only once (no retries)
            assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_async_client_connection_error_retry(self):
        """Test that async client retries APIConnectionError."""
        with patch('voyageai.Embedding.acreate') as mock_acreate:
            # Set up mock to raise APIConnectionError once, then succeed
            mock_acreate.side_effect = [
                error.APIConnectionError("Connection aborted", None, None),
                VoyageResponse.construct_from({
                    "data": [{"embedding": [0.1] * 1024}],
                    "usage": {"total_tokens": 10}
                })
            ]
            
            vo = voyageai.AsyncClient(max_retries=2)
            result = await vo.embed([self.sample_query], model=self.embed_model)
            
            # Verify it was called 2 times (1 initial attempt + 1 retry)
            assert mock_acreate.call_count == 2
            assert len(result.embeddings) == 1

    @pytest.mark.asyncio
    async def test_async_client_connection_error_max_retries_exceeded(self):
        """Test that async client fails after max_retries for APIConnectionError."""
        with patch('voyageai.Embedding.acreate') as mock_acreate:
            # Set up mock to always raise APIConnectionError
            mock_acreate.side_effect = error.APIConnectionError(
                "Connection aborted", None, None
            )
            
            vo = voyageai.AsyncClient(max_retries=1)
            
            with pytest.raises(error.APIConnectionError):
                await vo.embed([self.sample_query], model=self.embed_model)
            
            # Verify it was called 1 time only (max_retries=1 means 1 total attempt)
            assert mock_acreate.call_count == 1

    @pytest.mark.asyncio  
    async def test_async_client_connection_error_no_retry_when_max_retries_zero(self):
        """Test that async client doesn't retry when max_retries=0."""
        with patch('voyageai.Embedding.acreate') as mock_acreate:
            mock_acreate.side_effect = error.APIConnectionError(
                "Connection aborted", None, None
            )
            
            vo = voyageai.AsyncClient(max_retries=0)
            
            with pytest.raises(error.APIConnectionError):
                await vo.embed([self.sample_query], model=self.embed_model)
            
            # Verify it was called only once (no retries)
            assert mock_acreate.call_count == 1

    def test_sync_client_other_errors_still_retry(self):
        """Verify that other retryable errors still work (RateLimitError, etc.)."""
        with patch('voyageai.Embedding.create') as mock_create:
            # Set up mock to raise RateLimitError once, then succeed
            mock_create.side_effect = [
                error.RateLimitError("Rate limit exceeded", None, 429, None),
                VoyageResponse.construct_from({
                    "data": [{"embedding": [0.1] * 1024}],
                    "usage": {"total_tokens": 10}
                })
            ]
            
            vo = voyageai.Client(max_retries=2)
            result = vo.embed([self.sample_query], model=self.embed_model)
            
            # Verify it was called 2 times (max_retries=1 means 1 initial + 1 retry if first fails)
            assert mock_create.call_count == 2
            assert len(result.embeddings) == 1

    def test_sync_client_non_retryable_errors_not_retried(self):
        """Verify that non-retryable errors are not retried."""
        with patch('voyageai.Embedding.create') as mock_create:
            # Set up mock to raise InvalidRequestError (non-retryable)
            mock_create.side_effect = error.InvalidRequestError(
                "Invalid model", None, 400, None
            )
            
            vo = voyageai.Client(max_retries=2)
            
            with pytest.raises(error.InvalidRequestError):
                vo.embed([self.sample_query], model=self.embed_model)
            
            # Verify it was called only once (no retries for non-retryable errors)
            assert mock_create.call_count == 1