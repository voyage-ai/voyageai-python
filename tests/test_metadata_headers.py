import platform
from unittest.mock import patch

import pytest
import voyageai
from voyageai._base import _build_metadata_headers
from voyageai.version import VERSION


class TestBuildMetadataHeaders:
    def test_required_headers_present(self):
        headers = _build_metadata_headers()
        assert headers["X-VoyageAI-Lang"] == "python"
        assert headers["X-VoyageAI-Package"] == "voyageai"
        assert headers["X-VoyageAI-Package-Version"] == VERSION
        assert headers["X-VoyageAI-Runtime"] == platform.python_implementation()
        assert headers["X-VoyageAI-Runtime-Version"] == platform.python_version()
        assert headers["X-VoyageAI-OS"] == platform.system()
        assert headers["X-VoyageAI-Telemetry-Version"] == "1"

    def test_user_agent_format(self):
        headers = _build_metadata_headers()
        expected = (
            f"voyageai-python/{VERSION} "
            f"Python/{platform.python_version()} "
            f"{platform.system()}/{platform.machine()}"
        )
        assert headers["User-Agent"] == expected

    def test_fail_open_on_version_error(self):
        with patch("voyageai._base.platform.python_implementation", side_effect=OSError):
            headers = _build_metadata_headers()
        assert headers["X-VoyageAI-Lang"] == "python"
        assert "X-VoyageAI-Runtime" not in headers
        assert "X-VoyageAI-Runtime-Version" not in headers


class TestClientMetadataHeaders:
    def _make_client(self, **kwargs):
        return voyageai.Client(api_key="test-key", **kwargs)

    def test_metadata_headers_on_client(self):
        client = self._make_client()
        h = client._metadata_headers
        assert h["X-VoyageAI-Lang"] == "python"
        assert h["X-VoyageAI-Package-Version"] == VERSION
        assert h["X-VoyageAI-Runtime"] == platform.python_implementation()
        assert "X-VoyageAI-Wrapper" not in h

    def test_metadata_headers_in_params(self):
        client = self._make_client()
        assert client._params["headers"] is client._metadata_headers


class TestAppendClientMetadata:
    def _make_client(self):
        return voyageai.Client(api_key="test-key")

    def test_single_wrapper(self):
        client = self._make_client()
        client.append_client_metadata(name="mem0", version="1.2.3")
        assert client._metadata_headers["X-VoyageAI-Wrapper"] == "mem0/1.2.3"

    def test_multiple_wrappers(self):
        client = self._make_client()
        client.append_client_metadata(name="mem0", version="1.2.3")
        client.append_client_metadata(name="llamaindex", version="0.10.5")
        assert client._metadata_headers["X-VoyageAI-Wrapper"] == "mem0/1.2.3|llamaindex/0.10.5"

    def test_idempotent(self):
        client = self._make_client()
        client.append_client_metadata(name="mem0", version="1.2.3")
        client.append_client_metadata(name="mem0", version="1.2.3")
        assert client._metadata_headers["X-VoyageAI-Wrapper"] == "mem0/1.2.3"

    def test_idempotent_with_multiple(self):
        client = self._make_client()
        client.append_client_metadata(name="mem0", version="1.2.3")
        client.append_client_metadata(name="langchain", version="0.1.0")
        client.append_client_metadata(name="mem0", version="1.2.3")
        assert client._metadata_headers["X-VoyageAI-Wrapper"] == "mem0/1.2.3|langchain/0.1.0"

    @pytest.mark.parametrize(
        "name,version",
        [
            ("mem0\r\nX-Evil", "1.0"),  # CRLF injection attempt
            ("mem0", "1.0\nInjected: yes"),
            ("mem0|other", "1.0"),  # reserved list delimiter
            ("mem0/x", "1.0"),  # reserved name/version delimiter
            ("", "1.0"),  # empty name
            ("mem0", "   "),  # blank version
        ],
    )
    def test_invalid_wrapper_fields_rejected(self, name, version):
        client = self._make_client()
        with pytest.raises(ValueError):
            client.append_client_metadata(name=name, version=version)
        assert "X-VoyageAI-Wrapper" not in client._metadata_headers

    def test_wrapper_fields_are_trimmed(self):
        client = self._make_client()
        client.append_client_metadata(name="  mem0  ", version=" 1.2.3 ")
        assert client._metadata_headers["X-VoyageAI-Wrapper"] == "mem0/1.2.3"


class TestHeadersFlowToRequest:
    def test_metadata_headers_sent_on_wire(self, monkeypatch):
        captured_headers = {}

        class FakeResponse:
            status_code = 200
            headers = {"Content-Type": "application/json"}
            content = b'{"data": [{"embedding": [0.1]}], "usage": {"total_tokens": 5}}'

        def fake_request(self, method, url, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return FakeResponse()

        import requests

        monkeypatch.setattr(requests.Session, "request", fake_request)

        client = voyageai.Client(api_key="test-key")
        client.append_client_metadata(name="testlib", version="0.1.0")
        client.embed(texts=["hello"], model="voyage-2")

        assert captured_headers["X-VoyageAI-Lang"] == "python"
        assert captured_headers["X-VoyageAI-Package"] == "voyageai"
        assert captured_headers["X-VoyageAI-Package-Version"] == VERSION
        assert captured_headers["X-VoyageAI-Runtime"] == platform.python_implementation()
        assert captured_headers["X-VoyageAI-Runtime-Version"] == platform.python_version()
        assert captured_headers["X-VoyageAI-OS"] == platform.system()
        assert captured_headers["X-VoyageAI-Wrapper"] == "testlib/0.1.0"
        assert captured_headers["X-VoyageAI-Telemetry-Version"] == "1"
        assert f"voyageai-python/{VERSION}" in captured_headers["User-Agent"]
        assert captured_headers["Authorization"] == "Bearer test-key"

    def test_unexpected_headers_stripped(self, monkeypatch):
        captured_headers = {}

        class FakeResponse:
            status_code = 200
            headers = {"Content-Type": "application/json"}
            content = b'{"data": [{"embedding": [0.1]}], "usage": {"total_tokens": 5}}'

        def fake_request(self, method, url, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return FakeResponse()

        import requests

        monkeypatch.setattr(requests.Session, "request", fake_request)

        client = voyageai.Client(api_key="test-key")
        client._metadata_headers["X-Evil-Header"] = "should-be-stripped"
        client.embed(texts=["hello"], model="voyage-2")

        assert "X-Evil-Header" not in captured_headers
        assert "X-VoyageAI-Lang" in captured_headers

    def test_app_info_preserved_in_user_agent(self, monkeypatch):
        captured_headers = {}

        class FakeResponse:
            status_code = 200
            headers = {"Content-Type": "application/json"}
            content = b'{"data": [{"embedding": [0.1]}], "usage": {"total_tokens": 5}}'

        def fake_request(self, method, url, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return FakeResponse()

        import requests

        monkeypatch.setattr(requests.Session, "request", fake_request)
        monkeypatch.setattr(
            voyageai, "app_info", {"name": "myapp", "version": "2.0", "url": "https://example.com"}
        )

        client = voyageai.Client(api_key="test-key")
        client.embed(texts=["hello"], model="voyage-2")

        # The metadata User-Agent must still carry app_info (it isn't dropped
        # when the metadata headers are merged), and exactly once.
        ua = captured_headers["User-Agent"]
        assert f"voyageai-python/{VERSION}" in ua
        assert "myapp/2.0 (https://example.com)" in ua
        assert ua.count("myapp/2.0") == 1


class TestAsyncHeadersFlowToRequest:
    @pytest.mark.asyncio
    async def test_metadata_headers_sent_on_wire_async(self, monkeypatch):
        captured_headers = {}

        class FakeAioResponse:
            status = 200
            headers = {"Content-Type": "application/json"}
            content = b'{"data": [{"embedding": [0.1]}], "usage": {"total_tokens": 5}}'

            async def read(self):
                return self.content

            def release(self):
                return None

        async def fake_request(self, **kwargs):
            captured_headers.update(kwargs.get("headers") or {})
            return FakeAioResponse()

        import aiohttp

        monkeypatch.setattr(aiohttp.ClientSession, "request", fake_request)

        client = voyageai.AsyncClient(api_key="test-key")
        client.append_client_metadata(name="testlib", version="0.1.0")
        await client.embed(texts=["hello"], model="voyage-2")

        assert captured_headers["X-VoyageAI-Lang"] == "python"
        assert captured_headers["X-VoyageAI-Package"] == "voyageai"
        assert captured_headers["X-VoyageAI-Package-Version"] == VERSION
        assert captured_headers["X-VoyageAI-Runtime"] == platform.python_implementation()
        assert captured_headers["X-VoyageAI-OS"] == platform.system()
        assert captured_headers["X-VoyageAI-Wrapper"] == "testlib/0.1.0"
        assert captured_headers["X-VoyageAI-Telemetry-Version"] == "1"
        assert f"voyageai-python/{VERSION}" in captured_headers["User-Agent"]
        assert captured_headers["Authorization"] == "Bearer test-key"
