"""Unit tests for internal modules to improve coverage.

These tests do NOT call the Voyage API — they mock external dependencies
and exercise pure logic, error handling, and edge cases.
"""

import copy
import json
import logging
import pickle
import shutil
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests
import voyageai
from voyageai import error, util
from voyageai.api_resources.api_requestor import (
    APIRequestor,
    VoyageHttpResponse,
    _aiohttp_proxies_arg,
    _build_api_url,
    _make_session,
    _requests_proxies_arg,
)
from voyageai.api_resources.response import (
    VoyageResponse,
    convert_to_dict,
    convert_to_voyage_response,
)
from voyageai.video_utils import (
    _load_video_bytes,
    _parse_fps,
    _round_to_multiple,
)

# ---------------------------------------------------------------------------
# voyageai.api_resources.response
# ---------------------------------------------------------------------------


class TestVoyageResponse:
    def test_construct_from(self):
        resp = VoyageResponse.construct_from({"key": "value", "nested": {"a": 1}})
        assert resp["key"] == "value"
        assert isinstance(resp["nested"], VoyageResponse)

    def test_setattr_getattr(self):
        resp = VoyageResponse.construct_from({"foo": "bar"})
        resp.baz = "qux"
        assert resp["baz"] == "qux"
        assert resp.baz == "qux"

    def test_getattr_missing_raises(self):
        resp = VoyageResponse.construct_from({"foo": "bar"})
        with pytest.raises(AttributeError):
            _ = resp.nonexistent

    def test_getattr_private_raises(self):
        resp = VoyageResponse()
        with pytest.raises(AttributeError):
            _ = resp._nonexistent

    def test_delattr_raises(self):
        resp = VoyageResponse.construct_from({"foo": "bar"})
        with pytest.raises(NotImplementedError):
            del resp.foo

    def test_setitem_empty_string_raises(self):
        resp = VoyageResponse()
        with pytest.raises(ValueError, match="empty string"):
            resp["key"] = ""

    def test_delitem_raises(self):
        resp = VoyageResponse.construct_from({"foo": "bar"})
        with pytest.raises(NotImplementedError):
            del resp["foo"]

    def test_repr_with_object(self):
        resp = VoyageResponse.construct_from({"object": "embedding"})
        r = repr(resp)
        assert "VoyageResponse" in r
        assert "embedding" in r

    def test_repr_without_object(self):
        resp = VoyageResponse.construct_from({"foo": "bar"})
        r = repr(resp)
        assert "VoyageResponse" in r

    def test_str_json(self):
        resp = VoyageResponse.construct_from({"key": "value"})
        s = str(resp)
        parsed = json.loads(s)
        assert parsed["key"] == "value"

    def test_to_dict(self):
        resp = VoyageResponse.construct_from({"a": 1})
        d = resp.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_recursive(self):
        resp = VoyageResponse.construct_from({"nested": {"a": 1}, "items": [{"b": 2}]})
        d = resp.to_dict_recursive()
        assert isinstance(d["nested"], dict)
        assert not isinstance(d["nested"], VoyageResponse)
        assert isinstance(d["items"][0], dict)
        assert not isinstance(d["items"][0], VoyageResponse)

    def test_copy(self):
        resp = VoyageResponse.construct_from({"key": "value"})
        copied = copy.copy(resp)
        assert copied["key"] == "value"
        assert isinstance(copied, VoyageResponse)

    def test_deepcopy(self):
        resp = VoyageResponse.construct_from({"key": "value", "nested": {"a": 1}})
        copied = copy.deepcopy(resp)
        assert copied["key"] == "value"
        assert copied["nested"]["a"] == 1

    def test_pickle_unpickle(self):
        resp = VoyageResponse.construct_from({"key": "value"})
        data = pickle.dumps(resp)
        loaded = pickle.loads(data)
        assert loaded["key"] == "value"

    def test_refresh_from(self):
        resp = VoyageResponse.construct_from({"old": "data"})
        resp.refresh_from({"new": "data"})
        assert "new" in resp
        assert "old" not in resp


class TestConvertFunctions:
    def test_convert_to_voyage_response_dict(self):
        result = convert_to_voyage_response({"a": 1})
        assert isinstance(result, VoyageResponse)
        assert result["a"] == 1

    def test_convert_to_voyage_response_list(self):
        result = convert_to_voyage_response([{"a": 1}, {"b": 2}])
        assert isinstance(result, list)
        assert all(isinstance(r, VoyageResponse) for r in result)

    def test_convert_to_voyage_response_primitive(self):
        assert convert_to_voyage_response(42) == 42
        assert convert_to_voyage_response("hello") == "hello"

    def test_convert_to_voyage_response_http_response(self):
        http_resp = VoyageHttpResponse({"key": "val"}, {})
        result = convert_to_voyage_response(http_resp)
        assert isinstance(result, VoyageResponse)

    def test_convert_to_dict_nested(self):
        resp = VoyageResponse.construct_from({"nested": {"a": 1}})
        d = convert_to_dict(resp)
        assert isinstance(d, dict)
        assert isinstance(d["nested"], dict)

    def test_convert_to_dict_list(self):
        result = convert_to_dict([{"a": 1}])
        assert isinstance(result, list)

    def test_convert_to_dict_primitive(self):
        assert convert_to_dict(42) == 42


# ---------------------------------------------------------------------------
# voyageai.error
# ---------------------------------------------------------------------------


class TestVoyageError:
    def test_basic_error(self):
        e = error.VoyageError("test message")
        assert str(e) == "test message"
        assert e.user_message == "test message"

    def test_error_with_request_id(self):
        e = error.VoyageError("msg", headers={"request-id": "req_123"})
        assert "req_123" in str(e)
        assert e.request_id == "req_123"

    def test_error_repr(self):
        e = error.VoyageError("msg", http_status=400)
        r = repr(e)
        assert "VoyageError" in r
        assert "400" in r

    def test_error_with_bytes_body(self):
        e = error.VoyageError("msg", http_body=b"bytes body")
        assert e.http_body == "bytes body"

    def test_error_with_non_decodable_body(self):
        e = error.VoyageError("msg", http_body=b"\xff\xfe")
        assert e.http_body == "<Could not decode body as utf-8.>"

    def test_error_none_message(self):
        e = error.VoyageError(None)
        assert str(e) == "<empty message>"

    def test_construct_error_object_with_error_dict(self):
        # A structured {"error": {...}} body is parsed into an attribute-access
        # object so callers can read e.error.message / e.error.type.
        e = error.VoyageError(
            "msg",
            json_body={"error": {"message": "bad request", "type": "invalid"}},
        )
        assert e.error.message == "bad request"
        assert e.error.type == "invalid"

    def test_construct_error_object_without_error_key(self):
        e = error.VoyageError("msg", json_body={"detail": "something"})
        assert e.error is None

    def test_construct_error_object_none_json(self):
        e = error.VoyageError("msg", json_body=None)
        assert e.error is None

    def test_subclasses_exist(self):
        assert issubclass(error.APIError, error.VoyageError)
        assert issubclass(error.RateLimitError, error.VoyageError)
        assert issubclass(error.InvalidRequestError, error.VoyageError)
        assert issubclass(error.AuthenticationError, error.VoyageError)
        assert issubclass(error.Timeout, error.VoyageError)
        assert issubclass(error.ServerError, error.VoyageError)
        assert issubclass(error.ServiceUnavailableError, error.VoyageError)
        assert issubclass(error.VideoProcessingError, error.VoyageError)


# ---------------------------------------------------------------------------
# voyageai.util
# ---------------------------------------------------------------------------


class TestUtil:
    def test_api_key_to_header(self):
        h = util.api_key_to_header("test-key")
        assert h == {"Authorization": "Bearer test-key"}

    def test_logfmt_simple(self):
        result = util.logfmt({"key": "value"})
        assert "key=value" in result

    def test_logfmt_with_spaces(self):
        result = util.logfmt({"key": "has spaces"})
        assert "key=" in result

    def test_logfmt_bytes_value(self):
        result = util.logfmt({"key": b"bytes"})
        assert "key=bytes" in result

    def test_logfmt_non_string_value(self):
        result = util.logfmt({"key": 42})
        assert "key=42" in result

    def test_log_debug(self, capsys):
        with patch("voyageai.util._console_log_level", return_value="debug"):
            util.log_debug("test message", extra="param")
        err = capsys.readouterr().err
        assert "extra=param" in err
        assert "test message" in err

    def test_log_info(self, capsys):
        with patch("voyageai.util._console_log_level", return_value="info"):
            util.log_info("hello")
        assert "message=hello" in capsys.readouterr().err

    def test_log_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="voyage"):
            util.log_warn("test warning")
        assert "test warning" in caplog.text

    def test_console_log_level_from_module(self):
        original = voyageai.log
        try:
            voyageai.log = "debug"
            assert util._console_log_level() == "debug"
            voyageai.log = "info"
            assert util._console_log_level() == "info"
        finally:
            voyageai.log = original

    def test_console_log_level_none(self):
        original = voyageai.log
        try:
            voyageai.log = None
            with patch.dict("os.environ", {"VOYAGE_LOG": ""}, clear=False):
                assert util._console_log_level() is None
        finally:
            voyageai.log = original

    def test_default_api_key_from_module(self):
        original = voyageai.api_key
        try:
            voyageai.api_key = "test-key-123"
            assert util.default_api_key() == "test-key-123"
        finally:
            voyageai.api_key = original

    def test_default_api_key_missing_raises(self):
        original_key = voyageai.api_key
        original_path = voyageai.api_key_path
        try:
            voyageai.api_key = None
            voyageai.api_key_path = None
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(error.AuthenticationError):
                    util.default_api_key()
        finally:
            voyageai.api_key = original_key
            voyageai.api_key_path = original_path

    def test_default_api_key_from_file(self, tmp_path):
        key_file = tmp_path / "key.txt"
        key_file.write_text("  file-key-456  \n")
        original = voyageai.api_key_path
        try:
            voyageai.api_key_path = str(key_file)
            assert util.default_api_key() == "file-key-456"
        finally:
            voyageai.api_key_path = original

    def test_get_default_base_url_voyage(self):
        assert util.get_default_base_url("pa-abc") == "https://api.voyageai.com/v1"

    def test_get_default_base_url_mongodb(self):
        assert util.get_default_base_url("al-abc") == "https://ai.mongodb.com/v1"

    def test_resolve_numpy_dtype(self):
        assert util._resolve_numpy_dtype(None) == np.float32
        assert util._resolve_numpy_dtype("float") == np.float32
        assert util._resolve_numpy_dtype("int8") == np.int8
        assert util._resolve_numpy_dtype("uint8") == np.uint8

    def test_resolve_numpy_dtype_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            util._resolve_numpy_dtype("float64")

    def test_decode_base64_embedding(self):
        import base64

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        encoded = base64.b64encode(arr.tobytes()).decode()
        result = util.decode_base64_embedding(encoded)
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_decode_base64_embedding_int8(self):
        import base64

        arr = np.array([1, -1, 0], dtype=np.int8)
        encoded = base64.b64encode(arr.tobytes()).decode()
        result = util.decode_base64_embedding(encoded, dtype="int8")
        assert result == [1, -1, 0]


# ---------------------------------------------------------------------------
# voyageai.api_resources.api_requestor — helpers and error handling
# ---------------------------------------------------------------------------


class TestApiRequestorHelpers:
    def test_build_api_url_no_base_query(self):
        result = _build_api_url("https://api.example.com/v1/embed", "key=val")
        assert result == "https://api.example.com/v1/embed?key=val"

    def test_build_api_url_with_base_query(self):
        result = _build_api_url("https://api.example.com/v1/embed?a=1", "b=2")
        assert result == "https://api.example.com/v1/embed?a=1&b=2"

    def test_requests_proxies_none(self):
        assert _requests_proxies_arg(None) is None

    def test_requests_proxies_string(self):
        result = _requests_proxies_arg("http://proxy:8080")
        assert result == {"http": "http://proxy:8080", "https": "http://proxy:8080"}

    def test_requests_proxies_dict(self):
        result = _requests_proxies_arg({"http": "http://p1", "https": "http://p2"})
        assert result["http"] == "http://p1"

    def test_requests_proxies_invalid_raises(self):
        with pytest.raises(ValueError):
            _requests_proxies_arg(123)

    def test_aiohttp_proxies_none(self):
        assert _aiohttp_proxies_arg(None) is None

    def test_aiohttp_proxies_string(self):
        assert _aiohttp_proxies_arg("http://proxy") == "http://proxy"

    def test_aiohttp_proxies_dict_https(self):
        assert _aiohttp_proxies_arg({"https": "http://p"}) == "http://p"

    def test_aiohttp_proxies_dict_http_only(self):
        assert _aiohttp_proxies_arg({"http": "http://p"}) == "http://p"

    def test_aiohttp_proxies_invalid_raises(self):
        with pytest.raises(ValueError):
            _aiohttp_proxies_arg(42)

    def test_make_session_default(self):
        s = _make_session()
        assert isinstance(s, requests.Session)
        s.close()

    def test_make_session_custom_callable(self):
        mock_session = MagicMock(spec=requests.Session)
        original = voyageai.requestssession
        try:
            voyageai.requestssession = lambda: mock_session
            result = _make_session()
            assert result is mock_session
        finally:
            voyageai.requestssession = original

    def test_make_session_custom_instance(self):
        s = requests.Session()
        original = voyageai.requestssession
        try:
            voyageai.requestssession = s
            result = _make_session()
            assert result is s
        finally:
            voyageai.requestssession = original
            s.close()


class TestVoyageHttpResponse:
    def test_properties(self):
        headers = {
            "request-id": "req_123",
            "retry-after": "5",
            "operation-location": "/ops/1",
            "Voyage-Organization": "org_1",
            "Voyage-Processing-Ms": "42.5",
        }
        resp = VoyageHttpResponse({"result": "ok"}, headers)
        assert resp.request_id == "req_123"
        assert resp.retry_after == 5
        assert resp.operation_location == "/ops/1"
        assert resp.organization == "org_1"
        assert resp.response_ms == 42

    def test_retry_after_none(self):
        resp = VoyageHttpResponse({}, {})
        assert resp.retry_after is None

    def test_response_ms_none(self):
        resp = VoyageHttpResponse({}, {})
        assert resp.response_ms is None


class TestAPIRequestorErrorHandling:
    def setup_method(self):
        self.requestor = APIRequestor(key="test-key")

    def test_interpret_response_line_204(self):
        resp = self.requestor._interpret_response_line("", 204, {})
        assert resp.data is None

    def test_interpret_response_line_500(self):
        with pytest.raises(error.ServerError):
            self.requestor._interpret_response_line("error", 500, {})

    def test_interpret_response_line_503(self):
        with pytest.raises(error.ServiceUnavailableError):
            self.requestor._interpret_response_line("error", 503, {})

    def test_interpret_response_line_502(self):
        with pytest.raises(error.ServiceUnavailableError):
            self.requestor._interpret_response_line("error", 502, {})

    def test_interpret_response_line_504(self):
        with pytest.raises(error.ServiceUnavailableError):
            self.requestor._interpret_response_line("error", 504, {})

    def test_interpret_response_line_json_decode_error(self):
        with pytest.raises(error.APIError, match="HTTP code 200"):
            self.requestor._interpret_response_line(
                "not json", 200, {"Content-Type": "application/json"}
            )

    def test_interpret_response_line_text_plain(self):
        resp = self.requestor._interpret_response_line(
            "plain text", 200, {"Content-Type": "text/plain"}
        )
        assert resp.data == "plain text"

    def test_interpret_response_line_200_json(self):
        body = json.dumps({"key": "value"})
        resp = self.requestor._interpret_response_line(body, 200, {})
        assert resp.data == {"key": "value"}

    def test_handle_error_response_400(self):
        e = self.requestor.handle_error_response('{"detail":"bad"}', 400, {"detail": "bad"}, {})
        assert isinstance(e, error.InvalidRequestError)

    def test_handle_error_response_401(self):
        e = self.requestor.handle_error_response(
            '{"detail":"unauth"}', 401, {"detail": "unauth"}, {}
        )
        assert isinstance(e, error.AuthenticationError)

    def test_handle_error_response_422(self):
        e = self.requestor.handle_error_response(
            '{"detail":"malformed"}', 422, {"detail": "malformed"}, {}
        )
        assert isinstance(e, error.MalformedRequestError)

    def test_handle_error_response_429(self):
        e = self.requestor.handle_error_response(
            '{"detail":"rate limit"}', 429, {"detail": "rate limit"}, {}
        )
        assert isinstance(e, error.RateLimitError)

    def test_handle_error_response_other_4xx(self):
        e = self.requestor.handle_error_response('{"detail":"other"}', 418, {"detail": "other"}, {})
        assert isinstance(e, error.APIError)

    def test_handle_error_response_missing_detail(self):
        with pytest.raises(error.APIError, match="Invalid response"):
            self.requestor.handle_error_response("{}", 400, {}, {})

    def test_validate_headers_none(self):
        assert self.requestor._validate_headers(None) == {}

    def test_validate_headers_valid(self):
        h = self.requestor._validate_headers({"X-Custom": "value"})
        assert h["X-Custom"] == "value"

    def test_validate_headers_not_dict(self):
        with pytest.raises(TypeError, match="dictionary"):
            self.requestor._validate_headers("not a dict")

    def test_validate_headers_non_string_key(self):
        with pytest.raises(TypeError, match="keys must be strings"):
            self.requestor._validate_headers({123: "value"})

    def test_validate_headers_non_string_value(self):
        with pytest.raises(TypeError, match="values must be strings"):
            self.requestor._validate_headers({"key": 123})

    def test_prepare_request_raw_post(self):
        url, headers, data = self.requestor._prepare_request_raw(
            "/embed", None, "post", {"model": "voyage-3"}, None, None
        )
        assert "/embed" in url
        assert headers["Content-Type"] == "application/json"
        parsed = json.loads(data)
        assert parsed["model"] == "voyage-3"

    def test_prepare_request_raw_get_with_params(self):
        url, headers, data = self.requestor._prepare_request_raw(
            "/models", None, "get", {"limit": "10"}, None, None
        )
        assert "limit=10" in url
        assert data is None

    def test_prepare_request_raw_unrecognized_method(self):
        with pytest.raises(error.APIConnectionError, match="Unrecognized"):
            self.requestor._prepare_request_raw("/test", None, "patch", {}, None, None)

    def test_request_headers(self):
        headers = self.requestor.request_headers("post", {}, None)
        assert "Authorization" in headers

    def test_request_headers_with_request_id(self):
        headers = self.requestor.request_headers("post", {}, "req_abc")
        assert "Authorization" in headers


# ---------------------------------------------------------------------------
# voyageai.embeddings_utils — input validation
# ---------------------------------------------------------------------------


class TestEmbeddingsUtils:
    def test_check_input_type_valid(self):
        from voyageai.embeddings_utils import _check_input_type

        _check_input_type(None)
        _check_input_type("query")
        _check_input_type("document")

    def test_check_input_type_invalid(self):
        from voyageai.embeddings_utils import _check_input_type

        with pytest.raises(ValueError, match="invalid"):
            _check_input_type("invalid_type")

    def test_get_embeddings_internal(self):
        from voyageai.embeddings_utils import _get_embeddings

        mock_data = [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]
        mock_response = MagicMock()
        mock_response.data = mock_data
        with patch("voyageai.Embedding.create", return_value=mock_response):
            result = _get_embeddings(["text1", "text2"], model="voyage-3")
            assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_get_embeddings_internal_input_type_check(self):
        from voyageai.embeddings_utils import _get_embeddings

        with pytest.raises(ValueError, match="invalid"):
            _get_embeddings(["text"], model="voyage-3", input_type="bad")

    @pytest.mark.asyncio
    async def test_aget_embeddings_internal(self):
        from voyageai.embeddings_utils import _aget_embeddings

        mock_data = [{"embedding": [0.1]}, {"embedding": [0.2]}]
        mock_response = MagicMock()
        mock_response.data = mock_data
        with patch("voyageai.Embedding.acreate", return_value=mock_response):
            result = await _aget_embeddings(["a", "b"], model="voyage-3")
            assert result == [[0.1], [0.2]]

    def test_get_embedding_deprecation_warning(self):
        from voyageai.embeddings_utils import get_embedding

        with (
            pytest.warns(match="deprecated"),
            patch("voyageai.embeddings_utils._get_embeddings", return_value=[[0.1, 0.2]]),
        ):
            result = get_embedding("test")
            assert result == [0.1, 0.2]

    def test_get_embeddings_deprecation_warning(self):
        from voyageai.embeddings_utils import get_embeddings

        with (
            pytest.warns(match="deprecated"),
            patch(
                "voyageai.embeddings_utils._get_embeddings",
                return_value=[[0.1], [0.2]],
            ),
        ):
            result = get_embeddings(["a", "b"])
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_aget_embedding_deprecation_warning(self):
        from voyageai.embeddings_utils import aget_embedding

        with (
            pytest.warns(match="deprecated"),
            patch(
                "voyageai.embeddings_utils._aget_embeddings",
                return_value=[[0.1, 0.2]],
            ),
        ):
            result = await aget_embedding("test")
            assert result == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_aget_embeddings_deprecation_warning(self):
        from voyageai.embeddings_utils import aget_embeddings

        with (
            pytest.warns(match="deprecated"),
            patch(
                "voyageai.embeddings_utils._aget_embeddings",
                return_value=[[0.1], [0.2]],
            ),
        ):
            result = await aget_embeddings(["a", "b"])
            assert len(result) == 2


# ---------------------------------------------------------------------------
# voyageai.video_utils — pure helpers
# ---------------------------------------------------------------------------


class TestVideoUtilsHelpers:
    def test_parse_fps_fraction(self):
        assert _parse_fps("30/1") == 30.0

    def test_parse_fps_decimal(self):
        assert _parse_fps("29.97") == pytest.approx(29.97)

    def test_parse_fps_zero_denominator(self):
        assert _parse_fps("30/0") == 0.0

    def test_parse_fps_invalid(self):
        assert _parse_fps("abc") == 0.0

    def test_round_to_multiple(self):
        assert _round_to_multiple(100, 32) == 96
        assert _round_to_multiple(1, 32) == 32
        assert _round_to_multiple(48, 32) == 64

    def test_round_to_multiple_zero(self):
        assert _round_to_multiple(100, 0) == 100

    def test_load_video_bytes_from_bytes(self):
        data = b"fake video data"
        assert _load_video_bytes(data) == data

    def test_load_video_bytes_from_bytearray(self):
        data = bytearray(b"fake video data")
        result = _load_video_bytes(data)
        assert result == b"fake video data"
        assert isinstance(result, bytes)

    def test_load_video_bytes_from_path(self, tmp_path):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"video content")
        assert _load_video_bytes(str(f)) == b"video content"

    def test_load_video_bytes_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported video type"):
            _load_video_bytes(12345)

    def test_load_video_bytes_from_video_object(self):
        from voyageai.video_utils import Video

        v = Video(b"video data", model="test-model")
        assert _load_video_bytes(v) == b"video data"


# ---------------------------------------------------------------------------
# voyageai._base — client init and tokenizer
# ---------------------------------------------------------------------------


class TestBaseClient:
    def test_client_init_with_key(self):
        c = voyageai.Client(api_key="test-key-abc")
        assert c.api_key == "test-key-abc"

    def test_client_init_no_key_raises(self):
        original_key = voyageai.api_key
        original_path = voyageai.api_key_path
        try:
            voyageai.api_key = None
            voyageai.api_key_path = None
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(error.AuthenticationError):
                    voyageai.Client()
        finally:
            voyageai.api_key = original_key
            voyageai.api_key_path = original_path

    def test_tokenize_without_model_warns(self):
        c = voyageai.Client(api_key="test-key")
        with pytest.warns(match="specify the `model`"):
            c.tokenize(["hello world"], model=None)

    def test_count_tokens(self):
        c = voyageai.Client(api_key="test-key")
        count = c.count_tokens(["hello world"], model="voyage-3")
        assert isinstance(count, int)
        assert count > 0


class TestMakeSessionProxy:
    def test_make_session_with_proxy(self):
        original = voyageai.proxy
        try:
            voyageai.proxy = "http://myproxy:8080"
            s = _make_session()
            assert s.proxies == {
                "http": "http://myproxy:8080",
                "https": "http://myproxy:8080",
            }
            s.close()
        finally:
            voyageai.proxy = original


# ---------------------------------------------------------------------------
# video_utils — ffmpeg integration tests
# ---------------------------------------------------------------------------

# These exercise the real ffmpeg binary and the optional `ffmpeg-python`
# package (the `[video]` extra). Skip the whole class when either is missing so
# the suite stays runnable without the extra installed instead of erroring.
_FFMPEG_AVAILABLE = voyageai.video_utils.ffmpeg is not None and shutil.which("ffmpeg") is not None


@pytest.mark.skipif(
    not _FFMPEG_AVAILABLE,
    reason="requires ffmpeg on PATH and the voyageai[video] extra",
)
class TestVideoUtilsIntegration:
    EXAMPLE_VIDEO = "tests/example_video_01.mp4"

    def test_probe_video(self):
        from voyageai.video_utils import _probe_video

        meta = _probe_video(self.EXAMPLE_VIDEO)
        assert meta["width"] > 0
        assert meta["height"] > 0
        assert meta["duration"] > 0

    def test_compute_basic_usage_for_path(self):
        from voyageai.video_utils import _compute_basic_usage_for_path

        num_pixels, num_frames, _ = _compute_basic_usage_for_path(
            self.EXAMPLE_VIDEO, model="voyage-multimodal-3.5"
        )
        assert num_pixels is not None and num_pixels > 0
        assert num_frames is not None and num_frames > 0

    def test_compute_basic_usage_for_path_bad_file(self):
        from voyageai.video_utils import _compute_basic_usage_for_path

        assert _compute_basic_usage_for_path("/nonexistent.mp4", model="x") == (None, None, None)

    def test_get_video_token_config(self):
        from voyageai.video_utils import _get_video_token_config

        config = _get_video_token_config("voyage-multimodal-3.5")
        assert config is not None
        assert all(v > 0 for v in config)

    def test_get_video_token_config_bad_model(self):
        from voyageai.video_utils import _get_video_token_config

        assert _get_video_token_config("nonexistent-model-xyz") is None

    def test_ensure_ffmpeg_available(self):
        from voyageai.video_utils import _ensure_ffmpeg_available

        # Positive path: with ffmpeg present (guaranteed by the class skipif),
        # the check must pass without raising and return None.
        assert _ensure_ffmpeg_available() is None

    def test_video_from_file_no_optimize(self):
        from voyageai.video_utils import Video

        with open(self.EXAMPLE_VIDEO, "rb") as f:
            v = Video.from_file(f, model="voyage-multimodal-3.5", optimize=False)
        assert not v.optimized
        assert len(v.to_bytes()) > 0
        assert v.num_pixels is not None

    def test_optimize_video_from_path(self):
        from voyageai.video_utils import optimize_video

        r = optimize_video(self.EXAMPLE_VIDEO, model="voyage-multimodal-3.5", max_video_tokens=4000)
        assert r.optimized and len(r.to_bytes()) > 0

    def test_optimize_video_from_bytes(self):
        from voyageai.video_utils import optimize_video

        with open(self.EXAMPLE_VIDEO, "rb") as f:
            data = f.read()
        assert optimize_video(data, model="voyage-multimodal-3.5", max_video_tokens=4000).optimized

    def test_optimize_video_from_video_object(self):
        from voyageai.video_utils import Video, optimize_video

        v = Video.from_path(self.EXAMPLE_VIDEO, model="voyage-multimodal-3.5", optimize=False)
        assert optimize_video(v, model="voyage-multimodal-3.5", max_video_tokens=4000).optimized

    def test_optimize_video_no_resize(self):
        from voyageai.video_utils import optimize_video

        r = optimize_video(
            self.EXAMPLE_VIDEO,
            model="voyage-multimodal-3.5",
            resize=False,
            downsample_fps=False,
        )
        assert r.optimized

    def test_optimize_video_unsupported_type(self):
        from voyageai.video_utils import optimize_video

        with pytest.raises(TypeError, match="Unsupported"):
            optimize_video(12345, model="test")


# ---------------------------------------------------------------------------
# video_utils — error path tests (mocked)
# ---------------------------------------------------------------------------


class TestVideoUtilsErrorPaths:
    def test_ensure_ffmpeg_no_package(self):
        from voyageai import video_utils

        original = video_utils.ffmpeg
        try:
            video_utils.ffmpeg = None
            with pytest.raises(ImportError, match="ffmpeg-python"):
                video_utils._ensure_ffmpeg_available()
        finally:
            video_utils.ffmpeg = original

    def test_ensure_ffmpeg_not_on_path(self):
        with patch("shutil.which", return_value=None):
            from voyageai.video_utils import _ensure_ffmpeg_available

            with pytest.raises(EnvironmentError, match="not found on PATH"):
                _ensure_ffmpeg_available()

    def test_ensure_ffmpeg_execution_fails(self):
        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("subprocess.run", side_effect=OSError("exec failed")),
        ):
            from voyageai.video_utils import _ensure_ffmpeg_available

            with pytest.raises(EnvironmentError, match="Failed to execute"):
                _ensure_ffmpeg_available()

    def test_probe_video_no_video_stream(self):
        fake_probe = {"streams": [{"codec_type": "audio"}], "format": {}}
        with patch("ffmpeg.probe", return_value=fake_probe):
            from voyageai.video_utils import _probe_video

            with pytest.raises(ValueError, match="No video stream"):
                _probe_video("fake.mp4")

    def test_compute_basic_usage_zero_fps(self):
        with patch(
            "voyageai.video_utils._probe_video",
            return_value={"width": 640, "height": 480, "r_frame_rate": "0/0", "duration": 2.0},
        ):
            from voyageai.video_utils import _compute_basic_usage_for_path

            assert _compute_basic_usage_for_path("f.mp4", model="x") == (None, None, None)

    def test_compute_basic_usage_zero_duration(self):
        with patch(
            "voyageai.video_utils._probe_video",
            return_value={"width": 640, "height": 480, "r_frame_rate": "30/1", "duration": 0.0},
        ):
            from voyageai.video_utils import _compute_basic_usage_for_path

            assert _compute_basic_usage_for_path("f.mp4", model="x") == (None, None, None)

    def test_compute_basic_usage_no_video_config(self):
        with (
            patch(
                "voyageai.video_utils._probe_video",
                return_value={
                    "width": 640,
                    "height": 480,
                    "r_frame_rate": "30/1",
                    "duration": 1.0,
                },
            ),
            patch("voyageai.video_utils._get_video_token_config", return_value=None),
        ):
            from voyageai.video_utils import _compute_basic_usage_for_path

            num_pixels, num_frames, estimated_tokens = _compute_basic_usage_for_path(
                "f.mp4", model="x"
            )
            assert num_pixels is not None and num_pixels > 0
            assert num_frames is not None
            assert estimated_tokens is None


# ---------------------------------------------------------------------------
# _base — _get_client_config
# ---------------------------------------------------------------------------


class TestBaseClientConfig:
    def test_get_client_config_valid(self, tmp_path):
        from voyageai._base import _get_client_config

        # Hermetic: patch the HF Hub download so the test never hits the network
        # (this file's contract is "no external API / mock external deps").
        config_file = tmp_path / "client_config.json"
        config_file.write_text(json.dumps({"multimodal_image_pixels_min": 1}))

        with patch("voyageai._base.hf_hub_download", return_value=str(config_file)):
            config = _get_client_config("voyage-multimodal-3.5")

        assert isinstance(config, dict)
        assert "multimodal_image_pixels_min" in config

    def test_get_client_config_invalid_model(self):
        from huggingface_hub.utils import HfHubHTTPError
        from voyageai._base import _get_client_config

        # Reproduce what HF raises for a missing repo and assert the concrete
        # type _get_client_config re-raises, not a blanket Exception (which would
        # also pass on an error raised before the code under test). Pass a real
        # Response so construction works across huggingface_hub versions — newer
        # releases make `response` a required keyword-only argument.
        fake_response = requests.Response()
        fake_response.status_code = 404
        with patch(
            "voyageai._base.hf_hub_download",
            side_effect=HfHubHTTPError(
                "404 Client Error: model not found", response=fake_response
            ),
        ):
            with pytest.warns(match="Failed to load"):
                with pytest.raises(HfHubHTTPError):
                    _get_client_config("nonexistent-model-xyz-12345")
