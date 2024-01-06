import asyncio
import json
import time
import platform
import sys
import threading
import time
import warnings
from json import JSONDecodeError
from typing import (
    AsyncContextManager,
    AsyncGenerator,
    Callable,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
)
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp
import requests

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import voyageai
from voyageai import error, util, version
from voyageai.api_resources.voyage_response import VoyageResponse
from voyageai.util import ApiType

TIMEOUT_SECS = 600
MAX_SESSION_LIFETIME_SECS = 180
MAX_CONNECTION_RETRIES = 2

# Has one attribute per thread, 'session'.
_thread_context = threading.local()


def _build_api_url(url, query):
    scheme, netloc, path, base_query, fragment = urlsplit(url)

    if base_query:
        query = "%s&%s" % (base_query, query)

    return urlunsplit((scheme, netloc, path, query, fragment))


def _requests_proxies_arg(proxy) -> Optional[Dict[str, str]]:
    """Returns a value suitable for the 'proxies' argument to 'requests.request."""
    if proxy is None:
        return None
    elif isinstance(proxy, str):
        return {"http": proxy, "https": proxy}
    elif isinstance(proxy, dict):
        return proxy.copy()
    else:
        raise ValueError(
            "'voyageai.proxy' must be specified as either a string URL or a dict with string URL under the https and/or http keys."
        )


def _aiohttp_proxies_arg(proxy) -> Optional[str]:
    """Returns a value suitable for the 'proxies' argument to 'aiohttp.ClientSession.request."""
    if proxy is None:
        return None
    elif isinstance(proxy, str):
        return proxy
    elif isinstance(proxy, dict):
        return proxy["https"] if "https" in proxy else proxy["http"]
    else:
        raise ValueError(
            "'voyageai.proxy' must be specified as either a string URL or a dict with string URL under the https and/or http keys."
        )


def _make_session() -> requests.Session:
    if voyageai.requestssession:
        if isinstance(voyageai.requestssession, requests.Session):
            return voyageai.requestssession
        return voyageai.requestssession()
    if not voyageai.verify_ssl_certs:
        warnings.warn("verify_ssl_certs is ignored; voyageai always verifies.")
    s = requests.Session()
    proxies = _requests_proxies_arg(voyageai.proxy)
    if proxies:
        s.proxies = proxies
    s.mount(
        "https://",
        requests.adapters.HTTPAdapter(max_retries=MAX_CONNECTION_RETRIES),
    )
    return s


def parse_stream_helper(line: bytes) -> Optional[str]:
    if line and line.startswith(b"data:"):
        if line.startswith(b"data: "):
            # SSE event may be valid when it contain whitespace
            line = line[len(b"data: "):]
        else:
            line = line[len(b"data:"):]
        if line.strip() == b"[DONE]":
            # return here will cause GeneratorExit exception in urllib3
            # and it will close http connection with TCP Reset
            return None
        else:
            return line.decode("utf-8")
    return None


def parse_stream(rbody: Iterator[bytes]) -> Iterator[str]:
    for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line


async def parse_stream_async(rbody: aiohttp.StreamReader):
    async for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line


class APIRequestor:
    def __init__(
        self,
        key=None,
        api_base=None,
        api_type=None,
        api_version=None,
        organization=None,
    ):
        self.api_base = api_base or voyageai.api_base
        self.api_key = key or util.default_api_key()
        self.api_type = (
            ApiType.from_str(api_type)
            if api_type
            else ApiType.from_str(voyageai.api_type)
        )
        self.api_version = api_version or voyageai.api_version
        self.organization = organization or voyageai.organization

    @classmethod
    def format_app_info(cls, info):
        str = info["name"]
        if info["version"]:
            str += "/%s" % (info["version"],)
        if info["url"]:
            str += " (%s)" % (info["url"],)
        return str

    def _check_polling_response(self, response: VoyageResponse, predicate: Callable[[VoyageResponse], bool]):
        if not predicate(response):
            return
        error_data = response.data['error']
        message = error_data.get('message', 'Operation failed')
        code = error_data.get('code')
        raise error.VoyageAIError(message=message, code=code)

    def _poll(
        self,
        method,
        url,
        until,
        failed,
        params = None,
        headers = None,
        interval = None,
        delay = None
    ) -> Tuple[Iterator[VoyageResponse], bool, str]:
        if delay:
            time.sleep(delay)

        response, b, api_key = self.request(method, url, params, headers)
        self._check_polling_response(response, failed)
        start_time = time.time()
        while not until(response):
            if time.time() - start_time > TIMEOUT_SECS:
                raise error.Timeout("Operation polling timed out.")

            time.sleep(interval or response.retry_after or 10)
            response, b, api_key = self.request(method, url, params, headers)
            self._check_polling_response(response, failed)

        response.data = response.data['result']
        return response, b, api_key

    async def _apoll(
        self,
        method,
        url,
        until,
        failed,
        params = None,
        headers = None,
        interval = None,
        delay = None
    ) -> Tuple[Iterator[VoyageResponse], bool, str]:
        if delay:
            await asyncio.sleep(delay)

        response, b, api_key = await self.arequest(method, url, params, headers)
        self._check_polling_response(response, failed)
        start_time = time.time()
        while not until(response):
            if time.time() - start_time > TIMEOUT_SECS:
                raise error.Timeout("Operation polling timed out.")

            await asyncio.sleep(interval or response.retry_after or 10)
            response, b, api_key = await self.arequest(method, url, params, headers)
            self._check_polling_response(response, failed)

        response.data = response.data['result']
        return response, b, api_key

    @overload
    def request(
        self,
        method,
        url,
        params,
        headers,
        files,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Iterator[VoyageResponse], bool, str]:
        pass

    @overload
    def request(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        *,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Iterator[VoyageResponse], bool, str]:
        pass

    @overload
    def request(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: Literal[False] = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[VoyageResponse, bool, str]:
        pass

    @overload
    def request(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: bool = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Union[VoyageResponse, Iterator[VoyageResponse]], bool, str]:
        pass

    def request(
        self,
        method,
        url,
        params=None,
        headers=None,
        files=None,
        stream: bool = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> Tuple[Union[VoyageResponse, Iterator[VoyageResponse]], bool, str]:
        result = self.request_raw(
            method.lower(),
            url,
            params=params,
            supplied_headers=headers,
            files=files,
            stream=stream,
            request_id=request_id,
            request_timeout=request_timeout,
        )
        resp, got_stream = self._interpret_response(result, stream)
        return resp, got_stream, self.api_key

    @overload
    async def arequest(
        self,
        method,
        url,
        params,
        headers,
        files,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[AsyncGenerator[VoyageResponse, None], bool, str]:
        pass

    @overload
    async def arequest(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        *,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[AsyncGenerator[VoyageResponse, None], bool, str]:
        pass

    @overload
    async def arequest(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: Literal[False] = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[VoyageResponse, bool, str]:
        pass

    @overload
    async def arequest(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: bool = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Union[VoyageResponse, AsyncGenerator[VoyageResponse, None]], bool, str]:
        pass

    async def arequest(
        self,
        method,
        url,
        params=None,
        headers=None,
        files=None,
        stream: bool = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> Tuple[Union[VoyageResponse, AsyncGenerator[VoyageResponse, None]], bool, str]:
        ctx = AioHTTPSession()
        session = await ctx.__aenter__()
        result = None
        try:
            result = await self.arequest_raw(
                method.lower(),
                url,
                session,
                params=params,
                supplied_headers=headers,
                files=files,
                request_id=request_id,
                request_timeout=request_timeout,
            )
            resp, got_stream = await self._interpret_async_response(result, stream)
        except Exception:
            # Close the request before exiting session context.
            if result is not None:
                result.release()
            await ctx.__aexit__(None, None, None)
            raise
        if got_stream:

            async def wrap_resp():
                assert isinstance(resp, AsyncGenerator)
                try:
                    async for r in resp:
                        yield r
                finally:
                    # Close the request before exiting session context. Important to do it here
                    # as if stream is not fully exhausted, we need to close the request nevertheless.
                    result.release()
                    await ctx.__aexit__(None, None, None)

            return wrap_resp(), got_stream, self.api_key
        else:
            # Close the request before exiting session context.
            result.release()
            await ctx.__aexit__(None, None, None)
            return resp, got_stream, self.api_key

    def request_headers(
        self, method: str, extra, request_id: Optional[str]
    ) -> Dict[str, str]:
        user_agent = "Voyage/v1 PythonBindings/%s" % (version.VERSION,)
        if voyageai.app_info:
            user_agent += " " + self.format_app_info(voyageai.app_info)

        uname_without_node = " ".join(
            v for k, v in platform.uname()._asdict().items() if k != "node"
        )
        ua = {
            "bindings_version": version.VERSION,
            "httplib": "requests",
            "lang": "python",
            "lang_version": platform.python_version(),
            "platform": platform.platform(),
            "publisher": "voyageai",
            "uname": uname_without_node,
        }
        if voyageai.app_info:
            ua["application"] = voyageai.app_info

        headers = {
            "X-Voyage-Client-User-Agent": json.dumps(ua),
            "User-Agent": user_agent,
        }

        headers.update(util.api_key_to_header(self.api_type, self.api_key))

        if self.organization:
            headers["Voyage-Organization"] = self.organization

        if self.api_version is not None and self.api_type == ApiType.VOYAGEAI:
            headers["Voyage-Version"] = self.api_version
        if request_id is not None:
            headers["X-Request-Id"] = request_id
        if voyageai.debug:
            headers["Voyage-Debug"] = "true"
        headers.update(extra)
        for key in list(headers.keys()):
            if key not in ["Authorization", "Content-Type"]:
                headers.pop(key)
        return headers

    def _validate_headers(
        self, supplied_headers: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if supplied_headers is None:
            return headers

        if not isinstance(supplied_headers, dict):
            raise TypeError("Headers must be a dictionary")

        for k, v in supplied_headers.items():
            if not isinstance(k, str):
                raise TypeError("Header keys must be strings")
            if not isinstance(v, str):
                raise TypeError("Header values must be strings")
            headers[k] = v

        # NOTE: It is possible to do more validation of the headers, but a request could always
        # be made to the API manually with invalid headers, so we need to handle them server side.

        return headers

    def _prepare_request_raw(
        self,
        url,
        supplied_headers,
        method,
        params,
        files,
        request_id: Optional[str],
    ) -> Tuple[str, Dict[str, str], Optional[bytes]]:
        abs_url = "%s%s" % (self.api_base, url)
        headers = self._validate_headers(supplied_headers)

        data = None
        if method == "get" or method == "delete":
            if params:
                encoded_params = urlencode(
                    [(k, v) for k, v in params.items() if v is not None]
                )
                abs_url = _build_api_url(abs_url, encoded_params)
        elif method in {"post", "put"}:
            if params and files:
                data = params
            if params and not files:
                data = json.dumps(params).encode()
                headers["Content-Type"] = "application/json"
        else:
            raise error.APIConnectionError(
                "Unrecognized HTTP method %r. This may indicate a bug in the "
                "VoyageAI bindings. Please contact us through our help center at {{TODO}} for "
                "assistance." % (method,)
            )

        headers = self.request_headers(method, headers, request_id)

        util.log_debug("Request to Voyage API", method=method, path=abs_url)
        util.log_debug("Post details", data=data, api_version=self.api_version)

        return abs_url, headers, data

    def request_raw(
        self,
        method,
        url,
        *,
        params=None,
        supplied_headers: Optional[Dict[str, str]] = None,
        files=None,
        stream: bool = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> requests.Response:
        abs_url, headers, data = self._prepare_request_raw(
            url, supplied_headers, method, params, files, request_id
        )

        if not hasattr(_thread_context, "session"):
            _thread_context.session = _make_session()
            _thread_context.session_create_time = time.time()
        elif (
            time.time() - getattr(_thread_context, "session_create_time", 0)
            >= MAX_SESSION_LIFETIME_SECS
        ):
            _thread_context.session.close()
            _thread_context.session = _make_session()
            _thread_context.session_create_time = time.time()
        try:
            result = _thread_context.session.request(
                method,
                abs_url,
                headers=headers,
                data=data,
                files=files,
                stream=stream,
                timeout=request_timeout if request_timeout else TIMEOUT_SECS,
                proxies=_thread_context.session.proxies,
            )
        except requests.exceptions.Timeout as e:
            raise error.Timeout("Request timed out: {}".format(e)) from e
        except requests.exceptions.RequestException as e:
            raise error.APIConnectionError(
                "Error communicating with VoyageAI: {}".format(e)
            ) from e
        util.log_debug(
            "Voyage API response",
            path=abs_url,
            response_code=result.status_code,
            processing_ms=result.headers.get("Voyage-Processing-Ms"),
            request_id=result.headers.get("X-Request-Id"),
        )
        # Don't read the whole stream for debug logging unless necessary.
        if voyageai.log == "debug":
            util.log_debug(
                "API response body", body=result.content, headers=result.headers
            )
        return result

    async def arequest_raw(
        self,
        method,
        url,
        session,
        *,
        params=None,
        supplied_headers: Optional[Dict[str, str]] = None,
        files=None,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> aiohttp.ClientResponse:
        abs_url, headers, data = self._prepare_request_raw(
            url, supplied_headers, method, params, files, request_id
        )

        if isinstance(request_timeout, tuple):
            timeout = aiohttp.ClientTimeout(
                connect=request_timeout[0],
                total=request_timeout[1],
            )
        else:
            timeout = aiohttp.ClientTimeout(
                total=request_timeout if request_timeout else TIMEOUT_SECS
            )

        if files:
            # TODO: Use `aiohttp.MultipartWriter` to create the multipart form data here.
            # For now we use the private `requests` method that is known to have worked so far.
            data, content_type = requests.models.RequestEncodingMixin._encode_files(  # type: ignore
                files, data
            )
            headers["Content-Type"] = content_type
        request_kwargs = {
            "method": method,
            "url": abs_url,
            "headers": headers,
            "data": data,
            "proxy": _aiohttp_proxies_arg(voyageai.proxy),
            "timeout": timeout,
        }
        try:
            result = await session.request(**request_kwargs)
            util.log_info(
                "Voyage API response",
                path=abs_url,
                response_code=result.status,
                processing_ms=result.headers.get("Voyage-Processing-Ms"),
                request_id=result.headers.get("X-Request-Id"),
            )
            # Don't read the whole stream for debug logging unless necessary.
            if voyageai.log == "debug":
                util.log_debug(
                    "API response body", body=result.content, headers=result.headers
                )
            return result
        except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
            raise error.Timeout("Request timed out") from e
        except aiohttp.ClientError as e:
            raise error.APIConnectionError("Error communicating with Voyage") from e

    def _interpret_response(
        self, result: requests.Response, stream: bool
    ) -> Tuple[Union[VoyageResponse, Iterator[VoyageResponse]], bool]:
        """Returns the response(s) and a bool indicating whether it is a stream."""
        if stream and "text/event-stream" in result.headers.get("Content-Type", ""):
            return (
                self._interpret_response_line(
                    line, result.status_code, result.headers, stream=True
                )
                for line in parse_stream(result.iter_lines())
            ), True
        else:
            return (
                self._interpret_response_line(
                    result.content.decode("utf-8"),
                    result.status_code,
                    result.headers,
                    stream=False,
                ),
                False,
            )

    async def _interpret_async_response(
        self, result: aiohttp.ClientResponse, stream: bool
    ) -> Tuple[Union[VoyageResponse, AsyncGenerator[VoyageResponse, None]], bool]:
        """Returns the response(s) and a bool indicating whether it is a stream."""
        if stream and "text/event-stream" in result.headers.get("Content-Type", ""):
            return (
                self._interpret_response_line(
                    line, result.status, result.headers, stream=True
                )
                async for line in parse_stream_async(result.content)
            ), True
        else:
            try:
                await result.read()
            except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
                raise error.Timeout("Request timed out") from e
            except aiohttp.ClientError as e:
                util.log_warn(e, body=result.content)
            return (
                self._interpret_response_line(
                    (await result.read()).decode("utf-8"),
                    result.status,
                    result.headers,
                    stream=False,
                ),
                False,
            )

    def _interpret_response_line(
        self, rbody: str, rcode: int, rheaders, stream: bool
    ) -> VoyageResponse:
        # HTTP 204 response code does not have any content in the body.
        if rcode == 204:
            return VoyageResponse(None, rheaders)

        if rcode == 500:
            raise error.ServerError(
                "The server failed to process the request.",
                rbody,
                rcode,
                headers=rheaders,
            )
        elif rcode in [502, 503, 504]:
            raise error.ServiceUnavailableError(
                "The server is overloaded or not ready yet.",
                rbody,
                rcode,
                headers=rheaders,
            )
        try:
            if 'text/plain' in rheaders.get('Content-Type', ''):
                data = rbody
            else:
                data = json.loads(rbody)
        except (JSONDecodeError, UnicodeDecodeError) as e:
            raise error.APIError(
                f"HTTP code {rcode} from API ({rbody})", rbody, rcode, headers=rheaders
            ) from e

        resp = VoyageResponse(data, rheaders)
        if 400 <= rcode < 500:
            raise self.handle_error_response(
                rbody, rcode, resp.data, rheaders
            )
        return resp
    
    def handle_error_response(self, rbody, rcode, resp, rheaders, stream_error=False):
        try:
            error_message = resp["detail"]
        except (KeyError, TypeError):
            raise error.APIError(
                "Invalid response object from API: %r (HTTP response code "
                "was %d)" % (rbody, rcode),
                rbody,
                rcode,
                resp,
            )

        util.log_info(
            "Voyage API error received", error_message=error_message,
        )

        if rcode == 400:
            return error.InvalidRequestError(
                error_message, rbody, rcode, resp, rheaders
            )
        elif rcode == 401:
            return error.AuthenticationError(
                error_message, rbody, rcode, resp, rheaders
            )
        elif rcode == 422:
            return error.MalformedRequestError(
                error_message, rbody, rcode, resp, rheaders
            )
        elif rcode == 429:
            return error.RateLimitError(
                error_message, rbody, rcode, resp, rheaders
            )
        else:
            return error.APIError(
                f"{error_message} {rbody} {rcode} {resp} {rheaders}",
                rbody,
                rcode,
                resp,
                rheaders,
            )


class AioHTTPSession(AsyncContextManager):
    def __init__(self):
        self._session = None
        self._should_close_session = False

    async def __aenter__(self):
        self._session = voyageai.aiosession.get()
        if self._session is None:
            self._session = await aiohttp.ClientSession().__aenter__()
            self._should_close_session = True

        return self._session
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._session is None:
            raise RuntimeError("Session is not initialized")

        if self._should_close_session:
            await self._session.__aexit__(exc_type, exc_value, traceback)
