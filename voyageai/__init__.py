# Voyage AI Python bindings.
#
# Originally forked from the MIT-licensed OpenAI Python bindings.

import os
import sys
from typing import TYPE_CHECKING, Optional, Union, Callable

from contextvars import ContextVar

if "pkg_resources" not in sys.modules:
    # workaround for the following:
    # https://github.com/benoitc/gunicorn/pull/2539
    sys.modules["pkg_resources"] = object()  # type: ignore[assignment]
    import aiohttp

    del sys.modules["pkg_resources"]

VOYAGE_EMBED_BATCH_SIZE = 128
VOYAGE_EMBED_DEFAULT_MODEL = "voyage-02"

from voyageai.api_resources import Embedding
from voyageai.api_resources import voyage_object, voyage_response
from voyageai.version import VERSION
from voyageai.client import Client
from voyageai.client_async import AsyncClient
from voyageai.embeddings_utils import get_embedding, get_embeddings, aget_embedding, aget_embeddings

if TYPE_CHECKING:
    import requests
    from aiohttp import ClientSession

api_key = os.environ.get("VOYAGE_API_KEY")
# Path of a file with an API key, whose contents can change. Supercedes
# `api_key` if set.  The main use case is volume-mounted Kubernetes secrets,
# which are updated automatically.
api_key_path: Optional[str] = os.environ.get("VOYAGE_API_KEY_PATH")

organization = os.environ.get("VOYAGE_ORGANIZATION")
api_base = os.environ.get("VOYAGE_API_BASE", "https://api.voyageai.com/v1")
api_type = os.environ.get("VOYAGE_API_TYPE", "voyage")
api_version = os.environ.get("VOYAGE_API_VERSION", None)
verify_ssl_certs = True  # No effect. Certificates are always verified.
proxy = None
app_info = None
enable_telemetry = False  # Ignored; the telemetry feature was removed.
ca_bundle_path = None  # No longer used, feature was removed
debug = False
log = None  # Set to either 'debug' or 'info', controls console logging

requestssession: Optional[
    Union["requests.Session", Callable[[], "requests.Session"]]
] = None # Provide a requests.Session or Session factory.

aiosession: ContextVar[Optional["ClientSession"]] = ContextVar(
    "aiohttp-session", default=None
)  # Acts as a global aiohttp ClientSession that reuses connections.
# This is user-supplied; otherwise, a session is remade for each request.

__version__ = VERSION
