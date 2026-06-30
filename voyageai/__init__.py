# Voyage AI Python bindings.
#
# Originally forked from the MIT-licensed OpenAI Python bindings.

# ruff: noqa

import sys
from contextvars import ContextVar
from typing import TYPE_CHECKING, Callable, Optional, Union

if "pkg_resources" not in sys.modules:
    # workaround for the following:
    # https://github.com/benoitc/gunicorn/pull/2539
    sys.modules["pkg_resources"] = object()  # type: ignore[assignment]
    import aiohttp

    del sys.modules["pkg_resources"]

VOYAGE_EMBED_BATCH_SIZE = 128
VOYAGE_EMBED_DEFAULT_MODEL = "voyage-2"

from voyageai.api_resources import (
    ContextualizedEmbedding,
    Embedding,
    MultimodalEmbedding,
    Reranking,
)
from voyageai.chunking import default_chunk_fn
from voyageai.client import Client
from voyageai.client_async import AsyncClient
from voyageai.embeddings_utils import (
    aget_embedding,
    aget_embeddings,
    get_embedding,
    get_embeddings,
)
from voyageai.version import VERSION


def __getattr__(name: str):
    # `HAS_LOCAL` is resolved lazily so that `import voyageai` never imports
    # torch / sentence-transformers; module-level __getattr__ already provides
    # exactly this deferral on first attribute access.
    if name == "HAS_LOCAL":
        from voyageai.local import _is_local_available

        return _is_local_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    import requests
    from aiohttp import ClientSession

api_key: Optional[str] = None
api_key_path: Optional[str] = None

verify_ssl_certs = True  # No effect. Certificates are always verified.
proxy = None
app_info = None
debug = False
log = None  # Set to either 'debug' or 'info', controls console logging

requestssession: Optional[Union["requests.Session", Callable[[], "requests.Session"]]] = (
    None  # Provide a requests.Session or Session factory.
)

aiosession: ContextVar[Optional["ClientSession"]] = ContextVar(
    "aiohttp-session", default=None
)  # Acts as a global aiohttp ClientSession that reuses connections.
# This is user-supplied; otherwise, a session is remade for each request.

__version__ = VERSION
