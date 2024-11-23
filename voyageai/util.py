import logging
import os
import re
import sys
from typing import Optional
import voyageai

VOYAGE_LOG = os.environ.get("VOYAGE_LOG")

logger = logging.getLogger("voyage")

__all__ = [
    "log_info",
    "log_debug",
    "log_warn",
    "logfmt",
    "map_output_dtype",
]

api_key_to_header = lambda key: {"Authorization": f"Bearer {key}"}


def _console_log_level():
    if voyageai.log in ["debug", "info"]:
        return voyageai.log
    elif VOYAGE_LOG in ["debug", "info"]:
        return VOYAGE_LOG
    else:
        return None


def log_debug(message, **params):
    msg = logfmt(dict(message=message, **params))
    if _console_log_level() == "debug":
        print(msg, file=sys.stderr)
    logger.debug(msg)


def log_info(message, **params):
    msg = logfmt(dict(message=message, **params))
    if _console_log_level() in ["debug", "info"]:
        print(msg, file=sys.stderr)
    logger.info(msg)


def log_warn(message, **params):
    msg = logfmt(dict(message=message, **params))
    print(msg, file=sys.stderr)
    logger.warn(msg)


def logfmt(props):
    def fmt(key, val):
        # Handle case where val is a bytes or bytesarray
        if hasattr(val, "decode"):
            val = val.decode("utf-8")
        # Check if val is already a string to avoid re-encoding into ascii.
        if not isinstance(val, str):
            val = str(val)
        if re.search(r"\s", val):
            val = repr(val)
        # key should already be a string
        if re.search(r"\s", key):
            key = repr(key)
        return "{key}={val}".format(key=key, val=val)

    return " ".join([fmt(key, val) for key, val in sorted(props.items())])


def default_api_key() -> str:
    api_key_path = voyageai.api_key_path or os.environ.get("VOYAGE_API_KEY_PATH")
    api_key = voyageai.api_key or os.environ.get("VOYAGE_API_KEY")

    # When api_key_path is specified, it overwrites api_key
    if api_key_path:
        with open(api_key_path, "rt") as k:
            api_key = k.read().strip()
            return api_key
    elif api_key is not None:
        return api_key
    else:
        raise voyageai.error.AuthenticationError(
            "No API key provided. You can set your API key in code using 'voyageai.api_key = <API-KEY>', "
            "or set the environment variable VOYAGE_API_KEY=<API-KEY>). If your API key is stored "
            "in a file, you can point the voyageai module at it with 'voyageai.api_key_path = <PATH>', "
            "or set the environment variable VOYAGE_API_KEY_PATH=<PATH>. "
            "API keys can be generated in Voyage AI's dashboard (https://dash.voyageai.com)."
        )


def map_output_dtype(dtype: Optional[str] = None) -> str:
    if not dtype or dtype == "float":
        return "float32"
    if dtype == "int8" or dtype == "binary":
        return "int8"
    if dtype == "uint8" or dtype == "ubinary":
        return "uint8"
    raise ValueError(f"Unknown dtype {dtype}")