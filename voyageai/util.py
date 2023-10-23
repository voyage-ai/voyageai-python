import logging
import os
import re
import sys
from enum import Enum
from typing import Optional

import voyageai

VOYAGEAI_LOG = os.environ.get("VOYAGEAI_LOG")

logger = logging.getLogger("voyageai")

__all__ = [
    "log_info",
    "log_debug",
    "log_warn",
    "logfmt",
]

api_key_to_header = (
    lambda api, key: {"Authorization": f"Bearer {key}"}
)


class ApiType(Enum):
    VOYAGEAI = 1

    @staticmethod
    def from_str(label):
        if label.lower() in ("voyageai", "voyage_ai"):
            return ApiType.VOYAGEAI
        else:
            raise voyageai.error.InvalidAPIType(
                "The API type provided in invalid. Please select one of the supported API types: 'voyageai'"
            )


def _console_log_level():
    if voyageai.log in ["debug", "info"]:
        return voyageai.log
    elif VOYAGEAI_LOG in ["debug", "info"]:
        return VOYAGEAI_LOG
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


def convert_to_voyageai_object(
    resp,
    api_key=None,
    api_version=None,
    organization=None,
    engine=None,
    plain_old_data=False,
):
    # If we get a VoyageAIResponse, we'll want to return a VoyageAIObject.

    response_ms: Optional[int] = None
    if isinstance(resp, voyageai.voyageai_response.VoyageAIResponse):
        organization = resp.organization
        response_ms = resp.response_ms
        resp = resp.data

    if plain_old_data:
        return resp
    elif isinstance(resp, list):
        return [
            convert_to_voyageai_object(
                i, api_key, api_version, organization, engine=engine
            )
            for i in resp
        ]
    elif isinstance(resp, dict) and not isinstance(
        resp, voyageai.voyageai_object.VoyageAIObject
    ):
        resp = resp.copy()
        klass = voyageai.voyageai_object.VoyageAIObject

        return klass.construct_from(
            resp,
            api_key=api_key,
            api_version=api_version,
            organization=organization,
            response_ms=response_ms,
            engine=engine,
        )
    else:
        return resp


def convert_to_dict(obj):
    """Converts a VoyageAIObject back to a regular dict.

    Nested VoyageAIObjects are also converted back to regular dicts.

    :param obj: The VoyageAIObject to convert.

    :returns: The VoyageAIObject as a dict.
    """
    if isinstance(obj, list):
        return [convert_to_dict(i) for i in obj]
    # This works by virtue of the fact that VoyageAIObjects _are_ dicts. The dict
    # comprehension returns a regular dict and recursively applies the
    # conversion to each value.
    elif isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def default_api_key() -> str:
    if voyageai.api_key_path:
        with open(voyageai.api_key_path, "rt") as k:
            api_key = k.read().strip()
            if not api_key.startswith("sk-"):
                raise ValueError(f"Malformed API key in {voyageai.api_key_path}.")
            return api_key
    elif voyageai.api_key is not None:
        return voyageai.api_key
    else:
        raise voyageai.error.AuthenticationError(
            "No API key provided. You can set your API key in code using 'voyageai.api_key = <API-KEY>', "
            "or you can set the environment variable VOYAGEAI_API_KEY=<API-KEY>). If your API key is stored "
            "in a file, you can point the voyageai module at it with 'voyageai.api_key_path = <PATH>'. "
            "You can generate API keys in the VoyageAI web interface. See https://{{TODO}} for details."
        )
