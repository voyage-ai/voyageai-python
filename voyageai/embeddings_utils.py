import asyncio
from aiolimiter import AsyncLimiter
import warnings
from typing import List, Optional
from contextlib import AsyncExitStack
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from concurrent.futures import ThreadPoolExecutor

import voyageai


MAX_BATCH_SIZE = voyageai.VOYAGE_EMBED_BATCH_SIZE
MAX_LIST_LENGTH = voyageai.VOYAGE_EMBED_BATCH_SIZE
DEFAULT_CONCURRENCE = 1
DEFAULT_RPM = 300


@retry(
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(voyageai.error.RateLimitError),
)
def _get_embeddings(
    list_of_text: List[str],
    model: str = voyageai.VOYAGE_EMBED_DEFAULT_MODEL,
    input_type: Optional[str] = None,
    **kwargs,
) -> List[List[float]]:
    """Python wrapper for one Voyage API call."""
    _check_input_type(input_type)
    assert (
        len(list_of_text) <= MAX_BATCH_SIZE
    ), f"The length of list_of_text should not be larger than {MAX_BATCH_SIZE}."

    data = voyageai.Embedding.create(
        input=list_of_text, model=model, input_type=input_type, **kwargs
    ).data
    return [d["embedding"] for d in data]


@retry(
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(voyageai.error.RateLimitError),
)
async def _aget_embeddings(
    list_of_text: List[str],
    model: str = voyageai.VOYAGE_EMBED_DEFAULT_MODEL,
    input_type: Optional[str] = None,
    **kwargs,
) -> List[List[float]]:
    """Python wrapper for one async Voyage API call."""
    _check_input_type(input_type)
    assert (
        len(list_of_text) <= MAX_BATCH_SIZE
    ), f"The length of list_of_text should not be larger than {MAX_BATCH_SIZE}."

    semaphore = kwargs.pop("semaphore", AsyncExitStack())
    rate_limit = kwargs.pop("rate_limit", AsyncExitStack())

    async with semaphore:
        async with rate_limit:
            data = (
                await voyageai.Embedding.acreate(
                    input=list_of_text, model=model, **kwargs
                )
            ).data

    return [d["embedding"] for d in data]


def _check_input_type(input_type: Optional[str]):
    if input_type and input_type not in ["query", "document"]:
        raise ValueError(
            f"input_type {input_type} is invalid. Options: None, 'query', 'document'."
        )


def get_embedding(
    text: str,
    model: str = voyageai.VOYAGE_EMBED_DEFAULT_MODEL,
    input_type: Optional[str] = None,
    **kwargs,
) -> List[float]:
    """Get Voyage embedding for a text string.

    Args:
        text (str): A text string to be embed.
        model (str): Name of the model to use.
        input_type (str): Type of the input text. Defalut to None, meaning the type is unspecified.
            Other options include: "query", "document".
    """
    warnings.warn(
        "This function will be deprecated. We recommend using the `embed()` function in `voyageai.Client`."
    )
    return _get_embeddings([text], model, input_type, **kwargs)[0]


def get_embeddings(
    list_of_text: List[str],
    model: str = voyageai.VOYAGE_EMBED_DEFAULT_MODEL,
    input_type: Optional[str] = None,
    **kwargs,
) -> List[List[float]]:
    """Get Voyage embedding for a list of text strings.

    Args:
        list_of_text (list): A list of text strings to embed.
        model (str): Name of the model to use.
        input_type (str): Type of the input text. Defalut to None, meaning the type is unspecified.
            Other options include: "query", "document".
    """
    warnings.warn(
        "This function will be deprecated. We recommend using the `embed()` function in `voyageai.Client`."
    )
    if len(list_of_text) <= MAX_BATCH_SIZE:
        return _get_embeddings(list_of_text, model, input_type, **kwargs)

    assert (
        len(list_of_text) <= MAX_LIST_LENGTH
    ), f"The length of list_of_text should not be larger than {MAX_LIST_LENGTH}."

    batches = [
        list_of_text[i : i + MAX_BATCH_SIZE]
        for i in range(0, len(list_of_text), MAX_BATCH_SIZE)
    ]

    with ThreadPoolExecutor(max_workers=DEFAULT_CONCURRENCE) as executor:
        futures = [
            executor.submit(_get_embeddings, batch, model, input_type)
            for batch in batches
        ]

    results = [future.result() for future in futures]
    return sum(results, [])


async def aget_embedding(
    text: str,
    model: str = voyageai.VOYAGE_EMBED_DEFAULT_MODEL,
    input_type: Optional[str] = None,
    **kwargs,
) -> List[float]:
    """Get Voyage embedding for a text string (async).

    Args:
        text (str): A text string to be embed.
        model (str): Name of the model to use.
        input_type (str): Type of the input text. Defalut to None, meaning the type is unspecified.
            Other options include: "query", "document".
    """
    warnings.warn(
        "This function will be deprecated. We recommend using the `embed()` function in `voyageai.AsyncClient`."
    )
    return (await _aget_embeddings([text], model, input_type, **kwargs))[0]


async def aget_embeddings(
    list_of_text: List[str],
    model: str = voyageai.VOYAGE_EMBED_DEFAULT_MODEL,
    input_type: Optional[str] = None,
    **kwargs,
) -> List[List[float]]:
    """Get Voyage embedding for a list of text strings (async).

    Args:
        list_of_text (list): A list of text strings to embed.
        model (str): Name of the model to use.
        input_type (str): Type of the input text. Defalut to None, meaning the type is unspecified.
            Other options include: "query", "document".
    """
    warnings.warn(
        "This function will be deprecated. We recommend using the `embed()` function in `voyageai.AsyncClient`."
    )
    if len(list_of_text) <= MAX_BATCH_SIZE:
        return await _aget_embeddings(list_of_text, model, input_type, **kwargs)

    assert (
        len(list_of_text) <= MAX_LIST_LENGTH
    ), f"The length of list_of_text should not be larger than {MAX_LIST_LENGTH}."

    semaphore = asyncio.Semaphore(value=DEFAULT_CONCURRENCE)
    rate_limit = AsyncLimiter(DEFAULT_RPM, 60)

    batches = [
        list_of_text[i : i + MAX_BATCH_SIZE]
        for i in range(0, len(list_of_text), MAX_BATCH_SIZE)
    ]
    async_tasks = [
        _aget_embeddings(
            batch, model, input_type, semaphore=semaphore, rate_limit=rate_limit
        )
        for batch in batches
    ]
    results = await asyncio.gather(*async_tasks)
    return sum(results, [])
