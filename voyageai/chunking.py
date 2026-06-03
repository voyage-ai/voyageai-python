from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 2048
SEPARATORS = [
    "\n\n",
    "\n",
    "\uff0e",  # Fullwidth full stop
    "\u3002",  # Ideographic full stop
    "\uff0c",  # Fullwidth comma
    "\u3001",  # Ideographic comma
    ".",
    ",",
    " ",
    "\u200b",  # Zero-width space
    "",
]


def apply_chunking(
    inputs: List[List[str]],
    chunk_fn: Callable[[str], List[str]],
) -> List[List[str]]:
    """
    Apply chunk_fn to each string in a nested list of inputs and flatten the results
    """
    return [list(chain.from_iterable(chunk_fn(i) for i in input)) for input in inputs]


def validate_and_normalize_contextualized_inputs(
    inputs: Union[List[List[str]], List[str]],
    input_type: Optional[str],
    chunk_fn: Optional[Callable[[str], List[str]]],
    enable_auto_chunking: bool,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
) -> Tuple[List[List[str]], Dict[str, Any]]:
    """Validate contextualized-embedding params, normalize flat inputs, and
    build the auto-chunking kwargs to forward to the API."""
    if chunk_fn is not None and enable_auto_chunking:
        raise ValueError("chunk_fn cannot be combined with enable_auto_chunking=True")

    if not enable_auto_chunking and (chunk_size is not None or chunk_overlap is not None):
        raise ValueError("chunk_size and chunk_overlap require enable_auto_chunking=True")

    if chunk_size is not None and chunk_overlap is not None and chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be greater than or equal to 1")
    if chunk_overlap is not None and chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap is not None and chunk_size is None:
        raise ValueError("chunk_overlap requires chunk_size")

    if not inputs:
        raise ValueError("inputs must not be empty")

    if isinstance(inputs, list) and all(isinstance(s, str) for s in inputs):
        if input_type != "query" and not enable_auto_chunking:
            raise ValueError(
                "List[str] inputs requires enable_auto_chunking=True or input_type='query'"
            )
        inputs = [[s] for s in inputs]

    if enable_auto_chunking:
        if input_type != "document":
            raise ValueError("enable_auto_chunking=True requires input_type='document'")
        for i, doc in enumerate(inputs):
            if len(doc) != 1:
                raise ValueError(
                    f"inputs[{i}] has {len(doc)} chunks; auto-chunking expects one string per document"
                )

    extra_kwargs: Dict[str, Any] = {}
    if enable_auto_chunking:
        extra_kwargs["enable_auto_chunking"] = True
    if chunk_size is not None:
        extra_kwargs["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        extra_kwargs["chunk_overlap"] = chunk_overlap

    return inputs, extra_kwargs


def default_chunk_fn(chunk_size: int = DEFAULT_CHUNK_SIZE) -> Callable[[str], List[str]]:
    """
    Simple wrapper for LangChain RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        keep_separator="end",
        strip_whitespace=False,
        chunk_size=chunk_size,
        chunk_overlap=0,
    )

    def split(text: str) -> List[str]:
        chunks = splitter.create_documents([text])
        return [chunk.page_content for chunk in chunks]

    return split
