from itertools import chain
from typing import Callable, List

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
    return [
        list(chain.from_iterable(chunk_fn(i) for i in input)) for input in inputs
    ]


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
