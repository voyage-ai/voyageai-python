from itertools import chain
from typing import Callable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

def apply_chunking(
    inputs: List[List[str]],
    chunk_fn: Callable[[str], List[str]],
) -> List[List[str]]:
    return [list(chain.from_iterable(chunk_fn(i) for i in input)) for input in inputs]


def default_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Callable[[str], List[str]]:
    """ 
    Simple wrapper for LangChain RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    def split(text: str) -> List[str]:
        chunks = splitter.create_chunks([text])
        return [chunk.page_content for chunk in chunks]
    return split
