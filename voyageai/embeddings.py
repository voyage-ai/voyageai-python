import time
import warnings 
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential

import voyageai


MAX_BATCH_SIZE = 8


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="voyage-01", **kwargs) -> List[float]:
    """Get Voyage embedding for a text string.
    
    Args:
        text (str): A text string to be embed.
        model (str): Name of the model to use.
    """
    return voyageai.Embedding.create(input=[text], model=model, **kwargs)["data"][0]["embedding"]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings(list_of_text: List[str], model="voyage-01", **kwargs) -> List[List[float]]:
    """Get Voyage embedding for a list of text strings.
    
    Args:
        list_of_text (list): A list of text strings to embed.
        model (str): Name of the model to use. 
    """
    assert len(list_of_text) <= MAX_BATCH_SIZE, \
        f"The length of list_of_text should not be larger than {MAX_BATCH_SIZE}."

    data = voyageai.Embedding.create(input=list_of_text, model=model, **kwargs).data
    return [d["embedding"] for d in data]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def aget_embedding(text: str, model="voyage-01", **kwargs) -> List[float]:
    """Get Voyage embedding for a text string (async).
    
    Args:
        text (str): A text string to be embed.
        model (str): Name of the model to use.
    """
    return (await voyageai.Embedding.acreate(input=[text], model=model, **kwargs))["data"][0][
        "embedding"
    ]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def aget_embeddings(list_of_text: List[str], model="voyage-01", **kwargs) -> List[List[float]]:
    """Get Voyage embedding for a list of text strings (async).
    
    Args:
        list_of_text (list): A list of text strings to embed.
        model (str): Name of the model to use. 
    """
    assert len(list_of_text) <= MAX_BATCH_SIZE, \
        f"The length of list_of_text should not be larger than {MAX_BATCH_SIZE}."

    data = (await voyageai.Embedding.acreate(input=list_of_text, model=model, **kwargs)).data
    return [d["embedding"] for d in data]
