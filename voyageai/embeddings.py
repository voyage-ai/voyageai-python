import time
import warnings 
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential

import voyageai


MAX_BATCH_SIZE = 8


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(
    text: str, 
    model: str ="voyage-01", 
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
    _check_input_type(input_type)
    return voyageai.Embedding.create(
        input=[text], model=model, input_type=input_type, **kwargs
    )["data"][0]["embedding"]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings(
    list_of_text: List[str], 
    model: str ="voyage-01", 
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
    _check_input_type(input_type)
    assert len(list_of_text) <= MAX_BATCH_SIZE, \
        f"The length of list_of_text should not be larger than {MAX_BATCH_SIZE}."

    data = voyageai.Embedding.create(
        input=list_of_text, model=model, input_type=input_type, **kwargs
    ).data
    return [d["embedding"] for d in data]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def aget_embedding(text: str, 
    model: str ="voyage-01", 
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
    _check_input_type(input_type)
    return (await voyageai.Embedding.acreate(
        input=[text], model=model, input_type=input_type, **kwargs
    ))["data"][0]["embedding"]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def aget_embeddings(
    list_of_text: List[str], model: str ="voyage-01", 
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
    _check_input_type(input_type)
    assert len(list_of_text) <= MAX_BATCH_SIZE, \
        f"The length of list_of_text should not be larger than {MAX_BATCH_SIZE}."

    data = (await voyageai.Embedding.acreate(
        input=list_of_text, model=model, **kwargs)
    ).data
    return [d["embedding"] for d in data]


def _check_input_type(input_type: str):
    if input_type and input_type not in ["query", "document"]:
        raise ValueError(f"input_type {input_type} is invalid. Options: None, 'query', 'document'.")
