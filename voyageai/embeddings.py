import time
import warnings 
from typing import List, Optional

import voyageai


MAX_BATCH_SIZE = 8
MAX_NUM_REQUESTS_PER_SECOND = 10


def get_embedding(text: str, model="voyage-01", **kwargs) -> List[float]:
    """Get VoyageAI embedding for a text string.
    
    Args:
        text (str): A text string to be embed.
        model (str): Name of the model to use.
    """
    return voyageai.Embedding.create(input=[text], model=model, **kwargs)["data"][0]["embedding"]


def get_embeddings(
        list_of_text: List[str], 
        model="voyage-01", 
        batch_size: Optional[int] = None,
        cooldown: float = 0.1,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> List[List[float]]:
    """Get VoyageAI embedding for a list of text strings.
    
    Args:
        list_of_text (list): A list of text strings to embed.
        model (str): Name of the model to use. 
        batch_size (int): Number of texts in each API request.
        cooldown (float): Time in seconds to wait between consecutive API requests.
    """
    batch_size = batch_size or MAX_BATCH_SIZE

    if cooldown < 1 / MAX_NUM_REQUESTS_PER_SECOND:
        warnings.warn(f"Setting cooldown={cooldown} may exceed VoyageAI's rate limit.")

    if show_progress_bar:
        from tqdm.auto import tqdm
        _iter = tqdm(range(0, len(list_of_text), batch_size))
    else:
        _iter = range(0, len(list_of_text), batch_size)

    embeddings = []
    for i in _iter:
        data = voyageai.Embedding.create(input=list_of_text[i: i + batch_size], model=model, **kwargs).data
        embeddings.extend([d["embedding"] for d in data])
        time.sleep(cooldown)
    
    return embeddings
