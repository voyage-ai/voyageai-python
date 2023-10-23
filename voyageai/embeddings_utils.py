import textwrap as tr
from typing import List, Optional

import plotly.express as px
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tenacity import retry, stop_after_attempt, wait_random_exponential

import voyageai
from voyageai.datalib.numpy_helper import numpy as np
from voyageai.datalib.pandas_helper import pandas as pd


EMBEDDING_MAX_BATCH_SIZE = 4


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="voyage-01", **kwargs) -> List[float]:
    return voyageai.Embedding.create(input=[text], model=model, **kwargs)["data"][0]["embedding"]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def aget_embedding(text: str, model="voyage-01", **kwargs) -> List[float]:
    return (await voyageai.Embedding.acreate(input=[text], model=model, **kwargs))["data"][0][
        "embedding"
    ]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings(list_of_text: List[str], model="voyage-01", **kwargs) -> List[List[float]]:
    assert len(list_of_text) <= EMBEDDING_MAX_BATCH_SIZE, \
        f"The batch size should not be larger than {EMBEDDING_MAX_BATCH_SIZE}."

    data = voyageai.Embedding.create(input=list_of_text, model=model, **kwargs).data
    return [d["embedding"] for d in data]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def aget_embeddings(list_of_text: List[str], engine="voyage-api-v0", **kwargs) -> List[List[float]]:
    assert len(list_of_text) <= EMBEDDING_MAX_BATCH_SIZE, \
        f"The batch size should not be larger than {EMBEDDING_MAX_BATCH_SIZE}."

    data = (await voyageai.Embedding.acreate(input=list_of_text, engine=engine, **kwargs)).data
    return [d["embedding"] for d in data]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
    """Return a list of indices of nearest neighbors from a list of distances."""
    return np.argsort(distances)


def pca_components_from_embeddings(
    embeddings: List[List[float]], n_components=2
) -> np.ndarray:
    """Return the PCA components of a list of embeddings."""
    pca = PCA(n_components=n_components)
    array_of_embeddings = np.array(embeddings)
    return pca.fit_transform(array_of_embeddings)


def tsne_components_from_embeddings(
    embeddings: List[List[float]], n_components=2, **kwargs
) -> np.ndarray:
    """Returns t-SNE components of a list of embeddings."""
    # use better defaults if not specified
    if "init" not in kwargs.keys():
        kwargs["init"] = "pca"
    if "learning_rate" not in kwargs.keys():
        kwargs["learning_rate"] = "auto"
    tsne = TSNE(n_components=n_components, **kwargs)
    array_of_embeddings = np.array(embeddings)
    return tsne.fit_transform(array_of_embeddings)


def chart_from_components(
    components: np.ndarray,
    labels: Optional[List[str]] = None,
    strings: Optional[List[str]] = None,
    x_title="Component 0",
    y_title="Component 1",
    mark_size=5,
    **kwargs,
):
    """Return an interactive 2D chart of embedding components."""
    empty_list = ["" for _ in components]
    data = pd.DataFrame(
        {
            x_title: components[:, 0],
            y_title: components[:, 1],
            "label": labels if labels else empty_list,
            "string": ["<br>".join(tr.wrap(string, width=30)) for string in strings]
            if strings
            else empty_list,
        }
    )
    chart = px.scatter(
        data,
        x=x_title,
        y=y_title,
        color="label" if labels else None,
        symbol="label" if labels else None,
        hover_data=["string"] if strings else None,
        **kwargs,
    ).update_traces(marker=dict(size=mark_size))
    return chart


def chart_from_components_3D(
    components: np.ndarray,
    labels: Optional[List[str]] = None,
    strings: Optional[List[str]] = None,
    x_title: str = "Component 0",
    y_title: str = "Component 1",
    z_title: str = "Compontent 2",
    mark_size: int = 5,
    **kwargs,
):
    """Return an interactive 3D chart of embedding components."""
    empty_list = ["" for _ in components]
    data = pd.DataFrame(
        {
            x_title: components[:, 0],
            y_title: components[:, 1],
            z_title: components[:, 2],
            "label": labels if labels else empty_list,
            "string": ["<br>".join(tr.wrap(string, width=30)) for string in strings]
            if strings
            else empty_list,
        }
    )
    chart = px.scatter_3d(
        data,
        x=x_title,
        y=y_title,
        z=z_title,
        color="label" if labels else None,
        symbol="label" if labels else None,
        hover_data=["string"] if strings else None,
        **kwargs,
    ).update_traces(marker=dict(size=mark_size))
    return chart
