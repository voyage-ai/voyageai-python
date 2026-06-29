import base64
import functools
import io
import json
import platform
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
from huggingface_hub import hf_hub_download

import voyageai
from voyageai.object import EmbeddingsObject, RerankingObject
from voyageai.object.contextualized_embeddings import ContextualizedEmbeddingsObject
from voyageai.object.multimodal_embeddings import (
    MultimodalEmbeddingsObject,
    MultimodalInputRequest,
    MultimodalInputSegmentImageBase64,
    MultimodalInputSegmentImageURL,
    MultimodalInputSegmentText,
    MultimodalInputSegmentVideoBase64,
    MultimodalInputSegmentVideoURL,
)
from voyageai.util import default_api_key, get_default_base_url
from voyageai.video_utils import Video


def _build_metadata_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {
        "X-VoyageAI-Lang": "python",
        "X-VoyageAI-Package": "voyageai",
        "X-VoyageAI-Telemetry-Version": "1",
    }
    try:
        from voyageai.version import VERSION
    except Exception:
        VERSION = None

    if VERSION:
        headers["X-VoyageAI-Package-Version"] = VERSION
    try:
        headers["X-VoyageAI-Runtime"] = platform.python_implementation()
        headers["X-VoyageAI-Runtime-Version"] = platform.python_version()
    except Exception:
        pass
    try:
        headers["X-VoyageAI-OS"] = platform.system()
    except Exception:
        pass
    try:
        ua_parts = [
            f"voyageai-python/{VERSION}" if VERSION else "voyageai-python",
            f"Python/{platform.python_version()}",
            f"{platform.system()}/{platform.machine()}",
        ]
        headers["User-Agent"] = " ".join(ua_parts)
    except Exception:
        pass
    return headers


def _get_client_config(
    model: str,
) -> dict:
    try:
        config_path = hf_hub_download(repo_id=f"voyageai/{model}", filename="client_config.json")
        with open(config_path, "r") as f:
            data_dict = json.load(f)
    except:
        warnings.warn(
            f"Failed to load the client config for `{model}`. Please ensure that it is a valid model name."
            f"Currently, only multimodal models are supported."
        )
        raise

    return data_dict


class _BaseClient(ABC):
    """Voyage AI Client

    Args:
        api_key (str): Your API key.
        max_retries (int): Maximum number of retries if API call fails.
        timeout (float): Timeout in seconds.
        base_url (str): Base URL for the API endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 0,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or default_api_key()
        self.max_retries = max_retries
        base_url = base_url or get_default_base_url(self.api_key)

        self._metadata_headers = _build_metadata_headers()
        self._params = {
            "api_key": self.api_key,
            "request_timeout": timeout,
            "base_url": base_url,
            "headers": self._metadata_headers,
        }

    @staticmethod
    def _validate_wrapper_field(field: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"Wrapper {field} must be a string, got {type(value).__name__}.")
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"Wrapper {field} must be a non-empty string.")
        # Reject control characters (header-injection risk) and the "|"/"/"
        # delimiters used to encode the wrapper list.
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in stripped) or set("|/") & set(stripped):
            raise ValueError(
                f"Wrapper {field} {value!r} contains invalid characters "
                "(control characters or the reserved '|' / '/' delimiters)."
            )
        return stripped

    def append_client_metadata(self, name: str, version: str) -> None:
        """Record an integration wrapper (e.g. a framework that embeds this
        client) in the X-VoyageAI-Wrapper header."""
        name = self._validate_wrapper_field("name", name)
        version = self._validate_wrapper_field("version", version)
        wrapper_value = f"{name}/{version}"
        current = self._metadata_headers.get("X-VoyageAI-Wrapper", "")
        existing = [w for w in current.split("|") if w] if current else []
        if wrapper_value not in existing:
            existing.append(wrapper_value)
            self._metadata_headers["X-VoyageAI-Wrapper"] = "|".join(existing)

    @abstractmethod
    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: Optional[str] = None,
        truncation: bool = True,
        output_dtype: Optional[str] = None,
        output_dimension: Optional[int] = None,
    ) -> EmbeddingsObject:
        pass

    @abstractmethod
    def contextualized_embed(
        self,
        inputs: Union[List[List[str]], List[str]],
        model: str,
        input_type: Optional[str] = None,
        output_dtype: Optional[str] = None,
        output_dimension: Optional[int] = None,
        chunk_fn: Optional[Callable[[str], List[str]]] = None,
        enable_auto_chunking: bool = False,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> ContextualizedEmbeddingsObject:
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        model: str,
        top_k: Optional[int] = None,
        truncation: bool = True,
    ) -> RerankingObject:
        pass

    @abstractmethod
    def multimodal_embed(
        self,
        inputs: Union[List[Dict], List[List[Union[str, PIL.Image.Image, Video]]]],
        model: str,
        input_type: Optional[str] = None,
        truncation: bool = True,
    ) -> MultimodalEmbeddingsObject:
        pass

    @functools.lru_cache()
    def tokenizer(self, model: str):
        try:
            from tokenizers import Tokenizer  # type: ignore
        except ImportError:
            raise ImportError(
                "The package `tokenizers` is not found. Please run `pip install tokenizers` "
                "to install the dependency."
            )

        try:
            tokenizer = Tokenizer.from_pretrained(f"voyageai/{model}")
            tokenizer.no_truncation()
        except:
            warnings.warn(
                f"Failed to load the tokenizer for `{model}`. Please ensure that it is a valid model name."
            )
            raise

        return tokenizer

    def tokenize(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[Any]:
        if model is None:
            warnings.warn(
                "Please specify the `model` when using the tokenizer. Voyage's older models use the same "
                "tokenizer, but new models may use different tokenizers. If `model` is not specified, "
                "the old tokenizer will be used and the results might be different. `model` will be a "
                "required argument in the future."
            )
            model = voyageai.VOYAGE_EMBED_DEFAULT_MODEL

        return self.tokenizer(model).encode_batch(texts)

    def count_tokens(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> int:
        tokenized = self.tokenize(texts, model)
        return sum([len(t) for t in tokenized])

    def count_usage(
        self,
        inputs: Union[List[Dict], List[List[Union[str, PIL.Image.Image, Video]]]],
        model: str,
    ) -> Dict[str, int]:
        """
        This method returns estimated usage metrics for the provided input.
        Currently, only multimodal models are supported. Image and video URL segments are not supported.

        Args:
            inputs (list): a list of inputs
            model (str): the name of the model to be used for inference

        Returns:
            a dict, with the following keys:
            - for multimodal models:
              - "text_tokens": the number of tokens represented by the text in the items in the input
              - "image_pixels": the number of pixels represented by the images in the items in the input
              - "video_pixels": the number of pixels represented by the videos in the items in the input
              - "total_tokens": the total number of tokens represented by the items in the input
        """
        client_config = _get_client_config(model)

        min_pixels = client_config["multimodal_image_pixels_min"]
        max_pixels = client_config["multimodal_image_pixels_max"]
        pixel_to_token_ratio = client_config["multimodal_image_to_tokens_ratio"]

        request = MultimodalInputRequest.from_user_inputs(
            inputs=inputs,
            model=model,
        )

        image_tokens, image_pixels, text_tokens = 0, 0, 0
        video_tokens, video_pixels = 0, 0

        for item in request.inputs:
            text_segments = ""

            for segment in item.content:
                if isinstance(segment, MultimodalInputSegmentImageURL):
                    raise voyageai.error.InvalidRequestError(
                        "count_usage does not support image URL segments."
                    )

                elif isinstance(segment, MultimodalInputSegmentVideoURL):
                    raise voyageai.error.InvalidRequestError(
                        "count_usage does not support video URL segments."
                    )

                elif isinstance(segment, MultimodalInputSegmentImageBase64):
                    try:
                        image_str = segment.image_base64.split(",")[1]
                        image_data = base64.b64decode(image_str)
                        image = PIL.Image.open(io.BytesIO(image_data))
                        this_image_pixels = max(
                            min_pixels, min(max_pixels, image.height * image.width)
                        )
                        image_pixels += this_image_pixels
                        image_tokens += this_image_pixels // pixel_to_token_ratio

                    except Exception as e:
                        raise voyageai.error.InvalidRequestError(
                            f"Unable to process base64 image: {e}"
                        )

                elif isinstance(segment, MultimodalInputSegmentVideoBase64):
                    try:
                        video_str = segment.video_base64.split(",")[1]
                        video_data = base64.b64decode(video_str)
                        video = Video.from_file(io.BytesIO(video_data), model=model, optimize=False)
                        video_pixels += video.num_pixels
                        video_tokens += video.estimated_num_tokens

                    except Exception as e:
                        raise voyageai.error.InvalidRequestError(
                            f"Unable to process base64 video: {e}"
                        )

                elif isinstance(segment, MultimodalInputSegmentText):
                    text_segments += segment.text

            text_tokens += self.count_tokens([text_segments], model)

        return {
            "text_tokens": text_tokens,
            "image_pixels": image_pixels,
            "video_pixels": video_pixels,
            "total_tokens": image_tokens + video_tokens + text_tokens,
        }
