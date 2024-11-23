import base64
import io
import json
from abc import ABC, abstractmethod
import functools
import warnings
from typing import Any, List, Optional, Union, Dict

from huggingface_hub import hf_hub_download
import PIL.Image

import voyageai
import voyageai.error as error
from voyageai.object.multimodal_embeddings import MultimodalInputRequest, MultimodalInputSegmentText, \
    MultimodalInputSegmentImageURL, MultimodalInputSegmentImageBase64, MultimodalEmbeddingsObject
from voyageai.util import default_api_key
from voyageai.object import EmbeddingsObject, RerankingObject


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
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 0,
        timeout: Optional[float] = None,
    ) -> None:

        self.api_key = api_key or default_api_key()

        self._params = {
            "api_key": self.api_key,
            "request_timeout": timeout,
        }

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
        inputs: Union[List[Dict], List[List[Union[str, PIL.Image.Image]]]],
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
        inputs: Union[List[Dict], List[List[Union[str, PIL.Image.Image]]]],
        model: str,
    ) -> Dict[str, int]:
        """
        This method returns estimated usage metrics for the provided input.
        Currently, only multimodal models are supported. Image URL segments are not supported.

        Args:
            inputs (list): a list of inputs
            model (str): the name of the model to be used for inference

        Returns:
            a dict, with the following keys:
            - for multimodal models:
              - "text_tokens": the number of tokens represented by the text in the items in the input
              - "image_pixels": the number of pixels represented by the images in the items in the input
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

        for item in request.inputs:
            text_segments = ""

            for segment in item.content:
                if isinstance(segment, MultimodalInputSegmentImageURL):
                    raise voyageai.error.InvalidRequestError("count_usage does not support image URL segments.")

                elif isinstance(segment, MultimodalInputSegmentImageBase64):
                    try:
                        image_str = segment.image_base64.split(",")[1]
                        image_data = base64.b64decode(image_str)
                        image = PIL.Image.open(io.BytesIO(image_data))
                        this_image_pixels = max(
                            min_pixels,
                            min(max_pixels, image.height * image.width)
                        )
                        image_pixels += this_image_pixels
                        image_tokens += this_image_pixels // pixel_to_token_ratio

                    except Exception as e:
                        raise voyageai.error.InvalidRequestError(f"Unable to process base64 image: {e}")

                elif isinstance(segment, MultimodalInputSegmentText):
                    text_segments += segment.text

            text_tokens += self.count_tokens([text_segments], model)

        return {
            "text_tokens": text_tokens,
            "image_pixels": image_pixels,
            "total_tokens": image_tokens + text_tokens,
        }