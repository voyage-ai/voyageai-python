import base64
import PIL.Image
import PIL.ImageFile
from io import BytesIO
from enum import Enum
from typing import List, Optional, Union, Dict, Literal, Annotated, Any

from voyageai import error
from voyageai.api_resources import VoyageResponse

try:
    from pydantic.v1 import BaseModel, Field, Extra, ValidationError
except ImportError:
    from pydantic import BaseModel, Field, Extra, ValidationError

class MultimodalEmbeddingsObject:
    def __init__(self, response: Optional[VoyageResponse] = None):
        self.embeddings: List[List[float]] = []
        self.text_tokens: int = 0
        self.image_pixels: int = 0
        self.total_tokens: int = 0
        if response:
            self.update(response)

    def update(self, response: VoyageResponse):
        for d in response.data:
            self.embeddings.append(d.embedding)
        self.text_tokens += response.usage.text_tokens
        self.image_pixels += response.usage.image_pixels
        self.total_tokens += response.usage.total_tokens


class MultimodalInputSegmentType(str, Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_BASE64 = "image_base64"

    def __str__(self):
        return self.value


class MultimodalInputSegmentText(BaseModel):
    type: Literal[MultimodalInputSegmentType.TEXT] = MultimodalInputSegmentType.TEXT
    text: str

    class Config:
        extra = Extra.forbid


class MultimodalInputSegmentImageURL(BaseModel):
    type: Literal[
        MultimodalInputSegmentType.IMAGE_URL
    ] = MultimodalInputSegmentType.IMAGE_URL
    image_url: str

    class Config:
        extra = Extra.forbid


class MultimodalInputSegmentImageBase64(BaseModel):
    type: Literal[
        MultimodalInputSegmentType.IMAGE_BASE64
    ] = MultimodalInputSegmentType.IMAGE_BASE64
    image_base64: str

    class Config:
        extra = Extra.forbid


class MultimodalInput(BaseModel):
    content: List[
        Annotated[
            Union[
                MultimodalInputSegmentText,
                MultimodalInputSegmentImageURL,
                MultimodalInputSegmentImageBase64,
            ],
            Field(discriminator="type"),
        ]
    ] = Field(..., min_items=1)


class MultimodalInputRequest(BaseModel):
    inputs: List[MultimodalInput] = Field(default_factory=list)
    model: str
    input_type: Optional[str] = None
    truncation: bool = True

    @classmethod
    def from_user_inputs(
        cls,
        inputs: Union[List[Dict], List[List[Union[str, PIL.Image.Image]]]],
        model: str,
        input_type: Optional[str] = None,
        truncation: bool = True,
    ) -> "MultimodalInputRequest":
        """
        Create a MultimodalInputRequest from user inputs.

        :param inputs: Either a list of dictionaries (each with 'content') or a list of lists containing strings and/or PIL images.
        :param model: The model identifier.
        :param input_type: Optional input type.
        :param truncation: Whether to apply truncation.
        :return: An instance of MultimodalInputRequest.
        :raises error.InvalidRequestError: If input processing fails.
        """
        multimodal_inputs = []

        if not inputs or not isinstance(inputs, list):
            raise error.InvalidRequestError("'inputs' must be a non-empty list")

        first_input = inputs[0]
        if isinstance(first_input, dict):
            input_kind = "dict"
        elif isinstance(first_input, list):
            input_kind = "list"
        else:
            raise error.InvalidRequestError(
                f"Invalid input type: {type(first_input).__name__}. Must be dict or list."
            )

        # Ensure all inputs are of the same kind
        for idx, input_data in enumerate(inputs):
            if (input_kind == "dict" and not isinstance(input_data, dict)) or (
                input_kind == "list" and not isinstance(input_data, list)
            ):
                raise error.InvalidRequestError(
                    f"All inputs must be of type '{input_kind}'. Mismatch found at index {idx}."
                )

        # Process inputs based on their kind
        for idx, input_data in enumerate(inputs):
            try:
                if input_kind == "dict":
                    multimodal_input = cls._process_dict_input(input_data, idx)
                elif input_kind == "list":
                    multimodal_input = cls._process_list_input(input_data, idx)
                else:
                    # This should not happen due to earlier checks
                    raise error.InvalidRequestError(
                        f"Unsupported input kind at index {idx}."
                    )
                multimodal_inputs.append(multimodal_input)
            except (ValidationError, ValueError) as e:
                raise error.InvalidRequestError(
                    f"Error processing input at index {idx}: {e}"
                ) from e

        try:
            return cls(
                inputs=multimodal_inputs,
                model=model,
                input_type=input_type,
                truncation=truncation,
            )
        except ValidationError as e:
            raise error.InvalidRequestError(f"Invalid request structure: {e}") from e

    @classmethod
    def _process_dict_input(cls, input_data: Dict, idx: int) -> MultimodalInput:
        """
        Process a dictionary input and convert it to a MultimodalInput instance.

        :param input_data: The input dictionary.
        :param idx: The index of the input in the list.
        :return: A MultimodalInput instance.
        :raises ValueError: If 'content' key is missing or invalid.
        """
        if "content" not in input_data:
            raise ValueError(f"Input at index {idx} is missing the 'content' field.")

        try:
            return MultimodalInput.parse_obj(input_data)
        except ValidationError as ve:
            raise ValueError(f"Validation error for input at index {idx}: {ve}") from ve

    @classmethod
    def _process_list_input(
        cls, input_list: List[Union[str, PIL.Image.Image]], idx: int
    ) -> MultimodalInput:
        """
        Process a list input and convert it to a MultimodalInput instance.

        :param input_list: The input list containing strings or PIL images.
        :param idx: The index of the input in the list.
        :return: A MultimodalInput instance.
        :raises ValueError: If list items are not strings or PIL images.
        """
        if not input_list:
            raise ValueError(f"Input list at index {idx} is empty.")

        if not all(isinstance(item, (str, PIL.Image.Image)) for item in input_list):
            raise ValueError(
                f"All items in the list at index {idx} must be strings or PIL images."
            )

        segments = []
        for item_idx, item in enumerate(input_list):
            try:
                segment = cls._create_segment(item, item_idx, idx)
                segments.append(segment)
            except ValueError as ve:
                raise ValueError(
                    f"Error processing item {item_idx} in input list at index {idx}: {ve}"
                ) from ve

        return MultimodalInput(content=segments)

    @staticmethod
    def _create_segment(
        item: Union[str, PIL.Image.Image], item_idx: int, input_idx: int
    ) -> Union[MultimodalInputSegmentImageBase64, MultimodalInputSegmentText]:
        """
        Create a segment based on the type of the item.

        :param item: The item to create a segment from.
        :param item_idx: The index of the item in the list.
        :param input_idx: The index of the input in the main inputs list.
        :return: A MultimodalInputSegment instance.
        :raises ValueError: If the item type is unsupported.
        """
        if isinstance(item, str):
            return MultimodalInputSegmentText(text=item)
        elif isinstance(item, PIL.Image.Image):
            image_base64 = MultimodalInputRequest._image_to_base64(item, conversion_kwargs={"lossless": True})
            return MultimodalInputSegmentImageBase64(image_base64=image_base64)
        else:
            raise ValueError(
                f"Unsupported item type at input {input_idx}, item {item_idx}: {type(item).__name__}"
            )

    @staticmethod
    def _image_to_base64(
        image: PIL.Image.Image,
        target_format: str = "WEBP",
        target_mime_type: str = "image/webp",
        conversion_kwargs: Dict[str, Any] = {},
    ) -> str:
        """
        Convert a PIL Image to a Base64-encoded data URI.

        :param image: The PIL Image to convert.
        :param target_format: The format to save the image in.
        :param target_mime_type: The MIME type of the image.
        :return: A Base64-encoded data URI string.
        """
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format=target_format, **conversion_kwargs)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:{target_mime_type};base64,{img_base64}"
