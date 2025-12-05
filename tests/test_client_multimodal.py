import asyncio
import math
from inspect import iscoroutinefunction
from typing import List

import pytest
import voyageai
import voyageai.error as error
from PIL import Image


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        raise ValueError("Input lists must not be empty.")
    if len(a) != len(b):
        raise ValueError("Input lists must have the same length.")

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x**2 for x in a))
    magnitude_b = math.sqrt(sum(y**2 for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Input vectors must not be zero vectors.")
    similarity = dot_product / (magnitude_a * magnitude_b)
    print(f"Cosine similarity is {similarity}.")
    return similarity


@pytest.fixture(scope="function")
def client_kwargs():
    return {}


@pytest.fixture(scope="function")
def client(request, client_type: str, client_kwargs):
    client_kwargs = client_kwargs if client_kwargs else {}
    if client_type == "sync":
        return voyageai.Client(**client_kwargs)
    elif client_type == "async":
        return voyageai.AsyncClient(**client_kwargs)
    else:
        raise ValueError(f"Unknown client_type: {client_type}")


def embed_with_client(client, **kwargs):
    if iscoroutinefunction(client.multimodal_embed):

        async def embed_async(**async_kwargs):
            return await client.multimodal_embed(**async_kwargs)

        return asyncio.run(embed_async(**kwargs))
    else:
        return client.multimodal_embed(**kwargs)


@pytest.fixture(scope="function")
def multimodal_model():
    return "voyage-multimodal-3"


@pytest.fixture(scope="function")
def similarity_threshold():
    return 0.9995


@pytest.fixture(scope="function")
def embedding_dimension():
    return 1024


sample_input_dict_text_01 = {
    "content": [
        {"type": "text", "text": "this is an image of a blue sailboat on a lake."},
    ],
}

sample_input_dict_url_01 = {
    "content": [
        {
            "type": "image_url",
            "image_url": "https://www.voyageai.com/feature.png",
        },
    ],
}

sample_input_dict_b64_01 = {
    "content": [
        {
            "type": "image_base64",
            "image_base64": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAIAAADTED8xAAADMElEQVR4nOzVwQnAIBQFQYXff81RUkQCOyDj1YOPnbXWPmeTRef+/3O/OyBjzh3CD95BfqICMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMO0TAAD//2Anhf4QtqobAAAAAElFTkSuQmCC",
        },
    ],
}

sample_input_dict_mixed_01 = {
    "content": [
        {"type": "text", "text": "this is an image of a blue sailboat on a lake."},
        {
            "type": "image_url",
            "image_url": "https://www.voyageai.com/feature.png",
        },
    ],
}

sample_input_list_text_01 = ["this is an image of a blue sailboat on a lake."]

sample_input_list_img_01 = [Image.open("tests/example_image_01.jpg")]

sample_input_list_img_02 = [Image.open("tests/example_image_01.jpg").resize((256, 256))]

sample_input_list_img_03 = [Image.new("L", (400, 400), color=128)]

sample_input_list_img_04 = [Image.new("L", (10, 10), color=128)]

sample_input_list_img_05 = [Image.new("L", (4000, 4000), color=128)]

sample_input_list_mixed_01 = [
    "this is an image of a blue sailboat on a lake.",
    Image.new("L", (400, 400), color=128),
]

sample_input_invalid_text_01 = {
    "content": [
        {"type": "text", "wrong_key": "this is an image of a blue sailboat on a lake."},
    ],
}


@pytest.mark.parametrize("client_type", ["sync", "async"])
class TestClient:
    @pytest.mark.parametrize(
        "inputs",
        [
            ([sample_input_dict_text_01]),
            ([sample_input_dict_url_01]),
            ([sample_input_dict_b64_01]),
            ([sample_input_list_text_01]),
            ([sample_input_list_img_01]),
            ([sample_input_list_img_02]),
            ([sample_input_dict_text_01, sample_input_dict_url_01]),
            ([sample_input_dict_text_01, sample_input_dict_text_01]),
            ([sample_input_dict_b64_01, sample_input_dict_text_01]),
            ([sample_input_dict_b64_01, sample_input_dict_b64_01]),
            ([sample_input_list_text_01, sample_input_list_img_01]),
            ([sample_input_list_img_02, sample_input_list_text_01]),
            ([sample_input_list_text_01, sample_input_list_text_01]),
        ],
    )
    def test_client_multimodal_embed_valid_inputs(
        self, client, inputs, multimodal_model, embedding_dimension
    ):
        result = embed_with_client(client, inputs=inputs, model=multimodal_model)
        assert len(result.embeddings) == len(inputs)
        assert len(result.embeddings[0]) == embedding_dimension
        assert result.text_tokens > 0 or result.image_pixels > 0
        assert result.total_tokens > 0

    @pytest.mark.parametrize(
        "inputs, expected_exception",
        [
            ([sample_input_invalid_text_01], voyageai.error.InvalidRequestError),
            ([], voyageai.error.InvalidRequestError),
            ([{"another_key": ""}], voyageai.error.InvalidRequestError),
            ([12, sample_input_dict_text_01], voyageai.error.InvalidRequestError),
            ([sample_input_dict_text_01, 12], voyageai.error.InvalidRequestError),
            ([sample_input_invalid_text_01, []], voyageai.error.InvalidRequestError),
            (
                [sample_input_dict_text_01, sample_input_dict_b64_01] * 501,
                voyageai.error.InvalidRequestError,
            ),  # exceeds max batch size
            (
                [sample_input_list_text_01, sample_input_list_img_04] * 501,
                voyageai.error.InvalidRequestError,
            ),  # exceeds max batch size
            (
                [sample_input_dict_mixed_01, sample_input_list_mixed_01],
                voyageai.error.InvalidRequestError,
            ),  # no mixing dict and list
            ({}, voyageai.error.InvalidRequestError),  # inputs is not a list
        ],
    )
    def test_client_multimodal_embed_raises_exception(
        self, client, inputs, expected_exception, multimodal_model
    ):
        with pytest.raises(expected_exception):
            embed_with_client(client, inputs=inputs, model=multimodal_model)

    @pytest.mark.parametrize(
        "inputs",
        [
            ([sample_input_dict_text_01]),
            ([sample_input_dict_url_01]),
            ([sample_input_dict_b64_01]),
            ([sample_input_dict_text_01, sample_input_dict_b64_01]),
            ([sample_input_dict_text_01, sample_input_dict_url_01]),
            ([sample_input_list_img_02, sample_input_list_text_01]),
            ([sample_input_list_text_01, sample_input_list_text_01]),
        ],
    )
    def test_client_embed_input_type(
        self,
        client,
        inputs,
        multimodal_model,
        embedding_dimension,
        similarity_threshold,
    ):
        query_embd = embed_with_client(
            client, inputs=inputs, model=multimodal_model, input_type="query"
        )
        doc_embd = embed_with_client(
            client, inputs=inputs, model=multimodal_model, input_type="document"
        )

        assert len(query_embd.embeddings[0]) == embedding_dimension
        assert len(doc_embd.embeddings[0]) == embedding_dimension
        assert (
            cosine_similarity(query_embd.embeddings[0], doc_embd.embeddings[0])
            < similarity_threshold
        )
        assert query_embd.text_tokens == doc_embd.text_tokens
        assert query_embd.image_pixels == doc_embd.image_pixels
        assert query_embd.total_tokens == doc_embd.total_tokens

    def test_client_multimodal_embed_exceeds_context_length(
        self, client, multimodal_model, embedding_dimension
    ):
        long_input = [
            "this is an image of a blue sailboat on a lake." * 1500,
            Image.new("L", (400, 400), color=128),
        ] * 3

        inputs = [long_input]
        with pytest.raises(voyageai.error.InvalidRequestError):
            embed_with_client(client, inputs=inputs, model=multimodal_model, truncation=False)

        truncated_result = embed_with_client(
            client, inputs=inputs, model=multimodal_model, truncation=True
        )
        assert len(truncated_result.embeddings) == 1
        assert len(truncated_result.embeddings[0]) == embedding_dimension
        assert truncated_result.total_tokens <= 32000

    def test_client_embed_invalid_request(self, client, multimodal_model):
        with pytest.raises(error.InvalidRequestError):
            embed_with_client(client, inputs=[sample_input_list_mixed_01], model="wrong-model-name")

        with pytest.raises(error.InvalidRequestError):
            embed_with_client(
                client,
                inputs=[sample_input_list_mixed_01],
                model=multimodal_model,
                input_type="doc",
            )

        with pytest.raises(error.InvalidRequestError):
            embed_with_client(
                client,
                inputs=[sample_input_list_mixed_01],
                model=multimodal_model,
                truncation="test",
            )

        with pytest.raises(error.InvalidRequestError):
            embed_with_client(client, inputs=[], model=multimodal_model, truncation="test")

    def test_input_formats_yield_identical_result(
        self, client, multimodal_model, similarity_threshold
    ):
        input_1 = {
            "content": [
                {
                    "type": "image_url",
                    "image_url": "https://github.com/voyage-ai/voyageai-python/blob/4333a2eee7c4558cf3d9ad5ac2576a98c94c363a/tests/example_image_01.jpg?raw=true",
                },
            ],
        }
        input_2 = sample_input_list_img_01
        output_1 = embed_with_client(client, inputs=[input_1], model=multimodal_model)
        output_2 = embed_with_client(client, inputs=[input_2], model=multimodal_model)
        assert len(output_1.embeddings[0]) == len(output_2.embeddings[0])
        assert (
            cosine_similarity(output_1.embeddings[0], output_2.embeddings[0])
            >= similarity_threshold
        )

    @pytest.mark.parametrize(
        "client_kwargs",
        [{"timeout": 1, "max_retries": 1}],
    )
    def test_client_embed_timeout(self, client, client_kwargs, multimodal_model):
        with pytest.raises(error.Timeout):
            embed_with_client(
                client, inputs=[sample_input_dict_mixed_01] * 16, model=multimodal_model
            )

    @pytest.mark.parametrize(
        "inputs, expected_count",
        [
            ([sample_input_dict_text_01], (13, 0, 13)),
            ([sample_input_dict_b64_01], (0, 65536, 117)),
            ([sample_input_dict_text_01, sample_input_dict_b64_01], (13, 65536, 130)),
            ([sample_input_list_img_02, sample_input_list_text_01], (13, 65536, 130)),
            ([sample_input_list_text_01, sample_input_list_text_01], (26, 0, 26)),
            ([sample_input_list_text_01 * 2] * 2, (50, 0, 50)),
            ([sample_input_list_mixed_01, sample_input_list_mixed_01], (26, 320000, 596)),
            (
                [sample_input_list_img_04, sample_input_list_img_05, sample_input_list_text_01],
                (13, 2050000, 3673),
            ),
            ([], -1),
            ([sample_input_list_text_01, []], -1),
            ([sample_input_dict_url_01], -1),
            ([sample_input_dict_mixed_01], -1),
            ([sample_input_dict_text_01, sample_input_dict_url_01], -1),
        ],
    )
    def test_client_count_usage(self, client, inputs, expected_count, multimodal_model):
        if isinstance(expected_count, tuple):
            estimated_usage = client.count_usage(inputs=inputs, model=multimodal_model)
            assert estimated_usage["text_tokens"] == expected_count[0]
            assert estimated_usage["image_pixels"] == expected_count[1]
            assert estimated_usage["total_tokens"] == expected_count[2]

        else:
            with pytest.raises(voyageai.error.InvalidRequestError):
                client.count_usage(inputs=inputs, model=multimodal_model)
