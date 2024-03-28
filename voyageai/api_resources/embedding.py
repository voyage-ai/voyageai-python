import base64
import numpy as np

from voyageai.api_resources import APIResource


class Embedding(APIResource):

    OBJECT_NAME = "embeddings"

    @classmethod
    def create(cls, *args, **kwargs):
        """
        Creates a new embedding for the provided input and parameters.
        """
        user_provided_encoding_format = kwargs.get("encoding_format", None)

        # If encoding format was not explicitly specified, we opaquely use base64 for performance
        if not user_provided_encoding_format:
            kwargs["encoding_format"] = "base64"

        response = super().create(*args, **kwargs)

        # If a user specifies base64, we'll just return the encoded string.
        # This is only for the default case.
        if not user_provided_encoding_format:
            for data in response.data:
                # If an engine isn't using this optimization, don't do anything
                if type(data["embedding"]) == str:
                    data["embedding"] = np.frombuffer(
                        base64.b64decode(data["embedding"]), dtype="float32"
                    ).tolist()

        return response

    @classmethod
    async def acreate(cls, *args, **kwargs):
        """
        Creates a new embedding for the provided input and parameters.
        """
        user_provided_encoding_format = kwargs.get("encoding_format", None)

        # If encoding format was not explicitly specified, we opaquely use base64 for performance
        if not user_provided_encoding_format:
            kwargs["encoding_format"] = "base64"

        response = await super().acreate(*args, **kwargs)

        # If a user specifies base64, we'll just return the encoded string.
        # This is only for the default case.
        if not user_provided_encoding_format:
            for data in response.data:
                # If an engine isn't using this optimization, don't do anything
                if type(data["embedding"]) == str:
                    data["embedding"] = np.frombuffer(
                        base64.b64decode(data["embedding"]), dtype="float32"
                    ).tolist()

        return response
