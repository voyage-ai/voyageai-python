from voyageai.api_resources import APIResource
from voyageai.util import decode_base64_embedding


class ContextualizedEmbedding(APIResource):
    OBJECT_NAME = "contextualizedembeddings"

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
            for chunked_data in response.data:
                for chunk_embedding in chunked_data.data:
                # If an engine isn't using this optimization, don't do anything
                    if type(chunk_embedding["embedding"]) == str:
                        chunk_embedding["embedding"] = decode_base64_embedding(
                            chunk_embedding["embedding"], kwargs.get("output_dtype", None)
                        )

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
            for chunked_data in response.data:
                for chunk_embedding in chunked_data.data:
                # If an engine isn't using this optimization, don't do anything
                    if type(chunk_embedding["embedding"]) == str:
                        chunk_embedding["embedding"] = decode_base64_embedding(
                            chunk_embedding["embedding"], kwargs.get("output_dtype", None)
                        )

        return response
