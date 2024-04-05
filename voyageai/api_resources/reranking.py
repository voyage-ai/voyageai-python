from voyageai.api_resources import APIResource


class Reranking(APIResource):

    OBJECT_NAME = "rerank"

    @classmethod
    def create(cls, *args, **kwargs):
        """
        Creates a new reranking for the provided input and parameters.
        """
        response = super().create(*args, **kwargs)
        return response

    @classmethod
    async def acreate(cls, *args, **kwargs):
        """
        Creates a new reranking for the provided input and parameters.
        """
        response = await super().acreate(*args, **kwargs)
        return response
