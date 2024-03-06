import base64
import time
import numpy as np

from voyageai import util
from voyageai.api_resources import APIResource
from voyageai.error import TryAgain


class Reranking(APIResource):

    OBJECT_NAME = "rerank"

    @classmethod
    def create(cls, *args, **kwargs):
        """
        Creates a new reranking for the provided input and parameters.
        """
        start = time.time()
        timeout = kwargs.pop("timeout", None)

        while True:
            try:
                response = super().create(*args, **kwargs)
                return response
            except TryAgain as e:
                if timeout is not None and time.time() > start + timeout:
                    raise

                util.log_info("Waiting for model to warm up", error=e)

    @classmethod
    async def acreate(cls, *args, **kwargs):
        """
        Creates a new reranking for the provided input and parameters.
        """
        start = time.time()
        timeout = kwargs.pop("timeout", None)

        while True:
            try:
                response = await super().acreate(*args, **kwargs)
                return response
            except TryAgain as e:
                if timeout is not None and time.time() > start + timeout:
                    raise

                util.log_info("Waiting for model to warm up", error=e)
