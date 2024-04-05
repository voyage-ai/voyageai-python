from voyageai.api_resources import api_requestor
from voyageai.api_resources.response import VoyageResponse, convert_to_voyage_response


class APIResource(VoyageResponse):

    @classmethod
    def class_url(cls):
        if cls == APIResource:
            raise NotImplementedError(
                "APIResource is an abstract class. You should perform actions on its subclasses."
            )
        # Namespaces are separated in object names with periods (.) and in URLs
        # with forward slashes (/), so replace the former with the latter.
        base = cls.OBJECT_NAME.replace(".", "/")  # type: ignore
        return "/%s" % (base)

    @classmethod
    def __prepare_create_request(
        cls,
        api_key=None,
        api_base=None,
        **params,
    ):
        requestor = api_requestor.APIRequestor(
            api_key,
            api_base=api_base,
        )
        url = cls.class_url()
        headers = params.pop("headers", None)

        return requestor, url, params, headers

    @classmethod
    def create(
        cls,
        api_key=None,
        api_base=None,
        request_id=None,
        request_timeout=None,
        **params,
    ):
        requestor, url, params, headers = cls.__prepare_create_request(
            api_key, api_base, **params
        )

        response = requestor.request(
            "post",
            url,
            params=params,
            headers=headers,
            request_id=request_id,
            request_timeout=request_timeout,
        )

        obj = convert_to_voyage_response(response)
        return obj

    @classmethod
    async def acreate(
        cls,
        api_key=None,
        api_base=None,
        request_id=None,
        request_timeout=None,
        **params,
    ):
        requestor, url, params, headers = cls.__prepare_create_request(
            api_key, api_base, **params
        )
        response = await requestor.arequest(
            "post",
            url,
            params=params,
            headers=headers,
            request_id=request_id,
            request_timeout=request_timeout,
        )

        obj = convert_to_voyage_response(response)
        return obj
