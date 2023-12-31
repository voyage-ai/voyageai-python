import json
from copy import deepcopy
from typing import Optional, Tuple, Union

from voyageai import util
from voyageai.api_resources import api_requestor
from voyageai.api_resources.voyage_response import VoyageResponse


class VoyageObject(dict):

    def __init__(
        self,
        **params,
    ):
        super(VoyageObject, self).__init__()
        self._retrieve_params = params

    def __setattr__(self, k, v):
        if k[0] == "_" or k in self.__dict__:
            return super(VoyageObject, self).__setattr__(k, v)

        self[k] = v
        return None

    def __getattr__(self, k):
        if k[0] == "_":
            raise AttributeError(k)
        try:
            return self[k]
        except KeyError as err:
            raise AttributeError(*err.args)

    def __delattr__(self, k):
        if k[0] == "_" or k in self.__dict__:
            return super(VoyageObject, self).__delattr__(k)
        else:
            del self[k]

    def __setitem__(self, k, v):
        if v == "":
            raise ValueError(
                "You cannot set %s to an empty string. "
                "We interpret empty strings as None in requests."
                "You may set %s.%s = None to delete the property" % (k, str(self), k)
            )
        super(VoyageObject, self).__setitem__(k, v)

    def __delitem__(self, k):
        raise NotImplementedError("del is not supported")

    # Custom unpickling method that uses `update` to update the dictionary
    # without calling __setitem__, which would fail if any value is an empty
    # string
    def __setstate__(self, state):
        self.update(state)

    # Custom pickling method to ensure the instance is pickled as a custom
    # class and not as a dict, otherwise __setstate__ would not be called when
    # unpickling.
    def __reduce__(self):
        reduce_value = (
            type(self),  # callable
            (),  # args
            dict(self),  # state
        )
        return reduce_value

    @classmethod
    def construct_from(
        cls,
        values,
    ):
        instance = cls()
        instance.refresh_from(values)
        return instance

    def refresh_from(
        self,
        values,
    ):
        # Wipe old state before setting new.
        self.clear()
        for k, v in values.items():
            super(VoyageObject, self).__setitem__(
                k, util.convert_to_voyage_object(v)
            )

        self._previous = values

    def request(
        self,
        method,
        url,
        params=None,
        headers=None,
        stream=False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ):
        if params is None:
            params = self._retrieve_params
        requestor = api_requestor.APIRequestor(
            key=self.api_key,
        )
        response, stream, api_key = requestor.request(
            method,
            url,
            params=params,
            stream=stream,
            headers=headers,
            request_id=request_id,
            request_timeout=request_timeout,
        )

        if stream:
            assert not isinstance(response, VoyageResponse)  # must be an iterator
            return (
                util.convert_to_voyage_object(line)
                for line in response
            )
        else:
            return util.convert_to_voyage_object(response)

    async def arequest(
        self,
        method,
        url,
        params=None,
        headers=None,
        stream=False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ):
        if params is None:
            params = self._retrieve_params
        requestor = api_requestor.APIRequestor(
            key=self.api_key,
        )
        response, stream, api_key = await requestor.arequest(
            method,
            url,
            params=params,
            stream=stream,
            headers=headers,
            request_id=request_id,
            request_timeout=request_timeout,
        )

        if stream:
            assert not isinstance(response, VoyageResponse)  # must be an iterator
            return (
                util.convert_to_voyage_object(line)
                for line in response
            )
        else:
            return util.convert_to_voyage_object(response)

    def __repr__(self):
        ident_parts = [type(self).__name__]

        obj = self.get("object")
        if isinstance(obj, str):
            ident_parts.append(obj)

        unicode_repr = "<%s at %s> JSON: %s" % (
            " ".join(ident_parts),
            hex(id(self)),
            str(self),
        )

        return unicode_repr

    def __str__(self):
        obj = self.to_dict_recursive()
        return json.dumps(obj, indent=2)

    def to_dict(self):
        return dict(self)

    def to_dict_recursive(self):
        d = dict(self)
        for k, v in d.items():
            if isinstance(v, VoyageObject):
                d[k] = v.to_dict_recursive()
            elif isinstance(v, list):
                d[k] = [
                    e.to_dict_recursive() if isinstance(e, VoyageObject) else e
                    for e in v
                ]
        return d

    # This class overrides __setitem__ to throw exceptions on inputs that it
    # doesn't like. This can cause problems when we try to copy an object
    # wholesale because some data that's returned from the API may not be valid
    # if it was set to be set manually. Here we override the class' copy
    # arguments so that we can bypass these possible exceptions on __setitem__.
    def __copy__(self):
        copied = VoyageObject()

        copied._retrieve_params = self._retrieve_params

        for k, v in self.items():
            # Call parent's __setitem__ to avoid checks that we've added in the
            # overridden version that can throw exceptions.
            super(VoyageObject, copied).__setitem__(k, v)

        return copied

    # This class overrides __setitem__ to throw exceptions on inputs that it
    # doesn't like. This can cause problems when we try to copy an object
    # wholesale because some data that's returned from the API may not be valid
    # if it was set to be set manually. Here we override the class' copy
    # arguments so that we can bypass these possible exceptions on __setitem__.
    def __deepcopy__(self, memo):
        copied = self.__copy__()
        memo[id(self)] = copied

        for k, v in self.items():
            # Call parent's __setitem__ to avoid checks that we've added in the
            # overridden version that can throw exceptions.
            super(VoyageObject, copied).__setitem__(k, deepcopy(v, memo))

        return copied
