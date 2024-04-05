import json
from copy import deepcopy

from voyageai.api_resources.api_requestor import VoyageHttpResponse


class VoyageResponse(dict):

    def __init__(
        self,
        **params,
    ):
        super(VoyageResponse, self).__init__()

    def __setattr__(self, k, v):
        if k[0] == "_" or k in self.__dict__:
            return super(VoyageResponse, self).__setattr__(k, v)

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
            return super(VoyageResponse, self).__delattr__(k)
        else:
            del self[k]

    def __setitem__(self, k, v):
        if v == "":
            raise ValueError(
                "You cannot set %s to an empty string. "
                "We interpret empty strings as None in requests."
                "You may set %s.%s = None to delete the property" % (k, str(self), k)
            )
        super(VoyageResponse, self).__setitem__(k, v)

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
            super(VoyageResponse, self).__setitem__(k, convert_to_voyage_response(v))

        self._previous = values

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
            if isinstance(v, VoyageResponse):
                d[k] = v.to_dict_recursive()
            elif isinstance(v, list):
                d[k] = [
                    e.to_dict_recursive() if isinstance(e, VoyageResponse) else e
                    for e in v
                ]
        return d

    # This class overrides __setitem__ to throw exceptions on inputs that it
    # doesn't like. This can cause problems when we try to copy an object
    # wholesale because some data that's returned from the API may not be valid
    # if it was set to be set manually. Here we override the class' copy
    # arguments so that we can bypass these possible exceptions on __setitem__.
    def __copy__(self):
        copied = VoyageResponse()

        for k, v in self.items():
            # Call parent's __setitem__ to avoid checks that we've added in the
            # overridden version that can throw exceptions.
            super(VoyageResponse, copied).__setitem__(k, v)

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
            super(VoyageResponse, copied).__setitem__(k, deepcopy(v, memo))

        return copied


def convert_to_voyage_response(resp):
    # If we get a VoyageHttpResponse, we'll want to return a VoyageResponse.

    if isinstance(resp, VoyageHttpResponse):
        resp = resp.data

    if isinstance(resp, list):
        return [convert_to_voyage_response(i) for i in resp]
    elif isinstance(resp, dict) and not isinstance(resp, VoyageResponse):
        resp = resp.copy()
        return VoyageResponse.construct_from(resp)
    else:
        return resp


def convert_to_dict(obj):
    """Converts a VoyageResponse back to a regular dict.

    Nested VoyageResponse are also converted back to regular dicts.

    :param obj: The VoyageResponse to convert.

    :returns: The VoyageResponse as a dict.
    """
    if isinstance(obj, list):
        return [convert_to_dict(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    else:
        return obj
