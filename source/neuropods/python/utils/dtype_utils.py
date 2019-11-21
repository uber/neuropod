import six
import numpy as np


def get_dtype(arg):
    """
    Get numpy dtypes from strings in a python 2 and 3 compatible way
    """
    if arg == "string":
        arg = "str"

    return np.dtype(arg)


def get_dtype_name(arg):
    name = get_dtype(arg).name
    if name == "str":
        return "string"

    return name


def maybe_convert_bindings_types(items):
    if six.PY3:
        # Python 3 uses unicode for strings. We need to convert before passing to
        # the native bindings
        for key, value in items.items():
            if value.dtype.type == np.str_:
                items[key] = value.astype(np.string_)

    return items
