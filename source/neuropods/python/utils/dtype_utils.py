import six
import numpy as np

# Get numpy dtypes from strings in a python 2 and 3 compatible way
def get_dtype(arg):
    if arg == "string":
        arg = "str"

    return np.dtype(arg)

def get_dtype_name(arg):
    name = get_dtype(arg).name
    if name == "str":
        return "string"

    return name

# Python 3 represents strings as unicode. The native code doesn't suppport unicode
# arrays
def maybe_convert_bindings_types(items):
    if six.PY3:
        # Python 3 uses unicode for strings. We need to convert before passing to
        # the native bindings
        for key, value in items.items():
            if value.dtype.type == np.str_:
                items[key] = value.astype(np.string_)

    return items
