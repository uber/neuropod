# Copyright (c) 2020 UATC, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
