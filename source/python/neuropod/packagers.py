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
import sys


class PackagerLoader(object):
    """
    Create aliases for packagers so they can be used as

        `from neuropod.packagers import create_*_neuropod`

    instead of

        `from neuropod.backends.*.packager import create_*_neuropod`

    In order to make sure we don't load all the backends unnecessarily, we need to lazy-load
    them when the packaging function is called
    """

    # This is checked by the import system
    # See https://docs.python.org/3/reference/import.html#__path__
    __path__ = []

    def __getattr__(self, packaging_function):
        packager_name = packaging_function.replace("create_", "").replace(
            "_neuropod", ""
        )
        packager_module = "neuropod.backends." + packager_name + ".packager"

        if packager_name not in [
            "keras",
            "python",
            "pytorch",
            "tensorflow",
            "torchscript",
        ]:
            raise RuntimeError(
                "Tried to get an invalid attribute on neuropod.packagers ({})".format(
                    packaging_function
                )
            )

        module = __import__(packager_module, fromlist=[packaging_function])

        return getattr(module, packaging_function)


sys.modules[__name__] = PackagerLoader()
