#
# Uber, Inc. (c) 2019
#
import sys

class PackagerLoader(object):
    """
    Create aliases for packagers so they can be used as

        `from neuropods.packagers import create_*_neuropod`

    instead of

        `from neuropods.backends.*.packager import create_*_neuropod`

    In order to make sure we don't load all the backends unnecessarily, we need to lazy-load
    them when the packaging function is called
    """
    # This is checked by the import system
    # See https://docs.python.org/3/reference/import.html#__path__
    __path__ = []

    def __getattr__(self, packaging_function):
        packager_name = packaging_function.replace("create_", "").replace("_neuropod", "")
        packager_module = "neuropods.backends." + packager_name + ".packager"

        if packager_name not in ["keras", "python", "pytorch", "tensorflow", "torchscript"]:
            raise RuntimeError("Tried to get an invalid attribute on neuropods.packagers ({})".format(packaging_function))

        module = __import__(packager_module, fromlist=[packaging_function])

        return getattr(module, packaging_function)


sys.modules[__name__] = PackagerLoader()
