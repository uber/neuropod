#
# Uber, Inc. (c) 2020
#

from neuropod.neuropod_native import BackendLoadSpec

# A list of backends that are available for the native code to load
_REGISTERED_BACKENDS = []


def register_backend(platform, version, so_path):
    _REGISTERED_BACKENDS.append(BackendLoadSpec(platform, version, so_path))
