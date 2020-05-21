#
# Uber, Inc. (c) 2020
#

# A list of backends that are available for the native code to load
_REGISTERED_BACKENDS = []


def register_backend(platform, version, so_path):
    from neuropod.neuropod_native import BackendLoadSpec

    _REGISTERED_BACKENDS.append(BackendLoadSpec(platform, version, so_path))
