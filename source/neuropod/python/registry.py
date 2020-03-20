#
# Uber, Inc. (c) 2020
#

# A backend registry

_REGISTERED_BACKENDS = {}


def register_backend(platforms, so_path):
    for platform in platforms:
        _REGISTERED_BACKENDS[platform] = so_path
