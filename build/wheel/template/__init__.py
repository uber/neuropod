#
# Uber, Inc. (c) 2020
#

import os.path as osp
NATIVE_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "{SHARED_LIBRARY_NAME}")

from neuropod.registry import register_backend
register_backend({PLATFORM}, NATIVE_PATH)
