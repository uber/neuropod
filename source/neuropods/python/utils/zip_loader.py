#
# Uber, Inc. (c) 2019
#

import atexit
import os
import shutil
import tempfile
import zipfile

# Delete the created directories at process shutdown
TO_CLEANUP = []

def cleanup():
    for item in TO_CLEANUP:
        shutil.rmtree(item)

atexit.register(cleanup)

def extract_neuropod_if_necessary(path):
    if os.path.isdir(path):
        # `path` is already a directory
        return path

    # Assume it's a zipfile
    neuropod_path = tempfile.mkdtemp(suffix=".neuropod")
    z = zipfile.ZipFile(path)
    z.extractall(neuropod_path)
    z.close()

    # Make sure we delete this once we're done
    TO_CLEANUP.append(neuropod_path)

    return neuropod_path
