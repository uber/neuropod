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
