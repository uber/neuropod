# Copyright (c) 2022 The Neuropod Authors
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

import os

from neuropod.loader import load_neuropod
from neuropod.packagers import create_python_neuropod
from testpath.tempdir import TemporaryDirectory

_MODEL_SOURCE = """
def model():
    return {}

def get_model(_):
    return model
"""


def preinstall_deps(backend_version, requirements):
    """
    Preinstall python dependencies into the isolated python environments used to
    run Neuropod models.

    This can be used to reduce load times for models using large dependencies.

    :param  backend_version: The version of the python Neuropod backend to install the deps for (e.g. `3.6`)
    :param  requirements:    The deps to preinstall. See the docs for `create_python_neuropod` for details.
    """

    with TemporaryDirectory() as tmp_dir:
        neuropod_path = os.path.join(tmp_dir, "temp_neuropod")
        model_code_dir = os.path.join(tmp_dir, "model_code")
        os.makedirs(model_code_dir)

        with open(os.path.join(model_code_dir, "model.py"), "w") as f:
            f.write(_MODEL_SOURCE)

        # Creates and loads a python "model" that just serves to preload depsq
        create_python_neuropod(
            neuropod_path=neuropod_path,
            model_name="temp_preload_model",
            data_paths=[],
            code_path_spec=[
                {
                    "python_root": model_code_dir,
                    "dirs_to_package": [""],  # Package everything in the python_root
                }
            ],
            entrypoint_package="model",
            entrypoint="get_model",
            input_spec=[],
            output_spec=[],
            platform_version_semver=backend_version,
            requirements=requirements,
        )

        # Load the model to trigger installing the deps
        load_neuropod(neuropod_path)
