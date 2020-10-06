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

import os
import json
import tempfile
import shutil

from neuropod.utils.packaging_utils import packager
from neuropod.utils.pip_utils import compile_requirements


@packager(platform="python")
def create_python_neuropod(
    neuropod_path,
    data_paths,
    code_path_spec,
    entrypoint_package,
    entrypoint,
    requirements=None,
    **kwargs
):
    """
    Packages arbitrary python code as a neuropod package.

    {common_doc_pre}

    :param  data_paths:         A list of dicts containing the paths to any data files that needs to be packaged.

                                !!! note ""
                                    ***Example***:
                                    ```
                                    [{
                                        path: "/path/to/myfile.txt",
                                        packaged_name: "newfilename.txt"
                                    }]
                                    ```

    :param  code_path_spec:     The folder paths of all the code that will be packaged. Note that
                                *.pyc files are ignored.

                                !!! note ""
                                    This is specified as follows:
                                    ```
                                    [{
                                        "python_root": "/some/path/to/a/python/root",
                                        "dirs_to_package": ["relative/path/to/package"]
                                    }, ...]
                                    ```

    :param  entrypoint_package: The python package containing the entrypoint (e.g. some.package.something).
                                This must contain the entrypoint function specified below.

    :param  entrypoint:         The name of a function contained in the `entrypoint_package`. This
                                function must return a callable that takes in the inputs specified in
                                `input_spec` and returns a dict containing the outputs specified in
                                `output_spec`. The `entrypoint` function will be provided the path to
                                a directory containing the packaged data as its first parameter.

                                !!! note ""
                                    For example,
                                    a function like:

                                    ```
                                    def neuropod_init(data_path):

                                        def addition_model(x, y):
                                            return {
                                                "output": x + y
                                            }

                                        return addition_model
                                    ```
                                    contained in the package 'my.awesome.addition_model' would have
                                    `entrypoint_package='my.awesome.addition_model'` and
                                    `entrypoint='neuropod_init'`

    :param  requirements:       An optional string containing the runtime requirements of this model
                                (specified in a format that pip understands)

                                !!! note ""
                                    ***Example***:
                                    ```
                                    tensorflow=1.15.0
                                    numpy=1.8
                                    ```


    {common_doc_post}
    """
    neuropod_data_path = os.path.join(neuropod_path, "0", "data")
    neuropod_code_path = os.path.join(neuropod_path, "0", "code")

    # Create a folder to store the packaged data
    os.makedirs(neuropod_data_path)

    # Copy the data to be packaged
    for data_path_spec in data_paths:
        shutil.copyfile(
            data_path_spec["path"],
            os.path.join(neuropod_data_path, data_path_spec["packaged_name"]),
        )

    # Copy the specified source code while preserving package paths
    for copy_spec in code_path_spec:
        python_root = copy_spec["python_root"]

        if os.path.realpath(neuropod_path).startswith(
            os.path.realpath(python_root) + os.sep
        ):
            raise ValueError(
                "`neuropod_path` cannot be a subdirectory of `python_root`"
            )

        for dir_to_package in copy_spec["dirs_to_package"]:
            shutil.copytree(
                os.path.join(python_root, dir_to_package),
                os.path.join(neuropod_code_path, dir_to_package),
                ignore=shutil.ignore_patterns("*.pyc"),
            )

    # Add __init__.py files as needed
    for root, subdirs, files in os.walk(neuropod_code_path):
        if "__init__.py" not in files:
            with open(os.path.join(root, "__init__.py"), "w"):
                # We just need to create the file
                pass

    # Save requirements if specified
    if requirements is not None:
        # Write requirements to a temp file
        with tempfile.NamedTemporaryFile() as requirements_txt:
            requirements_txt.write(requirements.encode("utf-8"))
            requirements_txt.flush()

            # Write the lockfile
            lock_path = os.path.join(neuropod_path, "0", "requirements.lock")
            compile_requirements(requirements_txt.name, lock_path)

    # We also need to save the entrypoint package name so we know what to load at runtime
    # This is python specific config so it's not saved in the overall neuropod config
    with open(os.path.join(neuropod_path, "0", "config.json"), "w") as config_file:
        json.dump(
            {"entrypoint_package": entrypoint_package, "entrypoint": entrypoint},
            config_file,
        )
