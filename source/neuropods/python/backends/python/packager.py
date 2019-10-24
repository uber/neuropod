#
# Uber, Inc. (c) 2018
#

import os
import json
import shutil

from neuropods.utils.packaging_utils import packager


@packager(platform="python")
def create_python_neuropod(
        neuropod_path,
        data_paths,
        code_path_spec,
        entrypoint_package,
        entrypoint,
        skip_virtualenv=False,
        **kwargs):
    """
    Packages arbitrary python code as a neuropod package.

    {common_doc_pre}

    :param  data_paths:         A list of dicts containing the paths to any data files that needs to be packaged.
                                Ex: [{
                                    path: "/path/to/myfile.txt",
                                    packaged_name: "newfilename.txt"
                                }]

    :param  code_path_spec:     The folder paths of all the code that will be packaged. Note that
                                *.pyc files are ignored. This is specified as follows:
        [{
            "python_root": "/some/path/to/a/python/root",
            "dirs_to_package": ["relative/path/to/package"]
        }, ...]

    :param  entrypoint_package: The python package containing the entrypoint (e.g. some.package.something).
                                This must contain the entrypoint function specified below.

    :param  entrypoint:         The name of a function contained in the `entrypoint_package`. This
                                function must return a callable that takes in the inputs specified in
                                `input_spec` and returns a dict containing the outputs specified in
                                `output_spec`. The `entrypoint` function will be provided the path to
                                a directory containing the packaged data as its first parameter. For example,
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

    {common_doc_post}
    """
    neuropod_data_path = os.path.join(neuropod_path, "0", "data")
    neuropod_code_path = os.path.join(neuropod_path, "0", "code")

    # Create a folder to store the packaged data
    os.makedirs(neuropod_data_path)

    # Copy the data to be packaged
    for data_path_spec in data_paths:
        shutil.copyfile(data_path_spec["path"], os.path.join(neuropod_data_path, data_path_spec["packaged_name"]))

    # Copy the specified source code while preserving package paths
    for copy_spec in code_path_spec:
        python_root = copy_spec["python_root"]

        if os.path.realpath(neuropod_path).startswith(os.path.realpath(python_root) + os.sep):
            raise ValueError("`neuropod_path` cannot be a subdirectory of `python_root`")

        for dir_to_package in copy_spec["dirs_to_package"]:
            shutil.copytree(
                os.path.join(python_root, dir_to_package),
                os.path.join(neuropod_code_path, dir_to_package),
                ignore=shutil.ignore_patterns('*.pyc'),
            )

    # Add __init__.py files as needed
    for root, subdirs, files in os.walk(neuropod_code_path):
        if "__init__.py" not in files:
            with open(os.path.join(root, "__init__.py"), "w"):
                # We just need to create the file
                pass

    # We also need to save the entrypoint package name so we know what to load at runtime
    # This is python specific config so it's not saved in the overall neuropod config
    with open(os.path.join(neuropod_path, "0", "config.json"), "w") as config_file:
        json.dump({
            "entrypoint_package": entrypoint_package,
            "entrypoint": entrypoint
        }, config_file)
