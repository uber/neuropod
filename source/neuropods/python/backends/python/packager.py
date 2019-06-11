#
# Uber, Inc. (c) 2018
#

import os
import json
import shutil

from neuropods.backends import config_utils
from neuropods.utils.eval_utils import load_and_test_neuropod


def create_python_neuropod(
        neuropod_path,
        model_name,
        data_paths,
        code_path_spec,
        entrypoint_package,
        entrypoint,
        input_spec,
        output_spec,
        test_input_data=None,
        test_expected_out=None,
        test_deps=[],
        test_virtualenv=None,
        skip_virtualenv=False,
        default_input_device="GPU"):
    """
    Packages arbitrary python code as a neuropod package.

    :param  neuropod_path:      The output neuropod path

    :param  model_name:         The name of the model

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

    :param  input_spec:         A list of dicts specifying the input to the model. For each input, if shape
                                is set to `None`, no validation is done on the shape. If shape is a tuple, the
                                dimensions of the input are validated against that tuple.  A value of
                                `None` for any of the dimensions means that dimension will not be checked.
                                `dtype` can be any valid numpy datatype string. These can optionally specify
                                a device (either "CPU" or "GPU"). If not specified, the `default_input_device`
                                will be used.
                                Ex: [
                                    {"name": "x", "dtype": "float32", "shape": (None,), "device": "GPU"},
                                    {"name": "y", "dtype": "float32", "shape": (None,)},
                                ]

    :param  output_spec:        A list of dicts specifying the output of the model. See the documentation for
                                the `input_spec` parameter for more details.
                                Ex: [
                                    {"name": "out", "dtype": "float32", "shape": (None,)},
                                ]

    :param  test_input_data:    Optional sample input data. This is a dict mapping input names to
                                values. If this is provided, inference will be run in an isolated environment
                                immediately after packaging to ensure that the neuropod was created
                                successfully. Must be provided if `test_expected_out` is provided.

                                Throws a ValueError if inference failed.
                                Ex: {
                                    "x": np.arange(5),
                                    "y": np.arange(5),
                                }

    :param  test_expected_out:  Optional expected output. Throws a ValueError if the output of model inference
                                does not match the expected output.
                                Ex: {
                                    "out": np.arange(5) + np.arange(5)
                                }

    :param  test_deps:          Optional pip packages that the packaged code depends on
                                Ex: ['torch==0.4.1', 'numpy']

    :param  test_virtualenv:    The path to a virtualenv already containing the required deps to run the test in.
                                If not specified, a new temporary virtualenv is created.

    :param  skip_virtualenv:    If set to true, runs the test locally instead of in a virtualenv
    :param  default_input_device:   The default device that input tensors should be moved to before inference.
    """
    try:
        # Create the neuropod folder
        os.mkdir(neuropod_path)
    except OSError:
        raise ValueError("The specified neuropod path ({}) already exists! Aborting...".format(neuropod_path))

    # Write the neuropod config file
    config_utils.write_neuropod_config(
        neuropod_path=neuropod_path,
        model_name=model_name,
        platform="python",
        input_spec=input_spec,
        output_spec=output_spec,
        default_input_device=default_input_device
    )

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

    if test_input_data is not None:
        # Load and run the neuropod to make sure that packaging worked correctly
        # Throws a ValueError if the output doesn't match the expected output (if specified)
        load_and_test_neuropod(
            neuropod_path,
            test_input_data,
            test_expected_out,
            use_virtualenv=not skip_virtualenv,
            test_deps=test_deps,
            test_virtualenv=test_virtualenv,
        )
