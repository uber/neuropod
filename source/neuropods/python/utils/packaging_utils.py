#
# Uber, Inc. (c) 2019
#

import inspect
import os
import tempfile
import shutil
import zipfile

from neuropods.backends import config_utils
from neuropods.utils.eval_utils import save_test_data, load_and_test_neuropod

def _zipdir(path, zf):
    for root, dirs, files in os.walk(path):
        for file in files:
            abspath = os.path.join(root, file)
            relpath = os.path.relpath(abspath, path)
            zf.write(abspath, arcname=relpath)

# A docstring common to all packagers
COMMON_DOC_PRE = """
    :param  neuropod_path:      The output neuropod path

    :param  model_name:         The name of the model
"""

COMMON_DOC_POST = """
    :param  input_spec:         A list of dicts specifying the input to the model. For each input, if shape
                                is set to `None`, no validation is done on the shape. If shape is a tuple, the
                                dimensions of the input are validated against that tuple.  A value of
                                `None` for any of the dimensions means that dimension will not be checked.
                                `dtype` can be any valid numpy datatype string.
                                Ex: [
                                    {"name": "x", "dtype": "float32", "shape": (None,)},
                                    {"name": "y", "dtype": "float32", "shape": (None,)},
                                ]

    :param  output_spec:        A list of dicts specifying the output of the model. See the documentation for
                                the `input_spec` parameter for more details.
                                Ex: [
                                    {"name": "out", "dtype": "float32", "shape": (None,)},
                                ]

    :param  input_tensor_device:    A dict mapping input tensor names to the device
                                    that the model expects them to be on. This can
                                    either be `GPU` or `CPU`. Any tensors in `input_spec`
                                    not specified in this mapping will use the
                                    `default_input_tensor_device` specified below.

                                    If a GPU is selected at inference time, Neuropods
                                    will move tensors to the appropriate devices before
                                    running the model. Otherwise, it will attempt to run
                                    the model on CPU and move all tensors (and the model)
                                    to CPU.

                                    See the docstring for `load_neuropod` for more info.

                                    Ex: `{"x": "GPU"}`

    :param  default_input_tensor_device:    The default device that input tensors are expected
                                            to be on. This can either be `GPU` or `CPU`.

    :param  custom_ops:                     A list of paths to custom op shared libraries to include in the packaged neuropod.

                                            Note: Including custom ops ties your neuropod to the specific platform (e.g. Mac, Linux)
                                            that the custom ops were built for. It is the user's responsibility to ensure that their
                                            custom ops are built for the correct platform.

                                            Ex: `["/path/to/my/custom_op.so"]`

    :param  package_as_zip:     Whether to package the neuropod as a single file or as a directory. Default true (i.e. as a zipfile).

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

    :param  persist_test_data:  Optionally saves the test data within the packaged neuropod. default True.
"""

def set_packager_docstring(f):
    # Expects the functon to have a docstring including
    # {common_doc_pre} and {common_doc_post}
    # We can't easily use `.format` because the docstrings contain {}
    f.__doc__ = f.__doc__.replace("{common_doc_pre}", COMMON_DOC_PRE).replace("{common_doc_post}", COMMON_DOC_POST)
    return f

def packager(platform):
    # A decorator that wraps a `platform` specific packager with generic packaging and sets docstrings correctly

    def inner(f):
        # Expects the functon to have a docstring including
        # {common_doc_pre} and {common_doc_post}

        def wrapper(**kwargs):
            # Runs create neuropod
            _create_neuropod(
                packager_fn=f,
                platform=platform,
                **kwargs
            )

        # We can't easily use `.format` because the docstrings contain {}
        wrapper.__doc__  = f.__doc__.replace("{common_doc_pre}", COMMON_DOC_PRE).replace("{common_doc_post}", COMMON_DOC_POST)
        wrapper.__name__ = f.__name__

        # The default args for a packager come from combining the default args
        # of all of these functions
        wrapper.neuropod_default_arg_list = _generate_default_arg_map([
            config_utils.write_neuropod_config,
            load_and_test_neuropod,
            _create_neuropod,
            f,
        ])
        return wrapper

    return inner

def _get_default_args(f):
    """
    get the default args of a functon `f` as a list of tuples of the format
    (arg, default_value)
    """
    argspec = inspect.getargspec(f)
    if argspec.defaults:
        # Generate tuples of (arg, default_value)
        # According to https://docs.python.org/2/library/inspect.html#inspect.getargspec,
        # if defaults has n elements, they correspond to the last n elements listed in args.
        return list(zip(reversed(argspec.args), reversed(argspec.defaults)))

    return []

def _generate_default_arg_map(f_list):
    """
    Given a list of functions, generates a map from an arg name to a default value

    Note: later functions take priority
    """
    default_args = {}
    for f in f_list:
        for arg, default in _get_default_args(f):
            default_args[arg] = default

    return default_args

def _create_neuropod(
    neuropod_path,
    packager_fn,
    package_as_zip=True,
    custom_ops=[],
    test_input_data=None,
    test_expected_out=None,
    persist_test_data=True,
    **kwargs):
    if package_as_zip:
        package_path = tempfile.mkdtemp()
    else:
        try:
            # Create the neuropod folder
            os.mkdir(neuropod_path)
        except OSError:
            raise ValueError("The specified neuropod path ({}) already exists! Aborting...".format(neuropod_path))

        package_path = neuropod_path

    # Create the structure
    # Store the custom ops (if any)
    neuropod_custom_op_path = os.path.join(package_path, "0", "ops")
    os.makedirs(neuropod_custom_op_path)
    for op in custom_ops:
        shutil.copy(op, neuropod_custom_op_path)

    # Write the neuropod config file
    config_utils.write_neuropod_config(
        neuropod_path=package_path,
        custom_ops=[os.path.basename(op) for op in custom_ops],
        **kwargs
    )

    # Run the packager
    packager_fn(neuropod_path=package_path, **kwargs)

    # Persist the test data
    if test_input_data is not None and persist_test_data:
        save_test_data(package_path, test_input_data, test_expected_out)

    # Zip the directory if necessary
    if package_as_zip:
        if os.path.exists(neuropod_path):
            raise ValueError("The specified neuropod path ({}) already exists! Aborting...".format(neuropod_path))

        zf = zipfile.ZipFile(neuropod_path, 'w', zipfile.ZIP_DEFLATED)
        _zipdir(package_path, zf)
        zf.close()

        # Remove our tempdir
        shutil.rmtree(package_path)

    # Test the neuropod
    if test_input_data is not None:
        # Load and run the neuropod to make sure that packaging worked correctly
        # Throws a ValueError if the output doesn't match the expected output (if specified)
        load_and_test_neuropod(
            neuropod_path,
            test_input_data,
            test_expected_out,
            **kwargs
        )
